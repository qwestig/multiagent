from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone

from .evaluation import Evaluator
from .benchmark_protocol import is_proxy_meco_eligible, select_proxy_meco_subset
from .models import ModelAdapter
from .prompts import (
    build_baseline_prompt,
    build_critique_prompt,
    build_draft_prompt,
    build_protocol_prompt,
    build_revision_prompt,
    build_self_eval_prompt,
    build_step_judge_prompt,
)
from .proxy_meco import aggregate_proxy_samples, build_proxy_meco_trace, extract_reasoning_steps, parse_step_confidences, parse_step_judgments
from .reporting import render_markdown_report
from .schemas import (
    CritiqueResult,
    DatasetCase,
    ExperimentRun,
    IterationRecord,
    ModelRequest,
    ProxyMecoTrace,
    RunnerConfig,
    ScoreBreakdown,
)
from .storage import ProjectStorage
from .utils import extract_json_object


class StopPolicy:
    def __init__(self, max_iterations: int = 3, improvement_threshold: float = 0.03) -> None:
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

    def after_critique(self, critique: CritiqueResult) -> str | None:
        if not critique.needs_revision:
            return "Критик не нашёл существенных замечаний."
        return None

    def after_revision(
        self, iteration_index: int, previous_score: ScoreBreakdown, new_score: ScoreBreakdown
    ) -> str | None:
        improvement = new_score.overall - previous_score.overall
        if improvement < self.improvement_threshold:
            return "Улучшение ниже порога, дальнейшие итерации остановлены."
        if iteration_index >= self.max_iterations:
            return "Достигнут лимит итераций."
        return None


class TechniqueRunner:
    def __init__(
        self,
        model: ModelAdapter,
        evaluator: Evaluator,
        storage: ProjectStorage,
        stop_policy: StopPolicy | None = None,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.storage = storage
        self.stop_policy = stop_policy or StopPolicy()

    def run(self, case: DatasetCase, config: RunnerConfig) -> ExperimentRun:
        anti_errors = (
            self.storage.recent_anti_errors(case.bucket)
            if config.persist_anti_errors and config.mode == "interactive"
            else []
        )
        baseline_answer = self._call_model("baseline", case, config).text
        baseline_score = self.evaluator.score(case, baseline_answer)
        baseline_proxy_meco = self._build_proxy_meco(case, config, baseline_answer)

        draft_response = self._call_model(
            "draft",
            case,
            config,
            answer=None,
            critique=None,
            anti_errors=anti_errors,
        )
        current_answer = draft_response.text
        current_score = self.evaluator.score(case, current_answer)

        iterations: list[IterationRecord] = []
        for iteration_index in range(1, config.max_iterations + 1):
            critique_response = self._call_model(
                "critique",
                case,
                config,
                answer=current_answer,
            )
            critique = self._parse_critique(critique_response.text)
            protocol_response = self._call_model(
                "protocol",
                case,
                config,
                answer=current_answer,
                critique=critique,
            )

            stop_reason = self.stop_policy.after_critique(critique)
            if stop_reason:
                record = IterationRecord(
                    iteration_index=iteration_index,
                    draft=current_answer,
                    critique=critique.summary,
                    failure_hypothesis=critique.failure_hypothesis,
                    uncertainty_level=critique.uncertainty_level,
                    checks=critique.checks,
                    unresolved_questions=critique.unresolved_questions,
                    revision=current_answer,
                    anti_error_rule=protocol_response.text.strip(),
                    estimated_quality=current_score.overall,
                    tokens=critique_response.usage.total_tokens + protocol_response.usage.total_tokens,
                    latency_ms=critique_response.latency_ms + protocol_response.latency_ms,
                    stop_reason=stop_reason,
                )
                iterations.append(record)
                self._remember_rule(case, config, record.anti_error_rule)
                break

            revision_response = self._call_model(
                "revision",
                case,
                config,
                answer=current_answer,
                critique=critique,
            )
            revised_answer = revision_response.text
            revised_score = self.evaluator.score(case, revised_answer)
            stop_reason = self.stop_policy.after_revision(iteration_index, current_score, revised_score)
            record = IterationRecord(
                iteration_index=iteration_index,
                draft=current_answer,
                critique=critique.summary,
                failure_hypothesis=critique.failure_hypothesis,
                uncertainty_level=critique.uncertainty_level,
                checks=critique.checks,
                unresolved_questions=critique.unresolved_questions,
                revision=revised_answer,
                anti_error_rule=protocol_response.text.strip(),
                estimated_quality=revised_score.overall,
                tokens=(
                    critique_response.usage.total_tokens
                    + revision_response.usage.total_tokens
                    + protocol_response.usage.total_tokens
                ),
                latency_ms=(
                    critique_response.latency_ms
                    + revision_response.latency_ms
                    + protocol_response.latency_ms
                ),
                stop_reason=stop_reason,
            )
            iterations.append(record)
            self._remember_rule(case, config, record.anti_error_rule)
            current_answer = revised_answer
            current_score = revised_score
            if stop_reason:
                break

        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        run = ExperimentRun(
            run_id=uuid.uuid4().hex,
            case_id=case.id,
            case_title=case.title,
            bucket=case.bucket,
            mode=config.mode,
            created_at=created_at,
            proxy_meco_eligible=is_proxy_meco_eligible(case),
            baseline_answer=baseline_answer,
            baseline_score=baseline_score,
            baseline_proxy_meco=baseline_proxy_meco,
            final_answer=current_answer,
            final_score=current_score,
            final_proxy_meco=self._build_proxy_meco(case, config, current_answer),
            iterations=iterations,
            config=config.to_dict(),
            artifacts={},
        )
        if config.persist_runs:
            run_path = self.storage.save_run(run)
            run.artifacts["run_json"] = str(run_path)
        return run

    def run_batch(self, cases: list[DatasetCase], config: RunnerConfig) -> tuple[list[ExperimentRun], dict[str, str]]:
        batch_config = replace(config, mode="benchmark", persist_anti_errors=False, enable_proxy_meco=True)
        if batch_config.proxy_meco_subset_only:
            cases = select_proxy_meco_subset(cases, per_bucket=batch_config.proxy_meco_cases_per_bucket)
        runs = [self.run(case, batch_config) for case in cases]
        report_path = self.storage.save_report(render_markdown_report(runs))
        csv_path = self.storage.export_results_csv(runs)
        artifacts = {"report_md": str(report_path), "results_csv": str(csv_path)}
        return runs, artifacts

    def _build_proxy_meco(self, case: DatasetCase, config: RunnerConfig, answer: str) -> ProxyMecoTrace:
        if not config.enable_proxy_meco:
            return ProxyMecoTrace()

        steps = extract_reasoning_steps(answer)
        if not steps:
            return ProxyMecoTrace()

        repeats = max(1, config.proxy_meco_repeats)
        confidence_samples: list[list[tuple[float, str]]] = []
        llm_judgment_samples: list[list[tuple[float, bool, str]]] = []
        for _ in range(repeats):
            self_eval_response = self._call_model(
                "self_eval",
                case,
                config,
                answer=answer,
                anti_errors=steps,
            )
            confidence_samples.append(parse_step_confidences(self_eval_response.text, len(steps)))
            step_judge_response = self._call_model(
                "step_judge",
                case,
                config,
                answer=answer,
                anti_errors=steps,
            )
            llm_judgment_samples.append(parse_step_judgments(step_judge_response.text, len(steps)))

        confidence_pairs, llm_judgments, confidence_std_mean, llm_judge_consistency, sample_count = aggregate_proxy_samples(
            confidence_samples,
            llm_judgment_samples,
        )
        return build_proxy_meco_trace(
            case,
            answer,
            confidence_pairs,
            llm_judgments,
            self.evaluator,
            sample_count=sample_count,
            confidence_std_mean=confidence_std_mean,
            llm_judge_consistency=llm_judge_consistency,
        )

    def _remember_rule(self, case: DatasetCase, config: RunnerConfig, rule: str) -> None:
        if config.mode == "interactive" and config.persist_anti_errors:
            self.storage.remember_anti_error(case.bucket, rule)

    def _call_model(
        self,
        phase: str,
        case: DatasetCase,
        config: RunnerConfig,
        answer: str | None = None,
        critique: CritiqueResult | None = None,
        anti_errors: list[str] | None = None,
    ):
        if phase == "baseline":
            system_prompt, user_prompt = build_baseline_prompt(case)
        elif phase == "draft":
            system_prompt, user_prompt = build_draft_prompt(case, anti_errors or [])
        elif phase == "critique":
            system_prompt, user_prompt = build_critique_prompt(case, answer or "")
        elif phase == "revision":
            system_prompt, user_prompt = build_revision_prompt(case, answer or "", critique or CritiqueResult(summary=""))
        elif phase == "self_eval":
            steps = [str(item) for item in (anti_errors or []) if str(item).strip()]
            system_prompt, user_prompt = build_self_eval_prompt(case, answer or "", steps)
        elif phase == "step_judge":
            steps = [str(item) for item in (anti_errors or []) if str(item).strip()]
            system_prompt, user_prompt = build_step_judge_prompt(case, answer or "", steps)
        else:
            system_prompt, user_prompt = build_protocol_prompt(case, answer or "", critique or CritiqueResult(summary=""))

        request = ModelRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            metadata={
                "phase": phase,
                "case": case.to_dict(),
                "answer": answer,
                "steps": list(anti_errors or []),
                "critique": critique.to_dict() if critique else None,
                "anti_errors": anti_errors or [],
            },
        )
        return self.model.generate(request)

    def _parse_critique(self, raw_text: str) -> CritiqueResult:
        payload = extract_json_object(raw_text)
        if payload:
            return CritiqueResult(
                summary=payload.get("summary", "Диагностика выполнена."),
                failure_hypothesis=payload.get("failure_hypothesis", ""),
                issues=list(payload.get("issues", [])),
                checks=list(payload.get("checks", [])),
                unresolved_questions=list(payload.get("unresolved_questions", [])),
                uncertainty_level=payload.get("uncertainty_level", "medium"),
                needs_revision=bool(payload.get("needs_revision", payload.get("issues"))),
                estimated_quality=float(payload.get("estimated_quality", 0.0)),
            )
        lines = [line.strip("- ").strip() for line in raw_text.splitlines() if line.strip()]
        summary = lines[0] if lines else "Диагностика выполнена."
        issues = lines[1:4]
        return CritiqueResult(
            summary=summary,
            failure_hypothesis="unsupported_assumption" if issues else "no_major_issue",
            issues=issues,
            checks=["Проверь пересчёт.", "Проверь ограничения.", "Проверь обратную валидацию."],
            unresolved_questions=[],
            uncertainty_level="medium" if issues else "low",
            needs_revision=bool(issues),
            estimated_quality=0.5,
        )
