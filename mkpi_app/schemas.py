from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

BucketName = Literal["math_logic", "analysis", "constraint_planning"]


@dataclass(slots=True)
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ScoreComponent:
    name: str
    value: float
    rationale: str


@dataclass(slots=True)
class ScoreBreakdown:
    overall: float
    components: list[ScoreComponent] = field(default_factory=list)
    passed_checks: int = 0
    total_checks: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetCase:
    id: str
    bucket: BucketName
    title: str
    prompt: str
    reference: str
    constraints: list[str] = field(default_factory=list)
    scoring_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetCase":
        return cls(
            id=payload["id"],
            bucket=payload["bucket"],
            title=payload["title"],
            prompt=payload["prompt"],
            reference=payload.get("reference", ""),
            constraints=list(payload.get("constraints", [])),
            scoring_spec=dict(payload.get("scoring_spec", {})),
        )


@dataclass(slots=True)
class ModelRequest:
    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1200
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CritiqueResult:
    summary: str
    failure_hypothesis: str = ""
    issues: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    uncertainty_level: str = "medium"
    needs_revision: bool = True
    estimated_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IterationRecord:
    iteration_index: int
    draft: str
    critique: str
    failure_hypothesis: str
    uncertainty_level: str
    checks: list[str]
    unresolved_questions: list[str]
    revision: str
    anti_error_rule: str
    estimated_quality: float
    tokens: int
    latency_ms: int
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProxyMecoStep:
    step_index: int
    text: str
    confidence: float
    rationale: str
    cumulative_score: float
    marginal_gain: float
    is_supported: bool
    judge_score: float = 0.0
    judge_rationale: str = ""
    llm_judge_score: float = 0.0
    llm_judge_supported: bool = False
    llm_judge_rationale: str = ""


@dataclass(slots=True)
class ProxyMecoTrace:
    sample_count: int = 1
    step_count: int = 0
    supported_steps: int = 0
    confidence_mean: float = 0.0
    auroc: float = 0.0
    average_precision: float = 0.0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    overconfidence_rate: float = 0.0
    llm_judge_agreement: float = 0.0
    confidence_std_mean: float = 0.0
    llm_judge_consistency: float = 0.0
    dual_supported_steps: int = 0
    steps: list[ProxyMecoStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunnerConfig:
    model_name: str = "demo-meta-correction"
    max_iterations: int = 3
    temperature: float = 0.2
    max_tokens: int = 1200
    improvement_threshold: float = 0.03
    mode: Literal["interactive", "benchmark"] = "interactive"
    persist_runs: bool = True
    persist_anti_errors: bool = True
    enable_proxy_meco: bool = False
    proxy_meco_subset_only: bool = False
    proxy_meco_cases_per_bucket: int = 4
    proxy_meco_repeats: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentRun:
    run_id: str
    case_id: str
    case_title: str
    bucket: BucketName
    mode: str
    created_at: str
    baseline_answer: str
    baseline_score: ScoreBreakdown
    final_answer: str
    final_score: ScoreBreakdown
    proxy_meco_eligible: bool = False
    baseline_proxy_meco: ProxyMecoTrace = field(default_factory=ProxyMecoTrace)
    final_proxy_meco: ProxyMecoTrace = field(default_factory=ProxyMecoTrace)
    iterations: list[IterationRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=payload["run_id"],
            case_id=payload["case_id"],
            case_title=payload["case_title"],
            bucket=payload["bucket"],
            mode=payload["mode"],
            created_at=payload["created_at"],
            proxy_meco_eligible=payload.get("proxy_meco_eligible", False),
            baseline_answer=payload["baseline_answer"],
            baseline_score=ScoreBreakdown(
                overall=payload["baseline_score"]["overall"],
                components=[
                    ScoreComponent(**component)
                    for component in payload["baseline_score"].get("components", [])
                ],
                passed_checks=payload["baseline_score"].get("passed_checks", 0),
                total_checks=payload["baseline_score"].get("total_checks", 0),
                notes=list(payload["baseline_score"].get("notes", [])),
            ),
            baseline_proxy_meco=ProxyMecoTrace(
                sample_count=payload.get("baseline_proxy_meco", {}).get("sample_count", 1),
                step_count=payload.get("baseline_proxy_meco", {}).get("step_count", 0),
                supported_steps=payload.get("baseline_proxy_meco", {}).get("supported_steps", 0),
                confidence_mean=payload.get("baseline_proxy_meco", {}).get("confidence_mean", 0.0),
                auroc=payload.get("baseline_proxy_meco", {}).get("auroc", 0.0),
                average_precision=payload.get("baseline_proxy_meco", {}).get("average_precision", 0.0),
                brier_score=payload.get("baseline_proxy_meco", {}).get("brier_score", 0.0),
                expected_calibration_error=payload.get("baseline_proxy_meco", {}).get("expected_calibration_error", 0.0),
                precision=payload.get("baseline_proxy_meco", {}).get("precision", 0.0),
                recall=payload.get("baseline_proxy_meco", {}).get("recall", 0.0),
                f1=payload.get("baseline_proxy_meco", {}).get("f1", 0.0),
                overconfidence_rate=payload.get("baseline_proxy_meco", {}).get("overconfidence_rate", 0.0),
                llm_judge_agreement=payload.get("baseline_proxy_meco", {}).get("llm_judge_agreement", 0.0),
                confidence_std_mean=payload.get("baseline_proxy_meco", {}).get("confidence_std_mean", 0.0),
                llm_judge_consistency=payload.get("baseline_proxy_meco", {}).get("llm_judge_consistency", 0.0),
                dual_supported_steps=payload.get("baseline_proxy_meco", {}).get("dual_supported_steps", 0),
                steps=[
                    ProxyMecoStep(**step)
                    for step in payload.get("baseline_proxy_meco", {}).get("steps", [])
                ],
            ),
            final_answer=payload["final_answer"],
            final_score=ScoreBreakdown(
                overall=payload["final_score"]["overall"],
                components=[
                    ScoreComponent(**component)
                    for component in payload["final_score"].get("components", [])
                ],
                passed_checks=payload["final_score"].get("passed_checks", 0),
                total_checks=payload["final_score"].get("total_checks", 0),
                notes=list(payload["final_score"].get("notes", [])),
            ),
            final_proxy_meco=ProxyMecoTrace(
                sample_count=payload.get("final_proxy_meco", {}).get("sample_count", 1),
                step_count=payload.get("final_proxy_meco", {}).get("step_count", 0),
                supported_steps=payload.get("final_proxy_meco", {}).get("supported_steps", 0),
                confidence_mean=payload.get("final_proxy_meco", {}).get("confidence_mean", 0.0),
                auroc=payload.get("final_proxy_meco", {}).get("auroc", 0.0),
                average_precision=payload.get("final_proxy_meco", {}).get("average_precision", 0.0),
                brier_score=payload.get("final_proxy_meco", {}).get("brier_score", 0.0),
                expected_calibration_error=payload.get("final_proxy_meco", {}).get("expected_calibration_error", 0.0),
                precision=payload.get("final_proxy_meco", {}).get("precision", 0.0),
                recall=payload.get("final_proxy_meco", {}).get("recall", 0.0),
                f1=payload.get("final_proxy_meco", {}).get("f1", 0.0),
                overconfidence_rate=payload.get("final_proxy_meco", {}).get("overconfidence_rate", 0.0),
                llm_judge_agreement=payload.get("final_proxy_meco", {}).get("llm_judge_agreement", 0.0),
                confidence_std_mean=payload.get("final_proxy_meco", {}).get("confidence_std_mean", 0.0),
                llm_judge_consistency=payload.get("final_proxy_meco", {}).get("llm_judge_consistency", 0.0),
                dual_supported_steps=payload.get("final_proxy_meco", {}).get("dual_supported_steps", 0),
                steps=[
                    ProxyMecoStep(**step)
                    for step in payload.get("final_proxy_meco", {}).get("steps", [])
                ],
            ),
            iterations=[
                IterationRecord(
                    iteration_index=item["iteration_index"],
                    draft=item["draft"],
                    critique=item["critique"],
                    failure_hypothesis=item.get("failure_hypothesis", ""),
                    uncertainty_level=item.get("uncertainty_level", "medium"),
                    checks=list(item.get("checks", [])),
                    unresolved_questions=list(item.get("unresolved_questions", [])),
                    revision=item["revision"],
                    anti_error_rule=item.get("anti_error_rule", ""),
                    estimated_quality=item.get("estimated_quality", 0.0),
                    tokens=item.get("tokens", 0),
                    latency_ms=item.get("latency_ms", 0),
                    stop_reason=item.get("stop_reason"),
                )
                for item in payload.get("iterations", [])
            ],
            config=dict(payload.get("config", {})),
            artifacts=dict(payload.get("artifacts", {})),
        )


@dataclass(slots=True)
class StoragePaths:
    root: Path
    dataset_path: Path
    runs_dir: Path
    reports_dir: Path
    anti_error_path: Path
