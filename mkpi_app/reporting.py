from __future__ import annotations

from statistics import mean

from .schemas import ExperimentRun
from .utils import clamp


def build_results_rows(runs: list[ExperimentRun]) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for run in runs:
        delta = round(run.final_score.overall - run.baseline_score.overall, 4)
        total_tokens = sum(item.tokens for item in run.iterations)
        total_latency = sum(item.latency_ms for item in run.iterations)
        rows.append(
            {
                "run_id": run.run_id,
                "case_id": run.case_id,
                "case_title": run.case_title,
                "bucket": run.bucket,
                "mode": run.mode,
                "baseline_score": round(run.baseline_score.overall, 4),
                "final_score": round(run.final_score.overall, 4),
                "delta": delta,
                "iterations": len(run.iterations),
                "tokens": total_tokens,
                "latency_ms": total_latency,
                "baseline_proxy_auroc": round(run.baseline_proxy_meco.auroc, 4),
                "final_proxy_auroc": round(run.final_proxy_meco.auroc, 4),
                "baseline_proxy_brier": round(run.baseline_proxy_meco.brier_score, 4),
                "final_proxy_brier": round(run.final_proxy_meco.brier_score, 4),
                "baseline_proxy_ece": round(run.baseline_proxy_meco.expected_calibration_error, 4),
                "final_proxy_ece": round(run.final_proxy_meco.expected_calibration_error, 4),
                "baseline_proxy_f1": round(run.baseline_proxy_meco.f1, 4),
                "final_proxy_f1": round(run.final_proxy_meco.f1, 4),
                "baseline_proxy_conf_std": round(run.baseline_proxy_meco.confidence_std_mean, 4),
                "final_proxy_conf_std": round(run.final_proxy_meco.confidence_std_mean, 4),
                "baseline_proxy_judge_consistency": round(run.baseline_proxy_meco.llm_judge_consistency, 4),
                "final_proxy_judge_consistency": round(run.final_proxy_meco.llm_judge_consistency, 4),
                "baseline_proxy_overconfidence": round(run.baseline_proxy_meco.overconfidence_rate, 4),
                "final_proxy_overconfidence": round(run.final_proxy_meco.overconfidence_rate, 4),
                "baseline_proxy_agreement": round(run.baseline_proxy_meco.llm_judge_agreement, 4),
                "final_proxy_agreement": round(run.final_proxy_meco.llm_judge_agreement, 4),
            }
        )
    return rows


def aggregate_runs(runs: list[ExperimentRun]) -> dict[str, object]:
    if not runs:
        return {
            "runs": 0,
            "proxy_meco_protocol": {
                "eligible_runs": 0,
                "eligible_share": 0.0,
            },
            "baseline_mean": 0.0,
            "final_mean": 0.0,
            "delta_mean": 0.0,
            "failure_rate": 0.0,
            "proxy_meco": {
                "baseline_auroc_mean": 0.0,
                "final_auroc_mean": 0.0,
                "baseline_brier_mean": 0.0,
                "final_brier_mean": 0.0,
                "baseline_ece_mean": 0.0,
                "final_ece_mean": 0.0,
                "baseline_f1_mean": 0.0,
                "final_f1_mean": 0.0,
                "baseline_conf_std_mean": 0.0,
                "final_conf_std_mean": 0.0,
                "baseline_judge_consistency_mean": 0.0,
                "final_judge_consistency_mean": 0.0,
                "baseline_overconfidence_mean": 0.0,
                "final_overconfidence_mean": 0.0,
                "baseline_agreement_mean": 0.0,
                "final_agreement_mean": 0.0,
            },
            "bucket_breakdown": {},
        }

    baseline_scores = [run.baseline_score.overall for run in runs]
    final_scores = [run.final_score.overall for run in runs]
    deltas = [final - base for final, base in zip(final_scores, baseline_scores)]
    eligible_runs = [run for run in runs if run.proxy_meco_eligible]
    bucket_breakdown: dict[str, dict[str, float]] = {}
    for bucket in sorted({run.bucket for run in runs}):
        bucket_runs = [run for run in runs if run.bucket == bucket]
        bucket_eligible_runs = [run for run in bucket_runs if run.proxy_meco_eligible]
        bucket_breakdown[bucket] = {
            "count": len(bucket_runs),
            "proxy_eligible_count": len(bucket_eligible_runs),
            "baseline_mean": round(mean(run.baseline_score.overall for run in bucket_runs), 4),
            "final_mean": round(mean(run.final_score.overall for run in bucket_runs), 4),
            "delta_mean": round(
                mean(run.final_score.overall - run.baseline_score.overall for run in bucket_runs),
                4,
            ),
            "proxy_baseline_auroc": round(mean(run.baseline_proxy_meco.auroc for run in bucket_runs), 4),
            "proxy_final_auroc": round(mean(run.final_proxy_meco.auroc for run in bucket_runs), 4),
            "proxy_baseline_brier": round(mean(run.baseline_proxy_meco.brier_score for run in bucket_runs), 4),
            "proxy_final_brier": round(mean(run.final_proxy_meco.brier_score for run in bucket_runs), 4),
            "proxy_baseline_ece": round(mean(run.baseline_proxy_meco.expected_calibration_error for run in bucket_runs), 4),
            "proxy_final_ece": round(mean(run.final_proxy_meco.expected_calibration_error for run in bucket_runs), 4),
            "proxy_baseline_f1": round(mean(run.baseline_proxy_meco.f1 for run in bucket_runs), 4),
            "proxy_final_f1": round(mean(run.final_proxy_meco.f1 for run in bucket_runs), 4),
            "proxy_baseline_agreement": round(mean(run.baseline_proxy_meco.llm_judge_agreement for run in bucket_runs), 4),
            "proxy_final_agreement": round(mean(run.final_proxy_meco.llm_judge_agreement for run in bucket_runs), 4),
            "proxy_final_conf_std": round(mean(run.final_proxy_meco.confidence_std_mean for run in bucket_runs), 4),
            "proxy_final_judge_consistency": round(mean(run.final_proxy_meco.llm_judge_consistency for run in bucket_runs), 4),
        }

    failure_rate = sum(1 for run in runs if run.final_score.overall < run.baseline_score.overall) / len(runs)
    return {
        "runs": len(runs),
        "proxy_meco_protocol": {
            "eligible_runs": len(eligible_runs),
            "eligible_share": round(len(eligible_runs) / len(runs), 4),
        },
        "baseline_mean": round(mean(baseline_scores), 4),
        "final_mean": round(mean(final_scores), 4),
        "delta_mean": round(mean(deltas), 4),
        "failure_rate": round(clamp(failure_rate), 4),
        "proxy_meco": {
            "baseline_auroc_mean": round(mean(run.baseline_proxy_meco.auroc for run in runs), 4),
            "final_auroc_mean": round(mean(run.final_proxy_meco.auroc for run in runs), 4),
            "baseline_brier_mean": round(mean(run.baseline_proxy_meco.brier_score for run in runs), 4),
            "final_brier_mean": round(mean(run.final_proxy_meco.brier_score for run in runs), 4),
            "baseline_ece_mean": round(mean(run.baseline_proxy_meco.expected_calibration_error for run in runs), 4),
            "final_ece_mean": round(mean(run.final_proxy_meco.expected_calibration_error for run in runs), 4),
            "baseline_f1_mean": round(mean(run.baseline_proxy_meco.f1 for run in runs), 4),
            "final_f1_mean": round(mean(run.final_proxy_meco.f1 for run in runs), 4),
            "baseline_conf_std_mean": round(mean(run.baseline_proxy_meco.confidence_std_mean for run in runs), 4),
            "final_conf_std_mean": round(mean(run.final_proxy_meco.confidence_std_mean for run in runs), 4),
            "baseline_judge_consistency_mean": round(mean(run.baseline_proxy_meco.llm_judge_consistency for run in runs), 4),
            "final_judge_consistency_mean": round(mean(run.final_proxy_meco.llm_judge_consistency for run in runs), 4),
            "baseline_overconfidence_mean": round(mean(run.baseline_proxy_meco.overconfidence_rate for run in runs), 4),
            "final_overconfidence_mean": round(mean(run.final_proxy_meco.overconfidence_rate for run in runs), 4),
            "baseline_agreement_mean": round(mean(run.baseline_proxy_meco.llm_judge_agreement for run in runs), 4),
            "final_agreement_mean": round(mean(run.final_proxy_meco.llm_judge_agreement for run in runs), 4),
        },
        "bucket_breakdown": bucket_breakdown,
    }


def render_markdown_report(runs: list[ExperimentRun]) -> str:
    aggregate = aggregate_runs(runs)
    rows = build_results_rows(runs)
    best_run = max(runs, key=lambda run: run.final_score.overall - run.baseline_score.overall, default=None)
    worst_run = min(runs, key=lambda run: run.final_score.overall - run.baseline_score.overall, default=None)

    lines = [
        "# Отчёт по метакоррекции и итерации",
        "",
        "## Сводка",
        f"- Количество прогонов: {aggregate['runs']}",
        f"- Proxy-MECO eligible subset: {aggregate['proxy_meco_protocol']['eligible_runs']} ({aggregate['proxy_meco_protocol']['eligible_share']})",
        f"- Средний baseline score: {aggregate['baseline_mean']}",
        f"- Средний final score: {aggregate['final_mean']}",
        f"- Средняя дельта: {aggregate['delta_mean']}",
        f"- Доля деградаций: {aggregate['failure_rate']}",
        f"- Proxy-MECO baseline AUROC: {aggregate['proxy_meco']['baseline_auroc_mean']}",
        f"- Proxy-MECO final AUROC: {aggregate['proxy_meco']['final_auroc_mean']}",
        f"- Proxy-MECO baseline Brier: {aggregate['proxy_meco']['baseline_brier_mean']}",
        f"- Proxy-MECO final Brier: {aggregate['proxy_meco']['final_brier_mean']}",
        f"- Proxy-MECO baseline ECE: {aggregate['proxy_meco']['baseline_ece_mean']}",
        f"- Proxy-MECO final ECE: {aggregate['proxy_meco']['final_ece_mean']}",
        f"- Proxy-MECO baseline F1: {aggregate['proxy_meco']['baseline_f1_mean']}",
        f"- Proxy-MECO final F1: {aggregate['proxy_meco']['final_f1_mean']}",
        f"- Proxy-MECO baseline confidence std: {aggregate['proxy_meco']['baseline_conf_std_mean']}",
        f"- Proxy-MECO final confidence std: {aggregate['proxy_meco']['final_conf_std_mean']}",
        f"- Proxy-MECO baseline judge consistency: {aggregate['proxy_meco']['baseline_judge_consistency_mean']}",
        f"- Proxy-MECO final judge consistency: {aggregate['proxy_meco']['final_judge_consistency_mean']}",
        f"- Proxy-MECO baseline overconfidence: {aggregate['proxy_meco']['baseline_overconfidence_mean']}",
        f"- Proxy-MECO final overconfidence: {aggregate['proxy_meco']['final_overconfidence_mean']}",
        f"- Proxy-MECO baseline judge agreement: {aggregate['proxy_meco']['baseline_agreement_mean']}",
        f"- Proxy-MECO final judge agreement: {aggregate['proxy_meco']['final_agreement_mean']}",
        "",
        "## Разбивка по корзинам",
    ]
    for bucket, metrics in aggregate["bucket_breakdown"].items():
        lines.append(
            f"- {bucket}: count={metrics['count']}, proxy_eligible={metrics['proxy_eligible_count']}, baseline={metrics['baseline_mean']}, final={metrics['final_mean']}, delta={metrics['delta_mean']}, proxy_final_auroc={metrics['proxy_final_auroc']}, proxy_final_ece={metrics['proxy_final_ece']}, proxy_final_f1={metrics['proxy_final_f1']}, proxy_final_agreement={metrics['proxy_final_agreement']}, proxy_final_conf_std={metrics['proxy_final_conf_std']}, proxy_final_judge_consistency={metrics['proxy_final_judge_consistency']}"
        )

    lines.extend(["", "## Таблица результатов", "", "| case_id | bucket | baseline | final | delta | proxy AUROC | iterations |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['bucket']} | {row['baseline_score']} | {row['final_score']} | {row['delta']} | {row['final_proxy_auroc']} | {row['iterations']} |"
        )

    if best_run:
        lines.extend(
            [
                "",
                "## Лучший кейс",
                f"**{best_run.case_title}** (`{best_run.case_id}`)",
                "",
                f"- Baseline: {round(best_run.baseline_score.overall, 4)}",
                f"- Final: {round(best_run.final_score.overall, 4)}",
                f"- Delta: {round(best_run.final_score.overall - best_run.baseline_score.overall, 4)}",
                "",
                "### Финальный ответ",
                best_run.final_answer,
            ]
        )
        risky_steps = [step for step in best_run.final_proxy_meco.steps if step.confidence >= 0.7 and not step.is_supported]
        if risky_steps:
            lines.extend(["", "### Proxy-MECO: рискованные шаги"]) 
            for step in risky_steps[:3]:
                lines.append(
                    f"- Шаг {step.step_index}: conf={step.confidence:.2f}, judge={step.judge_score:.2f}, llm_judge={step.llm_judge_score:.2f} :: {step.text}"
                )
    if worst_run:
        lines.extend(
            [
                "",
                "## Сложный кейс",
                f"**{worst_run.case_title}** (`{worst_run.case_id}`)",
                "",
                f"- Baseline: {round(worst_run.baseline_score.overall, 4)}",
                f"- Final: {round(worst_run.final_score.overall, 4)}",
                f"- Delta: {round(worst_run.final_score.overall - worst_run.baseline_score.overall, 4)}",
                "",
                "### Замечания",
            ]
        )
        lines.extend(f"- {note}" for note in worst_run.final_score.notes or ["Отдельные замечания не зафиксированы."])
        disagreement_steps = [step for step in worst_run.final_proxy_meco.steps if step.is_supported != step.llm_judge_supported]
        if disagreement_steps:
            lines.extend(["", "### Proxy-MECO: расхождения judge-каналов"])
            for step in disagreement_steps[:3]:
                lines.append(
                    f"- Шаг {step.step_index}: rule={step.is_supported}, llm={step.llm_judge_supported}, conf={step.confidence:.2f} :: {step.text}"
                )
    return "\n".join(lines) + "\n"
