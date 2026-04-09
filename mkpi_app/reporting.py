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
            }
        )
    return rows


def aggregate_runs(runs: list[ExperimentRun]) -> dict[str, object]:
    if not runs:
        return {
            "runs": 0,
            "baseline_mean": 0.0,
            "final_mean": 0.0,
            "delta_mean": 0.0,
            "failure_rate": 0.0,
            "bucket_breakdown": {},
        }

    baseline_scores = [run.baseline_score.overall for run in runs]
    final_scores = [run.final_score.overall for run in runs]
    deltas = [final - base for final, base in zip(final_scores, baseline_scores)]
    bucket_breakdown: dict[str, dict[str, float]] = {}
    for bucket in sorted({run.bucket for run in runs}):
        bucket_runs = [run for run in runs if run.bucket == bucket]
        bucket_breakdown[bucket] = {
            "count": len(bucket_runs),
            "baseline_mean": round(mean(run.baseline_score.overall for run in bucket_runs), 4),
            "final_mean": round(mean(run.final_score.overall for run in bucket_runs), 4),
            "delta_mean": round(
                mean(run.final_score.overall - run.baseline_score.overall for run in bucket_runs),
                4,
            ),
        }

    failure_rate = sum(1 for run in runs if run.final_score.overall < run.baseline_score.overall) / len(runs)
    return {
        "runs": len(runs),
        "baseline_mean": round(mean(baseline_scores), 4),
        "final_mean": round(mean(final_scores), 4),
        "delta_mean": round(mean(deltas), 4),
        "failure_rate": round(clamp(failure_rate), 4),
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
        f"- Средний baseline score: {aggregate['baseline_mean']}",
        f"- Средний final score: {aggregate['final_mean']}",
        f"- Средняя дельта: {aggregate['delta_mean']}",
        f"- Доля деградаций: {aggregate['failure_rate']}",
        "",
        "## Разбивка по корзинам",
    ]
    for bucket, metrics in aggregate["bucket_breakdown"].items():
        lines.append(
            f"- {bucket}: count={metrics['count']}, baseline={metrics['baseline_mean']}, final={metrics['final_mean']}, delta={metrics['delta_mean']}"
        )

    lines.extend(["", "## Таблица результатов", "", "| case_id | bucket | baseline | final | delta | iterations |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['bucket']} | {row['baseline_score']} | {row['final_score']} | {row['delta']} | {row['iterations']} |"
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
    return "\n".join(lines) + "\n"
