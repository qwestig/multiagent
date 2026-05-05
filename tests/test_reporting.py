from __future__ import annotations

import unittest

from mkpi_app.reporting import aggregate_runs
from mkpi_app.schemas import ExperimentRun, ProxyMecoTrace, ScoreBreakdown


class ReportingTests(unittest.TestCase):
    def _run(self, case_id: str, bucket: str, baseline: float, final: float, auroc: float, ece: float, f1: float, agreement: float) -> ExperimentRun:
        return ExperimentRun(
            run_id=f"run-{case_id}",
            case_id=case_id,
            case_title=f"Кейс {case_id}",
            bucket=bucket,  # type: ignore[arg-type]
            mode="benchmark",
            created_at="2026-04-08T10:00:00+00:00",
            proxy_meco_eligible=True,
            baseline_answer="baseline",
            baseline_score=ScoreBreakdown(overall=baseline),
            final_answer="final",
            final_score=ScoreBreakdown(overall=final),
            baseline_proxy_meco=ProxyMecoTrace(auroc=auroc - 0.1, expected_calibration_error=ece + 0.05, f1=max(0.0, f1 - 0.1), llm_judge_agreement=max(0.0, agreement - 0.1)),
            final_proxy_meco=ProxyMecoTrace(auroc=auroc, expected_calibration_error=ece, f1=f1, llm_judge_agreement=agreement),
        )

    def test_aggregate_runs_includes_bucket_level_proxy_metrics(self) -> None:
        runs = [
            self._run("a1", "analysis", 0.4, 0.8, 0.72, 0.14, 0.7, 0.8),
            self._run("a2", "analysis", 0.5, 0.7, 0.68, 0.18, 0.6, 0.7),
            self._run("m1", "math_logic", 0.3, 0.9, 0.81, 0.09, 0.9, 0.95),
        ]
        aggregate = aggregate_runs(runs)
        analysis = aggregate["bucket_breakdown"]["analysis"]
        self.assertIn("proxy_final_auroc", analysis)
        self.assertIn("proxy_final_ece", analysis)
        self.assertIn("proxy_final_f1", analysis)
        self.assertIn("proxy_final_agreement", analysis)
        self.assertIn("proxy_eligible_count", analysis)
        self.assertEqual(analysis["count"], 2)
        self.assertAlmostEqual(analysis["proxy_final_auroc"], 0.7, places=4)
        self.assertEqual(aggregate["proxy_meco_protocol"]["eligible_runs"], 3)


if __name__ == "__main__":
    unittest.main()
