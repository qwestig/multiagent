from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mkpi_app.schemas import ScoreBreakdown, ScoreComponent
from mkpi_app.storage import ProjectStorage
from mkpi_app.schemas import ExperimentRun, IterationRecord


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.storage = ProjectStorage.default(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_save_and_load_run_roundtrip(self) -> None:
        run = ExperimentRun(
            run_id="run-1",
            case_id="case-1",
            case_title="Тестовый кейс",
            bucket="analysis",
            mode="interactive",
            created_at="2026-04-08T10:00:00+00:00",
            baseline_answer="baseline",
            baseline_score=ScoreBreakdown(
                overall=0.4,
                components=[ScoreComponent(name="a", value=0.4, rationale="baseline")],
                passed_checks=1,
                total_checks=2,
            ),
            final_answer="final",
            final_score=ScoreBreakdown(
                overall=0.8,
                components=[ScoreComponent(name="a", value=0.8, rationale="final")],
                passed_checks=2,
                total_checks=2,
            ),
            iterations=[
                IterationRecord(
                    iteration_index=1,
                    draft="draft",
                    critique="critique",
                    failure_hypothesis="hidden_constraint",
                    uncertainty_level="medium",
                    checks=["check"],
                    unresolved_questions=["question"],
                    revision="revision",
                    anti_error_rule="Если ..., то ...",
                    estimated_quality=0.8,
                    tokens=120,
                    latency_ms=12,
                )
            ],
            config={"mode": "interactive"},
            artifacts={},
        )
        path = self.storage.save_run(run)
        loaded = self.storage.load_run(path)
        self.assertEqual(loaded.run_id, run.run_id)
        self.assertEqual(loaded.final_score.overall, run.final_score.overall)
        listed = self.storage.list_runs()
        self.assertEqual(len(listed), 1)


if __name__ == "__main__":
    unittest.main()
