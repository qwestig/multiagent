from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from mkpi_app.evaluation import Evaluator
from mkpi_app.storage import ProjectStorage


class DatasetQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.storage = ProjectStorage.default(root)
        self.cases = self.storage.load_dataset()
        self.evaluator = Evaluator()

    def test_dataset_keeps_expected_bucket_balance(self) -> None:
        counts = Counter(case.bucket for case in self.cases)
        self.assertEqual(counts["math_logic"], 15)
        self.assertEqual(counts["analysis"], 15)
        self.assertEqual(counts["constraint_planning"], 15)

    def test_text_cases_keep_rich_scoring_specs(self) -> None:
        for case in self.cases:
            if case.bucket == "analysis":
                self.assertIn("required_sections", case.scoring_spec, case.id)
                self.assertIn("balance_markers", case.scoring_spec, case.id)
                self.assertIn("uncertainty_markers", case.scoring_spec, case.id)
                self.assertIn("follow_up_markers", case.scoring_spec, case.id)
                self.assertGreaterEqual(int(case.scoring_spec.get("min_words", 0)), 80, case.id)
            if case.bucket == "constraint_planning":
                self.assertIn("required_sections", case.scoring_spec, case.id)
                self.assertIn("quality_markers", case.scoring_spec, case.id)
                self.assertGreaterEqual(int(case.scoring_spec.get("min_steps", 0)), 4, case.id)

    def test_reference_answers_score_high(self) -> None:
        for case in self.cases:
            if case.bucket == "math_logic":
                continue
            score = self.evaluator.score(case, case.reference)
            self.assertGreaterEqual(
                score.overall,
                0.9,
                f"{case.id} reference score too low: {score.overall:.3f}",
            )


if __name__ == "__main__":
    unittest.main()
