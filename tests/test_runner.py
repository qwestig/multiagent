from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mkpi_app.evaluation import Evaluator
from mkpi_app.models import DemoModelAdapter
from mkpi_app.runner import StopPolicy, TechniqueRunner
from mkpi_app.schemas import DatasetCase, RunnerConfig
from mkpi_app.storage import ProjectStorage


class TechniqueRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.storage = ProjectStorage.default(Path(self.tempdir.name))
        self.evaluator = Evaluator()
        self.runner = TechniqueRunner(
            model=DemoModelAdapter(),
            evaluator=self.evaluator,
            storage=self.storage,
            stop_policy=StopPolicy(max_iterations=3, improvement_threshold=0.01),
        )
        self.case = DatasetCase(
            id="math-demo",
            bucket="math_logic",
            title="Контрольный пересчёт",
            prompt="Вычисли 37 * 24 - 158.",
            reference="Шаги: 37 * 24 = 888, 888 - 158 = 730. Итог: 730.",
            constraints=["Покажи шаги", "Выдели итог"],
            scoring_spec={"numeric_expected": 730, "must_include": ["шаг", "итог"]},
        )
        self.analysis_case = DatasetCase(
            id="analysis-demo",
            bucket="analysis",
            title="LLM в образовании",
            prompt="Оцени использование LLM в учебном процессе.",
            reference=(
                "LLM помогают персонализировать объяснения, однако усиливают риски для академической "
                "честности и требуют обязательной верификации ответов. Неопределённость связана с тем, "
                "как устойчиво студенты будут различать помощь и подмену самостоятельной работы. "
                "Вывод: использовать инструмент можно только как контролируемого помощника. "
                "Следующий шаг: провести пилот и проверить влияние на честность и качество обучения."
            ),
            constraints=["Укажи преимущества и риски", "Обозначь зоны неопределённости", "Дай вывод и следующий шаг"],
            scoring_spec={
                "required_keywords": ["академическая честность", "персонализ", "верификац"],
                "required_sections": {
                    "benefits": ["преимуществ", "ускоряет", "персонализ"],
                    "risks": ["риск", "угроз", "слаб"],
                    "uncertainty": ["неопредел", "недостаточно данных", "требует проверки"],
                    "recommendation": ["вывод", "рекоменда", "итог"],
                    "next_step": ["проверить", "уточнить", "пилот", "собрать данные"],
                },
                "balance_markers": ["однако", "при этом", "с другой стороны"],
                "uncertainty_markers": ["неопредел", "недостаточно данных", "требует проверки", "нужно проверить"],
                "follow_up_markers": ["проверить", "уточнить", "пилот", "дополнительно собрать"],
                "min_words": 80,
            },
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_runner_improves_result_and_persists_artifacts(self) -> None:
        run = self.runner.run(
            self.case,
            RunnerConfig(
                model_name="demo-meta-correction",
                max_iterations=3,
                improvement_threshold=0.01,
                mode="interactive",
                persist_runs=True,
                persist_anti_errors=True,
            ),
        )
        self.assertGreaterEqual(run.final_score.overall, run.baseline_score.overall)
        self.assertGreaterEqual(len(run.iterations), 1)
        self.assertTrue(run.iterations[0].failure_hypothesis)
        self.assertIn("run_json", run.artifacts)
        self.assertTrue(Path(run.artifacts["run_json"]).exists())
        anti_errors = self.storage.load_anti_errors()
        self.assertIn("math_logic", anti_errors)
        self.assertTrue(anti_errors["math_logic"])

    def test_batch_run_exports_csv_and_report(self) -> None:
        runs, artifacts = self.runner.run_batch(
            [self.case],
            RunnerConfig(
                model_name="demo-meta-correction",
                max_iterations=2,
                improvement_threshold=0.01,
                mode="benchmark",
                persist_runs=True,
                persist_anti_errors=False,
            ),
        )
        self.assertEqual(len(runs), 1)
        self.assertTrue(Path(artifacts["results_csv"]).exists())
        self.assertTrue(Path(artifacts["report_md"]).exists())

    def test_runner_improves_analysis_case(self) -> None:
        run = self.runner.run(
            self.analysis_case,
            RunnerConfig(
                model_name="demo-meta-correction",
                max_iterations=3,
                improvement_threshold=0.01,
                mode="benchmark",
                persist_runs=False,
                persist_anti_errors=False,
            ),
        )
        self.assertGreater(run.final_score.overall, run.baseline_score.overall)
        self.assertTrue(run.iterations)
        self.assertIn(
            run.iterations[0].failure_hypothesis,
            {"hidden_constraint", "unsupported_assumption", "decomposition_error"},
        )


if __name__ == "__main__":
    unittest.main()
