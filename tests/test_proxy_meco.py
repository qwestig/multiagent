from __future__ import annotations

import unittest

from mkpi_app.evaluation import Evaluator
from mkpi_app.proxy_meco import (
    aggregate_proxy_samples,
    build_proxy_meco_trace,
    extract_reasoning_steps,
    judge_step_support,
    parse_step_judgments,
)
from mkpi_app.schemas import DatasetCase


class ProxyMecoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator()
        self.case = DatasetCase(
            id="math-proxy",
            bucket="math_logic",
            title="Контрольный пересчёт",
            prompt="Вычисли 37 * 24 - 158.",
            reference="Шаги: 37 * 24 = 888, 888 - 158 = 730. Итог: 730.",
            constraints=["Покажи шаги", "Выдели итог"],
            scoring_spec={"numeric_expected": 730, "must_include": ["шаг", "итог"]},
        )
        self.analysis_case = DatasetCase(
            id="analysis-proxy",
            bucket="analysis",
            title="LLM в образовании",
            prompt="Оцени использование LLM в образовании.",
            reference="",
            constraints=["Укажи риски, неопределённость и следующий шаг"],
            scoring_spec={
                "required_keywords": ["академическая честность", "персонализ", "верификац"],
                "required_sections": {
                    "risks": ["риск"],
                    "uncertainty": ["неопредел", "недостаточно данных"],
                    "recommendation": ["вывод", "рекоменда"],
                    "next_step": ["проверить", "пилот"],
                },
                "balance_markers": ["однако"],
                "uncertainty_markers": ["неопредел", "недостаточно данных"],
                "follow_up_markers": ["проверить", "пилот"],
                "min_words": 40,
            },
        )

    def test_extract_reasoning_steps_prefers_numbered_lines(self) -> None:
        answer = "1. Считаем 37 * 24 = 888.\n2. Вычитаем 158 и получаем 730.\n3. Итог: 730."
        steps = extract_reasoning_steps(answer)
        self.assertEqual(len(steps), 3)
        self.assertIn("730", steps[-1])

    def test_build_proxy_meco_trace_marks_degrading_step_as_unsupported(self) -> None:
        answer = "1. Шаг: 37 * 24 = 888.\n2. Шаг: 888 - 158 = 729.\n3. Итог: 729."
        trace = build_proxy_meco_trace(
            self.case,
            answer,
            [
                (0.6, "Промежуточное вычисление выглядит правдоподобно."),
                (0.9, "Уверен в пересчёте."),
                (0.95, "Итог кажется точным."),
            ],
            [
                (0.7, True, "Шаг выглядит допустимым."),
                (0.4, False, "Есть риск ошибки в вычислении."),
                (0.2, False, "Финальный итог неверен."),
            ],
            self.evaluator,
        )
        self.assertEqual(trace.step_count, 3)
        self.assertLess(trace.supported_steps, trace.step_count)
        self.assertGreater(trace.overconfidence_rate, 0.0)
        self.assertTrue(any(not step.is_supported for step in trace.steps))
        self.assertLess(trace.llm_judge_agreement, 1.0)
        self.assertGreater(trace.brier_score, 0.0)
        self.assertGreaterEqual(trace.expected_calibration_error, 0.0)
        self.assertGreaterEqual(trace.f1, 0.0)

    def test_analysis_judge_rewards_uncertainty_and_follow_up(self) -> None:
        weak_score, weak_supported, _ = judge_step_support(
            self.analysis_case,
            "Сервис полезен и точно даст эффект.",
            is_final=True,
            cumulative_score=0.35,
            marginal_gain=0.0,
            support_tolerance=0.01,
            final_quality_threshold=0.8,
        )
        strong_score, strong_supported, rationale = judge_step_support(
            self.analysis_case,
            "Однако есть риск ошибки, неопределённость остаётся высокой. Вывод: запускать как пилот и проверить спрос.",
            is_final=True,
            cumulative_score=0.88,
            marginal_gain=0.2,
            support_tolerance=0.01,
            final_quality_threshold=0.8,
        )
        self.assertLess(weak_score, strong_score)
        self.assertFalse(weak_supported)
        self.assertTrue(strong_supported)
        self.assertIn("неопредел", rationale.lower())

    def test_parse_step_judgments_reads_supported_flags(self) -> None:
        raw_text = '{"steps": [{"step_index": 1, "score": 0.8, "supported": true, "rationale": "ok"}]}'
        parsed = parse_step_judgments(raw_text, 2)
        self.assertEqual(parsed[0], (0.8, True, "ok"))
        self.assertEqual(parsed[1][0], 0.5)
        self.assertFalse(parsed[1][1])

    def test_trace_computes_perfect_dual_judge_metrics_when_labels_match(self) -> None:
        answer = "1. Шаг: 37 * 24 = 888.\n2. Итог: 730."
        trace = build_proxy_meco_trace(
            self.case,
            answer,
            [
                (0.9, "Уверен в промежуточном вычислении."),
                (0.95, "Уверен в финальном результате."),
            ],
            [
                (0.9, True, "Шаг корректен."),
                (0.95, True, "Финальный шаг корректен."),
            ],
            self.evaluator,
        )
        self.assertEqual(trace.llm_judge_agreement, 1.0)
        self.assertEqual(trace.precision, 1.0)
        self.assertEqual(trace.recall, 1.0)
        self.assertEqual(trace.f1, 1.0)

    def test_aggregate_proxy_samples_tracks_variance_and_consistency(self) -> None:
        confidence_pairs, llm_judgments, confidence_std_mean, llm_judge_consistency, sample_count = aggregate_proxy_samples(
            [
                [(0.2, "a"), (0.8, "b")],
                [(0.4, "a"), (0.8, "b")],
                [(0.6, "a"), (0.8, "b")],
            ],
            [
                [(0.3, False, "x"), (0.9, True, "y")],
                [(0.6, True, "x"), (0.9, True, "y")],
                [(0.5, True, "x"), (0.8, True, "y")],
            ],
        )
        self.assertEqual(sample_count, 3)
        self.assertEqual(len(confidence_pairs), 2)
        self.assertGreater(confidence_std_mean, 0.0)
        self.assertLess(llm_judge_consistency, 1.0)
        self.assertTrue(llm_judgments[0][1])


if __name__ == "__main__":
    unittest.main()
