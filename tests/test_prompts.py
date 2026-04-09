from __future__ import annotations

import unittest

from mkpi_app.prompts import build_critique_prompt, build_draft_prompt, build_revision_prompt
from mkpi_app.schemas import CritiqueResult, DatasetCase


class PromptPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = DatasetCase(
            id="analysis-advanced",
            bucket="analysis",
            title="Сложный аналитический кейс",
            prompt="Оцени внедрение ИИ-системы в чувствительный домен.",
            reference="Нужен взвешенный разбор.",
            constraints=["риски", "неопределённость", "вывод"],
            scoring_spec={"required_keywords": ["риски", "неопределённость"], "min_words": 80},
        )

    def test_draft_prompt_includes_quality_frame_and_anti_errors(self) -> None:
        _, user_prompt = build_draft_prompt(self.case, ["Если данных мало, то обозначай неопределённость."])
        self.assertIn("Критерии качества", user_prompt)
        self.assertIn("антиошиб", user_prompt.lower())
        self.assertIn("неопредел", user_prompt.lower())

    def test_critique_prompt_requests_failure_hypothesis_and_uncertainty(self) -> None:
        _, user_prompt = build_critique_prompt(self.case, "Черновой ответ")
        self.assertIn("failure_hypothesis", user_prompt)
        self.assertIn("uncertainty_level", user_prompt)
        self.assertIn("unresolved_questions", user_prompt)

    def test_revision_prompt_includes_failure_hypothesis_and_unresolved_questions(self) -> None:
        critique = CritiqueResult(
            summary="Нужна доработка",
            failure_hypothesis="hidden_constraint",
            issues=["Не отражены ограничения"],
            checks=["Проверь ограничения"],
            unresolved_questions=["Нужны ли дополнительные данные?"],
            uncertainty_level="high",
            needs_revision=True,
            estimated_quality=0.4,
        )
        _, user_prompt = build_revision_prompt(self.case, "Черновой ответ", critique)
        self.assertIn("Гипотеза о типе сбоя", user_prompt)
        self.assertIn("Непроверенные или спорные места", user_prompt)
        self.assertIn("неопредел", user_prompt.lower())


if __name__ == "__main__":
    unittest.main()
