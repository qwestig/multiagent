from __future__ import annotations

import unittest

from mkpi_app.evaluation import Evaluator
from mkpi_app.schemas import DatasetCase


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator()

    def test_math_case_scores_exact_answer_high(self) -> None:
        case = DatasetCase(
            id="math-1",
            bucket="math_logic",
            title="Простое вычисление",
            prompt="Вычисли 37 * 24 - 158.",
            reference="Шаги: 37 * 24 = 888, 888 - 158 = 730. Итог: 730.",
            constraints=["Покажи шаги", "Выдели итог"],
            scoring_spec={"numeric_expected": 730, "must_include": ["шаг", "итог"]},
        )
        score = self.evaluator.score(case, "Шаги: 37 * 24 = 888, 888 - 158 = 730. Итог: 730.")
        self.assertGreaterEqual(score.overall, 0.95)

    def test_analysis_case_penalizes_missing_keywords(self) -> None:
        case = DatasetCase(
            id="analysis-1",
            bucket="analysis",
            title="LLM в образовании",
            prompt="Оцени внедрение LLM в университет.",
            reference="Нужно обсудить академическую честность, персонализацию и верификацию.",
            constraints=["Сделай вывод"],
            scoring_spec={
                "required_keywords": ["академическая честность", "персонализация", "верификация"],
                "min_words": 20,
            },
        )
        weak = self.evaluator.score(case, "Технология полезна, но требует контроля.")
        strong = self.evaluator.score(
            case,
            "Академическая честность требует новых правил, персонализация помогает студентам, а верификация "
            "нужна для проверки фактов. Вывод: внедрение возможно только при прозрачном контроле.",
        )
        self.assertLess(weak.overall, strong.overall)

    def test_analysis_rewards_uncertainty_and_follow_up(self) -> None:
        case = DatasetCase(
            id="analysis-2",
            bucket="analysis",
            title="Пилот цифровой услуги",
            prompt="Оцени запуск цифровой услуги.",
            reference="",
            constraints=["Укажи риски, неопределённость и следующий шаг"],
            scoring_spec={
                "required_keywords": ["метрики", "риски", "спрос"],
                "required_sections": {
                    "benefits": ["полез", "эффект"],
                    "risks": ["риск"],
                    "uncertainty": ["неопредел", "недостаточно данных"],
                    "recommendation": ["вывод"],
                    "next_step": ["проверить", "пилот"],
                },
                "balance_markers": ["однако"],
                "uncertainty_markers": ["неопредел", "недостаточно данных"],
                "follow_up_markers": ["проверить", "пилот"],
                "min_words": 40,
            },
        )
        weak = self.evaluator.score(
            case,
            "Сервис полезен и точно даст эффект. Риски минимальны. Вывод: запускать сразу.",
        )
        strong = self.evaluator.score(
            case,
            "Сервис может дать полезный эффект и улучшить метрики спроса, однако есть риск ошибки "
            "в позиционировании. Неопределённость остаётся высокой, потому что данных о реальном спросе "
            "недостаточно. Вывод: запуск возможен только как пилот. Следующий шаг: проверить спрос и "
            "ключевые метрики на ограниченной группе пользователей.",
        )
        self.assertLess(weak.overall, strong.overall)
        weak_uncertainty = next(component for component in weak.components if component.name == "uncertainty")
        strong_uncertainty = next(component for component in strong.components if component.name == "uncertainty")
        self.assertLess(weak_uncertainty.value, strong_uncertainty.value)

    def test_constraint_planning_rewards_governance_and_reserve_path(self) -> None:
        case = DatasetCase(
            id="planning-1",
            bucket="constraint_planning",
            title="Релиз сервиса",
            prompt="Составь план релиза.",
            reference="",
            constraints=["Добавь контрольную точку и резервный сценарий"],
            scoring_spec={
                "must_include": ["тесты", "мониторинг", "откат"],
                "forbidden_terms": ["без проверки"],
                "required_sections": {
                    "checkpoint": ["контрольная точка", "критерий готовности"],
                    "rollback": ["резервный сценарий", "откат", "эскалация"],
                    "owner": ["ответственный", "координатор"],
                    "completion": ["итог", "готовность"],
                },
                "quality_markers": ["проверка", "риск", "контроль"],
                "min_quality_markers": 2,
                "min_steps": 4,
            },
        )
        weak = self.evaluator.score(
            case,
            "1. Подготовить релиз.\n2. Выпустить сервис.\nИтог: релиз завершён.",
        )
        strong = self.evaluator.score(
            case,
            "1. Проверить тесты и назначить ответственного за выпуск.\n"
            "2. Подготовить мониторинг и критерий готовности.\n"
            "3. Провести контрольную точку и оценить риск перед запуском.\n"
            "4. Если возникает сбой, включить резервный сценарий с откатом и эскалацией.\n"
            "Итог: релиз проходит через проверки и управляемую готовность.",
        )
        self.assertLess(weak.overall, strong.overall)
        weak_governance = next(component for component in weak.components if component.name == "governance")
        strong_governance = next(component for component in strong.components if component.name == "governance")
        self.assertLess(weak_governance.value, strong_governance.value)


if __name__ == "__main__":
    unittest.main()
