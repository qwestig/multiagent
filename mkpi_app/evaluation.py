from __future__ import annotations

import re

from .schemas import DatasetCase, ScoreBreakdown, ScoreComponent
from .utils import (
    clamp,
    contains_any,
    has_close_numeric_value,
    keyword_coverage,
    marker_group_coverage,
    normalize_text,
    scaled_ratio,
)


class Evaluator:
    ANALYSIS_WEIGHTS = {
        "keyword_coverage": 0.25,
        "section_coverage": 0.2,
        "balance": 0.15,
        "uncertainty": 0.15,
        "follow_up": 0.15,
        "depth": 0.1,
    }
    PLANNING_WEIGHTS = {
        "must_include": 0.35,
        "forbidden_terms": 0.2,
        "structure": 0.15,
        "governance": 0.2,
        "quality_markers": 0.1,
    }

    def score(self, case: DatasetCase, answer: str) -> ScoreBreakdown:
        if case.bucket == "math_logic":
            return self._score_math_logic(case, answer)
        if case.bucket == "analysis":
            return self._score_analysis(case, answer)
        return self._score_constraint_planning(case, answer)

    def _weighted_average(self, components: list[ScoreComponent], weights: dict[str, float]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for component in components:
            weight = weights.get(component.name, 1.0)
            weighted_sum += component.value * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return clamp(weighted_sum / total_weight)

    def _score_math_logic(self, case: DatasetCase, answer: str) -> ScoreBreakdown:
        components: list[ScoreComponent] = []
        notes: list[str] = []

        expected = case.scoring_spec.get("numeric_expected")
        exact = 0.0
        if isinstance(expected, (int, float)) and has_close_numeric_value(answer, float(expected)):
            exact = 1.0
        elif normalize_text(case.reference) in normalize_text(answer):
            exact = 1.0
        elif isinstance(expected, (int, float)) and has_close_numeric_value(answer, float(expected), tolerance=1.0):
            exact = 0.4
            notes.append("Числовой итог близок к ожидаемому, но не точен.")
        else:
            notes.append("Не найден точный числовой итог.")
        components.append(ScoreComponent("accuracy", exact, "Точность финального ответа"))

        must_include = list(case.scoring_spec.get("must_include", []))
        passed, total, found = keyword_coverage(answer, must_include)
        coverage = 1.0 if total == 0 else passed / total
        components.append(
            ScoreComponent(
                "checks_coverage",
                coverage,
                f"Покрытие обязательных маркеров: {', '.join(found) if found else 'нет'}",
            )
        )

        structure = 1.0 if contains_any(answer, ["шаг", "итог", "вывод"]) else 0.4
        components.append(
            ScoreComponent("structure", structure, "Наличие явной структуры решения")
        )

        overall = clamp(sum(component.value for component in components) / len(components))
        return ScoreBreakdown(
            overall=overall,
            components=components,
            passed_checks=passed + int(exact == 1.0),
            total_checks=total + 1,
            notes=notes,
        )

    def _score_analysis(self, case: DatasetCase, answer: str) -> ScoreBreakdown:
        components: list[ScoreComponent] = []
        notes: list[str] = []
        spec = case.scoring_spec

        required_keywords = list(spec.get("required_keywords", []))
        passed, total, found = keyword_coverage(answer, required_keywords)
        keyword_score = 1.0 if total == 0 else passed / total
        components.append(
            ScoreComponent(
                "keyword_coverage",
                keyword_score,
                f"Опорные понятия: {', '.join(found) if found else 'не найдены'}",
            )
        )

        required_sections = spec.get("required_sections")
        if not required_sections:
            required_sections = {f"constraint_{index}": item for index, item in enumerate(case.constraints, start=1)}
        passed_sections, total_sections, found_sections = marker_group_coverage(answer, required_sections)
        section_score = 1.0 if total_sections == 0 else passed_sections / total_sections
        components.append(
            ScoreComponent(
                "section_coverage",
                section_score,
                f"Покрытые смысловые секции: {', '.join(found_sections) if found_sections else 'нет'}",
            )
        )

        balance_markers = list(spec.get("balance_markers", ["однако", "при этом", "с другой стороны"]))
        balance_hits, _, matched_balance = keyword_coverage(answer, balance_markers)
        balance_required = max(1, int(spec.get("min_balance_markers", 1)))
        balance_score = scaled_ratio(balance_hits, balance_required, minimum=0.0)
        components.append(
            ScoreComponent(
                "balance",
                balance_score,
                f"Маркеры баланса и контраргументации: {', '.join(matched_balance) if matched_balance else 'не найдены'}",
            )
        )

        uncertainty_markers = list(
            spec.get(
                "uncertainty_markers",
                ["неопредел", "недостаточно данных", "требует проверки", "неясно"],
            )
        )
        uncertainty_hits, _, matched_uncertainty = keyword_coverage(answer, uncertainty_markers)
        uncertainty_required = max(1, int(spec.get("min_uncertainty_markers", 1)))
        uncertainty_score = scaled_ratio(uncertainty_hits, uncertainty_required, minimum=0.0)
        overconfident_markers = list(spec.get("overconfident_markers", ["однозначно", "безусловно", "гарантированно"]))
        if contains_any(answer, overconfident_markers) and uncertainty_hits == 0:
            uncertainty_score = min(uncertainty_score, 0.2)
            notes.append("Ответ звучит слишком уверенно без явной калибровки неопределённости.")
        components.append(
            ScoreComponent(
                "uncertainty",
                uncertainty_score,
                f"Маркеры неопределённости: {', '.join(matched_uncertainty) if matched_uncertainty else 'не найдены'}",
            )
        )

        follow_up_markers = list(spec.get("follow_up_markers", ["проверить", "уточнить", "собрать данные", "пилот"]))
        follow_up_hits, _, matched_follow_up = keyword_coverage(answer, follow_up_markers)
        follow_up_required = max(1, int(spec.get("min_follow_up_markers", 1)))
        follow_up_score = scaled_ratio(follow_up_hits, follow_up_required, minimum=0.0)
        components.append(
            ScoreComponent(
                "follow_up",
                follow_up_score,
                f"Следующие шаги и верификация: {', '.join(matched_follow_up) if matched_follow_up else 'не найдены'}",
            )
        )

        min_words = int(spec.get("min_words", 70))
        length_score = scaled_ratio(len(answer.split()), min_words, minimum=0.2)
        if length_score < 1.0:
            notes.append("Ответ короче рекомендованного объёма.")
        components.append(
            ScoreComponent("depth", length_score, "Достаточность объёма и глубины анализа")
        )

        if section_score < 1.0:
            notes.append("Не все обязательные смысловые секции присутствуют.")
        if follow_up_score < 1.0:
            notes.append("Слабо выражен блок дополнительных проверок или следующего шага.")
        overall = self._weighted_average(components, self.ANALYSIS_WEIGHTS)
        return ScoreBreakdown(
            overall=overall,
            components=components,
            passed_checks=passed + passed_sections + min(balance_hits, balance_required) + min(uncertainty_hits, uncertainty_required) + min(follow_up_hits, follow_up_required),
            total_checks=total + total_sections + balance_required + uncertainty_required + follow_up_required,
            notes=notes,
        )

    def _score_constraint_planning(self, case: DatasetCase, answer: str) -> ScoreBreakdown:
        components: list[ScoreComponent] = []
        notes: list[str] = []
        spec = case.scoring_spec

        must_include = list(spec.get("must_include", []))
        passed, total, found = keyword_coverage(answer, must_include)
        inclusion_score = 1.0 if total == 0 else passed / total
        components.append(
            ScoreComponent(
                "must_include",
                inclusion_score,
                f"Найдены обязательные элементы: {', '.join(found) if found else 'нет'}",
            )
        )

        forbidden = list(spec.get("forbidden_terms", []))
        normalized = normalize_text(answer)
        violators = [item for item in forbidden if normalize_text(item) in normalized]
        forbidden_score = 1.0 if not forbidden else clamp(1 - (len(violators) / len(forbidden)))
        if violators:
            notes.append(f"Обнаружены запрещённые элементы: {', '.join(violators)}.")
        components.append(
            ScoreComponent(
                "forbidden_terms",
                forbidden_score,
                "Соблюдение прямых запретов в условиях",
            )
        )

        numbered_steps = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", answer))
        min_steps = int(spec.get("min_steps", 3))
        structure_score = scaled_ratio(numbered_steps, min_steps, minimum=0.2)
        components.append(
            ScoreComponent("structure", structure_score, f"Количество явных шагов: {numbered_steps}")
        )

        required_sections = spec.get("required_sections", {})
        matched_groups_count, total_groups, matched_groups = marker_group_coverage(answer, required_sections)
        governance_score = 1.0 if total_groups == 0 else matched_groups_count / total_groups
        components.append(
            ScoreComponent(
                "governance",
                governance_score,
                f"Контрольные и управляющие элементы: {', '.join(matched_groups) if matched_groups else 'не найдены'}",
            )
        )

        quality_markers = list(spec.get("quality_markers", ["проверка", "риск", "контрольная точка"]))
        quality_hits, _, matched_quality = keyword_coverage(answer, quality_markers)
        quality_required = max(1, int(spec.get("min_quality_markers", 2)))
        quality_score = scaled_ratio(quality_hits, quality_required, minimum=0.0)
        components.append(
            ScoreComponent(
                "quality_markers",
                quality_score,
                f"Маркеры качества и контролей: {', '.join(matched_quality) if matched_quality else 'не найдены'}",
            )
        )

        if governance_score < 1.0:
            notes.append("В плане не хватает управляющих элементов: контрольной точки, ответственного или эскалации.")
        overall = self._weighted_average(components, self.PLANNING_WEIGHTS)
        return ScoreBreakdown(
            overall=overall,
            components=components,
            passed_checks=passed + (len(forbidden) - len(violators)) + matched_groups_count + min(quality_hits, quality_required) + min(numbered_steps, min_steps),
            total_checks=total + len(forbidden) + total_groups + quality_required + min_steps,
            notes=notes,
        )
