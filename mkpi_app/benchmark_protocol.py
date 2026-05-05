from __future__ import annotations

from collections import defaultdict

from .schemas import DatasetCase


def score_proxy_meco_case(case: DatasetCase) -> float:
    score = 0.0
    spec = case.scoring_spec
    if case.bucket == "math_logic":
        if isinstance(spec.get("numeric_expected"), (int, float)):
            score += 0.5
        if spec.get("must_include"):
            score += 0.25
        if len(case.constraints) >= 2:
            score += 0.15
        if "шаг" in case.reference.lower() and "итог" in case.reference.lower():
            score += 0.1
        return score

    if case.bucket == "analysis":
        if spec.get("required_sections"):
            score += 0.3
        if spec.get("uncertainty_markers"):
            score += 0.2
        if spec.get("follow_up_markers"):
            score += 0.2
        if int(spec.get("min_words", 0)) >= 80:
            score += 0.15
        if spec.get("required_keywords"):
            score += 0.15
        return score

    if spec.get("required_sections"):
        score += 0.3
    if spec.get("forbidden_terms"):
        score += 0.2
    if spec.get("quality_markers"):
        score += 0.2
    if int(spec.get("min_steps", 0)) >= 4:
        score += 0.2
    if len(case.constraints) >= 2:
        score += 0.1
    return score


def is_proxy_meco_eligible(case: DatasetCase, minimum_score: float = 0.8) -> bool:
    return score_proxy_meco_case(case) >= minimum_score


def select_proxy_meco_subset(cases: list[DatasetCase], per_bucket: int = 4) -> list[DatasetCase]:
    buckets: dict[str, list[DatasetCase]] = defaultdict(list)
    for case in cases:
        if is_proxy_meco_eligible(case):
            buckets[case.bucket].append(case)

    selected: list[DatasetCase] = []
    for bucket in sorted(buckets):
        ranked = sorted(
            buckets[bucket],
            key=lambda item: (-score_proxy_meco_case(item), item.id),
        )
        selected.extend(ranked[: max(1, per_bucket)])
    return selected
