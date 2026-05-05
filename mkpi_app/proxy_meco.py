from __future__ import annotations

import re
from statistics import mean, pstdev

from .evaluation import Evaluator
from .schemas import DatasetCase, ProxyMecoStep, ProxyMecoTrace
from .utils import (
    clamp,
    contains_any,
    extract_json_object,
    extract_numeric_tokens,
    has_close_numeric_value,
    keyword_coverage,
    marker_group_coverage,
    normalize_text,
)

_STEP_LINE_RE = re.compile(r"(?m)^\s*(?:шаг\s*\d+[:.)-]?|\d+[.)]|[-*])\s+(.+)$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_reasoning_steps(text: str) -> list[str]:
    matches = [item.strip() for item in _STEP_LINE_RE.findall(text) if item.strip()]
    if matches:
        return matches[:8]

    paragraphs = [item.strip() for item in text.splitlines() if item.strip()]
    if len(paragraphs) > 1:
        return paragraphs[:8]

    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text.strip()) if item.strip()]
    if sentences:
        return sentences[:8]
    return [text.strip()] if text.strip() else []


def parse_step_confidences(raw_text: str, expected_count: int) -> list[tuple[float, str]]:
    payload = extract_json_object(raw_text) or {}
    raw_items = payload.get("steps", [])
    values: list[tuple[float, str]] = []
    for item in raw_items:
        confidence = clamp(float(item.get("confidence", 0.5)))
        rationale = str(item.get("rationale", "")).strip()
        values.append((confidence, rationale))
    while len(values) < expected_count:
        values.append((0.5, "Фолбэк: модель не вернула полную step-level самооценку."))
    return values[:expected_count]


def parse_step_judgments(raw_text: str, expected_count: int) -> list[tuple[float, bool, str]]:
    payload = extract_json_object(raw_text) or {}
    raw_items = payload.get("steps", [])
    values: list[tuple[float, bool, str]] = []
    for item in raw_items:
        score = clamp(float(item.get("score", 0.5)))
        supported = bool(item.get("supported", score >= 0.5))
        rationale = str(item.get("rationale", "")).strip()
        values.append((score, supported, rationale))
    while len(values) < expected_count:
        values.append((0.5, False, "Фолбэк: модель не вернула полную step-level judge оценку."))
    return values[:expected_count]


def _compute_prefix_scores(case: DatasetCase, steps: list[str], evaluator: Evaluator) -> list[float]:
    prefix_scores: list[float] = []
    prefix_parts: list[str] = []
    for step in steps:
        prefix_parts.append(step)
        prefix_scores.append(evaluator.score(case, "\n".join(prefix_parts)).overall)
    return prefix_scores


def _judge_math_step(case: DatasetCase, step: str, *, is_final: bool) -> tuple[float, str]:
    score = 0.0
    notes: list[str] = []
    normalized_step = normalize_text(step)

    expected = case.scoring_spec.get("numeric_expected")
    numeric_tokens = extract_numeric_tokens(step)
    if numeric_tokens:
        score += 0.35
        notes.append("Шаг содержит проверяемые числовые опоры.")
    else:
        notes.append("В шаге нет явных числовых опор.")

    must_include = list(case.scoring_spec.get("must_include", []))
    found, total, _ = keyword_coverage(step, must_include)
    if total:
        coverage = found / total
        score += 0.25 * coverage
        if coverage > 0:
            notes.append("Частично покрыты обязательные маркеры решения.")

    if contains_any(step, ["шаг", "итог", "вывод", "проверка"]):
        score += 0.15
        notes.append("Есть структурный маркер рассуждения.")

    if isinstance(expected, (int, float)) and has_close_numeric_value(step, float(expected), tolerance=0.01):
        score += 0.25
        notes.append("Шаг согласован с ожидаемым числовым итогом.")
    elif is_final and isinstance(expected, (int, float)):
        notes.append("Финальный шаг не подтверждает ожидаемый числовой итог.")

    return clamp(score), " ".join(notes)


def _judge_analysis_step(case: DatasetCase, step: str, *, is_final: bool) -> tuple[float, str]:
    score = 0.0
    notes: list[str] = []
    spec = case.scoring_spec

    required_keywords = list(spec.get("required_keywords", []))
    found, total, _ = keyword_coverage(step, required_keywords)
    if total:
        keyword_score = found / total
        score += 0.3 * keyword_score
        if keyword_score > 0:
            notes.append("Шаг опирается на обязательные понятия из scoring spec.")

    required_sections = spec.get("required_sections", {})
    matched_sections, total_sections, _ = marker_group_coverage(step, required_sections)
    if total_sections:
        section_score = matched_sections / total_sections
        score += 0.25 * section_score
        if section_score > 0:
            notes.append("Шаг покрывает часть обязательных смысловых секций.")

    uncertainty_markers = list(spec.get("uncertainty_markers", ["неопредел", "недостаточно данных", "требует проверки"]))
    if contains_any(step, uncertainty_markers):
        score += 0.15
        notes.append("Шаг явно калибрует неопределённость.")

    follow_up_markers = list(spec.get("follow_up_markers", ["проверить", "уточнить", "пилот", "собрать данные"]))
    if contains_any(step, follow_up_markers):
        score += 0.15
        notes.append("Шаг добавляет следующий шаг проверки или верификации.")

    balance_markers = list(spec.get("balance_markers", ["однако", "при этом", "с другой стороны"]))
    if contains_any(step, balance_markers):
        score += 0.1
        notes.append("Шаг содержит балансировку или контраргумент.")

    if is_final and contains_any(step, ["вывод", "итог", "рекоменду"]):
        score += 0.05
        notes.append("Финальный шаг оформлен как явный вывод.")

    return clamp(score), " ".join(notes) or "Шаг слабо покрывает исследовательские критерии."


def _judge_constraint_planning_step(case: DatasetCase, step: str, *, is_final: bool) -> tuple[float, str]:
    score = 0.0
    notes: list[str] = []
    spec = case.scoring_spec

    must_include = list(spec.get("must_include", []))
    found, total, _ = keyword_coverage(step, must_include)
    if total:
        inclusion_score = found / total
        score += 0.3 * inclusion_score
        if inclusion_score > 0:
            notes.append("Шаг покрывает обязательные элементы плана.")

    forbidden = list(spec.get("forbidden_terms", []))
    if forbidden and not contains_any(step, forbidden):
        score += 0.15
        notes.append("Шаг не нарушает прямые запреты.")
    elif forbidden:
        notes.append("Шаг затрагивает запрещённые формулировки или действия.")

    required_sections = spec.get("required_sections", {})
    matched_sections, total_sections, _ = marker_group_coverage(step, required_sections)
    if total_sections:
        governance_score = matched_sections / total_sections
        score += 0.25 * governance_score
        if governance_score > 0:
            notes.append("Шаг содержит управляющие или контрольные секции.")

    quality_markers = list(spec.get("quality_markers", ["проверка", "риск", "контрольная точка"]))
    quality_hits, quality_total, _ = keyword_coverage(step, quality_markers)
    if quality_total:
        score += 0.15 * (quality_hits / quality_total)
        if quality_hits > 0:
            notes.append("Шаг содержит маркеры качества или риска.")

    if re.match(r"^\s*(?:\d+[.)]|[-*])\s+", step):
        score += 0.1
        notes.append("Шаг сохраняет явную процедурную структуру.")
    elif is_final and contains_any(step, ["итог", "готовность"]):
        score += 0.1
        notes.append("Финальный шаг оформляет завершение плана.")

    return clamp(score), " ".join(notes) or "Шаг слабо соответствует процедурным ограничениям."


def judge_step_support(
    case: DatasetCase,
    step: str,
    *,
    is_final: bool,
    cumulative_score: float,
    marginal_gain: float,
    support_tolerance: float,
    final_quality_threshold: float,
) -> tuple[float, bool, str]:
    if case.bucket == "math_logic":
        judge_score, rationale = _judge_math_step(case, step, is_final=is_final)
    elif case.bucket == "analysis":
        judge_score, rationale = _judge_analysis_step(case, step, is_final=is_final)
    else:
        judge_score, rationale = _judge_constraint_planning_step(case, step, is_final=is_final)

    combined_score = clamp((judge_score * 0.65) + (cumulative_score * 0.35))
    threshold = 0.65 if is_final else 0.5
    is_supported = combined_score >= threshold and marginal_gain >= -support_tolerance
    if is_final and cumulative_score < final_quality_threshold:
        is_supported = False
        rationale = f"{rationale} Итоговое качество ответа ниже порога для финального шага."
    return combined_score, is_supported, rationale


def _rank_sum_auroc(pairs: list[tuple[float, bool]]) -> float:
    positives = [confidence for confidence, label in pairs if label]
    negatives = [confidence for confidence, label in pairs if not label]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else 0.0


def _average_precision(pairs: list[tuple[float, bool]]) -> float:
    positives = sum(1 for _, label in pairs if label)
    if positives == 0:
        return 0.0
    ranked = sorted(pairs, key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / index
    return precision_sum / positives


def _agreement_rate(left: list[bool], right: list[bool]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    matches = sum(1 for left_item, right_item in zip(left, right) if left_item == right_item)
    return matches / len(left)


def _brier_score(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return 0.0
    squared_errors = [((confidence - (1.0 if label else 0.0)) ** 2) for confidence, label in pairs]
    return sum(squared_errors) / len(squared_errors)


def _expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 5) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    error = 0.0
    for bin_index in range(bins):
        left = bin_index / bins
        right = (bin_index + 1) / bins
        if bin_index == bins - 1:
            bucket = [(confidence, label) for confidence, label in pairs if left <= confidence <= right]
        else:
            bucket = [(confidence, label) for confidence, label in pairs if left <= confidence < right]
        if not bucket:
            continue
        confidence_mean = sum(confidence for confidence, _ in bucket) / len(bucket)
        accuracy_mean = sum(1.0 if label else 0.0 for _, label in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(confidence_mean - accuracy_mean)
    return error


def _classification_metrics(rule_labels: list[bool], llm_labels: list[bool]) -> tuple[float, float, float]:
    if not rule_labels or len(rule_labels) != len(llm_labels):
        return 0.0, 0.0, 0.0
    true_positive = sum(1 for rule, llm in zip(rule_labels, llm_labels) if rule and llm)
    false_positive = sum(1 for rule, llm in zip(rule_labels, llm_labels) if not rule and llm)
    false_negative = sum(1 for rule, llm in zip(rule_labels, llm_labels) if rule and not llm)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def aggregate_proxy_samples(
    confidence_samples: list[list[tuple[float, str]]],
    llm_judgment_samples: list[list[tuple[float, bool, str]]],
) -> tuple[list[tuple[float, str]], list[tuple[float, bool, str]], float, float, int]:
    if not confidence_samples or not llm_judgment_samples:
        return [], [], 0.0, 0.0, 0

    sample_count = min(len(confidence_samples), len(llm_judgment_samples))
    step_count = min(len(sample) for sample in confidence_samples[:sample_count])
    step_count = min(step_count, min(len(sample) for sample in llm_judgment_samples[:sample_count]))
    if step_count <= 0:
        return [], [], 0.0, 0.0, sample_count

    aggregated_confidences: list[tuple[float, str]] = []
    aggregated_judgments: list[tuple[float, bool, str]] = []
    confidence_std_values: list[float] = []
    llm_consistency_values: list[float] = []

    for step_index in range(step_count):
        step_confidences = [sample[step_index][0] for sample in confidence_samples[:sample_count]]
        step_confidence_rationale = confidence_samples[0][step_index][1]
        confidence_std_values.append(pstdev(step_confidences) if len(step_confidences) > 1 else 0.0)
        aggregated_confidences.append((sum(step_confidences) / len(step_confidences), step_confidence_rationale))

        step_judge_scores = [sample[step_index][0] for sample in llm_judgment_samples[:sample_count]]
        step_judge_labels = [sample[step_index][1] for sample in llm_judgment_samples[:sample_count]]
        step_judge_rationale = llm_judgment_samples[0][step_index][2]
        positive_votes = sum(1 for label in step_judge_labels if label)
        majority_label = positive_votes >= (len(step_judge_labels) / 2)
        majority_count = max(positive_votes, len(step_judge_labels) - positive_votes)
        llm_consistency_values.append(majority_count / len(step_judge_labels))
        aggregated_judgments.append((sum(step_judge_scores) / len(step_judge_scores), majority_label, step_judge_rationale))

    confidence_std_mean = sum(confidence_std_values) / len(confidence_std_values) if confidence_std_values else 0.0
    llm_judge_consistency = sum(llm_consistency_values) / len(llm_consistency_values) if llm_consistency_values else 0.0
    return aggregated_confidences, aggregated_judgments, confidence_std_mean, llm_judge_consistency, sample_count


def build_proxy_meco_trace(
    case: DatasetCase,
    answer: str,
    confidence_pairs: list[tuple[float, str]],
    llm_judgments: list[tuple[float, bool, str]],
    evaluator: Evaluator,
    support_tolerance: float = 0.01,
    final_quality_threshold: float = 0.8,
    sample_count: int = 1,
    confidence_std_mean: float = 0.0,
    llm_judge_consistency: float = 0.0,
) -> ProxyMecoTrace:
    steps = extract_reasoning_steps(answer)
    if not steps:
        return ProxyMecoTrace()

    prefix_scores = _compute_prefix_scores(case, steps, evaluator)
    records: list[ProxyMecoStep] = []
    previous_score = 0.0
    overconfident = 0
    ranked_pairs: list[tuple[float, bool]] = []
    rule_labels: list[bool] = []
    llm_labels: list[bool] = []
    dual_supported_steps = 0

    for index, step in enumerate(steps, start=1):
        score = prefix_scores[index - 1]
        marginal_gain = score - previous_score
        judge_score, is_supported, judge_rationale = judge_step_support(
            case,
            step,
            is_final=index == len(steps),
            cumulative_score=score,
            marginal_gain=marginal_gain,
            support_tolerance=support_tolerance,
            final_quality_threshold=final_quality_threshold,
        )
        confidence, rationale = confidence_pairs[index - 1] if index - 1 < len(confidence_pairs) else (0.5, "")
        llm_judge_score, llm_judge_supported, llm_judge_rationale = (
            llm_judgments[index - 1] if index - 1 < len(llm_judgments) else (0.5, False, "")
        )
        if confidence >= 0.7 and not is_supported:
            overconfident += 1
        ranked_pairs.append((confidence, is_supported))
        rule_labels.append(is_supported)
        llm_labels.append(llm_judge_supported)
        if is_supported and llm_judge_supported:
            dual_supported_steps += 1
        records.append(
            ProxyMecoStep(
                step_index=index,
                text=step,
                confidence=confidence,
                rationale=rationale,
                cumulative_score=round(score, 4),
                marginal_gain=round(marginal_gain, 4),
                is_supported=is_supported,
                judge_score=round(judge_score, 4),
                judge_rationale=judge_rationale,
                llm_judge_score=round(llm_judge_score, 4),
                llm_judge_supported=llm_judge_supported,
                llm_judge_rationale=llm_judge_rationale,
            )
        )
        previous_score = score

    confidences = [item.confidence for item in records]
    brier_score = _brier_score(ranked_pairs)
    expected_calibration_error = _expected_calibration_error(ranked_pairs)
    precision, recall, f1 = _classification_metrics(rule_labels, llm_labels)
    return ProxyMecoTrace(
        sample_count=sample_count,
        step_count=len(records),
        supported_steps=sum(1 for item in records if item.is_supported),
        confidence_mean=round(mean(confidences), 4),
        auroc=round(_rank_sum_auroc(ranked_pairs), 4),
        average_precision=round(_average_precision(ranked_pairs), 4),
        brier_score=round(brier_score, 4),
        expected_calibration_error=round(expected_calibration_error, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        overconfidence_rate=round(overconfident / len(records), 4),
        llm_judge_agreement=round(_agreement_rate(rule_labels, llm_labels), 4),
        confidence_std_mean=round(confidence_std_mean, 4),
        llm_judge_consistency=round(llm_judge_consistency, 4),
        dual_supported_steps=dual_supported_steps,
        steps=records,
    )
