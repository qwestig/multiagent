from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9-]+")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))
    return collapsed


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def _common_prefix_length(left: str, right: str) -> int:
    prefix_length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        prefix_length += 1
    return prefix_length


def _tokens_similar(keyword_token: str, answer_token: str) -> bool:
    if keyword_token == answer_token:
        return True
    if len(keyword_token) >= 4 and answer_token.startswith(keyword_token):
        return True
    if len(answer_token) >= 4 and keyword_token.startswith(answer_token):
        return True

    min_length = min(len(keyword_token), len(answer_token))
    if min_length < 4:
        return False

    common_prefix = _common_prefix_length(keyword_token, answer_token)
    if min_length <= 6:
        threshold = 4
    elif min_length <= 8:
        threshold = 5
    else:
        threshold = min_length - 3
    return common_prefix >= threshold


def matches_marker(answer: str, marker: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_marker = normalize_text(marker)
    if not normalized_marker:
        return False
    if normalized_marker in normalized_answer:
        return True

    marker_tokens = tokenize_text(normalized_marker)
    answer_tokens = tokenize_text(normalized_answer)
    if not marker_tokens or not answer_tokens:
        return False

    return all(
        any(_tokens_similar(marker_token, answer_token) for answer_token in answer_tokens)
        for marker_token in marker_tokens
    )


def keyword_coverage(answer: str, keywords: list[str]) -> tuple[int, int, list[str]]:
    found = [keyword for keyword in keywords if matches_marker(answer, keyword)]
    return len(found), len(keywords), found


def marker_group_coverage(
    answer: str,
    groups: dict[str, str | list[str]] | list[str | list[str]] | None,
) -> tuple[int, int, list[str]]:
    if not groups:
        return 0, 0, []
    matched: list[str] = []
    total = 0
    if isinstance(groups, dict):
        items = groups.items()
    else:
        items = []
        for index, value in enumerate(groups, start=1):
            label = value if isinstance(value, str) else f"group_{index}"
            items.append((str(label), value))
    for label, value in items:
        total += 1
        variants = [value] if isinstance(value, str) else list(value)
        if any(matches_marker(answer, variant) for variant in variants):
            matched.append(str(label))
    return len(matched), total, matched


def contains_any(answer: str, items: list[str]) -> bool:
    return any(matches_marker(answer, item) for item in items)


def count_lines_with_prefixes(text: str, prefixes: list[str]) -> int:
    count = 0
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    for line in lines:
        if any(line.startswith(prefix.lower()) for prefix in prefixes):
            count += 1
    return count


def scaled_ratio(value: int, target: int, minimum: float = 0.0) -> float:
    if target <= 0:
        return 1.0
    return clamp(max(minimum, value / target))


def extract_numeric_tokens(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(" ", "")):
        values.append(float(token.replace(",", ".")))
    return values


def has_close_numeric_value(text: str, expected: float, tolerance: float = 0.01) -> bool:
    return any(abs(value - expected) <= tolerance for value in extract_numeric_tokens(text))


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def make_slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "run"
