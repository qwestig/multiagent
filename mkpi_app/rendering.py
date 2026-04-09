from __future__ import annotations

import re

DISPLAY_MATH_PATTERN = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
BRACKET_ENV_PATTERN = re.compile(
    r"\[\s*(\\begin\{(?:aligned|align\*?|gather\*?|cases|matrix|pmatrix|bmatrix)\}.*?\\end\{(?:aligned|align\*?|gather\*?|cases|matrix|pmatrix|bmatrix)\})\s*\]",
    flags=re.DOTALL,
)
BACKSLASH_BLOCK_PATTERN = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)
INLINE_BLOCK_PATTERN = re.compile(r"\\\((.*?)\\\)", flags=re.DOTALL)


def normalize_latex_body(body: str) -> str:
    normalized = body.strip()
    if "\\begin{aligned}" in normalized or "\\begin{align" in normalized:
        normalized = re.sub(r"(?<!\\)\\\s*&", r"\\\\ &", normalized)
    return normalized


def normalize_math_markdown(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    normalized = BRACKET_ENV_PATTERN.sub(
        lambda match: f"$$\n{normalize_latex_body(match.group(1))}\n$$",
        normalized,
    )
    normalized = BACKSLASH_BLOCK_PATTERN.sub(
        lambda match: f"$$\n{normalize_latex_body(match.group(1))}\n$$",
        normalized,
    )
    normalized = INLINE_BLOCK_PATTERN.sub(
        lambda match: f"${match.group(1).strip()}$",
        normalized,
    )
    normalized = DISPLAY_MATH_PATTERN.sub(
        lambda match: f"$$\n{normalize_latex_body(match.group(1))}\n$$",
        normalized,
    )
    return normalized


def split_markdown_and_math(text: str) -> list[tuple[str, str]]:
    normalized = normalize_math_markdown(text)
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in DISPLAY_MATH_PATTERN.finditer(normalized):
        if match.start() > cursor:
            before = normalized[cursor : match.start()]
            if before.strip():
                parts.append(("text", before))
        math_payload = normalize_latex_body(match.group(1))
        if math_payload.strip():
            parts.append(("math", math_payload))
        cursor = match.end()
    if cursor < len(normalized):
        tail = normalized[cursor:]
        if tail.strip():
            parts.append(("text", tail))
    return parts
