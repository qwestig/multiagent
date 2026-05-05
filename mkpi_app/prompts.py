from __future__ import annotations

from .schemas import CritiqueResult, DatasetCase

BASE_SYSTEM_PROMPT = """Ты — исследовательский агент метакогнитивной промпт-инженерии.
Работай на русском языке.
Для сложных задач пиши прозрачно и структурированно.
Не выдумывай факты, если данных недостаточно: явно отмечай допущения.
Соблюдай метакогнитивную честность: обозначай неопределённость, противоречия и границы знания.
Если проверка невозможна, не маскируй это уверенностью: либо запроси уточнение, либо зафиксируй риск."""


def _render_constraints(case: DatasetCase) -> str:
    if not case.constraints:
        return "Явных дополнительных ограничений нет."
    return "\n".join(f"- {item}" for item in case.constraints)


def _quality_frame(case: DatasetCase) -> str:
    if case.bucket == "math_logic":
        return "\n".join(
            [
                "- точный финальный результат",
                "- явные шаги вычисления",
                "- проверка скрытых условий и обратная сверка результата",
            ]
        )
    if case.bucket == "analysis":
        return "\n".join(
            [
                "- покрытие обязательных опорных понятий",
                "- явная фиксация рисков, ограничений и зон неопределённости",
                "- итоговый вывод без псевдоуверенности",
            ]
        )
    return "\n".join(
        [
            "- покрытие всех обязательных элементов плана",
            "- соблюдение запретов и ограничений",
            "- реалистичная пошаговая структура с контрольными точками",
        ]
    )


def build_baseline_prompt(case: DatasetCase) -> tuple[str, str]:
    return (
        BASE_SYSTEM_PROMPT,
        f"""Реши задачу без метакогнитивной итерации и без самокритики.

Название кейса: {case.title}
Корзина: {case.bucket}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Критерии качества:
{_quality_frame(case)}

Дай один финальный ответ.""",
    )


def build_draft_prompt(case: DatasetCase, anti_error_rules: list[str]) -> tuple[str, str]:
    memory_block = (
        "\n".join(f"- {rule}" for rule in anti_error_rules)
        if anti_error_rules
        else "- Пока нет накопленных антиошибок."
    )
    return (
        BASE_SYSTEM_PROMPT,
        f"""Выполни задачу в режиме первичного ответа для техники "метакоррекция и итерация".

Название кейса: {case.title}
Корзина: {case.bucket}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Активные антиошибки из прошлых интерактивных запусков:
{memory_block}

Критерии качества:
{_quality_frame(case)}

Сделай первичный ответ в явных шагах.
В начале кратко зафиксируй цель и ключевые ограничения.
Если для решения нужны допущения, перечисли их явно.
Не проводи ещё полноценную самокритику, но можешь отметить 1-3 потенциально уязвимых места.""",
    )


def build_critique_prompt(case: DatasetCase, answer: str) -> tuple[str, str]:
    return (
        BASE_SYSTEM_PROMPT,
        f"""Проведи диагностику ответа по технике "метакоррекция и итерация".

Кейс: {case.title}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Критерии качества:
{_quality_frame(case)}

Текущий ответ:
{answer}

Верни только JSON-объект со схемой:
{{
  "summary": "краткое резюме проблем",
  "failure_hypothesis": "главная гипотеза типа сбоя: logical_gap | hidden_constraint | unsupported_assumption | factual_risk | decomposition_error | no_major_issue",
  "issues": ["список проблем"],
  "checks": ["3-5 проверок в порядке применения: пересчёт, обратная валидация, скрытое условие, фальсификация вывода и т.д."],
  "unresolved_questions": ["что нельзя надёжно проверить без дополнительных данных"],
  "uncertainty_level": "low | medium | high",
  "needs_revision": true,
  "estimated_quality": 0.0
}}

Правила критика:
- сначала определи наиболее вероятный тип сбоя;
- диагностируй не только явные ошибки, но и правдоподобную, но ненадёжную логику;
- проверки должны быть воспроизводимыми и привязанными к конкретным уязвимостям;
- если независимая проверка невозможна, не фантазируй: зафиксируй это в unresolved_questions и повысь uncertainty_level;
- если серьёзных замечаний нет, верни пустой список issues, meaningful checks, failure_hypothesis=no_major_issue и needs_revision=false.""",
    )


def build_revision_prompt(
    case: DatasetCase, answer: str, critique: CritiqueResult
) -> tuple[str, str]:
    issues = "\n".join(f"- {issue}" for issue in critique.issues) or "- Существенных проблем не найдено."
    checks = "\n".join(f"- {check}" for check in critique.checks) or "- Явных проверок нет."
    unresolved = (
        "\n".join(f"- {item}" for item in critique.unresolved_questions)
        or "- Не зафиксированы."
    )
    return (
        BASE_SYSTEM_PROMPT,
        f"""Исправь ответ локально, не переписывая корректные части без необходимости.

Кейс: {case.title}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Текущий ответ:
{answer}

Замечания критика:
{issues}

Гипотеза о типе сбоя:
{critique.failure_hypothesis or "не указана"}

Уровень неопределённости:
{critique.uncertainty_level}

Проверки, которые надо применить:
{checks}

Непроверенные или спорные места:
{unresolved}

Критерии качества:
{_quality_frame(case)}

Инструкция на исправление:
- применяй проверки последовательно;
- исправляй только уязвимые фрагменты, не переписывая корректные части без причины;
- если после исправления остаётся неопределённость, явно обозначь её и не делай категоричный вывод;
- если задача опирается на формальные условия, добавь короткую контрольную сверку результата.

Верни улучшенный ответ целиком.""",
    )


def build_protocol_prompt(
    case: DatasetCase, answer: str, critique: CritiqueResult
) -> tuple[str, str]:
    issues = ", ".join(critique.issues) if critique.issues else "существенных ошибок не найдено"
    return (
        BASE_SYSTEM_PROMPT,
        f"""Сформулируй короткое правило-антиошибку для похожих задач.

Кейс: {case.title}
Проблемы: {issues}
Гипотеза сбоя: {critique.failure_hypothesis or "не указана"}
Уровень неопределённости: {critique.uncertainty_level}
Актуальный ответ:
{answer}

Верни одну короткую строку в формате:
Если [паттерн ошибки или сигнал риска], то проверяй [конкретная процедура]; если данных недостаточно, [эскалация или признание неопределённости].""",
    )


def build_self_eval_prompt(case: DatasetCase, answer: str, steps: list[str]) -> tuple[str, str]:
    step_block = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1)) or "- Шаги не выделены."
    return (
        BASE_SYSTEM_PROMPT,
        f"""Оцени уверенность в уже готовом ответе по шагам. Не исправляй сам ответ и не переписывай шаги.

Кейс: {case.title}
Корзина: {case.bucket}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Шаги ответа для оценки:
{step_block}

Полный ответ:
{answer}

Верни только JSON-объект вида:
{{
  "steps": [
    {{"step_index": 1, "confidence": 0.0, "rationale": "краткое объяснение"}}
  ]
}}

Правила:
- confidence задавай в диапазоне от 0.0 до 1.0;
- высокая уверенность допустима только если шаг хорошо опирается на условия задачи;
- если шаг зависит от спорного допущения, явно снижай confidence;
- не меняй число шагов и оцени каждый шаг по порядку.""",
    )


def build_step_judge_prompt(case: DatasetCase, answer: str, steps: list[str]) -> tuple[str, str]:
    step_block = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1)) or "- Шаги не выделены."
    return (
        BASE_SYSTEM_PROMPT,
        f"""Проведи независимую step-level оценку уже готового ответа. Не переписывай решение и не исправляй его.

Кейс: {case.title}
Корзина: {case.bucket}
Задача:
{case.prompt}

Ограничения:
{_render_constraints(case)}

Критерии качества:
{_quality_frame(case)}

Шаги ответа для оценки:
{step_block}

Полный ответ:
{answer}

Верни только JSON-объект вида:
{{
  "steps": [
    {{
      "step_index": 1,
      "score": 0.0,
      "supported": true,
      "rationale": "краткое объяснение, почему шаг поддержан или нет"
    }}
  ]
}}

Правила:
- score задавай в диапазоне от 0.0 до 1.0;
- supported=true только если шаг содержательно поддержан условиями задачи и не подрывает финальный вывод;
- если шаг выглядит правдоподобно, но не верифицируется, понижай score и объясняй риск;
- для финального шага особенно учитывай соответствие результата ограничениям и ожидаемому типу вывода;
- оцени каждый шаг по порядку и не меняй число шагов.""",
    )
