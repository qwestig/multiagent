from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mkpi_app import (
    DatasetCase,
    DemoModelAdapter,
    Evaluator,
    OpenAICompatibleAdapter,
    ProjectStorage,
    RunnerConfig,
    StopPolicy,
    TechniqueRunner,
    aggregate_runs,
)
from mkpi_app.models import OPENROUTER_BASE_URL, provider_configuration_hint
from mkpi_app.rendering import normalize_math_markdown, split_markdown_and_math

PROJECT_ROOT = Path(__file__).resolve().parent
storage = ProjectStorage.default(PROJECT_ROOT)
evaluator = Evaluator()


def build_adapter(
    adapter_mode: str, model_name: str, api_key: str, api_base_url: str
):
    if adapter_mode == "demo":
        return DemoModelAdapter(model=model_name or "demo-meta-correction")
    return OpenAICompatibleAdapter(
        model=model_name,
        api_key=api_key or None,
        base_url=api_base_url or None,
    )


def load_cases() -> list[DatasetCase]:
    return storage.load_dataset()


def build_custom_case(
    bucket: str,
    title: str,
    prompt: str,
    reference: str,
    constraints_raw: str,
    scoring_raw: str,
) -> DatasetCase:
    constraints = [line.strip() for line in constraints_raw.splitlines() if line.strip()]
    try:
        scoring_spec = json.loads(scoring_raw) if scoring_raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"scoring_spec должен быть валидным JSON: {exc.msg}") from exc
    return DatasetCase(
        id="custom-interactive-case",
        bucket=bucket,
        title=title or "Пользовательский кейс",
        prompt=prompt,
        reference=reference,
        constraints=constraints,
        scoring_spec=scoring_spec,
    )


def render_scorecard(title: str, score: float, delta: float | None = None) -> None:
    if delta is None:
        st.metric(title, f"{score:.3f}")
    else:
        st.metric(title, f"{score:.3f}", delta=f"{delta:+.3f}")


def render_markdown_block(
    title: str,
    content: str,
    *,
    show_source: bool = False,
    empty_label: str = "_Пусто_",
) -> None:
    st.markdown(f"**{title}**")
    normalized = content.strip()
    if normalized:
        parts = split_markdown_and_math(normalized)
        if parts:
            for part_type, payload in parts:
                if part_type == "math":
                    st.latex(payload)
                else:
                    st.markdown(payload)
        else:
            st.markdown(normalize_math_markdown(normalized))
    else:
        st.markdown(empty_label)
    if show_source and normalized:
        with st.expander(f"Показать исходный Markdown: {title}", expanded=False):
            st.code(content, language="markdown")


def render_markdown_list(title: str, items: list[str], *, show_source: bool = False) -> None:
    content = "\n".join(f"- {item}" for item in items) if items else ""
    render_markdown_block(title, content, show_source=show_source, empty_label="_Список пуст_")


def render_run(run, *, render_markdown: bool = True, show_markdown_source: bool = False) -> None:
    delta = run.final_score.overall - run.baseline_score.overall
    first_row, second_row, third_row = st.columns(3)
    with first_row:
        render_scorecard("Baseline", run.baseline_score.overall)
    with second_row:
        render_scorecard("MKPI", run.final_score.overall, delta)
    with third_row:
        st.metric("Итерации", len(run.iterations))

    left, right = st.columns(2)
    with left:
        st.subheader("Baseline-ответ")
        if render_markdown:
            render_markdown_block(
                "Baseline",
                run.baseline_answer,
                show_source=show_markdown_source,
            )
        else:
            st.write(run.baseline_answer)
    with right:
        st.subheader("Итог после метакоррекции")
        if render_markdown:
            render_markdown_block(
                "Финальный ответ",
                run.final_answer,
                show_source=show_markdown_source,
            )
        else:
            st.write(run.final_answer)

    st.subheader("Итерации")
    for item in run.iterations:
        with st.expander(f"Итерация {item.iteration_index}", expanded=item.iteration_index == 1):
            if render_markdown:
                render_markdown_block(
                    "Черновик",
                    item.draft,
                    show_source=show_markdown_source,
                )
                render_markdown_block(
                    "Критика",
                    item.critique,
                    show_source=show_markdown_source,
                )
                render_markdown_block(
                    "Гипотеза сбоя",
                    item.failure_hypothesis,
                    show_source=show_markdown_source,
                    empty_label="_Не зафиксирована_",
                )
                render_markdown_block(
                    "Уровень неопределённости",
                    item.uncertainty_level,
                    show_source=show_markdown_source,
                    empty_label="_Не указан_",
                )
                render_markdown_list(
                    "Проверки",
                    item.checks,
                    show_source=show_markdown_source,
                )
                render_markdown_list(
                    "Непроверенные вопросы",
                    item.unresolved_questions,
                    show_source=show_markdown_source,
                )
                render_markdown_block(
                    "Исправленный ответ",
                    item.revision,
                    show_source=show_markdown_source,
                )
                render_markdown_block(
                    "Антиошибка",
                    item.anti_error_rule,
                    show_source=show_markdown_source,
                )
            else:
                st.markdown("**Черновик**")
                st.write(item.draft)
                st.markdown("**Критика**")
                st.write(item.critique)
                st.markdown("**Гипотеза сбоя**")
                st.write(item.failure_hypothesis)
                st.markdown("**Уровень неопределённости**")
                st.write(item.uncertainty_level)
                st.markdown("**Проверки**")
                st.write(item.checks)
                st.markdown("**Непроверенные вопросы**")
                st.write(item.unresolved_questions)
                st.markdown("**Исправленный ответ**")
                st.write(item.revision)
                st.markdown("**Антиошибка**")
                st.code(item.anti_error_rule)
            st.caption(
                f"Оценка: {item.estimated_quality:.3f} | Токены: {item.tokens} | Задержка: {item.latency_ms} ms"
            )
            if item.stop_reason:
                st.info(item.stop_reason)

    st.subheader("Артефакты")
    if run.artifacts:
        for name, path in run.artifacts.items():
            st.write(f"`{name}` -> `{path}`")


def main() -> None:
    st.set_page_config(page_title="МКПИ: Метакоррекция и итерация", layout="wide")
    st.title("Метакогнитивная промпт-инженерия: метакоррекция и итерация")
    st.write(
        "Интерфейс для сравнения baseline-ответа и итеративной агентной техники без LangGraph. "
        "В batch-режиме перенос антиошибок отключается автоматически, чтобы benchmark оставался честным."
    )

    with st.sidebar:
        st.header("Конфигурация")
        adapter_mode = st.radio(
            "Режим модели",
            options=["demo", "openai-compatible"],
            format_func=lambda value: "Демо-адаптер" if value == "demo" else "Удалённый API",
        )
        model_name = st.text_input(
            "Модель",
            value="demo-meta-correction" if adapter_mode == "demo" else "gpt-4o-mini",
        )
        api_key = ""
        api_base_url = ""
        if adapter_mode == "openai-compatible":
            api_key = st.text_input(
                "API key",
                type="password",
                help="Если поле пустое, используется MKPI_API_KEY.",
            )
            api_base_url = st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                help="Поддерживается OpenAI-compatible endpoint /chat/completions.",
            )
            hint = provider_configuration_hint(api_key or None, api_base_url or None, model_name or None)
            if hint:
                st.info(hint)
            if model_name.strip().startswith("gpt-oss-"):
                st.caption(
                    f"Если используете OpenRouter для `gpt-oss-*`, обычно нужен `Base URL`: `{OPENROUTER_BASE_URL}`."
                )
        max_iterations = st.slider("Макс. итераций", min_value=1, max_value=5, value=3)
        improvement_threshold = st.slider(
            "Порог улучшения",
            min_value=0.0,
            max_value=0.2,
            value=0.03,
            step=0.01,
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.slider("Max tokens", min_value=256, max_value=4096, value=1200, step=128)
        persist_runs = st.checkbox("Сохранять JSON-прогоны", value=True)
        persist_anti_errors = st.checkbox(
            "Копить антиошибки в интерактивном режиме",
            value=True,
            help="В benchmark-режиме эти правила не применяются автоматически.",
        )
        render_markdown = st.checkbox(
            "Рендерить ответы как Markdown",
            value=True,
            help="Показывает ответы, итерации и антиошибки с markdown-форматированием.",
        )
        show_markdown_source = st.checkbox(
            "Показывать исходный Markdown",
            value=False,
            help="Добавляет раскрывающиеся блоки с исходным markdown-текстом.",
        )

    cases = load_cases()
    case_map = {f"{case.id} | {case.title}": case for case in cases}

    single_tab, batch_tab, history_tab = st.tabs(["Одиночный запуск", "Пакетный прогон", "История"])

    with single_tab:
        source_mode = st.radio(
            "Источник задания",
            options=["dataset", "custom"],
            format_func=lambda value: "Кейс из датасета" if value == "dataset" else "Свой кейс",
            horizontal=True,
        )
        if source_mode == "dataset":
            selected_key = st.selectbox("Выберите кейс", options=list(case_map.keys()))
            selected_case = case_map[selected_key]
        else:
            selected_case = None
            bucket = st.selectbox(
                "Корзина",
                options=["math_logic", "analysis", "constraint_planning"],
            )
            title = st.text_input("Название кейса", value="Пользовательский кейс")
            prompt = st.text_area("Задача", height=180)
            reference = st.text_area(
                "Эталонный ответ или краткий ориентир для оценки",
                height=120,
                help="Для custom-кейса оценка всё равно нужна; можно задать краткий ожидаемый итог.",
            )
            constraints_raw = st.text_area(
                "Ограничения, по одному на строке",
                height=120,
                help="Например: упомяни риски, добавь вывод, не используй ночную смену.",
            )
            scoring_raw = st.text_area(
                "JSON scoring_spec",
                value='{"required_keywords": ["риск", "вывод"], "min_words": 50}',
                height=120,
            )

        if st.button("Запустить сравнение", type="primary"):
            try:
                if source_mode == "custom":
                    selected_case = build_custom_case(
                        bucket,
                        title,
                        prompt,
                        reference,
                        constraints_raw,
                        scoring_raw,
                    )
                assert selected_case is not None
            except Exception as exc:
                st.error(str(exc))
                st.stop()
            adapter = build_adapter(adapter_mode, model_name, api_key, api_base_url)
            config = RunnerConfig(
                model_name=model_name,
                max_iterations=max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                improvement_threshold=improvement_threshold,
                mode="interactive",
                persist_runs=persist_runs,
                persist_anti_errors=persist_anti_errors,
            )
            runner = TechniqueRunner(
                model=adapter,
                evaluator=evaluator,
                storage=storage,
                stop_policy=StopPolicy(
                    max_iterations=max_iterations,
                    improvement_threshold=improvement_threshold,
                ),
            )
            try:
                run = runner.run(selected_case, config)
            except Exception as exc:
                st.error(str(exc))
            else:
                render_run(
                    run,
                    render_markdown=render_markdown,
                    show_markdown_source=show_markdown_source,
                )

    with batch_tab:
        selected_buckets = st.multiselect(
            "Корзины датасета",
            options=["math_logic", "analysis", "constraint_planning"],
            default=["math_logic", "analysis", "constraint_planning"],
        )
        benchmark_limit = st.number_input(
            "Сколько кейсов запустить",
            min_value=1,
            max_value=max(1, len(cases)),
            value=min(12, max(1, len(cases))),
            step=1,
        )
        if st.button("Запустить batch benchmark"):
            filtered_cases = [case for case in cases if case.bucket in selected_buckets][: benchmark_limit]
            adapter = build_adapter(adapter_mode, model_name, api_key, api_base_url)
            config = RunnerConfig(
                model_name=model_name,
                max_iterations=max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                improvement_threshold=improvement_threshold,
                mode="benchmark",
                persist_runs=persist_runs,
                persist_anti_errors=False,
            )
            runner = TechniqueRunner(
                model=adapter,
                evaluator=evaluator,
                storage=storage,
                stop_policy=StopPolicy(
                    max_iterations=max_iterations,
                    improvement_threshold=improvement_threshold,
                ),
            )
            try:
                runs, artifacts = runner.run_batch(filtered_cases, config)
            except Exception as exc:
                st.error(str(exc))
            else:
                summary = aggregate_runs(runs)
                first_row, second_row, third_row = st.columns(3)
                with first_row:
                    st.metric("Прогонов", summary["runs"])
                with second_row:
                    st.metric("Средний baseline", summary["baseline_mean"])
                with third_row:
                    st.metric("Средняя дельта", summary["delta_mean"])
                st.json(summary)
                for label, path in artifacts.items():
                    st.write(f"`{label}` -> `{path}`")
                    artifact_path = Path(path)
                    if artifact_path.exists():
                        st.download_button(
                            label=f"Скачать {artifact_path.name}",
                            data=artifact_path.read_text(encoding="utf-8"),
                            file_name=artifact_path.name,
                            mime="text/plain",
                        )

    with history_tab:
        run_files = storage.list_runs(limit=100)
        if not run_files:
            st.info("История пока пуста. Сначала выполните хотя бы один прогон.")
        else:
            selected_path = st.selectbox(
                "Сохранённые прогоны",
                options=run_files,
                format_func=lambda path: path.name,
            )
            run = storage.load_run(Path(selected_path))
            render_run(
                run,
                render_markdown=render_markdown,
                show_markdown_source=show_markdown_source,
            )
            st.subheader("Библиотека антиошибок")
            st.json(storage.load_anti_errors())


if __name__ == "__main__":
    main()
