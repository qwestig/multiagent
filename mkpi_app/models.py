from __future__ import annotations

import json
import http.client
import os
import re
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .schemas import CritiqueResult, DatasetCase, ModelRequest, ModelResponse, UsageStats
from .utils import has_close_numeric_value, keyword_coverage, marker_group_coverage, normalize_text

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


def provider_configuration_hint(api_key: str | None, base_url: str | None, model: str | None) -> str | None:
    normalized_base_url = (base_url or OPENAI_BASE_URL).rstrip("/").lower()
    normalized_model = (model or "").strip()
    if api_key and api_key.startswith("sk-or-v1") and "api.openai.com" in normalized_base_url:
        return (
            "Похоже, вы используете ключ OpenRouter (`sk-or-v1...`) с OpenAI endpoint. "
            f"Поменяйте `Base URL` на `{OPENROUTER_BASE_URL}`."
        )
    if "openrouter.ai" in normalized_base_url and api_key and not api_key.startswith("sk-or-v1"):
        return (
            "Похоже, выбран endpoint OpenRouter, но ключ не похож на OpenRouter API key (`sk-or-v1...`). "
            "Проверьте, что ключ и провайдер совпадают."
        )
    if "api.openai.com" in normalized_base_url and normalized_model.startswith("gpt-oss-"):
        return (
            "Модель `gpt-oss-*` часто используется через OpenRouter или другой совместимый провайдер. "
            "Если получите ошибку доступа, сначала проверьте `Base URL` и тип ключа."
        )
    return None


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "openai-compatible",
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("MKPI_API_KEY")
        self.base_url = (base_url or os.getenv("MKPI_API_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.provider_name = provider_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("Не найден API key. Передайте его в UI или через MKPI_API_KEY.")
        configuration_hint = provider_configuration_hint(self.api_key, self.base_url, self.model)
        if configuration_hint and self.api_key.startswith("sk-or-v1") and "api.openai.com" in self.base_url.lower():
            raise RuntimeError(configuration_hint)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            data=encoded,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=120) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            message = f"Ошибка API {exc.code}: {details}"
            if exc.code == 401:
                hint = provider_configuration_hint(self.api_key, self.base_url, self.model)
                if hint:
                    message = f"{message}\n\nПодсказка: {hint}"
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Не удалось подключиться к API: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        message = raw_payload["choices"][0]["message"]["content"]
        if isinstance(message, list):
            parts = [item.get("text", "") for item in message if item.get("type") == "text"]
            text = "\n".join(part for part in parts if part)
        else:
            text = str(message)
        usage = raw_payload.get("usage", {})
        return ModelResponse(
            text=text.strip(),
            provider=self.provider_name,
            model=self.model,
            usage=UsageStats(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            latency_ms=latency_ms,
            raw=raw_payload,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.base_url.lower():
            app_url = os.getenv("MKPI_APP_URL", "http://localhost:8501")
            app_title = os.getenv("MKPI_APP_TITLE", "MKPI Meta-Correction")
            headers["HTTP-Referer"] = app_url
            headers["X-Title"] = app_title
        return headers


class YandexAIStudioAdapter(ModelAdapter):
    def __init__(
        self,
        model: str = "yandexgpt-lite",
        api_key: str | None = None,
        folder_id: str | None = None,
        provider_name: str = "yandex-ai-studio",
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("MKPI_API_KEY")
        self.folder_id = folder_id or os.getenv("MKPI_YANDEX_FOLDER_ID") or ""
        self.provider_name = provider_name
        self.max_retries = 3
        self.verbose = os.getenv("MKPI_YANDEX_VERBOSE", "").lower() in {"1", "true", "yes"}

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("Не найден API key для Yandex AI Studio.")
        if not self.folder_id:
            raise RuntimeError("Не найден folder ID для Yandex AI Studio. Передайте его явно или через MKPI_YANDEX_FOLDER_ID.")

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}/latest",
            "completionOptions": {
                "stream": False,
                "temperature": request.temperature,
                "maxTokens": str(request.max_tokens),
            },
            "messages": [
                {"role": "system", "text": request.system_prompt},
                {"role": "user", "text": request.user_prompt},
            ],
        }
        encoded = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            url=f"{YANDEX_BASE_URL}/completion",
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            data=encoded,
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        raw_payload: dict[str, Any] | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(http_request, timeout=120) as response:
                    raw_payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ошибка Yandex API {exc.code}: {details}") from exc
            except (urllib.error.URLError, http.client.RemoteDisconnected) as exc:
                last_error = exc
                if self.verbose:
                    print(
                        f"[yandex-api] retry {attempt}/{self.max_retries} after error: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if attempt >= self.max_retries:
                    break
                time.sleep(1.5 * attempt)

        if raw_payload is None:
            raise RuntimeError(f"Не удалось подключиться к Yandex API после {self.max_retries} попыток: {last_error}")

        latency_ms = int((time.perf_counter() - started) * 1000)
        alternatives = raw_payload.get("result", {}).get("alternatives", [])
        if not alternatives:
            raise RuntimeError(f"Yandex API вернул пустой список alternatives: {raw_payload}")
        text = str(alternatives[0].get("message", {}).get("text", "")).strip()
        usage = raw_payload.get("result", {}).get("usage", {})
        return ModelResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            usage=UsageStats(
                prompt_tokens=int(usage.get("inputTextTokens", 0)),
                completion_tokens=int(usage.get("completionTokens", 0)),
                total_tokens=int(usage.get("totalTokens", 0)),
            ),
            latency_ms=latency_ms,
            raw=raw_payload,
        )


class DemoModelAdapter(ModelAdapter):
    def __init__(self, model: str = "demo-meta-correction") -> None:
        self.model = model

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        metadata = request.metadata
        case = DatasetCase.from_dict(metadata["case"])
        phase = metadata.get("phase", "baseline")

        if phase == "baseline":
            text = self._make_imperfect_answer(case, variant="baseline")
        elif phase == "draft":
            text = self._make_imperfect_answer(case, variant="draft")
        elif phase == "critique":
            critique = self._critique(case, metadata["answer"])
            text = json.dumps(critique.to_dict(), ensure_ascii=False, indent=2)
        elif phase == "revision":
            text = self._revise(case, metadata["answer"])
        elif phase == "protocol":
            critique = CritiqueResult(**metadata["critique"])
            text = self._protocol(case, critique)
        elif phase == "self_eval":
            text = self._self_eval(case, metadata.get("steps", []), metadata.get("answer", ""))
        elif phase == "step_judge":
            text = self._step_judge(case, metadata.get("steps", []), metadata.get("answer", ""))
        else:
            text = case.reference

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = UsageStats(
            prompt_tokens=max(1, len(request.user_prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            total_tokens=max(2, len(request.user_prompt + text) // 4),
        )
        return ModelResponse(
            text=text,
            provider="demo",
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            raw={"phase": phase},
        )

    def _make_imperfect_answer(self, case: DatasetCase, variant: str) -> str:
        if case.bucket == "math_logic":
            expected = case.scoring_spec.get("numeric_expected")
            wrong = expected + 1 if isinstance(expected, (int, float)) else case.reference
            return (
                f"Шаги решения: выполнил вычисления по задаче «{case.title}».\n"
                f"Промежуточный вывод выглядит правдоподобно.\n"
                f"Итог: {wrong}"
            )
        if case.bucket == "analysis":
            keywords = list(case.scoring_spec.get("required_keywords", []))
            kept = keywords[:-1] if len(keywords) > 1 else keywords
            coverage = ", ".join(kept) if kept else "основные аспекты"
            return (
                f"Краткий анализ кейса «{case.title}».\n"
                f"Рассмотрены такие элементы, как {coverage}.\n"
                "Вывод: решение в целом приемлемо, но анализ можно усилить."
            )
        must_include = list(case.scoring_spec.get("must_include", []))
        selected = must_include[:-1] if len(must_include) > 1 else must_include
        bullet_lines = "\n".join(f"{index}. {item}" for index, item in enumerate(selected, start=1))
        return f"План по кейсу «{case.title}»:\n{bullet_lines}\nИтог: план требует доработки."

    def _critique(self, case: DatasetCase, answer: str) -> CritiqueResult:
        issues: list[str] = []
        checks: list[str] = [
            "Сделай контрольный пересчёт или повторную сверку ключевого вывода.",
            "Проведи обратную валидацию результата от финального ответа к исходным условиям.",
            "Проверь скрытое условие или пропущенное ограничение.",
        ]
        if case.bucket == "math_logic":
            expected = case.scoring_spec.get("numeric_expected")
            if isinstance(expected, (int, float)) and not has_close_numeric_value(answer, float(expected)):
                issues.append("Числовой итог не совпадает с ожидаемым значением.")
            if "итог" not in normalize_text(answer):
                issues.append("В ответе не выделен финальный результат.")
        elif case.bucket == "analysis":
            required = list(case.scoring_spec.get("required_keywords", []))
            found_count, total, found = keyword_coverage(answer, required)
            if found_count < total:
                missing = [item for item in required if item not in found]
                issues.append(f"Не раскрыты обязательные опорные понятия: {', '.join(missing)}.")
            section_hits, section_total, section_found = marker_group_coverage(
                answer,
                case.scoring_spec.get("required_sections", {}),
            )
            if section_hits < section_total:
                missing_sections = [
                    item
                    for item in case.scoring_spec.get("required_sections", {})
                    if item not in section_found
                ]
                issues.append(
                    f"Не покрыты обязательные смысловые секции: {', '.join(missing_sections)}."
                )

            uncertainty_markers = list(
                case.scoring_spec.get(
                    "uncertainty_markers",
                    ["неопредел", "недостаточно данных", "требует проверки"],
                )
            )
            uncertainty_hits, _, _ = keyword_coverage(answer, uncertainty_markers)
            if uncertainty_hits == 0:
                issues.append("Не зафиксированы зоны неопределённости и границы уверенности.")

            follow_up_markers = list(
                case.scoring_spec.get(
                    "follow_up_markers",
                    ["проверить", "уточнить", "пилот", "собрать данные"],
                )
            )
            follow_up_hits, _, _ = keyword_coverage(answer, follow_up_markers)
            if follow_up_hits == 0:
                issues.append("Нет следующего шага проверки гипотезы или верификации вывода.")

            minimum_words = int(case.scoring_spec.get("min_words", 70) * 0.6)
            if len(answer.split()) < minimum_words:
                issues.append("Анализ слишком краткий для исследовательского вывода.")
            checks.append("Задай вопрос на фальсификацию вывода и проверь альтернативное объяснение.")
        else:
            required = list(case.scoring_spec.get("must_include", []))
            found_count, total, found = keyword_coverage(answer, required)
            if found_count < total:
                missing = [item for item in required if item not in found]
                issues.append(f"План пропускает обязательные элементы: {', '.join(missing)}.")
            forbidden = list(case.scoring_spec.get("forbidden_terms", []))
            violators = [item for item in forbidden if normalize_text(item) in normalize_text(answer)]
            if violators:
                issues.append(f"План нарушает запреты: {', '.join(violators)}.")
            governance_hits, governance_total, governance_found = marker_group_coverage(
                answer,
                case.scoring_spec.get("required_sections", {}),
            )
            if governance_hits < governance_total:
                missing_sections = [
                    item
                    for item in case.scoring_spec.get("required_sections", {})
                    if item not in governance_found
                ]
                issues.append(
                    f"План не содержит всех управляющих секций: {', '.join(missing_sections)}."
                )
            quality_markers = list(case.scoring_spec.get("quality_markers", []))
            quality_hits, _, _ = keyword_coverage(answer, quality_markers)
            minimum_quality = max(1, int(case.scoring_spec.get("min_quality_markers", 2)))
            if quality_hits < minimum_quality:
                issues.append("В плане слабо выражены проверки, критерии готовности или управление риском.")
            numbered_steps = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", answer))
            if numbered_steps < int(case.scoring_spec.get("min_steps", 4)):
                issues.append("План недостаточно детализирован по шагам.")
            checks.append("Проверь покрытие всех ограничений и наличие реалистичных шагов исполнения.")

        summary = "Существенных проблем не найдено." if not issues else "Найдены уязвимости, нужна локальная коррекция."
        failure_hypothesis = "no_major_issue"
        unresolved_questions: list[str] = []
        uncertainty_level = "low"
        if issues:
            uncertainty_level = "medium"
        if case.bucket == "math_logic":
            if any("Числовой итог" in issue for issue in issues):
                failure_hypothesis = "logical_gap"
            elif any("итог" in issue.lower() for issue in issues):
                failure_hypothesis = "decomposition_error"
        elif case.bucket == "analysis":
            if any("понятия" in issue.lower() for issue in issues):
                failure_hypothesis = "hidden_constraint"
            elif any("смысловые секции" in issue.lower() for issue in issues):
                failure_hypothesis = "decomposition_error"
            elif any("неопредел" in issue.lower() for issue in issues):
                failure_hypothesis = "unsupported_assumption"
            elif any("краткий" in issue.lower() for issue in issues):
                failure_hypothesis = "unsupported_assumption"
            unresolved_questions = [
                "Достаточно ли данных для уверенного вывода по всем обязательным аспектам?",
                "Нужна ли дополнительная проверка альтернативной интерпретации?",
            ] if issues else []
            uncertainty_level = "high" if issues else "medium"
        else:
            if any("запрет" in issue.lower() for issue in issues):
                failure_hypothesis = "hidden_constraint"
            elif any("обязательные" in issue.lower() for issue in issues):
                failure_hypothesis = "decomposition_error"
            unresolved_questions = [
                "Все ли ограничения покрыты без внутренних противоречий?",
            ] if issues else []
        return CritiqueResult(
            summary=summary,
            failure_hypothesis=failure_hypothesis,
            issues=issues,
            checks=checks[:4],
            unresolved_questions=unresolved_questions,
            uncertainty_level=uncertainty_level,
            needs_revision=bool(issues),
            estimated_quality=0.95 if not issues else 0.55,
        )

    def _revise(self, case: DatasetCase, answer: str) -> str:
        if case.bucket == "math_logic":
            must_include = case.scoring_spec.get("must_include", ["итог"])
            closing = "\n".join(f"- Проверка: {item}" for item in must_include[:2])
            return f"{case.reference}\n{closing}"
        if case.bucket == "analysis":
            keywords = list(case.scoring_spec.get("required_keywords", []))
            joined = ", ".join(keywords)
            return (
                f"{case.reference}\n\n"
                f"Опорные понятия явно раскрыты: {joined}. "
                "Зоны неопределённости отмечены явно, а следующий шаг проверки сформулирован как отдельная верификация гипотезы. "
                "Добавлена проверка альтернативной интерпретации и баланс рисков."
            )
        must_include = list(case.scoring_spec.get("must_include", []))
        bullets = "\n".join(f"{index}. {item}" for index, item in enumerate(must_include, start=1))
        return (
            f"{case.reference}\n\n"
            "Контроль покрытия ограничений:\n"
            f"{bullets}\n"
            "Контрольная точка: перед финализацией подтвердить критерий готовности и ответственного."
        )

    def _protocol(self, case: DatasetCase, critique: CritiqueResult) -> str:
        if case.bucket == "math_logic":
            return "Если задача содержит вычисления, то проверяй финальный итог обратным пересчётом и явной строкой с ответом; если результат нельзя перепроверить, явно фиксируй риск ошибки."
        if case.bucket == "analysis":
            return "Если пишешь аналитический вывод, то сверяй наличие всех опорных понятий и обязательно проверяй альтернативное объяснение; если данных мало, явно обозначай неопределённость."
        return "Если составляешь план с ограничениями, то перед финализацией проходи чеклист всех обязательных и запрещённых условий; если покрытие ограничений не доказано, не делай категоричный вывод."

    def _self_eval(self, case: DatasetCase, steps: list[str], answer: str) -> str:
        critique = self._critique(case, answer)
        payload_steps = []
        for index, step in enumerate(steps, start=1):
            confidence = 0.82
            rationale = "Шаг выглядит устойчивым относительно условий задачи."
            normalized_step = normalize_text(step)
            if critique.issues and index == len(steps):
                confidence = 0.36
                rationale = "Финальный шаг уязвим: критик нашёл проблемы, влияющие на вывод."
            elif case.bucket == "analysis" and "вывод" not in normalized_step and index == len(steps):
                confidence = 0.48
                rationale = "Итоговый вывод сформулирован неполно или без явной калибровки."
            elif case.bucket == "math_logic" and not any(char.isdigit() for char in step):
                confidence = 0.58
                rationale = "В шаге мало проверяемых числовых опор, поэтому уверенность умеренная."
            payload_steps.append(
                {
                    "step_index": index,
                    "confidence": confidence,
                    "rationale": rationale,
                }
            )
        return json.dumps({"steps": payload_steps}, ensure_ascii=False, indent=2)

    def _step_judge(self, case: DatasetCase, steps: list[str], answer: str) -> str:
        critique = self._critique(case, answer)
        payload_steps = []
        for index, step in enumerate(steps, start=1):
            score = 0.78
            supported = True
            rationale = "Шаг в целом согласован с условиями задачи."
            normalized_step = normalize_text(step)
            if critique.issues and index == len(steps):
                score = 0.24
                supported = False
                rationale = "Финальный шаг не поддержан: критик нашёл проблемы, влияющие на итоговый вывод."
            elif case.bucket == "math_logic" and not any(char.isdigit() for char in step):
                score = 0.45
                supported = False
                rationale = "Шаг не даёт проверяемой числовой опоры для вычислительной задачи."
            elif case.bucket == "analysis" and not any(marker in normalized_step for marker in ["риск", "неопредел", "вывод", "провер"]):
                score = 0.42
                supported = False
                rationale = "Шаг слишком общий и не фиксирует риск, неопределённость или проверяемый вывод."
            elif case.bucket == "constraint_planning" and not any(marker in normalized_step for marker in ["провер", "контроль", "ответствен", "откат", "риск"]):
                score = 0.46
                supported = False
                rationale = "Шаг слабо выражает управляемую процедуру или контроль исполнения."
            payload_steps.append(
                {
                    "step_index": index,
                    "score": score,
                    "supported": supported,
                    "rationale": rationale,
                }
            )
        return json.dumps({"steps": payload_steps}, ensure_ascii=False, indent=2)
