# Proxy-MECO: внедрение, ограничения и результаты на YandexGPT Lite

## Контекст

Цель работы состояла в том, чтобы приблизить текущий benchmark-пайплайн проекта к идеям `AutoMeco` / `MECO` из EMNLP 2025 без доступа к внутренним сигналам модели (`hidden states`, `logits`, `token probabilities`).

Поэтому был реализован не полный `AutoMeco`, а `proxy-MECO`: шаговая метакогнитивная оценка, основанная на:

- разбиении ответа на шаги;
- self-reported confidence модели по шагам;
- внешней proxy-разметке корректности шагов;
- независимом `LLM-as-a-judge` для шагов;
- калибровочных и agreement-метриках;
- batch-агрегации результатов.

Отдельно был выполнен полный прогон benchmark'ов на самой дешёвой текстовой модели Yandex AI Studio, `YandexGPT Lite`, с сопоставлением:

- `baseline` — голый ответ модели;
- `MKPI` — ответ с пайплайном метакоррекции и итерации.

## Что было в репозитории до начала работы

На старте проект уже содержал:

- `TechniqueRunner` с циклом `baseline -> draft -> critique -> revision -> protocol`;
- собственный mixed dataset (`math_logic`, `analysis`, `constraint_planning`);
- batch benchmark с генерацией CSV и Markdown-отчётов;
- офлайн `DemoModelAdapter`;
- `OpenAICompatibleAdapter`;
- систему хранения `runs/` и `reports/`;
- модульную оценку через `Evaluator`.

Этого было достаточно, чтобы внедрять proxy-бенчмарк поверх существующего пайплайна без полной перестройки архитектуры.

## Что было реализовано

### 1. Базовый proxy-MECO слой

Добавлен модуль `mkpi_app/proxy_meco.py`.

В нём реализованы:

- извлечение reasoning steps из ответа;
- построение proxy-trace по шагам;
- step-level confidence parsing;
- step-level proxy labels;
- первичные discriminative метрики.

Для сериализации результатов были расширены схемы:

- `ProxyMecoStep`
- `ProxyMecoTrace`

И добавлены поля в `ExperimentRun`:

- `baseline_proxy_meco`
- `final_proxy_meco`
- `proxy_meco_eligible`

### 2. Step-level self-evaluation

В `prompts.py` был добавлен `build_self_eval_prompt(...)`, а в раннер — новая фаза `self_eval`.

На этом шаге модель получает уже готовый ответ, разложенный на шаги, и для каждого шага возвращает:

- `confidence`
- краткое `rationale`

Это даёт прокси-оценку внутренней уверенности без доступа к внутренним активациям модели.

### 3. Bucket-specific rule-based step judge

Был реализован bucket-aware proxy judge для разметки шагов:

- `math_logic`
- `analysis`
- `constraint_planning`

Ключевая идея: шаг считается supported не по единственной общей эвристике, а по разным правилам для разных классов задач.

Примеры критериев:

- `math_logic`: числовые опоры, обязательные маркеры, согласованность с ожидаемым числовым итогом;
- `analysis`: обязательные понятия, секции, маркеры неопределённости, follow-up, balance;
- `constraint_planning`: must-include элементы, forbidden terms, управляющие секции, quality markers.

### 4. Независимый LLM-as-a-step-judge

Чтобы разметка шагов не была только rule-based, был добавлен второй канал:

- prompt `build_step_judge_prompt(...)`;
- фаза `step_judge` в раннере;
- обработка результата как `score + supported + rationale`.

В результате у каждого шага теперь есть:

- rule/evaluator-based proxy label;
- независимая LLM-оценка;
- возможность измерять agreement между judge-каналами.

### 5. Калибровка и proxy-метрики

В `ProxyMecoTrace` были добавлены метрики:

- `auroc`
- `average_precision`
- `brier_score`
- `expected_calibration_error`
- `precision`
- `recall`
- `f1`
- `overconfidence_rate`
- `llm_judge_agreement`
- `dual_supported_steps`

Эти метрики позволяют анализировать не только “помог ли пайплайн”, но и:

- насколько confidence калиброван;
- насколько шаговые сигналы различают supported / unsupported steps;
- насколько внешний LLM judge согласуется с rule-based judge.

### 6. Bucket-level агрегация

В `reporting.py` были добавлены aggregate-метрики по корзинам:

- `proxy_final_auroc`
- `proxy_final_brier`
- `proxy_final_ece`
- `proxy_final_f1`
- `proxy_final_agreement`

Это важно, потому что эффект MKPI и proxy-MECO неравномерен по классам задач.

### 7. Proxy-MECO subset protocol

Был добавлен модуль `mkpi_app/benchmark_protocol.py`, который формализует отбор задач, пригодных для proxy-MECO.

Реализованы:

- `score_proxy_meco_case(case)`
- `is_proxy_meco_eligible(case)`
- `select_proxy_meco_subset(cases, per_bucket=4)`

Протокол делает benchmark воспроизводимее: теперь можно отдельно считать,

- сколько прогонов вообще относятся к proxy-MECO-eligible subset;
- как меняются метрики именно на задачах, пригодных для шаговой оценки.

### 8. Multi-sample consistency

Финальным шагом был добавлен режим повторных proxy-оценок на одном и том же ответе.

Для этого в `RunnerConfig` добавлено:

- `proxy_meco_repeats`

В `ProxyMecoTrace` добавлены stability-метрики:

- `sample_count`
- `confidence_std_mean`
- `llm_judge_consistency`

Это переводит benchmark из single-run режима в более устойчивый режим оценки:

- насколько confidence скачет между повторными оценками;
- насколько judge labels стабильны между сэмплами.

## Интеграция Yandex AI Studio

### Почему понадобился отдельный адаптер

Текущий `OpenAICompatibleAdapter` не подходит для Yandex AI Studio напрямую, потому что Yandex использует другой completion endpoint и другой формат payload.

Поэтому был добавлен отдельный адаптер:

- `YandexAIStudioAdapter`

Он работает через endpoint:

- `https://llm.api.cloud.yandex.net/foundationModels/v1/completion`

и формирует `modelUri` вида:

- `gpt://<folder_id>/<model>/latest`

### С чем столкнулись

1. Первый тестовый запрос упал из-за неправильного `folder_id` в `modelUri`.

API вернул полезную ошибку с реальным `folder_id`, после чего запрос был исправлен.

2. Во время полного batch benchmark при работе с Yandex встречались сетевые сбои:

- `Remote end closed connection without response`
- `SSL: UNEXPECTED_EOF_WHILE_READING`

Из-за этого в адаптер были добавлены:

- retry attempts;
- линейный backoff;
- verbose logging retry-событий.

3. Полный batch в “молчаливом” режиме плохо подходит для реального исследования и ожидания пользователя. Поэтому прогон был переведён в режим live-progress с печатью:

- текущего кейса;
- промежуточного результата;
- delta baseline/final;
- proxy-AUROC по кейсу.

## Параметры полного прогона на YandexGPT Lite

Прогон был выполнен на:

- модель: `YandexGPT Lite`
- кейсов: `45`
- `max_iterations = 3`
- `temperature = 0.2`
- `max_tokens = 900`
- `proxy_meco_repeats = 1`

Артефакты сохранены в:

- `reports/report-yandexgpt-lite-full.md`
- `reports/results-yandexgpt-lite-full.csv`

## Итоговые результаты прогона

### Общие benchmark-метрики

- `runs`: `45`
- `baseline_mean`: `0.7802`
- `final_mean`: `0.8561`
- `delta_mean`: `+0.0759`
- `failure_rate`: `0.2222`

Интерпретация:

- пайплайн МКПИ в среднем улучшает качество;
- но примерно в 22% кейсов результат всё ещё деградирует.

### Разбивка по корзинам

#### math_logic

- `baseline_mean`: `0.7956`
- `final_mean`: `1.0000`
- `delta_mean`: `+0.2044`

Вывод: strongest gain. На формализуемых вычислительных задачах MKPI работает лучше всего.

#### analysis

- `baseline_mean`: `0.7384`
- `final_mean`: `0.7549`
- `delta_mean`: `+0.0164`

Вывод: gain есть, но слабый.

#### constraint_planning

- `baseline_mean`: `0.8067`
- `final_mean`: `0.8133`
- `delta_mean`: `+0.0067`

Вывод: эффект минимальный.

### Proxy-MECO summary

- `baseline_auroc_mean`: `0.0444`
- `final_auroc_mean`: `0.0503`
- `baseline_brier_mean`: `0.9051`
- `final_brier_mean`: `0.7212`
- `baseline_ece_mean`: `0.9332`
- `final_ece_mean`: `0.7799`
- `baseline_f1_mean`: `0.0338`
- `final_f1_mean`: `0.0089`
- `baseline_overconfidence_mean`: `0.9491`
- `final_overconfidence_mean`: `0.7096`
- `baseline_agreement_mean`: `0.0815`
- `final_agreement_mean`: `0.1676`

Интерпретация proxy-результатов:

- калибровка улучшается: Brier и ECE становятся лучше;
- overconfidence заметно снижается;
- agreement между judge-каналами растёт;
- но discriminative качество step-level сигналов остаётся слабым: AUROC и F1 низкие.

### Proxy-MECO по корзинам

#### math_logic

- `proxy_final_auroc`: `0.1510`
- `proxy_final_agreement`: `0.1945`

Лучшая корзина для step-level signal analysis.

#### analysis

- `proxy_final_auroc`: `0.0000`
- `proxy_final_agreement`: `0.1833`

Калибровка улучшается, но discriminative step label signal практически отсутствует.

#### constraint_planning

- `proxy_final_auroc`: `0.0000`
- `proxy_final_agreement`: `0.1250`

Прокси-разметка и модельная step-level оценка пока плохо согласуются содержательно.

## Основные выводы для статьи

### 1. MKPI реально помогает слабой/дешёвой модели

На `YandexGPT Lite` пайплайн МКПИ дал положительный средний эффект, особенно на `math_logic`.

Это хороший результат для статьи, потому что показывает не просто “абстрактную полезность” подхода, а прикладной выигрыш на дешёвой production-like модели.

### 2. Польза MKPI неравномерна по типам задач

Самый сильный эффект проявляется в задачах, где:

- есть проверяемый финальный результат;
- шаги можно локально верифицировать;
- critique/revision действительно может исправить ошибку, а не просто переформулировать ответ.

### 3. Калибровка confidence улучшается сильнее, чем step discrimination

На этом прогоне MKPI уменьшает самоуверенность и улучшает calibration-oriented показатели сильнее, чем улучшает способность proxy-benchmark'а различать supported / unsupported steps.

Это важный и интересный результат: пайплайн делает модель осторожнее, но пока не превращает её в сильного step-level self-judge.

### 4. Для аналитических и planning-задач proxy-MECO остаётся шумным

Это не обязательно означает, что MKPI “не работает” в этих задачах. Скорее это означает, что:

- шаговая разметка здесь сложнее;
- rule-based и LLM-as-a-judge proxy всё ещё грубы;
- нужен более сильный judge или доступ к более богатым сигналам.

## Что осталось за пределами proxy-подхода

Даже после всех улучшений текущий benchmark всё ещё остаётся `proxy-MECO`, а не полным `AutoMeco` / `MECO`.

Главные отсутствующие компоненты:

- внутренние сигналы модели (`hidden states`, `logits`);
- настоящие intrinsic lenses;
- full process reward model / PRM-like annotation;
- прямое сравнение внутренних представлений модели со step correctness.

То есть достигнут practical maximum без internal-state access, но не full paper replication.

## Файлы, добавленные и изменённые в ходе работы

### Новые модули

- `mkpi_app/proxy_meco.py`
- `mkpi_app/benchmark_protocol.py`

### Изменённые модули

- `mkpi_app/models.py`
- `mkpi_app/prompts.py`
- `mkpi_app/runner.py`
- `mkpi_app/reporting.py`
- `mkpi_app/schemas.py`
- `mkpi_app/__init__.py`

### Новые тесты

- `tests/test_proxy_meco.py`
- `tests/test_reporting.py`

## Проверка корректности изменений

После внедрения всех изменений был выполнен полный локальный прогон test suite:

- `python3 -m unittest discover -s tests -v`

Результат на момент фиксации:

- `29` тестов
- все `OK`

## Практическая рекомендация для следующего этапа исследования

Если проект развивается в сторону статьи, следующий качественный шаг уже не в усилении proxy-эвристик, а в переходе к одному из двух направлений:

1. либо curated research protocol:
   - фиксированный subset;
   - повторяемые модели;
   - жёсткая табличная отчётность для статьи;

2. либо переход к local open-weight моделям с доступом к внутренним сигналам для более близкой репликации `AutoMeco`.

На текущем этапе `proxy-MECO` уже пригоден как база для раздела `Method` и `Experimental Setup` в статье, а также для раздела `Limitations`, потому что его сильные стороны и ограничения теперь явно инструментализированы.
