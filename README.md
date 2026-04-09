# Система МКПИ: метакоррекция и итерация

Прототип исследовательского программного средства для проектной практики по метакогнитивной промпт-инженерии. Система реализует технику "метакоррекция и итерация" как управляемый Python-раннер без `LangGraph`, а поверх него даёт `Streamlit`-интерфейс для одиночных запусков, пакетных прогонов и просмотра истории.

## Что внутри

- `mkpi_app/models.py` — адаптеры модели: офлайн `DemoModelAdapter` и `OpenAICompatibleAdapter`.
- `mkpi_app/runner.py` — `TechniqueRunner`, `StopPolicy`, оркестрация `draft -> critique -> revise -> protocol`.
- `mkpi_app/evaluation.py` — bucket-aware оценка для `math_logic`, `analysis`, `constraint_planning`.
- `mkpi_app/storage.py` — загрузка датасета, JSON-логи прогонов, CSV и Markdown-отчёты.
- `mkpi_app/reporting.py` — агрегирование результатов и генерация отчёта.
- `app.py` — интерфейс для сравнения baseline и MKPI.
- `data/dataset.jsonl` — стартовый смешанный датасет.

## Быстрый старт

```bash
cd /Users/gpbu43278/Documents/opencode/mkpi_meta_correction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Для офлайн-проверки ничего кроме `streamlit` не нужно: demo-адаптер не обращается к внешним API.

Для живых запусков через удалённую модель можно:

```bash
export MKPI_API_KEY=...
export MKPI_API_BASE_URL=https://api.openai.com/v1
streamlit run app.py
```

Или ввести `API key` и `Base URL` прямо в боковой панели интерфейса.

## Режимы работы

- `Одиночный запуск` — baseline и MKPI на выбранном кейсе или пользовательской задаче.
- `Пакетный прогон` — benchmark по части датасета с автоматическим отключением накопленных антиошибок.
- `История` — повторное открытие JSON-прогонов и просмотр библиотеки антиошибок.

## Артефакты

- `runs/*.json` — трассы отдельных запусков.
- `runs/anti_errors.json` — накопленные правила для интерактивного режима.
- `reports/results-*.csv` — агрегированные численные результаты.
- `reports/report-*.md` — Markdown-отчёт для пояснительной записки и статьи.

## Тесты

Ядро покрыто `unittest`, поэтому запуск не зависит от сторонних тестовых библиотек:

```bash
cd /Users/gpbu43278/Documents/opencode/mkpi_meta_correction
python3 -m unittest discover -s tests -v
```
