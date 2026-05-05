# Отчёт по метакоррекции и итерации

## Сводка
- Количество прогонов: 45
- Proxy-MECO eligible subset: 45 (1.0)
- Средний baseline score: 0.7802
- Средний final score: 0.8561
- Средняя дельта: 0.0759
- Доля деградаций: 0.2222
- Proxy-MECO baseline AUROC: 0.0444
- Proxy-MECO final AUROC: 0.0503
- Proxy-MECO baseline Brier: 0.9051
- Proxy-MECO final Brier: 0.7212
- Proxy-MECO baseline ECE: 0.9332
- Proxy-MECO final ECE: 0.7799
- Proxy-MECO baseline F1: 0.0338
- Proxy-MECO final F1: 0.0089
- Proxy-MECO baseline confidence std: 0.0
- Proxy-MECO final confidence std: 0.0
- Proxy-MECO baseline judge consistency: 1.0
- Proxy-MECO final judge consistency: 1.0
- Proxy-MECO baseline overconfidence: 0.9491
- Proxy-MECO final overconfidence: 0.7096
- Proxy-MECO baseline judge agreement: 0.0815
- Proxy-MECO final judge agreement: 0.1676

## Разбивка по корзинам
- analysis: count=15, proxy_eligible=15, baseline=0.7384, final=0.7549, delta=0.0164, proxy_final_auroc=0.0, proxy_final_ece=0.68, proxy_final_f1=0.0, proxy_final_agreement=0.1833, proxy_final_conf_std=0.0, proxy_final_judge_consistency=1.0
- constraint_planning: count=15, proxy_eligible=15, baseline=0.8067, final=0.8133, delta=0.0067, proxy_final_auroc=0.0, proxy_final_ece=0.9225, proxy_final_f1=0.0, proxy_final_agreement=0.125, proxy_final_conf_std=0.0, proxy_final_judge_consistency=1.0
- math_logic: count=15, proxy_eligible=15, baseline=0.7956, final=1.0, delta=0.2044, proxy_final_auroc=0.151, proxy_final_ece=0.7371, proxy_final_f1=0.0267, proxy_final_agreement=0.1945, proxy_final_conf_std=0.0, proxy_final_judge_consistency=1.0

## Таблица результатов

| case_id | bucket | baseline | final | delta | proxy AUROC | iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ml-01 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-02 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.3 | 1 |
| ml-03 | math_logic | 1.0 | 1.0 | 0.0 | 0.0 | 1 |
| ml-04 | math_logic | 0.4667 | 1.0 | 0.5333 | 0.2143 | 1 |
| ml-05 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.125 | 1 |
| ml-06 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-07 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-08 | math_logic | 0.4667 | 1.0 | 0.5333 | 0.25 | 1 |
| ml-09 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-10 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.25 | 1 |
| ml-11 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.125 | 1 |
| ml-12 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.5 | 1 |
| ml-13 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-14 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.0 | 1 |
| ml-15 | math_logic | 0.8333 | 1.0 | 0.1667 | 0.5 | 1 |
| an-01 | analysis | 0.75 | 0.75 | 0.0 | 0.0 | 1 |
| an-02 | analysis | 0.7267 | 0.8333 | 0.1067 | 0.0 | 1 |
| an-03 | analysis | 0.81 | 0.81 | 0.0 | 0.0 | 1 |
| an-04 | analysis | 0.6833 | 0.9167 | 0.2333 | 0.0 | 2 |
| an-05 | analysis | 0.5367 | 0.81 | 0.2733 | 0.0 | 1 |
| an-06 | analysis | 0.6433 | 0.81 | 0.1667 | 0.0 | 1 |
| an-07 | analysis | 0.6433 | 0.6433 | 0.0 | 0.0 | 1 |
| an-08 | analysis | 0.6433 | 0.6433 | 0.0 | 0.0 | 1 |
| an-09 | analysis | 0.6433 | 0.4933 | -0.15 | 0.0 | 1 |
| an-10 | analysis | 0.9167 | 0.7267 | -0.19 | 0.0 | 1 |
| an-11 | analysis | 0.9167 | 0.9167 | 0.0 | 0.0 | 1 |
| an-12 | analysis | 0.6433 | 0.7267 | 0.0833 | 0.0 | 2 |
| an-13 | analysis | 0.7933 | 0.7933 | 0.0 | 0.0 | 1 |
| an-14 | analysis | 0.81 | 0.7667 | -0.0433 | 0.0 | 1 |
| an-15 | analysis | 0.9167 | 0.6833 | -0.2333 | 0.0 | 1 |
| cp-01 | constraint_planning | 0.8333 | 0.8333 | 0.0 | 0.0 | 1 |
| cp-02 | constraint_planning | 0.95 | 0.8 | -0.15 | 0.0 | 1 |
| cp-03 | constraint_planning | 0.7667 | 0.7833 | 0.0167 | 0.0 | 1 |
| cp-04 | constraint_planning | 1.0 | 0.9 | -0.1 | 0.0 | 1 |
| cp-05 | constraint_planning | 0.8333 | 0.8833 | 0.05 | 0.0 | 2 |
| cp-06 | constraint_planning | 0.7667 | 0.8833 | 0.1167 | 0.0 | 1 |
| cp-07 | constraint_planning | 0.7667 | 0.8833 | 0.1167 | 0.0 | 1 |
| cp-08 | constraint_planning | 0.7667 | 0.6667 | -0.1 | 0.0 | 1 |
| cp-09 | constraint_planning | 0.8833 | 0.8833 | 0.0 | 0.0 | 1 |
| cp-10 | constraint_planning | 1.0 | 0.9 | -0.1 | 0.0 | 1 |
| cp-11 | constraint_planning | 0.7667 | 0.6667 | -0.1 | 0.0 | 1 |
| cp-12 | constraint_planning | 0.7167 | 0.6167 | -0.1 | 0.0 | 1 |
| cp-13 | constraint_planning | 0.7167 | 0.7167 | 0.0 | 0.0 | 1 |
| cp-14 | constraint_planning | 0.6667 | 0.8333 | 0.1667 | 0.0 | 2 |
| cp-15 | constraint_planning | 0.6667 | 0.95 | 0.2833 | 0.0 | 1 |

## Лучший кейс
**Скидка на товар** (`ml-04`)

- Baseline: 0.4667
- Final: 1.0
- Delta: 0.5333

### Финальный ответ
**Цель:** вычислить стоимость книги после скидки в 15%.

**Ключевые ограничения:**
- показать шаги вычисления;
- выделить итоговый результат;
- не проводить полноценную самокритику, но отметить потенциально уязвимые места.

**Решение:**

1. Определить исходную стоимость книги: 1600 рублей.
2. Вычислить размер скидки: 15% от 1600 рублей. Для этого умножим 1600 на 0,15 (так как 15% = 0,15 в десятичном виде).
   $1600 \times 0,15 = 240$ рублей.
3. Вычесть размер скидки из исходной стоимости: $1600 - 240 = 1360$ рублей.

4. Проверка:
   - пересчитать размер скидки через пропорцию: $\frac{15}{100} \times 1600 = 240$ рублей.
   - проверить правильность преобразования процентов в десятичную дробь: 15% = 0,15.
   - выполнить обратную операцию: к итоговой цене 1360 рублей добавить размер скидки 240 рублей, получим 1600 рублей, что соответствует исходной цене.

**Итог:** после скидки 15% книга будет стоить 1360 рублей.

### Proxy-MECO: рискованные шаги
- Шаг 1: conf=1.00, judge=0.35, llm_judge=1.00 :: показать шаги вычисления;
- Шаг 2: conf=1.00, judge=0.41, llm_judge=1.00 :: выделить итоговый результат;
- Шаг 3: conf=1.00, judge=0.23, llm_judge=1.00 :: не проводить полноценную самокритику, но отметить потенциально уязвимые места.

## Сложный кейс
**Роботизация склада** (`an-15`)

- Baseline: 0.9167
- Final: 0.6833
- Delta: -0.2333

### Замечания
- Отдельные замечания не зафиксированы.

### Proxy-MECO: расхождения judge-каналов
- Шаг 1: rule=False, llm=True, conf=0.50 :: Отсутствие конкретных данных о текущем состоянии склада и его операционных процессах.
- Шаг 2: rule=False, llm=True, conf=0.50 :: Недостаток информации о специфике товаров, хранящихся на складе.
- Шаг 3: rule=False, llm=True, conf=0.50 :: Неопределённость в оценке финансовых и операционных рисков.
