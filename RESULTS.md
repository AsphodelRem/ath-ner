# Итоги экспериментов

Все значения ниже посчитаны официальным exact-span scorer на `data/dev.jsonl`.
Поскольку на dev уже сравнивалось много вариантов, это диагностические числа,
а не несмещённая оценка финального качества на закрытом тесте.

## Основные результаты

| Решение | Precision | Recall | Micro-F1 | Практический смысл |
|---|---:|---:|---:|---|
| Словарь train-поверхностей | 0.7078 | 0.5686 | 0.6306 | диагностика memorization |
| Distil-mBERT | 0.7989 | 0.7948 | 0.7968 | исходный baseline |
| XLM-R base | 0.8049 | 0.8516 | 0.8276 | больше recall |
| mmBERT base | 0.8216 | 0.8466 | 0.8339 | лучший исходный encoder |
| mmBERT + partial-window mask + center weighting | 0.8375 | 0.8602 | 0.8487 | лучший одиночный encoder |
| CRF без morph-признаков, threshold 0.50 | 0.8951 | 0.7037 | 0.7879 | дешёвая high-precision модель |
| Exact agreement двух mmBERT | **0.9269** | 0.8150 | 0.8674 | precision-oriented режим |
| 2-of-3: две mmBERT + CRF | 0.9196 | 0.8452 | 0.8808 | компромисс в пользу precision |
| 2-of-4: две mmBERT + XLM-R + CRF | 0.8956 | **0.8801** | **0.8878** | лучший dev F1 |
| Span stacker + два словарных сигнала, 5-fold OOF | **0.9120** | 0.8820 | **0.8968** | основная честная оценка нового ансамбля |
| Span stacker + два словарных сигнала, fitted dev | 0.9210 | **0.8948** | **0.9077** | оптимистичная оценка fitted-правила |
| Research stacker + Distil, 5-fold OOF | 0.9105 | 0.8852 | 0.8976 | выше score, но 1.027B encoder-параметров |
| Research stacker + Distil, fitted dev | 0.9128 | 0.8898 | **0.9012** | только research operating point |
| mmBERT BILOU + Viterbi | 0.9012 | 0.8687 | 0.8846 | сильный одиночный encoder |
| BILOU dual-decode stacker, 5-fold OOF | 0.9172 | 0.8826 | **0.8996** | новый лучший OOF под лимит |
| BILOU dual-decode stacker, fitted dev | **0.9270** | **0.8969** | **0.9117** | оптимистичный fitted score |
| mmBERT BILOU + Viterbi, seed 1337 | 0.8950 | 0.8802 | **0.8875** | лучший одиночный BILOU seed |
| Два BILOU seed, right-priority union | 0.8861 | 0.8936 | **0.8899** | простое объединение без обучения |
| Два BILOU seed + XLM-R + CRF + словари, 5-fold OOF | **0.9058** | **0.8946** | **0.9002** | итоговый вариант ветки `firgla` |
| Тот же итоговый stacker, fitted dev | 0.9106 | 0.9008 | 0.9056 | оптимистично; не использовать как честную оценку |

## Что показала статистическая гипотеза

Suffix-aware словарь улучшил exact-only словарь с `F1=0.5527` до `0.5587`,
но снизил precision. В CRF специализированные признаки выученных окончаний не
дали устойчивого прироста: вариант без них получил `F1=0.7900` против
`0.7880` без confidence-фильтра. Значит окончания полезны как слабый сигнал,
но не являются самостоятельным способом решить задачу.

## Почему улучшился mmBERT

В исходном sliding-window обучении сущность, разрезанная краем окна, получала
частичную BIO-разметку, хотя в соседнем окне она была видна целиком. Исправление
замаскировало 4 023 таких token-labels в 913 окнах. Center weighting дополнительно
уменьшил влияние токенов у краёв перекрывающихся окон. Вместе изменения подняли
mmBERT с `F1=0.8339` до `0.8487`.

Следующий mmBERT обучен с BILOU вместо BIO. На одинаковых весах независимый
argmax дал `F1=0.8453`, а constrained Viterbi — `F1=0.8846`. Следовательно,
основной прирост этого запуска дал именно структурный decoder, который не
разрешает незаконные переходы и незакрытые BILOU-сущности.

## Что брать в решение

- Для итогового offline/competition решения: два BILOU-mmBERT seed, XLM-R,
  CRF, exact/suffix-сигналы и logistic span stacker с запретом пересечений.
  Его основная оценка: `P=0.9058`, `R=0.8946`, `F1=0.9002` в 5-fold OOF.
- Для простого fallback: один mmBERT seed 1337 с BILOU и constrained Viterbi,
  `P=0.8950`, `R=0.8802`, `F1=0.8875`.
- Если precision важнее recall: exact intersection двух BILOU seed даёт
  `P=0.9331`, `R=0.8406`, `F1=0.8844`.
- Финальный ансамбль содержит `892 529 685` encoder-параметров, но три encoder
  forward всё равно требуют отдельного A100 throughput-теста на 30 RPS.

После первоначального majority vote добавлен обучаемый span stacker. Он получает
кандидатов от двух mmBERT, XLM-R, CRF, exact- и suffix-словарей, а затем использует
комбинацию голосов, согласие левой/правой границы, форму mention, контекст
границы и train-only gazetteer-признаки. HistGradientBoosting с отдельными
порогами `ORG=0.460`, `NAME=0.265`, `GEO=0.430` и запретом пересекающихся
результатов получил `F1=0.8968` на пяти out-of-fold частях dev. Это основной
следующий кандидат: три encoder-модели содержат 892 529 685 параметров, CRF,
словари и небольшой stacker добавляют мало вычислений. Более простой logistic
stacker практически не отстаёт (`OOF F1=0.8962`) и может лучше переноситься на
domain shift.

Добавление Distil-mBERT поднимает OOF до `0.8976`, а fitted-dev до `0.9012`,
но четыре encoder-модели вместе содержат 1 027 269 148 параметров. Этот вариант
оставлен как research-абляция, а не основной кандидат под ограничение 1B.

## Что делать дальше

1. Повторить исправленный mmBERT минимум на втором seed.
2. Сделать grouped holdout из train по нормализованным поверхностям и выбирать
   ensemble/threshold уже на нём, а не продолжать подбирать правила по dev.
3. Проверить controlled-эксперименты на кириллице, unseen-сущностях и длинных
   `ORG`, где остаётся больше всего ошибок.
4. На A100 измерить одиночный mmBERT, две mmBERT и четыре модели с batching;
   зафиксировать p50/p95, throughput и память.
5. После выбора operating point завернуть модель в API и прогнать
   `scripts/check_service.py` и `scripts/evaluate_service.py`.

## Где лежат финальные артефакты

- лучший одиночный encoder: `artifacts/experiments/mmbert-mask-partial/`;
- лучшая одиночная CRF: `artifacts/statistical/crf-no-morph-final/`;
- лучший ансамбль: `artifacts/experiments/four-model-vote/`;
- лучший OOF stacker под лимит: `artifacts/experiments/span-stacker-bilou-dual-decode/`;
- BILOU/Viterbi mmBERT: `artifacts/experiments/mmbert-bilou/`;
- второй BILOU seed: `artifacts/experiments/mmbert-bilou-seed1337/`;
- итоговый двух-seed stacker: `artifacts/experiments/span-stacker-bilou-two-seed/`;
- компактные git-артефакты и manifest: `final/`;
- research stacker с Distil-mBERT: `artifacts/experiments/span-stacker-with-suffix-distil/`;
- его предсказания: `dev_predictions_vote2_crf_t0.jsonl`;
- его метрики: `dev_metrics_vote2_crf_t0.json`;
- его разбор ошибок: `errors-vote2-crf-t0/error_report.md`.
