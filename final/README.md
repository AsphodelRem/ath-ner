# Финальная конфигурация ветки `firgla`

## Что выбрано

Итоговый score-oriented вариант использует восемь span-сигналов в строго
зафиксированном порядке:

1. mmBERT BILOU seed 42, constrained Viterbi;
2. тот же checkpoint, argmax;
3. mmBERT BILOU seed 1337, constrained Viterbi;
4. тот же checkpoint, argmax;
5. XLM-R base;
6. линейный CRF без confidence-фильтра;
7. suffix-aware train lexicon;
8. exact train lexicon.

Argmax и Viterbi каждого mmBERT строятся из одного encoder forward. Затем
логистический span stacker применяет отдельные пороги `ORG=0.345`,
`NAME=0.380`, `GEO=0.390` и удаляет пересекающиеся spans.

## Метрики

| Вариант | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| mmBERT BILOU seed 42 + Viterbi | 0.9012 | 0.8687 | 0.8846 |
| mmBERT BILOU seed 1337 + Viterbi | 0.8950 | 0.8802 | 0.8875 |
| Предыдущий BILOU stacker, 5-fold OOF | 0.9172 | 0.8826 | 0.8996 |
| **Финальный двух-seed stacker, 5-fold OOF** | **0.9058** | **0.8946** | **0.9002** |
| Финальный stacker, fitted dev | 0.9106 | 0.9008 | 0.9056 |
| Полный повторный MPS-инференс fitted stacker | 0.9102 | 0.9002 | 0.9052 |

Последняя строка оптимистична: serialized stacker обучен на всём dev для
применения к отдельному hidden test. Для ожидаемого качества используем OOF
`0.9002`.

Разница `0.9056 → 0.9052` на полном повторном MPS-прогоне составляет четыре
true positive и возникает из-за небольших численных изменений у пограничных
token logits. Фактический end-to-end результат сохранён в
`metrics/end_to_end_mps.json`.

## Файлы

- `solution.json` — machine-readable manifest и пути весов;
- `stacker.joblib` — fitted logistic span stacker;
- `stacker_experiment.json` — все OOF-абляции и выбранные thresholds;
- `lexicon_configs.json` — frozen exact/suffix параметры;
- `metrics/` — официальные exact-span JSON-метрики;
- `analysis/` — tokenizer density и аудит `ORG ↔ GEO`.

Большие encoder/CRF-веса лежат в `artifacts/` и не входят в git. Их можно
воспроизвести:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-statistical.txt
bash scripts/train_final_models.sh cuda
```

## Инференс

При наличии весов полный pipeline запускается одной командой:

```bash
bash scripts/run_final_inference.sh INPUT.jsonl OUTPUT.jsonl cuda
```

Для локального Apple Silicon вместо `cuda` передаётся `mps`, для
автоматического выбора — `auto`. Входу нужны только `hash` и `text`; `entities`
не обязательны.

Проверка на размеченном dev:

```bash
python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/final/dev_predictions.jsonl \
  --output artifacts/final/dev_metrics.json
```

## Ограничения

- три encoder checkpoint суммарно содержат `892 529 685` параметров, то есть
  проходят лимит 1B;
- нагрузочный тест 30 RPS и p95 latency на A100 ещё не проводился;
- dev использовался для выбора нескольких решений, поэтому окончательное
  подтверждение возможно только на hidden test;
- веса нужно передавать отдельно либо хранить в Git LFS/model registry.
