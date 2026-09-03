# Uzbek NER

## Задача

Нужно найти в тексте именованные сущности, определить их точные границы и
отнести каждую сущность к одному из трёх классов:

- `ORG` — организации и бренды;
- `NAME` — люди;
- `GEO` — географические объекты.

Подробные правила классов и определения точных границ приведены в
[`LABELING_GUIDE.md`](LABELING_GUIDE.md).

Практический разбор задачи, результаты EDA и baseline, а также очередь
следующих экспериментов собраны в [`ROADMAP.md`](ROADMAP.md).

## Итоговое решение ветки `firgla`

Основной вариант — span stacker над двумя независимо обученными
`jhu-clsp/mmBERT-base`, `FacebookAI/xlm-roberta-base`, линейным CRF и двумя
train-only словарными сигналами. Оба mmBERT используют BILOU; argmax и
constrained Viterbi для каждого checkpoint считаются из одного encoder forward.

| Оценка итогового варианта | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| 5-fold OOF, основная честная оценка | **0.9058** | **0.8946** | **0.9002** |
| Fitted dev, только диагностика | 0.9106 | 0.9008 | 0.9056 |
| Полный повторный MPS-инференс fitted stacker | 0.9102 | 0.9002 | 0.9052 |

OOF означает, что предсказание каждого документа получено stacker-моделью,
которая не обучалась на этом fold. Fitted-dev число оптимистично и не является
оценкой качества на новых данных. Сумма трёх encoder checkpoint —
`892 529 685` параметров, ниже ограничения 1B. Требование 30 RPS на A100 пока
не подтверждено нагрузочным тестом.

Небольшое отличие повторного MPS-прогона (`0.9052` против `0.9056`) связано с
численной нестабильностью нескольких пограничных token logits. Для сравнения
решений используется сохранённый OOF `0.9002`, а end-to-end строка показывает
фактически воспроизведённый результат команды ниже.

Все финальные пути, порядок источников и frozen thresholds находятся в
[`final/solution.json`](final/solution.json), подробная инструкция — в
[`final/README.md`](final/README.md). При наличии обученных весов полный
инференс выполняется одной командой:

```bash
bash scripts/run_final_inference.sh data/dev.jsonl artifacts/final/dev_predictions.jsonl auto
```

Модельные веса и `.venv` намеренно не хранятся в git. Веса можно воспроизвести
командой `bash scripts/train_final_models.sh cuda`; локальные пути перечислены
в `final/solution.json`.

## Состав комплекта

- `data/train.jsonl` — обучающая выборка с разметкой;
- `data/dev.jsonl` — валидационная выборка с разметкой;
- `data/dataset_manifest.json` — схема, статистика и SHA-256 файлов;
- `LABELING_GUIDE.md` — описание классов, границ и пограничных случаев;
- `baseline/` — минимальный baseline обучения и инференса;
- `statistical/` — suffix-aware словарь и линейный CRF без нейросети;
- `scripts/evaluate.py` — оценка файла предсказаний;
- `scripts/check_service.py` — проверка совместимости HTTP-сервиса;
- `scripts/evaluate_service.py` — прогон HTTP-сервиса и расчёт метрик;
- `API.md` — обязательный контракт HTTP API и Docker-контейнера;
- `requirements.txt` — зависимости baseline.
- `requirements-statistical.txt` — зависимости статистических baseline.

## Формат данных

Каждая строка `data/train.jsonl` и `data/dev.jsonl` — отдельный JSON-объект:

```json
{"hash":"example-001","text":"Ali Toshkent shahrida ishlaydi.","entities":[{"label":"NAME","start":0,"end":3},{"label":"GEO","start":4,"end":12}]}
```

Поля записи:

- `hash` — уникальный идентификатор текста;
- `text` — текст, относительно которого заданы координаты;
- `entities` — список сущностей. Если сущностей нет, список пустой.

Поля сущности:

- `label` — один из классов `ORG`, `NAME`, `GEO`;
- `start` — индекс первого символа сущности, начиная с нуля;
- `end` — индекс первого символа после сущности.

`end` не входит в интервал. Для каждой сущности выполняется:

```python
mention = text[start:end]
```

Координаты считаются по символам Unicode, как индексы строки Python, а не по
байтам.

## Baseline

Baseline — минимальная стартовая точка для модельного эксперимента, а не
полностью готовое итоговое решение. Его можно изменять или заменять. Для
выполнения всех требований кейса команда должна дополнительно обеспечить
воспроизводимость своего эксперимента и реализовать сервис по контракту из
[`API.md`](API.md).

Команды выполняются из корня каталога с данными. Требуется Python 3.10 или новее.

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Референсное окружение baseline использует PyTorch 2.6.0 для CUDA 12.4.
Команда может заменить эту сборку на совместимую со своим окружением и решением.

### macOS Apple Silicon

CUDA-сборка из `requirements.txt` не устанавливается на macOS. Для локального
запуска на Apple Silicon подготовлено отдельное окружение с теми же версиями
основных библиотек и обычной сборкой PyTorch:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
```

Baseline поддерживает `--device mps` и выбирает Apple GPU автоматически при
`--device auto`. Доступность MPS можно проверить командой:

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

## Анализ данных

Воспроизводимый EDA формирует JSON-статистику, Markdown-отчёт и диагностический
словарный baseline:

```bash
.venv/bin/python scripts/analyze_data.py \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/eda
```

Отдельно проверить пригодность токенизатора, длины окон и символьных границ:

```bash
.venv/bin/python scripts/analyze_tokenizer.py \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --model-name distilbert/distilbert-base-multilingual-cased \
  --max-length 256 \
  --stride 64 \
  --output artifacts/eda/distilmbert_tokenizer.json
```

Проверить словарные предсказания официальным exact-span scorer:

```bash
.venv/bin/python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/eda/lexicon_dev_predictions.jsonl \
  --output artifacts/eda/lexicon_dev_metrics.json
```

После получения предсказаний любой модели можно построить разрезы exact-span
ошибок и сохранить примеры для ручного разбора:

```bash
.venv/bin/python scripts/analyze_errors.py \
  --train data/train.jsonl \
  --gold data/dev.jsonl \
  --predictions artifacts/baseline/dev_predictions.jsonl \
  --output-dir artifacts/baseline
```

Обучить модель:

```bash
.venv/bin/python -m baseline.train \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/baseline \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --max-length 256 \
  --stride 64 \
  --seed 42 \
  --device auto
```

Получить предсказания на `dev`:

```bash
.venv/bin/python -m baseline.predict \
  --model-dir artifacts/baseline/model \
  --input data/dev.jsonl \
  --output artifacts/baseline/dev_predictions.jsonl \
  --batch-size 16 \
  --device auto
```

Скрипт автоматически выбирает CUDA, затем Apple MPS, затем CPU. Полный список
параметров доступен через `--help`.

### Воспроизведённый baseline

На предоставленном `dev` получены следующие результаты официального exact-span
scorer:

| Encoder | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| Distil-mBERT baseline | 0.7989 | 0.7948 | 0.7968 |
| XLM-R base | 0.8049 | 0.8516 | 0.8276 |
| mmBERT base | 0.8216 | 0.8466 | **0.8339** |
| mmBERT, partial-window mask + center weighting | 0.8375 | 0.8602 | **0.8487** |

Для mmBERT на Apple MPS обучение использует SDPA, а dev/inference — eager
attention: SDPA в `torch.inference_mode()` даёт нечисловые logits. Его
byte-level offsets также могут включать внешний пробел, поэтому общий декодер
удаляет внешний Unicode-whitespace из итоговых spans. Сырой результат до этой
коррекции сохранён с суффиксом `_raw_offsets`.

Подробные метрики, история обучения и разрезы ошибок находятся соответственно
в `artifacts/baseline` и `artifacts/experiments/<model>`.

### Статистический baseline и ансамбль

Проверить гипотезу про стемминг/присоединённые суффиксы без утечки из `dev`:

```bash
.venv/bin/python -m pip install -r requirements-statistical.txt
.venv/bin/python -m statistical.suffix_lexicon
```

Suffix-aware словарь с консервативным whitelist узбекских окончаний улучшил F1
на внутреннем holdout (`0.5569` → `0.5677`); на `dev` F1 вырос `0.5527` →
`0.5587`, хотя precision всё ещё снизился. Поэтому суффиксы используются дальше
как признаки, а не как безусловные предсказания.

Обучить линейный CRF с BILOU-разметкой, Viterbi, символьными и
train-only gazetteer-признаками:

```bash
.venv/bin/python -m statistical.crf
```

CRF с выученными морфологическими признаками получил `P=0.8944`, `R=0.7016`,
`F1=0.7863`; при отключении confidence-фильтра — `P=0.8745`, `R=0.7171`,
`F1=0.7880`. Контрольная абляция без морфологических признаков оказалась
чуть лучше как одиночная модель: `P=0.8951`, `R=0.7037`, `F1=0.7879`, а без
фильтра — `F1=0.7900`. Следовательно, полезность стемминга для этого корпуса
не подтверждена: gazetteer, форма слова и контекст уже забирают почти весь
его сигнал.

Сохранённая CRF занимает около 20 MB. Полный dev из 1 500 документов
обрабатывается локально на CPU за 7.0 секунды вместе с загрузкой процесса
(примерно 214 документов/с; это throughput-диагностика, не нагрузочный тест
HTTP-сервиса). Предсказания двух вариантов воспроизводятся командами:

```bash
.venv/bin/python -m statistical.predict \
  --model-dir artifacts/statistical/crf \
  --input data/dev.jsonl \
  --output artifacts/statistical/crf/dev_predictions_reproduced.jsonl

.venv/bin/python -m statistical.predict \
  --model-dir artifacts/statistical/crf-no-morph-final \
  --input data/dev.jsonl \
  --output artifacts/statistical/crf-no-morph-final/dev_predictions_reproduced.jsonl
```

Center-weighted агрегация окон mmBERT без переобучения дала `F1=0.8349`
вместо `0.8339`. Непересекающийся hybrid, где mmBERT имеет приоритет, а CRF
добавляет только отдельные spans, дал диагностические `P=0.8159`, `R=0.8619`,
`F1=0.8383`. Сравнение воспроизводится через `scripts/compare_predictions.py`;
правило ансамбля перед финальным использованием нужно подтвердить не на
публичном `dev`, а на внутреннем holdout или закрытом тесте.

Маскирование 4 023 противоречивых token-labels в 913 обрезанных train-окнах
дало существенно больший прирост: `F1=0.8487` вместе с center weighting.
Четырёхмодельный exact vote (две mmBERT, XLM-R и CRF, минимум два совпавших
голоса) получил `P=0.8956`, `R=0.8801`, `F1=0.8878`. Три encoder-модели вместе
содержат 892 529 685 параметров, то есть формально остаются ниже лимита 1B, но
их throughput на A100 необходимо измерить до выбора финального решения.
В этом ансамбле чуть лучше сработала исходная CRF с морфологическими признаками
и отключённым confidence-фильтром; вариант CRF без морфологии дал `F1=0.8873`.

Обучаемый span stacker поверх двух mmBERT, XLM-R, CRF, exact- и suffix-словарей
улучшил результат до `P=0.9120`, `R=0.8820`, `F1=0.8968` в пятифолдовом OOF-прогоне.
Fitted-версия, предназначенная для отдельного hidden test, имеет оптимистичный
dev-score `F1=0.9077`. Она воспроизводимо применяется так:

```bash
.venv/bin/python scripts/apply_span_stacker.py \
  --bundle artifacts/experiments/span-stacker-two-lexicons/stacker.joblib \
  --train data/train.jsonl \
  --input data/dev.jsonl \
  --predictions \
    artifacts/experiments/mmbert-base/dev_predictions_center_weighted.jsonl \
    artifacts/experiments/mmbert-mask-partial/dev_predictions_center.jsonl \
    artifacts/experiments/xlm-roberta-base/dev_predictions.jsonl \
    artifacts/statistical/crf/dev_predictions_threshold_0.jsonl \
    artifacts/statistical/suffix-lexicon/dev_predictions_suffix.jsonl \
    artifacts/statistical/suffix-lexicon/dev_predictions_exact.jsonl \
  --output artifacts/experiments/span-stacker-two-lexicons/dev_predictions_applied.jsonl
```

Research-вариант с дополнительным Distil-mBERT получил OOF `F1=0.8976` и
fitted-dev `F1=0.9012`, но сумма его encoder-параметров равна 1 027 269 148,
поэтому он не выбран основным вариантом под лимит 1B.

Новый mmBERT с BILOU подтвердил пользу структурного декодирования: argmax дал
`F1=0.8453`, constrained Viterbi на тех же весах — `P=0.9012`, `R=0.8687`,
`F1=0.8846`. После добавления обоих декодирований как бесплатных сигналов
stacker получил `P=0.9172`, `R=0.8826`, `F1=0.8996` в 5-fold OOF и
оптимистичный fitted-dev `F1=0.9117`.

Второй BILOU checkpoint (`seed=1337`) дал `P=0.8950`, `R=0.8802`,
`F1=0.8875`. Замена старого BIO-mmBERT на этот checkpoint и использование
обоих декодирований двух BILOU-моделей подняли итоговый stacker до
`P=0.9058`, `R=0.8946`, `F1=0.9002` в 5-fold OOF. Learned transition priors
повышали precision, но монотонно снижали F1 (`0.8844`, `0.8841`, `0.8835`,
`0.8826` для scale `0.01`, `0.025`, `0.05`, `0.10`), поэтому финал использует
только жёсткие ограничения допустимых BILOU-переходов.

Сводная таблица всех основных запусков и практическая рекомендация находятся
в [`RESULTS.md`](RESULTS.md).

```bash
.venv/bin/python -m baseline.predict \
  --model-dir artifacts/experiments/mmbert-base/model \
  --input data/dev.jsonl \
  --output artifacts/experiments/mmbert-base/dev_predictions_center_weighted.jsonl \
  --window-weighting center \
  --attn-implementation eager

.venv/bin/python scripts/compare_predictions.py \
  --left artifacts/experiments/mmbert-base/dev_predictions_center_weighted.jsonl \
  --right artifacts/statistical/crf/dev_predictions.jsonl \
  --left-name mmbert_center \
  --right-name crf \
  --output-dir artifacts/experiments/mmbert-center-crf-comparison

.venv/bin/python scripts/vote_predictions.py \
  --predictions \
    artifacts/experiments/mmbert-base/dev_predictions_center_weighted.jsonl \
    artifacts/experiments/mmbert-mask-partial/dev_predictions_center.jsonl \
    artifacts/statistical/crf/dev_predictions_threshold_0.jsonl \
    artifacts/experiments/xlm-roberta-base/dev_predictions.jsonl \
  --min-votes 2 \
  --output artifacts/experiments/four-model-vote/dev_predictions_vote2_crf_t0.jsonl
```

## Воспроизводимость решения

Материалы команды должны позволять повторить обучение и проверку результата
без скрытых ручных шагов. Необходимо предоставить:

- зафиксированные зависимости и описание окружения;
- точную версию предоставленных и дополнительных обучающих данных;
- конфигурацию, гиперпараметры и random seed итогового эксперимента;
- документированные команды подготовки данных, обучения, предикта и оценки;
- итоговую обученную модель, предсказания на `dev` и заявленные метрики.

Если использовались внешние или синтетические данные, команда должна сохранить
их итоговую подготовленную версию либо предоставить воспроизводимый способ их
получения и указать источники.

Повторный инференс предоставленной итоговой модели должен воспроизводить
заявленные метрики на `dev`. Повторное обучение должно давать сопоставимый
результат.

## Формат предсказаний

Каждая строка файла предсказаний содержит `hash` исходной записи и найденные
сущности:

```json
{"hash":"example-001","entities":[{"label":"NAME","start":0,"end":3},{"label":"GEO","start":4,"end":12}]}
```

Файл должен содержать ровно одну запись для каждого `hash` оцениваемой выборки.
Порядок записей и сущностей значения не имеет.

## Метрики

Запустить оценку на `dev`:

```bash
.venv/bin/python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/baseline/dev_predictions.jsonl \
  --output artifacts/baseline/dev_metrics.json
```

Сущность засчитывается только при точном совпадении `hash`, `label`, `start` и
`end` с эталоном. Скрипт выводит Precision, Recall и F1 для каждого класса, а
также micro- и macro-усреднение.

Итоговая оценка решений проводится на закрытой тестовой выборке.

## Инференс-сервис

Итоговое решение должно предоставлять `POST /api/v1/predict` и лёгкий
`GET /healthz`, запускаться в Docker и возвращать сущности в том же формате
exact spans, который используется в файле предсказаний.

Полный формат запросов, ответов и правила запуска контейнера приведены в
[`API.md`](API.md). CLI из каталога `baseline/` является примером модельного
инференса и не реализует HTTP-сервис за участника.

После запуска своего сервиса проверить совместимость можно командой:

```bash
python scripts/check_service.py --url http://localhost:8000
```

Проверяющий скрипт ждёт готовности `/healthz`, отправляет батч текстов в
`/api/v1/predict` и валидирует обязательные поля, классы и символьные
координаты. Качество предсказаний этой командой не оценивается.

Чтобы прогнать всю валидационную выборку через работающий сервис и рассчитать
метрики, используется команда:

```bash
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions artifacts/service/dev_predictions.jsonl \
  --output artifacts/service/dev_metrics.json
```

Скрипт отправляет сервису только `hash` и `text`, проверяет каждый ответ,
сохраняет предсказания и рассчитывает метрики тем же exact-span scorer. Размер
HTTP-батча можно изменить параметром `--batch-size`.
