# Воспроизведение итогового решения

Доменно-адаптированный RemBERT, две стадии на NVIDIA A100 80GB PCIe:

1. **MLM-адаптация** — `artifacts/mlm-rembert`, 11,8 часа. Корпус 405 585 пассажей,
   220,8 млн токенов. Разметка не используется.
2. **Дообучение под NER** — `artifacts/mlm_rembert_last`, 64 минуты. BILOU,
   декодирование Витерби, постобработка границ.

## Заявленный результат

`scripts/evaluate.py` на `data/dev.jsonl` (1500 записей, 7698 сущностей):

| класс | precision | recall | F1 | tp | fp | fn |
|-------|-----------|--------|------|------|-----|-----|
| ORG   | 0.8872 | 0.8665 | 0.8767 | 2304 | 293 | 355 |
| NAME  | 0.9182 | 0.8814 | 0.8994 | 2044 | 182 | 275 |
| GEO   | 0.9153 | 0.9096 | 0.9124 | 2474 | 229 | 246 |
| **micro** | **0.9065** | **0.8862** | **0.8962** | 6822 | 704 | 876 |
| macro | 0.9069 | 0.8858 | 0.8962 | — | — | — |

Точка работы декодирования — `transition_weight = 0.4`, см.
[«Сила ограничений Витерби»](#сила-ограничений-витерби).

Held-out перплексия энкодера после стадии 1: **11.50** (на старте 41.78).

## Комплект поставки

### Итоговая модель

`artifacts/mlm_rembert_last` — достаточно для инференса и проверки метрик.

| файл | размер | sha256 |
|------|--------|--------|
| `model/model.safetensors` | 2.2 ГБ | `c40399e4d8874bab10558b09ec0f7d1ae91cd79f67cec762cf0abbeec26d833e` |
| `model/config.json` | 1.3 КБ | `3f0a3310bd9878fd210b37510cca813c79a3fd7270cd9680c5b0d6983e085319` |
| `model/tokenizer.json` | 16 МБ | `f5973f76ef61c5b0f59cf39fe56b770a25b9484f953b6ab2d0f03795cbeb1fe5` |
| `model/tokenizer_config.json` | 446 Б | `2dc3bc1dd72c6d46e5dda24072df6beb4df6fe8fdde1ee989b3d897229fc9b87` |
| `run_config.json` | 1.1 КБ | `4e6183da43f272059c3eb08bf9e601aa6a5030d15b716c1c460837f0ea8bc191` |
| `transitions.json` | 18 КБ | `7e0bcd29823c50f24cc02e8f84c929fa53c0fa69fc189b72d51ccc224b9030cd` |

`transitions.json` обязателен: без него инференс с Витерби не запустится.
`run_config.json` задаёт схему тегов, окно и силу переходов для сервиса.

```bash
cd artifacts/mlm_rembert_last && sha256sum -c <<'EOF'
c40399e4d8874bab10558b09ec0f7d1ae91cd79f67cec762cf0abbeec26d833e  model/model.safetensors
3f0a3310bd9878fd210b37510cca813c79a3fd7270cd9680c5b0d6983e085319  model/config.json
f5973f76ef61c5b0f59cf39fe56b770a25b9484f953b6ab2d0f03795cbeb1fe5  model/tokenizer.json
2dc3bc1dd72c6d46e5dda24072df6beb4df6fe8fdde1ee989b3d897229fc9b87  model/tokenizer_config.json
4e6183da43f272059c3eb08bf9e601aa6a5030d15b716c1c460837f0ea8bc191  run_config.json
7e0bcd29823c50f24cc02e8f84c929fa53c0fa69fc189b72d51ccc224b9030cd  transitions.json
EOF
```

Дополнительно: `dev_predictions.jsonl`, `dev_metrics.json`, `history.json`, `metrics.csv`.

### Для переобучения

| артефакт | размер | sha256 |
|----------|--------|--------|
| `data/pretrain/corpus.jsonl` | 1,0 ГБ | `4042a4f837c6daec62a0b470c81dc6104b54c433f434f6f0b797ae932ab11ab5` |
| `data/pretrain/corpus.holdout.jsonl` | 3,2 МБ | `2280aea39937955b71d17705c719c38d609c4b96849d9683475fb50899a8968c` |
| `data/pretrain/corpus.manifest.json` | 1,5 КБ | `790a8059ba23f126ba4db909b7725e5d1daa7456aa0d5c6d00960ef375540742` |
| `logs/mlm_130653_0.log` | 47 КБ | лог стадии 1 |
| `logs/mlm_rembert_last130674_0.log` | 313 КБ | лог стадии 2 |

Корпус сверен с манифестом: 405 585 документов, 772 683 288 символов, битых строк 0.

**Промежуточный энкодер после MLM не передаётся.** Чтобы переобучить стадию 2,
сначала выполняется стадия 1 из приложенного корпуса — около 12 часов на A100.
Корпус, команда и лог стадии 1 приложены, так что результат воспроизводим целиком;
шаг просто не бесплатный. Ориентир корректности стадии 1 — held-out перплексия 11.5.

## Окружение

Python 3.12, CUDA 12.4, autocast `torch.bfloat16`, gradient checkpointing включён.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-solution.txt
```

```
torch==2.6.0+cu124        accelerate==1.14.0
transformers==5.14.1      numpy==2.2.6
tokenizers==0.22.2        tqdm==4.70.0
safetensors==0.8.0        huggingface-hub==1.29.0
```

Сверх `requirements.txt`: `numpy` (импорт в `solution/train.py`, `solution/viterbi.py`),
`accelerate` (Trainer в `solution/pretrain.py`), `pyarrow` (parquet-шарды в
`tools/prepare_pretrain.py`), `huggingface_hub`, `comet_ml` (только при `--comet`).
`sentencepiece` и `protobuf` не нужны — в чекпоинте есть `tokenizer.json`.

Сборка корпуса требует сети. Обучение и инференс идут офлайн
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).

## Данные

Внешняя и синтетическая разметка не привлекалась, пересплита нет, из train ничего
не удалялось.

| файл | записей | сущностей | sha256 |
|------|---------|-----------|--------|
| `data/train.jsonl` | 13000 | 66083 | `b3281c93aeb28353384b749627a4a82217cdd250a607a3ea3803363b31718346` |
| `data/dev.jsonl` | 1500 | 7698 | `7e5d96a1de83311e3acb33a3c28ed8e13783141143c8eb83e86fb329a795e4d0` |

Суммы совпадают с `data/dataset_manifest.json`:

```bash
python - <<'EOF'
import json, hashlib, pathlib
manifest = json.load(open("data/dataset_manifest.json", encoding="utf-8"))
for name, info in manifest["splits"].items():
    digest = hashlib.sha256(pathlib.Path(info["path"]).read_bytes()).hexdigest()
    print(name, "совпадает" if digest == info["sha256"] else "НЕ СОВПАДАЕТ")
EOF
```

### Корпус MLM

| источник | сплит | документов | доля символов |
|----------|-------|-----------:|--------------:|
| `tahrirchi/uz-crawl` | `telegram_blogs` | 207 909 | 22,3% |
| `tahrirchi/uz-books-v2` | `cyr` | 46 704 | 23,2% |
| `tahrirchi/uz-crawl` | `news` | 77 557 | 23,2% |
| `tahrirchi/uz-books-v2` | `lat` | 46 709 | 23,2% |
| `wikimedia/wikipedia` | `20231101.ru` | 17 552 | 4,0% |
| `wikimedia/wikipedia` | `20231101.en` | 9 154 | 4,0% |

405 585 пассажей, 772 683 288 символов, 220 755 753 токена (калибровка по
xlm-roberta-large, 3,5 симв./токен). Латиница 180 433, обе графики 140 225,
кириллица 84 927. Отсев: OCR-шум 248 056, короткие 128 529, near-дубли 11 636,
дубли по абзацам 8461, точные дубли 1485, не узбекский 426. Тексты кейса исключены
по нормализованному тексту.

## Стадия 1: сборка корпуса

```bash
python tools/prepare_pretrain.py \
  --sources telegram:all,books-cyr:3,news:2,books-lat:2,wiki-ru:1,wiki-en:1 \
  --max-tokens 300000000 \
  --foreign-share 0.08 \
  --tokenizer FacebookAI/xlm-roberta-large \
  --exclude data/train.jsonl data/dev.jsonl \
  --holdout 2000 \
  --passage-chars 4000 \
  --max-ocr-noise 0.01 \
  --near-dedup-bands 8 \
  --seed 42 \
  --output data/pretrain/corpus.jsonl
```

Калибровка пересчитывает общий бюджет, но квоты источников выведены из него раньше
и не обновляются. Сборка остановилась на 772,7 млн символов (`300 млн × 2,6`) вместо
калиброванных `300 млн × 3,5 = 1050` млн — корпус на 26% меньше запрошенного,
220,8 млн токенов вместо 300 млн. Итоговый прогон обучен на таком корпусе, поэтому
команда приведена как есть.

Датасеты версионируются на стороне Hugging Face: пересобранный корпус не будет
побитово равен исходному.

## Стадия 1: MLM-адаптация

```bash
python -m solution.pretrain \
  --corpus data/pretrain/corpus.jsonl \
  --eval-corpus data/pretrain/corpus.holdout.jsonl \
  --model-name google/rembert \
  --output-dir artifacts/mlm-rembert \
  --max-length 512 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 --warmup-ratio 0.06 \
  --max-steps 16800 \
  --eval-every 1000 --save-every 2000 --log-every 100 \
  --num-workers 8 --seed 42 --resume
```

Маскирование целыми словами, доля 15% (умолчание). За шаг 8 × 4 × 512 = 16384
токена, за 16800 шагов — 275,3 млн.

`--max-steps 16800` задано явно: автоформула в `tools/slurm/pretrain.sbatch`
(`estimated_tokens // 16384`) дала бы 13473. При повторе задавайте `MAX_STEPS`.

Скрипт пишет два состояния шага 16800: `checkpoint-16800` и `model/`. Стадия 2
брала `checkpoint-16800`.

## Стадия 2: дообучение под NER

```bash
python -m solution.train \
  --preset laptop \
  --model-name artifacts/mlm-rembert/checkpoint-16800 \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --max-length 512 --stride 128 \
  --batch-size 32 --gradient-accumulation-steps 4 \
  --eval-batch-size 32 \
  --learning-rate 3e-5 \
  --epochs 15 --patience 3 \
  --seed 42 \
  --tag-scheme bilou \
  --viterbi --transition-weight 0.25 --transition-mode conditional \
  --output-dir artifacts/mlm_rembert_last
```

`--transition-weight 0.25` здесь — значение, при котором `solution/train.py`
сравнивает эпохи и выбирает лучшую. Точка работы готового решения — 0.4, она
задаётся после обучения и на веса не влияет.

`--preset laptop` задаёт только `gradient_checkpointing=True`, остальное перекрыто
явными флагами. Эффективный батч 128, 13000 записей дают 14881 окно. Аугментации
выключены. Чекпоинт отбирается по exact-span micro-F1, а не по dev_loss.
`--epochs 15` — верхняя граница: ранняя остановка сработала после 8-й эпохи,
лучшая — 5-я.

| эпоха | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|
| train_loss | 0.6551 | 0.0576 | 0.0333 | 0.0195 | **0.0112** | 0.0063 | 0.0035 | 0.0021 |

Лучшая эпоха — пятая, ранняя остановка после восьмой. Полная кривая с метриками по
эпохам — в `artifacts/mlm_rembert_last/history.json` и в логе стадии 2.

Seed зафиксирован, но недетерминированные CUDA-ядра не отключались: повторное
обучение даёт сопоставимый, а не побитово идентичный результат.

## Инференс и проверка метрик

Матрица переходов оценена по train и сохранена рядом с моделью, обучающая выборка
для инференса не нужна. Прогон переданного чекпоинта воспроизводит
`dev_predictions.jsonl` побитово (проверено на 300 записях, расхождений 0).

```bash
python scripts/evaluate.py \
  --gold data/dev.jsonl \
  --predictions artifacts/mlm_rembert_last/dev_predictions.jsonl \
  --output artifacts/mlm_rembert_last/dev_metrics.json
```

Пересчёт предсказаний с нуля. Точка работы берётся из `run_config.json` рядом с
каталогом модели — те же поля, что читает сервис, поэтому CLI и HTTP дают
одинаковый ответ:

```bash
python -m solution.predict \
  --model-dir artifacts/mlm_rembert_last/model \
  --input data/dev.jsonl \
  --output artifacts/cli/dev_predictions.jsonl \
  --metrics artifacts/cli/dev_metrics.json
```

Тот же скрипт даёт предсказания по неразмеченному файлу — при отсутствии поля
`entities` метрики просто не считаются:

```bash
python -m solution.predict \
  --model-dir artifacts/mlm_rembert_last/model \
  --input public_test_inputs.jsonl \
  --output public_test_predictions.jsonl
```

Через работающий сервис:

```bash
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions artifacts/service/dev_predictions.jsonl \
  --output artifacts/service/dev_metrics.json
```

## Сервис

Точка работы задаётся файлами прогона: сервис читает `max_length`, `stride`,
`postprocess`, `tag_scheme`, `viterbi`, `transition_mode`, `transition_weight`
из `run_config.json`.

```
artifacts/
├── model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── run_config.json
└── transitions.json
```

```bash
docker compose up --build
python scripts/check_service.py --url http://localhost:8000
```

Загрузка с `local_files_only=True`, контейнер в сеть не ходит. Для CUDA при прямом
`docker run` добавьте `--gpus all -e NER_DEVICE=cuda`. Контракт — в [`API.md`](API.md).

## Декодирование Витерби

Модель оценивает каждый токен независимо, поэтому argmax по токенам умеет выдавать
невозможные последовательности тегов: `I-ORG` сразу после `O`, `L-NAME` после
`B-ORG`, сущность без начала. Витерби вместо этого ищет наиболее вероятную
последовательность целиком.

Алгоритм — динамическое программирование. Идём по токенам слева направо и для
каждого тега храним лучший счёт пути, которым в него можно прийти: счёт равен сумме
счёта предыдущего тега, логарифма вероятности перехода между ними и логарифма
вероятности самого тега на этом токене. Запоминаем, откуда пришли. На последнем
токене берём лучший тег и по сохранённым указателям разворачиваем путь назад. Это
даёт точный максимум за `O(T·S²)` вместо перебора `S^T` последовательностей, где
T — число токенов, S — число тегов.

Матрица переходов оценивается по биграммам тегов в `data/train.jsonl` и хранится в
`transitions.json`. Она **зависит от позиции сабтокена внутри слова** — начало,
середина, конец или слово целиком: границы сущностей в BILOU проходят по границам
слов, и переход, естественный между словами, внутри слова означает ошибку.

Режим `conditional` берёт log P(тег | предыдущий тег, позиция) как есть.
Альтернативный `structural` вычитает унарный приор log P(тег | позиция), оставляя
только сочетаемость: тогда редкие одиночные сущности (тег `U`) перестают
штрафоваться за саму свою редкость.

### Сила ограничений

Множитель логарифмической матрицы переходов при декодировании. Итоговое значение —
**0.4**: оно стоит в `run_config.json`, его читает сервис, при нём посчитаны все
числа выше и получен файл предсказаний. Обучение шло при 0.25 — это значение
влияет только на отбор лучшей эпохи.

Значение подобрано по dev. Рост веса обменивает полноту на точность, максимум F1
приходится на 0.4:

| transition_weight | precision | recall | F1 |
|-------------------|-----------|--------|------|
| **0.40** | **0.9065** | **0.8862** | **0.8962** |
| 0.50 | 0.9082 | 0.8828 | 0.8953 |
| 0.60 | 0.9105 | 0.8784 | 0.8941 |

Это параметр декодирования: на веса модели он не влияет и перебирается без
переобучения, флагом `--transition-weight` у `solution.predict`.

## Что не входит в решение

Аугментации (`solution/augment.py`, выключены), прогрев на внешнем размеченном
корпусе (ветка `@TWO_STAGE` в `tools/slurm/run.sbatch`),
ансамблирование. 

## Чек-лист

1. Суммы `data/train.jsonl` и `data/dev.jsonl` сходятся с манифестом.
2. Суммы переданной модели и корпуса сходятся с таблицами выше.
3. `scripts/evaluate.py` на `dev_predictions.jsonl` даёт micro-F1 0.8962.
4. `docker compose up --build` поднимается, `scripts/check_service.py` проходит.
5. `scripts/evaluate_service.py` на сервисе даёт те же 0.8962.

Пункты 1-5 не требуют обучения. Полный повтор с нуля — стадия 1 из корпуса до
перплексии около 11.5, затем стадия 2 до micro-F1 около 0.896.
