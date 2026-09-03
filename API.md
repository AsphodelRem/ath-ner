# Контракт инференс-сервиса

Сервис запускается в Docker, слушает `0.0.0.0:8000` и предоставляет два
эндпоинта. Все запросы и ответы используют JSON в UTF-8; авторизация не нужна.

## `GET /healthz`

После загрузки модели сервис возвращает `200 OK`,
`Content-Type: application/json` и тело:

```json
{"status":"ok"}
```

Проверка не запускает инференс. До завершения загрузки модели сервис может не
принимать соединения или отвечать `503 Service Unavailable`.

## `POST /api/v1/predict`

Тело запроса — непустой JSON-массив. `hash` должен быть непустой строкой и быть
уникальным в пределах запроса, `text` должен быть строкой:

```json
[
  {"hash":"example-001","text":"Ali Toshkent shahrida ishlaydi."},
  {"hash":"example-002","text":"Алишер Навоий ҳақида мақола."}
]
```

Успешный ответ имеет код `200 OK`:

```json
{
  "data": [
    {
      "hash": "example-001",
      "entities": [
        {"label":"NAME","start":0,"end":3},
        {"label":"GEO","start":4,"end":12}
      ]
    },
    {
      "hash": "example-002",
      "entities": [
        {"label":"NAME","start":0,"end":13}
      ]
    }
  ]
}
```

Правила ответа:

- в `data` находится ровно один результат на входной элемент, в том же порядке;
- `hash` копируется без изменений;
- `entities` всегда является массивом;
- `label` принимает только `ORG`, `NAME` или `GEO`;
- `start` и `end` — целые символьные координаты в исходном `text`;
- интервал полуоткрытый: `[start, end)`, `end` в сущность не входит;
- координаты считаются с нуля по символам Unicode, а не по байтам;
- выполняется `0 <= start < end <= len(text)`;
- одинаковые сущности не повторяются;
- некорректный запрос получает `4xx`, частичный успешный ответ не возвращается.

## Модель

Checkpoint Hugging Face должен находиться в `artifacts/model` до сборки образа.
Рядом с ним можно положить `artifacts/run_config.json` и
`artifacts/transitions.json`, созданные `solution.train`. Если checkpoint лежит
в `artifacts/<run>/model`, сервис автоматически выбирает запуск с наибольшим
`best_micro_f1` из соседнего `run_config.json`.

Переменные `NER_MODEL_DIR`, `NER_DEVICE` (`auto`, `cpu`, `cuda`) и
`NER_BATCH_SIZE` необязательны. Загрузка выполняется с `local_files_only=True`;
во время работы контейнер не обращается к сети. Если checkpoint не добавлен,
сервис запускается в техническом режиме и возвращает пустые `entities`: этого
достаточно для проверки HTTP-контракта, но не для оценки качества.

## Docker и Compose

Сборка и запуск через Compose:

```bash
docker compose up --build
```

Прямая сборка и запуск также поддерживаются:

```bash
docker build -t ner-uz-solution .
docker run --rm -p 8000:8000 ner-uz-solution
```

Для CUDA при прямом запуске добавьте `--gpus all -e NER_DEVICE=cuda`. После
старта контракт проверяется командой:

```bash
python scripts/check_service.py --url http://localhost:8000
```
