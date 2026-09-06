"""Предсказание сущностей обученным чекпоинтом по произвольному JSONL.

Точка работы берётся из run_config.json рядом с каталогом модели — те же поля,
что читает сервис (api/runtime.py), поэтому CLI и HTTP дают одинаковый ответ.
Любое поле перекрывается флагом.

    python -m solution.predict \
        --model-dir artifacts/mlm_rembert_last/model \
        --input public_test_inputs.jsonl \
        --output public_test_predictions.jsonl

Если во входном файле есть разметка, считаются exact-span метрики:

    python -m solution.predict \
        --model-dir artifacts/mlm_rembert_last/model \
        --input data/dev.jsonl --output artifacts/dev_predictions.jsonl \
        --metrics artifacts/dev_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification

from baseline.common import load_fast_tokenizer, read_records, resolve_device, validate_window
from solution.inference import predict_records
from solution.metrics import evaluate
from solution.tagging import tags_for
from solution.viterbi import load_transitions

JsonObject = dict[str, Any]

DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128


def parse_args() -> argparse.Namespace:
    """Разбирает пути и переопределения точки работы."""

    parser = argparse.ArgumentParser(description="Predict exact-span NER with a trained checkpoint.")
    parser.add_argument("--model-dir", type=Path, required=True, help="каталог чекпоинта")
    parser.add_argument("--input", type=Path, required=True, help="JSONL с полями hash и text")
    parser.add_argument("--output", type=Path, required=True, help="куда писать предсказания")
    parser.add_argument("--metrics", type=Path, help="куда писать метрики, если во входе есть разметка")
    parser.add_argument(
        "--run-config",
        type=Path,
        help="по умолчанию run_config.json рядом с каталогом модели",
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        help="по умолчанию transitions.json рядом с каталогом модели",
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, help="взять только первые N записей")
    parser.add_argument("--transition-weight", type=float)
    parser.add_argument("--transition-mode", choices=("conditional", "structural"))
    parser.add_argument("--no-viterbi", action="store_true", help="argmax вместо Витерби")
    parser.add_argument("--no-postprocess", action="store_true", help="отключить постобработку границ")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="без прогресс-бара")
    return parser.parse_args()


def read_run_config(path: Path) -> JsonObject:
    """Читает конфигурацию прогона; её отсутствие не ошибка, но меняет результат."""

    if not path.is_file():
        print(f"ВНИМАНИЕ: {path} не найден — Витерби выключен, окно берётся по умолчанию")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: ожидался JSON-объект")
    return payload


def resolve_scheme(id2label: dict[int, str], configured: Any) -> str:
    """Определяет схему тегов по конфигу, иначе по самим меткам чекпоинта."""

    if configured in {"bio", "bilou"}:
        scheme = str(configured)
    else:
        scheme = "bilou" if any(t.startswith(("L-", "U-")) for t in id2label.values()) else "bio"
    expected = set(tags_for(scheme))
    if set(id2label) != set(range(len(expected))) or set(id2label.values()) != expected:
        raise ValueError(f"метки чекпоинта не совпадают со схемой {scheme.upper()}")
    return scheme


def build_transitions(
    args: argparse.Namespace,
    config: JsonObject,
    id2label: dict[int, str],
) -> Any:
    """Собирает матрицу переходов с учётом переопределений; None означает argmax."""

    if args.no_viterbi or not config.get("viterbi", False):
        return None
    path = args.transitions or (args.model_dir.parent / "transitions.json")
    if not path.is_file():
        raise FileNotFoundError(f"Витерби включён, но матриц нет: {path}")
    mode = args.transition_mode or str(config.get("transition_mode", "conditional"))
    transitions, tags = load_transitions(path, mode=mode)
    if tags != [id2label[index] for index in range(len(id2label))]:
        raise ValueError("теги матрицы переходов не совпадают с метками чекпоинта")
    weight = args.transition_weight
    if weight is None:
        weight = float(config.get("transition_weight", 0.25))
    print(f"Витерби: {path} | режим {mode} | вес {weight}")
    return transitions * weight


def has_entities(path: Path) -> bool:
    """Проверяет по первой записи, размечен ли входной файл."""

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                return isinstance(json.loads(line).get("entities"), list)
    return False


def clean_entities(records: list[JsonObject], source: list[JsonObject]) -> list[JsonObject]:
    """Оставляет в ответе только поля контракта и проверяет координаты."""

    texts = {record["hash"]: record["text"] for record in source}
    result: list[JsonObject] = []
    for record in records:
        text_length = len(texts[record["hash"]])
        entities = []
        for entity in record["entities"]:
            start, end = int(entity["start"]), int(entity["end"])
            if not 0 <= start < end <= text_length:
                raise ValueError(f"{record['hash']}: координаты [{start}, {end}) вне текста")
            entities.append({"label": entity["label"], "start": start, "end": end})
        result.append({"hash": record["hash"], "entities": entities})
    return result


def run(args: argparse.Namespace) -> int:
    """Прогоняет входной файл через чекпоинт и пишет предсказания."""

    config = read_run_config(args.run_config or (args.model_dir.parent / "run_config.json"))
    max_length = args.max_length or int(config.get("max_length", DEFAULT_MAX_LENGTH))
    stride = args.stride if args.stride is not None else int(config.get("stride", DEFAULT_STRIDE))
    postprocess = bool(config.get("postprocess", True)) and not args.no_postprocess
    trust_remote_code = args.trust_remote_code or bool(config.get("trust_remote_code", False))

    labelled = has_entities(args.input)
    records = read_records(args.input, require_entities=labelled, limit=args.limit)
    inputs = [{"hash": record["hash"], "text": record["text"]} for record in records]
    print(f"Записей: {len(records)} | разметка во входе: {'есть' if labelled else 'нет'}")

    device = resolve_device(args.device)
    tokenizer = load_fast_tokenizer(str(args.model_dir))
    validate_window(tokenizer, max_length, stride)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()

    id2label = {int(index): str(label) for index, label in model.config.id2label.items()}
    scheme = resolve_scheme(id2label, config.get("tag_scheme"))
    log_transitions = build_transitions(args, config, id2label)
    print(f"Модель: {args.model_dir} | {device} | схема {scheme} | окно {max_length}/{stride}")

    with torch.no_grad():
        predictions = predict_records(
            model, tokenizer, inputs,
            max_length=max_length, stride=stride, batch_size=args.batch_size,
            device=device, id2label=id2label, scheme=scheme,
            postprocess=postprocess, progress=not args.quiet,
            log_transitions=log_transitions,
        )
    # predict_records в некоторых версиях добавляет score; контракт его не ждёт.
    predictions = clean_entities(predictions, inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = sum(len(record["entities"]) for record in predictions)
    print(f"Записано: {args.output} | сущностей {total}")

    if labelled:
        metrics = evaluate(records, predictions)
        micro = metrics["micro"]
        for label, values in metrics["by_label"].items():
            print(f"  {label:<5} P {values['precision']:.4f} R {values['recall']:.4f}"
                  f" F1 {values['f1']:.4f}")
        print(f"  micro P {micro['precision']:.4f} R {micro['recall']:.4f} F1 {micro['f1']:.4f}")
        if args.metrics:
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Метрики: {args.metrics}")
    elif args.metrics:
        print("ВНИМАНИЕ: во входном файле нет разметки, метрики не считались")
    return 0


def main() -> int:
    """Точка входа CLI."""

    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
