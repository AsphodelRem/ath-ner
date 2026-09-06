"""Ансамбль усреднением вероятностей, а не голосованием по спанам.

Голосование (tools/ensemble.py) видит только финальные решения моделей и теряет
всё, в чём они сомневались. Здесь усредняются распределения по классам на уровне
токенов, и спаны декодируются один раз из усреднённой матрицы.

Токенизаторы у моделей разные, поэтому сетки выравниваются по символьным
координатам: распределение чужого токена переносится на опорный токен
пропорционально длине пересечения. Схемы разметки тоже приводятся к общей: если
хоть одна модель обучена на BIO, все сворачиваются в BIO.

Веса задавать обязательно, если модели неравны по качеству. На трёх локальных
прогонах равные веса дали 0.8832 против 0.8890 у одиночного лидера — усреднение
со слабыми моделями тянет вниз. Веса 3:1:1 дали 0.8939 на отложенной половине.

    python -m tools.ensemble_probs --models artifacts/glot500-s42 artifacts/run04 --weights 3 1
    python -m tools.ensemble_probs --models artifacts/* --tune-weights --cache /tmp/ens.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from solution.inference import collect_token_scores, decode_scores
from solution.metrics import evaluate
from solution.tagging import tags_for
from solution.viterbi import load_transitions

JsonObject = dict[str, Any]
Grid = list[tuple[int, int]]


def read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scheme_of(labels: list[str]) -> str:
    """Определяет схему модели по набору её тегов."""

    for scheme in ("bilou", "bio"):
        if set(labels) == set(tags_for(scheme)):
            return scheme
    raise ValueError(f"не удалось опознать схему по тегам: {sorted(labels)}")


def projection(source: tuple[str, ...], target: tuple[str, ...]) -> list[list[int]]:
    """Для каждого целевого тега — индексы исходных, чью вероятность в него сливать.

    Каждый исходный тег попадает ровно в один целевой, иначе сумма вероятностей
    строки перестаёт быть единицей. Тег, существующий в целевой схеме, остаётся
    собой; остальные сворачиваются по префиксу: U означает начало сущности как и
    B, L — продолжение как и I. Обратное направление потребовало бы выдумать долю
    одиночных сущностей внутри B, поэтому не поддерживается.
    """

    fold = {"U": "B", "L": "I"}
    groups: list[list[int]] = [[] for _ in target]
    index_of = {tag: index for index, tag in enumerate(target)}
    for index, name in enumerate(source):
        if name in index_of:
            groups[index_of[name]].append(index)
            continue
        prefix, separator, label = name.partition("-")
        folded = f"{fold.get(prefix, prefix)}-{label}" if separator == "-" else name
        if folded not in index_of:
            raise ValueError(f"тег {name!r} некуда отобразить в схеме {target}")
        groups[index_of[folded]].append(index)
    empty = [target[i] for i, members in enumerate(groups) if not members]
    if empty:
        raise ValueError(f"теги {empty} нечем заполнить из схемы {source}")
    return groups


def model_scores(
    model_dir: Path,
    records: list[JsonObject],
    *,
    canonical: tuple[str, ...],
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> list[tuple[Grid, np.ndarray]]:
    """Прогоняет одну модель и возвращает по записи: сетку токенов и матрицу вероятностей."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    source = tuple(id2label[i] for i in sorted(id2label))
    # Схемы и порядок тегов у моделей различаются — приводим к общей.
    groups = projection(source, canonical)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        raw = collect_token_scores(
            model, tokenizer, records,
            max_length=max_length, stride=stride, batch_size=batch_size,
            device=device, progress=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out: list[tuple[Grid, np.ndarray]] = []
    for record_scores in raw:
        ordered = sorted(record_scores.items())
        if not ordered:
            out.append(([], np.zeros((0, len(canonical)))))
            continue
        matrix = torch.stack([total / count for _, (total, count) in ordered]).numpy()
        projected = np.stack([matrix[:, members].sum(axis=1) for members in groups], axis=1)
        out.append(([span for span, _ in ordered], projected))
    return out


def align(grid: Grid, spans: Grid, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Переносит матрицу с чужой сетки на опорную, взвешивая пересечением символов.

    Возвращает взвешенную сумму по опорным токенам и вес каждого — токены без
    пересечения получают нулевой вес и не портят усреднение.
    """

    total = np.zeros((len(grid), matrix.shape[1]), dtype=np.float64)
    weight = np.zeros(len(grid), dtype=np.float64)
    if not spans:
        return total, weight
    starts = np.fromiter((s for s, _ in spans), dtype=np.int64, count=len(spans))
    ends = np.fromiter((e for _, e in spans), dtype=np.int64, count=len(spans))
    for index, (grid_start, grid_end) in enumerate(grid):
        first = int(np.searchsorted(ends, grid_start, side="right"))
        for position in range(first, len(spans)):
            if starts[position] >= grid_end:
                break
            overlap = min(ends[position], grid_end) - max(starts[position], grid_start)
            if overlap <= 0:
                continue
            total[index] += matrix[position] * overlap
            weight[index] += overlap
    return total, weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="+", required=True,
                        help="каталоги артефактов (используется подкаталог model) или сами каталоги весов")
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"),
                        help="записи для разбора; без поля entities работает как чистый инференс")
    parser.add_argument("--weights", type=float, nargs="+", help="веса моделей, по умолчанию равные")
    parser.add_argument("--tune-weights", action="store_true",
                        help="подобрать веса на gold координатным спуском (подгонка под этот набор)")
    parser.add_argument("--scheme", default="auto", choices=("auto", "bio", "bilou"),
                        help="auto — самая грубая из схем моделей: BIO, если хоть одна модель на BIO")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--transitions", type=Path, help="матрицы переходов для Витерби")
    parser.add_argument("--transition-weight", type=float, default=0.25,
                        help="у ансамбля оптимум ниже, чем у одиночной модели (0.5): усреднение "
                             "вероятностей само подавляет несогласованные последовательности, "
                             "и переходы сверх 0.25 режут только полноту")
    parser.add_argument("--transition-mode", default="conditional", choices=("conditional", "structural"))
    parser.add_argument("--cache", type=Path, help="файл кэша прогонов моделей")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--per-model", action="store_true", help="показать каждую модель отдельно")
    args = parser.parse_args()

    records = read_jsonl(args.gold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = [d / "model" if (d / "model").is_dir() else d for d in args.models]

    schemes = []
    for model_dir in dirs:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        schemes.append(scheme_of(list(config["id2label"].values())))
    scheme = args.scheme if args.scheme != "auto" else ("bio" if "bio" in schemes else "bilou")
    canonical = tags_for(scheme)
    if len(set(schemes)) > 1:
        print("схемы моделей различаются: "
              + ", ".join(f"{d.parent.name}={s}" for d, s in zip(dirs, schemes))
              + f" -> усредняем в {scheme.upper()}")
    weights = args.weights or [1.0] * len(dirs)
    if len(weights) != len(dirs):
        raise SystemExit(f"весов {len(weights)}, а моделей {len(dirs)}")

    # Файл без разметки — это тестовый вход: считаем предсказания и не трогаем метрики.
    scored = all("entities" in record for record in records)
    if not scored:
        if args.tune_weights:
            raise SystemExit("--tune-weights требует разметки, а в записях нет поля entities")
        if args.per_model:
            raise SystemExit("--per-model требует разметки, а в записях нет поля entities")
        if not args.output:
            raise SystemExit("для входа без разметки нужен --output: метрики посчитать не по чему")
    print(f"моделей: {len(dirs)} | записей: {len(records)} | схема: {scheme} | устройство: {device}"
          + ("" if scored else " | режим: инференс без оценки"))

    cached: dict[str, Any] = {}
    if args.cache and args.cache.exists():
        with args.cache.open("rb") as handle:
            cached = pickle.load(handle)

    collected = []
    for model_dir in dirs:
        key = f"{model_dir}|{scheme}"
        if key in cached:
            print(f"  {model_dir}: из кэша")
            collected.append(cached[key])
            continue
        print(f"  {model_dir}: прогон")
        result = model_scores(model_dir, records, canonical=canonical,
                              max_length=args.max_length, stride=args.stride,
                              batch_size=args.batch_size, device=device)
        cached[key] = result
        collected.append(result)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with args.cache.open("wb") as handle:
            pickle.dump(cached, handle)

    id2label = dict(enumerate(canonical))
    log_transitions = None
    if args.transitions:
        log_transitions, _ = load_transitions(args.transitions, mode=args.transition_mode)
        if log_transitions.shape[-1] != len(canonical):
            raise SystemExit(
                f"{args.transitions} рассчитан на {log_transitions.shape[-1]} тегов, "
                f"а ансамбль сводится к {len(canonical)} ({scheme}). "
                "Возьмите transitions.json от прогона с той же схемой или уберите --transitions."
            )
        log_transitions = log_transitions * args.transition_weight

    def decode(per_record: list[dict[tuple[int, int], tuple[torch.Tensor, int]]]) -> list[JsonObject]:
        return decode_scores(records, per_record, id2label=id2label, scheme=scheme,
                             postprocess=True, log_transitions=log_transitions,
                             with_scores=False)

    def as_scores(grid: Grid, matrix: np.ndarray) -> dict[tuple[int, int], tuple[torch.Tensor, int]]:
        return {span: (torch.from_numpy(matrix[i]), 1) for i, span in enumerate(grid)}

    if args.per_model:
        print("\n=== одиночные модели ===")
        for model_dir, per_record in zip(dirs, collected, strict=True):
            single = [as_scores(grid, matrix) for grid, matrix in per_record]
            metrics = evaluate(records, decode(single))
            show(str(model_dir), metrics)

    # Опорная сетка — от первой модели; от весов не зависит, считаем один раз.
    aligned: list[list[tuple[np.ndarray, np.ndarray]]] = []
    grids: list[Grid] = []
    for index in range(len(records)):
        grid = collected[0][index][0]
        grids.append(grid)
        aligned.append([] if not grid else
                       [align(grid, *per_record[index]) for per_record in collected])

    def blend(model_weights: list[float]) -> list[dict[tuple[int, int], tuple[torch.Tensor, int]]]:
        merged = []
        for grid, parts in zip(grids, aligned, strict=True):
            if not grid:
                merged.append({})
                continue
            total = np.zeros((len(grid), len(canonical)), dtype=np.float64)
            weight = np.zeros(len(grid), dtype=np.float64)
            for (part, part_weight), model_weight in zip(parts, model_weights, strict=True):
                total += part * model_weight
                weight += part_weight * model_weight
            safe = np.where(weight > 0, weight, 1.0)[:, None]
            merged.append(as_scores(grid, total / safe))
        return merged

    if not args.weights and not args.tune_weights and len(dirs) > 1:
        print("\n  веса не заданы — усредняю с равными. Если модели различаются по качеству,"
              "\n  передайте --weights или --tune-weights: равные веса обычно хуже лидера")

    if args.tune_weights:
        print("\n=== подбор весов ===")
        grid = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
        best_f1 = evaluate(records, decode(blend(weights)))["micro"]["f1"]
        print(f"  старт {fmt(weights)}: micro {best_f1:.4f}")
        for round_index in range(1, 3):
            moved = False
            for position in range(len(dirs)):
                for value in grid:
                    trial = list(weights)
                    trial[position] = value
                    if not any(trial):
                        continue
                    f1 = evaluate(records, decode(blend(trial)))["micro"]["f1"]
                    if f1 > best_f1 + 1e-6:
                        weights, best_f1, moved = trial, f1, True
            print(f"  проход {round_index}: {fmt(weights)} micro {best_f1:.4f}")
            if not moved:
                break
        print("  подобранные веса подогнаны под этот набор — на тесте прибавка будет меньше")

    predictions = decode(blend(weights))
    metrics = evaluate(records, predictions) if scored else None
    if metrics is not None:
        print("\n=== ансамбль усреднением вероятностей ===")
        show(f"среднее по {len(dirs)} моделям, веса {fmt(weights)}", metrics)
    else:
        found = sum(len(record["entities"]) for record in predictions)
        empty = sum(1 for record in predictions if not record["entities"])
        print(f"\nансамбль по {len(dirs)} моделям, веса {fmt(weights)}: "
              f"сущностей {found} ({found / len(predictions):.2f} на запись), "
              f"записей без сущностей {empty} ({empty / len(predictions):.1%})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps({"hash": item["hash"], "entities": item["entities"]},
                                        ensure_ascii=False) + "\n")
        print(f"\nпредсказания: {args.output}")
    if args.metrics_output and metrics is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["ensemble"] = {"models": [str(d) for d in dirs], "weights": weights,
                               "mode": "probability-average"}
        args.metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"метрики:      {args.metrics_output}")
    return 0


def fmt(weights: list[float]) -> str:
    return ":".join(f"{w:g}" for w in weights)


def show(title: str, metrics: JsonObject) -> None:
    micro, macro = metrics["micro"], metrics["macro"]
    print(f"\n{title}")
    print(f"  {'':6s} {'precision':>10s} {'recall':>9s} {'f1':>9s} {'tp':>6s} {'fp':>6s} {'fn':>6s}")
    for label in ("ORG", "NAME", "GEO"):
        row = metrics["by_label"][label]
        print(f"  {label:6s} {row['precision']:10.4f} {row['recall']:9.4f} {row['f1']:9.4f}"
              f" {row['tp']:6d} {row['fp']:6d} {row['fn']:6d}")
    print(f"  {'micro':6s} {micro['precision']:10.4f} {micro['recall']:9.4f} {micro['f1']:9.4f}"
          f" {micro['tp']:6d} {micro['fp']:6d} {micro['fn']:6d}")
    print(f"  {'macro':6s} {macro['precision']:10.4f} {macro['recall']:9.4f} {macro['f1']:9.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
