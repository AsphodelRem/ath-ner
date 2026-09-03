from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LABELS = ("ORG", "NAME", "GEO")
EntityKey = tuple[str, int, int]
Cell = tuple[str, int]
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    doc_index: int
    key: EntityKey
    mask: int
    is_gold: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune exact-span model-combination rules with document-level cross-validation."
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        help="Beta-prior strengths used to smooth precision of rare vote patterns.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not isinstance(record.get("hash"), str) or not isinstance(
                record.get("entities"), list
            ):
                raise ValueError(f"{path}:{line_number}: invalid record")
            records.append(record)
    return records


def entity_key(entity: JsonObject) -> EntityKey:
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if label not in LABELS or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"invalid entity: {entity}")
    return label, start, end


def entity_set(record: JsonObject) -> set[EntityKey]:
    return {entity_key(entity) for entity in record["entities"]}


def stable_fold(record_hash: str, folds: int) -> int:
    digest = hashlib.sha256(record_hash.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_candidates(
    gold: list[JsonObject], models: list[list[JsonObject]]
) -> tuple[list[Candidate], list[set[EntityKey]]]:
    hashes = [record["hash"] for record in gold]
    for model in models:
        if [record["hash"] for record in model] != hashes:
            raise ValueError("gold and prediction hashes/order differ")

    candidates: list[Candidate] = []
    gold_sets: list[set[EntityKey]] = []
    for doc_index, rows in enumerate(zip(gold, *models, strict=True)):
        gold_keys = entity_set(rows[0])
        model_keys = [entity_set(row) for row in rows[1:]]
        gold_sets.append(gold_keys)
        for key in set().union(*model_keys):
            mask = sum(1 << index for index, keys in enumerate(model_keys) if key in keys)
            candidates.append(Candidate(doc_index, key, mask, key in gold_keys))
    return candidates, gold_sets


def counts_for_docs(
    candidates: Iterable[Candidate], doc_indices: set[int]
) -> dict[Cell, list[int]]:
    counts: dict[Cell, list[int]] = defaultdict(lambda: [0, 0])
    for candidate in candidates:
        if candidate.doc_index not in doc_indices:
            continue
        cell = (candidate.key[0], candidate.mask)
        counts[cell][0 if candidate.is_gold else 1] += 1
    return counts


def metric(tp: int, fp: int, gold_count: int) -> JsonObject:
    fn = gold_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold": gold_count,
        "predicted": tp + fp,
    }


def learn_allowlist(
    counts: dict[Cell, list[int]], gold_count: int, alpha: float
) -> tuple[set[Cell], JsonObject]:
    total_tp = sum(tp for tp, _ in counts.values())
    total_candidates = sum(tp + fp for tp, fp in counts.values())
    prior = total_tp / total_candidates if total_candidates else 0.5

    ranked = sorted(
        counts,
        key=lambda cell: (
            (counts[cell][0] + alpha * prior)
            / (sum(counts[cell]) + alpha if sum(counts[cell]) + alpha else 1.0),
            sum(counts[cell]),
            cell,
        ),
        reverse=True,
    )
    best_allowed: set[Cell] = set()
    best = metric(0, 0, gold_count)
    running_tp = 0
    running_fp = 0
    for index, cell in enumerate(ranked, start=1):
        tp, fp = counts[cell]
        running_tp += tp
        running_fp += fp
        current = metric(running_tp, running_fp, gold_count)
        if current["f1"] > best["f1"]:
            best = current
            best_allowed = set(ranked[:index])
    return best_allowed, {**best, "prior": prior, "selected_cells": len(best_allowed)}


def predict_with_rules(
    gold: list[JsonObject],
    candidates: list[Candidate],
    rules_by_fold: dict[int, set[Cell]],
    folds: int,
) -> list[JsonObject]:
    selected: list[list[JsonObject]] = [[] for _ in gold]
    for candidate in candidates:
        fold = stable_fold(gold[candidate.doc_index]["hash"], folds)
        if (candidate.key[0], candidate.mask) not in rules_by_fold[fold]:
            continue
        label, start, end = candidate.key
        selected[candidate.doc_index].append(
            {"label": label, "start": start, "end": end}
        )
    result: list[JsonObject] = []
    for record, entities in zip(gold, selected, strict=True):
        entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
        result.append({"hash": record["hash"], "entities": entities})
    return result


def write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def evaluate_records(gold: list[JsonObject], predictions: list[JsonObject]) -> JsonObject:
    tp = fp = gold_count = 0
    by_label: dict[str, JsonObject] = {}
    for label in LABELS:
        label_tp = label_fp = label_gold = 0
        for gold_record, prediction in zip(gold, predictions, strict=True):
            gold_keys = {key for key in entity_set(gold_record) if key[0] == label}
            predicted_keys = {key for key in entity_set(prediction) if key[0] == label}
            label_tp += len(gold_keys & predicted_keys)
            label_fp += len(predicted_keys - gold_keys)
            label_gold += len(gold_keys)
        by_label[label] = metric(label_tp, label_fp, label_gold)
        tp += label_tp
        fp += label_fp
        gold_count += label_gold
    return {"micro": metric(tp, fp, gold_count), "by_label": by_label}


def mask_name(mask: int, names: list[str]) -> str:
    return "+".join(name for index, name in enumerate(names) if mask & (1 << index))


def format_report(
    names: list[str],
    pattern_counts: dict[Cell, list[int]],
    alpha_results: list[JsonObject],
    selected: JsonObject,
    fitted_metrics: JsonObject,
) -> str:
    lines = [
        "# Span vote tuning",
        "",
        "Rules are learned from exact model-agreement patterns. OOF means each",
        "document was predicted by a rule learned without that document's fold.",
        "The fitted-dev number is intentionally optimistic and is reported only as",
        "the score of the rule serialized for applying to a separate hidden test.",
        "",
        "## Cross-validated smoothing search",
        "",
        "| Alpha | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|",
    ]
    for result in alpha_results:
        micro = result["metrics"]["micro"]
        lines.append(
            f"| {result['alpha']:g} | {micro['precision']:.4f} | "
            f"{micro['recall']:.4f} | {micro['f1']:.4f} |"
        )
    oof = selected["metrics"]["micro"]
    fitted = fitted_metrics["micro"]
    lines.extend(
        [
            "",
            f"Selected alpha: `{selected['alpha']:g}`.",
            "",
            f"OOF: `P={oof['precision']:.4f}`, `R={oof['recall']:.4f}`, "
            f"`F1={oof['f1']:.4f}`.",
            "",
            f"Fitted dev: `P={fitted['precision']:.4f}`, `R={fitted['recall']:.4f}`, "
            f"`F1={fitted['f1']:.4f}`.",
            "",
            "## Exact vote-pattern reliability on full dev",
            "",
            "| Label | Models | TP | FP | Precision |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for (label, mask), (tp, fp) in sorted(
        pattern_counts.items(),
        key=lambda item: (item[0][0], -(sum(item[1])), item[0][1]),
    ):
        precision = tp / (tp + fp)
        lines.append(
            f"| {label} | {mask_name(mask, names)} | {tp} | {fp} | {precision:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if len(args.predictions) < 2:
        raise ValueError("at least two prediction files are required")
    if args.names is not None and len(args.names) != len(args.predictions):
        raise ValueError("names must have the same length as predictions")
    if any(alpha < 0 for alpha in args.alphas):
        raise ValueError("alphas must be non-negative")

    names = args.names or [path.stem for path in args.predictions]
    gold = read_jsonl(args.gold)
    models = [read_jsonl(path) for path in args.predictions]
    candidates, gold_sets = build_candidates(gold, models)
    all_docs = set(range(len(gold)))
    fold_docs = {
        fold: {
            index
            for index, record in enumerate(gold)
            if stable_fold(record["hash"], args.folds) == fold
        }
        for fold in range(args.folds)
    }

    alpha_results: list[JsonObject] = []
    oof_predictions_by_alpha: dict[float, list[JsonObject]] = {}
    rules_by_alpha: dict[float, dict[int, set[Cell]]] = {}
    for alpha in args.alphas:
        rules_by_fold: dict[int, set[Cell]] = {}
        fold_training: dict[str, JsonObject] = {}
        for fold in range(args.folds):
            train_docs = all_docs - fold_docs[fold]
            counts = counts_for_docs(candidates, train_docs)
            train_gold_count = sum(len(gold_sets[index]) for index in train_docs)
            allowed, training_metric = learn_allowlist(counts, train_gold_count, alpha)
            rules_by_fold[fold] = allowed
            fold_training[str(fold)] = training_metric
        predictions = predict_with_rules(gold, candidates, rules_by_fold, args.folds)
        metrics = evaluate_records(gold, predictions)
        alpha_results.append(
            {"alpha": alpha, "metrics": metrics, "fold_training": fold_training}
        )
        oof_predictions_by_alpha[alpha] = predictions
        rules_by_alpha[alpha] = rules_by_fold

    selected = max(alpha_results, key=lambda result: result["metrics"]["micro"]["f1"])
    selected_alpha = float(selected["alpha"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "dev_predictions_oof.jsonl", oof_predictions_by_alpha[selected_alpha])

    full_counts = counts_for_docs(candidates, all_docs)
    full_gold_count = sum(len(keys) for keys in gold_sets)
    full_allowed, full_training = learn_allowlist(full_counts, full_gold_count, selected_alpha)
    fitted_predictions = predict_with_rules(
        gold, candidates, {0: full_allowed}, folds=1
    )
    fitted_metrics = evaluate_records(gold, fitted_predictions)
    write_jsonl(output_dir / "dev_predictions_fitted.jsonl", fitted_predictions)

    serialized_rules = [
        {
            "label": label,
            "mask": mask,
            "models": [name for index, name in enumerate(names) if mask & (1 << index)],
            "tp": full_counts[(label, mask)][0],
            "fp": full_counts[(label, mask)][1],
        }
        for label, mask in sorted(full_allowed)
    ]
    payload = {
        "schema_version": 1,
        "models": names,
        "prediction_files": [str(path) for path in args.predictions],
        "folds": args.folds,
        "selected_alpha": selected_alpha,
        "oof_metrics": selected["metrics"],
        "fitted_dev_metrics": fitted_metrics,
        "full_training": full_training,
        "allowed_patterns": serialized_rules,
        "alpha_results": alpha_results,
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        format_report(names, full_counts, alpha_results, selected, fitted_metrics),
        encoding="utf-8",
    )
    print(
        f"Selected alpha={selected_alpha:g}; "
        f"OOF F1={selected['metrics']['micro']['f1']:.6f}; "
        f"fitted-dev F1={fitted_metrics['micro']['f1']:.6f}"
    )
    print(f"Artifacts: {output_dir}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
