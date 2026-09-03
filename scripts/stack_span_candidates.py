from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("ORG", "NAME", "GEO")
EntityKey = tuple[str, int, int]
JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    doc_index: int
    key: EntityKey
    mask: int
    features: tuple[float, ...]
    is_gold: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated span stacker using agreement, boundary and gazetteer features."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-recombined-boundaries",
        action="store_true",
        help="Add start/end combinations from overlapping same-label predictions.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if (
                not isinstance(record.get("hash"), str)
                or not isinstance(record.get("entities"), list)
                or ("text" in record and not isinstance(record["text"], str))
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


def script_flags(text: str) -> tuple[float, float, float]:
    has_latin = any("LATIN" in unicodedata.name(char, "") for char in text if char.isalpha())
    has_cyrillic = any(
        "CYRILLIC" in unicodedata.name(char, "") for char in text if char.isalpha()
    )
    return float(has_latin), float(has_cyrillic), float(has_latin and has_cyrillic)


def boundary_kind(char: str) -> tuple[float, float, float]:
    if not char:
        return 1.0, 0.0, 0.0
    return 0.0, float(char.isspace()), float(unicodedata.category(char).startswith("P"))


def overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = overlap(left, right)
    if not intersection:
        return 0.0
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union


def build_gazetteer(
    train: list[JsonObject],
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]], dict[str, Counter[str]]]:
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    first_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    last_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train:
        text = record["text"]
        for entity in record["entities"]:
            label, start, end = entity_key(entity)
            mention = text[start:end].casefold()
            words = mention.split()
            surfaces[mention][label] += 1
            if words:
                first_tokens[words[0]][label] += 1
                last_tokens[words[-1]][label] += 1
    return dict(surfaces), dict(first_tokens), dict(last_tokens)


def counter_features(counter: Counter[str] | None, label: str) -> tuple[float, ...]:
    if not counter:
        return 0.0, 0.0, 0.0, 0.0
    total = sum(counter.values())
    label_count = counter[label]
    return (
        1.0,
        math.log1p(total),
        math.log1p(label_count),
        label_count / total,
    )


def connected_span_components(
    spans: list[tuple[int, int, int]],
) -> list[list[tuple[int, int, int]]]:
    remaining = set(range(len(spans)))
    components: list[list[tuple[int, int, int]]] = []
    while remaining:
        stack = [remaining.pop()]
        indices: list[int] = []
        while stack:
            current = stack.pop()
            indices.append(current)
            current_span = spans[current][:2]
            neighbours = {
                index
                for index in remaining
                if overlap(current_span, spans[index][:2]) > 0
            }
            remaining -= neighbours
            stack.extend(neighbours)
        components.append([spans[index] for index in indices])
    return components


def candidate_keys_for_document(
    model_sets: list[set[EntityKey]], include_recombined: bool
) -> set[EntityKey]:
    keys: set[EntityKey] = set().union(*model_sets)
    if not include_recombined:
        return keys
    for label in LABELS:
        spans = [
            (start, end, model_index)
            for model_index, model in enumerate(model_sets)
            for candidate_label, start, end in model
            if candidate_label == label
        ]
        for component in connected_span_components(spans):
            if len({model_index for _, _, model_index in component}) < 2:
                continue
            starts = {start for start, _, _ in component}
            ends = {end for _, end, _ in component}
            if len(starts) * len(ends) > 64:
                continue
            keys.update(
                (label, start, end)
                for start in starts
                for end in ends
                if start < end
            )
    return keys


def feature_names(model_names: list[str]) -> list[str]:
    names = [f"label={label}" for label in LABELS]
    names.extend(f"mask={mask}" for mask in range(1 << len(model_names)))
    names.extend(f"exact:{name}" for name in model_names)
    names.extend(
        [
            "exact_votes",
            "is_recombined",
            "char_length_log",
            "word_count",
            "is_upper",
            "is_title",
            "is_lower",
            "has_digit",
            "has_at",
            "has_hash",
            "has_hyphen",
            "has_apostrophe",
            "has_latin",
            "has_cyrillic",
            "mixed_script",
            "left_edge",
            "left_space",
            "left_punct",
            "right_edge",
            "right_space",
            "right_punct",
            "surface_seen",
            "surface_total_log",
            "surface_label_log",
            "surface_label_share",
            "first_seen",
            "first_total_log",
            "first_label_log",
            "first_label_share",
            "last_seen",
            "last_total_log",
            "last_label_log",
            "last_label_share",
            "start_vote_count",
            "end_vote_count",
            "overlap_model_count",
            "exact_span_other_label_votes",
        ]
    )
    for name in model_names:
        names.extend(
            [
                f"max_iou:{name}",
                f"same_start:{name}",
                f"same_end:{name}",
                f"contained_or_contains:{name}",
            ]
        )
    return names


def build_feature_vector(
    text: str,
    key: EntityKey,
    mask: int,
    model_sets: list[set[EntityKey]],
    gazetteer: tuple[
        dict[str, Counter[str]], dict[str, Counter[str]], dict[str, Counter[str]]
    ],
) -> tuple[float, ...]:
    label, start, end = key
    mention = text[start:end]
    normalized = mention.casefold()
    words = normalized.split()
    surfaces, first_tokens, last_tokens = gazetteer
    values: list[float] = [float(label == current) for current in LABELS]
    values.extend(float(mask == current) for current in range(1 << len(model_sets)))
    values.extend(float(bool(mask & (1 << index))) for index in range(len(model_sets)))
    exact_votes = mask.bit_count()
    values.extend(
        [
            float(exact_votes),
            float(exact_votes == 0),
            math.log1p(end - start),
            float(len(words)),
            float(bool(mention) and mention.isupper()),
            float(bool(mention) and mention.istitle()),
            float(bool(mention) and mention.islower()),
            float(any(char.isdigit() for char in mention)),
            float("@" in mention),
            float("#" in mention),
            float(any(char in "-–—" for char in mention)),
            float(any(char in "'’ʻʼ`" for char in mention)),
            *script_flags(mention),
            *boundary_kind(text[start - 1] if start else ""),
            *boundary_kind(text[end] if end < len(text) else ""),
            *counter_features(surfaces.get(normalized), label),
            *counter_features(first_tokens.get(words[0]) if words else None, label),
            *counter_features(last_tokens.get(words[-1]) if words else None, label),
        ]
    )

    same_label_spans = [
        [(candidate_start, candidate_end) for candidate_label, candidate_start, candidate_end in model if candidate_label == label]
        for model in model_sets
    ]
    values.extend(
        [
            float(sum(any(candidate_start == start for candidate_start, _ in spans) for spans in same_label_spans)),
            float(sum(any(candidate_end == end for _, candidate_end in spans) for spans in same_label_spans)),
            float(sum(any(overlap((start, end), span) > 0 for span in spans) for spans in same_label_spans)),
            float(
                sum(
                    any(
                        candidate_start == start
                        and candidate_end == end
                        and candidate_label != label
                        for candidate_label, candidate_start, candidate_end in model
                    )
                    for model in model_sets
                )
            ),
        ]
    )
    for spans in same_label_spans:
        max_iou = max((iou((start, end), span) for span in spans), default=0.0)
        same_start = any(candidate_start == start for candidate_start, _ in spans)
        same_end = any(candidate_end == end for _, candidate_end in spans)
        nested = any(
            (candidate_start <= start and end <= candidate_end)
            or (start <= candidate_start and candidate_end <= end)
            for candidate_start, candidate_end in spans
        )
        values.extend([max_iou, float(same_start), float(same_end), float(nested)])
    return tuple(values)


def build_candidates(
    train: list[JsonObject],
    gold: list[JsonObject],
    models: list[list[JsonObject]],
    include_recombined: bool,
) -> list[Candidate]:
    hashes = [record["hash"] for record in gold]
    for model in models:
        if [record["hash"] for record in model] != hashes:
            raise ValueError("gold and prediction hashes/order differ")
    gazetteer = build_gazetteer(train)
    candidates: list[Candidate] = []
    for doc_index, rows in enumerate(zip(gold, *models, strict=True)):
        gold_record = rows[0]
        text = gold_record.get("text")
        if not isinstance(text, str):
            raise ValueError("gold records must contain text")
        gold_keys = entity_set(gold_record)
        model_sets = [entity_set(row) for row in rows[1:]]
        for key in candidate_keys_for_document(model_sets, include_recombined):
            label, start, end = key
            if not 0 <= start < end <= len(text):
                continue
            mask = sum(
                1 << index for index, model in enumerate(model_sets) if key in model
            )
            candidates.append(
                Candidate(
                    doc_index=doc_index,
                    key=key,
                    mask=mask,
                    features=build_feature_vector(text, key, mask, model_sets, gazetteer),
                    is_gold=key in gold_keys,
                )
            )
    return candidates


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


def optimize_thresholds(
    candidates: list[Candidate], probabilities: np.ndarray, gold_counts: dict[str, int]
) -> dict[str, float]:
    thresholds = {label: 0.5 for label in LABELS}
    grid = np.linspace(0.05, 0.95, 181)
    for _ in range(3):
        for target_label in LABELS:
            best_threshold = thresholds[target_label]
            best_f1 = -1.0
            for threshold in grid:
                trial = {**thresholds, target_label: float(threshold)}
                tp = fp = 0
                for candidate, probability in zip(candidates, probabilities, strict=True):
                    if probability < trial[candidate.key[0]]:
                        continue
                    if candidate.is_gold:
                        tp += 1
                    else:
                        fp += 1
                score = metric(tp, fp, sum(gold_counts.values()))["f1"]
                if score > best_f1:
                    best_f1 = score
                    best_threshold = float(threshold)
            thresholds[target_label] = best_threshold
    return thresholds


def predictions_from_probabilities(
    gold: list[JsonObject],
    candidates: list[Candidate],
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    *,
    suppress_overlaps: bool,
) -> list[JsonObject]:
    by_doc: dict[int, list[tuple[float, Candidate]]] = defaultdict(list)
    for candidate, probability in zip(candidates, probabilities, strict=True):
        if probability >= thresholds[candidate.key[0]]:
            by_doc[candidate.doc_index].append((float(probability), candidate))

    result: list[JsonObject] = []
    for doc_index, record in enumerate(gold):
        selected: list[tuple[float, Candidate]] = []
        for probability, candidate in sorted(
            by_doc.get(doc_index, []), key=lambda item: (item[0], item[1].mask.bit_count()), reverse=True
        ):
            span = candidate.key[1:]
            if suppress_overlaps and any(overlap(span, other.key[1:]) > 0 for _, other in selected):
                continue
            selected.append((probability, candidate))
        entities = [
            {"label": candidate.key[0], "start": candidate.key[1], "end": candidate.key[2]}
            for _, candidate in selected
        ]
        entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
        result.append({"hash": record["hash"], "entities": entities})
    return result


def evaluate_records(gold: list[JsonObject], predictions: list[JsonObject]) -> JsonObject:
    by_label: dict[str, JsonObject] = {}
    total_tp = total_fp = total_gold = 0
    for label in LABELS:
        tp = fp = gold_count = 0
        for gold_record, prediction in zip(gold, predictions, strict=True):
            gold_keys = {key for key in entity_set(gold_record) if key[0] == label}
            predicted_keys = {key for key in entity_set(prediction) if key[0] == label}
            tp += len(gold_keys & predicted_keys)
            fp += len(predicted_keys - gold_keys)
            gold_count += len(gold_keys)
        by_label[label] = metric(tp, fp, gold_count)
        total_tp += tp
        total_fp += fp
        total_gold += gold_count
    return {"micro": metric(total_tp, total_fp, total_gold), "by_label": by_label}


def write_jsonl(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def estimator_factories(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        ),
        "histgb15": lambda: HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=5.0,
            random_state=seed,
        ),
        "histgb31": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=8.0,
            random_state=seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def run(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if args.names is not None and len(args.names) != len(args.predictions):
        raise ValueError("names must match predictions")
    names = args.names or [path.stem for path in args.predictions]
    train = read_jsonl(args.train)
    gold = read_jsonl(args.gold)
    models = [read_jsonl(path) for path in args.predictions]
    candidates = build_candidates(
        train, gold, models, include_recombined=args.include_recombined_boundaries
    )
    names_features = feature_names(names)
    if not candidates or len(candidates[0].features) != len(names_features):
        raise ValueError("feature construction mismatch")
    x = np.asarray([candidate.features for candidate in candidates], dtype=np.float32)
    y = np.asarray([candidate.is_gold for candidate in candidates], dtype=np.int8)
    doc_folds = [stable_fold(record["hash"], args.folds) for record in gold]
    candidate_folds = np.asarray([doc_folds[candidate.doc_index] for candidate in candidates])
    gold_counts = Counter(key[0] for record in gold for key in entity_set(record))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results: list[JsonObject] = []
    probabilities_by_model: dict[str, np.ndarray] = {}
    for model_name, factory in estimator_factories(args.seed).items():
        probabilities = np.zeros(len(candidates), dtype=np.float64)
        for fold in range(args.folds):
            train_mask = candidate_folds != fold
            test_mask = candidate_folds == fold
            estimator = factory()
            estimator.fit(x[train_mask], y[train_mask])
            probabilities[test_mask] = estimator.predict_proba(x[test_mask])[:, 1]
        thresholds = optimize_thresholds(candidates, probabilities, dict(gold_counts))
        variants: dict[str, JsonObject] = {}
        for suppress_overlaps in (False, True):
            variant_name = "nonoverlap" if suppress_overlaps else "all"
            predictions = predictions_from_probabilities(
                gold,
                candidates,
                probabilities,
                thresholds,
                suppress_overlaps=suppress_overlaps,
            )
            variants[variant_name] = evaluate_records(gold, predictions)
        model_results.append(
            {"model": model_name, "thresholds": thresholds, "variants": variants}
        )
        probabilities_by_model[model_name] = probabilities
        best_variant_name = max(variants, key=lambda name: variants[name]["micro"]["f1"])
        print(
            f"{model_name}: {best_variant_name} "
            f"OOF F1={variants[best_variant_name]['micro']['f1']:.6f}"
        )

    best_result = max(
        model_results,
        key=lambda result: max(
            variant["micro"]["f1"] for variant in result["variants"].values()
        ),
    )
    best_variant = max(
        best_result["variants"],
        key=lambda name: best_result["variants"][name]["micro"]["f1"],
    )
    best_name = best_result["model"]
    best_probabilities = probabilities_by_model[best_name]
    oof_predictions = predictions_from_probabilities(
        gold,
        candidates,
        best_probabilities,
        best_result["thresholds"],
        suppress_overlaps=best_variant == "nonoverlap",
    )
    write_jsonl(output_dir / "dev_predictions_oof.jsonl", oof_predictions)

    final_estimator = estimator_factories(args.seed)[best_name]()
    final_estimator.fit(x, y)
    fitted_probabilities = final_estimator.predict_proba(x)[:, 1]
    fitted_predictions = predictions_from_probabilities(
        gold,
        candidates,
        fitted_probabilities,
        best_result["thresholds"],
        suppress_overlaps=best_variant == "nonoverlap",
    )
    fitted_metrics = evaluate_records(gold, fitted_predictions)
    write_jsonl(output_dir / "dev_predictions_fitted.jsonl", fitted_predictions)
    joblib.dump(
        {
            "estimator": final_estimator,
            "feature_names": names_features,
            "model_names": names,
            "thresholds": best_result["thresholds"],
            "suppress_overlaps": best_variant == "nonoverlap",
            "include_recombined_boundaries": args.include_recombined_boundaries,
        },
        output_dir / "stacker.joblib",
        compress=3,
    )
    experiment = {
        "schema_version": 1,
        "models": names,
        "prediction_files": [str(path) for path in args.predictions],
        "folds": args.folds,
        "seed": args.seed,
        "candidates": len(candidates),
        "positive_candidates": int(y.sum()),
        "include_recombined_boundaries": args.include_recombined_boundaries,
        "model_results": model_results,
        "selected_model": best_name,
        "selected_variant": best_variant,
        "oof_metrics": best_result["variants"][best_variant],
        "fitted_dev_metrics": fitted_metrics,
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    oof_micro = experiment["oof_metrics"]["micro"]
    fitted_micro = fitted_metrics["micro"]
    report = (
        "# Span candidate stacker\n\n"
        f"Candidates: {len(candidates):,}; gold candidates in pool: {int(y.sum()):,}.\n\n"
        f"Selected OOF model: `{best_name}` with `{best_variant}` postprocessing.\n\n"
        f"OOF: `P={oof_micro['precision']:.4f}`, `R={oof_micro['recall']:.4f}`, "
        f"`F1={oof_micro['f1']:.4f}`.\n\n"
        f"Fitted dev (optimistic): `P={fitted_micro['precision']:.4f}`, "
        f"`R={fitted_micro['recall']:.4f}`, `F1={fitted_micro['f1']:.4f}`.\n\n"
        "The OOF probabilities are produced without training on the held-out document fold. "
        "The fitted-dev metric is not an unbiased estimate; the serialized model is intended "
        "for a separate hidden test.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        f"Selected {best_name}/{best_variant}: OOF F1={oof_micro['f1']:.6f}; "
        f"fitted-dev F1={fitted_micro['f1']:.6f}"
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
