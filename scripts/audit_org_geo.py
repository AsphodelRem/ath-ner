from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = ("ORG", "NAME", "GEO")
FOCUS_LABELS = ("ORG", "GEO")
DEFAULT_PREDICTIONS = (
    (
        "mmbert_bilou_viterbi",
        Path("artifacts/experiments/mmbert-bilou/dev_predictions_viterbi.jsonl"),
    ),
    (
        "span_stacker_bilou_dual_decode_oof",
        Path(
            "artifacts/experiments/span-stacker-bilou-dual-decode/"
            "dev_predictions_oof.jsonl"
        ),
    ),
)
DEFAULT_SOURCE_CONFIG = Path(
    "artifacts/experiments/span-stacker-bilou-dual-decode/experiment.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/experiments/org-geo-audit")

JsonObject = dict[str, Any]
Boundary = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact-boundary ORG/GEO label confusions, document-length slices, "
            "and source-model consensus."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument(
        "--prediction",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Prediction file to audit; repeat for multiple systems. If omitted, the "
            "MMBERT BILOU Viterbi and span-stacker OOF files are used."
        ),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=DEFAULT_SOURCE_CONFIG,
        help="Stacker experiment JSON containing models and prediction_files.",
    )
    parser.add_argument(
        "--consensus-fraction",
        type=float,
        default=0.70,
        help="Minimum fraction of all source models voting for the opposite label.",
    )
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def parse_prediction_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return list(DEFAULT_PREDICTIONS)
    specs: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --prediction {value!r}; expected NAME=PATH")
        if name in names:
            raise ValueError(f"duplicate prediction name: {name}")
        names.add(name)
        specs.append((name, Path(raw_path)))
    return specs


def require_entity(entity: Any, *, path: Path, record_number: int, text: str | None) -> None:
    if not isinstance(entity, dict):
        raise ValueError(f"{path}:{record_number}: entity is not an object")
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if label not in LABELS:
        raise ValueError(f"{path}:{record_number}: invalid entity label {label!r}")
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError(f"{path}:{record_number}: invalid entity start {start!r}")
    if not isinstance(end, int) or isinstance(end, bool) or start < 0 or end <= start:
        raise ValueError(f"{path}:{record_number}: invalid entity offsets {start!r}:{end!r}")
    if text is not None and end > len(text):
        raise ValueError(
            f"{path}:{record_number}: entity end {end} exceeds text length {len(text)}"
        )


def validate_records(
    records: list[JsonObject], *, path: Path, labeled: bool
) -> dict[str, JsonObject]:
    by_hash: dict[str, JsonObject] = {}
    for record_number, record in enumerate(records, start=1):
        record_hash = record.get("hash")
        entities = record.get("entities")
        text = record.get("text") if labeled else None
        if not isinstance(record_hash, str) or not record_hash:
            raise ValueError(f"{path}:{record_number}: invalid hash")
        if record_hash in by_hash:
            raise ValueError(f"{path}:{record_number}: duplicate hash {record_hash}")
        if labeled and not isinstance(text, str):
            raise ValueError(f"{path}:{record_number}: invalid text")
        if not isinstance(entities, list):
            raise ValueError(f"{path}:{record_number}: invalid entities")
        seen_entities: set[tuple[str, int, int]] = set()
        for entity in entities:
            require_entity(entity, path=path, record_number=record_number, text=text)
            key = (entity["label"], entity["start"], entity["end"])
            if key in seen_entities:
                raise ValueError(f"{path}:{record_number}: duplicate entity {key}")
            seen_entities.add(key)
        by_hash[record_hash] = record
    return by_hash


def ensure_same_hashes(
    gold_by_hash: dict[str, JsonObject],
    prediction_by_hash: dict[str, JsonObject],
    *,
    prediction_name: str,
) -> None:
    gold_hashes = set(gold_by_hash)
    prediction_hashes = set(prediction_by_hash)
    if gold_hashes == prediction_hashes:
        return
    missing = sorted(gold_hashes - prediction_hashes)[:5]
    extra = sorted(prediction_hashes - gold_hashes)[:5]
    raise ValueError(
        f"{prediction_name}: hashes differ from dev; missing={missing}, extra={extra}"
    )


def normalize_surface(surface: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", surface).casefold()).strip()


def document_length_bucket(length: int) -> str:
    if length <= 128:
        return "0000-0128"
    if length <= 512:
        return "0129-0512"
    if length <= 2048:
        return "0513-2048"
    return "2049+"


def dataset_summary(records: Iterable[JsonObject]) -> JsonObject:
    documents = 0
    entity_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    for record in records:
        documents += 1
        length_counts[document_length_bucket(len(record["text"]))] += 1
        entity_counts.update(entity["label"] for entity in record["entities"])
    return {
        "documents": documents,
        "entities": {
            "total": sum(entity_counts.values()),
            **{label: entity_counts[label] for label in LABELS},
        },
        "documents_by_character_length": {
            bucket: length_counts[bucket]
            for bucket in ("0000-0128", "0129-0512", "0513-2048", "2049+")
        },
    }


def boundary_labels(record: JsonObject) -> dict[Boundary, set[str]]:
    result: dict[Boundary, set[str]] = defaultdict(set)
    for entity in record["entities"]:
        result[(entity["start"], entity["end"])].add(entity["label"])
    return dict(result)


def focus_outcome(gold_label: str, predicted_labels: set[str]) -> str:
    if gold_label in predicted_labels:
        return "exact_correct"
    opposite = "GEO" if gold_label == "ORG" else "ORG"
    if opposite in predicted_labels:
        return "opposite_label"
    if predicted_labels:
        return "other_exact_label"
    return "no_exact_boundary"


def empty_focus_counts() -> Counter[str]:
    return Counter(
        {
            "gold_ORG": 0,
            "gold_GEO": 0,
            "exact_correct_ORG": 0,
            "exact_correct_GEO": 0,
            "ORG_to_GEO": 0,
            "GEO_to_ORG": 0,
            "other_exact_label_ORG": 0,
            "other_exact_label_GEO": 0,
            "no_exact_boundary_ORG": 0,
            "no_exact_boundary_GEO": 0,
        }
    )


def add_focus_outcome(counts: Counter[str], gold_label: str, outcome: str) -> None:
    counts[f"gold_{gold_label}"] += 1
    if outcome == "exact_correct":
        counts[f"exact_correct_{gold_label}"] += 1
    elif outcome == "opposite_label":
        opposite = "GEO" if gold_label == "ORG" else "ORG"
        counts[f"{gold_label}_to_{opposite}"] += 1
    else:
        counts[f"{outcome}_{gold_label}"] += 1


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def finalize_focus_counts(counts: Counter[str]) -> JsonObject:
    gold_org = counts["gold_ORG"]
    gold_geo = counts["gold_GEO"]
    exact_org = counts["exact_correct_ORG"]
    exact_geo = counts["exact_correct_GEO"]
    org_to_geo = counts["ORG_to_GEO"]
    geo_to_org = counts["GEO_to_ORG"]
    other_org = counts["other_exact_label_ORG"]
    other_geo = counts["other_exact_label_GEO"]
    no_boundary_org = counts["no_exact_boundary_ORG"]
    no_boundary_geo = counts["no_exact_boundary_GEO"]
    total_gold = gold_org + gold_geo
    total_confusions = org_to_geo + geo_to_org
    return {
        "gold": {"total": total_gold, "ORG": gold_org, "GEO": gold_geo},
        "exact_correct": {"total": exact_org + exact_geo, "ORG": exact_org, "GEO": exact_geo},
        "exact_boundary_org_geo_confusions": {
            "total": total_confusions,
            "ORG_to_GEO": org_to_geo,
            "GEO_to_ORG": geo_to_org,
        },
        "other_exact_label": {
            "total": other_org + other_geo,
            "gold_ORG": other_org,
            "gold_GEO": other_geo,
        },
        "no_exact_boundary": {
            "total": no_boundary_org + no_boundary_geo,
            "gold_ORG": no_boundary_org,
            "gold_GEO": no_boundary_geo,
        },
        "rates": {
            "ORG_to_GEO_per_gold_ORG": ratio(org_to_geo, gold_org),
            "GEO_to_ORG_per_gold_GEO": ratio(geo_to_org, gold_geo),
            "ORG_GEO_confusion_per_focus_gold": ratio(total_confusions, total_gold),
            "exact_correct_per_focus_gold": ratio(exact_org + exact_geo, total_gold),
        },
    }


def audit_model(
    dev_records: list[JsonObject], prediction_by_hash: dict[str, JsonObject]
) -> tuple[JsonObject, dict[tuple[str, int, int], str]]:
    total = empty_focus_counts()
    by_length: dict[str, Counter[str]] = defaultdict(empty_focus_counts)
    outcomes: dict[tuple[str, int, int], str] = {}
    boundary_conflicts = 0
    for record in dev_records:
        record_hash = record["hash"]
        predictions = boundary_labels(prediction_by_hash[record_hash])
        boundary_conflicts += sum(len(labels) > 1 for labels in predictions.values())
        bucket = document_length_bucket(len(record["text"]))
        for entity in record["entities"]:
            gold_label = entity["label"]
            if gold_label not in FOCUS_LABELS:
                continue
            entity_id = (record_hash, entity["start"], entity["end"])
            predicted_labels = predictions.get((entity["start"], entity["end"]), set())
            outcome = focus_outcome(gold_label, predicted_labels)
            outcomes[entity_id] = outcome
            add_focus_outcome(total, gold_label, outcome)
            add_focus_outcome(by_length[bucket], gold_label, outcome)
    buckets = ("0000-0128", "0129-0512", "0513-2048", "2049+")
    return (
        {
            "overall": finalize_focus_counts(total),
            "by_document_character_length": {
                bucket: finalize_focus_counts(by_length[bucket]) for bucket in buckets
            },
            "prediction_boundaries_with_multiple_labels": boundary_conflicts,
        },
        outcomes,
    )


def build_train_surface_counts(
    train_records: list[JsonObject],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train_records:
        text = record["text"]
        for entity in record["entities"]:
            surface = normalize_surface(text[entity["start"] : entity["end"]])
            counts[surface][entity["label"]] += 1
    return dict(counts)


def surface_evidence_report(
    train_surface_counts: dict[str, Counter[str]], dev_records: list[JsonObject]
) -> JsonObject:
    per_label_unique = Counter()
    ambiguous: list[JsonObject] = []
    for surface, counts in train_surface_counts.items():
        for label in FOCUS_LABELS:
            if counts[label]:
                per_label_unique[label] += 1
        if counts["ORG"] and counts["GEO"]:
            ambiguous.append(
                {
                    "normalized_surface": surface,
                    "ORG": counts["ORG"],
                    "GEO": counts["GEO"],
                    "total": counts["ORG"] + counts["GEO"],
                }
            )
    ambiguous.sort(
        key=lambda item: (min(item["ORG"], item["GEO"]), item["total"], item["normalized_surface"]),
        reverse=True,
    )

    dev_counts = Counter()
    for record in dev_records:
        text = record["text"]
        for entity in record["entities"]:
            gold_label = entity["label"]
            if gold_label not in FOCUS_LABELS:
                continue
            surface = normalize_surface(text[entity["start"] : entity["end"]])
            counts = train_surface_counts.get(surface, Counter())
            opposite = "GEO" if gold_label == "ORG" else "ORG"
            if counts[gold_label] or counts[opposite]:
                dev_counts["seen_as_ORG_or_GEO"] += 1
            else:
                dev_counts["unseen_as_ORG_or_GEO"] += 1
            if counts[gold_label] and counts[opposite]:
                dev_counts["seen_with_both_labels"] += 1
            if counts[opposite] > counts[gold_label]:
                dev_counts["training_majority_is_opposite"] += 1

    return {
        "normalization": "Unicode NFKC, casefold, whitespace collapse",
        "unique_normalized_surfaces": {
            "ORG": per_label_unique["ORG"],
            "GEO": per_label_unique["GEO"],
            "appearing_as_both_ORG_and_GEO": len(ambiguous),
        },
        "dev_focus_surface_evidence": {
            key: dev_counts[key]
            for key in (
                "seen_as_ORG_or_GEO",
                "unseen_as_ORG_or_GEO",
                "seen_with_both_labels",
                "training_majority_is_opposite",
            )
        },
        "top_ambiguous_training_surfaces": ambiguous[:30],
    }


def load_source_predictions(
    source_config_path: Path,
    dev_by_hash: dict[str, JsonObject],
) -> tuple[list[str], dict[str, dict[str, JsonObject]], list[str]]:
    config = read_json(source_config_path)
    model_names = config.get("models")
    prediction_files = config.get("prediction_files")
    if not isinstance(model_names, list) or not all(isinstance(name, str) for name in model_names):
        raise ValueError(f"{source_config_path}: invalid models")
    if not isinstance(prediction_files, list) or not all(
        isinstance(path, str) for path in prediction_files
    ):
        raise ValueError(f"{source_config_path}: invalid prediction_files")
    if len(model_names) != len(prediction_files):
        raise ValueError(f"{source_config_path}: models/prediction_files length mismatch")

    source_maps: dict[str, dict[str, JsonObject]] = {}
    resolved_files: list[str] = []
    for name, raw_path in zip(model_names, prediction_files):
        path = Path(raw_path).expanduser().resolve()
        records = read_jsonl(path)
        source_map = validate_records(records, path=path, labeled=False)
        ensure_same_hashes(dev_by_hash, source_map, prediction_name=f"source {name}")
        source_maps[name] = source_map
        resolved_files.append(display_path(path))
    return model_names, source_maps, resolved_files


def entity_context(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def labels_for_boundary(record: JsonObject, start: int, end: int) -> list[str]:
    return sorted(
        {
            entity["label"]
            for entity in record["entities"]
            if entity["start"] == start and entity["end"] == end
        }
    )


def pairwise_and_consensus(
    dev_records: list[JsonObject],
    model_names: list[str],
    prediction_maps: dict[str, dict[str, JsonObject]],
    model_outcomes: dict[str, dict[tuple[str, int, int], str]],
    source_names: list[str],
    source_maps: dict[str, dict[str, JsonObject]],
    train_surface_counts: dict[str, Counter[str]],
    *,
    consensus_fraction: float,
    max_examples: int,
) -> tuple[JsonObject, JsonObject]:
    pair_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    opposite_vote_histogram: Counter[int] = Counter()
    suspicious_examples: list[JsonObject] = []
    threshold = math.ceil(consensus_fraction * len(source_names)) if source_names else 0

    for record in dev_records:
        record_hash = record["hash"]
        text = record["text"]
        for entity in record["entities"]:
            gold_label = entity["label"]
            if gold_label not in FOCUS_LABELS:
                continue
            start = entity["start"]
            end = entity["end"]
            entity_id = (record_hash, start, end)
            states = [model_outcomes[name][entity_id] for name in model_names]
            pair_counts["focus_gold_spans"] += 1
            pair_counts[" | ".join(states)] += 1
            if all(state == "exact_correct" for state in states):
                pair_counts["all_models_exact_correct"] += 1
            if all(state == "opposite_label" for state in states):
                pair_counts["all_models_same_opposite_label"] += 1

            votes: Counter[str] = Counter()
            votes_by_source: dict[str, list[str]] = {}
            for source_name in source_names:
                labels = labels_for_boundary(source_maps[source_name][record_hash], start, end)
                votes_by_source[source_name] = labels
                votes.update(labels)
            opposite = "GEO" if gold_label == "ORG" else "ORG"
            opposite_votes = votes[opposite]
            true_votes = votes[gold_label]
            opposite_vote_histogram[opposite_votes] += 1
            source_counts["focus_gold_spans"] += 1
            source_counts["at_least_one_source_exact_boundary"] += bool(sum(votes.values()))
            source_counts["high_consensus_correct"] += true_votes >= threshold
            source_counts["high_consensus_opposite"] += opposite_votes >= threshold

            if source_names and opposite_votes >= threshold:
                surface = text[start:end]
                normalized = normalize_surface(surface)
                train_counts = train_surface_counts.get(normalized, Counter())
                final_labels = {
                    name: labels_for_boundary(prediction_maps[name][record_hash], start, end)
                    for name in model_names
                }
                signals = ["source_consensus_opposes_gold"]
                if all(state == "opposite_label" for state in states):
                    signals.append("all_audited_models_predict_opposite")
                if train_counts[opposite] > train_counts[gold_label]:
                    signals.append("training_surface_majority_is_opposite")
                if train_counts[opposite] and train_counts[gold_label]:
                    signals.append("surface_has_mixed_ORG_GEO_training_labels")
                source_counts["high_consensus_opposite_all_audited_opposite"] += all(
                    state == "opposite_label" for state in states
                )
                source_counts["high_consensus_opposite_in_le_128"] += len(text) <= 128
                source_counts["high_consensus_opposite_training_majority_is_opposite"] += (
                    train_counts[opposite] > train_counts[gold_label]
                )
                suspicious_examples.append(
                    {
                        "hash": record_hash,
                        "document_character_length": len(text),
                        "document_length_bucket": document_length_bucket(len(text)),
                        "entity": {
                            "gold_label": gold_label,
                            "consensus_opposite_label": opposite,
                            "start": start,
                            "end": end,
                            "surface": surface,
                        },
                        "source_votes": {
                            "true_label": true_votes,
                            "opposite_label": opposite_votes,
                            "threshold": threshold,
                            "number_of_sources": len(source_names),
                            "by_label": {label: votes[label] for label in LABELS},
                            "by_source": votes_by_source,
                        },
                        "audited_model_boundary_labels": final_labels,
                        "training_normalized_surface_counts": {
                            label: train_counts[label] for label in LABELS
                        },
                        "signals": signals,
                        "context": entity_context(text, start, end),
                    }
                )

    suspicious_examples.sort(
        key=lambda item: (
            item["source_votes"]["opposite_label"],
            "all_audited_models_predict_opposite" in item["signals"],
            item["training_normalized_surface_counts"][
                item["entity"]["consensus_opposite_label"]
            ]
            - item["training_normalized_surface_counts"][item["entity"]["gold_label"]],
            -item["document_character_length"],
            item["hash"],
        ),
        reverse=True,
    )

    pairwise = {
        "model_order": model_names,
        "focus_gold_spans": pair_counts["focus_gold_spans"],
        "all_models_exact_correct": pair_counts["all_models_exact_correct"],
        "all_models_same_opposite_label": pair_counts["all_models_same_opposite_label"],
        "outcome_cross_tab": {
            key: value
            for key, value in sorted(pair_counts.items())
            if " | " in key
        },
    }
    consensus = {
        "source_models": source_names,
        "threshold": {
            "fraction": consensus_fraction,
            "minimum_opposite_votes": threshold,
            "number_of_sources": len(source_names),
        },
        "summary": {
            key: source_counts[key]
            for key in (
                "focus_gold_spans",
                "at_least_one_source_exact_boundary",
                "high_consensus_correct",
                "high_consensus_opposite",
                "high_consensus_opposite_all_audited_opposite",
                "high_consensus_opposite_in_le_128",
                "high_consensus_opposite_training_majority_is_opposite",
            )
        },
        "opposite_vote_histogram": {
            str(votes): opposite_vote_histogram[votes]
            for votes in range(len(source_names) + 1)
        },
        "suspicious_annotation_candidates": suspicious_examples[:max_examples],
        "candidate_count_before_example_limit": len(suspicious_examples),
        "interpretation_warning": (
            "These are review candidates, not proven annotation errors. The source models are "
            "correlated, and the stacker is trained from their predictions."
        ),
    }
    return pairwise, consensus


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def markdown_escape(value: Any, *, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().replace("|", "\\|")
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_report(report: JsonObject) -> str:
    model_names = list(report["models"])
    lines = [
        "# ORG/GEO exact-boundary audit",
        "",
        "This audit counts a confusion only when a gold ORG/GEO span has an exact-boundary "
        "prediction with the opposite label. Document length is measured in Unicode characters.",
        "",
        "## Data",
        "",
        "| Split | Documents | Entities | ORG | GEO | Documents <=128 chars |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev"):
        item = report["datasets"][split]
        lines.append(
            f"| {split} | {item['documents']:,} | {item['entities']['total']:,} | "
            f"{item['entities']['ORG']:,} | {item['entities']['GEO']:,} | "
            f"{item['documents_by_character_length']['0000-0128']:,} |"
        )

    surface = report["training_surface_evidence"]
    lines.extend(
        [
            "",
            "## Training-surface evidence",
            "",
            f"After {surface['normalization']}, train contains "
            f"{surface['unique_normalized_surfaces']['appearing_as_both_ORG_and_GEO']:,} "
            "surfaces observed as both ORG and GEO.",
            "",
            "| Dev ORG/GEO surface signal | Count |",
            "|---|---:|",
        ]
    )
    for key, value in surface["dev_focus_surface_evidence"].items():
        lines.append(f"| {key} | {value:,} |")

    first_name, second_name = model_names
    first_overall = report["models"][first_name]["overall"]
    second_overall = report["models"][second_name]["overall"]
    first_short = report["models"][first_name]["by_document_character_length"]["0000-0128"]
    second_short = report["models"][second_name]["by_document_character_length"]["0000-0128"]
    first_swaps = first_overall["exact_boundary_org_geo_confusions"]["total"]
    second_swaps = second_overall["exact_boundary_org_geo_confusions"]["total"]
    first_short_swaps = first_short["exact_boundary_org_geo_confusions"]["total"]
    second_short_swaps = second_short["exact_boundary_org_geo_confusions"]["total"]
    pairwise = report["pairwise_audited_models"]
    consensus = report["source_consensus"]
    source_summary = consensus["summary"]
    cross_tab = pairwise["outcome_cross_tab"]
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"- `{second_name}` reduces exact-boundary ORG/GEO swaps from {first_swaps:,} "
            f"to {second_swaps:,} ({percentage(ratio(first_swaps - second_swaps, first_swaps))} "
            "fewer).",
            f"- In documents <=128 characters, swaps fall from {first_short_swaps:,} "
            f"({percentage(first_short['rates']['ORG_GEO_confusion_per_focus_gold'])}) to "
            f"{second_short_swaps:,} "
            f"({percentage(second_short['rates']['ORG_GEO_confusion_per_focus_gold'])}). "
            f"The short-text swap rates are respectively "
            f"{first_short['rates']['ORG_GEO_confusion_per_focus_gold'] / first_overall['rates']['ORG_GEO_confusion_per_focus_gold']:.2f}x "
            f"and {second_short['rates']['ORG_GEO_confusion_per_focus_gold'] / second_overall['rates']['ORG_GEO_confusion_per_focus_gold']:.2f}x "
            "their all-length rates.",
            f"- Of the first model's {first_swaps:,} swaps, the second model makes "
            f"{cross_tab.get('opposite_label | exact_correct', 0):,} exact-correct, retains "
            f"{cross_tab.get('opposite_label | opposite_label', 0):,}, and has no exact-boundary "
            f"prediction for {cross_tab.get('opposite_label | no_exact_boundary', 0):,}. It adds "
            f"{cross_tab.get('exact_correct | opposite_label', 0) + cross_tab.get('no_exact_boundary | opposite_label', 0) + cross_tab.get('other_exact_label | opposite_label', 0):,} "
            "new opposite-label outcomes.",
            f"- Source consensus flags {source_summary['high_consensus_opposite']:,} spans for "
            f"annotation review; {source_summary['high_consensus_opposite_all_audited_opposite']:,} "
            "are also assigned the opposite label by both audited outputs. These remain review "
            "candidates because contextual ORG/GEO roles can legitimately override surface priors.",
        ]
    )

    lines.extend(
        [
            "",
            "## Exact-boundary ORG/GEO confusions",
            "",
            "| Model | Gold ORG | ORG->GEO | Rate | Gold GEO | GEO->ORG | Rate | Total swaps | Rate / ORG+GEO gold |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in model_names:
        item = report["models"][name]["overall"]
        confusion = item["exact_boundary_org_geo_confusions"]
        rates = item["rates"]
        lines.append(
            f"| {name} | {item['gold']['ORG']:,} | {confusion['ORG_to_GEO']:,} | "
            f"{percentage(rates['ORG_to_GEO_per_gold_ORG'])} | {item['gold']['GEO']:,} | "
            f"{confusion['GEO_to_ORG']:,} | {percentage(rates['GEO_to_ORG_per_gold_GEO'])} | "
            f"{confusion['total']:,} | {percentage(rates['ORG_GEO_confusion_per_focus_gold'])} |"
        )

    lines.extend(
        [
            "",
            "## Document-length slices",
            "",
            "| Model | Character-length bucket | ORG+GEO gold | ORG->GEO | GEO->ORG | Total swaps | Swap rate | Exact-correct rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in model_names:
        for bucket, item in report["models"][name]["by_document_character_length"].items():
            confusion = item["exact_boundary_org_geo_confusions"]
            lines.append(
                f"| {name} | {bucket} | {item['gold']['total']:,} | "
                f"{confusion['ORG_to_GEO']:,} | {confusion['GEO_to_ORG']:,} | "
                f"{confusion['total']:,} | "
                f"{percentage(item['rates']['ORG_GEO_confusion_per_focus_gold'])} | "
                f"{percentage(item['rates']['exact_correct_per_focus_gold'])} |"
            )

    lines.extend(
        [
            "",
            "## Agreement between audited models",
            "",
            f"Model order: `{pairwise['model_order'][0]}` then `{pairwise['model_order'][1]}`.",
            "",
            f"Both models exactly recover {pairwise['all_models_exact_correct']:,} of "
            f"{pairwise['focus_gold_spans']:,} gold ORG/GEO spans; both assign the same opposite "
            f"ORG/GEO label on {pairwise['all_models_same_opposite_label']:,} spans.",
            "",
            "| First model outcome | Second model outcome | Count |",
            "|---|---|---:|",
        ]
    )
    for cross_key, count in pairwise["outcome_cross_tab"].items():
        left, right = cross_key.split(" | ", maxsplit=1)
        lines.append(f"| {left} | {right} | {count:,} |")

    threshold = consensus["threshold"]
    summary = consensus["summary"]
    lines.extend(
        [
            "",
            "## Source-model consensus and annotation-review candidates",
            "",
            f"A high-consensus opposite label requires at least "
            f"{threshold['minimum_opposite_votes']} of {threshold['number_of_sources']} source "
            f"models ({percentage(threshold['fraction'])} configured threshold). "
            f"This flags {summary['high_consensus_opposite']:,} of "
            f"{summary['focus_gold_spans']:,} dev ORG/GEO spans. "
            "These are review candidates, not proven annotation errors: sources are correlated "
            "and the stacker is derived from them.",
            "",
            "| Hash | Doc chars | Gold -> vote | Votes (opp/true) | Audited outputs | Surface | Train ORG/GEO | Signals |",
            "|---|---:|---|---:|---|---|---:|---|",
        ]
    )
    for example in consensus["suspicious_annotation_candidates"]:
        entity = example["entity"]
        source_votes = example["source_votes"]
        final = ", ".join(
            f"{name}={'+'.join(labels) if labels else '-'}"
            for name, labels in example["audited_model_boundary_labels"].items()
        )
        train_counts = example["training_normalized_surface_counts"]
        lines.append(
            f"| `{example['hash']}` | {example['document_character_length']:,} | "
            f"{entity['gold_label']}->{entity['consensus_opposite_label']} | "
            f"{source_votes['opposite_label']}/{source_votes['true_label']} | "
            f"{markdown_escape(final, limit=90)} | "
            f"{markdown_escape(entity['surface'], limit=60)} | "
            f"{train_counts['ORG']}/{train_counts['GEO']} | "
            f"{markdown_escape(', '.join(example['signals']), limit=100)} |"
        )

    lines.extend(
        [
            "",
            "### Candidate contexts",
            "",
        ]
    )
    for index, example in enumerate(consensus["suspicious_annotation_candidates"], start=1):
        entity = example["entity"]
        lines.append(
            f"{index}. `{example['hash']}` ({entity['gold_label']}->"
            f"{entity['consensus_opposite_label']}, {example['source_votes']['opposite_label']}/"
            f"{example['source_votes']['number_of_sources']} opposite votes): "
            f"**{markdown_escape(entity['surface'], limit=80)}** — "
            f"{markdown_escape(example['context'], limit=240)}"
        )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "python3 scripts/audit_org_geo.py",
            "```",
            "",
            "Machine-readable details, including all per-source votes for each listed candidate, "
            "are in `audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.consensus_fraction <= 1.0:
        raise ValueError("--consensus-fraction must be in (0, 1]")
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")

    train_path = args.train.expanduser().resolve()
    dev_path = args.dev.expanduser().resolve()
    source_config_path = args.source_config.expanduser().resolve()
    prediction_specs = [
        (name, path.expanduser().resolve())
        for name, path in parse_prediction_specs(args.prediction)
    ]
    if len(prediction_specs) != 2:
        raise ValueError("exactly two --prediction files are required for pairwise analysis")

    train_records = read_jsonl(train_path)
    dev_records = read_jsonl(dev_path)
    validate_records(train_records, path=train_path, labeled=True)
    dev_by_hash = validate_records(dev_records, path=dev_path, labeled=True)

    prediction_maps: dict[str, dict[str, JsonObject]] = {}
    for name, path in prediction_specs:
        records = read_jsonl(path)
        prediction_map = validate_records(records, path=path, labeled=False)
        ensure_same_hashes(dev_by_hash, prediction_map, prediction_name=name)
        prediction_maps[name] = prediction_map

    source_names, source_maps, source_files = load_source_predictions(
        source_config_path, dev_by_hash
    )
    train_surface_counts = build_train_surface_counts(train_records)
    models: dict[str, JsonObject] = {}
    model_outcomes: dict[str, dict[tuple[str, int, int], str]] = {}
    for name, _ in prediction_specs:
        model_report, outcomes = audit_model(dev_records, prediction_maps[name])
        models[name] = model_report
        model_outcomes[name] = outcomes

    model_names = [name for name, _ in prediction_specs]
    pairwise, consensus = pairwise_and_consensus(
        dev_records,
        model_names,
        prediction_maps,
        model_outcomes,
        source_names,
        source_maps,
        train_surface_counts,
        consensus_fraction=args.consensus_fraction,
        max_examples=args.max_examples,
    )
    report: JsonObject = {
        "schema_version": 1,
        "inputs": {
            "train": display_path(train_path),
            "dev": display_path(dev_path),
            "predictions": {
                name: display_path(path) for name, path in prediction_specs
            },
            "source_config": display_path(source_config_path),
            "source_prediction_files": {
                name: path for name, path in zip(source_names, source_files)
            },
        },
        "definitions": {
            "exact_boundary_confusion": (
                "A gold ORG/GEO entity whose exact [start,end) boundary is predicted with the "
                "opposite ORG/GEO label and not with the gold label."
            ),
            "document_length": "Number of Unicode code points returned by Python len(text).",
            "document_length_buckets": ["0000-0128", "0129-0512", "0513-2048", "2049+"],
        },
        "validation": {
            "train_records": len(train_records),
            "dev_records": len(dev_records),
            "prediction_hash_sets_match_dev": True,
            "source_prediction_hash_sets_match_dev": True,
        },
        "datasets": {
            "train": dataset_summary(train_records),
            "dev": dataset_summary(dev_records),
        },
        "training_surface_evidence": surface_evidence_report(
            train_surface_counts, dev_records
        ),
        "models": models,
        "pairwise_audited_models": pairwise,
        "source_consensus": consensus,
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Audit JSON: {json_path}")
    print(f"Audit Markdown: {markdown_path}")
    for name in model_names:
        overall = models[name]["overall"]
        short = models[name]["by_document_character_length"]["0000-0128"]
        print(
            f"{name}: swaps={overall['exact_boundary_org_geo_confusions']['total']} "
            f"({overall['rates']['ORG_GEO_confusion_per_focus_gold']:.4%}); "
            f"<=128 swaps={short['exact_boundary_org_geo_confusions']['total']} "
            f"({short['rates']['ORG_GEO_confusion_per_focus_gold']:.4%})"
        )
    print(
        "High-consensus annotation-review candidates: "
        f"{consensus['summary']['high_consensus_opposite']}"
    )


def main() -> int:
    try:
        run(parse_args())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
