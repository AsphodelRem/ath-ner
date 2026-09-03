from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("ORG", "NAME", "GEO")
APOSTROPHES = frozenset("'’ʻʼ‘`")

JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact-span NER errors.")
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def entity_key(entity: JsonObject) -> EntityKey:
    return entity["label"], entity["start"], entity["end"]


def overlap(left: EntityKey, right: EntityKey) -> bool:
    return left[1] < right[2] and right[1] < left[2]


def script_kind(text: str) -> str:
    latin = False
    cyrillic = False
    for character in text:
        name = unicodedata.name(character, "")
        latin = latin or "LATIN" in name
        cyrillic = cyrillic or "CYRILLIC" in name
    if latin and cyrillic:
        return "mixed"
    if latin:
        return "latin"
    if cyrillic:
        return "cyrillic"
    return "other"


def length_bucket(length: int) -> str:
    if length <= 5:
        return "01-05"
    if length <= 10:
        return "06-10"
    if length <= 20:
        return "11-20"
    return "21+"


def word_bucket(surface: str) -> str:
    words = len(surface.split())
    if words == 1:
        return "1"
    if words == 2:
        return "2"
    return "3+"


def document_bucket(length: int) -> str:
    if length <= 128:
        return "0000-0128"
    if length <= 512:
        return "0129-0512"
    if length <= 2048:
        return "0513-2048"
    return "2049+"


def classify_gold(entity: EntityKey, predictions: set[EntityKey]) -> str:
    if entity in predictions:
        return "exact"
    same_boundary = [item for item in predictions if item[1:] == entity[1:]]
    if same_boundary:
        return "wrong_label_exact_boundary"
    same_label_overlap = [
        item for item in predictions if item[0] == entity[0] and overlap(entity, item)
    ]
    if same_label_overlap:
        return "boundary_same_label"
    wrong_label_overlap = [item for item in predictions if overlap(entity, item)]
    if wrong_label_overlap:
        return "boundary_and_label"
    return "missed"


def classify_prediction(entity: EntityKey, gold: set[EntityKey]) -> str:
    if entity in gold:
        return "exact"
    same_boundary = [item for item in gold if item[1:] == entity[1:]]
    if same_boundary:
        return "wrong_label_exact_boundary"
    same_label_overlap = [item for item in gold if item[0] == entity[0] and overlap(entity, item)]
    if same_label_overlap:
        return "boundary_same_label"
    wrong_label_overlap = [item for item in gold if overlap(entity, item)]
    if wrong_label_overlap:
        return "boundary_and_label"
    return "spurious"


def build_train_surfaces(train: list[JsonObject]) -> dict[str, set[str]]:
    result = {label: set() for label in LABELS}
    for record in train:
        text = record["text"]
        for entity in record["entities"]:
            surface = text[entity["start"] : entity["end"]].casefold()
            result[entity["label"]].add(surface)
    return result


def example(
    record_hash: str,
    text: str,
    entity: EntityKey,
    overlapping: list[EntityKey],
) -> JsonObject:
    context_start = max(0, entity[1] - 80)
    context_end = min(len(text), entity[2] + 80)
    return {
        "hash": record_hash,
        "entity": {
            "label": entity[0],
            "start": entity[1],
            "end": entity[2],
            "surface": text[entity[1] : entity[2]],
        },
        "overlapping": [
            {
                "label": item[0],
                "start": item[1],
                "end": item[2],
                "surface": text[item[1] : item[2]],
            }
            for item in overlapping
        ],
        "context": text[context_start:context_end].replace("\n", " "),
    }


def analyze(
    train: list[JsonObject],
    gold_records: list[JsonObject],
    prediction_records: list[JsonObject],
) -> dict[str, Any]:
    predictions_by_hash = {record["hash"]: record for record in prediction_records}
    train_surfaces = build_train_surfaces(train)
    gold_categories = {label: Counter() for label in LABELS}
    prediction_categories = {label: Counter() for label in LABELS}
    recall_slices: dict[str, dict[str, Counter[str]]] = {
        "script": defaultdict(Counter),
        "seen_in_train": defaultdict(Counter),
        "character_length": defaultdict(Counter),
        "word_length": defaultdict(Counter),
        "document_length": defaultdict(Counter),
        "apostrophe": defaultdict(Counter),
    }
    examples: dict[str, list[JsonObject]] = defaultdict(list)
    empty_documents = {"documents": 0, "documents_with_predictions": 0, "predicted_entities": 0}

    for gold_record in gold_records:
        record_hash = gold_record["hash"]
        text = gold_record["text"]
        prediction_record = predictions_by_hash[record_hash]
        gold = {entity_key(entity) for entity in gold_record["entities"]}
        predicted = {entity_key(entity) for entity in prediction_record["entities"]}

        if not gold:
            empty_documents["documents"] += 1
            empty_documents["documents_with_predictions"] += bool(predicted)
            empty_documents["predicted_entities"] += len(predicted)

        for entity in gold:
            category = classify_gold(entity, predicted)
            label, start, end = entity
            surface = text[start:end]
            gold_categories[label][category] += 1
            slices = {
                "script": script_kind(text),
                "seen_in_train": "seen" if surface.casefold() in train_surfaces[label] else "unseen",
                "character_length": length_bucket(len(surface)),
                "word_length": word_bucket(surface),
                "document_length": document_bucket(len(text)),
                "apostrophe": "yes" if APOSTROPHES.intersection(surface) else "no",
            }
            for slice_name, bucket in slices.items():
                recall_slices[slice_name][bucket]["gold"] += 1
                recall_slices[slice_name][bucket]["exact"] += category == "exact"

            if category != "exact" and len(examples[category]) < 30:
                overlapping = sorted(
                    (item for item in predicted if overlap(entity, item)),
                    key=lambda item: (item[1], item[2], item[0]),
                )
                examples[category].append(example(record_hash, text, entity, overlapping))

        for entity in predicted:
            category = classify_prediction(entity, gold)
            prediction_categories[entity[0]][category] += 1
            if category == "spurious" and len(examples["spurious"]) < 30:
                examples["spurious"].append(example(record_hash, text, entity, []))

    serializable_slices: dict[str, dict[str, Any]] = {}
    for slice_name, buckets in recall_slices.items():
        serializable_slices[slice_name] = {}
        for bucket, counts in sorted(buckets.items()):
            gold_count = counts["gold"]
            exact_count = counts["exact"]
            serializable_slices[slice_name][bucket] = {
                "gold": gold_count,
                "exact": exact_count,
                "exact_recall": exact_count / gold_count if gold_count else 0.0,
            }

    return {
        "schema_version": 1,
        "gold_error_categories": {
            label: dict(gold_categories[label]) for label in LABELS
        },
        "prediction_error_categories": {
            label: dict(prediction_categories[label]) for label in LABELS
        },
        "recall_slices": serializable_slices,
        "empty_documents": empty_documents,
        "examples": dict(examples),
    }


def markdown_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Exact-span error analysis",
        "",
        "## Gold entity outcomes",
        "",
        "| Label | Exact | Boundary, same label | Wrong label, exact boundary | Boundary and label | Missed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        item = analysis["gold_error_categories"][label]
        lines.append(
            f"| {label} | {item.get('exact', 0):,} | {item.get('boundary_same_label', 0):,} | "
            f"{item.get('wrong_label_exact_boundary', 0):,} | {item.get('boundary_and_label', 0):,} | "
            f"{item.get('missed', 0):,} |"
        )

    empty = analysis["empty_documents"]
    lines.extend(
        [
            "",
            "## Empty documents",
            "",
            f"- Gold-empty documents: {empty['documents']:,}",
            f"- Empty documents with at least one prediction: {empty['documents_with_predictions']:,}",
            f"- Predicted entities in empty documents: {empty['predicted_entities']:,}",
            "",
            "## Exact recall slices",
            "",
        ]
    )
    for slice_name, buckets in analysis["recall_slices"].items():
        lines.extend(
            [
                f"### {slice_name}",
                "",
                "| Bucket | Exact | Gold | Recall |",
                "|---|---:|---:|---:|",
            ]
        )
        for bucket, item in buckets.items():
            lines.append(
                f"| {bucket} | {item['exact']:,} | {item['gold']:,} | {item['exact_recall']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    train = read_jsonl(args.train.expanduser().resolve())
    gold = read_jsonl(args.gold.expanduser().resolve())
    predictions = read_jsonl(args.predictions.expanduser().resolve())
    if {record["hash"] for record in gold} != {record["hash"] for record in predictions}:
        raise ValueError("prediction hashes differ from gold hashes")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(train, gold, predictions)
    summary_path = output_dir / "error_analysis.json"
    report_path = output_dir / "error_report.md"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(result), encoding="utf-8")
    print(f"Error analysis: {summary_path}")
    print(f"Error report: {report_path}")


def main() -> int:
    try:
        run(parse_args())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
