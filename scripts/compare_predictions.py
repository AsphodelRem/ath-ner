from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from statistical.common import JsonObject, evaluate, read_jsonl, write_jsonl


def read_predictions(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not isinstance(record.get("hash"), str) or not isinstance(record.get("entities"), list):
                raise ValueError(f"{path}:{line_number}: invalid prediction")
            records.append({"hash": record["hash"], "entities": record["entities"]})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two exact-span NER prediction files.")
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def key(entity: JsonObject) -> tuple[str, int, int]:
    return entity["label"], entity["start"], entity["end"]


def overlaps(left: JsonObject, right: JsonObject, *, same_label: bool = False) -> bool:
    return (
        (not same_label or left["label"] == right["label"])
        and left["start"] < right["end"]
        and right["start"] < left["end"]
    )


def prioritize(primary: Sequence[JsonObject], secondary: Sequence[JsonObject]) -> list[JsonObject]:
    result = list(primary)
    for entity in secondary:
        if not any(overlaps(entity, existing) for existing in result):
            result.append(entity)
    return sorted(result, key=lambda item: (item["start"], item["end"], item["label"]))


def combine_record(left: JsonObject, right: JsonObject, method: str) -> JsonObject:
    left_entities = left["entities"]
    right_entities = right["entities"]
    left_keys = {key(entity) for entity in left_entities}
    right_keys = {key(entity) for entity in right_entities}
    if method == "exact_union":
        entities_by_key = {key(entity): entity for entity in [*left_entities, *right_entities]}
        entities = sorted(entities_by_key.values(), key=lambda item: (item["start"], item["end"], item["label"]))
    elif method == "exact_intersection":
        entities = [entity for entity in left_entities if key(entity) in right_keys]
    elif method == "left_priority_union":
        entities = prioritize(left_entities, right_entities)
    elif method == "right_priority_union":
        entities = prioritize(right_entities, left_entities)
    elif method == "left_overlap_filter":
        entities = [
            entity
            for entity in left_entities
            if any(overlaps(entity, other, same_label=True) for other in right_entities)
        ]
    elif method == "right_overlap_filter":
        entities = [
            entity
            for entity in right_entities
            if any(overlaps(entity, other, same_label=True) for other in left_entities)
        ]
    else:
        raise ValueError(f"unknown method: {method}")
    return {"hash": left["hash"], "entities": entities}


def run(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold)
    left = read_predictions(args.left)
    right = read_predictions(args.right)
    if [record["hash"] for record in gold] != [record["hash"] for record in left]:
        raise ValueError("left hashes/order differ from gold")
    if [record["hash"] for record in gold] != [record["hash"] for record in right]:
        raise ValueError("right hashes/order differ from gold")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = {args.left_name: left, args.right_name: right}
    methods = (
        "exact_union",
        "exact_intersection",
        "left_priority_union",
        "right_priority_union",
        "left_overlap_filter",
        "right_overlap_filter",
    )
    for method in methods:
        all_predictions[method] = [
            combine_record(left_record, right_record, method)
            for left_record, right_record in zip(left, right, strict=True)
        ]

    report = {name: evaluate(gold, predictions) for name, predictions in all_predictions.items()}
    for name, predictions in all_predictions.items():
        write_jsonl(output_dir / f"{name}.jsonl", predictions)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("method                    precision    recall        f1")
    print("-------------------------------------------------------")
    for name, metrics in report.items():
        micro = metrics["micro"]
        print(f"{name:25s} {micro['precision']:.4f}      {micro['recall']:.4f}    {micro['f1']:.4f}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
