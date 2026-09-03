from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = {"ORG", "NAME", "GEO"}
JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-span majority vote for NER predictions.")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not isinstance(record.get("hash"), str) or not isinstance(record.get("entities"), list):
                raise ValueError(f"{path}:{line_number}: invalid prediction")
            records.append(record)
    return records


def entity_key(entity: JsonObject) -> tuple[str, int, int]:
    label = entity.get("label")
    start = entity.get("start")
    end = entity.get("end")
    if label not in LABELS or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"invalid entity: {entity}")
    return label, start, end


def run(args: argparse.Namespace) -> None:
    if args.min_votes < 1 or args.min_votes > len(args.predictions):
        raise ValueError("min-votes must be between 1 and the number of prediction files")
    models = [read_predictions(path) for path in args.predictions]
    hashes = [record["hash"] for record in models[0]]
    if any([record["hash"] for record in model] != hashes for model in models[1:]):
        raise ValueError("prediction hashes/order differ")

    result: list[JsonObject] = []
    for rows in zip(*models, strict=True):
        votes: Counter[tuple[str, int, int]] = Counter()
        for row in rows:
            votes.update({entity_key(entity) for entity in row["entities"]})
        entities = [
            {"label": label, "start": start, "end": end}
            for (label, start, end), count in votes.items()
            if count >= args.min_votes
        ]
        entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
        result.append({"hash": rows[0]["hash"], "entities": entities})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in result:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    print(f"Predictions: {args.output} ({len(result)} records, min_votes={args.min_votes})")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
