from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from stack_span_candidates import (
    build_candidates,
    feature_names,
    predictions_from_probabilities,
    read_jsonl,
    write_jsonl,
)

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a fitted span stacker to aligned model predictions."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_input(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not isinstance(record.get("hash"), str) or not isinstance(
                record.get("text"), str
            ):
                raise ValueError(f"{path}:{line_number}: input requires hash and text")
            records.append(
                {"hash": record["hash"], "text": record["text"], "entities": []}
            )
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def run(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.bundle)
    required = {
        "estimator",
        "feature_names",
        "model_names",
        "thresholds",
        "suppress_overlaps",
        "include_recombined_boundaries",
    }
    if not isinstance(bundle, dict) or not required <= set(bundle):
        raise ValueError("invalid stacker bundle")
    if len(args.predictions) != len(bundle["model_names"]):
        raise ValueError(
            f"expected {len(bundle['model_names'])} prediction files in this order: "
            + ", ".join(bundle["model_names"])
        )

    train = read_jsonl(args.train)
    inputs = read_input(args.input)
    models = [read_jsonl(path) for path in args.predictions]
    candidates = build_candidates(
        train,
        inputs,
        models,
        include_recombined=bool(bundle["include_recombined_boundaries"]),
    )
    expected_features = feature_names(list(bundle["model_names"]))
    if expected_features != bundle["feature_names"]:
        raise ValueError("bundle feature schema does not match current code")
    x = np.asarray([candidate.features for candidate in candidates], dtype=np.float32)
    probabilities = bundle["estimator"].predict_proba(x)[:, 1]
    predictions = predictions_from_probabilities(
        inputs,
        candidates,
        probabilities,
        dict(bundle["thresholds"]),
        suppress_overlaps=bool(bundle["suppress_overlaps"]),
    )
    write_jsonl(args.output, predictions)
    print(f"Records: {len(inputs)}, candidates: {len(candidates)}")
    print(f"Model order: {', '.join(bundle['model_names'])}")
    print(f"Predictions: {args.output}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
