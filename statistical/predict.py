from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from statistical.common import read_jsonl, write_jsonl
from statistical.crf import load_tagger, predict_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved statistical CRF NER model.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/statistical/crf"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model_dir / "bundle.joblib")
    model = load_tagger(args.model_dir / "model.crfsuite")
    records = read_jsonl(args.input, require_entities=False)
    threshold = (
        bundle["min_confidence"]
        if args.min_confidence is None
        else args.min_confidence
    )
    predictions = predict_records(
        records,
        model,
        bundle["extractor"],
        max_tokens=bundle["max_tokens"],
        min_confidence=threshold,
    )
    write_jsonl(args.output, predictions)
    print(f"Predictions: {args.output} ({len(predictions)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
