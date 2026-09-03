from __future__ import annotations

import argparse
import json
from pathlib import Path

from statistical.common import build_lexicon, learn_suffixes, read_jsonl, write_jsonl
from statistical.suffix_lexicon import MatcherConfig, predict_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a fixed exact or suffix-aware train lexicon to new JSONL."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("final/lexicon_configs.json"))
    parser.add_argument(
        "--variant",
        choices=("exact", "suffix"),
        required=True,
        help="Select the frozen configuration used by the final stacker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        config = MatcherConfig(**payload[args.variant])
        train = read_jsonl(args.train)
        records = read_jsonl(args.input, require_entities=False)
        lexicon = build_lexicon(train)
        suffixes = learn_suffixes(lexicon)
        predictions, source_counts = predict_records(records, lexicon, suffixes, config)
        write_jsonl(args.output, predictions)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"Variant: {args.variant}; records: {len(records)}; sources: {source_counts}")
    print(f"Predictions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
