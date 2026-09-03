from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from baseline.common import align_labels, load_fast_tokenizer, read_records, tags_for_scheme
from baseline.transition_confidence import estimate_transition_priors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate BIO/BILOU transition priors from gold train spans."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Checkpoint/tokenizer directory; tag scheme is read from baseline_config.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag-scheme", choices=("bio", "bilou"))
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _baseline_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "baseline_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _tag_scheme(args: argparse.Namespace, config: dict[str, Any]) -> str:
    tag_scheme = args.tag_scheme or config.get("tag_scheme")
    if tag_scheme not in {"bio", "bilou"}:
        raise ValueError(
            "tag scheme is absent from baseline_config.json; pass --tag-scheme"
        )
    configured_tags = config.get("tags")
    expected_tags = list(tags_for_scheme(tag_scheme))
    if configured_tags is not None and configured_tags != expected_tags:
        raise ValueError("baseline_config.json tags do not match its tag scheme")
    return str(tag_scheme)


def _document_tag_sequence(
    tokenizer: Any, text: str, entities: list[dict[str, Any]], tag_scheme: str
) -> list[str]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    raw_offsets = encoded["offset_mapping"]
    offsets = [(int(start), int(end)) for start, end in raw_offsets]
    tags = tags_for_scheme(tag_scheme)
    labels = align_labels(offsets, entities, tag_scheme=tag_scheme)
    return [tags[label] for label in labels if label != -100]


def run(args: argparse.Namespace) -> Path:
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("max-records must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"output already exists: {output}; pass --overwrite to replace it")

    model_dir = args.model_dir.expanduser().resolve()
    config = _baseline_config(model_dir)
    tag_scheme = _tag_scheme(args, config)
    tokenizer = load_fast_tokenizer(str(model_dir))
    train_path = args.train.expanduser().resolve()
    records = read_records(
        train_path, require_entities=True, limit=args.max_records
    )
    sequences = [
        _document_tag_sequence(
            tokenizer, record["text"], record["entities"], tag_scheme
        )
        for record in tqdm(records, desc="Gold tag sequences", unit="doc")
    ]
    payload = estimate_transition_priors(
        sequences, tags_for_scheme(tag_scheme), smoothing=args.smoothing
    )
    payload["source"] = {
        "train": str(train_path),
        "model_dir": str(model_dir),
        "record_count": len(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Records: {len(records)}")
    print(f"Non-empty sequences: {payload['sequence_count']}")
    print(f"Tokens: {payload['token_count']}")
    print(f"Transition priors: {output}")
    return output


def main() -> int:
    try:
        run(parse_args())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
