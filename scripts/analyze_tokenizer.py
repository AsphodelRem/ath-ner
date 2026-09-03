from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

LABELS = ("ORG", "NAME", "GEO")
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tokenizer fit for exact-span Uzbek NER.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument(
        "--model-name",
        default="distilbert/distilbert-base-multilingual-cased",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eda/tokenizer_report.json"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantiles(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{int(quantile * 100):02d}": ordered[round(quantile * (len(ordered) - 1))]
        for quantile in QUANTILES
    }


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


def window_ranges(
    tokenizer: Any,
    text: str,
    max_length: int,
    stride: int,
) -> list[tuple[int, int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    chunks = encoded["offset_mapping"]
    if chunks and isinstance(chunks[0], tuple):
        chunks = [chunks]
    ranges: list[tuple[int, int]] = []
    for offsets in chunks:
        content = [(int(start), int(end)) for start, end in offsets if start != end]
        if content:
            ranges.append((content[0][0], content[-1][1]))
    return ranges


def analyze_split(
    records: list[JsonObject],
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    content_limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    document_tokens: list[int] = []
    document_windows: list[int] = []
    documents_over_limit = 0
    unk_tokens = 0
    total_tokens = 0
    tokens_by_script: dict[str, Counter[str]] = defaultdict(Counter)
    entity_pieces = {label: [] for label in LABELS}
    entity_character_lengths = {label: [] for label in LABELS}
    exact_token_boundaries = Counter()
    unknown_entities = Counter()
    entities_with_partial_windows = Counter()
    entities_without_complete_window = Counter()
    partial_window_instances = Counter()
    fragmented_examples: list[JsonObject] = []

    for record in records:
        text = record["text"]
        script = script_kind(text)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = [int(item) for item in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        token_count = len(input_ids)
        document_tokens.append(token_count)
        total_tokens += token_count
        unk_count = sum(token_id == tokenizer.unk_token_id for token_id in input_ids)
        unk_tokens += unk_count
        tokens_by_script[script]["tokens"] += token_count
        tokens_by_script[script]["unk_tokens"] += unk_count
        tokens_by_script[script]["documents"] += 1
        documents_over_limit += token_count > content_limit

        ranges = window_ranges(tokenizer, text, max_length, stride)
        document_windows.append(len(ranges))
        for entity in record["entities"]:
            label = entity["label"]
            start = entity["start"]
            end = entity["end"]
            surface = text[start:end]
            overlapping_indices = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_start < end and start < token_end
            ]
            pieces = len(overlapping_indices)
            entity_pieces[label].append(pieces)
            entity_character_lengths[label].append(len(surface))
            if overlapping_indices:
                first_offset = offsets[overlapping_indices[0]]
                last_offset = offsets[overlapping_indices[-1]]
                exact = first_offset[0] == start and last_offset[1] == end
                exact_token_boundaries[(label, "exact" if exact else "inexact")] += 1
                if any(input_ids[index] == tokenizer.unk_token_id for index in overlapping_indices):
                    unknown_entities[label] += 1
            else:
                exact_token_boundaries[(label, "inexact")] += 1

            overlapping_windows = [
                window
                for window in ranges
                if window[0] < end and start < window[1]
            ]
            complete_windows = [
                window
                for window in overlapping_windows
                if window[0] <= start and end <= window[1]
            ]
            partial_count = len(overlapping_windows) - len(complete_windows)
            if partial_count:
                entities_with_partial_windows[label] += 1
                partial_window_instances[label] += partial_count
            if not complete_windows:
                entities_without_complete_window[label] += 1

            if pieces >= 10:
                fragmented_examples.append(
                    {
                        "hash": record["hash"],
                        "label": label,
                        "surface": surface,
                        "pieces": pieces,
                        "characters": len(surface),
                        "script": script,
                    }
                )

    fragmented_examples.sort(key=lambda item: (-item["pieces"], -item["characters"]))
    by_label: dict[str, Any] = {}
    for label in LABELS:
        total_entities = len(entity_pieces[label])
        exact_count = exact_token_boundaries[(label, "exact")]
        by_label[label] = {
            "entities": total_entities,
            "subtokens_per_entity": quantiles(entity_pieces[label]),
            "characters_per_entity": quantiles(entity_character_lengths[label]),
            "exact_token_boundary_entities": exact_count,
            "exact_token_boundary_share": exact_count / total_entities if total_entities else 0.0,
            "entities_with_unk": unknown_entities[label],
            "entities_with_partial_windows": entities_with_partial_windows[label],
            "partial_window_instances": partial_window_instances[label],
            "entities_without_complete_window": entities_without_complete_window[label],
        }

    return {
        "records": len(records),
        "content_token_limit": content_limit,
        "document_tokens": quantiles(document_tokens),
        "document_windows": quantiles(document_windows),
        "documents_over_limit": documents_over_limit,
        "documents_over_limit_share": documents_over_limit / len(records) if records else 0.0,
        "total_tokens": total_tokens,
        "unk_tokens": unk_tokens,
        "unk_token_share": unk_tokens / total_tokens if total_tokens else 0.0,
        "tokens_by_script": {
            script: {
                **dict(counts),
                "unk_token_share": counts["unk_tokens"] / counts["tokens"] if counts["tokens"] else 0.0,
            }
            for script, counts in sorted(tokens_by_script.items())
        },
        "by_label": by_label,
        "most_fragmented_entities": fragmented_examples[:50],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    train = summary["splits"]["train"]
    dev = summary["splits"]["dev"]
    lines = [
        "# Tokenizer fit report",
        "",
        f"Model: `{summary['model_name']}`; max length: {summary['max_length']}; stride: {summary['stride']}.",
        "",
        "## Documents and windows",
        "",
        "| Metric | Train | Dev |",
        "|---|---:|---:|",
        f"| Documents | {train['records']:,} | {dev['records']:,} |",
        f"| Documents over content limit | {train['documents_over_limit']:,} ({train['documents_over_limit_share']:.1%}) | {dev['documents_over_limit']:,} ({dev['documents_over_limit_share']:.1%}) |",
        f"| Median tokens | {train['document_tokens']['p50']:,} | {dev['document_tokens']['p50']:,} |",
        f"| p95 tokens | {train['document_tokens']['p95']:,} | {dev['document_tokens']['p95']:,} |",
        f"| Maximum tokens | {train['document_tokens']['p100']:,} | {dev['document_tokens']['p100']:,} |",
        f"| Maximum windows | {train['document_windows']['p100']:,} | {dev['document_windows']['p100']:,} |",
        f"| UNK token share | {train['unk_token_share']:.4%} | {dev['unk_token_share']:.4%} |",
        "",
        "## Entity tokenization and window boundaries",
        "",
        "| Label | Split | Median pieces | p95 pieces | Exact token boundaries | Entities touching a partial window | No complete window |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        for split_name, split in (("train", train), ("dev", dev)):
            item = split["by_label"][label]
            lines.append(
                f"| {label} | {split_name} | {item['subtokens_per_entity']['p50']} | "
                f"{item['subtokens_per_entity']['p95']} | {item['exact_token_boundary_share']:.1%} | "
                f"{item['entities_with_partial_windows']:,} | {item['entities_without_complete_window']:,} |"
            )
    lines.extend(
        [
            "",
            "`Entities touching a partial window` means at least one overlapping training window contains only part of the entity. The same entity can still be complete in another overlapping window.",
            "",
            "## Most fragmented dev entities",
            "",
            "| Label | Pieces | Characters | Surface |",
            "|---|---:|---:|---|",
        ]
    )
    for item in dev["most_fragmented_entities"][:25]:
        surface = item["surface"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['label']} | {item['pieces']} | {item['characters']} | `{surface}` |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.max_length < 2 or args.stride < 0:
        raise ValueError("invalid max-length or stride")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer with offset mappings is required")
    content_limit = args.max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= args.stride < content_limit:
        raise ValueError("stride must be smaller than the content token limit")

    train = read_jsonl(args.train.expanduser().resolve())
    dev = read_jsonl(args.dev.expanduser().resolve())
    summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "stride": args.stride,
        "splits": {
            "train": analyze_split(train, tokenizer, args.max_length, args.stride),
            "dev": analyze_split(dev, tokenizer, args.max_length, args.stride),
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_path.with_suffix(".md")
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(f"Tokenizer analysis: {output_path}")
    print(f"Tokenizer report: {report_path}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
