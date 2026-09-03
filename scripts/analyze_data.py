from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

LABELS = ("ORG", "NAME", "GEO")
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
APOSTROPHES = frozenset("'’ʻʼ‘`")
ATTACHED_SUFFIXES = tuple(
    sorted(
        {
            "provinsiyasidagi",
            "respublikasidagi",
            "viloyatidagi",
            "tumanidagi",
            "shahridagi",
            "вилоятидаги",
            "туманидаги",
            "шаҳридаги",
            "ларининг",
            "larining",
            "ларидан",
            "laridan",
            "ларига",
            "lariga",
            "ларини",
            "larini",
            "ларнинг",
            "larning",
            "dagi",
            "нинг",
            "ning",
            "дан",
            "dan",
            "га",
            "ga",
            "да",
            "da",
            "ни",
            "ni",
        },
        key=len,
        reverse=True,
    )
)

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Uzbek NER train/dev data.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    return parser.parse_args()


def read_records(path: Path) -> tuple[list[JsonObject], dict[str, Any]]:
    records: list[JsonObject] = []
    hashes: set[str] = set()
    invalid_entities = 0
    overlapping_entities = 0
    duplicate_hashes = 0

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            record_hash = record.get("hash")
            text = record.get("text")
            entities = record.get("entities")
            if not isinstance(record_hash, str) or not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: invalid hash or text")
            if not isinstance(entities, list):
                raise ValueError(f"{path}:{line_number}: entities must be a list")
            if record_hash in hashes:
                duplicate_hashes += 1
            hashes.add(record_hash)

            ordered = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))
            for entity in ordered:
                if (
                    entity.get("label") not in LABELS
                    or not isinstance(entity.get("start"), int)
                    or not isinstance(entity.get("end"), int)
                    or not 0 <= entity["start"] < entity["end"] <= len(text)
                ):
                    invalid_entities += 1
            overlapping_entities += sum(
                right["start"] < left["end"]
                for left, right in zip(ordered, ordered[1:], strict=False)
            )
            records.append(record)

    integrity = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duplicate_hashes": duplicate_hashes,
        "invalid_entities": invalid_entities,
        "overlapping_entities": overlapping_entities,
    }
    return records, integrity


def nearest_quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        f"p{int(quantile * 100):02d}": ordered[round(quantile * (len(ordered) - 1))]
        for quantile in QUANTILES
    }


def detect_script(text: str) -> str:
    has_latin = False
    has_cyrillic = False
    for character in text:
        name = unicodedata.name(character, "")
        has_latin = has_latin or "LATIN" in name
        has_cyrillic = has_cyrillic or "CYRILLIC" in name
    if has_latin and has_cyrillic:
        return "mixed"
    if has_latin:
        return "latin"
    if has_cyrillic:
        return "cyrillic"
    return "other"


def mention(record: JsonObject, entity: JsonObject) -> str:
    return record["text"][entity["start"] : entity["end"]]


def hash_date(record_hash: str) -> str | None:
    candidate = record_hash[-8:]
    try:
        datetime.strptime(candidate, "%Y%m%d")
    except ValueError:
        return None
    return candidate


def summarize_split(records: list[JsonObject]) -> dict[str, Any]:
    entities = [
        (record, entity, mention(record, entity))
        for record in records
        for entity in record["entities"]
    ]
    scripts = Counter(detect_script(record["text"]) for record in records)
    exact_texts = Counter(record["text"] for record in records)
    normalized_texts = Counter(
        re.sub(r"\s+", " ", record["text"]).strip().casefold()
        for record in records
    )
    dates = Counter(
        date
        for record in records
        if (date := hash_date(record["hash"])) is not None
    )
    by_label: dict[str, Any] = {}
    for label in LABELS:
        label_mentions = [surface for _, entity, surface in entities if entity["label"] == label]
        by_label[label] = {
            "count": len(label_mentions),
            "character_length": nearest_quantiles([len(surface) for surface in label_mentions]),
            "word_length": nearest_quantiles([len(surface.split()) for surface in label_mentions]),
            "multiword": sum(len(surface.split()) > 1 for surface in label_mentions),
            "with_apostrophe": sum(bool(APOSTROPHES.intersection(surface)) for surface in label_mentions),
            "with_hyphen": sum("-" in surface for surface in label_mentions),
            "with_dot": sum("." in surface for surface in label_mentions),
            "unique_case_sensitive": len(set(label_mentions)),
            "unique_casefolded": len({surface.casefold() for surface in label_mentions}),
        }

    return {
        "records": len(records),
        "records_with_entities": sum(bool(record["entities"]) for record in records),
        "empty_records": sum(not record["entities"] for record in records),
        "entities": len(entities),
        "scripts": dict(sorted(scripts.items())),
        "text_character_length": nearest_quantiles([len(record["text"]) for record in records]),
        "text_word_length": nearest_quantiles([len(record["text"].split()) for record in records]),
        "records_with_newlines": sum("\n" in record["text"] for record in records),
        "records_with_urls": sum(
            bool(re.search(r"https?://|www\.", record["text"], flags=re.IGNORECASE))
            for record in records
        ),
        "records_with_mentions": sum("@" in record["text"] for record in records),
        "records_with_hashtags": sum("#" in record["text"] for record in records),
        "exact_duplicate_rows": sum(count - 1 for count in exact_texts.values() if count > 1),
        "normalized_duplicate_rows": sum(
            count - 1 for count in normalized_texts.values() if count > 1
        ),
        "hashes_without_date_suffix": sum(hash_date(record["hash"]) is None for record in records),
        "top_hash_date_suffixes": dict(dates.most_common(20)),
        "by_label": by_label,
    }


def surface_counts(records: list[JsonObject], *, casefold: bool) -> dict[str, Counter[str]]:
    result = {label: Counter() for label in LABELS}
    for record in records:
        for entity in record["entities"]:
            surface = mention(record, entity)
            key = surface.casefold() if casefold else surface
            result[entity["label"]][key] += 1
    return result


def surface_analysis(train: list[JsonObject], dev: list[JsonObject]) -> dict[str, Any]:
    train_folded = surface_counts(train, casefold=True)
    dev_folded = surface_counts(dev, casefold=True)
    train_raw = surface_counts(train, casefold=False)

    seen_dev: dict[str, Any] = {}
    unseen_examples: dict[str, list[str]] = {}
    for label in LABELS:
        seen = sum(count for surface, count in dev_folded[label].items() if surface in train_folded[label])
        total = sum(dev_folded[label].values())
        seen_dev[label] = {
            "seen_mentions": seen,
            "total_mentions": total,
            "share": seen / total if total else 0.0,
        }
        unseen_examples[label] = [
            surface
            for surface, _ in dev_folded[label].most_common()
            if surface not in train_folded[label]
        ][:20]

    labels_by_surface: dict[str, Counter[str]] = defaultdict(Counter)
    display_surface: dict[str, str] = {}
    for record in train:
        for entity in record["entities"]:
            surface = mention(record, entity)
            folded = surface.casefold()
            labels_by_surface[folded][entity["label"]] += 1
            display_surface.setdefault(folded, surface)
    ambiguous = [
        {
            "surface": display_surface[surface],
            "labels": dict(counts),
            "count": sum(counts.values()),
        }
        for surface, counts in labels_by_surface.items()
        if len(counts) > 1
    ]
    ambiguous.sort(key=lambda item: (-item["count"], item["surface"].casefold()))

    return {
        "dev_seen_in_train_casefolded": seen_dev,
        "unseen_dev_examples_casefolded": unseen_examples,
        "ambiguous_train_surface_count": len(ambiguous),
        "ambiguous_train_surfaces": ambiguous[:100],
        "top_train_surfaces": {
            label: [
                {"surface": surface, "count": count}
                for surface, count in train_raw[label].most_common(30)
            ]
            for label in LABELS
        },
    }


def suffix_analysis(
    train: list[JsonObject],
    dev: list[JsonObject],
) -> dict[str, Any]:
    known = {
        label: set(surface_counts(train, casefold=True)[label])
        for label in LABELS
    }
    result: dict[str, Any] = {}
    for split_name, records in (("train", train), ("dev", dev)):
        split_result: dict[str, Any] = {}
        for label in LABELS:
            suffix_counter: Counter[str] = Counter()
            examples: list[dict[str, str]] = []
            for record in records:
                for entity in record["entities"]:
                    if entity["label"] != label:
                        continue
                    surface = mention(record, entity)
                    folded = surface.casefold()
                    for suffix in ATTACHED_SUFFIXES:
                        if not folded.endswith(suffix):
                            continue
                        base = folded[: -len(suffix)].rstrip("'’ʻʼ‘`")
                        if len(base) >= 4 and base in known[label]:
                            suffix_counter[suffix] += 1
                            if len(examples) < 20:
                                examples.append(
                                    {"surface": surface, "base_casefolded": base, "suffix": suffix}
                                )
                            break
            split_result[label] = {
                "matched_mentions": sum(suffix_counter.values()),
                "suffixes": dict(suffix_counter.most_common()),
                "examples": examples,
            }
        result[split_name] = split_result
    return result


def _wordish(character: str) -> bool:
    return character.isalnum() or character in APOSTROPHES or character == "_"


def build_lexicon_trie(train: list[JsonObject]) -> dict[str, Any]:
    labels_by_surface: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train:
        for entity in record["entities"]:
            surface = mention(record, entity)
            if len(surface) >= 2:
                labels_by_surface[surface][entity["label"]] += 1

    trie: dict[str, Any] = {}
    for surface, labels in labels_by_surface.items():
        node = trie
        for character in surface:
            node = node.setdefault(character, {})
        node[""] = labels.most_common(1)[0][0]
    return trie


def lexicon_entities(text: str, trie: dict[str, Any]) -> list[JsonObject]:
    candidates: list[tuple[int, int, str]] = []
    for start in range(len(text)):
        if start and _wordish(text[start - 1]):
            continue
        node = trie
        end = start
        matches: list[tuple[int, int, str]] = []
        while end < len(text) and text[end] in node:
            node = node[text[end]]
            end += 1
            if "" in node and (end == len(text) or not _wordish(text[end])):
                matches.append((start, end, node[""]))
        if matches:
            candidates.append(max(matches, key=lambda item: item[1] - item[0]))

    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if not any(
            candidate[0] < existing[1] and existing[0] < candidate[1]
            for existing in selected
        ):
            selected.append(candidate)
    selected.sort()
    return [
        {"label": label, "start": start, "end": end}
        for start, end, label in selected
    ]


def write_jsonl(path: Path, records: list[JsonObject]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def metric_values(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_predictions(gold: list[JsonObject], predicted: list[JsonObject]) -> dict[str, Any]:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for gold_record, pred_record in zip(gold, predicted, strict=True):
        gold_entities = {
            (entity["label"], entity["start"], entity["end"])
            for entity in gold_record["entities"]
        }
        pred_entities = {
            (entity["label"], entity["start"], entity["end"])
            for entity in pred_record["entities"]
        }
        for label in LABELS:
            gold_label = {entity for entity in gold_entities if entity[0] == label}
            pred_label = {entity for entity in pred_entities if entity[0] == label}
            counts[label]["tp"] += len(gold_label & pred_label)
            counts[label]["fp"] += len(pred_label - gold_label)
            counts[label]["fn"] += len(gold_label - pred_label)
    by_label = {
        label: metric_values(**label_counts)
        for label, label_counts in counts.items()
    }
    micro = metric_values(
        tp=sum(item["tp"] for item in counts.values()),
        fp=sum(item["fp"] for item in counts.values()),
        fn=sum(item["fn"] for item in counts.values()),
    )
    return {"by_label": by_label, "micro": micro}


def markdown_report(summary: dict[str, Any]) -> str:
    train = summary["splits"]["train"]
    dev = summary["splits"]["dev"]
    surfaces = summary["surfaces"]
    lexicon = summary["lexicon_baseline"]

    lines = [
        "# Uzbek NER: data analysis",
        "",
        "Generated by `scripts/analyze_data.py`. Percentages are mention-level unless noted otherwise.",
        "",
        "## Dataset overview",
        "",
        "| Metric | Train | Dev |",
        "|---|---:|---:|",
        f"| Records | {train['records']:,} | {dev['records']:,} |",
        f"| Records without entities | {train['empty_records']:,} ({train['empty_records']/train['records']:.1%}) | {dev['empty_records']:,} ({dev['empty_records']/dev['records']:.1%}) |",
        f"| Entities | {train['entities']:,} | {dev['entities']:,} |",
        f"| Median characters | {train['text_character_length']['p50']:,} | {dev['text_character_length']['p50']:,} |",
        f"| p95 characters | {train['text_character_length']['p95']:,} | {dev['text_character_length']['p95']:,} |",
        f"| Maximum characters | {train['text_character_length']['p100']:,} | {dev['text_character_length']['p100']:,} |",
        f"| Exact duplicate rows | {train['exact_duplicate_rows']:,} | {dev['exact_duplicate_rows']:,} |",
        f"| Casefolded/whitespace-normalized duplicate rows | {train['normalized_duplicate_rows']:,} | {dev['normalized_duplicate_rows']:,} |",
        "",
        "## Entity counts",
        "",
        "| Label | Train | Dev | Train unique (casefolded) | Dev unique (casefolded) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        lines.append(
            f"| {label} | {train['by_label'][label]['count']:,} | {dev['by_label'][label]['count']:,} | "
            f"{train['by_label'][label]['unique_casefolded']:,} | {dev['by_label'][label]['unique_casefolded']:,} |"
        )

    lines.extend(
        [
            "",
            "## Scripts",
            "",
            "| Script | Train documents | Dev documents |",
            "|---|---:|---:|",
        ]
    )
    for script in sorted(set(train["scripts"]) | set(dev["scripts"])):
        train_count = train["scripts"].get(script, 0)
        dev_count = dev["scripts"].get(script, 0)
        lines.append(
            f"| {script} | {train_count:,} ({train_count/train['records']:.1%}) | "
            f"{dev_count:,} ({dev_count/dev['records']:.1%}) |"
        )

    lines.extend(
        [
            "",
            "## Dev surface coverage by train (casefolded)",
            "",
            "| Label | Seen mentions | Total | Share |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        item = surfaces["dev_seen_in_train_casefolded"][label]
        lines.append(
            f"| {label} | {item['seen_mentions']:,} | {item['total_mentions']:,} | {item['share']:.1%} |"
        )

    lines.extend(
        [
            "",
            f"Train contains **{surfaces['ambiguous_train_surface_count']:,}** casefolded surfaces observed with more than one label.",
            "",
            "## Case-sensitive lexicon diagnostic",
            "",
            "The diagnostic memorizes train entity surfaces, keeps the majority label, requires word-like boundaries, chooses longest non-overlapping matches, and is evaluated on dev with exact spans.",
            "",
            "| Scope | Precision | Recall | F1 | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in (*LABELS, "micro"):
        item = lexicon["micro"] if label == "micro" else lexicon["by_label"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | "
            f"{item['tp']:,} | {item['fp']:,} | {item['fn']:,} |"
        )

    lines.extend(["", "## Most frequent train surfaces", ""])
    for label in LABELS:
        items = surfaces["top_train_surfaces"][label][:15]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(", ".join(f"`{item['surface']}` ({item['count']})" for item in items))
        lines.append("")

    lines.extend(["## Frequent ambiguous train surfaces", ""])
    lines.append("| Surface | Label counts | Total |")
    lines.append("|---|---|---:|")
    for item in surfaces["ambiguous_train_surfaces"][:25]:
        label_counts = ", ".join(f"{label}={count}" for label, count in item["labels"].items())
        lines.append(f"| `{item['surface']}` | {label_counts} | {item['count']} |")

    lines.extend(["", "## Representative unseen dev surfaces", ""])
    for label in LABELS:
        examples = surfaces["unseen_dev_examples_casefolded"][label][:15]
        lines.append(f"- **{label}:** " + ", ".join(f"`{item}`" for item in examples))

    lines.extend(
        [
            "",
            "## Practical implications",
            "",
            "- Long documents require overlapping windows or sentence-aware chunking; hard truncation is not acceptable.",
            "- Empty documents must remain in validation because they strongly affect false-positive precision.",
            "- High surface coverage makes a lexicon useful as a feature, but ambiguous surfaces require contextual classification.",
            "- NAME has the lowest surface coverage and is the clearest test of true generalization.",
            "- Attached Uzbek suffixes and apostrophe variants must be preserved in exact output spans.",
            "- Dev should not be used as training data; repeated tuning should use an internal train split.",
            "- Train and dev hashes show very similar date-suffix distributions, so the public dev is not a temporal holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    train_path = args.train.expanduser().resolve()
    dev_path = args.dev.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train, train_integrity = read_records(train_path)
    dev, dev_integrity = read_records(dev_path)
    train_hashes = {record["hash"] for record in train}
    train_texts = {record["text"] for record in train}

    trie = build_lexicon_trie(train)
    lexicon_predictions = [
        {
            "hash": record["hash"],
            "entities": lexicon_entities(record["text"], trie),
        }
        for record in dev
    ]
    lexicon_metrics = evaluate_predictions(dev, lexicon_predictions)

    summary = {
        "schema_version": 1,
        "paths": {"train": str(train_path), "dev": str(dev_path)},
        "integrity": {"train": train_integrity, "dev": dev_integrity},
        "cross_split": {
            "duplicate_hashes": sum(record["hash"] in train_hashes for record in dev),
            "exact_duplicate_texts": sum(record["text"] in train_texts for record in dev),
        },
        "splits": {
            "train": summarize_split(train),
            "dev": summarize_split(dev),
        },
        "surfaces": surface_analysis(train, dev),
        "attached_suffixes": suffix_analysis(train, dev),
        "lexicon_baseline": lexicon_metrics,
    }

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    predictions_path = output_dir / "lexicon_dev_predictions.jsonl"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    write_jsonl(predictions_path, lexicon_predictions)

    print(f"Train: {len(train):,} records")
    print(f"Dev: {len(dev):,} records")
    print(f"Lexicon dev micro-F1: {lexicon_metrics['micro']['f1']:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Lexicon predictions: {predictions_path}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
