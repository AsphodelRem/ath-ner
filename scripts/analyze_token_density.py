from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import tokenizers as tokenizers_library
import transformers
from transformers import AutoTokenizer


DEFAULT_TOKENIZERS = (
    ("mmbert_base", Path("artifacts/pretrained/mmbert-base")),
    ("xlm_roberta_base", Path("artifacts/experiments/xlm-roberta-base/model")),
    ("baseline_distilmbert", Path("artifacts/baseline/model")),
)
SCRIPT_NAMES = ("latin", "cyrillic")
SPLIT_NAMES = ("train", "dev", "combined")

JsonObject = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local tokenizer density on pure-Latin and pure-Cyrillic "
            "documents from the Uzbek NER train and dev sets."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument(
        "--tokenizer",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Local tokenizer to analyze; repeat for multiple tokenizers. "
            "Defaults to mmBERT, fine-tuned XLM-R, and the baseline model."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/eda/token_density.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, split: str) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: 'text' must be a string")
            script, letter_count = classify_document(text)
            records.append(
                {
                    "split": split,
                    "text": text,
                    "script": script,
                    "letters": letter_count,
                    "words": count_words(text),
                }
            )
    return records


def classify_document(text: str) -> tuple[str, int]:
    """Return a strict script class and the count of letters in that script.

    Unicode modifier letters (category Lm), which include common Uzbek apostrophe
    characters, are script-neutral. Other Unicode letter scripts make a document
    ineligible even when Latin or Cyrillic letters are also present.
    """

    latin = 0
    cyrillic = 0
    other = 0
    for character in text:
        category = unicodedata.category(character)
        if not category.startswith("L") or category == "Lm":
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            latin += 1
        elif "CYRILLIC" in name:
            cyrillic += 1
        else:
            other += 1

    if other:
        return "other_or_multiscript", 0
    if latin and cyrillic:
        return "mixed_latin_cyrillic", 0
    if latin:
        return "latin", latin
    if cyrillic:
        return "cyrillic", cyrillic
    return "letterless", 0


def count_words(text: str) -> int:
    """Count whitespace-delimited fields containing a letter or a digit."""

    return sum(any(character.isalnum() for character in field) for field in text.split())


def parse_tokenizers(values: list[str] | None) -> list[tuple[str, Path]]:
    if values is None:
        return list(DEFAULT_TOKENIZERS)
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --tokenizer {value!r}; expected NAME=PATH")
        if name in seen:
            raise ValueError(f"duplicate tokenizer name: {name}")
        seen.add(name)
        parsed.append((name, Path(raw_path)))
    return parsed


def batched(values: list[JsonObject], size: int) -> Iterable[list[JsonObject]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def rounded(value: float) -> float:
    return round(value, 6)


def median(values: list[int | float]) -> int | float:
    result = statistics.median(values)
    return rounded(result) if isinstance(result, float) else result


def summarize_documents(documents: list[JsonObject]) -> JsonObject:
    total_tokens = sum(int(document["tokens"]) for document in documents)
    total_letters = sum(int(document["letters"]) for document in documents)
    total_words = sum(int(document["words"]) for document in documents)
    if not documents or total_letters == 0 or total_words == 0:
        raise ValueError("eligible document group unexpectedly has a zero denominator")

    per_document_letters = [
        100.0 * int(document["tokens"]) / int(document["letters"])
        for document in documents
    ]
    per_document_words = [
        int(document["tokens"]) / int(document["words"])
        for document in documents
    ]
    return {
        "documents": len(documents),
        "total_letters": total_letters,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_100_letters": rounded(100.0 * total_tokens / total_letters),
        "tokens_per_word": rounded(total_tokens / total_words),
        "median_per_document": {
            "tokens": median([int(document["tokens"]) for document in documents]),
            "letters": median([int(document["letters"]) for document in documents]),
            "words": median([int(document["words"]) for document in documents]),
            "tokens_per_100_letters": median(per_document_letters),
            "tokens_per_word": median(per_document_words),
        },
    }


def compare_scripts(split_summary: JsonObject) -> JsonObject:
    latin = split_summary["latin"]
    cyrillic = split_summary["cyrillic"]
    fields = (
        "tokens_per_100_letters",
        "tokens_per_word",
    )
    comparison: JsonObject = {}
    for field in fields:
        latin_value = float(latin[field])
        cyrillic_value = float(cyrillic[field])
        comparison[field] = {
            "cyrillic_over_latin_ratio": rounded(cyrillic_value / latin_value),
            "cyrillic_minus_latin_gap": rounded(cyrillic_value - latin_value),
        }

    median_comparison: JsonObject = {}
    for field in fields:
        latin_value = float(latin["median_per_document"][field])
        cyrillic_value = float(cyrillic["median_per_document"][field])
        median_comparison[field] = {
            "cyrillic_over_latin_ratio": rounded(cyrillic_value / latin_value),
            "cyrillic_minus_latin_gap": rounded(cyrillic_value - latin_value),
        }
    comparison["median_per_document"] = median_comparison
    return comparison


def analyze_tokenizer(
    name: str,
    path: Path,
    records: list[JsonObject],
    batch_size: int,
) -> JsonObject:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_dir():
        raise ValueError(f"tokenizer directory does not exist: {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        local_files_only=True,
        use_fast=True,
    )

    eligible = [record.copy() for record in records if record["script"] in SCRIPT_NAMES]
    for batch in batched(eligible, batch_size):
        encoded = tokenizer(
            [str(document["text"]) for document in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            verbose=False,
        )
        for document, input_ids in zip(batch, encoded["input_ids"], strict=True):
            document["tokens"] = len(input_ids)

    splits: JsonObject = {}
    for split in SPLIT_NAMES:
        split_documents = (
            eligible if split == "combined" else [d for d in eligible if d["split"] == split]
        )
        script_summary = {
            script: summarize_documents(
                [document for document in split_documents if document["script"] == script]
            )
            for script in SCRIPT_NAMES
        }
        splits[split] = {
            **script_summary,
            "cyrillic_vs_latin": compare_scripts(script_summary),
        }

    return {
        "path": str(path),
        "resolved_path": str(resolved_path),
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocabulary_size": int(tokenizer.vocab_size),
        "splits": splits,
    }


def selection_summary(records: list[JsonObject]) -> JsonObject:
    summary: JsonObject = {}
    for split in SPLIT_NAMES:
        selected = records if split == "combined" else [r for r in records if r["split"] == split]
        counts = Counter(str(record["script"]) for record in selected)
        summary[split] = {
            "documents": len(selected),
            "document_classes": dict(sorted(counts.items())),
        }
    return summary


def markdown_report(summary: JsonObject) -> str:
    lines = [
        "# Latin vs Cyrillic tokenizer density",
        "",
        "All tokenizers were loaded from local directories with `local_files_only=True`. "
        "Tokenizer counts exclude model special tokens.",
        "",
        "Reproduce from the repository root with "
        "`.venv/bin/python scripts/analyze_token_density.py`.",
        "",
        "A pure-Latin document contains Latin letters and no Cyrillic or other-script "
        "letters; pure Cyrillic is defined symmetrically. Unicode modifier letters "
        "(including common Uzbek apostrophe characters) are script-neutral. Letters "
        "are Unicode Latin/Cyrillic letters. A word is a whitespace-delimited field "
        "containing at least one Unicode letter or digit.",
        "",
        "Corpus rates are micro-averages (`total tokens / total denominator`). Median "
        "rates are medians of per-document rates. Ratios are Cyrillic / Latin and "
        "gaps are Cyrillic - Latin.",
        "",
        "## Dataset selection",
        "",
        "| Split | All | Pure Latin | Pure Cyrillic | Mixed Latin/Cyrillic | Other/multiscript | Letterless |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLIT_NAMES:
        item = summary["dataset"]["selection"][split]
        classes = item["document_classes"]
        lines.append(
            f"| {split} | {item['documents']:,} | {classes.get('latin', 0):,} | "
            f"{classes.get('cyrillic', 0):,} | {classes.get('mixed_latin_cyrillic', 0):,} | "
            f"{classes.get('other_or_multiscript', 0):,} | {classes.get('letterless', 0):,} |"
        )

    lines.extend(
        [
            "",
            "## Combined train + dev",
            "",
            "| Tokenizer | Script | Docs | Tokens | Tokens/100 letters | Tokens/word | Median tokens/doc | Median tokens/100 letters | Median tokens/word |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in summary["tokenizers"].items():
        for script in SCRIPT_NAMES:
            item = model["splits"]["combined"][script]
            med = item["median_per_document"]
            lines.append(
                f"| {name} | {script} | {item['documents']:,} | {item['total_tokens']:,} | "
                f"{item['tokens_per_100_letters']:.3f} | {item['tokens_per_word']:.3f} | "
                f"{med['tokens']:.1f} | {med['tokens_per_100_letters']:.3f} | "
                f"{med['tokens_per_word']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Cyrillic overhead on combined train + dev",
            "",
            "| Tokenizer | Corpus T/100 ratio | Corpus T/100 gap | Corpus T/word ratio | Corpus T/word gap | Median T/100 ratio | Median T/100 gap | Median T/word ratio | Median T/word gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in summary["tokenizers"].items():
        comparison = model["splits"]["combined"]["cyrillic_vs_latin"]
        letters = comparison["tokens_per_100_letters"]
        words = comparison["tokens_per_word"]
        med_letters = comparison["median_per_document"]["tokens_per_100_letters"]
        med_words = comparison["median_per_document"]["tokens_per_word"]
        lines.append(
            f"| {name} | {letters['cyrillic_over_latin_ratio']:.3f} | "
            f"{letters['cyrillic_minus_latin_gap']:+.3f} | "
            f"{words['cyrillic_over_latin_ratio']:.3f} | "
            f"{words['cyrillic_minus_latin_gap']:+.3f} | "
            f"{med_letters['cyrillic_over_latin_ratio']:.3f} | "
            f"{med_letters['cyrillic_minus_latin_gap']:+.3f} | "
            f"{med_words['cyrillic_over_latin_ratio']:.3f} | "
            f"{med_words['cyrillic_minus_latin_gap']:+.3f} |"
        )

    for split in ("train", "dev"):
        lines.extend(
            [
                "",
                f"## {split.title()} corpus rates",
                "",
                "| Tokenizer | Latin tokens/100 letters | Cyrillic tokens/100 letters | Cyr/Lat ratio | Latin tokens/word | Cyrillic tokens/word | Cyr/Lat ratio |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, model in summary["tokenizers"].items():
            item = model["splits"][split]
            lines.append(
                f"| {name} | {item['latin']['tokens_per_100_letters']:.3f} | "
                f"{item['cyrillic']['tokens_per_100_letters']:.3f} | "
                f"{item['cyrillic_vs_latin']['tokens_per_100_letters']['cyrillic_over_latin_ratio']:.3f} | "
                f"{item['latin']['tokens_per_word']:.3f} | "
                f"{item['cyrillic']['tokens_per_word']:.3f} | "
                f"{item['cyrillic_vs_latin']['tokens_per_word']['cyrillic_over_latin_ratio']:.3f} |"
            )

    lines.extend(["", "## Local tokenizer paths", ""])
    for name, model in summary["tokenizers"].items():
        lines.append(f"- `{name}`: `{model['path']}` ({model['tokenizer_class']})")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    train_path = args.train.expanduser().resolve()
    dev_path = args.dev.expanduser().resolve()
    records = read_jsonl(train_path, "train") + read_jsonl(dev_path, "dev")
    tokenizers = parse_tokenizers(args.tokenizer)

    summary: JsonObject = {
        "schema_version": 1,
        "definitions": {
            "pure_script": (
                "At least one Unicode Latin/Cyrillic letter, no letter from the other "
                "script, and no non-modifier letter from another script."
            ),
            "letters": "Unicode non-modifier letters whose names contain LATIN or CYRILLIC.",
            "words": (
                "Whitespace-delimited nonempty fields containing at least one Unicode "
                "letter or digit."
            ),
            "tokens": "Tokenizer input IDs with add_special_tokens=False and no truncation.",
            "corpus_rate": "Micro-average: total tokens divided by total letters or words.",
            "comparison_direction": "Ratio is Cyrillic / Latin; gap is Cyrillic - Latin.",
        },
        "dataset": {
            "train": {"path": str(args.train), "sha256": sha256(train_path)},
            "dev": {"path": str(args.dev), "sha256": sha256(dev_path)},
            "selection": selection_summary(records),
        },
        "runtime": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "tokenizers": tokenizers_library.__version__,
        },
        "tokenizers": {},
    }
    for name, path in tokenizers:
        print(f"Analyzing {name}: {path}", flush=True)
        summary["tokenizers"][name] = analyze_tokenizer(
            name, path, records, args.batch_size
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_path.with_suffix(".md")
    report_path.write_text(markdown_report(summary), encoding="utf-8")
    print(f"JSON report: {output_path}")
    print(f"Markdown report: {report_path}")


def main() -> int:
    try:
        run(parse_args())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
