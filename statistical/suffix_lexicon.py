from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from statistical.common import (
    LABELS,
    KNOWN_ATTACHED_SUFFIXES,
    JsonObject,
    LexiconEntry,
    Token,
    boundary_alignment_stats,
    build_lexicon,
    evaluate,
    learn_suffixes,
    print_metrics,
    read_jsonl,
    stable_train_holdout,
    tokenize,
    write_jsonl,
)

TERMINAL = ""


@dataclass(frozen=True)
class MatcherConfig:
    exact_min_count: int = 1
    exact_min_purity: float = 0.75
    suffix_min_support: int = 3
    suffix_root_min_count: int = 2
    suffix_root_min_purity: float = 0.9
    allow_suffixes: bool = True
    suffix_policy: str = "learned"


@dataclass(frozen=True)
class Candidate:
    label: str
    start: int
    end: int
    score: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a suffix-aware lexicon NER baseline.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/statistical/suffix-lexicon"))
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    return parser.parse_args()


def build_trie(lexicon: dict[tuple[str, ...], LexiconEntry]) -> dict[str, Any]:
    trie: dict[str, Any] = {}
    for key, entry in lexicon.items():
        node = trie
        for token in key:
            node = node.setdefault(token, {})
        node[TERMINAL] = entry
    return trie


def suffix_index(
    suffixes: dict[str, dict[str, int]], min_support: int, suffix_policy: str
) -> dict[str, list[tuple[str, str, int]]]:
    result = [
        (suffix, label, support)
        for label in LABELS
        for suffix, support in suffixes[label].items()
        if support >= min_support
        and (suffix_policy == "learned" or suffix in KNOWN_ATTACHED_SUFFIXES)
    ]
    by_last_character: dict[str, list[tuple[str, str, int]]] = {}
    for item in sorted(result, key=lambda item: (-len(item[0]), -item[2], item[0], item[1])):
        by_last_character.setdefault(item[0][-1], []).append(item)
    return by_last_character


def accepted_exact(entry: LexiconEntry, config: MatcherConfig) -> bool:
    return entry.count >= config.exact_min_count and entry.purity >= config.exact_min_purity


def find_candidates(
    tokens: Sequence[Token],
    trie: dict[str, Any],
    learned_suffixes: dict[str, list[tuple[str, str, int]]],
    config: MatcherConfig,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for start_index, start_token in enumerate(tokens):
        node = trie
        for end_index in range(start_index, min(start_index + 24, len(tokens))):
            token = tokens[end_index]

            if config.allow_suffixes and len(token.norm) >= 3:
                for suffix, suffix_label, support in learned_suffixes.get(token.norm[-1], []):
                    if not token.norm.endswith(suffix) or len(token.norm) <= len(suffix) + 1:
                        continue
                    base = token.norm[: -len(suffix)].rstrip("'")
                    base_node = node.get(base)
                    if not isinstance(base_node, dict):
                        continue
                    entry = base_node.get(TERMINAL)
                    if (
                        isinstance(entry, LexiconEntry)
                        and entry.label == suffix_label
                        and entry.count >= config.suffix_root_min_count
                        and entry.purity >= config.suffix_root_min_purity
                    ):
                        score = (
                            1_000.0
                            + 15.0 * math.log1p(entry.count)
                            + 10.0 * math.log1p(support)
                            + token.end
                            - start_token.start
                        )
                        candidates.append(
                            Candidate(
                                label=entry.label,
                                start=start_token.start,
                                end=token.end,
                                score=score,
                                source="suffix",
                            )
                        )

            child = node.get(token.norm)
            if not isinstance(child, dict):
                break
            node = child
            entry = node.get(TERMINAL)
            if isinstance(entry, LexiconEntry) and accepted_exact(entry, config):
                score = (
                    2_000.0
                    + 20.0 * math.log1p(entry.count)
                    + 10.0 * entry.purity
                    + token.end
                    - start_token.start
                )
                candidates.append(
                    Candidate(
                        label=entry.label,
                        start=start_token.start,
                        end=token.end,
                        score=score,
                        source="exact",
                    )
                )
    return candidates


def select_non_overlapping(candidates: Sequence[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    unique = {(item.label, item.start, item.end, item.source): item for item in candidates}
    for candidate in sorted(
        unique.values(),
        key=lambda item: (-item.score, -(item.end - item.start), item.start, item.label),
    ):
        if any(candidate.start < old.end and old.start < candidate.end for old in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.start, item.end, item.label))


def predict_records(
    records: Sequence[JsonObject],
    lexicon: dict[tuple[str, ...], LexiconEntry],
    suffixes: dict[str, dict[str, int]],
    config: MatcherConfig,
) -> tuple[list[JsonObject], dict[str, int]]:
    trie = build_trie(lexicon)
    learned_suffixes = suffix_index(
        suffixes, config.suffix_min_support, config.suffix_policy
    )
    source_counts = {"exact": 0, "suffix": 0}
    predictions: list[JsonObject] = []
    for record in records:
        selected = select_non_overlapping(
            find_candidates(tokenize(record["text"]), trie, learned_suffixes, config)
        )
        for item in selected:
            source_counts[item.source] += 1
        predictions.append(
            {
                "hash": record["hash"],
                "entities": [
                    {"label": item.label, "start": item.start, "end": item.end}
                    for item in selected
                ],
            }
        )
    return predictions, source_counts


def tune_config(
    fitting: Sequence[JsonObject], holdout: Sequence[JsonObject]
) -> tuple[MatcherConfig, list[JsonObject]]:
    lexicon = build_lexicon(fitting)
    suffixes = learn_suffixes(lexicon)
    candidates = [
        MatcherConfig(exact_min_count=count, exact_min_purity=purity, allow_suffixes=False)
        for count in (1, 2, 3)
        for purity in (0.6, 0.8, 1.0)
    ]
    candidates += [
        MatcherConfig(
            exact_min_count=count,
            exact_min_purity=purity,
            suffix_min_support=suffix_support,
            suffix_root_min_count=root_count,
            suffix_root_min_purity=0.9,
            suffix_policy=suffix_policy,
        )
        for count in (1, 2)
        for purity in (0.8, 1.0)
        for suffix_support in (2, 4, 8)
        for root_count in (1, 2)
        for suffix_policy in ("learned", "uzbek_whitelist")
    ]
    trials: list[JsonObject] = []
    for config in candidates:
        predictions, source_counts = predict_records(holdout, lexicon, suffixes, config)
        metrics = evaluate(holdout, predictions)
        trials.append(
            {
                "config": asdict(config),
                "metrics": metrics,
                "source_counts": source_counts,
            }
        )

    # The case holder values precision, so F0.75 is the tie-breaking objective.
    def objective(trial: JsonObject) -> tuple[float, float]:
        precision = trial["metrics"]["micro"]["precision"]
        recall = trial["metrics"]["micro"]["recall"]
        beta2 = 0.75**2
        f075 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
        return f075, trial["metrics"]["micro"]["f1"]

    trials.sort(key=objective, reverse=True)
    return MatcherConfig(**trials[0]["config"]), trials


def serializable_suffixes(suffixes: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        label: dict(sorted(values.items(), key=lambda item: (-item[1], -len(item[0]), item[0])))
        for label, values in suffixes.items()
    }


def f075(trial: JsonObject) -> tuple[float, float]:
    precision = trial["metrics"]["micro"]["precision"]
    recall = trial["metrics"]["micro"]["recall"]
    beta2 = 0.75**2
    score = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
    return score, trial["metrics"]["micro"]["f1"]


def run(args: argparse.Namespace) -> None:
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    fitting, holdout = stable_train_holdout(train, holdout_fraction=args.holdout_fraction)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Internal split: fitting={len(fitting)} holdout={len(holdout)}")
    print(f"Token boundary coverage train: {boundary_alignment_stats(train)}")
    best_config, trials = tune_config(fitting, holdout)
    best_exact_trial = max(
        (trial for trial in trials if not trial["config"]["allow_suffixes"]),
        key=f075,
    )
    best_exact_config = MatcherConfig(**best_exact_trial["config"])
    best_suffix_trial = max(
        (trial for trial in trials if trial["config"]["allow_suffixes"]),
        key=lambda trial: trial["metrics"]["micro"]["f1"],
    )
    best_suffix_config = MatcherConfig(**best_suffix_trial["config"])
    print(f"Selected config: {best_config}")
    print_metrics("Internal holdout", trials[0]["metrics"])
    print_metrics("Internal holdout exact-only", best_exact_trial["metrics"])
    print(f"Best suffix config: {best_suffix_config}")
    print_metrics("Internal holdout with suffixes", best_suffix_trial["metrics"])

    full_lexicon = build_lexicon(train)
    full_suffixes = learn_suffixes(full_lexicon)
    dev_predictions, source_counts = predict_records(dev, full_lexicon, full_suffixes, best_config)
    dev_metrics = evaluate(dev, dev_predictions)
    print_metrics("Dev", dev_metrics)
    exact_predictions, exact_source_counts = predict_records(
        dev, full_lexicon, full_suffixes, best_exact_config
    )
    exact_dev_metrics = evaluate(dev, exact_predictions)
    print_metrics("Dev exact-only", exact_dev_metrics)
    suffix_predictions, suffix_source_counts = predict_records(
        dev, full_lexicon, full_suffixes, best_suffix_config
    )
    suffix_dev_metrics = evaluate(dev, suffix_predictions)
    print_metrics("Dev with suffixes", suffix_dev_metrics)

    write_jsonl(output_dir / "dev_predictions.jsonl", dev_predictions)
    write_jsonl(output_dir / "dev_predictions_exact.jsonl", exact_predictions)
    write_jsonl(output_dir / "dev_predictions_suffix.jsonl", suffix_predictions)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "selected_f0.75_config": dev_metrics,
                "best_exact_config": exact_dev_metrics,
                "best_suffix_config": suffix_dev_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "selected_config": asdict(best_config),
                "best_exact_config": asdict(best_exact_config),
                "best_suffix_config": asdict(best_suffix_config),
                "split": {"fitting": len(fitting), "holdout": len(holdout)},
                "alignment": {
                    "train": boundary_alignment_stats(train),
                    "dev": boundary_alignment_stats(dev),
                },
                "learned_suffixes": serializable_suffixes(full_suffixes),
                "dev_source_counts": source_counts,
                "exact_dev_source_counts": exact_source_counts,
                "suffix_dev_source_counts": suffix_source_counts,
                "internal_trials": trials,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
