from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import joblib
import pycrfsuite

from statistical.common import (
    JsonObject,
    Token,
    bilou_to_entities,
    boundary_alignment_stats,
    build_lexicon,
    chunk_slices,
    entities_to_bilou,
    evaluate,
    print_metrics,
    read_jsonl,
    stable_train_holdout,
    tokenize,
    write_jsonl,
)
from statistical.features import FeatureExtractor


@dataclass(frozen=True)
class CrfConfig:
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 70
    max_tokens: int = 384
    gazetteer_min_count: int = 2
    gazetteer_min_purity: float = 0.9
    suffix_min_support: int = 4
    suffix_policy: str = "learned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature-based linear-chain CRF baseline.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/statistical/crf"))
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--max-iterations", type=int, default=70)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-final-fit", action="store_true")
    parser.add_argument(
        "--suffix-policy",
        choices=("learned", "uzbek_whitelist", "none"),
        default="learned",
    )
    parser.add_argument(
        "--fixed-confidence-threshold",
        type=float,
        help="Skip internal fit and use a threshold selected by an earlier internal run.",
    )
    return parser.parse_args()


def safe_chunk_slices(tokens: Sequence[Token], tags: Sequence[str], max_tokens: int) -> list[tuple[int, int]]:
    """Moves training cuts to the end of a gold entity when necessary."""

    result = chunk_slices(tokens, max_tokens)
    adjusted: list[tuple[int, int]] = []
    start = 0
    for _, proposed_end in result:
        if proposed_end <= start:
            continue
        end = proposed_end
        while end < len(tags) and tags[end].startswith(("I-", "L-")):
            end += 1
        adjusted.append((start, end))
        start = end
    if start < len(tokens):
        adjusted.append((start, len(tokens)))
    return adjusted


def entity_confidence(
    entity: JsonObject,
    tokens: Sequence[Token],
    predicted_tag_confidences: Sequence[float],
) -> float:
    indices = [
        index
        for index, token in enumerate(tokens)
        if token.start < entity["end"] and entity["start"] < token.end
    ]
    if not indices:
        return 0.0
    return min(predicted_tag_confidences[index] for index in indices)


def predict_records_scored(
    records: Sequence[JsonObject],
    model: pycrfsuite.Tagger,
    extractor: FeatureExtractor,
    *,
    max_tokens: int,
) -> list[JsonObject]:
    predictions: list[JsonObject] = []
    for record in records:
        tokens = tokenize(record["text"])
        document_entities: list[JsonObject] = []
        for start, end in chunk_slices(tokens, max_tokens):
            chunk = tokens[start:end]
            features = extractor.transform(chunk)
            model.set(features)
            tags = model.tag()
            tag_confidences = [
                model.marginal(tag, index) for index, tag in enumerate(tags)
            ]
            for entity in bilou_to_entities(chunk, tags):
                entity["_confidence"] = entity_confidence(entity, chunk, tag_confidences)
                document_entities.append(entity)
        predictions.append(
            {
                "hash": record["hash"],
                "entities": sorted(
                    document_entities,
                    key=lambda entity: (entity["start"], entity["end"], entity["label"]),
                ),
            }
        )
    return predictions


def filter_scored_predictions(
    predictions: Sequence[JsonObject], min_confidence: float
) -> list[JsonObject]:
    return [
        {
            "hash": record["hash"],
            "entities": [
                {"label": entity["label"], "start": entity["start"], "end": entity["end"]}
                for entity in record["entities"]
                if entity["_confidence"] >= min_confidence
            ],
        }
        for record in predictions
    ]


def predict_records(
    records: Sequence[JsonObject],
    model: pycrfsuite.Tagger,
    extractor: FeatureExtractor,
    *,
    max_tokens: int,
    min_confidence: float,
) -> list[JsonObject]:
    return filter_scored_predictions(
        predict_records_scored(records, model, extractor, max_tokens=max_tokens),
        min_confidence,
    )


def threshold_trials(
    gold: Sequence[JsonObject],
    model: pycrfsuite.Tagger,
    extractor: FeatureExtractor,
    max_tokens: int,
) -> tuple[float, list[JsonObject]]:
    trials: list[JsonObject] = []
    scored_predictions = predict_records_scored(
        gold, model, extractor, max_tokens=max_tokens
    )
    for threshold in (0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95):
        predictions = filter_scored_predictions(scored_predictions, threshold)
        metrics = evaluate(gold, predictions)
        precision = metrics["micro"]["precision"]
        recall = metrics["micro"]["recall"]
        beta2 = 0.75**2
        f075 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0
        trials.append({"threshold": threshold, "f0.75": f075, "metrics": metrics})
    trials.sort(key=lambda item: (item["f0.75"], item["metrics"]["micro"]["f1"]), reverse=True)
    return float(trials[0]["threshold"]), trials


def fit_model(
    records: Sequence[JsonObject], config: CrfConfig, model_path: Path
) -> tuple[FeatureExtractor, dict[str, int], float]:
    extractor = FeatureExtractor(
        build_lexicon(records),
        gazetteer_min_count=config.gazetteer_min_count,
        gazetteer_min_purity=config.gazetteer_min_purity,
        suffix_min_support=config.suffix_min_support,
        suffix_policy=config.suffix_policy,
    )
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.select("lbfgs", "crf1d")
    trainer.set_params(
        {
            "c1": config.c1,
            "c2": config.c2,
            "max_iterations": config.max_iterations,
            "feature.possible_transitions": True,
        }
    )
    stats = {"documents": len(records), "tokens": 0, "sequences": 0, "all_o_sequences": 0}
    started = time.perf_counter()
    for record_index, record in enumerate(records, start=1):
        tokens = tokenize(record["text"])
        if not tokens:
            continue
        tags = entities_to_bilou(tokens, record["entities"])
        document_features = extractor.transform(tokens)
        stats["tokens"] += len(tokens)
        for start, end in safe_chunk_slices(tokens, tags, config.max_tokens):
            chunk_tags = tags[start:end]
            trainer.append(document_features[start:end], chunk_tags)
            stats["sequences"] += 1
            stats["all_o_sequences"] += all(tag == "O" for tag in chunk_tags)
        if record_index % 2_000 == 0:
            print(f"  appended {record_index}/{len(records)} documents", flush=True)
    trainer.train(str(model_path))
    elapsed = time.perf_counter() - started
    return extractor, stats, elapsed


def load_tagger(path: Path) -> pycrfsuite.Tagger:
    tagger = pycrfsuite.Tagger()
    tagger.open(str(path))
    return tagger


def run(args: argparse.Namespace) -> None:
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)
    if args.limit is not None:
        train = train[: args.limit]
        dev = dev[: max(50, args.limit // 5)]
    fitting, holdout = stable_train_holdout(train, holdout_fraction=args.holdout_fraction)
    config = CrfConfig(
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        suffix_policy=args.suffix_policy,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Internal split: fitting={len(fitting)} holdout={len(holdout)}", flush=True)
    internal_fit: JsonObject | None = None
    trials: list[JsonObject] = []
    if args.fixed_confidence_threshold is None:
        print("Fitting internal CRF...", flush=True)
        internal_path = output_dir / "internal.crfsuite"
        internal_extractor, internal_stats, internal_seconds = fit_model(
            fitting, config, internal_path
        )
        internal_model = load_tagger(internal_path)
        threshold, trials = threshold_trials(
            holdout, internal_model, internal_extractor, config.max_tokens
        )
        internal_fit = {"seconds": internal_seconds, "stats": internal_stats}
        print(f"Selected confidence threshold: {threshold:.2f}", flush=True)
        print_metrics("Internal holdout", trials[0]["metrics"])
    else:
        threshold = args.fixed_confidence_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("fixed-confidence-threshold must be in [0, 1]")
        print(f"Using fixed confidence threshold: {threshold:.2f}", flush=True)

    metadata: JsonObject = {
        "config": asdict(config),
        "selected_confidence_threshold": threshold,
        "split": {"fitting": len(fitting), "holdout": len(holdout)},
        "alignment": {
            "train": boundary_alignment_stats(train),
            "dev": boundary_alignment_stats(dev),
        },
        "internal_fit": internal_fit,
        "internal_threshold_trials": trials,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "sklearn-crfsuite": version("sklearn-crfsuite"),
            "python-crfsuite": version("python-crfsuite"),
        },
    }

    if not args.skip_final_fit:
        print("Fitting final CRF on all train records...", flush=True)
        final_path = output_dir / "model.crfsuite"
        final_extractor, final_stats, final_seconds = fit_model(train, config, final_path)
        final_model = load_tagger(final_path)
        dev_predictions = predict_records(
            dev,
            final_model,
            final_extractor,
            max_tokens=config.max_tokens,
            min_confidence=threshold,
        )
        dev_metrics = evaluate(dev, dev_predictions)
        print_metrics("Dev", dev_metrics)
        write_jsonl(output_dir / "dev_predictions.jsonl", dev_predictions)
        (output_dir / "dev_metrics.json").write_text(
            json.dumps(dev_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        joblib.dump(
            {
                "extractor": final_extractor,
                "max_tokens": config.max_tokens,
                "min_confidence": threshold,
            },
            output_dir / "bundle.joblib",
            compress=3,
        )
        metadata["final_fit"] = {"seconds": final_seconds, "stats": final_stats}
        metadata["dev_metrics"] = dev_metrics

    (output_dir / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
