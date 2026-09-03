from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification

from .common import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_STRIDE,
    BIO_TAGS,
    BILOU_TAGS,
    JsonObject,
    ModelFeature,
    Offsets,
    decode_tagged_tokens,
    load_fast_tokenizer,
    read_records,
    resolve_device,
    tokenize_windows,
    trim_entity_whitespace,
    validate_window,
)
from .transition_confidence import (
    TransitionPriors,
    add_span_confidence,
    legal_end,
    legal_start,
    legal_transition,
    load_transition_priors,
    tag_parts,
    viterbi_decode,
)

Window = tuple[int, ModelFeature, Offsets]


def parse_args() -> argparse.Namespace:
    """Разбирает пути модели, входа и предсказаний."""

    parser = argparse.ArgumentParser(description="Run the minimal NER baseline.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/baseline/model"))
    parser.add_argument("--input", type=Path, default=Path("dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("predictions.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"))
    parser.add_argument(
        "--window-weighting",
        choices=("uniform", "center"),
        help="How to combine token probabilities from overlapping windows.",
    )
    parser.add_argument(
        "--decoder",
        choices=("argmax", "viterbi"),
        help="Decode token labels independently or with legal BIO/BILOU transitions.",
    )
    parser.add_argument(
        "--secondary-output",
        type=Path,
        help="Optional second JSONL decoded from the same encoder forward pass.",
    )
    parser.add_argument(
        "--secondary-decoder",
        choices=("argmax", "viterbi"),
        help="Decoder used for --secondary-output.",
    )
    parser.add_argument(
        "--transition-priors",
        type=Path,
        help="Optional learned start/transition/end potentials for Viterbi.",
    )
    parser.add_argument(
        "--transition-scale",
        type=float,
        default=0.05,
        help="Multiplier for learned transition potentials (default: 0.05).",
    )
    parser.add_argument(
        "--confidence-output",
        type=Path,
        help="Optional JSONL sidecar with per-span token confidence statistics.",
    )
    return parser.parse_args()


def _read_baseline_config(model_dir: Path) -> JsonObject:
    """Читает параметры окон, сохранённые скриптом обучения."""

    path = model_dir / "baseline_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _validate_args(args: argparse.Namespace) -> None:
    """Проверяет параметры batch и ограничения отладочной выборки."""

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("max-records must be positive")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("max-length must be positive")
    if not torch.isfinite(torch.tensor(args.transition_scale)) or args.transition_scale < 0:
        raise ValueError("transition-scale must be a non-negative finite number")
    if (args.secondary_output is None) != (args.secondary_decoder is None):
        raise ValueError(
            "--secondary-output and --secondary-decoder must be provided together"
        )
    if args.confidence_output is not None and args.secondary_output is not None:
        raise ValueError(
            "--confidence-output currently describes only the primary decoder; "
            "omit --secondary-output when exporting confidence"
        )


def _model_labels(model: torch.nn.Module) -> dict[int, str]:
    """Извлекает и проверяет BIO-метки из model config."""

    labels = {int(index): str(label) for index, label in model.config.id2label.items()}
    values = set(labels.values())
    expected = next(
        (tags for tags in (BIO_TAGS, BILOU_TAGS) if values == set(tags)),
        None,
    )
    if expected is None or set(labels) != set(range(len(expected))):
        raise ValueError("model labels must be a supported BIO or BILOU tag set")
    return labels


def _tag_parts(tag: str) -> tuple[str, str | None]:
    return tag_parts(tag)


def _legal_start(tag: str, bilou: bool) -> bool:
    return legal_start(tag, bilou)


def _legal_end(tag: str, bilou: bool) -> bool:
    return legal_end(tag, bilou)


def _legal_transition(left: str, right: str, bilou: bool) -> bool:
    return legal_transition(left, right, bilou)


def _viterbi_decode(
    probabilities: torch.Tensor,
    id2label: dict[int, str],
    priors: TransitionPriors | None = None,
    prior_scale: float = 1.0,
) -> list[int]:
    """Ищет наиболее вероятную последовательность среди легальных BIO/BILOU путей."""

    return viterbi_decode(
        probabilities, id2label, priors=priors, prior_scale=prior_scale
    )


def _build_windows(
    records: list[JsonObject],
    tokenizer: Any,
    *,
    max_length: int,
    stride: int,
) -> list[Window]:
    """Токенизирует все документы и связывает окна с индексами записей."""

    windows: list[Window] = []
    for record_index, record in enumerate(tqdm(records, desc="Tokenize", unit="doc")):
        for feature, offsets in tokenize_windows(
            tokenizer,
            record["text"],
            max_length=max_length,
            stride=stride,
        ):
            windows.append((record_index, feature, offsets))
    return windows


@torch.inference_mode()
def _predict_token_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    windows: list[Window],
    record_count: int,
    *,
    batch_size: int,
    device: torch.device,
    window_weighting: str,
) -> list[dict[tuple[int, int], tuple[torch.Tensor, float]]]:
    """Усредняет вероятности одинаковых токенов из перекрывающихся окон."""

    aggregated: list[dict[tuple[int, int], tuple[torch.Tensor, float]]] = [
        {} for _ in range(record_count)
    ]
    model.eval()
    for batch_start in tqdm(
        range(0, len(windows), batch_size),
        desc="Predict",
        unit="batch",
    ):
        batch_windows = windows[batch_start : batch_start + batch_size]
        batch = tokenizer.pad(
            [feature for _, feature, _ in batch_windows],
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities = torch.softmax(model(**batch).logits.float(), dim=-1).cpu()

        for row_index, (record_index, _, offsets) in enumerate(batch_windows):
            record_scores = aggregated[record_index]
            content_indices = [
                index for index, (start, end) in enumerate(offsets) if start != end
            ]
            content_position = {index: position for position, index in enumerate(content_indices)}
            for token_index, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                key = (start, end)
                score = probabilities[row_index, token_index]
                if window_weighting == "center" and len(content_indices) > 1:
                    position = content_position[token_index]
                    center_proximity = 1.0 - abs(
                        2.0 * position / (len(content_indices) - 1) - 1.0
                    )
                    weight = 0.25 + 0.75 * center_proximity
                else:
                    weight = 1.0
                if key in record_scores:
                    previous, weight_sum = record_scores[key]
                    record_scores[key] = (previous + score * weight, weight_sum + weight)
                else:
                    record_scores[key] = (score.clone() * weight, weight)
    return aggregated


def _decode_records(
    records: list[JsonObject],
    scores: list[dict[tuple[int, int], tuple[torch.Tensor, float]]],
    id2label: dict[int, str],
    decoder: str,
    *,
    transition_priors: TransitionPriors | None = None,
    transition_scale: float = 1.0,
    confidence_records: list[JsonObject] | None = None,
) -> list[JsonObject]:
    """Преобразует усреднённые token scores в JSONL-предсказания spans."""

    predictions: list[JsonObject] = []
    for record, record_scores in zip(records, scores, strict=True):
        ordered_scores = [
            ((start, end), score_sum / weight_sum)
            for (start, end), (score_sum, weight_sum) in sorted(record_scores.items())
        ]
        if decoder == "viterbi":
            probability_matrix = (
                torch.stack([score for _, score in ordered_scores])
                if ordered_scores
                else torch.empty((0, len(id2label)))
            )
            label_ids = _viterbi_decode(
                probability_matrix,
                id2label,
                transition_priors,
                transition_scale,
            )
        else:
            label_ids = [int(score.argmax().item()) for _, score in ordered_scores]
        tagged_tokens = []
        for ((start, end), _), label_id in zip(ordered_scores, label_ids, strict=True):
            tagged_tokens.append((start, end, id2label[label_id]))
        tag_scheme = "bilou" if set(id2label.values()) == set(BILOU_TAGS) else "bio"
        entities = decode_tagged_tokens(tagged_tokens, tag_scheme=tag_scheme)
        trimmed_entities = trim_entity_whitespace(record["text"], entities)
        predictions.append(
            {
                "hash": record["hash"],
                "entities": trimmed_entities,
            }
        )
        if confidence_records is not None:
            confidence_records.append(
                {
                    "hash": record["hash"],
                    "entities": add_span_confidence(
                        trimmed_entities,
                        [offset for offset, _ in ordered_scores],
                        [score for _, score in ordered_scores],
                        label_ids,
                        id2label,
                    ),
                }
            )
    return predictions


def _write_jsonl(path: Path, records: list[JsonObject]) -> None:
    """Записывает предсказания по одному JSON-объекту на строку."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def run(args: argparse.Namespace) -> Path:
    """Загружает checkpoint и строит exact-span предсказания для JSONL."""

    _validate_args(args)
    model_dir = args.model_dir.expanduser().resolve()
    config = _read_baseline_config(model_dir)
    max_length = (
        args.max_length
        if args.max_length is not None
        else int(config.get("max_length", DEFAULT_MAX_LENGTH))
    )
    stride = args.stride if args.stride is not None else int(config.get("stride", DEFAULT_STRIDE))
    device = resolve_device(args.device)
    tokenizer = load_fast_tokenizer(str(model_dir))
    validate_window(tokenizer, max_length, stride)
    attn_implementation = args.attn_implementation or config.get(
        "inference_attn_implementation"
    )
    window_weighting = args.window_weighting or str(
        config.get("window_weighting", "uniform")
    )
    decoder = args.decoder or str(config.get("decoder", "argmax"))
    transition_priors = (
        load_transition_priors(args.transition_priors.expanduser().resolve())
        if args.transition_priors is not None
        else None
    )
    if transition_priors is not None and decoder != "viterbi":
        raise ValueError("transition priors require --decoder viterbi")
    if transition_priors is not None and args.secondary_decoder == "argmax":
        raise ValueError(
            "transition priors cannot be shared with an argmax secondary decoder"
        )
    model_kwargs: dict[str, Any] = {}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        **model_kwargs,
    ).to(device)
    id2label = _model_labels(model)

    records = read_records(
        args.input.expanduser().resolve(),
        require_entities=False,
        limit=args.max_records,
    )
    windows = _build_windows(
        records,
        tokenizer,
        max_length=max_length,
        stride=stride,
    )
    scores = _predict_token_scores(
        model,
        tokenizer,
        windows,
        len(records),
        batch_size=args.batch_size,
        device=device,
        window_weighting=window_weighting,
    )
    confidence_records: list[JsonObject] | None = (
        [] if args.confidence_output is not None else None
    )
    predictions = _decode_records(
        records,
        scores,
        id2label,
        decoder,
        transition_priors=transition_priors,
        transition_scale=args.transition_scale,
        confidence_records=confidence_records,
    )
    output_path = args.output.expanduser().resolve()
    _write_jsonl(output_path, predictions)
    secondary_path: Path | None = None
    if args.secondary_output is not None:
        assert args.secondary_decoder is not None
        secondary_predictions = _decode_records(
            records,
            scores,
            id2label,
            args.secondary_decoder,
        )
        secondary_path = args.secondary_output.expanduser().resolve()
        _write_jsonl(secondary_path, secondary_predictions)
    confidence_path: Path | None = None
    if args.confidence_output is not None:
        confidence_path = args.confidence_output.expanduser().resolve()
        assert confidence_records is not None
        _write_jsonl(confidence_path, confidence_records)
    print(f"Device: {device}")
    print(f"Records: {len(records)}, windows: {len(windows)}")
    print(f"Window weighting: {window_weighting}")
    print(f"Decoder: {decoder}")
    if transition_priors is not None:
        print(f"Transition priors: {args.transition_priors} (scale={args.transition_scale})")
    print(f"Predictions: {output_path}")
    if secondary_path is not None:
        print(f"Secondary decoder: {args.secondary_decoder}")
        print(f"Secondary predictions: {secondary_path}")
    if confidence_path is not None:
        print(f"Span confidence: {confidence_path}")
    return output_path


def main() -> int:
    """Запускает CLI инференса с компактным сообщением об ошибке."""

    try:
        run(parse_args())
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
