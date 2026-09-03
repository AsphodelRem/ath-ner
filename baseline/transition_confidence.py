from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .common import BILOU_TAGS, BIO_TAGS, JsonObject

PRIOR_SCHEMA_VERSION = 1


def tag_parts(tag: str) -> tuple[str, str | None]:
    """Splits an O/BIO/BILOU tag into its prefix and entity label."""

    if tag == "O":
        return "O", None
    prefix, separator, label = tag.partition("-")
    if separator != "-" or not label:
        raise ValueError(f"invalid tag {tag!r}")
    return prefix, label


def is_bilou(tags: Sequence[str]) -> bool:
    """Returns the tag scheme after validating the complete label set."""

    values = set(tags)
    if values == set(BILOU_TAGS) and len(tags) == len(BILOU_TAGS):
        return True
    if values == set(BIO_TAGS) and len(tags) == len(BIO_TAGS):
        return False
    raise ValueError("tags must be the complete supported BIO or BILOU tag set")


def legal_start(tag: str, bilou: bool) -> bool:
    prefix, _ = tag_parts(tag)
    return prefix in ({"O", "B", "U"} if bilou else {"O", "B"})


def legal_end(tag: str, bilou: bool) -> bool:
    prefix, _ = tag_parts(tag)
    return prefix in ({"O", "L", "U"} if bilou else {"O", "B", "I"})


def legal_transition(left: str, right: str, bilou: bool) -> bool:
    left_prefix, left_label = tag_parts(left)
    right_prefix, right_label = tag_parts(right)
    if not bilou:
        if right_prefix != "I":
            return True
        return left_prefix in {"B", "I"} and left_label == right_label
    if left_prefix in {"B", "I"}:
        return right_prefix in {"I", "L"} and left_label == right_label
    return right_prefix in {"O", "B", "U"}


@dataclass(frozen=True)
class TransitionPriors:
    """Relative start/transition/end log-potentials learned from gold tags."""

    tags: tuple[str, ...]
    start: tuple[float, ...]
    transitions: tuple[tuple[float, ...], ...]
    end: tuple[float, ...]

    def aligned_tensors(
        self,
        target_tags: Sequence[str],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorders potentials to match a model's id2label order."""

        if set(target_tags) != set(self.tags) or len(target_tags) != len(self.tags):
            raise ValueError("transition priors tags do not match model labels")
        source_index = {tag: index for index, tag in enumerate(self.tags)}
        order = [source_index[tag] for tag in target_tags]
        start = torch.tensor(
            [self.start[index] for index in order], dtype=dtype, device=device
        )
        transitions = torch.tensor(
            [
                [self.transitions[left][right] for right in order]
                for left in order
            ],
            dtype=dtype,
            device=device,
        )
        end = torch.tensor(
            [self.end[index] for index in order], dtype=dtype, device=device
        )
        return start, transitions, end


def _as_finite_float(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source} must be finite")
    return result


def parse_transition_priors(payload: Any) -> TransitionPriors:
    """Validates a transition-prior JSON object."""

    if not isinstance(payload, dict):
        raise ValueError("transition priors must be a JSON object")
    if payload.get("schema_version") != PRIOR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported transition priors schema_version: {payload.get('schema_version')!r}"
        )
    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not all(
        isinstance(tag, str) for tag in raw_tags
    ):
        raise ValueError("transition priors tags must be an array of strings")
    tags = tuple(raw_tags)
    is_bilou(tags)
    size = len(tags)

    raw_start = payload.get("start")
    raw_transitions = payload.get("transitions")
    raw_end = payload.get("end")
    if not isinstance(raw_start, list) or len(raw_start) != size:
        raise ValueError("transition priors start has the wrong shape")
    if not isinstance(raw_end, list) or len(raw_end) != size:
        raise ValueError("transition priors end has the wrong shape")
    if (
        not isinstance(raw_transitions, list)
        or len(raw_transitions) != size
        or any(not isinstance(row, list) or len(row) != size for row in raw_transitions)
    ):
        raise ValueError("transition priors transitions has the wrong shape")

    return TransitionPriors(
        tags=tags,
        start=tuple(
            _as_finite_float(value, f"start[{index}]")
            for index, value in enumerate(raw_start)
        ),
        transitions=tuple(
            tuple(
                _as_finite_float(value, f"transitions[{left}][{right}]")
                for right, value in enumerate(row)
            )
            for left, row in enumerate(raw_transitions)
        ),
        end=tuple(
            _as_finite_float(value, f"end[{index}]")
            for index, value in enumerate(raw_end)
        ),
    )


def load_transition_priors(path: Path) -> TransitionPriors:
    """Loads and validates transition priors from JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    try:
        return parse_transition_priors(payload)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _relative_log_potentials(
    counts: Sequence[int], allowed: Sequence[bool], smoothing: float
) -> list[float]:
    """Builds max-centred log probabilities over legal outcomes."""

    allowed_indices = [index for index, value in enumerate(allowed) if value]
    denominator = sum(counts[index] for index in allowed_indices) + smoothing * len(
        allowed_indices
    )
    values = [0.0] * len(counts)
    legal_values = {
        index: math.log((counts[index] + smoothing) / denominator)
        for index in allowed_indices
    }
    maximum = max(legal_values.values())
    for index, value in legal_values.items():
        values[index] = value - maximum
    return values


def estimate_transition_priors(
    tag_sequences: Iterable[Sequence[str]],
    tags: Sequence[str],
    *,
    smoothing: float = 1.0,
) -> JsonObject:
    """Estimates relative log-potentials from complete document tag sequences."""

    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be a positive finite number")
    tags = tuple(tags)
    bilou = is_bilou(tags)
    tag_to_id = {tag: index for index, tag in enumerate(tags)}
    size = len(tags)
    start_counts = [0] * size
    transition_counts = [[0] * size for _ in range(size)]
    end_counts = [0] * size
    sequence_count = 0
    token_count = 0

    for sequence_number, raw_sequence in enumerate(tag_sequences, start=1):
        sequence = list(raw_sequence)
        if not sequence:
            continue
        try:
            indices = [tag_to_id[tag] for tag in sequence]
        except KeyError as error:
            raise ValueError(
                f"sequence {sequence_number} contains unknown tag {error.args[0]!r}"
            ) from error
        if not legal_start(sequence[0], bilou):
            raise ValueError(f"sequence {sequence_number} has an illegal start")
        if not legal_end(sequence[-1], bilou):
            raise ValueError(f"sequence {sequence_number} has an illegal end")
        for left, right in zip(sequence, sequence[1:], strict=False):
            if not legal_transition(left, right, bilou):
                raise ValueError(
                    f"sequence {sequence_number} has illegal transition {left} -> {right}"
                )
        start_counts[indices[0]] += 1
        end_counts[indices[-1]] += 1
        for left, right in zip(indices, indices[1:], strict=False):
            transition_counts[left][right] += 1
        sequence_count += 1
        token_count += len(sequence)

    if not sequence_count:
        raise ValueError("no non-empty tag sequences")

    start_allowed = [legal_start(tag, bilou) for tag in tags]
    end_allowed = [legal_end(tag, bilou) for tag in tags]
    transition_allowed = [
        [legal_transition(left, right, bilou) for right in tags] for left in tags
    ]
    return {
        "schema_version": PRIOR_SCHEMA_VERSION,
        "kind": "max_centered_log_conditional",
        "tag_scheme": "bilou" if bilou else "bio",
        "tags": list(tags),
        "smoothing": smoothing,
        "sequence_count": sequence_count,
        "token_count": token_count,
        "start": _relative_log_potentials(start_counts, start_allowed, smoothing),
        "transitions": [
            _relative_log_potentials(counts, allowed, smoothing)
            for counts, allowed in zip(
                transition_counts, transition_allowed, strict=True
            )
        ],
        "end": _relative_log_potentials(end_counts, end_allowed, smoothing),
        "counts": {
            "start": start_counts,
            "transitions": transition_counts,
            "end": end_counts,
        },
    }


def viterbi_decode(
    probabilities: torch.Tensor,
    id2label: dict[int, str],
    *,
    priors: TransitionPriors | None = None,
    prior_scale: float = 1.0,
) -> list[int]:
    """Finds the best legal path, optionally adding learned transition priors."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional tensor")
    if probabilities.shape[0] == 0:
        return []
    if not math.isfinite(prior_scale) or prior_scale < 0:
        raise ValueError("prior_scale must be a non-negative finite number")
    if set(id2label) != set(range(len(id2label))):
        raise ValueError("id2label indices must be contiguous from zero")
    tags = [id2label[index] for index in range(len(id2label))]
    bilou = is_bilou(tags)
    if probabilities.shape[1] != len(tags):
        raise ValueError("probabilities label dimension does not match id2label")

    emissions = probabilities.clamp_min(1e-12).log()
    negative = torch.finfo(emissions.dtype).min / 2
    transitions = torch.full(
        (len(tags), len(tags)), negative, dtype=emissions.dtype, device=emissions.device
    )
    for left_index, left in enumerate(tags):
        for right_index, right in enumerate(tags):
            if legal_transition(left, right, bilou):
                transitions[left_index, right_index] = 0.0

    prior_start = prior_transitions = prior_end = None
    if priors is not None and prior_scale:
        prior_start, prior_transitions, prior_end = priors.aligned_tensors(
            tags, dtype=emissions.dtype, device=emissions.device
        )
        transitions = transitions + prior_transitions * prior_scale

    scores = emissions[0].clone()
    if prior_start is not None:
        scores += prior_start * prior_scale
    for tag_index, tag in enumerate(tags):
        if not legal_start(tag, bilou):
            scores[tag_index] = negative
    backpointers: list[torch.Tensor] = []
    for position in range(1, emissions.shape[0]):
        transition_scores = scores[:, None] + transitions
        best_scores, best_left = transition_scores.max(dim=0)
        scores = best_scores + emissions[position]
        backpointers.append(best_left)
    if prior_end is not None:
        scores += prior_end * prior_scale
    for tag_index, tag in enumerate(tags):
        if not legal_end(tag, bilou):
            scores[tag_index] = negative
    last = int(scores.argmax().item())
    path = [last]
    for backpointer in reversed(backpointers):
        last = int(backpointer[last].item())
        path.append(last)
    return list(reversed(path))


def add_span_confidence(
    entities: Sequence[JsonObject],
    token_offsets: Sequence[tuple[int, int]],
    token_probabilities: Sequence[torch.Tensor],
    label_ids: Sequence[int],
    id2label: dict[int, str],
) -> list[JsonObject]:
    """Adds min/mean/geometric token confidence to decoded entities."""

    if not (
        len(token_offsets) == len(token_probabilities) == len(label_ids)
    ):
        raise ValueError("token offsets, probabilities and label ids must align")
    selected: list[tuple[int, int, str | None, float]] = []
    for (start, end), probabilities, label_id in zip(
        token_offsets, token_probabilities, label_ids, strict=True
    ):
        if probabilities.ndim != 1 or label_id < 0 or label_id >= len(probabilities):
            raise ValueError("invalid token probability vector or label id")
        _, label = tag_parts(id2label[label_id])
        selected.append(
            (start, end, label, float(probabilities[label_id].clamp(0.0, 1.0).item()))
        )

    scored: list[JsonObject] = []
    for entity in entities:
        values = [
            confidence
            for start, end, label, confidence in selected
            if label == entity["label"]
            and start < entity["end"]
            and entity["start"] < end
        ]
        if values:
            minimum = min(values)
            mean = sum(values) / len(values)
            geometric_mean = math.exp(
                sum(math.log(max(value, 1e-12)) for value in values) / len(values)
            )
        else:
            minimum = mean = geometric_mean = 0.0
        scored.append(
            {
                "label": entity["label"],
                "start": entity["start"],
                "end": entity["end"],
                "_confidence": minimum,
                "mean_token_confidence": mean,
                "geometric_mean_token_confidence": geometric_mean,
                "token_count": len(values),
            }
        )
    return scored
