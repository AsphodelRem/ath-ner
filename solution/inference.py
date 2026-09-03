"""Инференс со скользящим окном и постобработкой границ."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from baseline.common import tokenize_windows
from solution.postprocess import postprocess_entities
from solution.tagging import decode_tokens
from solution.viterbi import token_category, viterbi_decode

JsonObject = dict[str, Any]


@torch.no_grad()
def collect_token_scores(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    records: list[JsonObject],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    progress: bool = False,
) -> list[dict[tuple[int, int], tuple[torch.Tensor, int]]]:
    """Считает вероятности классов по токенам, усредняя перекрывающиеся окна."""

    windows: list[tuple[int, JsonObject, list[tuple[int, int]]]] = []
    for index, record in enumerate(records):
        for feature, offsets in tokenize_windows(
            tokenizer, record["text"], max_length=max_length, stride=stride
        ):
            windows.append((index, feature, offsets))

    scores: list[dict[tuple[int, int], tuple[torch.Tensor, int]]] = [{} for _ in records]
    model.eval()
    iterator = range(0, len(windows), batch_size)
    if progress:
        iterator = tqdm(iterator, desc="Predict", unit="batch", leave=False)
    for start in iterator:
        chunk = windows[start : start + batch_size]
        batch = tokenizer.pad([feature for _, feature, _ in chunk], padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        probabilities = torch.softmax(model(**batch).logits.float(), dim=-1).cpu()
        for row, (record_index, _, offsets) in enumerate(chunk):
            record_scores = scores[record_index]
            for position, (span_start, span_end) in enumerate(offsets):
                if span_start == span_end:
                    continue
                key = (span_start, span_end)
                value = probabilities[row, position]
                if key in record_scores:
                    previous, count = record_scores[key]
                    record_scores[key] = (previous + value, count + 1)
                else:
                    record_scores[key] = (value.clone(), 1)
    return scores


def decode_scores(
    records: list[JsonObject],
    scores: list[dict[tuple[int, int], tuple[torch.Tensor, int]]],
    *,
    id2label: dict[int, str],
    scheme: str = "bio",
    postprocess: bool = True,
    log_transitions: Any = None,
) -> list[JsonObject]:
    """Переводит усреднённые вероятности в spans: argmax либо Витерби."""

    predictions: list[JsonObject] = []
    for record, record_scores in zip(records, scores, strict=True):
        ordered = sorted(record_scores.items())
        if not ordered:
            predictions.append({"hash": record["hash"], "entities": []})
            continue
        spans = [span for span, _ in ordered]
        matrix = torch.stack([total / count for _, (total, count) in ordered]).numpy()
        if log_transitions is None:
            label_ids = matrix.argmax(axis=1).tolist()
        else:
            categories = np.array(
                [token_category(record["text"], start, end) for start, end in spans],
                dtype=np.int64,
            )
            label_ids = viterbi_decode(matrix, categories, log_transitions)
        tagged = [
            (start, end, id2label[int(label)])
            for (start, end), label in zip(spans, label_ids, strict=True)
        ]
        entities = decode_tokens(tagged, scheme)
        if postprocess:
            entities = postprocess_entities(record["text"], entities)
        predictions.append({"hash": record["hash"], "entities": entities})
    return predictions


@torch.no_grad()
def predict_records(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    records: list[JsonObject],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    id2label: dict[int, str],
    scheme: str = "bio",
    postprocess: bool = True,
    progress: bool = False,
    log_transitions: Any = None,
) -> list[JsonObject]:
    """Предсказывает сущности, усредняя вероятности по перекрывающимся окнам."""

    scores = collect_token_scores(
        model, tokenizer, records,
        max_length=max_length, stride=stride, batch_size=batch_size,
        device=device, progress=progress,
    )
    return decode_scores(
        records, scores, id2label=id2label, scheme=scheme,
        postprocess=postprocess, log_transitions=log_transitions,
    )
