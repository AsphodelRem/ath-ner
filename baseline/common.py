"""Common validation and sliding-window tokenization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ENTITY_LABELS = ("ORG", "NAME", "GEO")
JsonObject = dict[str, Any]
ModelFeature = dict[str, list[int]]
Offsets = list[tuple[int, int]]


def read_records(
    path: Path,
    *,
    require_entities: bool,
    limit: int | None = None,
) -> list[JsonObject]:
    """Read and validate JSONL records used by the training commands."""

    records: list[JsonObject] = []
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty line")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            record = _validate_record(raw, path, line_number, require_entities)
            if record["hash"] in seen_hashes:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record['hash']}")
            seen_hashes.add(record["hash"])
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _validate_record(
    raw: Any,
    path: Path,
    line_number: int,
    require_entities: bool,
) -> JsonObject:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    record_hash = raw.get("hash")
    text = raw.get("text")
    if not isinstance(record_hash, str) or not record_hash:
        raise ValueError(f"{path}:{line_number}: hash must be a non-empty string")
    if not isinstance(text, str):
        raise ValueError(f"{path}:{line_number}: text must be a string")
    result: JsonObject = {"hash": record_hash, "text": text}
    if require_entities:
        result["entities"] = _validate_entities(
            raw.get("entities"), text, f"{path}:{line_number}"
        )
    return result


def _validate_entities(raw: Any, text: str, source: str) -> list[JsonObject]:
    if not isinstance(raw, list):
        raise ValueError(f"{source}: entities must be an array")
    entities: list[JsonObject] = []
    seen: set[tuple[str, int, int]] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}/entities[{index}]: entity must be an object")
        label, start, end = entity.get("label"), entity.get("start"), entity.get("end")
        if label not in ENTITY_LABELS:
            raise ValueError(f"{source}/entities[{index}]: invalid label {label!r}")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"{source}/entities[{index}]: invalid offsets")
        key = (label, start, end)
        if key in seen:
            raise ValueError(f"{source}/entities[{index}]: duplicate entity")
        seen.add(key)
        entities.append({"label": label, "start": start, "end": end})
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    for left, right in zip(entities, entities[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"{source}: overlapping entities are not supported")
    return entities


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic torch device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable")
    return torch.device(requested)


def load_fast_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer required for exact character offsets."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with offset mappings is required")
    return tokenizer


def validate_window(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
) -> None:
    """Validate sliding-window parameters."""

    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("max-length is too small for tokenizer special tokens")
    if not 0 <= stride < content_length:
        raise ValueError(f"stride must be between 0 and {content_length - 1}")


def tokenize_windows(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[ModelFeature, Offsets]]:
    """Split text into overlapping token windows with character offsets."""

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    input_chunks = encoded["input_ids"]
    offset_chunks = encoded["offset_mapping"]
    if input_chunks and isinstance(input_chunks[0], int):
        input_chunks = [input_chunks]
        offset_chunks = [offset_chunks]

    windows: list[tuple[ModelFeature, Offsets]] = []
    for chunk_index, offsets in enumerate(offset_chunks):
        feature: ModelFeature = {}
        for key in ("input_ids", "attention_mask"):
            if key not in encoded:
                continue
            values = encoded[key]
            feature[key] = (
                values[chunk_index] if values and isinstance(values[0], list) else values
            )
        windows.append((feature, [(int(start), int(end)) for start, end in offsets]))
    return windows
