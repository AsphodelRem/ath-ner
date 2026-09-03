from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

LABELS = ("ORG", "NAME", "GEO")
APOSTROPHES = frozenset("'’ʻʼ‘`´")
KNOWN_ATTACHED_SUFFIXES = frozenset(
    {
        "da",
        "dagi",
        "dan",
        "ga",
        "gacha",
        "ka",
        "qa",
        "lar",
        "lari",
        "laridan",
        "lariga",
        "larini",
        "larining",
        "larning",
        "lik",
        "likda",
        "ni",
        "ning",
        "'da",
        "'dagi",
        "'dan",
        "'ga",
        "'ni",
        "'ning",
        "да",
        "даги",
        "дан",
        "га",
        "гача",
        "ка",
        "қа",
        "лар",
        "лари",
        "ларидан",
        "ларига",
        "ларини",
        "ларининг",
        "ларнинг",
        "лик",
        "ликда",
        "ни",
        "нинг",
        "'да",
        "'даги",
        "'дан",
        "'га",
        "'ни",
        "'нинг",
    }
)
# Apostrophes are part of Uzbek words. Dots and dashes stay separate: otherwise text such
# as ``gap.Xarakat`` becomes one token, while ``Kun.uz`` is still representable as three.
WORD_RE = re.compile(r"[^\W_]+(?:['’ʻʼ‘`´][^\W_]+)*|_+|[^\w\s]", re.UNICODE)
SENTENCE_END = frozenset(".!?。！？")

JsonObject = dict[str, Any]
EntityKey = tuple[str, int, int]


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    norm: str


@dataclass(frozen=True)
class LexiconEntry:
    label: str
    count: int
    total: int

    @property
    def purity(self) -> float:
        return self.count / self.total


def normalize(text: str) -> str:
    """Normalizes comparison keys without touching source text or offsets."""

    return "".join("'" if char in APOSTROPHES else char for char in text.casefold())


def read_jsonl(path: Path, *, require_entities: bool = True) -> list[JsonObject]:
    records: list[JsonObject] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            record_hash = raw.get("hash")
            text = raw.get("text")
            if not isinstance(record_hash, str) or not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: invalid hash or text")
            if record_hash in seen:
                raise ValueError(f"{path}:{line_number}: duplicate hash {record_hash}")
            seen.add(record_hash)
            record: JsonObject = {"hash": record_hash, "text": text}
            if require_entities:
                entities = raw.get("entities")
                if not isinstance(entities, list):
                    raise ValueError(f"{path}:{line_number}: entities must be a list")
                record["entities"] = sorted(
                    (
                        {
                            "label": entity["label"],
                            "start": int(entity["start"]),
                            "end": int(entity["end"]),
                        }
                        for entity in entities
                    ),
                    key=lambda entity: (entity["start"], entity["end"]),
                )
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def stable_train_holdout(
    records: Sequence[JsonObject], *, holdout_fraction: float
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Makes a deterministic split that is stable across Python versions."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    cutoff = round(holdout_fraction * 10_000)
    fitting: list[JsonObject] = []
    holdout: list[JsonObject] = []
    for record in records:
        bucket = int.from_bytes(
            hashlib.blake2b(record["hash"].encode(), digest_size=8).digest(), "big"
        ) % 10_000
        (holdout if bucket < cutoff else fitting).append(record)
    return fitting, holdout


def tokenize(text: str) -> list[Token]:
    return [
        Token(match.group(), match.start(), match.end(), normalize(match.group()))
        for match in WORD_RE.finditer(text)
    ]


def entity_token_keys(text: str, entities: Sequence[JsonObject]) -> Iterator[tuple[tuple[str, ...], str]]:
    tokens = tokenize(text)
    for entity in entities:
        covered = [
            token
            for token in tokens
            if token.start >= entity["start"] and token.end <= entity["end"]
        ]
        if covered and covered[0].start == entity["start"] and covered[-1].end == entity["end"]:
            yield tuple(token.norm for token in covered), entity["label"]


def build_lexicon(records: Sequence[JsonObject]) -> dict[tuple[str, ...], LexiconEntry]:
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for record in records:
        for key, label in entity_token_keys(record["text"], record["entities"]):
            counts[key][label] += 1

    result: dict[tuple[str, ...], LexiconEntry] = {}
    for key, labels in counts.items():
        label, count = labels.most_common(1)[0]
        result[key] = LexiconEntry(label=label, count=count, total=sum(labels.values()))
    return result


def learn_suffixes(
    lexicon: dict[tuple[str, ...], LexiconEntry],
    *,
    max_suffix_length: int = 12,
) -> dict[str, Counter[str]]:
    """Learns attached endings when both base and suffixed forms occur in train."""

    suffixes = {label: Counter() for label in LABELS}
    for key, suffixed_entry in lexicon.items():
        last = key[-1]
        for suffix_length in range(1, min(max_suffix_length, len(last) - 1) + 1):
            base_last = last[:-suffix_length]
            if len(base_last) < 2:
                continue
            base_entry = lexicon.get((*key[:-1], base_last))
            if base_entry is None or base_entry.label != suffixed_entry.label:
                continue
            suffix = last[-suffix_length:]
            support = min(base_entry.count, suffixed_entry.count)
            suffixes[suffixed_entry.label][suffix] += support
    return suffixes


def boundary_alignment_stats(records: Sequence[JsonObject]) -> dict[str, int | float]:
    total = 0
    representable = 0
    for record in records:
        boundaries = {0, len(record["text"])}
        for token in tokenize(record["text"]):
            boundaries.add(token.start)
            boundaries.add(token.end)
        for entity in record["entities"]:
            total += 1
            representable += entity["start"] in boundaries and entity["end"] in boundaries
    return {
        "total": total,
        "representable": representable,
        "not_representable": total - representable,
        "share": representable / total if total else 0.0,
    }


def entities_to_bilou(tokens: Sequence[Token], entities: Sequence[JsonObject]) -> list[str]:
    tags = ["O"] * len(tokens)
    for entity in entities:
        indices = [
            index
            for index, token in enumerate(tokens)
            if token.start < entity["end"] and entity["start"] < token.end
        ]
        if not indices:
            continue
        label = entity["label"]
        if len(indices) == 1:
            tags[indices[0]] = f"U-{label}"
            continue
        tags[indices[0]] = f"B-{label}"
        for index in indices[1:-1]:
            tags[index] = f"I-{label}"
        tags[indices[-1]] = f"L-{label}"
    return tags


def bilou_to_entities(tokens: Sequence[Token], tags: Sequence[str]) -> list[JsonObject]:
    """Decodes BILOU defensively; invalid transitions become local spans."""

    entities: list[JsonObject] = []
    active_label: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def close(end: int) -> None:
        nonlocal active_label, active_start, active_end
        if active_label is not None and active_start is not None:
            entities.append({"label": active_label, "start": active_start, "end": end})
        active_label = None
        active_start = None
        active_end = None

    for token, tag in zip(tokens, tags, strict=True):
        if tag == "O" or "-" not in tag:
            if active_label is not None:
                close(active_end if active_end is not None else token.start)
            continue
        prefix, label = tag.split("-", 1)
        if label not in LABELS:
            if active_label is not None:
                close(active_end if active_end is not None else token.start)
            continue
        if prefix == "U":
            if active_label is not None:
                close(active_end if active_end is not None else token.start)
            entities.append({"label": label, "start": token.start, "end": token.end})
        elif prefix == "B":
            if active_label is not None:
                close(active_end if active_end is not None else token.start)
            active_label = label
            active_start = token.start
            active_end = token.end
        elif prefix == "I" and active_label == label:
            active_end = token.end
        elif prefix == "L" and active_label == label:
            close(token.end)
        else:
            if active_label is not None:
                close(active_end if active_end is not None else token.start)
            active_label = label
            active_start = token.start
            active_end = token.end
            if prefix == "L":
                close(token.end)
    if active_label is not None and tokens:
        close(active_end if active_end is not None else tokens[-1].end)
    return entities


def chunk_slices(tokens: Sequence[Token], max_tokens: int) -> list[tuple[int, int]]:
    """Chunks long documents, preferring paragraph and sentence boundaries."""

    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    slices: list[tuple[int, int]] = []
    start = 0
    while start < len(tokens):
        hard_end = min(start + max_tokens, len(tokens))
        end = hard_end
        if hard_end < len(tokens):
            search_start = max(start + max_tokens // 2, hard_end - 80)
            for candidate in range(hard_end - 1, search_start - 1, -1):
                gap = tokens[candidate].start - tokens[candidate - 1].end if candidate > start else 0
                if gap > 1 or tokens[candidate - 1].text in SENTENCE_END:
                    end = candidate
                    break
        slices.append((start, end))
        start = end
    return slices


def script_name(text: str) -> str:
    latin = False
    cyrillic = False
    for character in text:
        name = unicodedata.name(character, "")
        latin |= "LATIN" in name
        cyrillic |= "CYRILLIC" in name
    if latin and cyrillic:
        return "mixed"
    if latin:
        return "latin"
    if cyrillic:
        return "cyrillic"
    return "other"


def word_shape(text: str) -> str:
    raw = []
    for character in text:
        if character.isupper():
            value = "X"
        elif character.islower():
            value = "x"
        elif character.isdigit():
            value = "d"
        else:
            value = character
        if not raw or raw[-1] != value:
            raw.append(value)
    return "".join(raw)[:20]


def metric_values(tp: int, fp: int, fn: int) -> JsonObject:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(gold: Sequence[JsonObject], predictions: Sequence[JsonObject]) -> JsonObject:
    if len(gold) != len(predictions):
        raise ValueError("gold and predictions have different lengths")
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for gold_record, prediction in zip(gold, predictions, strict=True):
        if gold_record["hash"] != prediction["hash"]:
            raise ValueError("gold and prediction hashes are misaligned")
        gold_set = {
            (entity["label"], entity["start"], entity["end"])
            for entity in gold_record["entities"]
        }
        pred_set = {
            (entity["label"], entity["start"], entity["end"])
            for entity in prediction["entities"]
        }
        for label in LABELS:
            gold_label = {item for item in gold_set if item[0] == label}
            pred_label = {item for item in pred_set if item[0] == label}
            counts[label]["tp"] += len(gold_label & pred_label)
            counts[label]["fp"] += len(pred_label - gold_label)
            counts[label]["fn"] += len(gold_label - pred_label)
    by_label = {label: metric_values(**counts[label]) for label in LABELS}
    micro = metric_values(
        tp=sum(value["tp"] for value in counts.values()),
        fp=sum(value["fp"] for value in counts.values()),
        fn=sum(value["fn"] for value in counts.values()),
    )
    return {"by_label": by_label, "micro": micro}


def print_metrics(title: str, metrics: JsonObject) -> None:
    micro = metrics["micro"]
    print(
        f"{title}: P={micro['precision']:.4f} R={micro['recall']:.4f} "
        f"F1={micro['f1']:.4f} (TP={micro['tp']} FP={micro['fp']} FN={micro['fn']})"
    )
