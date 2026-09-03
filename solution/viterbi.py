"""
Декодирование меток алгоритмом Витерби с учётом позиции сабтокена в слове.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from baseline.common import tokenize_windows
from solution.tagging import align_labels, tags_for

JsonObject = dict[str, Any]

# Позиция сабтокена внутри слова.
WORD, INIT, MIDDLE, FIN = 0, 1, 2, 3
CATEGORIES = ("word", "init", "middle", "fin")

APOSTROPHES = "'‘’ʻʼʽ`´"
WORD_CHAR = re.compile(r"[^\W_]", re.UNICODE)
SMOOTHING = 0.1
EPS = 1e-12


def _is_word_char(char: str) -> bool:
    """Буква, цифра или внутренний апостроф — часть одного слова."""

    return bool(WORD_CHAR.match(char)) or char in APOSTROPHES


def token_category(text: str, start: int, end: int) -> int:
    """Определяет, чем токен является внутри слова: целым словом, началом и т.д."""

    at_start = start == 0 or not _is_word_char(text[start - 1])
    at_end = end >= len(text) or not _is_word_char(text[end])
    if at_start and at_end:
        return WORD
    if at_start:
        return INIT
    if at_end:
        return FIN
    return MIDDLE


def document_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Упорядоченные уникальные символьные координаты токенов документа.

    Повторяет структуру последовательности, которая складывается на инференсе
    после усреднения перекрывающихся окон.
    """

    seen: dict[tuple[int, int], None] = {}
    for _, offsets in tokenize_windows(tokenizer, text, max_length=max_length, stride=stride):
        for start, end in offsets:
            if start != end:
                seen.setdefault((start, end), None)
    return sorted(seen)


def estimate_transitions(
    records: list[JsonObject],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    stride: int,
    scheme: str,
    progress: bool = True,
) -> dict[str, Any]:
    """Оценивает матрицы переходов подсчётом биграмм тегов по выборке."""

    tags = tags_for(scheme)
    size = len(tags)
    counts = np.zeros((len(CATEGORIES), size, size), dtype=np.float64)
    iterator = tqdm(records, desc="Estimate transitions", unit="doc") if progress else records
    for record in iterator:
        text = record["text"]
        spans = document_tokens(tokenizer, text, max_length=max_length, stride=stride)
        if not spans:
            continue
        labels = align_labels(spans, record["entities"], scheme)
        previous = 0  # виртуальное начало последовательности — состояние O
        for (start, end), label in zip(spans, labels, strict=True):
            if label < 0:
                continue
            counts[token_category(text, start, end), previous, label] += 1.0
            previous = label

    return {
        "scheme": scheme,
        "tags": list(tags),
        "categories": list(CATEGORIES),
        "max_length": max_length,
        "stride": stride,
        "records": len(records),
        "counts": counts.tolist(),
        "log_transitions": build_log_transitions({"counts": counts}, mode="conditional").tolist(),
        "observed_counts": counts.sum(axis=(1, 2)).tolist(),
    }


def build_log_transitions(payload: dict[str, Any], *, mode: str = "conditional") -> np.ndarray:
    """Строит матрицы переходов из счётчиков биграмм.

    conditional — log P(текущий | предыдущий, позиция): смешивает структурные
        ограничения с априорной частотой тегов, а она перекошена в сторону O;
    structural — из того же логарифма вычитается унарный приор log P(текущий |
        позиция). Остаётся только информация о сочетаемости, поэтому редкие
        одиночные сущности (тег U) перестают штрафоваться за саму редкость.
    """

    counts = np.asarray(payload["counts"], dtype=np.float64) + SMOOTHING
    conditional = counts / counts.sum(axis=2, keepdims=True)
    log_conditional = np.log(conditional)
    if mode == "conditional":
        return log_conditional
    if mode != "structural":
        raise ValueError(f"неизвестный режим матриц переходов: {mode}")
    marginal = counts.sum(axis=1) / counts.sum(axis=(1, 2), keepdims=True)[:, 0, :]
    return log_conditional - np.log(marginal)[:, None, :]


def load_transitions(path: Path, *, mode: str = "conditional") -> tuple[np.ndarray, list[str]]:
    """Читает сохранённые матрицы и пересобирает их в выбранном режиме."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "counts" in payload:
        return build_log_transitions(payload, mode=mode), payload["tags"]
    if mode != "conditional":
        raise ValueError("в файле нет счётчиков; пересоберите матрицы")
    return np.asarray(payload["log_transitions"], dtype=np.float64), payload["tags"]


def viterbi_decode(
    probabilities: np.ndarray,
    categories: np.ndarray,
    log_transitions: np.ndarray,
) -> list[int]:
    """Ищет наиболее вероятную последовательность меток.

    probabilities — (T, S) вероятности классов на токен, categories — (T,)
    позиции сабтокенов, log_transitions — (C, S, S) в логарифмах.
    """

    steps, size = probabilities.shape
    if steps == 0:
        return []
    emission = np.log(np.maximum(probabilities, EPS))
    # Начинаем из виртуального состояния O (индекс 0).
    scores = log_transitions[categories[0], 0] + emission[0]
    backpointers = np.zeros((steps, size), dtype=np.int32)
    for step in range(1, steps):
        # candidates[i, j] — прийти в j из i.
        candidates = scores[:, None] + log_transitions[categories[step]]
        backpointers[step] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + emission[step]

    path = [int(scores.argmax())]
    for step in range(steps - 1, 0, -1):
        path.append(int(backpointers[step, path[-1]]))
    path.reverse()
    return path


def parse_args() -> argparse.Namespace:
    """Разбирает параметры оценки матриц переходов."""

    parser = argparse.ArgumentParser(description="Estimate Viterbi transition matrices.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--model-dir", type=Path, required=True, help="каталог с токенизатором")
    parser.add_argument("--scheme", choices=("bio", "bilou"), default="bilou")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    """Считает матрицы переходов и сохраняет их в JSON."""

    args = parse_args()
    from baseline.common import load_fast_tokenizer, read_records

    records = read_records(args.train, require_entities=True, limit=args.limit)
    tokenizer = load_fast_tokenizer(str(args.model_dir))
    payload = estimate_transitions(
        records, tokenizer,
        max_length=args.max_length, stride=args.stride, scheme=args.scheme,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    matrices = np.asarray(payload["log_transitions"])
    tags = payload["tags"]
    print(f"\nТегов: {len(tags)} | категорий: {len(CATEGORIES)} | документов: {len(records)}")
    for index, name in enumerate(CATEGORIES):
        print(f"  {name:<8} токенов: {payload['observed_counts'][index]:>10.0f}")
    print("\nСамые запрещённые переходы внутри слова (категория middle):")
    middle = matrices[MIDDLE]
    flat = [(middle[i, j], tags[i], tags[j]) for i in range(len(tags)) for j in range(len(tags))]
    for score, source, target in sorted(flat)[:6]:
        print(f"  {source:>8} -> {target:<8} log p = {score:7.2f}")
    print(f"\nСохранено: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
