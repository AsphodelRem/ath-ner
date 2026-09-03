"""Схемы разметки токенов: BIO и BILOU."""

from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from baseline.common import ENTITY_LABELS, tokenize_windows

JsonObject = dict[str, Any]
Offsets = list[tuple[int, int]]

SCHEMES = ("bio", "bilou")


def tags_for(scheme: str) -> tuple[str, ...]:
    """Возвращает упорядоченный набор тегов схемы."""

    if scheme == "bio":
        prefixes = ("B", "I")
    elif scheme == "bilou":
        prefixes = ("B", "I", "L", "U")
    else:
        raise ValueError(f"неизвестная схема разметки: {scheme}")
    return ("O", *(f"{prefix}-{label}" for label in ENTITY_LABELS for prefix in prefixes))


def align_labels(offsets: Offsets, entities: list[JsonObject], scheme: str) -> list[int]:
    """Переводит символьные spans в метки токенов одного окна."""

    tag_to_id = {tag: index for index, tag in enumerate(tags_for(scheme))}
    ordered = sorted(entities, key=lambda item: (item["start"], item["end"]))
    labels: list[int] = []
    cursor = 0
    for start, end in offsets:
        if start == end:  # спецтокены и пустые фрагменты
            labels.append(-100)
            continue
        while cursor < len(ordered) and ordered[cursor]["end"] <= start:
            cursor += 1
        if cursor >= len(ordered):
            labels.append(tag_to_id["O"])
            continue
        entity = ordered[cursor]
        if end <= entity["start"] or start >= entity["end"]:
            labels.append(tag_to_id["O"])
            continue
        # Границы определяем по тому, содержит ли токен первый/последний символ.
        first = start <= entity["start"] < end
        last = start < entity["end"] <= end
        if scheme == "bio":
            prefix = "B" if first else "I"
        elif first and last:
            prefix = "U"
        elif first:
            prefix = "B"
        elif last:
            prefix = "L"
        else:
            prefix = "I"
        labels.append(tag_to_id[f"{prefix}-{entity['label']}"])
    return labels


def decode_tokens(tokens: list[tuple[int, int, str]], scheme: str) -> list[JsonObject]:
    """Собирает символьные spans из предсказанных меток токенов.

    Декодирование снисходительное: некорректные последовательности вроде I без
    предшествующего B не отбрасываются, а начинают новую сущность — так модель
    теряет меньше найденного.
    """

    if scheme not in SCHEMES:
        raise ValueError(f"неизвестная схема разметки: {scheme}")
    entities: list[JsonObject] = []
    current: JsonObject | None = None

    def flush() -> None:
        """Закрывает накопленную сущность."""

        nonlocal current
        if current is not None:
            entities.append(current)
            current = None

    for start, end, tag in tokens:
        if tag == "O":
            flush()
            continue
        prefix, separator, label = tag.partition("-")
        if separator != "-" or label not in ENTITY_LABELS:
            raise ValueError(f"модель вернула неподдерживаемый тег {tag!r}")

        if prefix == "U":
            flush()
            entities.append({"label": label, "start": start, "end": end})
            continue
        if prefix == "B":
            flush()
            current = {"label": label, "start": start, "end": end}
            continue
        # I и L: продолжаем открытую сущность того же класса либо начинаем новую.
        if current is None or current["label"] != label:
            flush()
            current = {"label": label, "start": start, "end": end}
        else:
            current["end"] = max(current["end"], end)
        if prefix == "L":
            flush()
    flush()
    return entities


class NerDataset(Dataset):
    """Окна с метками выбранной схемы разметки."""

    def __init__(
        self,
        records: list[JsonObject],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        stride: int,
        scheme: str,
        description: str,
    ) -> None:
        self.features: list[JsonObject] = []
        for record in tqdm(records, desc=description, unit="doc"):
            for feature, offsets in tokenize_windows(
                tokenizer, record["text"], max_length=max_length, stride=stride
            ):
                labels = align_labels(offsets, record["entities"], scheme)
                if all(label == -100 for label in labels):
                    continue
                self.features.append({**feature, "labels": labels})
        if not self.features:
            raise ValueError(f"{description}: не получилось ни одного окна с метками")

    def __len__(self) -> int:
        """Количество окон."""

        return len(self.features)

    def __getitem__(self, index: int) -> JsonObject:
        """Одно окно для DataLoader."""

        return self.features[index]


def _self_test() -> int:
    """Кодирует и декодирует эталонные разметки обратно в spans."""

    failures = 0
    text = "Toshkentda Oqtepa Lavash ochildi"
    entities = [
        {"label": "GEO", "start": 0, "end": 10},
        {"label": "ORG", "start": 11, "end": 24},
    ]
    # Токенизация по словам — достаточно, чтобы проверить сами схемы.
    offsets: Offsets = [(0, 0), (0, 10), (11, 17), (18, 24), (25, 32), (0, 0)]
    for scheme in SCHEMES:
        tags = tags_for(scheme)
        ids = align_labels(offsets, entities, scheme)
        tagged = [
            (start, end, tags[label])
            for (start, end), label in zip(offsets, ids, strict=True)
            if label != -100
        ]
        decoded = decode_tokens(tagged, scheme)
        ok = decoded == entities
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {scheme}: {[t[2] for t in tagged]}")
        print(f"         -> {[(item['label'], text[item['start']:item['end']]) for item in decoded]}")

    # Две подряд идущие сущности одного класса: BIO их склеит, BILOU — нет.
    pair = [{"label": "GEO", "start": 0, "end": 8}, {"label": "GEO", "start": 9, "end": 18}]
    pair_offsets: Offsets = [(0, 8), (9, 18)]
    for scheme, expected in (("bio", 2), ("bilou", 2)):
        tags = tags_for(scheme)
        ids = align_labels(pair_offsets, pair, scheme)
        tagged = [(s, e, tags[i]) for (s, e), i in zip(pair_offsets, ids, strict=True)]
        decoded = decode_tokens(tagged, scheme)
        ok = len(decoded) == expected
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {scheme}: две соседние GEO -> {len(decoded)} сущности")

    print(f"\nтегов: bio {len(tags_for('bio'))}, bilou {len(tags_for('bilou'))}")
    print(f"провалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
