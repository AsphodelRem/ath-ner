"""Аугментации"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

# Режимы искажения регистра и их относительные веса.
CASE_MODES = (
    ("entity_upper", 3),
    ("entity_lower", 3),
    ("entity_title", 2),
    ("document_lower", 1),
    ("document_upper", 1),
)
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# Частотные окончания узбекских сущностей.
AFFIXES = (
    "larining", "ларининг", "lardagi", "лардаги", "gacha", "гача", "dagi", "даги",
    "ning", "нинг", "lari", "лари", "ligi", "лиги", "lik", "лик", "dan", "дан",
    "lar", "лар", "ga", "га", "da", "да", "ni", "ни", "si", "си", "i", "и",
)
MIN_CANDIDATES = 5


def safe_lower(text: str) -> str:
    """Приводит к нижнему регистру, не меняя длину строки.

    Турецкая `İ` в нижнем регистре занимает два кодпоинта, немецкая `ß` в
    верхнем — тоже. Такие символы оставляем как есть, иначе поедут офсеты.
    """

    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def safe_upper(text: str) -> str:
    """Приводит к верхнему регистру, не меняя длину строки."""

    return "".join(c.upper() if len(c.upper()) == 1 else c for c in text)


def safe_title(text: str) -> str:
    """Делает первую букву каждого слова заглавной, длину не меняя."""

    result = []
    new_word = True
    for char in text:
        if LETTER.match(char):
            result.append(safe_upper(char) if new_word else safe_lower(char))
            new_word = False
        else:
            result.append(char)
            new_word = True
    return "".join(result)


def script_of(mention: str) -> str:
    """Кириллица или латиница — по преобладанию букв."""

    return "cyr" if CYRILLIC.search(mention) else "lat"


def detect_affix(mention: str) -> str:
    """Возвращает поверхностное окончание упоминания или пустую строку."""

    lowered = mention.lower()
    for affix in AFFIXES:
        if len(lowered) > len(affix) + 2 and lowered.endswith(affix):
            return affix
    return ""


def build_gazetteer(records: list[JsonObject], *, min_length: int = 2) -> dict[str, list[str]]:
    """Собирает формы по трём уровням ключа, от точного к общему.

    `метка:графика:окончание` — основной ключ: замена сохраняет и графику,
    и морфологический рисунок. Если таких форм мало, поиск падает на
    `метка:графика`, затем на просто `метка`.
    """

    forms: dict[str, set[str]] = defaultdict(set)
    for record in records:
        text = record["text"]
        for entity in record["entities"]:
            mention = text[entity["start"]:entity["end"]]
            if len(mention) < min_length:
                continue
            label, script = entity["label"], script_of(mention)
            forms[f"{label}:{script}:{detect_affix(mention)}"].add(mention)
            forms[f"{label}:{script}"].add(mention)
            forms[label].add(mention)
    return {key: sorted(values) for key, values in forms.items()}


def augment_case(record: JsonObject, rng: random.Random) -> JsonObject:
    """Меняет регистр текста или отдельных сущностей; координаты сохраняются."""

    text = record["text"]
    entities = record["entities"]
    mode = rng.choices([name for name, _ in CASE_MODES], [w for _, w in CASE_MODES])[0]

    if mode == "document_lower":
        return {**record, "text": safe_lower(text)}
    if mode == "document_upper":
        return {**record, "text": safe_upper(text)}
    if not entities:
        return record

    transform = {"entity_upper": safe_upper, "entity_lower": safe_lower, "entity_title": safe_title}[mode]
    chars = list(text)
    for entity in entities:
        if rng.random() > 0.5:  # искажаем не все сущности сразу
            continue
        start, end = entity["start"], entity["end"]
        replacement = transform(text[start:end])
        if len(replacement) != end - start:  # страховка, не должно случаться
            continue
        chars[start:end] = list(replacement)
    return {**record, "text": "".join(chars)}


def _match_case(source: str, replacement: str) -> str:
    """Переносит рисунок регистра исходного упоминания на новое."""

    if source.isupper() and len(source) > 1:
        return safe_upper(replacement)
    first = next((c for c in source if LETTER.match(c)), "")
    if first and first.islower():
        return safe_lower(replacement)
    return replacement


def augment_swap(
    record: JsonObject,
    gazetteer: dict[str, list[str]],
    rng: random.Random,
    *,
    probability: float = 0.5,
) -> JsonObject:
    """Заменяет упоминания на другие формы той же метки, пересчитывая координаты."""

    text = record["text"]
    entities = sorted(record["entities"], key=lambda item: item["start"])
    if not entities:
        return record

    pieces: list[str] = []
    new_entities: list[JsonObject] = []
    cursor = 0
    for entity in entities:
        start, end, label = entity["start"], entity["end"], entity["label"]
        mention = text[start:end]
        # От точного ключа к общему: окончание, затем графика, затем метка.
        candidates: list[str] = []
        for key in (
            f"{label}:{script_of(mention)}:{detect_affix(mention)}",
            f"{label}:{script_of(mention)}",
            label,
        ):
            candidates = gazetteer.get(key) or []
            if len(candidates) >= MIN_CANDIDATES:
                break
        replacement = mention
        if candidates and rng.random() < probability:
            candidate = candidates[rng.randrange(len(candidates))]
            if candidate != mention:
                replacement = _match_case(mention, candidate)
        pieces.append(text[cursor:start])
        offset = sum(len(piece) for piece in pieces)
        pieces.append(replacement)
        new_entities.append({"label": label, "start": offset, "end": offset + len(replacement)})
        cursor = end
    pieces.append(text[cursor:])
    return {**record, "text": "".join(pieces), "entities": new_entities}


class Augmenter:
    """Применяет аугментации к записи с заданными вероятностями."""

    def __init__(
        self,
        gazetteer: dict[str, list[str]] | None = None,
        *,
        case_probability: float = 0.0,
        swap_probability: float = 0.0,
        swap_share: float = 0.5,
    ) -> None:
        self.gazetteer = gazetteer or {}
        self.case_probability = case_probability
        self.swap_probability = swap_probability
        self.swap_share = swap_share

    @property
    def enabled(self) -> bool:
        """Включена ли хотя бы одна аугментация."""

        return self.case_probability > 0 or self.swap_probability > 0

    def apply(self, record: JsonObject, rng: random.Random) -> JsonObject:
        """Возвращает изменённую копию записи (или её саму, если ничего не выпало)."""

        result = record
        if self.gazetteer and rng.random() < self.swap_probability:
            result = augment_swap(result, self.gazetteer, rng, probability=self.swap_share)
        if rng.random() < self.case_probability:
            result = augment_case(result, rng)
        return result

    def apply_all(self, records: list[JsonObject], seed: int) -> list[JsonObject]:
        """Прогоняет всю выборку; seed делает эпоху воспроизводимой."""

        rng = random.Random(seed)
        return [self.apply(record, rng) for record in records]


def validate(record: JsonObject) -> None:
    """Проверяет инварианты формата кейса; бросает ValueError при нарушении."""

    text = record["text"]
    previous_end = -1
    for entity in sorted(record["entities"], key=lambda item: item["start"]):
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError(f"спан вне текста: {start}:{end} при длине {len(text)}")
        mention = text[start:end]
        if mention != mention.strip():
            raise ValueError(f"пробел на краю спана: {mention!r}")
        if start < previous_end:
            raise ValueError("спаны пересеклись")
        previous_end = end


def _self_test() -> int:
    """Прогоняет аугментации по обучающей выборке и проверяет координаты."""

    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "train.jsonl"
    if not path.exists():
        print(f"нет {path}, пропускаю проверку на данных")
        return 0

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:2000]]
    gazetteer = build_gazetteer(records)
    print("### газеттир")
    for label in ("ORG", "NAME", "GEO"):
        exact = {k: v for k, v in gazetteer.items() if k.count(":") == 2 and k.startswith(label)}
        big = sum(1 for v in exact.values() if len(v) >= MIN_CANDIDATES)
        print(f"  {label:<6}{len(gazetteer[label]):>6} форм | групп по окончанию: {len(exact)}"
              f", из них пригодных (>= {MIN_CANDIDATES} форм): {big}")

    rng = random.Random(0)
    print("\n### регистр: длина текста и координаты не меняются")
    changed = 0
    for record in records:
        augmented = augment_case(record, rng)
        if len(augmented["text"]) != len(record["text"]):
            raise SystemExit(f"ДЛИНА ИЗМЕНИЛАСЬ: {record['hash']}")
        validate(augmented)
        if augmented["text"] != record["text"]:
            changed += 1
    print(f"  изменено {changed} из {len(records)}, длина совпадает везде, координаты валидны")

    print("\n### подмена сущностей: координаты пересчитаны верно")
    swapped_total = affix_kept = 0
    for record in records:
        augmented = augment_swap(record, gazetteer, rng, probability=0.5)
        validate(augmented)
        if len(augmented["entities"]) != len(record["entities"]):
            raise SystemExit(f"ПОТЕРЯНЫ СУЩНОСТИ: {record['hash']}")
        for old, new in zip(record["entities"], augmented["entities"], strict=True):
            if old["label"] != new["label"]:
                raise SystemExit("метка изменилась")
            before = record["text"][old["start"]:old["end"]]
            after = augmented["text"][new["start"]:new["end"]]
            if after != before:
                swapped_total += 1
                if detect_affix(before) == detect_affix(after):
                    affix_kept += 1
    print(f"  подменено {swapped_total} упоминаний, все координаты указывают на новые строки")
    print(f"  окончание сохранено в {affix_kept} случаях ({100 * affix_kept / max(swapped_total, 1):.0f}%)")

    print("\n### примеры")
    sample = next(r for r in records if len(r["entities"]) >= 2 and len(r["text"]) < 200)
    show = lambda r: [(e["label"], r["text"][e["start"]:e["end"]]) for e in r["entities"]]
    print(f"  исходный : {sample['text'][:110]!r}")
    print(f"             {show(sample)}")
    case_rng = random.Random(7)
    for _ in range(4):
        variant = augment_case(sample, case_rng)
        if variant["text"] != sample["text"]:
            break
    print(f"  регистр  : {variant['text'][:110]!r}")
    print(f"             {show(variant)}")
    swap = augment_swap(sample, gazetteer, random.Random(3), probability=1.0)
    print(f"  подмена  : {swap['text'][:110]!r}")
    print(f"             {show(swap)}")

    print("\n### крайние случаи")
    tricky = {"hash": "x", "text": "İstanbul va Toshkent", "entities": [
        {"label": "GEO", "start": 0, "end": 8}, {"label": "GEO", "start": 12, "end": 20}]}
    for mode_rng in (random.Random(i) for i in range(12)):
        out = augment_case(tricky, mode_rng)
        if len(out["text"]) != len(tricky["text"]):
            raise SystemExit("İ сломала длину")
    print("  турецкая İ обработана: длина сохраняется во всех режимах")
    empty = {"hash": "y", "text": "no entities here", "entities": []}
    validate(augment_case(empty, rng))
    validate(augment_swap(empty, gazetteer, rng))
    print("  записи без сущностей проходят обе аугментации")
    print("\nпровалов нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
