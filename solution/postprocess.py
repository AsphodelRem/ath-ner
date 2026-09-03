"""
Приведение предсказанных spans к конвенциям LABELING_GUIDE.md.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

JsonObject = dict[str, Any]

APOSTROPHES = "'‘’ʻʼʽ`´"
# Кавычки и скобки внутри спана означают, что модель захватила лишнее.
BRACKETS = "«»\"“”„()[]{}<>"
LEAD_STRIP = BRACKETS + APOSTROPHES + "#@.,:;!?…-–—*/\\|№ \t\n\r "
TAIL_STRIP = BRACKETS + APOSTROPHES + ".,:;!?…-–—*/\\|&№ \t\n\r "
INVISIBLE = re.compile(r"[​-‏⁠﻿]")
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}", "«": "»", "“": "”", "„": "“"}

LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
WORD_CHAR = re.compile(r"[^\W_]", re.UNICODE)
MAX_SUFFIX = 10


def _is_word_char(char: str) -> bool:
    """Буква, цифра или внутренний апостроф — часть одного слова."""

    return bool(WORD_CHAR.match(char)) or char in APOSTROPHES


def _is_abbreviation_dot(text: str, start: int, end: int) -> bool:
    """Точка после одиночной буквы — часть аббревиатуры: A., ш., F.C."""

    if end - 2 < start or not LETTER.match(text[end - 2]):
        return False
    return end - 3 < start or not LETTER.match(text[end - 3])


def normalize_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Приводит одну пару координат к границам по правилам гайда."""

    if not 0 <= start < end <= len(text):
        return None

    # Внутренние кавычки намеренно не трогаем: в эталоне они встречаются
    # внутри официальных названий (`Oʻzsuvta'minot" AJ`, `Боруссия"си`),
    # и обрезка по ним портит 0,48% gold-спанов.
    while start < end and text[start] in LEAD_STRIP:
        # Открывающая скобка со своей парой внутри спана — часть названия: (G)I-DLE.
        closing = BRACKET_PAIRS.get(text[start])
        if closing is not None and closing in text[start + 1:end - 1]:
            break
        start += 1
    while end > start and text[end - 1] in TAIL_STRIP:
        # Точка после заглавной буквы — часть аббревиатуры: Zverev A., Arsenal F.C.
        if text[end - 1] == "." and _is_abbreviation_dot(text, start, end):
            break
        end -= 1
    if start >= end:
        return None

    # Начало внутри слова — сдвигаем к началу слова.
    while start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
        start -= 1

    # Конец внутри слова — дотягиваем аффикс: Toshkent -> Toshkentda.
    # Только строчные буквы и апострофы, чтобы не склеить два слова.
    if end < len(text) and _is_word_char(text[end]):
        cursor = end
        limit = min(len(text), end + MAX_SUFFIX)
        while cursor < limit:
            char = text[cursor]
            if char in APOSTROPHES or (LETTER.match(char) and char.islower()):
                cursor += 1
                continue
            break
        if cursor >= len(text) or not _is_word_char(text[cursor]):
            end = cursor

    mention = text[start:end]
    if mention != mention.strip() or not LETTER.search(mention):
        return None
    if INVISIBLE.fullmatch(mention):
        return None
    return start, end


def postprocess_entities(text: str, entities: list[JsonObject]) -> list[JsonObject]:
    """Нормализует границы, снимает дубли и пересечения внутри записи."""

    normalized: list[tuple[int, int, str]] = []
    for entity in entities:
        span = normalize_span(text, int(entity["start"]), int(entity["end"]))
        if span is None:
            continue
        normalized.append((span[0], span[1], entity["label"]))

    # Разметка плоская: при пересечении остаётся более длинная сущность.
    normalized = sorted(set(normalized), key=lambda item: (item[0] - item[1], item[0]))
    kept: list[tuple[int, int, str]] = []
    for span in normalized:
        if any(span[0] < other[1] and other[0] < span[1] for other in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda item: item[0])
    return [{"label": label, "start": start, "end": end} for start, end, label in kept]


def _self_test() -> int:
    """Прогоняет примеры из LABELING_GUIDE.md и печатает результат."""

    cases = [
        ("«EVOS»ga chiqdik", 1, 5, "EVOS", "суффикс за кавычкой не входит"),
        ("«EVOS»ga chiqdik", 0, 6, "EVOS", "внешние кавычки снимаются"),
        ("KFC da ovqatlandik", 0, 3, "KFC", "отдельное служебное слово не втягивается"),
        ("Toshkentda yashaydi", 0, 8, "Toshkentda", "аффикс втягивается в границы"),
        ("Farg'ona viloyati markazi", 0, 17, "Farg'ona viloyati", "составное название не трогаем"),
        ("Kelgan joyi Samarqand.", 12, 22, "Samarqand", "конечная точка не входит"),
        ("#Toshkent shahri", 0, 9, "Toshkent", "решётка не входит"),
        ("@mobiuz rasmiy kanali", 0, 7, "mobiuz", "собака не входит"),
        ("Mobiuz'dan xabar", 0, 6, "Mobiuz'dan", "аффикс через апостроф втягивается"),
        ("Oqtepa Lavash Xadra", 0, 13, "Oqtepa Lavash", "многословный бренд сохраняется"),
        ("2024 yilda", 0, 4, None, "спан без букв отбрасывается"),
    ]
    failures = 0
    for text, start, end, expected, comment in cases:
        span = normalize_span(text, start, end)
        actual = None if span is None else text[span[0]:span[1]]
        status = "ok " if actual == expected else "FAIL"
        if actual != expected:
            failures += 1
        print(f"  [{status}] {comment}: {text[start:end]!r} -> {actual!r} (ожидалось {expected!r})")

    text = "Toshkent shahrida Toshkent metrosi ishlaydi"
    overlapping = [
        {"label": "GEO", "start": 0, "end": 8},
        {"label": "GEO", "start": 0, "end": 17},
        {"label": "ORG", "start": 18, "end": 26},
    ]
    result = postprocess_entities(text, overlapping)
    mentions = [(item["label"], text[item["start"]:item["end"]]) for item in result]
    expected_mentions = [("GEO", "Toshkent shahrida"), ("ORG", "Toshkent")]
    status = "ok " if mentions == expected_mentions else "FAIL"
    if mentions != expected_mentions:
        failures += 1
    print(f"  [{status}] пересечения снимаются: {mentions}")

    print(f"\nпровалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
