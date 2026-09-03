"""Разбор предсказаний по срезам, а не только общим micro F1.

Считает exact-span метрики отдельно по языку документа, его длине и по тому,
встречалась ли поверхностная форма сущности в обучающей выборке. Принимает
несколько файлов предсказаний сразу, чтобы сравнивать конфигурации.

    python tools/evaluate_slices.py --gold data/dev.jsonl \
        --predictions artifacts/xlmr-s42/dev_predictions.jsonl \
                      artifacts/glot500-s42/dev_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

JsonObject = dict[str, Any]
Key = tuple[str, int, int]

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
UZ_CYR_CHARS = set("ўқғҳЎҚҒҲ")

# Стоп-слова для эвристического определения языка документа.
STOPWORDS = {
    "узб. латиница": """va bilan uchun bo'lgan bolgan deb ham bu kerak yil lekin ammo shuningdek hamda
        yoki emas edi qilib haqida ular uning bir necha ko'proq koproq bosh yangi degan bo'yicha esa
        hozir juda qilgan bergan berilgan amalga oshirildi mumkin kelib chiqqan tomonidan davomida
        oldin keyin barcha qildi bo'lib bolib qanday nima qachon qayerda sababli natijasida ushbu""".split(),
    "узб. кириллица": """ва билан учун бўлган деб ҳам бу керак йил лекин аммо шунингдек ҳамда ёки эмас
        эди қилиб ҳақида улар унинг бир неча кўпроқ бош янги деган бўйича эса ҳозир жуда қилган берган
        берилган амалга оширилди мумкин келиб чиққан томонидан давомида олдин кейин барча қилди
        бўлиб қандай нима қачон қаерда сабабли натижасида ушбу мазкур""".split(),
    "русский": """что это который которые были был была как для не на по из за или но также если уже
        когда очень более менее чтобы этом этой этого всех своих может года году лет россии сообщает
        отметил заявил около после перед между тысяч рублей""".split(),
    "английский": """the and of to in is for with on that this are was were will has have been from as
        at by it its their they which about would could should not but or you your we our more than
        said also can may new one two first year years""".split(),
    "турецкий": "ve bir için ile bu olarak daha çok sonra kadar".split(),
}
STOPSETS = {name: set(words) for name, words in STOPWORDS.items()}


def detect_language(text: str) -> str:
    """Определяет язык документа по стоп-словам и алфавиту (эвристика)."""

    tokens = [word.lower() for word in WORD.findall(text)]
    if not tokens:
        return "короткий/неопр."
    total = len(tokens)
    scores = {name: sum(token in words for token in tokens) / total for name, words in STOPSETS.items()}
    has_cyrillic = bool(CYRILLIC.search(text))
    has_latin = bool(LATIN.search(text))
    if UZ_CYR_CHARS & set(text):
        scores["узб. кириллица"] += 0.05
    if not has_cyrillic:
        scores["узб. кириллица"] = scores["русский"] = 0.0
    if not has_latin:
        scores["узб. латиница"] = scores["английский"] = scores["турецкий"] = 0.0
    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.03 else "короткий/неопр."


def length_bucket(text: str) -> str:
    """Группирует документ по длине."""

    size = len(text)
    if size < 200:
        return "1. <200 симв."
    if size < 1000:
        return "2. 200-1000"
    if size < 3000:
        return "3. 1000-3000"
    return "4. >3000"


def read_jsonl(path: Path) -> list[JsonObject]:
    """Читает JSONL."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keys_of(record: JsonObject) -> set[Key]:
    """Переводит сущности записи в множество ключей."""

    return {(item["label"], int(item["start"]), int(item["end"])) for item in record["entities"]}


def f1_of(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Считает precision, recall и F1."""

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def slice_report(
    gold: list[JsonObject],
    predictions: dict[str, dict[str, set[Key]]],
    bucket_of: Callable[[JsonObject], str],
    title: str,
) -> None:
    """Печатает метрики по срезу для всех наборов предсказаний."""

    counts: dict[str, dict[str, list[int]]] = {
        name: defaultdict(lambda: [0, 0, 0]) for name in predictions
    }
    sizes: Counter = Counter()
    for record in gold:
        bucket = bucket_of(record)
        truth = keys_of(record)
        sizes[bucket] += len(truth)
        for name, by_hash in predictions.items():
            predicted = by_hash.get(record["hash"], set())
            cell = counts[name][bucket]
            cell[0] += len(truth & predicted)
            cell[1] += len(predicted - truth)
            cell[2] += len(truth - predicted)

    names = list(predictions)
    print(f"\n### {title}")
    header = f"  {'группа':<20}{'сущн.':>7}" + "".join(f"{name[:14]:>16}" for name in names)
    print(header)
    for bucket in sorted(sizes, key=lambda item: -sizes[item]):
        cells = []
        for name in names:
            tp, fp, fn = counts[name][bucket]
            cells.append(f"{f1_of(tp, fp, fn)[2]:>16.4f}")
        print(f"  {bucket:<20}{sizes[bucket]:>7}" + "".join(cells))
    if len(names) == 2:
        print(f"\n  разница в пунктах: {names[1]} минус {names[0]}")
        for bucket in sorted(sizes, key=lambda item: -sizes[item]):
            first = f1_of(*counts[names[0]][bucket])[2]
            second = f1_of(*counts[names[1]][bucket])[2]
            print(f"  {bucket:<20}{sizes[bucket]:>7}{100 * (second - first):>+16.2f}")


def main() -> int:
    """Точка входа CLI."""

    parser = argparse.ArgumentParser(description="Evaluate predictions by data slices.")
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    predictions = {}
    for path in args.predictions:
        name = path.parent.name if path.name.startswith("dev_") else path.stem
        predictions[name] = {record["hash"]: keys_of(record) for record in read_jsonl(path)}

    vocabulary: dict[str, set[str]] = defaultdict(set)
    for record in read_jsonl(args.train):
        for entity in record["entities"]:
            vocabulary[entity["label"]].add(record["text"][entity["start"]:entity["end"]])

    slice_report(gold, predictions, lambda record: detect_language(record["text"]), "F1 по языку документа")
    slice_report(gold, predictions, lambda record: length_bucket(record["text"]), "F1 по длине документа")

    print("\n### recall по знакомости формы (главный индикатор обобщения)")
    print(f"  {'группа':<20}{'сущн.':>7}" + "".join(f"{name[:14]:>16}" for name in predictions))
    stats: dict[str, dict[str, list[int]]] = {name: defaultdict(lambda: [0, 0]) for name in predictions}
    sizes: Counter = Counter()
    for record in gold:
        for entity in record["entities"]:
            mention = record["text"][entity["start"]:entity["end"]]
            bucket = "знакомая форма" if mention in vocabulary[entity["label"]] else "НЕ встречалась"
            sizes[bucket] += 1
            key = (entity["label"], entity["start"], entity["end"])
            for name, by_hash in predictions.items():
                cell = stats[name][bucket]
                cell[1] += 1
                cell[0] += key in by_hash.get(record["hash"], set())
    for bucket in ("знакомая форма", "НЕ встречалась"):
        cells = "".join(
            f"{100 * stats[name][bucket][0] / max(stats[name][bucket][1], 1):>15.1f}%"
            for name in predictions
        )
        print(f"  {bucket:<20}{sizes[bucket]:>7}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
