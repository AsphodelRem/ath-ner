"""Сборка корпуса для доменной адаптации энкодера.

Источники (обе лицензии допускают использование):
  tahrirchi/uz-crawl        telegram_blogs — 368k постов, ровно наш домен
                            news — новостные тексты
  tahrirchi/uz-books-v2     cyr — 9,2 ГБ узбекской кириллицы
                            lat — 6,8 ГБ латиницы

Кириллический сплит здесь ключевой: измеренное отставание модели по кириллице
составляет несколько пунктов F1, а своих кириллических данных у нас 20% корпуса.

Тексты выборки кейса исключаются: пересечения по нормализованному тексту не
найдено ни одного, но проверка остаётся, чтобы метрика на dev не стала
транcдуктивной при смене версии источника.

    python tools/prepare_pretrain.py --sources telegram:all,books-cyr:3,news:2 \
        --max-tokens 300000000 --output data/pretrain/corpus.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

JsonObject = dict[str, Any]

SOURCES = {
    "telegram": ("tahrirchi/uz-crawl", "telegram_blogs"),
    "news": ("tahrirchi/uz-crawl", "news"),
    "books-cyr": ("tahrirchi/uz-books-v2", "cyr"),
    "books-lat": ("tahrirchi/uz-books-v2", "lat"),
    "books-train": ("tahrirchi/uz-books-v2", "train"),
    # Иноязычная подмешка против катастрофического забывания: адаптация на
    # чистом узбекском подъедает мультиязычность, а 5,2% нашей выборки —
    # тексты на английском и русском, и они дают F1 0,90-0,93.
    "wiki-ru": ("wikimedia/wikipedia", "20231101.ru"),
    "wiki-en": ("wikimedia/wikipedia", "20231101.en"),
}
FOREIGN = ("wiki-ru", "wiki-en")

APOSTROPHES = "'‘’ʻʼʽ`´"
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
# Запасное значение, если токенизатор не передан. Измерено на источниках
# для xlm-roberta-large: 2,67 у телеграма, 2,43 у кириллических книг.
# Величина сильно зависит от графики и модели, поэтому лучше калибровать.
CHARS_PER_TOKEN = 2.6
CALIBRATION_SAMPLE = 200

# Признаки OCR-шума, измеренные на кириллическом шарде uz-books-v2:
# смешение алфавитов внутри слова 0,54%, одиночные буквы как слова 1,95%.
# В чистом тексте оба показателя близки к нулю, поэтому пороги ставим низко.
MIXED_SCRIPT = re.compile(r"[A-Za-z][Ѐ-ӿ]|[Ѐ-ӿ][A-Za-z]")
LETTER_DIGIT = re.compile(r"[^\W\d_][0-9]|[0-9][^\W\d_]", re.UNICODE)
WORD = re.compile(r"\S+")
PARAGRAPH = re.compile(r"\n\s*\n+")


def parse_args() -> argparse.Namespace:
    """Разбирает состав корпуса и ограничения."""

    parser = argparse.ArgumentParser(description="Build a corpus for domain-adaptive pretraining.")
    parser.add_argument(
        "--sources",
        default="telegram:all,books-cyr:3,news:2,books-lat:2,wiki-ru:1,wiki-en:1",
        help="через запятую «имя:число шардов» либо «имя:all»",
    )
    parser.add_argument("--output", type=Path, default=Path("data/pretrain/corpus.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=300_000_000, help="бюджет токенов")
    parser.add_argument(
        "--foreign-share",
        type=float,
        default=0.07,
        help="доля бюджета на иноязычные источники (wiki-ru, wiki-en): страховка "
             "от забывания мультиязычности; 0 отключает",
    )
    parser.add_argument("--min-chars", type=int, default=200, help="минимальная длина документа")
    parser.add_argument("--max-chars", type=int, default=100_000, help="документы длиннее режутся")
    parser.add_argument(
        "--passage-chars",
        type=int,
        default=4000,
        help="длинные документы режутся на пассажи по абзацам; книги в uz-books-v2 "
             "имеют медианную длину 260 тыс. символов, целиком они бесполезны",
    )
    parser.add_argument(
        "--max-ocr-noise",
        type=float,
        default=0.01,
        help="доля слов со смешением алфавитов или буквой рядом с цифрой",
    )
    parser.add_argument(
        "--max-single-letters",
        type=float,
        default=0.06,
        help="доля одиночных букв как отдельных слов: колонтитулы и номера страниц",
    )
    parser.add_argument("--holdout", type=int, default=2000, help="документов в held-out для перплексии")
    parser.add_argument("--near-dedup-bands", type=int, default=8, help="0 отключает near-dedup")
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[Path("data/train.jsonl"), Path("data/dev.jsonl")],
        help="JSONL кейса: совпадающие тексты исключаются",
    )
    parser.add_argument(
        "--tokenizer",
        help="модель или каталог для калибровки бюджета; без неё берётся запасное значение",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize(text: str) -> str:
    """Нормализует текст для сравнения и дедупликации."""

    folded = unicodedata.normalize("NFKC", text or "").casefold()
    for mark in APOSTROPHES:
        folded = folded.replace(mark, "'")
    return re.sub(r"\s+", " ", folded).strip()


def digest(text: str) -> str:
    """Короткий хеш нормализованного текста."""

    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def read_excluded(paths: list[Path]) -> set[str]:
    """Собирает хеши текстов выборки кейса."""

    hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    hashes.add(digest(json.loads(line).get("text", "")))
    return hashes


def split_passages(text: str, limit: int) -> Iterator[str]:
    """Режет длинный документ на пассажи, не разрывая абзацы без нужды.

    Сначала пробуем границы абзацев (пустая строка), затем одиночные переводы
    строки: в книгах uz-books-v2 пустых строк нет вовсе, и без этого запасного
    варианта вся книга уезжает одним куском. Блок, который и так длиннее
    лимита, режется жёстко.
    """

    if len(text) <= limit:
        yield text
        return
    blocks = [b for b in PARAGRAPH.split(text) if b.strip()]
    if len(blocks) < 2:
        blocks = [b for b in text.split("\n") if b.strip()]
    buffer = ""
    for block in blocks:
        block = block.strip()
        while len(block) > limit:  # абзац длиннее лимита режем по месту
            if buffer:
                yield buffer
                buffer = ""
            yield block[:limit]
            block = block[limit:]
        if buffer and len(buffer) + len(block) + 1 > limit:
            yield buffer
            buffer = block
        else:
            buffer = f"{buffer}\n{block}" if buffer else block
    if buffer:
        yield buffer


def ocr_noise(text: str) -> tuple[float, float]:
    """Возвращает доли слов с признаками OCR-шума и одиночных букв."""

    words = WORD.findall(text)
    if not words:
        return 1.0, 1.0
    noisy = sum(1 for w in words if MIXED_SCRIPT.search(w) or LETTER_DIGIT.search(w))
    single = sum(1 for w in words if len(re.sub(r"[^\w]", "", w)) == 1 and w[:1].isalpha())
    return noisy / len(words), single / len(words)


class NearDuplicateIndex:
    """Лёгкий MinHash-LSH без внешних зависимостей.

    Новости массово перепечатываются, и точная дедупликация их не ловит:
    достаточно поменять заголовок. Сигнатура строится по 5-словным шинглам,
    делится на полосы, и документ считается дублем при совпадении полосы.
    """

    def __init__(self, bands: int = 8, rows: int = 4) -> None:
        self.bands = bands
        self.rows = rows
        self.size = bands * rows
        self.buckets: list[set[int]] = [set() for _ in range(bands)]

    def _signature(self, text: str) -> list[int]:
        """Считает MinHash по шинглам из пяти слов."""

        words = WORD.findall(normalize(text))
        shingles = {" ".join(words[i:i + 5]) for i in range(max(len(words) - 4, 1))}
        if not shingles:
            return [0] * self.size
        hashed = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
                  for s in shingles]
        # Разные перестановки эмулируем разными смещениями хеша.
        return [min((h ^ (seed * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF for h in hashed)
                for seed in range(self.size)]

    def add_if_new(self, text: str) -> bool:
        """True, если документ новый; иначе регистрирует дубль и возвращает False."""

        signature = self._signature(text)
        keys = []
        for band in range(self.bands):
            chunk = signature[band * self.rows:(band + 1) * self.rows]
            keys.append(int.from_bytes(
                hashlib.blake2b(str(chunk).encode(), digest_size=8).digest(), "big"))
        if any(key in self.buckets[band] for band, key in enumerate(keys)):
            return False
        for band, key in enumerate(keys):
            self.buckets[band].add(key)
        return True


def is_uzbek_like(text: str) -> bool:
    """Отсеивает документы без осмысленной доли букв нужных алфавитов."""

    letters = sum(1 for char in text if LETTER.match(char))
    if letters < len(text) * 0.4:
        return False
    return bool(CYRILLIC.search(text) or LATIN.search(text))


def iter_shard_texts(repo: str, split: str, limit: int | None) -> Iterator[str]:
    """Читает parquet-шарды указанного сплита по одному."""

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    # Две раскладки: у tahrirchi это data/<split>-00000-of-N.parquet,
    # у wikimedia/wikipedia — <config>/train-00000-of-N.parquet.
    files = sorted(
        item.rfilename
        for item in HfApi().dataset_info(repo).siblings
        if item.rfilename.endswith(".parquet")
        and (f"/{split}-" in item.rfilename or item.rfilename.startswith(f"{split}/"))
    )
    if not files:
        raise ValueError(f"{repo}: не найдено шардов для сплита {split}")
    if limit is not None:
        files = files[:limit]
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset")
        table = pq.read_table(path)
        column = next(
            (c for c in table.column_names if c.lower() in ("text", "content", "body")),
            table.column_names[0],
        )
        print(f"    {name}: {table.num_rows} строк")
        for value in table.column(column).to_pylist():
            if value:
                yield value


def run(args: argparse.Namespace) -> int:
    """Собирает корпус и пишет JSONL с манифестом."""

    excluded = read_excluded(args.exclude)
    print(f"Исключаемых текстов кейса: {len(excluded)}")

    chars_per_token = CHARS_PER_TOKEN
    calibration: list[str] = []
    if args.tokenizer:
        from transformers import AutoTokenizer

        calibrator = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Бюджет калибруется по {args.tokenizer}")
    else:
        calibrator = None
        print(f"Токенизатор не задан, бюджет считается по {CHARS_PER_TOKEN} симв./токен")

    seen: set[str] = set()
    seen_paragraphs: set[str] = set()
    near = NearDuplicateIndex(args.near_dedup_bands) if args.near_dedup_bands else None
    stats: Counter = Counter()
    scripts: Counter = Counter()
    per_source: Counter = Counter()
    total_chars = 0
    foreign_chars = 0
    budget_chars = int(args.max_tokens * chars_per_token)
    # Квота иноязычных резервируется заранее: без этого узбекские источники
    # выбирают весь бюджет и до подмешки дело не доходит вовсе.
    foreign_budget = int(budget_chars * args.foreign_share)
    native_budget = budget_chars - foreign_budget
    # Квота делится поровну между источниками каждой группы: без этого первый
    # по порядку выбирает всё, а остальные получают по одному пассажу — состав
    # корпуса определяется порядком в --sources, а не замыслом.
    requested = [spec.split(":")[0].strip() for spec in args.sources.split(",") if spec.strip()]
    requested_foreign = [n for n in requested if n in FOREIGN]
    requested_native = [n for n in requested if n not in FOREIGN]
    per_foreign_budget = foreign_budget // max(len(requested_foreign), 1)
    per_native_budget = native_budget // max(len(requested_native), 1)
    chars_by_source: Counter = Counter()
    foreign_by_source: Counter = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    holdout_path = args.output.with_name(args.output.stem + ".holdout.jsonl")
    with args.output.open("w", encoding="utf-8") as out, holdout_path.open("w", encoding="utf-8") as holdout:
        for spec in args.sources.split(","):
            name, _, count = spec.partition(":")
            name = name.strip()
            if name not in SOURCES:
                raise ValueError(f"неизвестный источник {name}; доступны: {', '.join(SOURCES)}")
            repo, split = SOURCES[name]
            is_foreign = name in FOREIGN
            if is_foreign and args.foreign_share <= 0:
                continue
            limit = None if count.strip() in ("", "all") else int(count)

            def source_full(foreign: bool = is_foreign, source: str = name) -> bool:
                """Исчерпана ли квота текущего источника."""

                quota = per_foreign_budget if foreign else per_native_budget
                return chars_by_source[source] >= quota
            print(f"\n=== {name} ({repo}, сплит {split}, шардов: {count or 'all'})")

            for document in iter_shard_texts(repo, split, limit):
                stats["прочитано документов"] += 1
                for text in split_passages(document[: args.max_chars], args.passage_chars):
                    stats["пассажей"] += 1
                    if len(text) < args.min_chars:
                        stats["коротких"] += 1
                        continue
                    if not is_uzbek_like(text):
                        stats["не похоже на узбекский"] += 1
                        continue
                    noise, single = ocr_noise(text)
                    if noise > args.max_ocr_noise or single > args.max_single_letters:
                        stats["OCR-шум"] += 1
                        continue
                    key = digest(text)
                    if key in excluded:
                        stats["совпадает с выборкой кейса"] += 1
                        continue
                    if key in seen:
                        stats["точные дубли"] += 1
                        continue
                    # Абзацная дедупликация: перепечатки делят абзацы целиком.
                    paragraphs = [p for p in PARAGRAPH.split(text) if len(p.strip()) > 80]
                    if paragraphs and all(digest(p) in seen_paragraphs for p in paragraphs):
                        stats["дубли по абзацам"] += 1
                        continue
                    for paragraph in paragraphs:
                        seen_paragraphs.add(digest(paragraph))
                    if near is not None and not near.add_if_new(text):
                        stats["near-дубли"] += 1
                        continue
                    seen.add(key)

                    # Первые --holdout пассажей уходят в отложенную выборку:
                    # по ней меряется перплексия и решается, когда остановиться.
                    target = holdout if stats["held-out"] < args.holdout else out
                    if target is holdout:
                        stats["held-out"] += 1
                    else:
                        stats["записано"] += 1
                        per_source[name] += 1
                        total_chars += len(text)
                        chars_by_source[name] += len(text)
                        if is_foreign:
                            foreign_chars += len(text)
                            foreign_by_source[name] += len(text)
                        # Первые пассажи используем для калибровки бюджета:
                        # брать константу вслепую значит промахнуться в
                        # числе шагов обучения на десятки процентов.
                        if calibrator is not None and len(calibration) < CALIBRATION_SAMPLE:
                            calibration.append(text[:20000])
                            if len(calibration) == CALIBRATION_SAMPLE:
                                measured = sum(len(t) for t in calibration) / sum(
                                    len(calibrator(t, add_special_tokens=False)["input_ids"])
                                    for t in calibration
                                )
                                chars_per_token = measured
                                budget_chars = int(args.max_tokens * chars_per_token)
                                print(f"    калибровка: {measured:.2f} симв./токен,"
                                      f" бюджет {budget_chars / 1e6:.0f} млн символов")
                        has_cyrillic = bool(CYRILLIC.search(text))
                        has_latin = bool(LATIN.search(text))
                        scripts["обе графики" if has_cyrillic and has_latin
                                else "кириллица" if has_cyrillic else "латиница"] += 1
                    target.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    if source_full():
                        print(f"    квота источника выбрана"
                              f" ({chars_by_source[name] / 1e6:.1f} млн символов)")
                        break
                # Выход из цикла по пассажам не прекращает чтение документов —
                # нужен явный второй выход, иначе бюджет не соблюдается.
                if source_full():
                    break
            if total_chars >= budget_chars:
                break

    tokens = int(total_chars / chars_per_token)
    manifest = {
        "sources": args.sources,
        "documents": stats["записано"],
        "characters": total_chars,
        "estimated_tokens": tokens,
        "chars_per_token_used": round(chars_per_token, 3),
        "chars_per_token_calibrated": bool(args.tokenizer),
        "tokenizer": args.tokenizer,
        "by_source": dict(per_source),
        "foreign_share_requested": args.foreign_share,
        "foreign_share_actual": round(foreign_chars / max(total_chars, 1), 4),
        "foreign_by_source": {k: round(v / max(total_chars, 1), 4) for k, v in foreign_by_source.items()},
        "share_by_source": {k: round(v / max(total_chars, 1), 4) for k, v in chars_by_source.items()},
        "by_script": dict(scripts),
        "filters": dict(stats),
        "excluded_from": [str(p) for p in args.exclude],
        "output": str(args.output),
        "holdout": str(holdout_path),
        "holdout_documents": stats["held-out"],
        "passage_chars": args.passage_chars,
        "max_ocr_noise": args.max_ocr_noise,
        "near_dedup_bands": args.near_dedup_bands,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Итог")
    for key, value in stats.most_common():
        print(f"  {key:<30}{value:>12}")
    print(f"\n  символов: {total_chars / 1e6:.1f} млн | токенов: {tokens / 1e6:.1f} млн"
          f" ({chars_per_token:.2f} симв./токен)")
    print(f"  шагов при батче B, накоплении A и окне L: {tokens} / (B*A*L)")
    print("  по графикам: " + ", ".join(f"{k} {100 * v / max(stats['записано'], 1):.1f}%"
                                        for k, v in scripts.most_common()))
    print("  доли по источникам: " + ", ".join(
        f"{k} {100 * v / max(total_chars, 1):.0f}%" for k, v in chars_by_source.most_common()))
    print(f"  иноязычных: {100 * foreign_chars / max(total_chars, 1):.1f}%"
          f" (запрошено {100 * args.foreign_share:.0f}%)")
    print(f"\n  корпус:   {args.output}")
    print(f"  held-out: {holdout_path} ({stats['held-out']} пассажей)")
    print(f"  манифест: {manifest_path}")
    return 0


def main() -> int:
    """Точка входа CLI."""

    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())


