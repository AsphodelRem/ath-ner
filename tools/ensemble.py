"""Ансамблирование предсказаний нескольких прогонов голосованием по спанам.

Каждый спан (start, end, label) получает по голосу от каждого прогона, который
его предсказал. В финальный ответ идут спаны, набравшие не меньше порога голосов;
пересечения снимаются жадно — приоритет у более согласованных, затем у длинных.

    python -m tools.ensemble --gold data/dev.jsonl --runs artifacts/glot500-s42 artifacts/run04
    python -m tools.ensemble --gold data/dev.jsonl --runs artifacts/* --sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from solution.metrics import evaluate
from solution.postprocess import postprocess_entities

JsonObject = dict[str, Any]
Span = tuple[int, int, str]


def read_jsonl(path: Path) -> list[JsonObject]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_run(path: Path) -> Path:
    """Каталог артефакта или прямой путь до файла предсказаний."""

    if path.is_dir():
        candidate = path / "dev_predictions.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"в {path} нет dev_predictions.jsonl")
        return candidate
    return path


def load_runs(paths: Iterable[Path]) -> dict[str, dict[str, set[Span]]]:
    runs: dict[str, dict[str, set[Span]]] = {}
    for raw in paths:
        path = resolve_run(Path(raw))
        name = path.parent.name if path.name == "dev_predictions.jsonl" else path.stem
        runs[name] = {
            record["hash"]: {
                (int(item["start"]), int(item["end"]), item["label"])
                for item in record["entities"]
            }
            for record in read_jsonl(path)
        }
    return runs


def vote(
    runs: dict[str, dict[str, set[Span]]],
    hashes: Iterable[str],
    threshold: int,
    *,
    prefer_votes: bool = True,
) -> dict[str, list[Span]]:
    """Отбирает спаны с числом голосов >= threshold и снимает пересечения."""

    result: dict[str, list[Span]] = {}
    for key in hashes:
        tally: dict[Span, int] = {}
        for run in runs.values():
            for span in run.get(key, ()):
                tally[span] = tally.get(span, 0) + 1

        candidates = [span for span, votes in tally.items() if votes >= threshold]
        # Больше голосов — выше приоритет; при равенстве побеждает длинный спан.
        if prefer_votes:
            candidates.sort(key=lambda s: (-tally[s], s[0] - s[1], s[0]))
        else:
            candidates.sort(key=lambda s: (s[0] - s[1], -tally[s], s[0]))

        kept: list[Span] = []
        for span in candidates:
            if any(span[0] < other[1] and other[0] < span[1] for other in kept):
                continue
            kept.append(span)
        result[key] = sorted(kept, key=lambda s: s[0])
    return result


def to_records(gold: list[JsonObject], chosen: dict[str, list[Span]], *, postprocess: bool) -> list[JsonObject]:
    records = []
    for record in gold:
        entities = [
            {"label": label, "start": start, "end": end}
            for start, end, label in chosen.get(record["hash"], [])
        ]
        if postprocess:
            entities = postprocess_entities(record["text"], entities)
        records.append({"hash": record["hash"], "text": record["text"], "entities": entities})
    return records


def show(title: str, metrics: JsonObject) -> None:
    micro, macro = metrics["micro"], metrics["macro"]
    print(f"\n{title}")
    print(f"  {'':6s} {'precision':>10s} {'recall':>9s} {'f1':>9s} {'tp':>6s} {'fp':>6s} {'fn':>6s}")
    for label in ("ORG", "NAME", "GEO"):
        row = metrics["by_label"][label]
        print(
            f"  {label:6s} {row['precision']:10.4f} {row['recall']:9.4f} {row['f1']:9.4f}"
            f" {row['tp']:6d} {row['fp']:6d} {row['fn']:6d}"
        )
    print(
        f"  {'micro':6s} {micro['precision']:10.4f} {micro['recall']:9.4f} {micro['f1']:9.4f}"
        f" {micro['tp']:6d} {micro['fp']:6d} {micro['fn']:6d}"
    )
    print(f"  {'macro':6s} {macro['precision']:10.4f} {macro['recall']:9.4f} {macro['f1']:9.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=Path("data/dev.jsonl"))
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="каталоги артефактов или .jsonl")
    parser.add_argument("--threshold", type=int, default=None, help="минимум голосов (по умолчанию большинство)")
    parser.add_argument("--sweep", action="store_true", help="перебрать все пороги и показать одиночные прогоны")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="куда записать предсказания ансамбля")
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    runs = load_runs(args.runs)
    hashes = [record["hash"] for record in gold]
    postprocess = not args.no_postprocess
    total = len(runs)
    print(f"прогонов в ансамбле: {total} ({', '.join(runs)})")

    if args.sweep:
        print("\n=== одиночные прогоны ===")
        singles = []
        for name in runs:
            single = {key: sorted(runs[name].get(key, ()), key=lambda s: s[0]) for key in hashes}
            metrics = evaluate(gold, to_records(gold, single, postprocess=postprocess))
            singles.append((name, metrics))
            show(name, metrics)
        best_name, best = max(singles, key=lambda item: item[1]["micro"]["f1"])

        print("\n=== голосование ===")
        results = []
        for threshold in range(1, total + 1):
            chosen = vote(runs, hashes, threshold)
            metrics = evaluate(gold, to_records(gold, chosen, postprocess=postprocess))
            results.append((threshold, metrics))
            show(f"порог >= {threshold}/{total}", metrics)
        top_threshold, top = max(results, key=lambda item: item[1]["micro"]["f1"])
        delta = top["micro"]["f1"] - best["micro"]["f1"]
        print(
            f"\nлучший одиночный: {best_name} = {best['micro']['f1']:.4f}"
            f"\nлучший ансамбль:  порог {top_threshold}/{total} = {top['micro']['f1']:.4f} ({delta:+.4f})"
        )
        threshold, metrics, chosen = top_threshold, top, vote(runs, hashes, top_threshold)
    else:
        threshold = args.threshold if args.threshold is not None else total // 2 + 1
        chosen = vote(runs, hashes, threshold)
        metrics = evaluate(gold, to_records(gold, chosen, postprocess=postprocess))
        show(f"порог >= {threshold}/{total}", metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in to_records(gold, chosen, postprocess=postprocess):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nпредсказания: {args.output}")
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["ensemble"] = {"runs": list(runs), "threshold": threshold}
        args.metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"метрики:      {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
