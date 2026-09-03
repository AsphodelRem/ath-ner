"""Сводная таблица по каталогам прогонов.

Собирает run_config.json и dev_metrics.json из artifacts/* и печатает
сравнение конфигураций с метриками. Помогает не сравнивать шесть прогонов
глазами по логам.

    python tools/summarize_runs.py artifacts
    python tools/summarize_runs.py artifacts --csv artifacts/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

COLUMNS = (
    ("run", "прогон"),
    ("model", "модель"),
    ("tag_scheme", "теги"),
    ("window", "окно"),
    ("viterbi", "витерби"),
    ("epochs", "эпох"),
    ("best_epoch", "лучшая"),
    ("micro_f1", "micro F1"),
    ("precision", "P"),
    ("recall", "R"),
    ("ORG", "ORG"),
    ("NAME", "NAME"),
    ("GEO", "GEO"),
    ("minutes", "минут"),
)


def collect(root: Path) -> list[JsonObject]:
    """Читает все завершённые прогоны в каталоге."""

    rows: list[JsonObject] = []
    for directory in sorted(root.iterdir()):
        metrics_path = directory / "dev_metrics.json"
        if not directory.is_dir() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = directory / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        weight = config.get("transition_weight")
        rows.append({
            "run": directory.name,
            "model": str(config.get("model_name", "?")).split("/")[-1],
            "tag_scheme": config.get("tag_scheme", "?"),
            "window": f"{config.get('max_length', '?')}/{config.get('stride', '?')}",
            "viterbi": f"w={weight}" if config.get("viterbi") else "нет",
            "epochs": config.get("epochs", "?"),
            "best_epoch": config.get("best_epoch", "?"),
            "micro_f1": metrics["micro"]["f1"],
            "precision": metrics["micro"]["precision"],
            "recall": metrics["micro"]["recall"],
            "ORG": metrics["by_label"]["ORG"]["f1"],
            "NAME": metrics["by_label"]["NAME"]["f1"],
            "GEO": metrics["by_label"]["GEO"]["f1"],
            "minutes": round(config.get("total_seconds", 0) / 60) if config.get("total_seconds") else "?",
        })
    return sorted(rows, key=lambda row: -row["micro_f1"])


def render(rows: list[JsonObject]) -> None:
    """Печатает таблицу, отсортированную по micro F1."""

    if not rows:
        print("завершённых прогонов не найдено")
        return
    widths = {key: max(len(title), *(len(_fmt(row[key])) for row in rows)) for key, title in COLUMNS}
    header = "  ".join(title.ljust(widths[key]) for key, title in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(_fmt(row[key]).ljust(widths[key]) for key, _ in COLUMNS))
    best = rows[0]
    print(f"\nлучший: {best['run']} -> micro F1 {best['micro_f1']:.4f}")


def _fmt(value: Any) -> str:
    """Форматирует ячейку таблицы."""

    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    """Точка входа CLI."""

    parser = argparse.ArgumentParser(description="Summarize training runs.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts"))
    parser.add_argument("--csv", type=Path, help="дополнительно выгрузить таблицу в CSV")
    args = parser.parse_args()

    rows = collect(args.root)
    render(rows)
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[key for key, _ in COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
