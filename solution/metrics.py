"""Обёртка над официальным scorer кейса для использования внутри цикла обучения."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
ROOT = Path(__file__).resolve().parent.parent


def _load_scorer() -> Any:
    """Импортирует scripts/evaluate.py без копирования его логики."""

    spec = importlib.util.spec_from_file_location("case_evaluate", ROOT / "scripts" / "evaluate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("не удалось загрузить scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = _load_scorer()


def _keys(entities: list[JsonObject]) -> set[tuple[str, int, int]]:
    """Переводит список сущностей в множество ключей скорера."""

    return {(item["label"], int(item["start"]), int(item["end"])) for item in entities}


def evaluate(gold_records: list[JsonObject], predicted: list[JsonObject]) -> JsonObject:
    """Считает exact-span метрики тем же кодом, что и scripts/evaluate.py."""

    gold = {record["hash"]: {"entities": _keys(record["entities"])} for record in gold_records}
    predictions = {record["hash"]: _keys(record["entities"]) for record in predicted}
    missing = set(gold) - set(predictions)
    if missing:
        raise ValueError(f"нет предсказаний для {len(missing)} записей")
    return SCORER.calculate_metrics(gold, predictions)
