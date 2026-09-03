"""Логирование метрик обучения в Comet ML.

Три режима, задаются флагом --comet:
  off      — ничего не логируем (по умолчанию);
  online   — отправка на comet.com, нужен COMET_API_KEY;
  offline  — всё пишется в локальный .zip, сеть не используется вовсе.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  
    import comet_ml

    COMET_AVAILABLE = True
except ImportError:  # pragma: no cover 
    comet_ml = None
    COMET_AVAILABLE = False

JsonObject = dict[str, Any]
MODES = ("off", "online", "offline")


class Tracker:
    """Заглушка: единый интерфейс, когда логирование выключено."""

    enabled = False

    def log_parameters(self, parameters: JsonObject) -> None:
        """Записывает гиперпараметры запуска."""

    def log_metrics(self, metrics: JsonObject, *, step: int | None = None, epoch: int | None = None) -> None:
        """Записывает набор метрик на шаге или эпохе."""

    def log_asset(self, path: Path) -> None:
        """Прикладывает файл к эксперименту."""

    def set_name(self, name: str) -> None:
        """Задаёт человекочитаемое имя запуска."""

    def end(self) -> str | None:
        """Завершает эксперимент и возвращает ссылку либо путь к архиву."""

        return None


class CometTracker(Tracker):
    """Обёртка над comet_ml.Experiment / OfflineExperiment."""

    enabled = True

    def __init__(self, experiment: Any, *, offline_dir: Path | None) -> None:
        self._experiment = experiment
        self._offline_dir = offline_dir

    def log_parameters(self, parameters: JsonObject) -> None:
        """Записывает гиперпараметры запуска."""

        self._experiment.log_parameters(parameters)

    def log_metrics(self, metrics: JsonObject, *, step: int | None = None, epoch: int | None = None) -> None:
        """Записывает набор метрик на шаге или эпохе."""

        self._experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_asset(self, path: Path) -> None:
        """Прикладывает файл к эксперименту."""

        if path.exists():
            self._experiment.log_asset(str(path), file_name=path.name)

    def set_name(self, name: str) -> None:
        """Задаёт человекочитаемое имя запуска."""

        self._experiment.set_name(name)

    def end(self) -> str | None:
        """Завершает эксперимент и возвращает ссылку либо путь к архиву."""

        self._experiment.end()
        if self._offline_dir is not None:
            archives = sorted(self._offline_dir.glob("*.zip"), key=lambda item: item.stat().st_mtime)
            return str(archives[-1]) if archives else str(self._offline_dir)
        try:
            return self._experiment.url
        except AttributeError:  # pragma: no cover - зависит от версии SDK
            return None


def load_env_file(path: Path) -> list[str]:
    """Подхватывает KEY=VALUE из .env; уже заданные переменные не перетирает.

    comet_ml сам .env не читает, поэтому разбираем его вручную и без
    дополнительных зависимостей.
    """

    if not path.exists():
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def check_mode(mode: str) -> None:
    """Проверяет режим и доступность учётных данных до начала обучения."""

    if mode not in MODES:
        raise ValueError(f"неизвестный режим comet: {mode}")
    if mode == "off":
        return
    if not COMET_AVAILABLE:
        raise SystemExit(
            "comet_ml не установлен. Поставьте его (pip install comet_ml) или используйте --comet off"
        )
    if mode == "online" and not (os.environ.get("COMET_API_KEY") or os.environ.get("COMET_CONFIG")):
        raise SystemExit(
            "для --comet online нужен COMET_API_KEY. Положите его в один из вариантов:\n"
            "  1) ~/.comet.config   ->  [comet]\n"
            "                           api_key = <ключ>\n"
            "  2) .env в корне проекта  ->  COMET_API_KEY=<ключ>\n"
            "  3) export COMET_API_KEY=<ключ> в ~/.bashrc\n"
            "Либо используйте --comet offline: он не ходит в сеть, а архив\n"
            "потом загружается командой comet upload."
        )


def create_tracker(
    mode: str,
    *,
    project: str,
    workspace: str | None = None,
    offline_dir: Path | None = None,
    tags: list[str] | None = None,
) -> Tracker:
    """Создаёт трекер выбранного режима; при mode='off' возвращает заглушку."""

    if mode not in MODES:
        raise ValueError(f"неизвестный режим comet: {mode}")
    if mode == "off":
        return Tracker()
    if not COMET_AVAILABLE:
        raise SystemExit(
            "comet_ml не установлен. Поставьте его (pip install comet_ml) или используйте --comet off"
        )

    common = dict(
        project_name=project,
        workspace=workspace,
        # Логируем метрики руками, чтобы не мешать автологгеру transformers.
        auto_metric_logging=False,
        auto_param_logging=False,
        auto_output_logging="simple",
        parse_args=False,
        log_code=False,
        log_graph=False,
    )

    if mode == "offline":
        directory = offline_dir or Path("artifacts/comet")
        directory.mkdir(parents=True, exist_ok=True)
        experiment = comet_ml.OfflineExperiment(
            offline_directory=str(directory),
            # В офлайне сведения о среде собираются локально и уходят в архив.
            log_env_details=True,
            log_env_gpu=True,
            log_env_host=False,
            **common,
        )
        tracker = CometTracker(experiment, offline_dir=directory)
    else:
        if not (os.environ.get("COMET_API_KEY") or os.environ.get("COMET_CONFIG")):
            raise SystemExit(
                "для --comet online нужен COMET_API_KEY в окружении "
                "(или используйте --comet offline, он не ходит в сеть)"
            )
        experiment = comet_ml.Experiment(log_env_details=True, **common)
        tracker = CometTracker(experiment, offline_dir=None)

    if tags:
        experiment.add_tags(tags)
    return tracker


def flatten_metrics(metrics: JsonObject) -> JsonObject:
    """Разворачивает отчёт scorer в плоский набор скаляров для графиков."""

    flat: JsonObject = {}
    for scope in ("micro", "macro"):
        for key, value in metrics.get(scope, {}).items():
            if isinstance(value, (int, float)):
                flat[f"dev_{scope}_{key}"] = value
    for label, values in metrics.get("by_label", {}).items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                flat[f"dev_{label}_{key}"] = value
    return flat
