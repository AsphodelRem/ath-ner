"""Offline predictor loading and model inference for the FastAPI service."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128
DEFAULT_BATCH_SIZE = 16

JsonObject = dict[str, Any]


class Predictor(Protocol):
    """Small interface used by the HTTP layer and tests."""

    def predict(self, records: list[JsonObject]) -> list[JsonObject]:
        """Return one result per input record in the same order."""


class EmptyPredictor:
    """Contract-compatible fallback used when no checkpoint is packaged.

    The repository intentionally does not contain trained weights. Keeping this
    fallback makes the API and container testable before a checkpoint is copied
    into ``artifacts/model``. It is not intended for quality evaluation.
    """

    def predict(self, records: list[JsonObject]) -> list[JsonObject]:
        return [{"hash": item["hash"], "entities": []} for item in records]


class TransformerPredictor:
    """Thread-safe wrapper around the existing sliding-window inference code."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        from solution.tagging import tags_for

        self._torch = torch
        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._run_config = _read_run_config(model_dir)
        self._device = _resolve_device(torch)
        self._max_length = int(self._run_config.get("max_length", DEFAULT_MAX_LENGTH))
        self._stride = int(self._run_config.get("stride", DEFAULT_STRIDE))
        self._batch_size = _positive_int_env("NER_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        self._postprocess = bool(self._run_config.get("postprocess", True))

        trust_remote_code = bool(self._run_config.get("trust_remote_code", False))
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        if not self._tokenizer.is_fast:
            raise ValueError("checkpoint must contain a fast tokenizer with offset mappings")
        _validate_window(self._tokenizer, self._max_length, self._stride)

        self._model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()

        self._id2label = {
            int(index): str(label) for index, label in self._model.config.id2label.items()
        }
        configured_scheme = self._run_config.get("tag_scheme")
        if configured_scheme in {"bio", "bilou"}:
            self._scheme = str(configured_scheme)
        else:
            self._scheme = (
                "bilou"
                if any(label.startswith(("L-", "U-")) for label in self._id2label.values())
                else "bio"
            )
        expected_tags = set(tags_for(self._scheme))
        if set(self._id2label) != set(range(len(expected_tags))):
            raise ValueError("checkpoint label ids must be contiguous and start at zero")
        if set(self._id2label.values()) != expected_tags:
            raise ValueError(
                f"checkpoint labels do not match the {self._scheme.upper()} NER schema"
            )

        self._log_transitions = self._load_transitions()
        LOGGER.info(
            "Loaded checkpoint %s on %s (scheme=%s, window=%s/%s)",
            model_dir,
            self._device,
            self._scheme,
            self._max_length,
            self._stride,
        )

    def _load_transitions(self) -> Any:
        if not self._run_config.get("viterbi", False):
            return None
        transitions_path = self._model_dir.parent / "transitions.json"
        if not transitions_path.is_file():
            raise FileNotFoundError(
                f"Viterbi is enabled but transitions are missing: {transitions_path}"
            )
        from solution.viterbi import load_transitions

        mode = str(self._run_config.get("transition_mode", "conditional"))
        transitions, tags = load_transitions(transitions_path, mode=mode)
        if tags != [self._id2label[index] for index in range(len(self._id2label))]:
            raise ValueError("transition tags do not match checkpoint labels")
        weight = float(self._run_config.get("transition_weight", 0.25))
        return transitions * weight

    def predict(self, records: list[JsonObject]) -> list[JsonObject]:
        from solution.inference import predict_records

        # A single lock prevents concurrent requests from overcommitting GPU RAM.
        with self._lock:
            return predict_records(
                self._model,
                self._tokenizer,
                records,
                max_length=self._max_length,
                stride=self._stride,
                batch_size=self._batch_size,
                device=self._device,
                id2label=self._id2label,
                scheme=self._scheme,
                postprocess=self._postprocess,
                progress=False,
                log_transitions=self._log_transitions,
            )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_device(torch: Any) -> Any:
    requested = os.getenv("NER_DEVICE", "auto")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("NER_DEVICE must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NER_DEVICE=cuda, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _validate_window(tokenizer: Any, max_length: int, stride: int) -> None:
    content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if content_length < 1:
        raise ValueError("max_length is too small for tokenizer special tokens")
    if not 0 <= stride < content_length:
        raise ValueError(f"stride must be between 0 and {content_length - 1}")


def _read_run_config(model_dir: Path) -> JsonObject:
    path = model_dir.parent / "run_config.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _find_model_dir(root: Path) -> Path | None:
    explicit = os.getenv("NER_MODEL_DIR")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"NER_MODEL_DIR is not a checkpoint: {path}")
        return path

    preferred = root / "artifacts" / "model"
    if (preferred / "config.json").is_file():
        return preferred

    candidates = sorted((root / "artifacts").glob("*/model/config.json"))
    if not candidates:
        return None

    def score(config_path: Path) -> float:
        run_config = _read_run_config(config_path.parent)
        return float(run_config.get("best_micro_f1", -1.0))

    return max(candidates, key=score).parent


def create_predictor(root: Path | None = None) -> Predictor:
    """Load the best packaged local checkpoint, without any network access."""

    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    model_dir = _find_model_dir(project_root)
    if model_dir is None:
        LOGGER.warning(
            "No checkpoint found under %s/artifacts; API starts in empty fallback mode",
            project_root,
        )
        return EmptyPredictor()
    return TransformerPredictor(model_dir)
