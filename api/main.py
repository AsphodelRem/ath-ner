"""FastAPI application implementing the required Uzbek NER contract."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException

from api.runtime import Predictor, create_predictor
from api.schemas import HealthResponse, PredictItem, PredictResponse

LOGGER = logging.getLogger(__name__)


def _validate_predictions(
    predictions: Any,
    items: list[PredictItem],
) -> list[dict[str, Any]]:
    """Protect the public contract from malformed model output."""

    if not isinstance(predictions, list) or len(predictions) != len(items):
        raise RuntimeError("predictor returned a result count different from the request")

    validated: list[dict[str, Any]] = []
    for index, (prediction, item) in enumerate(zip(predictions, items, strict=True)):
        if not isinstance(prediction, dict) or prediction.get("hash") != item.hash:
            raise RuntimeError(f"predictor changed hash or order at item {index}")
        entities = prediction.get("entities")
        if not isinstance(entities, list):
            raise RuntimeError(f"predictor returned invalid entities at item {index}")

        seen: set[tuple[str, int, int]] = set()
        clean_entities: list[dict[str, Any]] = []
        for entity_index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise RuntimeError(f"predictor returned an invalid entity at item {index}")
            label = entity.get("label")
            start = entity.get("start")
            end = entity.get("end")
            if label not in {"ORG", "NAME", "GEO"}:
                raise RuntimeError(f"predictor returned an invalid label at item {index}")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not 0 <= start < end <= len(item.text)
            ):
                raise RuntimeError(
                    f"predictor returned invalid offsets at item {index}, entity {entity_index}"
                )
            key = (label, start, end)
            if key in seen:
                raise RuntimeError(f"predictor returned a duplicate entity at item {index}")
            seen.add(key)
            clean_entities.append({"label": label, "start": start, "end": end})
        validated.append({"hash": item.hash, "entities": clean_entities})
    return validated


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Build the app; dependency injection keeps contract tests lightweight."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor or create_predictor()
        yield

    app = FastAPI(
        title="Uzbek Exact-Span NER API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Return readiness without performing inference."""

        return HealthResponse()

    @app.post("/api/v1/predict", response_model=PredictResponse)
    def predict(items: list[PredictItem]) -> PredictResponse:
        """Run exact-span NER for a non-empty ordered batch."""

        if not items:
            raise HTTPException(status_code=422, detail="request batch must not be empty")
        hashes = [item.hash for item in items]
        if len(set(hashes)) != len(hashes):
            raise HTTPException(
                status_code=422,
                detail="hash must be unique within the request batch",
            )

        records = [{"hash": item.hash, "text": item.text} for item in items]
        try:
            predictions = app.state.predictor.predict(records)
            clean = _validate_predictions(predictions, items)
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="prediction failed") from error
        return PredictResponse(data=clean)

    return app


app = create_app()
