"""Pydantic schemas for the public HTTP contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class PredictItem(BaseModel):
    """One document in a prediction batch."""

    model_config = ConfigDict(extra="ignore")

    hash: StrictStr = Field(min_length=1)
    text: StrictStr


class Entity(BaseModel):
    """An exact, half-open character span in the source text."""

    label: Literal["ORG", "NAME", "GEO"]
    start: int
    end: int


class Prediction(BaseModel):
    """NER result for one input document."""

    hash: str
    entities: list[Entity]


class PredictResponse(BaseModel):
    """Envelope required by API.md."""

    data: list[Prediction]


class HealthResponse(BaseModel):
    """Lightweight service readiness response."""

    status: Literal["ok"] = "ok"
