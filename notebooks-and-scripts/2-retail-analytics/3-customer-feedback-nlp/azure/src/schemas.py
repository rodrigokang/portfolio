"""Request and response objects used by the inference layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MAX_TEXT_LENGTH = 10_000


@dataclass(frozen=True)
class PredictionRequest:
    """Validated input for one sentiment prediction."""

    text: str

    @classmethod
    def from_value(cls, value: Any) -> "PredictionRequest":
        if not isinstance(value, str):
            raise TypeError("'text' must be a string.")

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("'text' must not be empty.")

        if len(cleaned_value) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"'text' must not exceed {MAX_TEXT_LENGTH:,} characters."
            )

        return cls(text=cleaned_value)


@dataclass(frozen=True)
class PredictionResponse:
    """Serializable sentiment prediction returned by the service."""

    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
