"""Application-facing sentiment prediction service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .model_loader import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_metadata,
    load_pipeline,
)
from .schemas import PredictionRequest, PredictionResponse


class SentimentPredictor:
    """Load the exported pipeline once and expose a stable prediction API."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self._pipeline = load_pipeline(model_path)
        self._metadata = load_metadata(metadata_path)
        self._classes = self._resolve_classes()

    def _resolve_classes(self) -> list[str]:
        classes: Any = getattr(self._pipeline, "classes_", None)

        if classes is None:
            classifier = getattr(self._pipeline, "named_steps", {}).get(
                "classifier"
            )
            classes = getattr(classifier, "classes_", None)

        if classes is None:
            raise RuntimeError(
                "The loaded pipeline does not expose fitted class labels."
            )

        return [str(label) for label in classes]

    def predict(self, text: str) -> dict[str, Any]:
        """Predict sentiment for one review and return a JSON-ready mapping."""
        request = PredictionRequest.from_value(text)

        predicted_label = str(self._pipeline.predict([request.text])[0])
        raw_probabilities = np.asarray(
            self._pipeline.predict_proba([request.text])[0],
            dtype=float,
        )

        if raw_probabilities.shape[0] != len(self._classes):
            raise RuntimeError(
                "The number of probabilities does not match the class labels."
            )

        probabilities = {
            label: round(float(probability), 6)
            for label, probability in zip(
                self._classes,
                raw_probabilities,
                strict=True,
            )
        }

        confidence = probabilities[predicted_label]
        
        artifact_metadata = self._metadata.get("artifact", {})
        model_version = artifact_metadata.get("model_version")

        response = PredictionResponse(
            sentiment=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=(
                str(model_version) if model_version is not None else None
            ),
        )

        return response.to_dict()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a defensive copy of the model metadata."""
        return dict(self._metadata)
