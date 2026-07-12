"""Utilities for loading the exported sentiment model and its metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS_DIR / "sentiment_pipeline.joblib"
DEFAULT_METADATA_PATH = (
    DEFAULT_ARTIFACTS_DIR / "sentiment_pipeline_metadata.json"
)


class ArtifactLoadError(RuntimeError):
    """Raised when a required production artifact cannot be loaded."""


def _require_file(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()

    if not resolved_path.is_file():
        raise ArtifactLoadError(
            f"{description} was not found at: {resolved_path}. "
            "Run the Python notebook export section first."
        )

    return resolved_path


@lru_cache(maxsize=1)
def load_pipeline(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load and cache the fitted scikit-learn sentiment pipeline."""
    path = _require_file(Path(model_path), "Sentiment pipeline")

    try:
        pipeline = joblib.load(path)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Could not load the sentiment pipeline from: {path}"
        ) from exc

    required_methods = ("predict", "predict_proba")
    missing_methods = [
        method for method in required_methods if not hasattr(pipeline, method)
    ]

    if missing_methods:
        raise ArtifactLoadError(
            "The loaded artifact is not a compatible classification pipeline. "
            f"Missing methods: {', '.join(missing_methods)}."
        )

    return pipeline


@lru_cache(maxsize=1)
def load_metadata(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Load and cache the JSON metadata associated with the model."""
    path = _require_file(Path(metadata_path), "Model metadata")

    try:
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(
            f"Could not load valid model metadata from: {path}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ArtifactLoadError("Model metadata must contain a JSON object.")

    return metadata
