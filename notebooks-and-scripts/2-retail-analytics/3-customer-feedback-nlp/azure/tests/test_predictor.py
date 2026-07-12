from unittest.mock import Mock

import numpy as np

from src.predictor import SentimentPredictor


def test_predictor_returns_json_ready_response(monkeypatch) -> None:
    fake_pipeline = Mock()
    fake_pipeline.classes_ = np.array(
        ["negative", "neutral", "positive"]
    )
    fake_pipeline.predict.return_value = np.array(["positive"])
    fake_pipeline.predict_proba.return_value = np.array(
        [[0.02, 0.08, 0.90]]
    )

    monkeypatch.setattr(
        "src.predictor.load_pipeline",
        lambda model_path: fake_pipeline,
    )
    monkeypatch.setattr(
        "src.predictor.load_metadata",
        lambda metadata_path: {
            "artifact": {
                "model_version": "1.0.0"
            }
        },
    )

    predictor = SentimentPredictor()
    result = predictor.predict("Excellent product.")

    assert result == {
        "sentiment": "positive",
        "confidence": 0.9,
        "probabilities": {
            "negative": 0.02,
            "neutral": 0.08,
            "positive": 0.9,
        },
        "model_version": "1.0.0",
    }