"""Azure Functions HTTP entry points for sentiment inference."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import azure.functions as func

from src import SentimentPredictor
from src.model_loader import ArtifactLoadError

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    """Create the predictor once per worker process."""
    return SentimentPredictor()


def json_response(
    payload: dict[str, Any],
    status_code: int = 200,
) -> func.HttpResponse:
    """Build a consistent UTF-8 JSON response."""
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
        charset="utf-8",
    )


@app.function_name(name="PredictSentiment")
@app.route(route="predict", methods=["POST"])
def predict_sentiment(req: func.HttpRequest) -> func.HttpResponse:
    """Predict sentiment from a JSON body containing a non-empty `text` field."""
    try:
        body = req.get_json()
    except ValueError:
        return json_response(
            {
                "error": {
                    "code": "invalid_json",
                    "message": "Request body must contain valid JSON.",
                }
            },
            status_code=400,
        )

    if not isinstance(body, dict):
        return json_response(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Request body must be a JSON object.",
                }
            },
            status_code=400,
        )

    try:
        result = get_predictor().predict(body.get("text"))
    except (TypeError, ValueError) as exc:
        return json_response(
            {
                "error": {
                    "code": "validation_error",
                    "message": str(exc),
                }
            },
            status_code=400,
        )
    except ArtifactLoadError:
        logging.exception("The model artifacts could not be loaded.")
        return json_response(
            {
                "error": {
                    "code": "model_unavailable",
                    "message": "The sentiment model is unavailable.",
                }
            },
            status_code=503,
        )
    except Exception:
        logging.exception("Unexpected sentiment inference failure.")
        return json_response(
            {
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected inference error occurred.",
                }
            },
            status_code=500,
        )

    return json_response(result)


@app.function_name(name="HealthCheck")
@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Report whether the function process can load the model artifacts."""
    try:
        predictor = get_predictor()
        model_version = (
            predictor.metadata.get("artifact", {}).get("model_version")
        )
    except Exception:
        logging.exception("Health check failed while loading model artifacts.")
        return json_response(
            {
                "status": "unhealthy",
                "model_loaded": False,
            },
            status_code=503,
        )

    return json_response(
        {
            "status": "healthy",
            "model_loaded": True,
            "model_version": model_version,
        }
    )
