import json
from unittest.mock import Mock

import azure.functions as func
import pytest

import function_app


def make_request(
    body: bytes,
    method: str = "POST",
) -> func.HttpRequest:
    return func.HttpRequest(
        method=method,
        url="http://localhost:7071/api/predict",
        headers={"content-type": "application/json"},
        params={},
        route_params={},
        body=body,
    )


def response_json(response: func.HttpResponse) -> dict:
    return json.loads(response.get_body().decode("utf-8"))


def test_predict_endpoint_returns_prediction(monkeypatch) -> None:
    fake_predictor = Mock()
    fake_predictor.predict.return_value = {
        "sentiment": "positive",
        "confidence": 0.9,
        "probabilities": {
            "negative": 0.02,
            "neutral": 0.08,
            "positive": 0.9,
        },
        "model_version": "1.0.0",
    }
    monkeypatch.setattr(function_app, "get_predictor", lambda: fake_predictor)

    response = function_app.predict_sentiment(
        make_request(json.dumps({"text": "Excellent product."}).encode())
    )

    assert response.status_code == 200
    assert response_json(response)["sentiment"] == "positive"
    fake_predictor.predict.assert_called_once_with("Excellent product.")


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (b"not-json", "invalid_json"),
        (b"[]", "invalid_request"),
        (b"{}", "validation_error"),
        (b'{"text": "   "}', "validation_error"),
    ],
)
def test_predict_endpoint_rejects_invalid_requests(
    body: bytes,
    expected_code: str,
) -> None:
    response = function_app.predict_sentiment(make_request(body))

    assert response.status_code == 400
    assert response_json(response)["error"]["code"] == expected_code


def test_health_endpoint_reports_model_version(monkeypatch) -> None:
    fake_predictor = Mock()
    fake_predictor.metadata = {
        "artifact": {
            "model_version": "1.0.0",
        }
    }
    monkeypatch.setattr(function_app, "get_predictor", lambda: fake_predictor)

    request = func.HttpRequest(
        method="GET",
        url="http://localhost:7071/api/health",
        headers={},
        params={},
        route_params={},
        body=b"",
    )
    response = function_app.health_check(request)

    assert response.status_code == 200
    assert response_json(response) == {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.0.0",
    }
