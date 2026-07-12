import pytest

from src.schemas import PredictionRequest


def test_prediction_request_strips_whitespace() -> None:
    request = PredictionRequest.from_value("  useful product  ")
    assert request.text == "useful product"


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_prediction_request_rejects_empty_text(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        PredictionRequest.from_value(value)


def test_prediction_request_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        PredictionRequest.from_value(123)
