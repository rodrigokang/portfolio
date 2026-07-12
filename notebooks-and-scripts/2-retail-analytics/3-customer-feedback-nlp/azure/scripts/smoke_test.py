"""Run one local inference without starting an HTTP service."""

from __future__ import annotations

import json
import sys
from pathlib import Path


AZURE_DIR = Path(__file__).resolve().parents[1]
if str(AZURE_DIR) not in sys.path:
    sys.path.insert(0, str(AZURE_DIR))

from src import SentimentPredictor  # noqa: E402


def main() -> None:
    predictor = SentimentPredictor()
    result = predictor.predict(
        "The product arrived on time and works exactly as expected."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
