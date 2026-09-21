from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api import app, get_predictor


client = TestClient(app)


def test_predict_csv_returns_predictions():
    """Test that a valid CSV upload returns a predictions CSV."""

    csv_content = (
        "review\n"
        "This movie was fantastic!\n"
        "This movie was terrible!\n"
    )

    predictor = MagicMock()

    predictor.predict_batch.return_value = [
        {
            "sentiment": "Positive",
            "confidence": 0.99
        },
        {
            "sentiment": "Negative",
            "confidence": 0.98
        }
    ]

    app.dependency_overrides[get_predictor] = lambda: predictor

    try:
        response = client.post(
            "/predict/csv",
            files={
                "file": (
                    "reviews.csv",
                    csv_content,
                    "text/csv"
                )
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    assert "predicted_sentiment" in response.text
    assert "confidence" in response.text
    assert "Positive" in response.text
    assert "Negative" in response.text


def test_predict_csv_rejects_missing_review_column():
    """Test that a CSV without a review column returns HTTP 400."""

    csv_content = (
        "text\n"
        "This movie was fantastic!\n"
    )

    predictor = MagicMock()

    app.dependency_overrides[get_predictor] = lambda: predictor

    try:
        response = client.post(
            "/predict/csv",
            files={
                "file": (
                    "invalid.csv",
                    csv_content,
                    "text/csv"
                )
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400

    assert response.json() == {
        "detail": "CSV must contain a 'review' column."
    }


def test_predict_csv_handles_unexpected_error():
    """Test that an unexpected inference error returns HTTP 500."""

    csv_content = (
        "review\n"
        "This movie was fantastic!\n"
    )

    predictor = MagicMock()

    predictor.predict_batch.side_effect = RuntimeError(
        "Unexpected inference error"
    )

    app.dependency_overrides[get_predictor] = lambda: predictor

    try:
        response = client.post(
            "/predict/csv",
            files={
                "file": (
                    "reviews.csv",
                    csv_content,
                    "text/csv"
                )
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal server error"
    }