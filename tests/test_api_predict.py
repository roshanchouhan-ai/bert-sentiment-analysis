from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api import app, get_predictor

client = TestClient(app)


def test_predict_returns_prediction():
    """Test that a valid prediction request returns sentiment and confidence."""

    predictor = MagicMock()

    predictor.predict.return_value = {
        "sentiment": "Positive",
        "confidence": 0.99
    }

    app.dependency_overrides[get_predictor] = lambda: predictor

    try:
        response = client.post(
            "/predict",
            json={
                "text": "This movie was absolutely fantastic!"
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] == "Positive"
    assert 0 <= data["confidence"] <= 1


def test_predict_rejects_empty_text():
    """Test that an empty review is rejected by request validation."""

    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 422


def test_predict_rejects_whitespace_text():
    """Test that a whitespace-only review is rejected."""

    response = client.post(
        "/predict",
        json={
            "text": "   "
        }
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_text_type():
    """Test that a non-string review is rejected."""

    response = client.post(
        "/predict",
        json={
            "text": 123
        }
    )

    assert response.status_code == 422

def test_health_check():
    """Test that the health endpoint returns a healthy status."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}





def test_predict_handles_unexpected_error():
    """Test that an unexpected inference error returns HTTP 500."""

    predictor = MagicMock()

    predictor.predict.side_effect = RuntimeError(
        "Unexpected inference error"
    )

    app.dependency_overrides[get_predictor] = lambda: predictor

    try:
        response = client.post(
            "/predict",
            json={
                "text": "This movie was great!"
            }
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Internal server error"
    }




