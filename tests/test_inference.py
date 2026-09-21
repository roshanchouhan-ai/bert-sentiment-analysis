from unittest.mock import patch

import torch
import pytest
from config.settings import MODEL_PATH
from src.inference import SentimentPredictor

@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")


def test_predict_rejects_non_string(mock_model, mock_tokenizer):
    """Test that predict rejects non-string input."""

    mock_model.return_value.to.return_value.eval.return_value = None
    mock_tokenizer.return_value = None

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(TypeError):
        predictor.predict(123)


@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_rejects_empty_text(mock_model, mock_tokenizer):
    """Test that predict rejects empty input."""

    mock_model.return_value.to.return_value.eval.return_value = None
    mock_tokenizer.return_value = None

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(ValueError):
        predictor.predict("")


@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_returns_sentiment_and_confidence(mock_model, mock_tokenizer):
    """Test that predict returns sentiment and confidence for valid input."""

    mock_tokenizer_instance = mock_tokenizer.return_value

    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    mock_model_instance = mock_model.return_value

    mock_model_instance.eval.return_value = None
    mock_model_instance.to.return_value = mock_model_instance

    mock_outputs = mock_model_instance.return_value

    mock_outputs.logits = torch.tensor(
        [[0.1, 3.0]]
    )

    predictor = SentimentPredictor(MODEL_PATH)

    result = predictor.predict(
        "This movie was absolutely fantastic!"
    )

    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"] == "Positive"
    assert 0 <= result["confidence"] <= 1


@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_returns_negative_sentiment(mock_model,mock_tokenizer):
    """Test that predict correctly handles a negative prediction."""

    mock_tokenizer_instance = mock_tokenizer.return_value

    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    mock_model_instance = mock_model.return_value

    mock_model_instance.eval.return_value = None
    mock_model_instance.to.return_value = mock_model_instance

    mock_outputs = mock_model_instance.return_value

    mock_outputs.logits = torch.tensor(
        [[3.0, 0.1]]
    )

    predictor = SentimentPredictor(MODEL_PATH)

    result = predictor.predict(
        "This movie was terrible and boring."
    )

    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"] == "Negative"
    assert 0 <= result["confidence"] <= 1