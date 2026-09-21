import torch
from unittest.mock import patch

import pytest
from config.settings import MODEL_PATH
from src.inference import SentimentPredictor

@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_batch_returns_predictions(mock_model, mock_tokenizer):
    """Test that predict_batch returns predictions for valid reviews."""

    mock_tokenizer_instance = mock_tokenizer.return_value

    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([
            [101, 2023, 102],
            [101, 2024, 102],
            [101, 2025, 102]
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
    }

    mock_model_instance = mock_model.return_value

    mock_model_instance.eval.return_value = None
    mock_model_instance.to.return_value = mock_model_instance

    mock_outputs = mock_model_instance.return_value

    mock_outputs.logits = torch.tensor([
        [0.1, 3.0],
        [3.0, 0.1],
        [0.1, 3.0]
    ])

    predictor = SentimentPredictor(MODEL_PATH)

    reviews = [
        "This movie was fantastic!",
        "This movie was terrible!",
        "I really enjoyed this film."
    ]

    results = predictor.predict_batch(reviews)

    assert len(results) == 3

    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"
    assert results[2]["sentiment"] == "Positive"

    for result in results:
        assert "sentiment" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1



@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_batch_rejects_non_list(mock_model, mock_tokenizer):
    """Test that predict_batch rejects non-list input."""

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(TypeError):
        predictor.predict_batch("This is not a list")



@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_batch_rejects_empty_list(mock_model, mock_tokenizer):
    """Test that predict_batch rejects an empty list."""

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(ValueError):
        predictor.predict_batch([])



@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_batch_rejects_non_string_review(mock_model,mock_tokenizer):
    """Test that predict_batch rejects non-string reviews."""

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(TypeError):
        predictor.predict_batch([
            "This movie was great!",
            123
        ])



@patch("src.inference.AutoTokenizer.from_pretrained")
@patch("src.inference.AutoModelForSequenceClassification.from_pretrained")
def test_predict_batch_rejects_empty_review(mock_model,mock_tokenizer):
    """Test that predict_batch rejects empty reviews."""

    predictor = SentimentPredictor(MODEL_PATH)

    with pytest.raises(ValueError):
        predictor.predict_batch([
            "This movie was great!",
            ""
        ])