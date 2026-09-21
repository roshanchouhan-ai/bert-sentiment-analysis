import pandas as pd
import pytest

from src.batch_inference import predict_csv

class MockPredictor:
    pass


def test_predict_csv(tmp_path):
    """Test that predict_csv processes a valid CSV correctly."""

    input_file = tmp_path / "input.csv"

    df = pd.DataFrame({
        "review": [
            "This movie was fantastic!",
            "This movie was terrible!"
        ]
    })

    df.to_csv(input_file, index=False)

    class MockPredictor:

        def predict_batch(self, reviews):
            return [
                {"sentiment": "Positive", "confidence": 0.99},
                {"sentiment": "Negative", "confidence": 0.98}
            ]
    
    predictor = MockPredictor()

    result = predict_csv(
        input_file,
        predictor
    )

    assert "predicted_sentiment" in result.columns
    assert "confidence" in result.columns

    assert result["predicted_sentiment"].tolist() == [
        "Positive",
        "Negative"
    ]

    assert result["confidence"].tolist() == [
        0.99,
        0.98
    ]



def test_predict_csv_file_not_found(tmp_path):
    """Test that predict_csv raises an error when the input file is missing."""

    input_file = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):

        predict_csv(
            input_file,
            MockPredictor()
        )



def test_predict_csv_missing_review_column(tmp_path):
    """Test that predict_csv raises an error when the review column is missing."""

    input_file = tmp_path / "invalid.csv"

    df = pd.DataFrame({
        "text": [
            "This movie was great!"
        ]
    })

    df.to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="review"):

        predict_csv(
            input_file,
            MockPredictor()
        )