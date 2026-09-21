import pandas as pd
import logging

logger = logging.getLogger(__name__)


def predict_csv(input_file, predictor):
    try:
        logger.info("Reading input CSV")

        df = pd.read_csv(input_file)

        logger.info("Loaded %d reviews", len(df))

    except Exception:
        logger.exception("Failed to read input CSV")
        raise

    if "review" not in df.columns:
        logger.error("Input CSV must contain a 'review' column")
        raise ValueError("CSV must contain a 'review' column.")

    logger.info("Starting batch prediction")

    results = predictor.predict_batch(
        df["review"].tolist()
    )

    df["predicted_sentiment"] = [
        result["sentiment"]
        for result in results
    ]

    df["confidence"] = [
        result["confidence"]
        for result in results
    ]

    logger.info("Batch prediction completed")

    return df