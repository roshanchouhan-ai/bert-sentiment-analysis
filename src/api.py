from functools import lru_cache

import logging

from src.logging_config import setup_logging

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator

from fastapi import UploadFile, File
import tempfile
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from io import StringIO


from src.inference import SentimentPredictor
from src.batch_inference import predict_csv
from config.settings import MODEL_PATH

setup_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="API for sentiment prediction using fine-tuned BERT.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Input text cannot be empty or whitespace")
        return value
    


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float


@lru_cache
def get_predictor():
    return SentimentPredictor(MODEL_PATH)


@app.post("/predict", response_model = PredictionResponse)
def predict(request: PredictionRequest, 
    predictor: SentimentPredictor = Depends(get_predictor)):

    logger.info("Prediction request received")

    try:
        result = predictor.predict(request.text)

        logger.info(
            "Prediction completed: %s",
            result["sentiment"]
        )

        return result
        
    
    except Exception:
        logger.exception("Unexpected error during prediction")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/health")
def health_check():
    logger.info("Health check requested")

    return {
        "status": "healthy"
    }


@app.post("/predict/csv")
def predict_csv_endpoint(
    file: UploadFile = File(...),
    predictor: SentimentPredictor = Depends(get_predictor)
):
    logger.info(
        "CSV prediction request received: %s",
        file.filename
    )

    try:
        df = predict_csv(file.file,predictor)

        csv_buffer = StringIO()

        df.to_csv(csv_buffer,index=False)

        csv_buffer.seek(0)

        logger.info(
            "CSV prediction completed: %s",
            file.filename
        )

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": ("attachment; filename=predictions.csv")}
        )

    except ValueError as exc:
        logger.error("Invalid CSV file: %s",exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        logger.exception("Unexpected error during CSV prediction")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )