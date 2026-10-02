from __future__ import annotations

from pathlib import Path
import sys
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


def batch_preprocess(texts):
  if isinstance(texts, str):
    return [texts]
  return list(texts)


sys.modules["__main__"].batch_preprocess = batch_preprocess

app = FastAPI(
    title="Airline Sentiment Classification API",
    description=(
        "Production-grade NLP inference service built for Project 5 internship"
        " deliverable."
    ),
    version="1.0.0",
)

# Load the trained model artifact
MODEL_PATH = Path(__file__).parent.parent / "day3" / "artifacts" / "baseline.joblib"
if not MODEL_PATH.exists():
  MODEL_PATH = Path(__file__).parent.parent / "baseline_tfidf.joblib"

try:
  model = joblib.load(MODEL_PATH)
  print(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
  model = None
  print(f"Warning: Could not load model artifact: {e}")


class SentimentRequest(BaseModel):
  text: str = Field(
      ...,
      min_length=2,
      max_length=1000,
      description="Customer review text to classify.",
  )


class SentimentResponse(BaseModel):
  text: str
  sentiment: str
  status: str = "success"


@app.get("/")
def root():
  return {
      "message": "Welcome to the Airline Sentiment API!",
      "status": "healthy",
      "docs_url": "/docs",
  }


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(payload: SentimentRequest):
  if model is None:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model artifact is not loaded on the server.",
    )

  try:
    prediction = model.predict([payload.text])[0]
    return SentimentResponse(text=payload.text, sentiment=str(prediction))
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Inference failed: {str(e)}",
    )