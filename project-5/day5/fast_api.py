from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from statistics import fmean
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field


def batch_preprocess(texts):
    from text_preprocessing import preprocess

    if isinstance(texts, str):
        texts = [texts]
    return [preprocess(text) for text in texts]


# Older joblib artifacts refer to this helper as __main__.batch_preprocess.
sys.modules["__main__"].batch_preprocess = batch_preprocess

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_DIR / "day3/artifacts/baseline.joblib"))
BASELINE_PATH = Path(os.getenv("DRIFT_BASELINE_PATH", APP_DIR / "drift_baseline.json"))
WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_Z_THRESHOLD", "0.5"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("sentiment_api")


def text_features(text: str) -> dict[str, float]:
    """Small, label-independent text features used for input drift checks."""
    words = text.split()
    return {
        "char_count": float(len(text)),
        "word_count": float(len(words)),
        "punctuation_ratio": sum(not char.isalnum() and not char.isspace() for char in text)
        / max(len(text), 1),
    }


def predict_sentiment(model, text: str) -> str:
    """Run one prediction and normalize the classifier's label to a string."""
    prediction = model.predict([text])
    if len(prediction) != 1:
        raise ValueError("The classifier must return exactly one prediction.")
    return str(prediction[0])


def load_baseline(path: Path) -> dict:
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        for feature in ("char_count", "word_count", "punctuation_ratio"):
            if feature not in baseline or baseline[feature]["std"] <= 0:
                raise ValueError(f"Invalid drift baseline for {feature}.")
        return baseline
    except FileNotFoundError:
        logger.warning(json.dumps({"event": "drift_baseline_missing", "path": str(path)}))
        return {}


app = FastAPI(
    title="Airline Sentiment Classification API",
    description="Airline review sentiment inference with input drift reporting.",
    version="2.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info(json.dumps({"event": "model_loaded", "model_path": MODEL_PATH.name}))
except Exception as exc:  # Keep the server alive so health checks expose readiness accurately.
    model = None
    logger.error(json.dumps({"event": "model_load_failed", "error": str(exc)}))

baseline = load_baseline(BASELINE_PATH)
live_features: deque[dict[str, float]] = deque(maxlen=WINDOW_SIZE)


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=1000, description="Customer review text.")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    status: str = "success"


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/")
def root():
    return {"message": "Airline Sentiment API", "status": "healthy", "docs_url": "/docs"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model": "tfidf-linear-svm"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment_endpoint(payload: SentimentRequest, request: Request):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is unavailable.")

    started = time.perf_counter()
    features = text_features(payload.text)
    try:
        prediction = predict_sentiment(model, payload.text)
    except Exception:
        logger.exception(json.dumps({"event": "prediction_failed", "request_id": request.state.request_id}))
        raise HTTPException(status_code=500, detail="Prediction failed.") from None

    live_features.append(features)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    # Do not log review text; the digest supports deduplication without storing customer content.
    logger.info(json.dumps({
        "event": "prediction", "request_id": request.state.request_id,
        "text_sha256": hashlib.sha256(payload.text.encode("utf-8")).hexdigest(),
        "features": features, "prediction": prediction, "latency_ms": latency_ms,
    }))
    return SentimentResponse(text=payload.text, sentiment=prediction)


@app.get("/drift")
def drift_report():
    sample = list(live_features)
    if not baseline:
        return {"status": "baseline_unavailable", "sample_size": len(sample), "features": {}}
    if len(sample) < 30:
        return {"status": "collecting", "sample_size": len(sample), "required_samples": 30, "features": {}}

    report = {}
    drifted = False
    for feature in ("char_count", "word_count", "punctuation_ratio"):
        values = [item[feature] for item in sample]
        center = fmean(values)
        z_score = (center - baseline[feature]["mean"]) / baseline[feature]["std"]
        flagged = abs(z_score) >= DRIFT_THRESHOLD
        drifted |= flagged
        report[feature] = {
            "baseline_mean": baseline[feature]["mean"],
            "live_mean": round(center, 4),
            "standardized_mean_shift": round(z_score, 4),
            "drift": flagged,
        }
    return {"status": "drift_detected" if drifted else "ok", "sample_size": len(sample), "features": report}
