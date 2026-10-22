from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fast_api
from fastapi.testclient import TestClient


class StubModel:
    def __init__(self, label="positive"):
        self.label = label

    def predict(self, texts):
        assert len(texts) == 1
        return [self.label]


def test_predict_returns_classifier_label():
    assert fast_api.predict_sentiment(StubModel("negative"), "Awful flight") == "negative"


def test_predict_converts_non_string_label():
    assert fast_api.predict_sentiment(StubModel(2), "Review") == "2"


def test_predict_rejects_wrong_prediction_count():
    class BadModel:
        def predict(self, texts):
            return []

    try:
        fast_api.predict_sentiment(BadModel(), "Review")
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("Expected a prediction shape error")


def test_text_features_are_computed_consistently():
    assert fast_api.text_features("Hi! there") == {
        "char_count": 9.0,
        "word_count": 2.0,
        "punctuation_ratio": 1 / 9,
    }


def test_prediction_endpoint_returns_response_and_request_id(monkeypatch):
    monkeypatch.setattr(fast_api, "model", StubModel("neutral"))
    fast_api.live_features.clear()
    with TestClient(fast_api.app) as client:
        response = client.post("/predict", json={"text": "The flight was okay"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "neutral"
    assert response.headers["x-request-id"]
    assert len(fast_api.live_features) == 1


def test_saved_model_inference_uses_training_preprocessing():
    assert fast_api.model is not None
    prediction = fast_api.predict_sentiment(fast_api.model, "The crew was kind and helpful")
    assert prediction in {"negative", "neutral", "positive"}


def test_garbage_input_fails_with_validation_error():
    with TestClient(fast_api.app) as client:
        short = client.post("/predict", json={"text": "x"})
        wrong_type = client.post("/predict", json={"text": 123})
    assert short.status_code == 422
    assert wrong_type.status_code == 422


def test_unavailable_model_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(fast_api, "model", None)
    with TestClient(fast_api.app) as client:
        response = client.post("/predict", json={"text": "A valid review"})
    assert response.status_code == 503


def test_drift_report_waits_for_minimum_sample(monkeypatch):
    monkeypatch.setattr(fast_api, "baseline", {
        name: {"mean": 1.0, "std": 1.0}
        for name in ("char_count", "word_count", "punctuation_ratio")
    })
    fast_api.live_features.clear()
    fast_api.live_features.append(fast_api.text_features("A review"))
    report = fast_api.drift_report()
    assert report["status"] == "collecting"
    assert report["required_samples"] == 30
