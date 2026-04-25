from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app

APP = FastAPI(title="Fraud Detection API")

PREDICTION_LATENCY = Histogram("fraud_api_latency_seconds", "Inference latency")
PREDICTION_COUNT = Counter("fraud_api_requests_total", "Total inference requests")
PREDICTION_ERRORS = Counter("fraud_api_errors_total", "Total inference errors")

MODEL_PATH = Path("artifacts/model.joblib")
MODEL = None


class InferencePayload(BaseModel):
    features: dict[str, float | int]


@APP.on_event("startup")
def load_model() -> None:
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)


@APP.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@APP.post("/predict")
def predict(payload: InferencePayload) -> dict[str, float | int]:
    start = time.perf_counter()
    PREDICTION_COUNT.inc()

    if MODEL is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feature_vector = np.array([list(payload.features.values())], dtype=float)
        probability = float(MODEL.predict_proba(feature_vector)[0][1])
        prediction = int(probability >= 0.5)
        return {"prediction": prediction, "fraud_probability": probability}
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


metrics_app = make_asgi_app()
APP.mount("/metrics", metrics_app)
