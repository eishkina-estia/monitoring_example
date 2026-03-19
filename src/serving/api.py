# python 3.10+
# https://fastapi.tiangolo.com/advanced/testing-events/
from contextlib import asynccontextmanager

from time import perf_counter

from fastapi import FastAPI, Request, Response, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.monitoring.prometheus.metrics import (
    MODEL_INFO,
    PREDICTION_LATENCY,
    REQUEST_COUNT,
    REQUEST_ERROR_COUNT,
)

from src.serving.inference import load_inference_artifacts, predict_quality
from src.serving.schemas import WineFeatures, PredictionResponse, HealthResponse


app_state = {
    "model": None,
    "preprocessor": None,
    "model_signature": None,
    "model_name": None,
    "model_version": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API starts."""
    print("Start API")
    app_state.update(load_inference_artifacts())

    MODEL_INFO.labels(
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
    ).set(1)

    yield
    print("Stop API")

app = FastAPI(
    title="Wine Quality Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint.

    JSON response schema:
    - message: service description
    """
    return {"message": "Wine Quality Prediction API"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: WineFeatures, request: Request):
    """Wine quality prediction endpoint."""

    # endpoint = "/predict"
    endpoint = request.scope["route"].path
    REQUEST_COUNT.labels(endpoint=endpoint).inc()

    start_time = perf_counter()

    try:
        predicted_quality = predict_quality(
            payload=payload,
            preprocessor=app_state["preprocessor"],
            model=app_state["model"],
            model_signature=app_state["model_signature"]
        )
        latency = perf_counter() - start_time
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency)

    except Exception as e:
        REQUEST_ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictionResponse(
        predicted_quality=predicted_quality,
        model_name=app_state["model_name"],
        model_version=app_state["model_version"],
    )

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        # Collects all registered Prometheus metrics and formats them into the Prometheus text format
        content=generate_latest(),
        # Sets the correct content type required for Prometheus to parse metrics correctly
        media_type=CONTENT_TYPE_LATEST,
    )