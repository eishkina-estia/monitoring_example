from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint"],
)

REQUEST_ERROR_COUNT = Counter(
    "prediction_request_errors_total",
    "Total number of prediction request errors",
    ["endpoint"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
)

MODEL_INFO = Gauge(
    "model_info",
    "Loaded model information",
    ["model_name", "model_version"],
)