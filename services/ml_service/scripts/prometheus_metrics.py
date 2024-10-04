from pathlib import Path
from prometheus_client import Counter, Histogram

from scripts.env import env_vars
from scripts.utils import read_yaml


INVALID_REQUEST_COUNT = Counter(
    "invalid_request_count",
    (
        "Number of invalid requests. 400 status code is used to determine "
        "this error"
    ),
)
FAILED_PREDICTIONS_COUNT = Counter(
    "failed_predictions_count",
    (
        "Number of failed predictions due to error in the model. 500 status "
        "code is used to determine this error"
    ),
)
TOO_MANY_REQUESTS_COUNT = Counter(
    "too_many_requests_count",
    "Number of requests that exceeded the rate limit",
)
UNEXPECTED_ERROR_COUNT = Counter(
    "unexpected_error_count",
    "Number of unexpected errors while processing requests",
)

PREDICTED_PRICE_HISTOGRAM = Histogram(
    "prediction_price_histogram",
    "Histogram of the predicted price by model",
    buckets=list(
        read_yaml(Path(env_vars.param_dir, "quantiles_price.yaml")).values()
    ),
)

REQUEST_DURATION_HISTOGRAM = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["handler", "is_valid"],
    buckets=[0.1, 0.2, 0.5, 0.5, 1.0, 1.5, 2, 2.5, 5.0, 10.0],
)
