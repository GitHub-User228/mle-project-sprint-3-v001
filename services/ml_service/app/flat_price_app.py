import time
import aioredis
from pathlib import Path
from typing import Dict, Any
from fastapi_limiter import FastAPILimiter
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Body, HTTPException, Depends

from scripts import logger
from scripts.env import env_vars
from scripts.settings import config
from scripts.utils import read_json
from scripts.fast_api_handler import FastApiHandler
from scripts.limiters import global_limiter, ip_limiter
from scripts.prometheus_metrics import (
    INVALID_REQUEST_COUNT,
    FAILED_PREDICTIONS_COUNT,
    UNEXPECTED_ERROR_COUNT,
    PREDICTED_PRICE_HISTOGRAM,
    REQUEST_DURATION_HISTOGRAM,
)

# Reading input example
input_example = read_json(Path(env_vars.param_dir, "example_input.json"))

# Creating FastApi app
app = FastAPI()

# Initializing FastApiHandler
app.handler = FastApiHandler(**config["handler"])


@app.on_event("startup")
async def startup():
    """
    Initializes the FastAPILimiter with a Redis URL. This function is
    called on startup of the FastAPI application to set up the rate
    limiting functionality.
    """
    try:
        redis_client = aioredis.from_url(
            f"redis://redis:{env_vars.redis_vm_port}",
            encoding="utf-8",
            decode_responses=True,
        )
        await FastAPILimiter.init(redis_client)
    except aioredis.RedisError as e:
        message = f"Redis error: {str(e)}"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)
    except Exception as e:
        logger.error(e)
        raise e


instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.post(
    config["prediction_endpoint"],
    dependencies=[Depends(global_limiter), Depends(ip_limiter)],
)
async def get_prediction(
    user_id: str,
    model_params: Dict[str, Any] = Body(..., example=input_example),
) -> Dict[str, Any]:
    """
    Process the prediction request for a given user_id and model
    parameters (model input).

    Args:
        user_id (str):
            The unique identifier for the user.
        model_params (Dict[str, Any]):
            The parameters (input) to be used for the prediction model

    Returns:
        Dict[str, Any]:
            The prediction response.

    Raises:
        HTTPException:
            If there is an error validating the input parameters or
            predicting the price.
        Exception:
            If there is an unexpected error while processing the
            request.
    """
    start_time = time.time()
    try:
        params = {"user_id": user_id, "model_params": model_params}
        response = await app.handler.handle(params=params)
        REQUEST_DURATION_HISTOGRAM.labels(
            handler=config["prediction_endpoint"], is_valid="true"
        ).observe(time.time() - start_time)
        PREDICTED_PRICE_HISTOGRAM.observe(response["prediction"])
        return response
    except HTTPException as e:
        if e.status_code == 400:
            INVALID_REQUEST_COUNT.inc()
        if e.status_code == 500:
            FAILED_PREDICTIONS_COUNT.inc()
        REQUEST_DURATION_HISTOGRAM.labels(
            handler=config["prediction_endpoint"], is_valid="false"
        ).observe(time.time() - start_time)
        raise e
    except Exception as e:
        REQUEST_DURATION_HISTOGRAM.labels(
            handler=config["prediction_endpoint"], is_valid="false"
        ).observe(time.time() - start_time)
        UNEXPECTED_ERROR_COUNT.inc()
        logger.error(
            "Unexpected error occurred while processing the request",
            exc_info=True,
        )

        raise HTTPException(status_code=500, detail=str(e))
