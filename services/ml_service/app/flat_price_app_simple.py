from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, Body, HTTPException

from scripts import logger
from scripts.env import env_vars
from scripts.settings import config
from scripts.utils import read_json
from scripts.fast_api_handler import FastApiHandler


# Reading input example
input_example = read_json(Path(env_vars.param_dir, "example_input.json"))

# Creating FastApi app
app = FastAPI()

# Initializing FastApiHandler
app.handler = FastApiHandler(**config["handler"])


@app.post(config["prediction_endpoint"])
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
    try:
        params = {"user_id": user_id, "model_params": model_params}
        response = await app.handler.handle(params=params)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            "Unexpected error occurred while processing the request",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
