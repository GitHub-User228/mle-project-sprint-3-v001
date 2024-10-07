import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException
from catboost import CatBoostError
from pydantic import ValidationError
from sklearn.exceptions import NotFittedError

from scripts import logger
from scripts.env import env_vars
from scripts.utils import read_json, read_pkl
from scripts.settings import QueryParams, ModelParams, config


class FastApiHandler:
    """
    Handles a request to the FastAPI endpoint, validating the input
    parameters and predicting flat price for the given input.

    Attributes:
        model_filename (str):
            The name of the model file to load.
        fe_pipeline_filename (str):
            The name of the fe_pipeline file to load.
        model (CatBoostRegressor):
            The loaded CatBoostRegressor used to predict the price.
        fe_pipeline (Pipeline):
            The loaded Pipeline used to transform the input features.

    Raises:
        TypeError:
            If the input class arguments are not of the correct type.
        FileNotFoundError:
            If the model or feature engineering pipeline file is not
            found.
        ImportError:
            If a function or a class used in the feature engineering
            pipeline
            cannot be imported.
        AttributeError:
            If there is an attribute error occured in the feature
            engineering pipeline.
        RuntimeError:
            If there is an unexpected error loading the model or
            feature engineering pipeline file.
        HTTPException:
            If there is an error validating the input parameters or
            predicting the price.
    """

    def __init__(self, model_filename: str, fe_pipeline_filename: str) -> None:

        self.model_filename = model_filename
        self.fe_pipeline_filename = fe_pipeline_filename
        self._validate_class_input()
        self.load_models()
        self.init_fe_pipeline()
        logger.info(
            f"[{self.__class__.__name__}] Handler has been initialised."
        )

    def load_models(self):
        try:
            self.model = read_pkl(
                Path(env_vars.model_dir, self.model_filename)
            )
        except FileNotFoundError:
            msg = f"Model file '{self.model_filename}' not found"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise FileNotFoundError(msg)
        except Exception as e:
            msg = f"Error loading model file '{self.model_filename}'"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise RuntimeError(msg) from e

        try:
            self.fe_pipeline = read_pkl(
                Path(env_vars.model_dir, self.fe_pipeline_filename)
            )
        except FileNotFoundError:
            msg = f"FE pipeline file '{self.fe_pipeline_filename}' not found"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise FileNotFoundError(msg)
        except ImportError as e:
            msg = "Error loading FE pipeline: missing function or class"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise ImportError(msg) from e
        except AttributeError as e:
            msg = "Error loading FE pipeline: attribute error"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise AttributeError(msg) from e
        except Exception as e:
            msg = (
                f"Unexpected error loading FE pipeline file "
                f"'{self.fe_pipeline_filename}'"
            )
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise RuntimeError(msg) from e

    def _validate_class_input(self) -> None:
        """
        Validates the input parameters for the FastApiHandler class.

        Raises:
            TypeError:
                If the input arguments are not of the correct type.
        """
        if not isinstance(self.model_filename, str):
            raise TypeError("Model filename must be a string")
        if not isinstance(self.fe_pipeline_filename, str):
            raise TypeError("FE pipeline filename must be a string")

    def init_fe_pipeline(self) -> None:
        """
        Initialises the feature engineering pipeline. Since
        `AutoFeatRegressor` is not yet initialised, we pass the test
        data to it.
        """
        logger.info(
            f"[{self.__class__.__name__}] Initialising the FE pipeline..."
        )
        model_params = read_json(
            Path(env_vars.param_dir, "example_input.json")
        )
        model_params = ModelParams(**model_params).dict()
        _ = self.fe_pipeline.transform(pd.DataFrame([model_params]))

    async def predict(self, model_params: Dict[str, Any]) -> float:
        """
        Predicts the price of a flat for the given input.

        Args:
            model_params (dict):
                A dictionary containing the input for the model.

        Returns:
            float:
                The flat price for the given input.

        Raises:
            HTTPException:
                If an error occurs during prediction or transformation.
        """
        try:
            # Generating features via fe_pipeline
            transformed_features = await asyncio.to_thread(
                self.fe_pipeline.transform, pd.DataFrame([model_params])
            )
        except NotFittedError as e:
            msg = "Feature engineering pipeline is not fitted"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )
        except ValueError as e:
            msg = "Invalid input data for feature engineering pipeline"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={"error_message": msg, "error_info": str(e)},
            )
        except Exception as e:
            msg = "Error during feature transformation"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )

        try:
            # Predicting price
            prediction = await asyncio.to_thread(
                self.model.predict, transformed_features
            )
            prediction = 10 ** prediction.item() - 1
            return prediction
        except CatBoostError as e:
            msg = "Error during model prediction with CatBoost"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )
        except Exception as e:
            msg = "Unexpected error during model prediction with CatBoost"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validates the input parameters for the flat price prediction
        request.

        Args:
            params (Dict[str, Any]):
                The input parameters for the request.

        Raises:
            HTTPException:
                If any of the query parameters or model parameters are
                missing or invalid.
        """
        # Check if all query params exist
        try:
            _ = QueryParams(**params)
            logger.info(f"[{self.__class__.__name__}]: All query params exist")
        except ValidationError as e:
            msg = "Not all query params exist"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={"error_message": msg, "error_info": str(e)},
            )

        # Check if all model params exist and are valid
        try:
            params["model_params"] = ModelParams(
                **params["model_params"]
            ).dict()
            logger.info(
                f"[{self.__class__.__name__}]: All model params exist and are valid"
            )
        except ValidationError as e:
            msg = "Not all model params exist or some of them are invalid"
            logger.error(f"[{self.__class__.__name__}]: {msg}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={"error_message": msg, "error_info": str(e)},
            )

    async def handle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a request to the FastAPI endpoint, validating the input
        parameters and predicting the flat price for the given input.

        Args:
            params (Dict[str, Any]):
                A dictionary containing the input parameters.

        Returns:
            Dict[str, Any]:
                A dictionary containing the response. The response
                includes the user_id and the predicted flat price.
                If there is an error, the response will contain an
                "Error" key with an error message.
        """
        self.validate_params(params=params)
        logger.info(
            f"[{self.__class__.__name__}]: "
            f"Predicting flat price for user_id "
            f"{params['user_id']} and model_params "
            f"{params['model_params']}"
        )
        prediction = await self.predict(model_params=params["model_params"])
        return {
            "user_id": params["user_id"],
            "prediction": prediction,
        }


if __name__ == "__main__":

    # Loading example input
    test_params = read_json(Path(env_vars.param_dir, "example_input.json"))

    # Initializing the Fast API handler
    handler = FastApiHandler(**config["handler"])

    # Handling the request
    response = handler.handle(params=test_params)
    print(f"Response: {response}")
