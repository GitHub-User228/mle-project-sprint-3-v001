import random
from pathlib import Path
from functools import lru_cache
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field, validator

from scripts.env import env_vars
from scripts.utils import read_yaml

# Reading config data
config = read_yaml(Path(env_vars.config_dir, "config.yaml"))


class ModelParams(BaseModel):
    """
    Defines the `ModelParams` class, which represents the input
    parameters for a prediction model. Used for validation purposes.
    """

    class Config:
        extra = "forbid"

    # Identifiers and unnecessary parameters (specified for compatibility)
    id: int = Field(
        default=1,
        ge=1,
        alias="id",
        description=(
            "Unique ID of an observation (specified for compatibility)"
        ),
    )
    flat_id: int = Field(
        default=1,
        ge=1,
        alias="flat_id",
        description=("Unique ID of a flat (specified for compatibility)"),
    )
    building_id: int = Field(
        default=1,
        ge=1,
        alias="building_id",
        description=("Unique ID of a building (specified for compatibility)"),
    )
    subset: Literal["train", "test"] = Field(
        default="test",
        alias="subset",
        description=("Data subset (specified for compatibility)"),
    )
    is_duplicated: bool = Field(
        default=False,
        alias="is_duplicated",
        description=(
            "Whether a correponding observation is duplicated "
            "(specified for compatibility)"
        ),
    )
    log1p_target: float = Field(
        default=7.0,
        alias="log1p_target",
        description=(
            "Logarithm of the price of a flat (specified for compatibility)",
        ),
    )

    # Binary parameters
    has_elevator: bool = Field(
        alias="has_elevator",
        description="Whether the corresponding building has an elevator",
    )
    is_apartment: bool = Field(
        alias="is_apartment",
        description=("Whether a correponding property is an apartment"),
    )

    # Categorical parameters
    building_type_int: int = Field(
        ge=0,
        le=6,
        alias="building_type_int",
        description=(
            "Building type as integer. If 5 then type is transformed to "
            "4 (most popular) since the model does not support this type"
        ),
    )
    _extract_building_type_int = validator(
        "building_type_int", pre=False, allow_reuse=True
    )(lambda x: x if x != 5 else 4)

    # Discrete numerical parameters
    rooms: int = Field(
        ge=1,
        le=6,
        alias="rooms",
        description=(
            "Number of rooms in a flat. Only <= 6 rooms is supported",
        ),
    )
    floor: int = Field(
        ge=1,
        le=31,
        alias="floor",
        description=("Floor of a flat. Only <= 31 floors is supported",),
    )
    floors_total: int = Field(
        ge=1,
        le=41,
        alias="floors_total",
        description=(
            "Total number of floors in a building. Only <= 41 floors is supported",
        ),
    )
    flats_count: int = Field(
        ge=1,
        le=1000,
        alias="flats_count",
        description=(
            "Number of flats in a building. Only <= 1000 flats is supported",
        ),
    )
    build_year: int = Field(
        ge=1914,
        le=2024,
        alias="build_year",
        description=("Year of building construction. Must be >= 1914"),
    )

    # Continuous numerical parameters
    ceiling_height: float = Field(
        ge=2,
        le=4,
        alias="ceiling_height",
        description=("Height of the ceiling. Must be between 2m and 4m",),
    )
    kitchen_area: float = Field(
        ge=0,
        le=35,
        alias="kitchen_area",
        description=("Area of the kitchen. Must be between 0m^2 and 35m^2",),
    )
    living_area: float = Field(
        ge=1,
        le=150,
        alias="living_area",
        description=(
            "Area of the living space. Must be between 1m^2 and 150m^2",
        ),
    )
    total_area: float = Field(
        ge=11,
        le=200,
        alias="total_area",
        description=("Total area of a flat. Must be between 11m^2 and 200^2",),
    )
    latitude: float = Field(
        ge=55.4,
        le=56.1,
        alias="latitude",
        description=("Latitude of a flat. Must be within or close to Moscow",),
    )
    longitude: float = Field(
        ge=37.0,
        le=38.0,
        alias="longitude",
        description=(
            "Longitude of a flat. Must be within or close to Moscow",
        ),
    )


class QueryParams(BaseModel):
    """
    Defines the parameters for a query to the service.
    Used for validation purposes.
    """

    user_id: str = Field(
        alias="user_id",
        description="User ID",
    )
    model_params: dict = Field(
        alias="model_params",
        description="Input for the prediction model",
    )


class GlobalRequestRateLimitSettings(BaseModel):
    """
    Settings for global request rate limiting.
    """

    times: int = Field(
        config["constraints"]["request_rate_limit"]["global"]["times"],
        description="Maximum number of requests allowed within a time window",
        ge=1,
        le=10,
    )

    seconds: Literal[60] = Field(
        config["constraints"]["request_rate_limit"]["global"]["seconds"],
        description="Time window in seconds",
    )


class IPRequestRateLimitSettings(BaseModel):
    """
    Settings for ip request rate limiting.
    """

    times: int = Field(
        config["constraints"]["request_rate_limit"]["per_ip"]["times"],
        description="Maximum number of requests allowed within a time window",
        ge=1,
        le=4,
    )

    seconds: Literal[60] = Field(
        config["constraints"]["request_rate_limit"]["per_ip"]["seconds"],
        description="Time window in seconds",
    )


@lru_cache
def get_global_request_rate_limit_settings() -> GlobalRequestRateLimitSettings:
    """
    Returns the global request rate limit settings.
    """
    return GlobalRequestRateLimitSettings()


@lru_cache
def get_ip_request_rate_limit_settings() -> IPRequestRateLimitSettings:
    """
    Returns the IP request rate limit settings.
    """
    return IPRequestRateLimitSettings()


def generate_random_model_params(invalid_rate: float) -> Dict[str, Any]:
    """
    Generates random model parameters based on the restrictions
    defined in the ModelParams class.

    Args:
        invalid_rate (float):
            The rate at which invalid parameters should be generated.
            Must be between 0 and 1.

    Returns:
        Dict[str, Any]:
            A dictionary containing the generated random model
            parameters.
    """
    params = {}

    for field_name, field in ModelParams.__fields__.items():
        if random.random() < invalid_rate:
            continue
        if field.default:
            params[field_name] = field.default
        elif field.type_.__name__ == "ConstrainedIntValue":
            params[field_name] = random.randint(
                field.field_info.ge, field.field_info.le
            )
        elif field.type_.__name__ == "ConstrainedFloatValue":
            params[field_name] = random.uniform(
                field.field_info.ge, field.field_info.le
            )
        elif field.type_.__name__ == "bool":
            params[field_name] = random.choice([True, False])
        elif field.type_.__name__ == "Literal":
            params[field_name] = random.choice(list(field.type_.__args__))

    return params
