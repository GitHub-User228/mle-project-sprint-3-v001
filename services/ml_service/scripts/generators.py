import random
from typing import Dict, Any

from scripts.settings import ModelParams


def generate_random_ip() -> str:
    """
    Generates a random IP address as a string.

    Returns:
        str:
            A random IP address in the format "x.x.x.x".
    """
    return (
        f"{random.randint(1, 255)}.{random.randint(0, 255)}."
        f"{random.randint(0, 255)}.{random.randint(0, 255)}"
    )


def generate_random_model_params(invalid_rate: float = 0.0) -> Dict[str, Any]:
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

    if random.random() < invalid_rate:
        return params
    for field_name, field in ModelParams.__fields__.items():
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
