import math

import yaml
import json
import random
import joblib
from pathlib import Path
from typing import Dict, Any

from scripts import logger


def read_yaml(path: Path, verbose: bool = True) -> Dict:
    """
    Reads a yaml file, and returns a dict.

    Args:
        path_to_yaml (Path):
            Path to the yaml file

    Returns:
        Dict:
            The yaml content as a dict.
        verbose:
            Whether to do any info logs

    Raises:
        ValueError:
            If the file is not a YAML file
        FileNotFoundError:
            If the file is not found.
        yaml.YAMLError:
            If there is an error parsing the yaml file.
    """
    if path.suffix not in [".yaml", ".yml"]:
        msg = f"The file {path} is not a YAML file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)
    try:
        with open(path, "r") as file:
            content = yaml.safe_load(file)
        if verbose:
            logger.info(f"YAML file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise yaml.YAMLError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading YAML file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_json(path: Path, verbose: bool = True) -> Dict:
    """
    Reads a JSON file and returns a dict.

    Args:
        path (Path):
            Path to the JSON file
        verbose (bool):
            Whether to do any info logs

    Returns:
        Dict:
            The JSON content as a dict.

    Raises:
        ValueError:
            If the file is not a JSON file
        FileNotFoundError:
            If the file is not found.
        json.JSONDecodeError:
            If there is an error parsing the JSON file.
    """
    if path.suffix != ".json":
        msg = f"The file {path} is not a JSON file"
        logger.error(f"{msg}")
        raise ValueError(msg)

    try:
        with open(path, "r") as file:
            content = json.load(file)
        if verbose:
            logger.info(f"JSON file {path} has been loaded")
        return content
    except FileNotFoundError as e:
        msg = f"File {path} not found"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except json.JSONDecodeError as e:
        msg = f"Error parsing JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise json.JSONDecodeError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while reading JSON file {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def read_pkl(path: Path) -> object:
    """
    Reads a model object from a file using joblib.

    Args:
        path (Path):
            The path to the file with the model to load.

    Returns:
        object:
            The loaded model object.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        FileNotFoundError:
            If the file does not exist.
        IOError:
            If an I/O error occurs during the loading process.
        Exception:
            If an unexpected error occurs while loading the model.
    """

    if path.suffix != ".pkl":
        msg = f"The file {path} is not a pkl file"
        logger.error(f"{msg}: {e}")
        raise ValueError(msg)

    try:
        with open(path, "rb") as f:
            model = joblib.load(f)
        logger.info(f"Model {path} has been loaded")
        return model
    except FileNotFoundError as e:
        msg = f"File '{path}' does not exist"
        logger.error(f"{msg}: {e}")
        raise FileNotFoundError(msg) from e
    except IOError as e:
        msg = f"An I/O error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise IOError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred while loading a model from {path}"
        logger.error(f"{msg}: {e}")
        raise Exception(msg) from e


def calculate_expire_time(pexpire: int) -> int:
    """
    Calculates the expiration time in seconds from a
    provided expiration time in milliseconds.

    Args:
        pexpire (int):
        The expiration time in milliseconds.

    Returns:
        int:
            The expiration time in seconds, rounded up to the
            nearest integer.
    """
    return math.ceil(pexpire / 1000)


def boolean(arg: Any):
    """
    Converts the given argument to a boolean value.
    Used in the argument parser.

    Args:
        arg (Any):
            The input value to be converted to a boolean.

    Returns:
        bool:
            The boolean value corresponding to the input.

    Raises:
        ValueError:
            If the input value is not "True" or "False".
    """
    arg = str(arg)
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise ValueError("invalid value for boolean argument")


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
