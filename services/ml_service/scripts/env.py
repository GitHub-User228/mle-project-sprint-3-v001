from pathlib import Path
from pydantic import BaseSettings, validator


class EnviromentVariables(BaseSettings):
    """
    Defines an `EnviromentVariables` class that loads and validates
    environment variables that are used in the project.
    """

    app_docker_port: int
    app_vm_port: int
    redis_vm_port: int
    log_dir: Path
    config_dir: Path
    param_dir: Path
    model_dir: Path

    class Config:
        env_file = ".env"

    @validator("*")
    def check_variable(cls, v: str | Path, field: str) -> str | Path:
        """
        Validates and checks the environment variable specified by the
        `field` parameter. Returns the validated variable.

        Args:
            v (str | Path):
                The environment variable value to be validated.
            field (str):
                The name of the environment variable.

        Raises:
            ValueError:
                If the environment variable is not set.

        Returns:
            str | Path:
                The validated environment variable value.
        """
        if v is None:
            raise ValueError(
                f"Environment variable '{field.name}' is not set."
            )
        if isinstance(v, Path):
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_env(cls) -> "EnviromentVariables":
        """
        Create a new instance of EnviromentVariables by loading
        environment variables.

        Returns:
            EnviromentVariables:
                A new instance of EnviromentVariables with loaded and
                validated environment variables.

        Raises:
            ValueError:
                If any required environment variable is not set.
        """
        return cls()


env_vars = EnviromentVariables.from_env()
