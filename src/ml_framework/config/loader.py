"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ml_framework.config.base import PipelineConfig

T = TypeVar("T", bound=BaseModel)


def _resolve_env_vars(config: dict) -> dict:
    """Recursively resolve environment variables in config values.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    """
    def resolve_value(value: str) -> str:
        if not isinstance(value, str):
            return value

        import re
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return re.sub(pattern, replacer, value)

    def resolve_recursive(obj: dict | list | str) -> dict | list | str:
        if isinstance(obj, dict):
            return {k: resolve_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return resolve_value(obj)
        return obj

    return resolve_recursive(config)


def load_yaml(path: str | Path) -> dict:
    """Load and parse a YAML file with environment variable resolution."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not path.suffix.lower() in (".yaml", ".yml"):
        raise ValueError(f"Expected YAML file, got: {path.suffix}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Empty config file: {path}")

    return _resolve_env_vars(raw_config)


def load_config(path: str | Path, config_class: type[T] = PipelineConfig) -> T:
    """Load and validate a pipeline configuration from YAML.

    Args:
        path: Path to the YAML configuration file.
        config_class: Pydantic model class for validation.

    Returns:
        Validated configuration instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config doesn't match schema.
    """
    raw_config = load_yaml(path)
    return config_class.model_validate(raw_config)


def validate_config(config: dict, config_class: type[T] = PipelineConfig) -> T:
    """Validate a config dict against a Pydantic model.

    Useful for programmatic config creation.
    """
    return config_class.model_validate(config)
