from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("configs/rag_config.yaml")


def load_yaml_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    keys = key_path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def load_rag_config() -> Dict[str, Any]:
    return load_yaml_config(DEFAULT_CONFIG_PATH)
