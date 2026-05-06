from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_by_path(config: dict, path: str, default=None):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def require_by_path(config: dict, path: str):
    value = get_by_path(config, path, default=None)
    if value is None:
        raise KeyError(f"Missing required config path: {path}")
    return value


def set_by_path(config: dict, path: str, value) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _coerce_override_value(raw_value: str, reference):
    if raw_value.lower() == "null":
        return None
    if isinstance(reference, bool):
        return raw_value.lower() in ("1", "true", "yes", "y", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw_value)
    if isinstance(reference, float):
        return float(raw_value)
    if isinstance(reference, list):
        return [item.strip() for item in raw_value.split(",")]
    if reference is None:
        low = raw_value.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(raw_value)
        except ValueError:
            try:
                return float(raw_value)
            except ValueError:
                return raw_value
    return raw_value


def apply_cli_overrides(config: dict, overrides: list[str]) -> dict:
    if len(overrides) % 2 != 0:
        raise ValueError("CLI overrides must be provided as --path value pairs.")
    updated = dict(config)
    for index in range(0, len(overrides), 2):
        key_token = overrides[index]
        if not key_token.startswith("--"):
            raise ValueError(f"Unexpected override token: {key_token}")
        path = key_token[2:]
        raw_value = overrides[index + 1]
        reference = get_by_path(updated, path, default=None)
        set_by_path(updated, path, _coerce_override_value(raw_value, reference))
    return updated
