"""
VIGIL — Configuration loader

Loads and validates project configuration from YAML files.
Supports environment variable overrides for dataset paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config. Defaults to configs/default.yaml.
        overrides: Dictionary of overrides to apply on top of the loaded config.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Environment variable override for dataset root
    env_root = os.environ.get("VIGIL_DATA_ROOT")
    if env_root:
        config["dataset"]["root"] = env_root

    # Apply overrides
    if overrides:
        _deep_update(config, overrides)

    return config


def _deep_update(base: dict, updates: dict) -> None:
    """Recursively update a nested dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def get_project_root() -> Path:
    """Return the absolute path to the project root (VIGIL/)."""
    return Path(__file__).resolve().parents[1]


def get_dataset_root(config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the dataset root path.

    Checks in order:
    1. VIGIL_DATA_ROOT environment variable
    2. config['dataset']['root']
    3. Default: <project_root>/dataset
    """
    env_root = os.environ.get("VIGIL_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()

    if config and "dataset" in config:
        root = config["dataset"].get("root", "dataset")
        root_path = Path(root)
        if root_path.is_absolute():
            return root_path
        return get_project_root() / root_path

    return get_project_root() / "dataset"
