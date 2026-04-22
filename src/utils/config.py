"""Configuration helpers.

Centralizes loading of ``config/params.yaml`` so every module reads the
same settings. Keep this file tiny -- no business logic belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "params.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML config file and return it as a dict.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML config. Defaults to ``config/params.yaml``.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Return absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]
