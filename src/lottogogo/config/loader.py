"""Load engine config from YAML or JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schema import EngineConfig

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments only
    yaml = None


class ConfigLoadError(ValueError):
    """Raised when config cannot be loaded or parsed."""


def load_config(path: str | Path) -> EngineConfig:
    """Load config file from YAML/JSON and validate with Pydantic."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(config_path)
    elif suffix == ".json":
        data = _load_json(config_path)
    else:
        raise ConfigLoadError(
            f"Unsupported config format '{suffix}'. Use .yaml/.yml or .json."
        )

    if not isinstance(data, dict):
        raise ConfigLoadError("Config root must be a JSON/YAML object.")

    try:
        return EngineConfig.model_validate(data)
    except ValidationError as exc:
        raise exc


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ConfigLoadError("PyYAML is required to load YAML config files.")

    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)

    return parsed or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        parsed = json.load(file)

    return parsed or {}

