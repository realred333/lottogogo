from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lottogogo.config.loader import ConfigLoadError, load_config


def test_load_json_config_with_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"sample_size": 12345, "prior_alpha": 2.0}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.sample_size == 12345
    assert config.prior_alpha == 2.0
    assert config.prior_beta == 1.0
    assert config.output_count == 5


def test_load_yaml_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "recent_n: 20\n" "sample_size: 2000\n" "min_prob_floor: 0.005\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.recent_n == 20
    assert config.sample_size == 2000
    assert config.min_prob_floor == 0.005


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_invalid_config_value_raises_validation_error(tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"sample_size": 0}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_unsupported_extension_raises(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("sample_size=10", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Unsupported config format"):
        load_config(config_path)

