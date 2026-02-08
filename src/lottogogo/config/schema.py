"""Pydantic schema for engine configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EngineConfig(BaseModel):
    """Validated runtime configuration with defaults."""

    model_config = ConfigDict(extra="forbid")

    recent_n: int = Field(default=50, gt=0)
    prior_alpha: float = Field(default=1.0, gt=0.0)
    prior_beta: float = Field(default=1.0, gt=0.0)

    hot_threshold: int = Field(default=2, ge=1)
    hot_window: int = Field(default=5, ge=1)
    hot_weight: float = Field(default=0.4, ge=0.0)
    cold_window: int = Field(default=10, ge=1)
    cold_weight: float = Field(default=0.15, ge=0.0)
    neighbor_weight: float = Field(default=0.3, ge=0.0)
    carryover_weight: float = Field(default=0.2, ge=0.0)
    reverse_weight: float = Field(default=0.1, ge=0.0)

    poisson_window: int = Field(default=20, ge=1)
    poisson_lambda: float = Field(default=0.5, ge=0.0)
    markov_lambda: float = Field(default=0.3, ge=0.0)

    sample_size: int = Field(default=50000, gt=0)
    min_prob_floor: float = Field(default=0.001, ge=0.0, lt=1.0)
    output_count: int = Field(default=5, gt=0)

    max_overlap: int = Field(default=3, ge=0, le=6)
    seed: int = Field(default=42)

