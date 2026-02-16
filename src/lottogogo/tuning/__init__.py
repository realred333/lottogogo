"""Hyper-parameter tuning helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PenaltyTuneConfig",
    "PenaltyTuneResult",
    "run_penalty_grid_search",
    "FitnessEvaluator",
    "FitnessResult",
    "GAOptimizer",
    "GAConfig",
    "OptimizationResult",
    "FeatureBuilder",
    "XGBRanker",
    "RankerResult",
]


def __getattr__(name: str) -> Any:
    if name in ("PenaltyTuneConfig", "PenaltyTuneResult", "run_penalty_grid_search"):
        from . import penalty_search

        return getattr(penalty_search, name)
    if name in ("FitnessEvaluator", "FitnessResult"):
        from . import fitness

        return getattr(fitness, name)
    if name in ("GAOptimizer", "GAConfig", "OptimizationResult"):
        from . import ga_optimizer

        return getattr(ga_optimizer, name)
    if name in ("FeatureBuilder",):
        from . import feature_builder

        return getattr(feature_builder, name)
    if name in ("XGBRanker", "RankerResult"):
        from . import xgb_ranker

        return getattr(xgb_ranker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
