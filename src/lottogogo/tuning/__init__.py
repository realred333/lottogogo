"""Hyper-parameter tuning helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["PenaltyTuneConfig", "PenaltyTuneResult", "run_penalty_grid_search"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import penalty_search

        return getattr(penalty_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
