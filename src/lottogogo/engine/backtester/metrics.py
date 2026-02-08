"""Metrics for backtesting results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .walk_forward import RoundBacktestResult


def summarize_results(results: Sequence[RoundBacktestResult]) -> dict[str, object]:
    """Aggregate backtest metrics from per-round results."""
    hit_counts: list[int] = []
    for result in results:
        hit_counts.extend(int(value) for value in result.hit_counts)

    total_recommendations = len(hit_counts)
    if total_recommendations == 0:
        return {
            "p_match_ge_3": 0.0,
            "p_match_ge_4": 0.0,
            "average_match_count": 0.0,
            "std_match_count": 0.0,
            "distribution": {},
            "distribution_graph": [],
            "total_recommendations": 0,
            "total_rounds": len(results),
        }

    hit_array = np.array(hit_counts, dtype=np.float64)
    distribution = Counter(int(value) for value in hit_counts)
    distribution_graph = [{"match_count": key, "count": distribution[key]} for key in sorted(distribution)]

    return {
        "p_match_ge_3": float(np.mean(hit_array >= 3)),
        "p_match_ge_4": float(np.mean(hit_array >= 4)),
        "average_match_count": float(np.mean(hit_array)),
        "std_match_count": float(np.std(hit_array)),
        "distribution": dict(sorted(distribution.items())),
        "distribution_graph": distribution_graph,
        "total_recommendations": total_recommendations,
        "total_rounds": len(results),
    }


def compare_summaries(
    strategy_summary: dict[str, object],
    baseline_summary: dict[str, object],
) -> dict[str, float]:
    """Compare strategy and baseline metrics."""
    p3_strategy = float(strategy_summary.get("p_match_ge_3", 0.0))
    p3_baseline = float(baseline_summary.get("p_match_ge_3", 0.0))
    p4_strategy = float(strategy_summary.get("p_match_ge_4", 0.0))
    p4_baseline = float(baseline_summary.get("p_match_ge_4", 0.0))
    avg_strategy = float(strategy_summary.get("average_match_count", 0.0))
    avg_baseline = float(baseline_summary.get("average_match_count", 0.0))

    return {
        "p_match_ge_3_diff": p3_strategy - p3_baseline,
        "p_match_ge_4_diff": p4_strategy - p4_baseline,
        "average_match_count_diff": avg_strategy - avg_baseline,
    }

