from __future__ import annotations

import pandas as pd
import pytest

from lottogogo.engine.backtester import (
    RandomBaselineGenerator,
    RoundBacktestResult,
    WalkForwardBacktester,
    compare_summaries,
    generate_backtest_report,
    summarize_results,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 5],
            "n1": [1, 1, 2, 3, 4],
            "n2": [2, 2, 3, 4, 5],
            "n3": [3, 3, 4, 5, 6],
            "n4": [4, 4, 5, 6, 7],
            "n5": [5, 5, 6, 7, 8],
            "n6": [6, 6, 7, 8, 9],
        }
    )


def test_t611_walk_forward_loop_sequential_and_accumulated():
    history = _history()
    train_round_sizes: list[int] = []

    def recommender(train_df: pd.DataFrame, output_count: int, seed: int | None):
        train_round_sizes.append(len(train_df))
        return [(1, 2, 3, 4, 5, 6)] * output_count

    backtester = WalkForwardBacktester()
    results = backtester.run(
        history=history,
        recommender=recommender,
        start_round=3,
        end_round=5,
        recent_n=2,
        output_count=2,
        seed=42,
    )

    assert [result.round_id for result in results] == [3, 4, 5]
    assert train_round_sizes == [2, 2, 2]
    assert all(len(result.recommendations) == 2 for result in results)


def test_t612_baseline_generator_is_seed_reproducible():
    generator = RandomBaselineGenerator()
    baseline1 = generator.generate(output_count=5, seed=7)
    baseline2 = generator.generate(output_count=5, seed=7)

    assert baseline1 == baseline2
    assert len(baseline1) == 5
    assert all(len(set(combo)) == 6 for combo in baseline1)


def _sample_round_results() -> list[RoundBacktestResult]:
    return [
        RoundBacktestResult(
            round_id=100,
            winning_numbers=(1, 2, 3, 4, 5, 6),
            recommendations=[(1, 2, 3, 10, 11, 12), (1, 2, 7, 8, 9, 10)],
            hit_counts=[3, 2],
        ),
        RoundBacktestResult(
            round_id=101,
            winning_numbers=(7, 8, 9, 10, 11, 12),
            recommendations=[(7, 8, 9, 10, 20, 21), (1, 2, 3, 4, 5, 6)],
            hit_counts=[4, 0],
        ),
    ]


def test_t621_probability_match_ge_3_and_baseline_comparison():
    strategy_summary = summarize_results(_sample_round_results())
    baseline_summary = {
        "p_match_ge_3": 0.25,
        "p_match_ge_4": 0.0,
        "average_match_count": 1.0,
        "std_match_count": 0.5,
        "distribution": {0: 1, 1: 2, 2: 1},
        "total_recommendations": 4,
        "total_rounds": 2,
    }

    comparison = compare_summaries(strategy_summary, baseline_summary)

    assert strategy_summary["p_match_ge_3"] == pytest.approx(0.5)
    assert comparison["p_match_ge_3_diff"] == pytest.approx(0.25)


def test_t622_supporting_metrics_avg_ge4_std():
    summary = summarize_results(_sample_round_results())

    assert summary["average_match_count"] == pytest.approx(2.25)
    assert summary["p_match_ge_4"] == pytest.approx(0.25)
    assert summary["std_match_count"] > 0


def test_t623_report_generation_contains_json_markdown_and_config_snapshot():
    strategy_results = _sample_round_results()
    baseline_results = [
        RoundBacktestResult(
            round_id=100,
            winning_numbers=(1, 2, 3, 4, 5, 6),
            recommendations=[(10, 11, 12, 13, 14, 15)],
            hit_counts=[0],
        )
    ]
    report = generate_backtest_report(
        strategy_results=strategy_results,
        baseline_results=baseline_results,
        config_snapshot={"seed": 42, "output_count": 2},
    )

    assert "strategy" in report.json_report
    assert "baseline" in report.json_report
    assert "comparison" in report.json_report
    assert "distribution_graph" in report.json_report["strategy"]
    assert report.json_report["config_snapshot"]["seed"] == 42
    assert "## Backtest Summary" in report.markdown_report

