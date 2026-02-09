from __future__ import annotations

import pytest

import lottogogo.tuning.penalty_search as penalty_search
from lottogogo.tuning.penalty_search import (
    PenaltyTuneConfig,
    PenaltyTuneResult,
    _WorkerInput,
    _evaluate_worker,
    _validate_lambda_bounds,
    build_float_grid,
)


def test_build_float_grid_inclusive():
    values = build_float_grid(0.0, 0.3, 0.1)
    assert values == [0.0, 0.1, 0.2, 0.3]


def test_objective_key_prefers_higher_p3_then_lower_std():
    better = PenaltyTuneResult(
        poisson_lambda=0.2,
        markov_lambda=0.3,
        p_match_ge_3=0.40,
        p_match_ge_4=0.10,
        average_match_count=2.0,
        std_match_count=1.1,
        total_rounds=30,
        total_recommendations=150,
    )
    worse = PenaltyTuneResult(
        poisson_lambda=0.2,
        markov_lambda=0.4,
        p_match_ge_3=0.35,
        p_match_ge_4=0.12,
        average_match_count=2.1,
        std_match_count=0.9,
        total_rounds=30,
        total_recommendations=150,
    )
    assert better.objective_key() > worse.objective_key()


def test_validate_lambda_bounds_rejects_above_half():
    with pytest.raises(ValueError):
        _validate_lambda_bounds([0.0, 0.6], "poisson_lambda")


def test_evaluate_worker_forces_full_history_window(monkeypatch):
    history = penalty_search.pd.DataFrame(
        {
            "round": [1, 2, 3],
            "n1": [1, 2, 3],
            "n2": [4, 5, 6],
            "n3": [7, 8, 9],
            "n4": [10, 11, 12],
            "n5": [13, 14, 15],
            "n6": [16, 17, 18],
        }
    )
    config = PenaltyTuneConfig(recent_n=2, output_count=1, sample_size=10, rank_top_k=5, workers=1)
    worker_input = _WorkerInput(
        history=history,
        config=config,
        poisson_lambda=0.1,
        markov_lambda=0.1,
    )

    captured: dict[str, object] = {}

    class DummyBacktester:
        def run(self, **kwargs):
            captured["recent_n"] = kwargs["recent_n"]
            return []

    monkeypatch.setattr(penalty_search, "WalkForwardBacktester", DummyBacktester)
    monkeypatch.setattr(
        penalty_search,
        "summarize_results",
        lambda _results: {
            "p_match_ge_3": 0.0,
            "p_match_ge_4": 0.0,
            "average_match_count": 0.0,
            "std_match_count": 0.0,
            "total_rounds": 0,
            "total_recommendations": 0,
        },
    )

    _evaluate_worker(worker_input)
    assert captured["recent_n"] is None
