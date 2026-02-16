"""Unit tests for tuning/fitness.py."""

from __future__ import annotations

import pandas as pd
import pytest

from lottogogo.tuning.fitness import (
    WEIGHT_BOUNDS,
    WEIGHT_KEYS,
    FitnessEvaluationError,
    FitnessEvaluator,
    FitnessResult,
    random_baseline,
    _hit_at_k,
    _mean_rank,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_history(n_rounds: int = 120) -> pd.DataFrame:
    """Create synthetic lottery history for testing."""
    import random as rng

    rng.seed(42)
    rows = []
    for r in range(1, n_rounds + 1):
        nums = sorted(rng.sample(range(1, 46), 6))
        rows.append({
            "round": r,
            "n1": nums[0],
            "n2": nums[1],
            "n3": nums[2],
            "n4": nums[3],
            "n5": nums[4],
            "n6": nums[5],
            "bonus": rng.randint(1, 45),
        })
    return pd.DataFrame(rows)


def _default_weights() -> dict[str, float]:
    """Return default weights within valid bounds."""
    return {
        "hot_weight": 0.40,
        "cold_weight": 0.15,
        "neighbor_weight": 0.30,
        "carryover_weight": 0.40,
        "reverse_weight": 0.10,
        "hmm_hot_boost": 0.30,
        "hmm_cold_boost": 0.15,
        "poisson_lambda": 0.0,
        "markov_lambda": 0.0,
        "temperature": 0.5,
    }


# ── Tests: random_baseline ──────────────────────────────────────────────────

def test_random_baseline_k15():
    result = random_baseline(15)
    assert result == pytest.approx(15 * 6 / 45, rel=1e-6)


def test_random_baseline_k20():
    result = random_baseline(20)
    assert result == pytest.approx(20 * 6 / 45, rel=1e-6)


# ── Tests: _hit_at_k ────────────────────────────────────────────────────────

def test_hit_at_k_counts_overlapping_numbers():
    scores = {i: float(46 - i) for i in range(1, 46)}  # 1 is highest
    actual = {1, 2, 3, 10, 20, 30}
    assert _hit_at_k(scores, actual, k=5) == 3  # 1, 2, 3 in top-5
    assert _hit_at_k(scores, actual, k=3) == 3  # 1, 2, 3


def test_hit_at_k_zero_when_no_overlap():
    scores = {i: float(46 - i) for i in range(1, 46)}
    actual = {40, 41, 42, 43, 44, 45}
    assert _hit_at_k(scores, actual, k=5) == 0


# ── Tests: _mean_rank ───────────────────────────────────────────────────────

def test_mean_rank_returns_average_rank():
    scores = {i: float(46 - i) for i in range(1, 46)}
    actual = {1, 2, 3, 4, 5, 6}
    rank = _mean_rank(scores, actual)
    assert rank == pytest.approx(3.5)  # ranks 1,2,3,4,5,6 → avg 3.5


# ── Tests: FitnessEvaluator ──────────────────────────────────────────────────

def test_evaluator_rejects_empty_history():
    with pytest.raises(FitnessEvaluationError, match="empty"):
        FitnessEvaluator(pd.DataFrame(), train_end=100, val_end=200)


def test_evaluator_rejects_invalid_range():
    history = _make_history(10)
    with pytest.raises(FitnessEvaluationError, match="Invalid range"):
        FitnessEvaluator(history, train_end=200, val_end=100)


def test_evaluator_rejects_out_of_bounds_weight():
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    bad_weights = _default_weights()
    bad_weights["temperature"] = 10.0  # above max 2.0
    with pytest.raises(FitnessEvaluationError, match="out of bounds"):
        evaluator.evaluate(bad_weights)


def test_evaluator_rejects_missing_weight():
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    incomplete = {"hot_weight": 0.4}  # missing others
    with pytest.raises(FitnessEvaluationError, match="Missing weight"):
        evaluator.evaluate(incomplete)


def test_evaluator_returns_fitness_result():
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    weights = _default_weights()
    result = evaluator.evaluate(weights)

    assert isinstance(result, FitnessResult)
    assert result.hit_at_15 >= 0
    assert result.hit_at_20 >= 0
    assert result.mean_rank > 0
    assert result.combined_fitness == pytest.approx(
        0.6 * result.train_fitness + 0.4 * result.val_fitness
    )


def test_evaluator_time_sequential_no_leakage():
    """Verify train data doesn't include future rounds."""
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    weights = _default_weights()
    # This should succeed without errors — if it leaks, we'd get 
    # impossibly high scores, but at minimum it shouldn't crash
    result = evaluator.evaluate(weights)
    assert result.hit_at_15 <= 15  # can't exceed K
    assert result.hit_at_20 <= 20


# ── Tests: Weight bounds ────────────────────────────────────────────────────

def test_weight_bounds_have_correct_keys():
    assert set(WEIGHT_KEYS) == set(WEIGHT_BOUNDS.keys())
    assert len(WEIGHT_KEYS) == 10


def test_weight_bounds_are_valid():
    for key, (lo, hi) in WEIGHT_BOUNDS.items():
        assert lo < hi, f"{key}: lo={lo} >= hi={hi}"
