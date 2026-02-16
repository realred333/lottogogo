"""Unit tests for tuning/ga_optimizer.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from lottogogo.tuning.fitness import FitnessEvaluator, FitnessResult
from lottogogo.tuning.ga_optimizer import (
    GAConfig,
    GAOptimizer,
    OptimizationResult,
    _clamp_individual,
    _vec_to_weights,
    _weights_to_vec,
    save_result,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_history(n_rounds: int = 120) -> pd.DataFrame:
    rng = random.Random(42)
    rows = []
    for r in range(1, n_rounds + 1):
        nums = sorted(rng.sample(range(1, 46), 6))
        rows.append({
            "round": r,
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": rng.randint(1, 45),
        })
    return pd.DataFrame(rows)


# ── Tests: GAConfig ──────────────────────────────────────────────────────────

def test_ga_config_defaults():
    cfg = GAConfig()
    assert cfg.population_size == 100
    assert cfg.generations == 200
    assert cfg.seed == 42


def test_ga_config_rejects_small_population():
    with pytest.raises(ValueError, match="population_size"):
        GAConfig(population_size=5)


def test_ga_config_rejects_zero_generations():
    with pytest.raises(ValueError, match="generations"):
        GAConfig(generations=0)


def test_ga_config_rejects_high_elitism():
    with pytest.raises(ValueError, match="elitism_count"):
        GAConfig(population_size=10, elitism_count=6)


# ── Tests: _vec_to_weights / _weights_to_vec ─────────────────────────────────

def test_vec_to_weights_round_trip():
    weights = {
        "hot_weight": 0.4, "cold_weight": 0.15, "neighbor_weight": 0.3,
        "carryover_weight": 0.4, "reverse_weight": 0.1,
        "hmm_hot_boost": 0.3, "hmm_cold_boost": 0.15,
        "poisson_lambda": 0.0, "markov_lambda": 0.0, "temperature": 0.5,
    }
    vec = _weights_to_vec(weights)
    result = _vec_to_weights(vec)
    assert result == weights


# ── Tests: _clamp_individual ──────────────────────────────────────────────────

def test_clamp_individual_clips_out_of_bounds():
    # Create an individual with extreme values
    from lottogogo.tuning.fitness import WEIGHT_KEYS, WEIGHT_BOUNDS

    ind = [10.0] * len(WEIGHT_KEYS)
    clamped = _clamp_individual(ind)
    for i, key in enumerate(WEIGHT_KEYS):
        _, hi = WEIGHT_BOUNDS[key]
        assert clamped[i] <= hi


# ── Tests: GAOptimizer ───────────────────────────────────────────────────────

def test_ga_optimizer_runs_small():
    """Run GA with tiny parameters to verify it completes without error."""
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    config = GAConfig(
        population_size=10,
        generations=3,
        seed=42,
        elitism_count=2,
        tournament_size=3,
    )
    optimizer = GAOptimizer(evaluator, config)
    result = optimizer.run(verbose=False)

    assert isinstance(result, OptimizationResult)
    assert len(result.best_weights) == 10
    assert result.best_fitness.hit_at_15 >= 0
    assert len(result.generation_log) == 3


def test_ga_optimizer_reproducible():
    """Same seed → same result."""
    history = _make_history(120)

    def run_once(seed: int) -> dict:
        evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
        config = GAConfig(population_size=10, generations=3, seed=seed, elitism_count=2)
        optimizer = GAOptimizer(evaluator, config)
        result = optimizer.run(verbose=False)
        return result.best_weights

    w1 = run_once(42)
    w2 = run_once(42)
    assert w1 == w2


def test_ga_optimizer_weights_within_bounds():
    """All output weights must be within defined bounds."""
    from lottogogo.tuning.fitness import WEIGHT_BOUNDS

    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    config = GAConfig(population_size=10, generations=5, seed=42, elitism_count=2)
    optimizer = GAOptimizer(evaluator, config)
    result = optimizer.run(verbose=False)

    for key, val in result.best_weights.items():
        lo, hi = WEIGHT_BOUNDS[key]
        assert lo <= val <= hi, f"{key}={val} not in [{lo}, {hi}]"


# ── Tests: Checkpoint ────────────────────────────────────────────────────────

def test_checkpoint_save_load(tmp_path: Path):
    """Save checkpoint mid-run, then verify it can be loaded."""
    history = _make_history(120)
    evaluator = FitnessEvaluator(history, train_end=80, val_end=120)
    config = GAConfig(population_size=10, generations=5, seed=42, elitism_count=2)
    cp_path = tmp_path / "cp.json"

    optimizer = GAOptimizer(evaluator, config)
    result = optimizer.run(checkpoint_path=cp_path, verbose=False)

    assert cp_path.exists()
    data = json.loads(cp_path.read_text())
    assert "generation" in data
    assert "population" in data
    assert len(data["population"]) == 10


# ── Tests: save_result ───────────────────────────────────────────────────────

def test_save_result_creates_json(tmp_path: Path):
    result = OptimizationResult(
        best_weights={"hot_weight": 0.4, "cold_weight": 0.15,
                      "neighbor_weight": 0.3, "carryover_weight": 0.4,
                      "reverse_weight": 0.1, "hmm_hot_boost": 0.3,
                      "hmm_cold_boost": 0.15, "poisson_lambda": 0.0,
                      "markov_lambda": 0.0, "temperature": 0.5},
        best_fitness=FitnessResult(
            hit_at_15=2.3, hit_at_20=3.0, mean_rank=20.0,
            train_fitness=2.4, val_fitness=2.2, combined_fitness=2.32,
        ),
        baseline_fitness=FitnessResult(
            hit_at_15=2.0, hit_at_20=2.67, mean_rank=23.0,
            train_fitness=2.0, val_fitness=2.0, combined_fitness=2.0,
        ),
        generation_log=[],
    )
    out_path = tmp_path / "weights.json"
    save_result(result, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "weights" in data
    assert data["weights"]["hot_weight"] == 0.4
    assert "fitness" in data
    assert "baseline" in data
