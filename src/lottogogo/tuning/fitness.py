"""Fitness evaluation for GA weight optimization.

Evaluates a weight vector by running time-sequential backtesting
against the existing engine pipeline and computing hit@K metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

import numpy as np
import pandas as pd

from lottogogo.engine.score.calculator import BaseScoreCalculator, ScoreEnsembler
from lottogogo.engine.score.booster import BoostCalculator
from lottogogo.engine.score.hmm_scorer import HMMScorer
from lottogogo.engine.score.penalizer import PenaltyCalculator
from lottogogo.engine.score.normalizer import ProbabilityNormalizer

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_NUMBERS = 45


class FitnessEvaluationError(Exception):
    """Raised when fitness evaluation fails."""


@dataclass
class FitnessResult:
    """Result of a single fitness evaluation."""

    hit_at_15: float
    hit_at_20: float
    mean_rank: float
    train_fitness: float
    val_fitness: float
    combined_fitness: float  # 0.6 * train + 0.4 * val


# Chromosome key names and their valid ranges.
#
# NOTE: `temperature` used to live here and was removed deliberately.
# Fitness is computed from the *ranking* of raw scores (see `_hit_at_k` /
# `_mean_rank`), and `CachedScoreComputer.compute` returns raw scores without
# ever applying the softmax. Temperature therefore had zero gradient in this
# objective: GA could not optimize it, so it drifted to whatever value random
# initialization plus mutation happened to leave it at — in the last run, the
# lower bound 0.1. That value was then consumed by recommend.py and
# build_frontend_model.py, where it *does* matter, collapsing ~31% of the
# sampling mass onto a single number.
#
# Sampling temperature is now a fixed constant
# (`lottogogo.engine.score.normalizer.DEFAULT_TEMPERATURE`) or a per-preset
# value (`mvp.service.PresetConfig.temperature`). If it should ever be tuned
# again, the fitness function must first be changed to actually sample.
WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "hot_weight": (0.0, 1.0),
    "cold_weight": (0.0, 0.5),
    "neighbor_weight": (0.0, 1.0),
    "carryover_weight": (0.0, 1.0),
    "reverse_weight": (0.0, 0.5),
    "hmm_hot_boost": (0.0, 1.0),
    "hmm_cold_boost": (0.0, 0.5),
    "poisson_lambda": (0.0, 0.5),
    "markov_lambda": (0.0, 0.5),
}

# HMM-disabled configuration (7D instead of 9D)
WEIGHT_BOUNDS_NO_HMM: dict[str, tuple[float, float]] = {
    "hot_weight": (0.0, 1.0),
    "cold_weight": (0.0, 0.5),
    "neighbor_weight": (0.0, 1.0),
    "carryover_weight": (0.0, 1.0),
    "reverse_weight": (0.0, 0.5),
    "poisson_lambda": (0.0, 0.5),
    "markov_lambda": (0.0, 0.5),
}

WEIGHT_KEYS = list(WEIGHT_BOUNDS.keys())


def random_baseline(k: int = 15) -> float:
    """Theoretical expected hit@K for random selection.

    When choosing K numbers out of 45, expected overlap with 6 winning numbers:
    E[hit@K] = K * 6 / 45
    """
    return k * 6 / TOTAL_NUMBERS


BOOST_WEIGHT_KEYS = (
    "hot_weight",
    "cold_weight",
    "neighbor_weight",
    "carryover_weight",
    "reverse_weight",
)
_NUMBERS = tuple(range(1, TOTAL_NUMBERS + 1))
_UNIT_LAMBDA = 0.5  # largest value PenaltyCalculator accepts


def _compute_scores(
    history: pd.DataFrame,
    weights: dict[str, float],
) -> dict[int, float]:
    """Run the engine pipeline with given weights and return raw scores.

    This is the reference implementation. `CachedScoreComputer` must return
    the same scores; it only avoids redoing the weight-independent work.
    """
    booster = BoostCalculator(
        hot_threshold=2,
        hot_window=5,
        cold_window=10,
        **{key: weights[key] for key in BOOST_WEIGHT_KEYS},
    )
    penalizer = PenaltyCalculator(
        poisson_window=20,
        poisson_lambda=weights["poisson_lambda"],
        markov_lambda=weights["markov_lambda"],
    )
    base_scores = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0).calculate_scores(
        history, recent_n=50
    )
    boosts, _ = booster.calculate_boosts(history)  # HMM boosts are skipped
    penalties = penalizer.calculate_penalties(history)
    return ScoreEnsembler(minimum_score=0.0).combine(base_scores, boosts, penalties)


class CachedScoreComputer:
    """Score computer that does the weight-independent work only once.

    Every layer is linear in its weight (`base + sum(w * boost) - sum(l *
    penalty)`, floored at 0), so the engine is run once per unit weight and
    each later `compute` call is a small weighted sum. The GA evaluates
    thousands of weight vectors against the same histories, and rebuilding the
    Markov matrix for each one made a full run take days.
    """

    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history
        self._components: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
        self._lock = threading.Lock()  # the GA evaluates from several threads

    def _build_components(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        def as_vector(by_number: dict[int, float]) -> np.ndarray:
            return np.array([by_number[number] for number in _NUMBERS], dtype=float)

        history = self.history
        base = as_vector(
            BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0).calculate_scores(
                history, recent_n=50
            )
        )
        # Signed unit responses: boosts add, penalties subtract.
        units: dict[str, np.ndarray] = {}
        for key in BOOST_WEIGHT_KEYS:
            unit_weights = {name: float(name == key) for name in BOOST_WEIGHT_KEYS}
            boosts, _ = BoostCalculator(
                hot_threshold=2, hot_window=5, cold_window=10, **unit_weights
            ).calculate_boosts(history)
            units[key] = as_vector(boosts)

        penalizer = PenaltyCalculator(
            poisson_window=20, poisson_lambda=_UNIT_LAMBDA, markov_lambda=_UNIT_LAMBDA
        )
        units["poisson_lambda"] = -as_vector(penalizer.calculate_poisson_penalty(history)) / _UNIT_LAMBDA
        units["markov_lambda"] = -as_vector(penalizer.calculate_markov_penalty(history)) / _UNIT_LAMBDA
        return base, units

    def compute(self, weights: dict[str, float]) -> dict[int, float]:
        """Compute raw scores for the given weights."""
        with self._lock:
            if self._components is None:
                self._components = self._build_components()
        base, units = self._components

        raw = base.copy()
        for key, unit in units.items():
            raw += weights[key] * unit
        np.maximum(raw, 0.0, out=raw)
        return dict(zip(_NUMBERS, raw.tolist()))


def _hit_at_k(
    scores: dict[int, float],
    actual_numbers: set[int],
    k: int,
) -> int:
    """Count how many of the top-K scored numbers are in actual_numbers."""
    ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    top_k = set(ranked[:k])
    return len(top_k & actual_numbers)


def _mean_rank(scores: dict[int, float], actual_numbers: set[int]) -> float:
    """Average rank of the actual winning numbers (1-indexed, lower is better)."""
    ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    rank_map = {n: i + 1 for i, n in enumerate(ranked)}
    ranks = [rank_map[n] for n in actual_numbers if n in rank_map]
    return float(np.mean(ranks)) if ranks else float(TOTAL_NUMBERS / 2)


class FitnessEvaluator:
    """Evaluate a weight vector using time-sequential backtesting."""

    def __init__(
        self,
        history: pd.DataFrame,
        train_end: int,
        val_end: int,
    ) -> None:
        if history.empty:
            raise FitnessEvaluationError("history cannot be empty")
        if train_end <= 0 or val_end <= train_end:
            raise FitnessEvaluationError(
                f"Invalid range: train_end={train_end}, val_end={val_end}"
            )
        self.history = history.copy()
        if "round" in self.history.columns:
            self.history = self.history.sort_values("round").reset_index(drop=True)
        self.train_end = train_end
        self.val_end = val_end
        
        # Per target round: score computer over the prior rounds + actual numbers
        self._round_cache: dict[int, tuple[CachedScoreComputer, set[int]] | None] = {}
        self._round_cache_lock = threading.Lock()

    def evaluate(self, weights: dict[str, float]) -> FitnessResult:
        """Evaluate a weight vector.

        Runs rolling backtest: for each validation round t, uses rounds 1..(t-1)
        for scoring and checks hit@K against round t's actual numbers.
        """
        self._validate_weights(weights)

        history = self.history
        if "round" not in history.columns:
            raise FitnessEvaluationError("history must have 'round' column")

        # Split into train and validation rounds
        all_rounds = sorted(history["round"].unique())
        train_rounds = [r for r in all_rounds if r <= self.train_end]
        val_rounds = [r for r in all_rounds if self.train_end < r <= self.val_end]

        if len(train_rounds) < 50:
            raise FitnessEvaluationError(
                f"Insufficient training data: {len(train_rounds)} rounds (need >= 50)"
            )
        if len(val_rounds) < 10:
            raise FitnessEvaluationError(
                f"Insufficient validation data: {len(val_rounds)} rounds (need >= 10)"
            )

        # Evaluate on train (sample for speed)
        train_sample = train_rounds[-20:]  # last 20 of training (reduced from 100 for speed)
        train_hits_15, _, _ = self._evaluate_window(train_sample, weights)

        # Evaluate on validation
        val_hits_15, val_hits_20, val_ranks = self._evaluate_window(val_rounds, weights)

        train_fitness = float(np.mean(train_hits_15))
        val_fitness = float(np.mean(val_hits_15))
        mean_rank = float(np.mean(val_ranks))
        
        # Redesigned fitness: increase val weight, add rank bonus
        # Rank bonus: (45 - mean_rank) / 45 ∈ [0, 1], higher is better
        rank_bonus = (TOTAL_NUMBERS - mean_rank) / TOTAL_NUMBERS
        combined = 0.4 * train_fitness + 0.5 * val_fitness + 0.1 * rank_bonus

        return FitnessResult(
            hit_at_15=float(np.mean(val_hits_15)),
            hit_at_20=float(np.mean(val_hits_20)),
            mean_rank=float(np.mean(val_ranks)),
            train_fitness=train_fitness,
            val_fitness=val_fitness,
            combined_fitness=combined,
        )

    def _round_context(self, target_round: int) -> tuple[CachedScoreComputer, set[int]] | None:
        """Score computer over the rounds before `target_round`, plus its actual numbers."""
        with self._round_cache_lock:
            return self._round_context_locked(target_round)

    def _round_context_locked(self, target_round: int) -> tuple[CachedScoreComputer, set[int]] | None:
        if target_round not in self._round_cache:
            history = self.history
            train_data = history[history["round"] < target_round]
            actual_row = history[history["round"] == target_round]
            if len(train_data) < 20 or actual_row.empty:
                self._round_cache[target_round] = None
            else:
                actual_numbers = set(int(actual_row.iloc[0][col]) for col in NUMBER_COLUMNS)
                self._round_cache[target_round] = (CachedScoreComputer(train_data), actual_numbers)
        return self._round_cache[target_round]

    def _evaluate_window(
        self,
        target_rounds: list[int],
        weights: dict[str, float],
    ) -> tuple[list[int], list[int], list[float]]:
        """Compute hit@15, hit@20 and mean rank for each target round using prior data."""
        hits_15: list[int] = []
        hits_20: list[int] = []
        ranks: list[float] = []
        for target_round in target_rounds:
            context = self._round_context(target_round)
            if context is None:
                continue
            computer, actual_numbers = context
            try:
                scores = computer.compute(weights)
                hits_15.append(_hit_at_k(scores, actual_numbers, 15))
                hits_20.append(_hit_at_k(scores, actual_numbers, 20))
                ranks.append(_mean_rank(scores, actual_numbers))
            except Exception:
                hits_15.append(0)
                hits_20.append(0)
                ranks.append(float(TOTAL_NUMBERS / 2))
        return (
            hits_15 if hits_15 else [0],
            hits_20 if hits_20 else [0],
            ranks if ranks else [float(TOTAL_NUMBERS / 2)],
        )

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> None:
        """Validate weight keys and bounds."""
        # Validate only the keys present in weights (supports both full and no-HMM sets)
        for key, value in weights.items():
            if key in WEIGHT_BOUNDS:
                lo, hi = WEIGHT_BOUNDS[key]
                if not (lo <= value <= hi):
                    raise FitnessEvaluationError(
                        f"Weight {key}={value} out of bounds [{lo}, {hi}]"
                    )
