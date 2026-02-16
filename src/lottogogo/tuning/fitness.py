"""Fitness evaluation for GA weight optimization.

Evaluates a weight vector by running time-sequential backtesting
against the existing engine pipeline and computing hit@K metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# Chromosome key names and their valid ranges
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
    "temperature": (0.1, 2.0),
}

# HMM-disabled configuration (8D instead of 10D)
WEIGHT_BOUNDS_NO_HMM: dict[str, tuple[float, float]] = {
    "hot_weight": (0.0, 1.0),
    "cold_weight": (0.0, 0.5),
    "neighbor_weight": (0.0, 1.0),
    "carryover_weight": (0.0, 1.0),
    "reverse_weight": (0.0, 0.5),
    "poisson_lambda": (0.0, 0.5),
    "markov_lambda": (0.0, 0.5),
    "temperature": (0.1, 2.0),
}

WEIGHT_KEYS = list(WEIGHT_BOUNDS.keys())


def random_baseline(k: int = 15) -> float:
    """Theoretical expected hit@K for random selection.

    When choosing K numbers out of 45, expected overlap with 6 winning numbers:
    E[hit@K] = K * 6 / 45
    """
    return k * 6 / TOTAL_NUMBERS


class CachedScoreComputer:
    """Reusable score computer with cached calculator instances."""
    
    def __init__(self, history: pd.DataFrame) -> None:
        """Initialize with history and create reusable calculator instances."""
        self.history = history
        # Pre-create calculator instances (reused across evaluations)
        self.base_calc = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
        self.ensembler = ScoreEnsembler(minimum_score=0.0)
    
    def compute(self, weights: dict[str, float]) -> dict[int, float]:
        """Compute scores with given weights using cached instances."""
        # Create weight-dependent calculators (lightweight)
        booster = BoostCalculator(
            hot_threshold=2,
            hot_window=5,
            hot_weight=weights["hot_weight"],
            cold_window=10,
            cold_weight=weights["cold_weight"],
            neighbor_weight=weights["neighbor_weight"],
            carryover_weight=weights["carryover_weight"],
            reverse_weight=weights["reverse_weight"],
        )
        penalizer = PenaltyCalculator(
            poisson_window=20,
            poisson_lambda=weights["poisson_lambda"],
            markov_lambda=weights["markov_lambda"],
        )
        
        base_scores = self.base_calc.calculate_scores(self.history, recent_n=50)
        boosts, _ = booster.calculate_boosts(self.history)
        
        # HMM scorer (DISABLED for performance)
        combined_boosts = boosts  # Skip HMM boosts
        
        penalties = penalizer.calculate_penalties(self.history)
        raw_scores = self.ensembler.combine(base_scores, combined_boosts, penalties)
        return raw_scores


def _compute_scores(
    history: pd.DataFrame,
    weights: dict[str, float],
) -> dict[int, float]:
    """Run the engine pipeline with given weights and return raw scores.
    
    NOTE: This function is kept for backward compatibility.
    For performance, use CachedScoreComputer instead.
    """
    computer = CachedScoreComputer(history)
    return computer.compute(weights)


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
        
        # Cache for CachedScoreComputer instances (keyed by history hash)
        self._score_computer_cache: dict[int, CachedScoreComputer] = {}

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
        train_hits_15 = self._evaluate_rounds(history, train_sample, weights, k=15)
        train_hits_20 = self._evaluate_rounds(history, train_sample, weights, k=20)

        # Evaluate on validation
        val_hits_15 = self._evaluate_rounds(history, val_rounds, weights, k=15)
        val_hits_20 = self._evaluate_rounds(history, val_rounds, weights, k=20)
        val_ranks = self._evaluate_ranks(history, val_rounds, weights)

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

    def _evaluate_rounds(
        self,
        history: pd.DataFrame,
        target_rounds: list[int],
        weights: dict[str, float],
        k: int,
    ) -> list[int]:
        """Compute hit@K for each target round using prior data."""
        hits: list[int] = []
        for target_round in target_rounds:
            train_data = history[history["round"] < target_round]
            if len(train_data) < 20:
                continue
            actual_row = history[history["round"] == target_round]
            if actual_row.empty:
                continue
            actual_numbers = set(
                int(actual_row.iloc[0][col]) for col in NUMBER_COLUMNS
            )
            try:
                # Use cached score computer for this training data
                data_hash = hash(tuple(train_data["round"].values))
                if data_hash not in self._score_computer_cache:
                    self._score_computer_cache[data_hash] = CachedScoreComputer(train_data)
                
                scores = self._score_computer_cache[data_hash].compute(weights)
                hits.append(_hit_at_k(scores, actual_numbers, k))
            except Exception:
                hits.append(0)
        return hits if hits else [0]

    def _evaluate_ranks(
        self,
        history: pd.DataFrame,
        target_rounds: list[int],
        weights: dict[str, float],
    ) -> list[float]:
        """Compute mean rank for each target round."""
        ranks: list[float] = []
        for target_round in target_rounds:
            train_data = history[history["round"] < target_round]
            if len(train_data) < 20:
                continue
            actual_row = history[history["round"] == target_round]
            if actual_row.empty:
                continue
            actual_numbers = set(
                int(actual_row.iloc[0][col]) for col in NUMBER_COLUMNS
            )
            try:
                # Use cached score computer for this training data
                data_hash = hash(tuple(train_data["round"].values))
                if data_hash not in self._score_computer_cache:
                    self._score_computer_cache[data_hash] = CachedScoreComputer(train_data)
                
                scores = self._score_computer_cache[data_hash].compute(weights)
                ranks.append(_mean_rank(scores, actual_numbers))
            except Exception:
                ranks.append(float(TOTAL_NUMBERS / 2))
        return ranks if ranks else [float(TOTAL_NUMBERS / 2)]

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
