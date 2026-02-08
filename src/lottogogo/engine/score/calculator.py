"""Base score calculation with Beta posterior mean."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from scipy.stats import beta

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


class BaseScoreCalculator:
    """Calculate number-level base scores using a Beta-Bernoulli model."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0.")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def posterior_params(self, successes: int, total_rounds: int) -> tuple[float, float]:
        """Return posterior alpha/beta after Bayesian update."""
        if successes < 0:
            raise ValueError("successes cannot be negative.")
        if total_rounds <= 0:
            raise ValueError("total_rounds must be greater than 0.")
        if successes > total_rounds:
            raise ValueError("successes cannot exceed total_rounds.")

        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + (total_rounds - successes)
        return posterior_alpha, posterior_beta

    def posterior_mean(self, successes: int, total_rounds: int) -> float:
        """Compute posterior mean for a single number."""
        posterior_alpha, posterior_beta = self.posterior_params(successes, total_rounds)
        return float(beta.mean(posterior_alpha, posterior_beta))

    def calculate_scores(self, history: pd.DataFrame, recent_n: int | None = None) -> dict[int, float]:
        """Calculate base score for each number 1~45 from history."""
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")

        missing = [column for column in NUMBER_COLUMNS if column not in history.columns]
        if missing:
            raise ValueError(f"Missing number columns: {missing}")

        working = history.copy()
        if recent_n is not None:
            if recent_n <= 0:
                raise ValueError("recent_n must be > 0")
            if "round" in working.columns:
                working = working.sort_values("round")
            working = working.tail(recent_n)

        total_rounds = len(working)
        counts = self._appearance_counts(working)

        return {
            number: self.posterior_mean(successes=counts.get(number, 0), total_rounds=total_rounds)
            for number in range(1, 46)
        }

    @staticmethod
    def _appearance_counts(history: pd.DataFrame) -> Mapping[int, int]:
        counts = {number: 0 for number in range(1, 46)}

        for column in NUMBER_COLUMNS:
            for value, freq in history[column].astype(int).value_counts().to_dict().items():
                if 1 <= int(value) <= 45:
                    counts[int(value)] += int(freq)
        return counts


class ScoreEnsembler:
    """Combine base/boost/penalty layers into final non-negative raw scores."""

    def __init__(self, minimum_score: float = 0.0) -> None:
        if minimum_score < 0:
            raise ValueError("minimum_score must be >= 0.")
        self.minimum_score = float(minimum_score)

    def combine(
        self,
        base_scores: Mapping[int, float],
        boost_scores: Mapping[int, float],
        penalties: Mapping[int, float],
    ) -> dict[int, float]:
        """Combine scores with formula: base + boost - penalty."""
        keys = set(base_scores.keys()) | set(boost_scores.keys()) | set(penalties.keys())
        combined: dict[int, float] = {}
        for key in keys:
            raw = float(base_scores.get(key, 0.0)) + float(boost_scores.get(key, 0.0)) - float(
                penalties.get(key, 0.0)
            )
            combined[key] = max(self.minimum_score, raw)
        return combined

    @staticmethod
    def normalize(raw_scores: Mapping[int, float]) -> dict[int, float]:
        """Normalize raw scores so the sum becomes 1.0."""
        if not raw_scores:
            raise ValueError("raw_scores cannot be empty.")

        total = float(sum(raw_scores.values()))
        if total <= 0:
            uniform = 1.0 / len(raw_scores)
            return {key: uniform for key in raw_scores}

        return {key: float(value) / total for key, value in raw_scores.items()}
