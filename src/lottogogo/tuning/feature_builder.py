"""XGBoost feature builder for lottery number ranking.

Transforms engine module outputs into a (round × 45, n_features) feature matrix
for XGBoost training and prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lottogogo.engine.score.calculator import BaseScoreCalculator
from lottogogo.engine.score.booster import BoostCalculator
from lottogogo.engine.score.hmm_scorer import HMMScorer
from lottogogo.engine.score.penalizer import PenaltyCalculator

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_NUMBERS = 45


class FeatureBuildError(Exception):
    """Raised when feature extraction fails."""


FEATURE_NAMES = [
    # Original 15 features
    "base_score",
    "hot_boost",
    "cold_boost",
    "neighbor_boost",
    "carryover_boost",
    "reverse_boost",
    "hmm_hot_prob",
    "hmm_cold_prob",
    "poisson_penalty",
    "markov_penalty",
    "frequency_recent_10",
    "frequency_recent_20",
    "frequency_all",
    "gap_since_last",
    "number_value",
    # New features (13 added)
    "streak_length",  # consecutive appearances
    "pair_freq_recent",  # co-occurrence with last round numbers
    "zone_low",  # 1-15
    "zone_mid",  # 16-30
    "zone_high",  # 31-45
    "lag_1_appeared",  # appeared in previous round
    "lag_2_appeared",  # appeared 2 rounds ago
    "odd_even",  # 1 if odd, 0 if even
    "frequency_variance",  # variance of appearance frequency
    "rank_percentile",  # percentile rank by base_score
    "recency_score",  # weighted recent frequency
    "gap_variance",  # variance of gaps between appearances
    "cycle_phase",  # number % 7 (weekly cycle hypothesis)
]


class FeatureBuilder:
    """Build feature matrices for XGBoost from engine outputs."""

    def __init__(self, history: pd.DataFrame, weights: dict[str, float] | None = None) -> None:
        if history.empty:
            raise FeatureBuildError("history cannot be empty")
        required_cols = ["round"] + NUMBER_COLUMNS
        missing = [c for c in required_cols if c not in history.columns]
        if missing:
            raise FeatureBuildError(f"Missing columns: {missing}")
        self.history = history.sort_values("round").reset_index(drop=True)
        self.weights = weights or {}

    def build(
        self,
        round_range: tuple[int, int],
        min_train_rounds: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and labels for the given round range.

        Args:
            round_range: (start_round, end_round) inclusive
            min_train_rounds: minimum prior rounds needed for feature extraction

        Returns:
            X: shape (n_rounds * 45, n_features)
            y: shape (n_rounds * 45,) — 1 if number was drawn, 0 otherwise
        """
        start, end = round_range
        target_rounds = [
            r for r in self.history["round"].unique()
            if start <= r <= end
        ]

        if not target_rounds:
            raise FeatureBuildError(f"No rounds in range [{start}, {end}]")

        all_X = []
        all_y = []

        for target_round in sorted(target_rounds):
            prior = self.history[self.history["round"] < target_round]
            if len(prior) < min_train_rounds:
                continue

            actual_row = self.history[self.history["round"] == target_round]
            if actual_row.empty:
                continue
            actual_numbers = set(
                int(actual_row.iloc[0][col]) for col in NUMBER_COLUMNS
            )

            features = self._extract_features(prior, target_round, weights=self.weights)
            labels = np.array(
                [1.0 if n in actual_numbers else 0.0 for n in range(1, 46)]
            )

            all_X.append(features)
            all_y.append(labels)

        if not all_X:
            raise FeatureBuildError("No valid rounds for feature extraction")

        return np.vstack(all_X), np.concatenate(all_y)

    def _extract_features(
        self,
        prior: pd.DataFrame,
        target_round: int,
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Extract features for all 45 numbers using prior data.
        
        Args:
            prior: Historical data before target_round
            target_round: Round number to extract features for
            weights: Optional custom weights for boost/HMM calculations

        Returns: shape (45, n_features)
        """
        # Use provided weights or defaults
        w = weights or {}
        
        # Base scores
        base_calc = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
        base_scores = base_calc.calculate_scores(prior, recent_n=50)

        # Boost scores with custom weights
        booster = BoostCalculator(
            hot_threshold=2, hot_window=5,
            hot_weight=w.get("hot_weight", 1.0),
            cold_window=10,
            cold_weight=w.get("cold_weight", 1.0),
            neighbor_weight=w.get("neighbor_weight", 1.0),
            carryover_weight=w.get("carryover_weight", 1.0),
            reverse_weight=w.get("reverse_weight", 1.0),
        )
        boosts, boost_tags = booster.calculate_boosts(prior)

        # Individual boost types from tags
        hot_boost = {}
        cold_boost = {}
        neighbor_boost = {}
        carryover_boost = {}
        reverse_boost = {}
        for n in range(1, 46):
            tags = boost_tags.get(n, [])
            hot_boost[n] = 1.0 if "hot" in tags else 0.0
            cold_boost[n] = 1.0 if "cold" in tags else 0.0
            neighbor_boost[n] = 1.0 if "neighbor" in tags else 0.0
            carryover_boost[n] = 1.0 if "carryover" in tags or "carryover2" in tags else 0.0
            reverse_boost[n] = 1.0 if "reverse" in tags else 0.0

        # HMM scores with custom weights
        try:
            hmm_scorer = HMMScorer(
                hot_boost=w.get("hmm_hot_boost", 1.0),
                cold_boost=w.get("hmm_cold_boost", 1.0),
                window=100
            )
            hmm_boosts, hmm_tags = hmm_scorer.calculate_boosts(prior)
            hmm_hot = {n: 1.0 if "hmm_hot" in hmm_tags.get(n, []) else 0.0 for n in range(1, 46)}
            hmm_cold = {n: 1.0 if "hmm_cold" in hmm_tags.get(n, []) else 0.0 for n in range(1, 46)}
        except Exception:
            hmm_hot = {n: 0.0 for n in range(1, 46)}
            hmm_cold = {n: 0.0 for n in range(1, 46)}

        # Penalty scores
        penalizer = PenaltyCalculator(
            poisson_window=20, poisson_lambda=0.5, markov_lambda=0.3
        )
        poisson = penalizer.calculate_poisson_penalty(prior)
        markov = penalizer.calculate_markov_penalty(prior)

        # Frequency features
        freq_10 = self._frequency(prior, 10)
        freq_20 = self._frequency(prior, 20)
        freq_all = self._frequency(prior, len(prior))

        # Gap since last appearance
        gaps = self._gap_since_last(prior)
        
        # New features (T7)
        streaks = self._streak_length(prior)
        pair_freq = self._pair_frequency(prior)
        lag_1 = self._lag_features(prior, 1)
        lag_2 = self._lag_features(prior, 2)
        freq_var = self._frequency_variance(prior)
        gap_var = self._gap_variance(prior)
        recency = self._recency_score(prior)
        
        # Compute rank percentile from base_scores
        base_values = [base_scores.get(n, 0.0) for n in range(1, 46)]
        rank_percentiles = {}
        for n in range(1, 46):
            score = base_scores.get(n, 0.0)
            percentile = sum(1 for v in base_values if v < score) / 45.0
            rank_percentiles[n] = percentile

        # Build feature matrix: (45, n_features = 28)
        features = np.zeros((TOTAL_NUMBERS, len(FEATURE_NAMES)))
        for i, n in enumerate(range(1, 46)):
            features[i] = [
                # Original 15 features
                base_scores.get(n, 0.0),
                hot_boost.get(n, 0.0),
                cold_boost.get(n, 0.0),
                neighbor_boost.get(n, 0.0),
                carryover_boost.get(n, 0.0),
                reverse_boost.get(n, 0.0),
                hmm_hot.get(n, 0.0),
                hmm_cold.get(n, 0.0),
                poisson.get(n, 0.0),
                markov.get(n, 0.0),
                freq_10.get(n, 0.0),
                freq_20.get(n, 0.0),
                freq_all.get(n, 0.0),
                gaps.get(n, 0.0),
                n / TOTAL_NUMBERS,
                # New 13 features
                streaks.get(n, 0.0),
                pair_freq.get(n, 0.0),
                1.0 if 1 <= n <= 15 else 0.0,  # zone_low
                1.0 if 16 <= n <= 30 else 0.0,  # zone_mid
                1.0 if 31 <= n <= 45 else 0.0,  # zone_high
                lag_1.get(n, 0.0),
                lag_2.get(n, 0.0),
                1.0 if n % 2 == 1 else 0.0,  # odd_even
                freq_var.get(n, 0.0),
                rank_percentiles.get(n, 0.0),
                recency.get(n, 0.0),
                gap_var.get(n, 0.0),
                float(n % 7) / 7.0,  # cycle_phase
            ]
        return features

    @staticmethod
    def _frequency(history: pd.DataFrame, window: int) -> dict[int, float]:
        """Compute normalized appearance frequency in last `window` rounds."""
        recent = history.tail(window)
        counts = {n: 0 for n in range(1, 46)}
        for col in NUMBER_COLUMNS:
            for val in recent[col]:
                v = int(val)
                if 1 <= v <= 45:
                    counts[v] += 1
        total = len(recent) * 6
        return {n: counts[n] / total if total > 0 else 0.0 for n in range(1, 46)}

    @staticmethod
    def _gap_since_last(history: pd.DataFrame) -> dict[int, float]:
        """Compute the gap (number of rounds since last appearance)."""
        ordered = history.sort_values("round")
        max_round = int(ordered["round"].max())
        last_seen = {n: 0 for n in range(1, 46)}

        for _, row in ordered.iterrows():
            r = int(row["round"])
            for col in NUMBER_COLUMNS:
                v = int(row[col])
                if 1 <= v <= 45:
                    last_seen[v] = r

        return {
            n: (max_round - last_seen[n]) / max_round if max_round > 0 else 0.0
            for n in range(1, 46)
        }

    @staticmethod
    def scale_pos_weight(y: np.ndarray) -> float:
        """Compute scale_pos_weight for imbalanced binary classification."""
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        return float(n_neg / n_pos) if n_pos > 0 else 1.0
    
    # New feature extraction methods for T7
    @staticmethod
    def _streak_length(history: pd.DataFrame) -> dict[int, float]:
        """Compute consecutive appearance streak length."""
        ordered = history.sort_values("round")
        streaks = {n: 0 for n in range(1, 46)}
        current_streak = {n: 0 for n in range(1, 46)}
        
        for _, row in ordered.iterrows():
            appeared = {int(row[col]) for col in NUMBER_COLUMNS if 1 <= int(row[col]) <= 45}
            for n in range(1, 46):
                if n in appeared:
                    current_streak[n] += 1
                    streaks[n] = max(streaks[n], current_streak[n])
                else:
                    current_streak[n] = 0
        return {n: float(streaks[n]) / 10.0 for n in range(1, 46)}  # Normalize by max expected streak
    
    @staticmethod
    def _pair_frequency(history: pd.DataFrame) -> dict[int, float]:
        """Co-occurrence frequency with last round numbers."""
        if len(history) < 2:
            return {n: 0.0 for n in range(1, 46)}
        
        ordered = history.sort_values("round")
        last_round = ordered.tail(1).iloc[0]
        last_numbers = {int(last_round[col]) for col in NUMBER_COLUMNS if 1 <= int(last_round[col]) <= 45}
        
        pair_counts = {n: 0 for n in range(1, 46)}
        for _, row in ordered[:-1].iterrows():
            current_numbers = {int(row[col]) for col in NUMBER_COLUMNS if 1 <= int(row[col]) <= 45}
            for n in range(1, 46):
                if n in current_numbers and len(current_numbers & last_numbers) > 0:
                    pair_counts[n] += 1
        
        total = len(ordered) - 1
        return {n: pair_counts[n] / total if total > 0 else 0.0 for n in range(1, 46)}
    
    @staticmethod
    def _lag_features(history: pd.DataFrame, lag: int) -> dict[int, float]:
        """Check if number appeared N rounds ago."""
        if len(history) < lag:
            return {n: 0.0 for n in range(1, 46)}
        
        ordered = history.sort_values("round")
        lag_round = ordered.tail(lag).iloc[0]
        lag_numbers = {int(lag_round[col]) for col in NUMBER_COLUMNS if 1 <= int(lag_round[col]) <= 45}
        
        return {n: 1.0 if n in lag_numbers else 0.0 for n in range(1, 46)}
    
    @staticmethod
    def _frequency_variance(history: pd.DataFrame) -> dict[int, float]:
        """Variance of appearance frequency over time windows."""
        windows = [10, 20, 50]
        variances = {n: [] for n in range(1, 46)}
        
        for window in windows:
            if len(history) >= window:
                freq = FeatureBuilder._frequency(history, window)
                for n in range(1, 46):
                    variances[n].append(freq[n])
        
        return {n: float(np.var(variances[n])) if variances[n] else 0.0 for n in range(1, 46)}
    
    @staticmethod
    def _gap_variance(history: pd.DataFrame) -> dict[int, float]:
        """Variance of gaps between appearances."""
        ordered = history.sort_values("round")
        gaps_list = {n: [] for n in range(1, 46)}
        last_seen = {n: 0 for n in range(1, 46)}
        
        for _, row in ordered.iterrows():
            r = int(row["round"])
            for col in NUMBER_COLUMNS:
                v = int(row[col])
                if 1 <= v <= 45:
                    if last_seen[v] > 0:
                        gaps_list[v].append(r - last_seen[v])
                    last_seen[v] = r
        
        return {n: float(np.var(gaps_list[n])) if len(gaps_list[n]) > 1 else 0.0 for n in range(1, 46)}
    
    @staticmethod
    def _recency_score(history: pd.DataFrame) -> dict[int, float]:
        """Weighted recent frequency (more recent = higher weight)."""
        ordered = history.sort_values("round")
        scores = {n: 0.0 for n in range(1, 46)}
        total_rounds = len(ordered)
        
        for idx, (_, row) in enumerate(ordered.iterrows()):
            weight = (idx + 1) / total_rounds  # Linear weight: recent rounds have higher weight
            for col in NUMBER_COLUMNS:
                v = int(row[col])
                if 1 <= v <= 45:
                    scores[v] += weight
        
        max_score = max(scores.values()) if scores.values() else 1.0
        return {n: scores[n] / max_score if max_score > 0 else 0.0 for n in range(1, 46)}

