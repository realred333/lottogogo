"""Penalty layer for over-represented number patterns."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_NUMBERS = 45
NUMBERS_PER_ROUND = 6


class PenaltyCalculator:
    """Calculate Poisson/Markov style penalties."""

    def __init__(
        self,
        poisson_window: int = 20,
        poisson_lambda: float = 0.5,
        markov_lambda: float = 0.3,
    ) -> None:
        if poisson_window <= 0:
            raise ValueError("poisson_window must be > 0.")
        if poisson_lambda < 0 or markov_lambda < 0:
            raise ValueError("penalty lambdas must be >= 0.")
        if poisson_lambda > 0.5 or markov_lambda > 0.5:
            raise ValueError("penalty lambdas must be <= 0.5 to avoid over-penalization.")

        self.poisson_window = poisson_window
        self.poisson_lambda = float(poisson_lambda)
        self.markov_lambda = float(markov_lambda)

    def calculate_poisson_penalty(self, history: pd.DataFrame) -> dict[int, float]:
        """Calculate Poisson-like penalties from recent appearance counts."""
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")
        self._ensure_columns(history)

        ordered = self._ordered_history(history)
        windowed = ordered.tail(self.poisson_window)
        counts = self._appearance_counts(windowed)

        effective_window = len(windowed)
        expected_mean = (effective_window * NUMBERS_PER_ROUND) / TOTAL_NUMBERS
        penalties: dict[int, float] = {}
        for number in range(1, TOTAL_NUMBERS + 1):
            excess = max(0.0, counts.get(number, 0) - expected_mean)
            penalties[number] = self.poisson_lambda * excess
        return penalties

    def build_transition_matrix(self, history: pd.DataFrame) -> np.ndarray:
        """Build a row-normalized Markov transition matrix (46x46, index 1~45 used)."""
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")
        self._ensure_columns(history)

        ordered = self._ordered_history(history)
        matrix = np.zeros((TOTAL_NUMBERS + 1, TOTAL_NUMBERS + 1), dtype=float)

        rows = [self._row_numbers(row) for _, row in ordered.iterrows()]
        for previous, current in zip(rows, rows[1:]):
            for prev_number in previous:
                for curr_number in current:
                    matrix[prev_number, curr_number] += 1.0

        row_sums = matrix.sum(axis=1)
        nonzero_rows = row_sums > 0
        matrix[nonzero_rows] = matrix[nonzero_rows] / row_sums[nonzero_rows, None]
        return matrix

    def calculate_markov_penalty(self, history: pd.DataFrame) -> dict[int, float]:
        """Calculate transition-probability based Markov penalties."""
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")
        self._ensure_columns(history)

        ordered = self._ordered_history(history)
        if len(ordered) < 2:
            return {number: 0.0 for number in range(1, TOTAL_NUMBERS + 1)}

        matrix = self.build_transition_matrix(ordered)
        last_round_numbers = self._row_numbers(ordered.tail(1).iloc[0])

        penalties: dict[int, float] = {}
        for number in range(1, TOTAL_NUMBERS + 1):
            transition_probs = [float(matrix[prev_number, number]) for prev_number in last_round_numbers]
            mean_probability = float(np.mean(transition_probs)) if transition_probs else 0.0
            penalties[number] = self.markov_lambda * mean_probability
        return penalties

    def calculate_penalties(self, history: pd.DataFrame) -> dict[int, float]:
        """Return combined Poisson + Markov penalties."""
        poisson = self.calculate_poisson_penalty(history)
        markov = self.calculate_markov_penalty(history)
        return {number: poisson[number] + markov[number] for number in range(1, TOTAL_NUMBERS + 1)}

    @staticmethod
    def _ensure_columns(history: pd.DataFrame) -> None:
        missing = [column for column in NUMBER_COLUMNS if column not in history.columns]
        if missing:
            raise ValueError(f"Missing number columns: {missing}")

    @staticmethod
    def _ordered_history(history: pd.DataFrame) -> pd.DataFrame:
        ordered = history.copy()
        if "round" in ordered.columns:
            ordered = ordered.sort_values("round")
        return ordered

    @staticmethod
    def _appearance_counts(history: pd.DataFrame) -> Mapping[int, int]:
        counts = {number: 0 for number in range(1, TOTAL_NUMBERS + 1)}
        for column in NUMBER_COLUMNS:
            for value, freq in history[column].astype(int).value_counts().to_dict().items():
                number = int(value)
                if 1 <= number <= TOTAL_NUMBERS:
                    counts[number] += int(freq)
        return counts

    @staticmethod
    def _row_numbers(row: pd.Series) -> set[int]:
        return {
            int(row[column])
            for column in NUMBER_COLUMNS
            if 1 <= int(row[column]) <= TOTAL_NUMBERS
        }
