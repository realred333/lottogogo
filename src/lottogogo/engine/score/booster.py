"""Heuristic boost layer for number scores."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


class BoostCalculator:
    """Calculate heuristic boosts for numbers 1~45."""

    def __init__(
        self,
        hot_threshold: int = 2,
        hot_window: int = 5,
        hot_weight: float = 0.4,
        cold_window: int = 10,
        cold_weight: float = 0.15,
        neighbor_weight: float = 0.3,
        carryover_weight: float = 0.2,
        reverse_weight: float = 0.1,
    ) -> None:
        if hot_threshold <= 0:
            raise ValueError("hot_threshold must be > 0")
        if hot_window <= 0 or cold_window <= 0:
            raise ValueError("hot_window and cold_window must be > 0")
        if min(hot_weight, cold_weight, neighbor_weight, carryover_weight, reverse_weight) < 0:
            raise ValueError("All boost weights must be >= 0")

        self.hot_threshold = hot_threshold
        self.hot_window = hot_window
        self.hot_weight = float(hot_weight)
        self.cold_window = cold_window
        self.cold_weight = float(cold_weight)
        self.neighbor_weight = float(neighbor_weight)
        self.carryover_weight = float(carryover_weight)
        self.reverse_weight = float(reverse_weight)

    def calculate_boosts(self, history: pd.DataFrame) -> tuple[dict[int, float], dict[int, list[str]]]:
        """Return per-number boost score and active boost labels."""
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")
        self._ensure_columns(history)

        ordered = self._ordered_history(history)
        boosts = {number: 0.0 for number in range(1, 46)}
        tags = {number: [] for number in range(1, 46)}

        self._apply_hot_cold(ordered=ordered, boosts=boosts, tags=tags)
        self._apply_neighbor_carryover(ordered=ordered, boosts=boosts, tags=tags)
        self._apply_reverse(ordered=ordered, boosts=boosts, tags=tags)

        return boosts, tags

    def _apply_hot_cold(
        self, ordered: pd.DataFrame, boosts: dict[int, float], tags: dict[int, list[str]]
    ) -> None:
        hot_counts = self._appearance_counts(ordered.tail(self.hot_window))
        cold_counts = self._appearance_counts(ordered.tail(self.cold_window))

        for number in range(1, 46):
            if hot_counts.get(number, 0) >= self.hot_threshold:
                boosts[number] += self.hot_weight
                tags[number].append("hot")
            if cold_counts.get(number, 0) == 0:
                boosts[number] += self.cold_weight
                tags[number].append("cold")

    def _apply_neighbor_carryover(
        self, ordered: pd.DataFrame, boosts: dict[int, float], tags: dict[int, list[str]]
    ) -> None:
        last_round_numbers = set(
            int(value)
            for value in ordered.tail(1)[NUMBER_COLUMNS].astype(int).values.flatten().tolist()
            if 1 <= int(value) <= 45
        )

        neighbor_candidates: set[int] = set()
        for number in last_round_numbers:
            for candidate in (number - 1, number + 1):
                if 1 <= candidate <= 45 and candidate not in last_round_numbers:
                    neighbor_candidates.add(candidate)

        for number in range(1, 46):
            if number in last_round_numbers:
                boosts[number] += self.carryover_weight
                tags[number].append("carryover")
            if number in neighbor_candidates:
                boosts[number] += self.neighbor_weight
                tags[number].append("neighbor")

    def _apply_reverse(
        self, ordered: pd.DataFrame, boosts: dict[int, float], tags: dict[int, list[str]]
    ) -> None:
        last_round_numbers = set(
            int(value)
            for value in ordered.tail(1)[NUMBER_COLUMNS].astype(int).values.flatten().tolist()
            if 1 <= int(value) <= 45
        )
        reverse_candidates = {46 - number for number in last_round_numbers}

        for number in range(1, 46):
            if number in reverse_candidates:
                boosts[number] += self.reverse_weight
                tags[number].append("reverse")

    @staticmethod
    def _ordered_history(history: pd.DataFrame) -> pd.DataFrame:
        ordered = history.copy()
        if "round" in ordered.columns:
            ordered = ordered.sort_values("round")
        return ordered

    @staticmethod
    def _ensure_columns(history: pd.DataFrame) -> None:
        missing = [column for column in NUMBER_COLUMNS if column not in history.columns]
        if missing:
            raise ValueError(f"Missing number columns: {missing}")

    @staticmethod
    def _appearance_counts(history: pd.DataFrame) -> Mapping[int, int]:
        counts = {number: 0 for number in range(1, 46)}
        for column in NUMBER_COLUMNS:
            for value, freq in history[column].astype(int).value_counts().to_dict().items():
                number = int(value)
                if 1 <= number <= 45:
                    counts[number] += int(freq)
        return counts
