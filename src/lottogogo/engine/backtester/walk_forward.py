"""Walk-forward backtesting engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]

RecommendationFn = Callable[[pd.DataFrame, int, int | None], Sequence[Sequence[int]]]


@dataclass(frozen=True)
class RoundBacktestResult:
    """Per-round backtest outcome."""

    round_id: int
    winning_numbers: tuple[int, ...]
    recommendations: list[tuple[int, ...]]
    hit_counts: list[int]


class WalkForwardBacktester:
    """Execute sequential walk-forward validation."""

    def run(
        self,
        *,
        history: pd.DataFrame,
        recommender: RecommendationFn,
        start_round: int | None = None,
        end_round: int | None = None,
        recent_n: int | None = None,
        output_count: int = 5,
        seed: int | None = 42,
    ) -> list[RoundBacktestResult]:
        """Run walk-forward loop and accumulate round-level results."""
        if output_count <= 0:
            raise ValueError("output_count must be > 0.")
        if history.empty:
            raise ValueError("history dataframe cannot be empty.")
        missing = [column for column in ["round", *NUMBER_COLUMNS] if column not in history.columns]
        if missing:
            raise ValueError(f"Missing required history columns: {missing}")

        ordered = history.reset_index(drop=True).sort_values("round").reset_index(drop=True).copy()
        results: list[RoundBacktestResult] = []

        for idx, row in ordered.iterrows():
            round_id = int(row["round"])
            if start_round is not None and round_id < start_round:
                continue
            if end_round is not None and round_id > end_round:
                continue
            if idx == 0:
                continue

            train_df = ordered.iloc[:idx].copy()
            if recent_n is not None:
                if recent_n <= 0:
                    raise ValueError("recent_n must be > 0 when provided.")
                train_df = train_df.tail(recent_n)

            round_seed = None if seed is None else seed + idx
            recommendations = recommender(train_df, output_count, round_seed)
            normalized = [tuple(sorted(int(value) for value in combo)) for combo in recommendations]
            winning_numbers = tuple(sorted(int(row[column]) for column in NUMBER_COLUMNS))
            winning_set = set(winning_numbers)
            hit_counts = [len(winning_set.intersection(combo)) for combo in normalized]

            results.append(
                RoundBacktestResult(
                    round_id=round_id,
                    winning_numbers=winning_numbers,
                    recommendations=normalized,
                    hit_counts=hit_counts,
                )
            )

        return results
