"""Combination scoring and ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CombinationRank:
    """Ranked combination with score."""

    rank: int
    numbers: tuple[int, ...]
    combo_score: float


class CombinationRanker:
    """Compute combo_score = sum(raw_scores[number]) and rank descending."""

    def rank(
        self,
        combinations: Sequence[Sequence[int]],
        raw_scores: Mapping[int, float],
        top_k: int | None = None,
    ) -> list[CombinationRank]:
        """Rank combinations by aggregate score."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")

        scored: list[tuple[tuple[int, ...], float]] = []
        for combination in combinations:
            numbers = tuple(sorted(int(value) for value in combination))
            combo_score = sum(float(raw_scores.get(number, 0.0)) for number in numbers)
            scored.append((numbers, combo_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]

        return [
            CombinationRank(rank=index + 1, numbers=numbers, combo_score=combo_score)
            for index, (numbers, combo_score) in enumerate(scored)
        ]

