"""Ending-digit filter."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .base import BaseFilter, FilterDecision


class TailFilter(BaseFilter):
    """Limit repeated ending digits."""

    name = "ending"

    def __init__(self, max_same_tail: int = 2) -> None:
        if max_same_tail <= 0:
            raise ValueError("max_same_tail must be > 0.")
        self.max_same_tail = max_same_tail

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)
        tails = [number % 10 for number in numbers]
        max_count = max(Counter(tails).values())
        if max_count <= self.max_same_tail:
            return FilterDecision(True)
        return FilterDecision(
            False,
            reason=f"same_tail_count={max_count} exceeds {self.max_same_tail}",
        )

