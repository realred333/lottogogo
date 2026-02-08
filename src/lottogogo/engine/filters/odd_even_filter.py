"""Odd-even balance filter."""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseFilter, FilterDecision


class OddEvenFilter(BaseFilter):
    """Allow only combinations with odd count in configured range."""

    name = "odd_even"

    def __init__(self, min_odd: int = 2, max_odd: int = 4) -> None:
        if min_odd < 0 or max_odd > 6 or min_odd > max_odd:
            raise ValueError("Invalid odd count range.")
        self.min_odd = min_odd
        self.max_odd = max_odd

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)
        odd_count = sum(1 for number in numbers if number % 2 == 1)
        if self.min_odd <= odd_count <= self.max_odd:
            return FilterDecision(True)
        return FilterDecision(
            False,
            reason=f"odd_count={odd_count} outside {self.min_odd}~{self.max_odd}",
        )

