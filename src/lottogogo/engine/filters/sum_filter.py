"""Sum-range filter."""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseFilter, FilterDecision


class SumFilter(BaseFilter):
    """Pass only combinations whose sum falls in configured bounds."""

    name = "sum"

    def __init__(self, min_sum: int = 100, max_sum: int = 175) -> None:
        if min_sum > max_sum:
            raise ValueError("min_sum cannot be greater than max_sum.")
        self.min_sum = min_sum
        self.max_sum = max_sum

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)
        total = sum(numbers)
        if self.min_sum <= total <= self.max_sum:
            return FilterDecision(True)
        return FilterDecision(False, reason=f"sum={total} outside {self.min_sum}~{self.max_sum}")

