"""High-low balance filter."""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseFilter, FilterDecision


class HighLowFilter(BaseFilter):
    """Balance high (23+) and low (1~22) numbers."""

    name = "high_low"

    def __init__(self, min_high: int = 2, max_high: int = 4, high_start: int = 23) -> None:
        if min_high < 0 or max_high > 6 or min_high > max_high:
            raise ValueError("Invalid high count range.")
        if not (1 <= high_start <= 45):
            raise ValueError("high_start must be in range 1~45.")
        self.min_high = min_high
        self.max_high = max_high
        self.high_start = high_start

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)
        high_count = sum(1 for number in numbers if number >= self.high_start)
        if self.min_high <= high_count <= self.max_high:
            return FilterDecision(True)
        return FilterDecision(
            False,
            reason=f"high_count={high_count} outside {self.min_high}~{self.max_high}",
        )

