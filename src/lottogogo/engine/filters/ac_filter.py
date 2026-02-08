"""AC value filter."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from .base import BaseFilter, FilterDecision


def calculate_ac_value(combination: Sequence[int]) -> int:
    """Calculate AC value for a lotto combination."""
    numbers = tuple(sorted(int(value) for value in combination))
    differences = {b - a for a, b in combinations(numbers, 2)}
    return len(differences) - 5


class ACFilter(BaseFilter):
    """Pass only combinations with AC value >= min_ac."""

    name = "ac"

    def __init__(self, min_ac: int = 7) -> None:
        if min_ac < 0:
            raise ValueError("min_ac must be >= 0.")
        self.min_ac = min_ac

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)
        ac_value = calculate_ac_value(numbers)
        if ac_value >= self.min_ac:
            return FilterDecision(True)
        return FilterDecision(False, reason=f"ac={ac_value} below {self.min_ac}")

