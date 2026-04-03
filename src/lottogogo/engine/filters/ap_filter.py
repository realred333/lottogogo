"""Arithmetic progression filter."""

from __future__ import annotations

from itertools import combinations
from collections.abc import Sequence

from .base import BaseFilter, FilterDecision


class ArithmeticProgressionFilter(BaseFilter):
    """Reject combinations where 5 or 6 numbers form an arithmetic progression.

    6-number AP (e.g. 1,8,15,22,29,36) and 5-number AP hidden among the 6
    (e.g. 1,3,5,7,9,43 → [1,3,5,7,9] is AP) are both rejected.
    """

    name = "ap"

    def __init__(self, min_ap_length: int = 5) -> None:
        if min_ap_length < 2:
            raise ValueError("min_ap_length must be >= 2.")
        self.min_ap_length = min_ap_length

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)

        for length in range(len(numbers), self.min_ap_length - 1, -1):
            for combo in combinations(numbers, length):
                diffs = [combo[i + 1] - combo[i] for i in range(len(combo) - 1)]
                if len(set(diffs)) == 1:
                    return FilterDecision(
                        False,
                        reason=f"AP{length} detected: {list(combo)} (d={diffs[0]})",
                    )

        return FilterDecision(True)
