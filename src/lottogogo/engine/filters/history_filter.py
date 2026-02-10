"""Historical overlap filter."""

from __future__ import annotations

from itertools import combinations
from collections.abc import Iterable, Sequence

from .base import BaseFilter, FilterDecision


class HistoryFilter(BaseFilter):
    """Reject combinations with too much overlap to historical winning draws."""

    name = "history"

    def __init__(self, historical_draws: Iterable[Sequence[int]], match_threshold: int = 5) -> None:
        if not (1 <= match_threshold <= 6):
            raise ValueError("match_threshold must be between 1 and 6.")
        self.match_threshold = match_threshold
        self._normalized_history = [self.normalize_combination(draw) for draw in historical_draws]
        self._history_masks = [self._to_bitmask(draw) for draw in self._normalized_history]
        self._exact_history = set(self._normalized_history)
        self._history_five_subsets: set[tuple[int, ...]] = set()

        if self.match_threshold == 5:
            for draw in self._normalized_history:
                self._history_five_subsets.update(combinations(draw, 5))

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)

        if self.match_threshold == 6:
            if numbers in self._exact_history:
                return FilterDecision(False, reason="history_overlap=6 >= 6")
            return FilterDecision(True)

        if self.match_threshold == 5:
            if numbers in self._exact_history:
                return FilterDecision(False, reason="history_overlap=6 >= 5")

            for subset in combinations(numbers, 5):
                if subset in self._history_five_subsets:
                    return FilterDecision(False, reason="history_overlap=5 >= 5")
            return FilterDecision(True)

        candidate_mask = self._to_bitmask(numbers)

        for history_mask in self._history_masks:
            overlap = (candidate_mask & history_mask).bit_count()
            if overlap >= self.match_threshold:
                return FilterDecision(
                    False,
                    reason=f"history_overlap={overlap} >= {self.match_threshold}",
                )
        return FilterDecision(True)

    @staticmethod
    def _to_bitmask(combination: Sequence[int]) -> int:
        mask = 0
        for number in combination:
            mask |= 1 << (int(number) - 1)
        return mask
