"""Zone distribution filter."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .base import BaseFilter, FilterDecision

DEFAULT_ZONES = ((1, 11), (12, 22), (23, 33), (34, 45))


class ZoneFilter(BaseFilter):
    """Ensure numbers are not over-concentrated in one zone."""

    name = "zone"

    def __init__(
        self,
        zones: tuple[tuple[int, int], ...] = DEFAULT_ZONES,
        max_per_zone: int = 3,
    ) -> None:
        if max_per_zone <= 0:
            raise ValueError("max_per_zone must be > 0.")
        self.zones = zones
        self.max_per_zone = max_per_zone

    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        numbers = self.normalize_combination(combination)

        counts = Counter(self._zone_index(number) for number in numbers)
        max_count = max(counts.values()) if counts else 0
        if max_count <= self.max_per_zone:
            return FilterDecision(True)
        return FilterDecision(
            False,
            reason=f"zone concentration={max_count} exceeds {self.max_per_zone}",
        )

    def _zone_index(self, number: int) -> int:
        for idx, (start, end) in enumerate(self.zones):
            if start <= number <= end:
                return idx
        raise ValueError(f"Number {number} does not belong to configured zones.")

