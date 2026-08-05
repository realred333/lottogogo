"""Diversity constraints for ranked combinations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

DEFAULT_NUMBER_FREQUENCY_RATIO = 0.4


def default_number_frequency(output_count: int, ratio: float = DEFAULT_NUMBER_FREQUENCY_RATIO) -> int:
    """Return how many games a single number may appear in, for `output_count` games.

    Combination scores are the sum of member number scores, so the highest scoring
    numbers otherwise occupy the entire top of the ranking and end up in every
    selected game. Capping per-number frequency is what keeps the output diverse
    at the *number* level; `max_overlap` only constrains pairs of combinations.
    """
    if output_count <= 0:
        raise ValueError("output_count must be > 0.")
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must be in (0, 1].")
    return max(2, math.ceil(output_count * ratio))


class DiversitySelector:
    """Select combinations while enforcing overlap, duplicate, and frequency constraints."""

    def __init__(self, max_overlap: int = 3, max_number_frequency: int | None = None) -> None:
        if not (0 <= max_overlap <= 6):
            raise ValueError("max_overlap must be between 0 and 6.")
        if max_number_frequency is not None and max_number_frequency < 1:
            raise ValueError("max_number_frequency must be >= 1 when provided.")
        self.max_overlap = max_overlap
        self.max_number_frequency = max_number_frequency

    def select(self, candidates: Sequence[Sequence[int]], output_count: int) -> list[tuple[int, ...]]:
        """Select up to output_count combinations satisfying diversity rules."""
        if output_count <= 0:
            raise ValueError("output_count must be > 0.")

        selected: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        number_counts: Counter[int] = Counter()

        for candidate in candidates:
            normalized = tuple(sorted(int(value) for value in candidate))
            if normalized in seen:
                continue
            if self._violates_overlap(normalized, selected):
                continue
            if self._violates_frequency(normalized, number_counts):
                continue

            selected.append(normalized)
            seen.add(normalized)
            number_counts.update(normalized)
            if len(selected) >= output_count:
                break

        return selected

    def _violates_overlap(
        self,
        candidate: tuple[int, ...],
        selected: list[tuple[int, ...]],
    ) -> bool:
        candidate_set = set(candidate)
        for existing in selected:
            overlap = len(candidate_set.intersection(existing))
            if overlap >= (self.max_overlap + 1):
                return True
        return False

    def _violates_frequency(self, candidate: tuple[int, ...], number_counts: Counter[int]) -> bool:
        if self.max_number_frequency is None:
            return False
        return any(number_counts[number] >= self.max_number_frequency for number in candidate)


def select_with_relaxation(
    candidates: Sequence[Sequence[int]],
    output_count: int,
    *,
    max_overlap: int = 3,
    max_number_frequency: int | None = None,
) -> list[tuple[int, ...]]:
    """Select diverse combinations, progressively relaxing constraints when short.

    Relaxes the per-number frequency cap first (it is the tighter constraint), then
    the pairwise overlap limit. Returns the best partial result if no plan fills
    `output_count`, so callers never get an empty list when candidates exist.
    """
    if output_count <= 0:
        raise ValueError("output_count must be > 0.")

    plans: list[tuple[int, int | None]] = []
    if max_number_frequency is not None:
        for frequency in range(max_number_frequency, output_count):
            plans.append((max_overlap, frequency))
    plans.append((max_overlap, None))
    for overlap in range(max_overlap + 1, 7):
        plans.append((overlap, None))

    best: list[tuple[int, ...]] = []
    for overlap, frequency in plans:
        selector = DiversitySelector(max_overlap=overlap, max_number_frequency=frequency)
        selected = selector.select(candidates, output_count=output_count)
        if len(selected) > len(best):
            best = selected
        if len(best) >= output_count:
            break

    return best[:output_count]
