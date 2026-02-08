"""Diversity constraints for ranked combinations."""

from __future__ import annotations

from collections.abc import Sequence


class DiversitySelector:
    """Select combinations while enforcing overlap and duplicate constraints."""

    def __init__(self, max_overlap: int = 3) -> None:
        if not (0 <= max_overlap <= 6):
            raise ValueError("max_overlap must be between 0 and 6.")
        self.max_overlap = max_overlap

    def select(self, candidates: Sequence[Sequence[int]], output_count: int) -> list[tuple[int, ...]]:
        """Select up to output_count combinations satisfying diversity rules."""
        if output_count <= 0:
            raise ValueError("output_count must be > 0.")

        selected: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()

        for candidate in candidates:
            normalized = tuple(sorted(int(value) for value in candidate))
            if normalized in seen:
                continue
            if self._violates_overlap(normalized, selected):
                continue

            selected.append(normalized)
            seen.add(normalized)
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

