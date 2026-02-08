"""Filter pipeline orchestration."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .base import BaseFilter


@dataclass(frozen=True)
class PipelineDecision:
    """Decision returned by a filter pipeline run."""

    passed: bool
    filters_passed: tuple[str, ...]
    failed_filter: str | None = None
    reason: str | None = None


class FilterPipeline:
    """Apply filters sequentially with early-exit and rejection logging."""

    def __init__(self, filters: Sequence[BaseFilter] | None = None) -> None:
        self.filters = list(filters or [])
        self._rejection_counts: dict[str, int] = defaultdict(int)

    def add_filter(self, filter_obj: BaseFilter) -> None:
        """Add a filter to the pipeline."""
        self.filters.append(filter_obj)

    def evaluate(self, combination: Sequence[int]) -> PipelineDecision:
        """Evaluate one combination against all filters in order."""
        passed_filters: list[str] = []
        for filter_obj in self.filters:
            decision = filter_obj.evaluate(combination)
            if not decision.passed:
                self._rejection_counts[filter_obj.name] += 1
                return PipelineDecision(
                    passed=False,
                    filters_passed=tuple(passed_filters),
                    failed_filter=filter_obj.name,
                    reason=decision.reason,
                )
            passed_filters.append(filter_obj.name)

        return PipelineDecision(passed=True, filters_passed=tuple(passed_filters))

    def filter_combinations(self, combinations: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
        """Return combinations that pass all filters."""
        accepted: list[tuple[int, ...]] = []
        for combination in combinations:
            decision = self.evaluate(combination)
            if decision.passed:
                accepted.append(tuple(int(value) for value in sorted(combination)))
        return accepted

    @property
    def rejection_counts(self) -> dict[str, int]:
        """Return per-filter rejection counts."""
        return dict(self._rejection_counts)

