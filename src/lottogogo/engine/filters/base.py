"""Base types for lotto filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

COMBINATION_SIZE = 6
NUMBER_MIN = 1
NUMBER_MAX = 45


@dataclass(frozen=True)
class FilterDecision:
    """Filter evaluation result."""

    passed: bool
    reason: str | None = None


class BaseFilter(ABC):
    """Base class for all combination filters."""

    name: str

    @abstractmethod
    def evaluate(self, combination: Sequence[int]) -> FilterDecision:
        """Evaluate whether the given combination passes this filter."""

    @staticmethod
    def normalize_combination(combination: Sequence[int]) -> tuple[int, ...]:
        """Return validated, sorted lotto combination."""
        numbers = tuple(sorted(int(value) for value in combination))
        if len(numbers) != COMBINATION_SIZE:
            raise ValueError(f"Combination must contain {COMBINATION_SIZE} numbers.")
        if len(set(numbers)) != COMBINATION_SIZE:
            raise ValueError("Combination numbers must be unique.")
        if any(number < NUMBER_MIN or number > NUMBER_MAX for number in numbers):
            raise ValueError("Combination numbers must be in range 1~45.")
        return numbers

