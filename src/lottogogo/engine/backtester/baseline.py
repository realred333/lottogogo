"""Baseline generators for backtesting."""

from __future__ import annotations

import numpy as np

NUMBER_MIN = 1
NUMBER_MAX = 45


class RandomBaselineGenerator:
    """Generate random lotto recommendations with fixed seed support."""

    def generate(self, *, output_count: int = 5, seed: int | None = 42) -> list[tuple[int, ...]]:
        """Generate random combinations under the same output count constraint."""
        if output_count <= 0:
            raise ValueError("output_count must be > 0.")

        rng = np.random.default_rng(seed)
        numbers = np.arange(NUMBER_MIN, NUMBER_MAX + 1, dtype=np.int64)
        combinations: list[tuple[int, ...]] = []

        for _ in range(output_count):
            sampled = rng.choice(numbers, size=6, replace=False)
            sampled.sort()
            combinations.append(tuple(int(value) for value in sampled.tolist()))
        return combinations

