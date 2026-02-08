"""Monte Carlo sampler for lotto combinations."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

COMBINATION_SIZE = 6


class MonteCarloSampler:
    """Generate weighted lotto combinations without replacement."""

    def __init__(self, sample_size: int = 50000, chunk_size: int = 20000) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")

        self.sample_size = sample_size
        self.chunk_size = chunk_size

    def sample_one(
        self,
        probabilities: Mapping[int, float],
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, ...]:
        """Sample a single 6-number combination with weights."""
        numbers, probs = self._normalize_probabilities(probabilities)
        generator = rng or np.random.default_rng(seed)
        sampled = generator.choice(numbers, size=COMBINATION_SIZE, replace=False, p=probs)
        sampled.sort()
        return tuple(int(value) for value in sampled.tolist())

    def sample(
        self,
        probabilities: Mapping[int, float],
        *,
        sample_size: int | None = None,
        seed: int | None = None,
    ) -> list[tuple[int, ...]]:
        """Sample many weighted combinations, using chunked vectorized generation."""
        numbers, probs = self._normalize_probabilities(probabilities)
        n_samples = sample_size if sample_size is not None else self.sample_size
        if n_samples <= 0:
            raise ValueError("sample_size must be > 0.")

        generator = np.random.default_rng(seed)
        if n_samples == 1:
            return [self.sample_one(probabilities, rng=generator)]

        sampled = self._sample_vectorized(
            numbers=numbers,
            probabilities=probs,
            sample_size=n_samples,
            rng=generator,
        )
        return [tuple(int(value) for value in row.tolist()) for row in sampled]

    def sample_array(
        self,
        probabilities: Mapping[int, float],
        *,
        sample_size: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample combinations and return as an ndarray of shape (N, 6)."""
        numbers, probs = self._normalize_probabilities(probabilities)
        n_samples = sample_size if sample_size is not None else self.sample_size
        if n_samples <= 0:
            raise ValueError("sample_size must be > 0.")

        generator = np.random.default_rng(seed)
        if n_samples == 1:
            return np.array([self.sample_one(probabilities, rng=generator)], dtype=np.int64)

        return self._sample_vectorized(
            numbers=numbers,
            probabilities=probs,
            sample_size=n_samples,
            rng=generator,
        )

    def _sample_vectorized(
        self,
        *,
        numbers: np.ndarray,
        probabilities: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Vectorized weighted sampling without replacement via Gumbel top-k."""
        log_prob = np.log(probabilities)
        chunks = math.ceil(sample_size / self.chunk_size)
        collected: list[np.ndarray] = []

        for chunk_idx in range(chunks):
            start = chunk_idx * self.chunk_size
            remaining = sample_size - start
            current = min(self.chunk_size, remaining)
            gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(current, len(numbers)))
            scores = log_prob + gumbel
            topk_idx = np.argpartition(scores, -COMBINATION_SIZE, axis=1)[:, -COMBINATION_SIZE:]
            sampled = numbers[topk_idx]
            sampled.sort(axis=1)
            collected.append(sampled.astype(np.int64, copy=False))

        return np.vstack(collected)

    @staticmethod
    def _normalize_probabilities(probabilities: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
        if not probabilities:
            raise ValueError("probabilities cannot be empty.")

        keys = sorted(int(key) for key in probabilities.keys())
        values = np.array([float(probabilities[key]) for key in keys], dtype=np.float64)

        if len(keys) < COMBINATION_SIZE:
            raise ValueError("At least 6 probabilities are required.")
        if np.any(values < 0):
            raise ValueError("probabilities must be non-negative.")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("Sum of probabilities must be > 0.")

        normalized = values / total
        return np.array(keys, dtype=np.int64), normalized

