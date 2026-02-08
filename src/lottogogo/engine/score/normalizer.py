"""Probability normalization utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class ProbabilityNormalizer:
    """Convert raw scores to stable sampling probabilities."""

    @staticmethod
    def softmax(raw_scores: Mapping[int, float], temperature: float = 1.0) -> dict[int, float]:
        """Apply temperature-controlled softmax normalization."""
        if not raw_scores:
            raise ValueError("raw_scores cannot be empty.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        keys = list(raw_scores.keys())
        values = np.array([float(raw_scores[key]) for key in keys], dtype=np.float64)
        scaled = values / float(temperature)
        shifted = scaled - np.max(scaled)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values)
        return {key: float(prob) for key, prob in zip(keys, probabilities)}

    @staticmethod
    def apply_floor(probabilities: Mapping[int, float], min_prob_floor: float) -> dict[int, float]:
        """Apply minimum probability floor and renormalize while preserving sum=1."""
        if not probabilities:
            raise ValueError("probabilities cannot be empty.")
        if min_prob_floor < 0:
            raise ValueError("min_prob_floor must be >= 0.")

        keys = list(probabilities.keys())
        n = len(keys)
        if min_prob_floor * n >= 1.0:
            raise ValueError("min_prob_floor is too large for the number of items.")

        values = np.array([float(probabilities[key]) for key in keys], dtype=np.float64)
        values = np.clip(values, 0.0, None)
        total = np.sum(values)
        if total <= 0:
            values = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            values = values / total

        residual = 1.0 - (min_prob_floor * n)
        adjusted = min_prob_floor + residual * values
        adjusted = adjusted / np.sum(adjusted)
        return {key: float(prob) for key, prob in zip(keys, adjusted)}

    @classmethod
    def to_sampling_probabilities(
        cls, raw_scores: Mapping[int, float], temperature: float = 1.0, min_prob_floor: float = 0.0
    ) -> dict[int, float]:
        """Convert raw scores to sampling probabilities with optional floor."""
        probabilities = cls.softmax(raw_scores=raw_scores, temperature=temperature)
        if min_prob_floor > 0:
            return cls.apply_floor(probabilities=probabilities, min_prob_floor=min_prob_floor)
        return probabilities

