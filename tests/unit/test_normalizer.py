from __future__ import annotations

import pytest

from lottogogo.engine.score.normalizer import ProbabilityNormalizer


def test_softmax_sum_and_order():
    probabilities = ProbabilityNormalizer.softmax({1: 1.0, 2: 2.0, 3: 3.0})
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[3] > probabilities[2] > probabilities[1]


def test_softmax_temperature_controls_sharpness():
    cold = ProbabilityNormalizer.softmax({1: 1.0, 2: 3.0}, temperature=0.5)
    hot = ProbabilityNormalizer.softmax({1: 1.0, 2: 3.0}, temperature=2.0)

    # Lower temperature should make top choice sharper.
    assert cold[2] > hot[2]


def test_floor_applies_minimum_and_renormalizes():
    probabilities = ProbabilityNormalizer.apply_floor({1: 0.9, 2: 0.1, 3: 0.0}, min_prob_floor=0.05)

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(value >= 0.05 for value in probabilities.values())


def test_floor_rejects_invalid_large_floor():
    with pytest.raises(ValueError, match="too large"):
        ProbabilityNormalizer.apply_floor({1: 0.5, 2: 0.5}, min_prob_floor=0.5)

