from __future__ import annotations

import pytest

from lottogogo.engine.score.calculator import ScoreEnsembler


def test_ensemble_formula_and_clip():
    ensembler = ScoreEnsembler(minimum_score=0.0)
    combined = ensembler.combine(
        base_scores={1: 0.3, 2: 0.2},
        boost_scores={1: 0.2, 2: 0.0},
        penalties={1: 0.1, 2: 0.5},
    )

    assert combined[1] == pytest.approx(0.4)  # 0.3 + 0.2 - 0.1
    assert combined[2] == pytest.approx(0.0)  # clipped from -0.3


def test_ensemble_normalization_sum_is_one():
    normalized = ScoreEnsembler.normalize({1: 2.0, 2: 3.0, 3: 5.0})
    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized[3] > normalized[2] > normalized[1]


def test_ensemble_normalization_handles_all_zero():
    normalized = ScoreEnsembler.normalize({1: 0.0, 2: 0.0, 3: 0.0})
    assert normalized[1] == pytest.approx(1 / 3)
    assert sum(normalized.values()) == pytest.approx(1.0)

