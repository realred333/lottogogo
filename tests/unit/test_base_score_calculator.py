from __future__ import annotations

import pandas as pd
import pytest

from lottogogo.engine.score.calculator import BaseScoreCalculator


def test_posterior_mean_matches_expected_formula():
    calculator = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
    posterior = calculator.posterior_mean(successes=2, total_rounds=10)

    # (alpha + successes) / (alpha + beta + total_rounds) = 3 / 12
    assert posterior == pytest.approx(0.25)


def test_calculate_scores_uses_recent_n_only():
    history = pd.DataFrame(
        {
            "round": [1, 2, 3, 4],
            "n1": [1, 1, 2, 2],
            "n2": [3, 3, 3, 3],
            "n3": [4, 4, 4, 4],
            "n4": [5, 5, 5, 5],
            "n5": [6, 6, 6, 6],
            "n6": [7, 7, 7, 7],
        }
    )

    calculator = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
    scores_all = calculator.calculate_scores(history)
    scores_recent = calculator.calculate_scores(history, recent_n=2)

    assert scores_all[1] == pytest.approx(0.5)  # successes=2 out of 4
    assert scores_recent[1] == pytest.approx(0.25)  # successes=0 out of last 2 rounds
    assert scores_recent[2] == pytest.approx(0.75)  # successes=2 out of last 2 rounds


def test_calculate_scores_returns_all_numbers():
    history = pd.DataFrame(
        {
            "round": [1],
            "n1": [1],
            "n2": [2],
            "n3": [3],
            "n4": [4],
            "n5": [5],
            "n6": [6],
        }
    )

    calculator = BaseScoreCalculator()
    scores = calculator.calculate_scores(history)

    assert len(scores) == 45
    assert all(0.0 < value < 1.0 for value in scores.values())

