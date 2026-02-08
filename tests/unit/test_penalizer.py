from __future__ import annotations

import pandas as pd
import pytest

from lottogogo.engine.score.penalizer import PenaltyCalculator


def test_poisson_penalty_increases_for_overrepresented_numbers():
    history = pd.DataFrame(
        {
            "round": [1, 2, 3, 4],
            "n1": [1, 1, 1, 1],
            "n2": [2, 3, 4, 5],
            "n3": [6, 7, 8, 9],
            "n4": [10, 11, 12, 13],
            "n5": [14, 15, 16, 17],
            "n6": [18, 19, 20, 21],
        }
    )
    calculator = PenaltyCalculator(poisson_window=4, poisson_lambda=0.5, markov_lambda=0.0)

    penalties = calculator.calculate_poisson_penalty(history)

    assert penalties[1] > penalties[45]
    assert penalties[45] == pytest.approx(0.0)


def test_poisson_penalty_scales_with_lambda():
    history = pd.DataFrame(
        {
            "round": [1, 2],
            "n1": [1, 1],
            "n2": [2, 3],
            "n3": [4, 5],
            "n4": [6, 7],
            "n5": [8, 9],
            "n6": [10, 11],
        }
    )
    low = PenaltyCalculator(poisson_window=2, poisson_lambda=0.1).calculate_poisson_penalty(history)
    high = PenaltyCalculator(poisson_window=2, poisson_lambda=0.6).calculate_poisson_penalty(history)

    assert high[1] > low[1]


def test_markov_penalty_uses_transition_matrix():
    history = pd.DataFrame(
        {
            "round": [1, 2, 3],
            "n1": [1, 10, 10],
            "n2": [2, 11, 11],
            "n3": [3, 12, 12],
            "n4": [4, 13, 13],
            "n5": [5, 14, 14],
            "n6": [6, 15, 15],
        }
    )
    calculator = PenaltyCalculator(poisson_lambda=0.0, markov_lambda=0.3)

    matrix = calculator.build_transition_matrix(history)
    penalties = calculator.calculate_markov_penalty(history)

    assert matrix.shape == (46, 46)
    # Row corresponding to 10 has outgoing transitions to 10~15.
    assert matrix[10, 10] > 0.0
    assert penalties[10] > penalties[1]
    assert penalties[10] >= 0.0
