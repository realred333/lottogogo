from __future__ import annotations

import pandas as pd
import pytest

from lottogogo.engine.score.booster import BoostCalculator


def _history_for_hot_cold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "n1": [1, 1, 1, 1, 1, 6, 7, 8, 9, 10],
            "n2": [2, 2, 2, 11, 12, 13, 14, 15, 16, 17],
            "n3": [3, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            "n4": [4, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            "n5": [5, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            "n6": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        }
    )


def test_hot_cold_threshold_and_windows():
    history = _history_for_hot_cold()
    calculator = BoostCalculator(
        hot_threshold=2,
        hot_window=5,
        hot_weight=0.4,
        cold_window=10,
        cold_weight=0.15,
        neighbor_weight=0.0,
        carryover_weight=0.0,
        reverse_weight=0.0,
    )

    boosts, tags = calculator.calculate_boosts(history)

    # Number 13 appears twice in the last 5 rounds -> hot.
    assert "hot" in tags[13]
    assert boosts[13] == pytest.approx(0.4)

    # Number 45 does not appear in the last 10 rounds -> cold.
    assert "cold" in tags[45]
    assert boosts[45] == pytest.approx(0.15)


def test_booster_requires_number_columns():
    history = pd.DataFrame({"round": [1], "n1": [1]})
    calculator = BoostCalculator()
    with pytest.raises(ValueError, match="Missing number columns"):
        calculator.calculate_boosts(history)


def test_neighbor_and_carryover_from_last_round():
    history = pd.DataFrame(
        {
            "round": [1, 2],
            "n1": [1, 10],
            "n2": [2, 20],
            "n3": [3, 30],
            "n4": [4, 40],
            "n5": [5, 44],
            "n6": [6, 45],
        }
    )

    calculator = BoostCalculator(
        hot_weight=0.0,
        cold_weight=0.0,
        neighbor_weight=0.3,
        carryover_weight=0.2,
        reverse_weight=0.0,
    )
    boosts, tags = calculator.calculate_boosts(history)

    # Carryover: numbers in last round.
    assert "carryover" in tags[10]
    assert boosts[10] == pytest.approx(0.2)

    # Neighbor: ±1 around last round numbers.
    assert "neighbor" in tags[9]
    assert boosts[9] == pytest.approx(0.3)

    # Edge handling around 45.
    assert "neighbor" in tags[43]
    assert boosts[43] == pytest.approx(0.3)


def test_reverse_mapping_from_last_round():
    history = pd.DataFrame(
        {
            "round": [1],
            "n1": [1],
            "n2": [2],
            "n3": [3],
            "n4": [10],
            "n5": [20],
            "n6": [45],
        }
    )
    calculator = BoostCalculator(
        hot_weight=0.0,
        cold_weight=0.0,
        neighbor_weight=0.0,
        carryover_weight=0.0,
        reverse_weight=0.1,
    )

    boosts, tags = calculator.calculate_boosts(history)

    # reverse(1)=45, reverse(2)=44, reverse(10)=36, reverse(20)=26
    assert "reverse" in tags[44]
    assert "reverse" in tags[36]
    assert "reverse" in tags[26]
    assert boosts[44] == pytest.approx(0.1)
