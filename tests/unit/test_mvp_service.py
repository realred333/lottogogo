"""Tests for MVP service helper behavior."""

from __future__ import annotations

from lottogogo.mvp.service import PRESET_CONFIGS, contains_rare_pair, exceeds_carryover_limit


def test_presets_are_defined_for_a_and_b() -> None:
    assert set(PRESET_CONFIGS.keys()) == {"A", "B"}
    assert PRESET_CONFIGS["A"].min_ac > PRESET_CONFIGS["B"].min_ac


def test_contains_rare_pair_detects_known_rare_pair() -> None:
    assert contains_rare_pair((1, 8, 12, 22, 33, 44))
    assert not contains_rare_pair((1, 2, 3, 4, 5, 6))


def test_exceeds_carryover_limit() -> None:
    combination = (3, 7, 11, 19, 29, 41)
    carryovers = {3, 7, 11, 40}

    assert exceeds_carryover_limit(combination, carryovers, limit=2)
    assert not exceeds_carryover_limit(combination, carryovers, limit=3)
