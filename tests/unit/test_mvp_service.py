"""Tests for MVP service helper behavior."""

from __future__ import annotations

from collections import deque
from threading import Lock

from lottogogo.mvp.service import PRESET_CONFIGS, contains_rare_pair, exceeds_carryover_limit
from lottogogo.mvp.service import ALLOWED_GAMES, RecommendationService


def test_presets_are_defined_for_a_and_b() -> None:
    assert set(PRESET_CONFIGS.keys()) == {"A", "B"}
    assert PRESET_CONFIGS["A"].min_ac > PRESET_CONFIGS["B"].min_ac
    assert PRESET_CONFIGS["A"].sample_size == 100_000
    assert PRESET_CONFIGS["B"].sample_size == 100_000


def test_contains_rare_pair_detects_known_rare_pair() -> None:
    assert contains_rare_pair((1, 8, 12, 22, 33, 44))
    assert not contains_rare_pair((1, 2, 3, 4, 5, 6))


def test_exceeds_carryover_limit() -> None:
    combination = (3, 7, 11, 19, 29, 41)
    carryovers = {3, 7, 11, 40}

    assert exceeds_carryover_limit(combination, carryovers, limit=2)
    assert not exceeds_carryover_limit(combination, carryovers, limit=3)


def _make_minimal_service() -> RecommendationService:
    service = RecommendationService.__new__(RecommendationService)
    service._pool_max = 8
    service._pool_lock = Lock()
    service._result_pool = {
        (preset, games): deque(maxlen=service._pool_max)
        for preset in PRESET_CONFIGS.keys()
        for games in ALLOWED_GAMES
    }
    service._pool_cursor = {
        (preset, games): 0
        for preset in PRESET_CONFIGS.keys()
        for games in ALLOWED_GAMES
    }
    service._latest_result = {
        (preset, games): None
        for preset in PRESET_CONFIGS.keys()
        for games in ALLOWED_GAMES
    }
    return service


def test_pool_read_does_not_deplete_bucket() -> None:
    service = _make_minimal_service()
    result = {"meta": {"preset": "A", "percentile": 1}, "recommendations": []}

    service._store_in_pool("A", 5, result)
    first = service._read_from_pool("A", 5)
    second = service._read_from_pool("A", 5)

    assert first is not None
    assert second is not None
    assert len(service._result_pool[("A", 5)]) == 1


def test_pool_read_rotates_between_pre_generated_items() -> None:
    service = _make_minimal_service()
    first_result = {"meta": {"preset": "A", "percentile": 1}, "recommendations": [{"numbers": [1, 2, 3, 4, 5, 6]}]}
    second_result = {"meta": {"preset": "A", "percentile": 2}, "recommendations": [{"numbers": [7, 8, 9, 10, 11, 12]}]}

    service._store_in_pool("A", 5, first_result)
    service._store_in_pool("A", 5, second_result)

    one = service._read_from_pool("A", 5)
    two = service._read_from_pool("A", 5)
    three = service._read_from_pool("A", 5)

    assert one is not None and two is not None and three is not None
    assert one["recommendations"][0]["numbers"] == [1, 2, 3, 4, 5, 6]
    assert two["recommendations"][0]["numbers"] == [7, 8, 9, 10, 11, 12]
    assert three["recommendations"][0]["numbers"] == [1, 2, 3, 4, 5, 6]


def test_recommend_ignores_seed_and_reads_from_pool() -> None:
    service = _make_minimal_service()
    service._pool_target = 4
    service._warming_keys = set()
    service._start_pool_refill = lambda preset, games: None
    service._bootstrap_sample_size = 6000

    pooled = {
        "meta": {"preset": "A", "percentile": 1},
        "recommendations": [{"numbers": [1, 2, 3, 4, 5, 6], "score": 1.0, "tags": [], "reasons": []}],
    }
    service._store_in_pool("A", 5, pooled)

    result = service.recommend(preset="A", games=5, seed=123456)

    assert result["recommendations"][0]["numbers"] == [1, 2, 3, 4, 5, 6]
    assert len(service._result_pool[("A", 5)]) == 1
