from __future__ import annotations

from collections import Counter

import pytest

from lottogogo.engine.ranker import (
    CombinationRanker,
    DiversitySelector,
    default_number_frequency,
    select_with_relaxation,
)


def test_t511_combination_score_and_top_k():
    ranker = CombinationRanker()
    combinations = [
        (1, 2, 3, 4, 5, 6),
        (10, 11, 12, 13, 14, 15),
        (1, 10, 20, 30, 40, 45),
    ]
    raw_scores = {
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        5: 0.1,
        6: 0.1,
        10: 0.2,
        11: 0.2,
        12: 0.2,
        13: 0.2,
        14: 0.2,
        15: 0.2,
        20: 0.3,
        30: 0.3,
        40: 0.3,
        45: 0.3,
    }

    ranked = ranker.rank(combinations=combinations, raw_scores=raw_scores, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].numbers == (1, 10, 20, 30, 40, 45)
    assert ranked[0].combo_score > ranked[1].combo_score


def test_t521_overlap_threshold_blocks_too_similar_combos():
    selector = DiversitySelector(max_overlap=3)
    candidates = [
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 7, 8),  # overlap 4 with first -> should be removed
        (9, 10, 11, 12, 13, 14),
    ]

    selected = selector.select(candidates, output_count=3)

    assert (1, 2, 3, 4, 7, 8) not in selected
    assert (1, 2, 3, 4, 5, 6) in selected
    assert (9, 10, 11, 12, 13, 14) in selected


def test_t522_duplicate_removal_and_output_count_fill():
    selector = DiversitySelector(max_overlap=3)
    candidates = [
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6),  # duplicate
        (7, 8, 9, 10, 11, 12),
        (13, 14, 15, 16, 17, 18),
        (19, 20, 21, 22, 23, 24),
    ]

    selected = selector.select(candidates, output_count=3)

    assert len(selected) == 3
    assert selected.count((1, 2, 3, 4, 5, 6)) == 1


# ── Number-frequency cap ─────────────────────────────────────────────────────

def test_default_number_frequency_scales_with_game_count():
    assert default_number_frequency(5) == 2  # ceil(5 * 0.4)
    assert default_number_frequency(10) == 4  # ceil(10 * 0.4)
    assert default_number_frequency(1) == 2  # floor of 2 always applies
    assert default_number_frequency(10, ratio=0.6) == 6


def test_frequency_cap_blocks_a_number_from_every_game():
    """A top-scoring number must not appear in more games than the cap allows."""
    # Every candidate contains 32; without a cap all three would be selected.
    candidates = [
        (1, 2, 3, 4, 5, 32),
        (6, 7, 8, 9, 10, 32),
        (11, 12, 13, 14, 15, 32),
        (16, 17, 18, 19, 20, 21),
    ]

    uncapped = DiversitySelector(max_overlap=3).select(candidates, output_count=3)
    assert sum(1 for combo in uncapped if 32 in combo) == 3

    capped = DiversitySelector(max_overlap=3, max_number_frequency=2).select(
        candidates, output_count=3
    )
    assert sum(1 for combo in capped if 32 in combo) == 2
    assert (16, 17, 18, 19, 20, 21) in capped


def test_frequency_cap_rejects_invalid_value():
    with pytest.raises(ValueError, match="max_number_frequency"):
        DiversitySelector(max_number_frequency=0)


def test_select_with_relaxation_fills_count_by_loosening_frequency():
    """When the cap is too tight to fill the request, it relaxes instead of returning short."""
    candidates = [
        (1, 2, 3, 4, 5, 32),
        (6, 7, 8, 9, 10, 32),
        (11, 12, 13, 14, 15, 32),
    ]

    selected = select_with_relaxation(
        candidates, output_count=3, max_overlap=3, max_number_frequency=1
    )

    assert len(selected) == 3


def test_select_with_relaxation_respects_cap_when_it_can():
    candidates = [
        (1, 2, 3, 4, 5, 32),
        (6, 7, 8, 9, 10, 32),
        (11, 12, 13, 14, 15, 32),
        (16, 17, 18, 19, 20, 21),
        (22, 23, 24, 25, 26, 27),
    ]

    selected = select_with_relaxation(
        candidates, output_count=3, max_overlap=3, max_number_frequency=2
    )

    assert len(selected) == 3
    counts = Counter(number for combo in selected for number in combo)
    assert counts[32] <= 2


def test_select_with_relaxation_returns_best_effort_when_candidates_run_out():
    selected = select_with_relaxation(
        [(1, 2, 3, 4, 5, 6)], output_count=5, max_overlap=3, max_number_frequency=2
    )

    assert selected == [(1, 2, 3, 4, 5, 6)]

