from __future__ import annotations

from lottogogo.engine.ranker import CombinationRanker, DiversitySelector


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

