from __future__ import annotations

from lottogogo.engine.filters import (
    ACFilter,
    HighLowFilter,
    HistoryFilter,
    OddEvenFilter,
    SumFilter,
    TailFilter,
    ZoneFilter,
)
from lottogogo.engine.filters.ac_filter import calculate_ac_value


def test_t411_sum_filter_boundaries():
    filt = SumFilter(min_sum=100, max_sum=175)

    assert filt.evaluate([1, 5, 10, 20, 30, 34]).passed  # sum=100
    assert filt.evaluate([21, 24, 28, 30, 34, 38]).passed  # sum=175
    assert not filt.evaluate([1, 2, 3, 4, 5, 6]).passed
    assert not filt.evaluate([40, 41, 42, 43, 44, 45]).passed


def test_t412_ac_filter_threshold():
    filt = ACFilter(min_ac=7)

    assert calculate_ac_value([1, 2, 3, 4, 8, 16]) == 7
    assert filt.evaluate([1, 2, 3, 4, 8, 16]).passed
    assert not filt.evaluate([1, 2, 3, 4, 5, 6]).passed


def test_t413_zone_filter_distribution():
    filt = ZoneFilter(max_per_zone=3)

    assert filt.evaluate([1, 2, 12, 24, 34, 45]).passed
    assert not filt.evaluate([1, 2, 3, 4, 23, 34]).passed


def test_t414_tail_filter_limit():
    filt = TailFilter(max_same_tail=2)

    assert filt.evaluate([1, 12, 23, 34, 40, 45]).passed
    assert not filt.evaluate([1, 11, 21, 32, 43, 44]).passed


def test_t415_odd_even_filter_range():
    filt = OddEvenFilter(min_odd=2, max_odd=4)

    assert filt.evaluate([1, 3, 2, 4, 6, 8]).passed
    assert filt.evaluate([1, 3, 5, 7, 2, 4]).passed
    assert not filt.evaluate([1, 2, 4, 6, 8, 10]).passed


def test_t416_high_low_filter_range():
    filt = HighLowFilter(min_high=2, max_high=4, high_start=23)

    assert filt.evaluate([1, 2, 3, 4, 23, 24]).passed
    assert filt.evaluate([1, 2, 23, 24, 25, 26]).passed
    assert not filt.evaluate([1, 2, 3, 4, 5, 23]).passed


def test_t417_history_filter_overlap():
    history = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
    ]
    filt = HistoryFilter(historical_draws=history, match_threshold=5)

    assert not filt.evaluate([1, 2, 3, 4, 5, 20]).passed
    assert filt.evaluate([1, 2, 3, 4, 20, 21]).passed


def test_t418_history_filter_exact_match_threshold_6():
    history = [
        [1, 2, 3, 4, 5, 6],
    ]
    filt = HistoryFilter(historical_draws=history, match_threshold=6)

    assert not filt.evaluate([1, 2, 3, 4, 5, 6]).passed
    assert filt.evaluate([1, 2, 3, 4, 5, 7]).passed
