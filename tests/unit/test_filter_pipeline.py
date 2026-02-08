from __future__ import annotations

from lottogogo.engine.filters import FilterDecision, FilterPipeline, OddEvenFilter, SumFilter
from lottogogo.engine.filters.base import BaseFilter


class CountingFilter(BaseFilter):
    def __init__(self, name: str, should_pass: bool) -> None:
        self.name = name
        self.should_pass = should_pass
        self.calls = 0

    def evaluate(self, combination):
        self.calls += 1
        if self.should_pass:
            return FilterDecision(True)
        return FilterDecision(False, reason=f"{self.name} failed")


def test_t421_pipeline_sequential_execution_with_early_exit():
    first = CountingFilter("first", should_pass=True)
    second = CountingFilter("second", should_pass=False)
    third = CountingFilter("third", should_pass=True)
    pipeline = FilterPipeline([first, second, third])

    decision = pipeline.evaluate([1, 2, 3, 4, 23, 24])

    assert not decision.passed
    assert decision.failed_filter == "second"
    assert decision.filters_passed == ("first",)
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 0


def test_t422_pipeline_rejection_counts_logging():
    pipeline = FilterPipeline([SumFilter(min_sum=100, max_sum=175), OddEvenFilter(min_odd=2, max_odd=4)])

    pipeline.evaluate([1, 2, 3, 4, 5, 6])  # sum fail
    pipeline.evaluate([1, 2, 3, 4, 5, 7])  # sum fail
    pipeline.evaluate([10, 12, 14, 16, 18, 31])  # odd_even fail

    assert pipeline.rejection_counts["sum"] == 2
    assert pipeline.rejection_counts["odd_even"] == 1

