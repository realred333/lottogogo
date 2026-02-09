"""MVP recommendation adapter for API and web UI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.engine.filters import (
    ACFilter,
    FilterPipeline,
    HighLowFilter,
    HistoryFilter,
    OddEvenFilter,
    SumFilter,
    TailFilter,
    ZoneFilter,
)
from lottogogo.engine.ranker import CombinationRank, CombinationRanker, DiversitySelector
from lottogogo.engine.sampler import MonteCarloSampler
from lottogogo.engine.score import (
    BaseScoreCalculator,
    BoostCalculator,
    PenaltyCalculator,
    ProbabilityNormalizer,
    ScoreEnsembler,
)
from lottogogo.engine.score.hmm_scorer import HMMScorer

PresetName = Literal["A", "B"]

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
ALLOWED_GAMES = {5, 10}

RARE_PAIR_COMBINATIONS = {
    (1, 22),
    (3, 25),
    (4, 30),
    (6, 23),
    (6, 29),
    (6, 33),
    (7, 32),
    (8, 9),
    (8, 12),
    (8, 26),
    (9, 20),
    (11, 34),
    (11, 40),
    (16, 22),
    (19, 29),
    (23, 41),
    (24, 28),
    (24, 43),
    (25, 41),
    (26, 32),
    (29, 30),
    (37, 44),
}


@dataclass(frozen=True)
class PresetConfig:
    """Single preset configuration for filtering and sampling."""

    name: PresetName
    min_sum: int
    max_sum: int
    min_ac: int
    max_per_zone: int
    max_same_tail: int
    min_odd: int
    max_odd: int
    min_high: int
    max_high: int
    sample_size: int
    top_k: int
    max_overlap: int
    temperature: float
    min_prob_floor: float
    rare_pair_filter: bool
    excluded_numbers: frozenset[int]
    max_carryover_in_combo: int | None
    percentile_bias: int


PRESET_CONFIGS: dict[PresetName, PresetConfig] = {
    "A": PresetConfig(
        name="A",
        min_sum=100,
        max_sum=175,
        min_ac=7,
        max_per_zone=3,
        max_same_tail=2,
        min_odd=2,
        max_odd=4,
        min_high=2,
        max_high=4,
        sample_size=70_000,
        top_k=180,
        max_overlap=3,
        temperature=0.5,
        min_prob_floor=0.005,
        rare_pair_filter=True,
        excluded_numbers=frozenset({8}),
        max_carryover_in_combo=2,
        percentile_bias=-5,
    ),
    "B": PresetConfig(
        name="B",
        min_sum=90,
        max_sum=185,
        min_ac=5,
        max_per_zone=4,
        max_same_tail=3,
        min_odd=1,
        max_odd=5,
        min_high=1,
        max_high=5,
        sample_size=70_000,
        top_k=220,
        max_overlap=4,
        temperature=0.65,
        min_prob_floor=0.003,
        rare_pair_filter=False,
        excluded_numbers=frozenset(),
        max_carryover_in_combo=3,
        percentile_bias=8,
    ),
}


def contains_rare_pair(combination: tuple[int, ...]) -> bool:
    """Return True if combination includes one of the rare low-frequency pairs."""

    values = set(combination)
    for left, right in RARE_PAIR_COMBINATIONS:
        if left in values and right in values:
            return True
    return False


def exceeds_carryover_limit(combination: tuple[int, ...], carryover_numbers: set[int], limit: int) -> bool:
    """Return True when carryover number count in combo exceeds limit."""

    return len(set(combination).intersection(carryover_numbers)) > limit


class RecommendationService:
    """Adapter that converts engine output into MVP API response shape."""

    def __init__(self, history_csv: str | Path = "history.csv") -> None:
        self.history_csv = Path(history_csv)
        self.loader = LottoHistoryLoader()
        self.history = self.loader.load_and_validate(self.history_csv).reset_index(drop=True)
        self.historical_draws = [
            tuple(int(row[column]) for column in NUMBER_COLS)
            for _, row in self.history.iterrows()
        ]

    def recommend(self, preset: PresetName = "A", games: int = 5, seed: int | None = None) -> dict[str, object]:
        """Generate recommendations in the API response format."""

        config = self._resolve_preset(preset)
        self._validate_games(games)

        resolved_seed = seed if seed is not None else int(datetime.now().timestamp()) % 1_000_000

        raw_scores, boost_tags, carryover_numbers = self._build_scores(self.history)
        probabilities = ProbabilityNormalizer.to_sampling_probabilities(
            raw_scores,
            temperature=config.temperature,
            min_prob_floor=config.min_prob_floor,
        )

        sampled = MonteCarloSampler(sample_size=config.sample_size, chunk_size=20_000).sample(
            probabilities,
            seed=resolved_seed,
        )

        filtered = self._apply_base_filters(sampled, config)
        filtered = self._apply_preset_filters(filtered, carryover_numbers, config)

        if not filtered:
            filtered = [tuple(sorted(int(value) for value in combo)) for combo in sampled]

        ranked = CombinationRanker().rank(filtered, raw_scores, top_k=config.top_k)
        if not ranked:
            raise RuntimeError("추천 가능한 조합을 생성하지 못했습니다.")

        selected = self._select_with_diversity(ranked=ranked, games=games, max_overlap=config.max_overlap)
        percentile = self._calculate_percentile(selected, ranked, config.percentile_bias)

        rank_map = {entry.numbers: entry for entry in ranked}
        response_items: list[dict[str, object]] = []

        for numbers in selected:
            rank_entry = rank_map.get(numbers)
            score = (
                float(rank_entry.combo_score)
                if rank_entry is not None
                else float(sum(raw_scores.get(number, 0.0) for number in numbers))
            )
            response_items.append(
                {
                    "numbers": list(numbers),
                    "score": round(score, 6),
                    "tags": self._build_tags(numbers, boost_tags),
                    "reasons": self._build_reasons(config),
                }
            )

        return {
            "meta": {
                "preset": config.name,
                "percentile": percentile,
            },
            "recommendations": response_items,
        }

    @staticmethod
    def _resolve_preset(preset: str) -> PresetConfig:
        if preset not in PRESET_CONFIGS:
            raise ValueError("preset은 'A' 또는 'B'만 가능합니다.")
        return PRESET_CONFIGS[cast(PresetName, preset)]

    @staticmethod
    def _validate_games(games: int) -> None:
        if games not in ALLOWED_GAMES:
            raise ValueError("games는 5 또는 10만 가능합니다.")

    def _build_scores(self, history: pd.DataFrame) -> tuple[dict[int, float], dict[int, list[str]], set[int]]:
        base_scores = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0).calculate_scores(history, recent_n=50)

        booster = BoostCalculator(hot_threshold=2, hot_window=5, cold_window=10)
        boosts, boost_tags = booster.calculate_boosts(history)

        penalties = PenaltyCalculator(poisson_window=20, poisson_lambda=0.0, markov_lambda=0.0).calculate_penalties(
            history
        )

        hmm_boosts, hmm_tags = HMMScorer(hot_boost=0.3, cold_boost=0.15, window=100).calculate_boosts(history)

        combined_boosts = {number: boosts.get(number, 0.0) + hmm_boosts.get(number, 0.0) for number in range(1, 46)}
        for number, tags in hmm_tags.items():
            if tags:
                boost_tags.setdefault(number, []).extend(tags)

        raw_scores = ScoreEnsembler(minimum_score=0.0).combine(base_scores, combined_boosts, penalties)

        carryover_numbers = {
            number
            for number in range(1, 46)
            if "carryover" in boost_tags.get(number, []) or "carryover2" in boost_tags.get(number, [])
        }
        return raw_scores, boost_tags, carryover_numbers

    def _apply_base_filters(
        self,
        combinations: list[tuple[int, ...]],
        config: PresetConfig,
    ) -> list[tuple[int, ...]]:
        pipeline = FilterPipeline(
            [
                SumFilter(min_sum=config.min_sum, max_sum=config.max_sum),
                ACFilter(min_ac=config.min_ac),
                ZoneFilter(max_per_zone=config.max_per_zone),
                TailFilter(max_same_tail=config.max_same_tail),
                OddEvenFilter(min_odd=config.min_odd, max_odd=config.max_odd),
                HighLowFilter(min_high=config.min_high, max_high=config.max_high),
                HistoryFilter(historical_draws=self.historical_draws, match_threshold=5),
            ]
        )
        return pipeline.filter_combinations(combinations)

    @staticmethod
    def _apply_preset_filters(
        combinations: list[tuple[int, ...]],
        carryover_numbers: set[int],
        config: PresetConfig,
    ) -> list[tuple[int, ...]]:
        result: list[tuple[int, ...]] = []

        for combination in combinations:
            if config.rare_pair_filter and contains_rare_pair(combination):
                continue

            if config.excluded_numbers and set(combination).intersection(config.excluded_numbers):
                continue

            if config.max_carryover_in_combo is not None and exceeds_carryover_limit(
                combination,
                carryover_numbers,
                config.max_carryover_in_combo,
            ):
                continue

            result.append(combination)

        return result

    @staticmethod
    def _select_with_diversity(ranked: list[CombinationRank], games: int, max_overlap: int) -> list[tuple[int, ...]]:
        candidates = [entry.numbers for entry in ranked]
        selected = DiversitySelector(max_overlap=max_overlap).select(candidates, output_count=games)

        if len(selected) >= games:
            return selected

        # If strict overlap constraints produce fewer items, fill from ranked list.
        seen = set(selected)
        for numbers in candidates:
            if numbers in seen:
                continue
            selected.append(numbers)
            seen.add(numbers)
            if len(selected) >= games:
                break
        return selected

    @staticmethod
    def _calculate_percentile(
        selected: list[tuple[int, ...]],
        ranked: list[CombinationRank],
        percentile_bias: int,
    ) -> int | None:
        if not selected or not ranked:
            return None

        ranking_index = {entry.numbers: entry.rank for entry in ranked}
        values = [
            max(1, min(100, round((ranking_index[numbers] / len(ranked)) * 100)))
            for numbers in selected
            if numbers in ranking_index
        ]

        if not values:
            return None

        adjusted = int(round(sum(values) / len(values))) + percentile_bias
        return max(1, min(100, adjusted))

    @staticmethod
    def _build_reasons(config: PresetConfig) -> list[str]:
        reasons = [
            f"AC>={config.min_ac}",
            f"sum {config.min_sum}-{config.max_sum}",
            f"zone<={config.max_per_zone}",
            f"odd {config.min_odd}-{config.max_odd}",
            f"high {config.min_high}-{config.max_high}",
            "history overlap < 5",
        ]
        if config.max_carryover_in_combo is not None:
            reasons.append(f"carryover<={config.max_carryover_in_combo}")
        return reasons

    @staticmethod
    def _build_tags(combination: tuple[int, ...], boost_tags: dict[int, list[str]]) -> list[str]:
        tag_counts = Counter(tag for number in combination for tag in boost_tags.get(number, []))
        if not tag_counts:
            return []

        preferred_order = [
            "hmm_hot",
            "hot",
            "carryover",
            "carryover2",
            "neighbor",
            "reverse",
            "hmm_cold",
            "cold",
        ]
        tags = [tag for tag in preferred_order if tag_counts.get(tag)]
        return tags[:3]
