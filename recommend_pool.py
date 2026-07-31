#!/usr/bin/env python3
"""LottoGoGo pool-based recommender.

This experiment keeps the original recommend.py untouched. It does not use
carryover, neighbor, or reverse signals to decide exclusions. Instead, it removes
high-risk low-tier numbers first, then samples broadly from the remaining pool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.engine.filters import (
    ACFilter,
    ArithmeticProgressionFilter,
    FilterPipeline,
    HighLowFilter,
    HistoryFilter,
    OddEvenFilter,
    SumFilter,
    TailFilter,
    ZoneFilter,
)
from lottogogo.engine.ranker.diversity import DiversitySelector
from lottogogo.engine.ranker.scorer import CombinationRanker

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
ALL_NUMBERS = set(range(1, 46))

# Hard exclusions are treated as unavailable from the start, not as post-filters.
HARD_EXCLUDE_NUMBERS = {2, 5, 8, 9, 22, 32, 39}

RARE_PAIR_COMBINATIONS = {
    (8, 12),
    (8, 26),
    (24, 43),
    (26, 32),
    (1, 22),
    (3, 25),
    (4, 30),
    (6, 33),
    (8, 9),
    (11, 40),
    (23, 41),
    (6, 23),
    (6, 29),
    (7, 32),
    (9, 20),
    (11, 34),
    (16, 22),
    (19, 29),
    (24, 28),
    (25, 41),
    (29, 30),
    (37, 44),
}


@dataclass(frozen=True)
class PoolPlan:
    """Selected number pool and score details."""

    pool: list[int]
    hard_excluded: list[int]
    risk_excluded: list[int]
    risk_scores: dict[int, float]
    detail_rows: list[dict[str, float | int]]


def _normalize(values: dict[int, float], candidates: list[int], *, invert: bool = False) -> dict[int, float]:
    selected = [float(values.get(number, 0.0)) for number in candidates]
    lo = min(selected) if selected else 0.0
    hi = max(selected) if selected else 0.0
    if hi <= lo:
        return {number: 0.0 for number in candidates}
    result = {number: (float(values.get(number, 0.0)) - lo) / (hi - lo) for number in candidates}
    if invert:
        result = {number: 1.0 - value for number, value in result.items()}
    return result


def _number_rounds(history: pd.DataFrame, number: int) -> list[int]:
    mask = history[NUMBER_COLS].eq(number).any(axis=1)
    return history.loc[mask, "round"].astype(int).tolist()


def _frequency(history: pd.DataFrame, window: int) -> dict[int, int]:
    recent = history.tail(window)
    counts = Counter()
    for _, row in recent.iterrows():
        counts.update(int(row[column]) for column in NUMBER_COLS)
    return {number: counts.get(number, 0) for number in range(1, 46)}


def _pair_counts(history: pd.DataFrame) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for _, row in history.iterrows():
        numbers = sorted(int(row[column]) for column in NUMBER_COLS)
        counts.update(combinations(numbers, 2))
    return counts


def calculate_risk_scores(history: pd.DataFrame) -> tuple[dict[int, float], list[dict[str, float | int]]]:
    """Score numbers for exclusion risk.

    Higher score means "more suitable to remove from the pool." This deliberately
    avoids carryover, neighbor, and reverse signals.
    """

    ordered = history.sort_values("round").reset_index(drop=True)
    latest_round = int(ordered["round"].max())
    candidates = sorted(ALL_NUMBERS - HARD_EXCLUDE_NUMBERS)

    f5 = _frequency(ordered, 5)
    f20 = _frequency(ordered, 20)
    f50 = _frequency(ordered, 50)
    pairs = _pair_counts(ordered)

    raw: dict[str, dict[int, float]] = {
        "avg_gap": {},
        "max_gap": {},
        "p90_gap": {},
        "gap_ge30": {},
        "current_gap": {},
        "recent50_low": {},
        "rare_pair_le10": {},
        "rare_pair_le12": {},
        "pair_avg_low": {},
        "overheat": {},
    }

    detail_rows: list[dict[str, float | int]] = []

    for number in candidates:
        rounds = _number_rounds(ordered, number)
        gaps = [right - left for left, right in zip(rounds, rounds[1:])]
        current_gap = latest_round - rounds[-1] if rounds else latest_round

        if gaps:
            gap_array = np.array(gaps, dtype=float)
            avg_gap = float(gap_array.mean())
            max_gap = float(gap_array.max())
            p90_gap = float(np.quantile(gap_array, 0.9))
            gap_ge30 = float((gap_array >= 30).sum())
        else:
            avg_gap = 0.0
            max_gap = 0.0
            p90_gap = 0.0
            gap_ge30 = 0.0

        pair_values = [
            pairs.get((min(number, other), max(number, other)), 0)
            for other in range(1, 46)
            if other != number
        ]
        rare_le10 = float(sum(value <= 10 for value in pair_values))
        rare_le12 = float(sum(value <= 12 for value in pair_values))
        pair_avg = float(np.mean(pair_values)) if pair_values else 0.0

        raw["avg_gap"][number] = avg_gap
        raw["max_gap"][number] = max_gap
        raw["p90_gap"][number] = p90_gap
        raw["gap_ge30"][number] = gap_ge30
        raw["current_gap"][number] = float(current_gap)
        raw["recent50_low"][number] = float(f50[number])
        raw["rare_pair_le10"][number] = rare_le10
        raw["rare_pair_le12"][number] = rare_le12
        raw["pair_avg_low"][number] = pair_avg
        raw["overheat"][number] = float(f20[number] * 0.65 + f5[number] * 1.4)

        detail_rows.append(
            {
                "number": number,
                "count": len(rounds),
                "avg_gap": round(avg_gap, 3),
                "max_gap": int(max_gap),
                "p90_gap": round(p90_gap, 3),
                "gap_ge30": int(gap_ge30),
                "current_gap": int(current_gap),
                "recent50": f50[number],
                "recent20": f20[number],
                "recent5": f5[number],
                "rare_pair_le10": int(rare_le10),
                "rare_pair_le12": int(rare_le12),
                "pair_avg": round(pair_avg, 3),
            }
        )

    avg_gap_n = _normalize(raw["avg_gap"], candidates)
    max_gap_n = _normalize(raw["max_gap"], candidates)
    p90_gap_n = _normalize(raw["p90_gap"], candidates)
    gap_ge30_n = _normalize(raw["gap_ge30"], candidates)
    current_gap_n = _normalize(raw["current_gap"], candidates)
    recent50_low_n = _normalize(raw["recent50_low"], candidates, invert=True)
    rare10_n = _normalize(raw["rare_pair_le10"], candidates)
    rare12_n = _normalize(raw["rare_pair_le12"], candidates)
    pair_avg_low_n = _normalize(raw["pair_avg_low"], candidates, invert=True)
    overheat_n = _normalize(raw["overheat"], candidates)

    risk_scores: dict[int, float] = {}
    for number in candidates:
        long_gap = (
            0.35 * avg_gap_n[number]
            + 0.25 * gap_ge30_n[number]
            + 0.20 * max_gap_n[number]
            + 0.20 * p90_gap_n[number]
        )
        silence_low = 0.55 * current_gap_n[number] + 0.45 * recent50_low_n[number]
        rare_pair = 0.50 * rare10_n[number] + 0.25 * rare12_n[number] + 0.25 * pair_avg_low_n[number]
        overheat = overheat_n[number]
        risk_scores[number] = 0.40 * long_gap + 0.25 * silence_low + 0.25 * rare_pair + 0.10 * overheat

    for row in detail_rows:
        number = int(row["number"])
        row["risk"] = round(risk_scores[number], 6)

    return risk_scores, detail_rows


def build_pool(history: pd.DataFrame, pool_size: int) -> PoolPlan:
    if pool_size < 6:
        raise ValueError("pool_size must be >= 6")

    risk_scores, detail_rows = calculate_risk_scores(history)
    base_pool = sorted(ALL_NUMBERS - HARD_EXCLUDE_NUMBERS)
    target_size = min(pool_size, len(base_pool))
    risk_exclude_count = max(0, len(base_pool) - target_size)
    risk_excluded = [
        number
        for number, _ in sorted(risk_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[
            :risk_exclude_count
        ]
    ]
    pool = sorted(number for number in base_pool if number not in set(risk_excluded))

    return PoolPlan(
        pool=pool,
        hard_excluded=sorted(HARD_EXCLUDE_NUMBERS),
        risk_excluded=sorted(risk_excluded),
        risk_scores=risk_scores,
        detail_rows=sorted(detail_rows, key=lambda row: (float(row["risk"]), int(row["number"])), reverse=True),
    )


def _sample_from_pool(pool: list[int], sample_size: int, seed: int | None) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    pool_array = np.array(pool, dtype=int)
    result: list[tuple[int, ...]] = []
    for _ in range(sample_size):
        picked = rng.choice(pool_array, size=6, replace=False)
        result.append(tuple(sorted(int(value) for value in picked)))
    return result


def _contains_rare_pair(combination: tuple[int, ...]) -> bool:
    values = set(combination)
    return any(left in values and right in values for left, right in RARE_PAIR_COMBINATIONS)


def _historical_draws(history: pd.DataFrame) -> list[tuple[int, ...]]:
    return [tuple(int(row[column]) for column in NUMBER_COLS) for _, row in history.iterrows()]


def generate_recommendations(
    history: pd.DataFrame,
    *,
    games: int = 5,
    seed: int | None = 42,
    pool_size: int = 30,
    sample_size: int = 100_000,
    max_overlap: int = 3,
) -> tuple[list[tuple[int, ...]], PoolPlan, dict[str, int]]:
    plan = build_pool(history, pool_size=pool_size)
    sampled = _sample_from_pool(plan.pool, sample_size=sample_size, seed=seed)

    pipeline = FilterPipeline(
        [
            SumFilter(min_sum=100, max_sum=175),
            ACFilter(min_ac=7),
            ZoneFilter(max_per_zone=3),
            TailFilter(max_same_tail=2),
            OddEvenFilter(min_odd=2, max_odd=4),
            HighLowFilter(min_high=2, max_high=4),
            HistoryFilter(historical_draws=_historical_draws(history), match_threshold=5),
            ArithmeticProgressionFilter(min_ap_length=5),
        ]
    )

    filtered = pipeline.filter_combinations(sampled)
    filtered = [combo for combo in filtered if not _contains_rare_pair(combo)]
    unique_filtered = list(dict.fromkeys(filtered))

    # Higher score means lower exclusion risk. This is intentionally weak and broad.
    score_map = {number: 1.0 - plan.risk_scores.get(number, 0.0) for number in range(1, 46)}
    ranked = CombinationRanker().rank(unique_filtered, score_map)
    selected = DiversitySelector(max_overlap=max_overlap).select(
        [entry.numbers for entry in ranked],
        output_count=games,
    )

    stats = {
        "sampled": len(sampled),
        "filtered": len(filtered),
        "unique_filtered": len(unique_filtered),
        "pool_size": len(plan.pool),
    }
    return selected, plan, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Pool-based LottoGoGo recommender")
    parser.add_argument("--csv", default="history.csv")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--show-risk", type=int, default=12, help="Show top N risk-excluded candidates/details")
    args = parser.parse_args()

    history = LottoHistoryLoader().load_and_validate(args.csv).reset_index(drop=True)
    latest_round = int(history["round"].max())
    selected, plan, stats = generate_recommendations(
        history,
        games=args.games,
        seed=args.seed,
        pool_size=args.pool_size,
        sample_size=args.sample_size,
    )

    print("=" * 60)
    print("LottoGoGo Pool Recommender")
    print("=" * 60)
    print(f"data rows       : {len(history)}")
    print(f"latest round    : {latest_round}")
    print(f"target round    : {latest_round + 1}")
    print(f"seed            : {args.seed}")
    print(f"hard excluded   : {plan.hard_excluded}")
    print(f"risk excluded   : {plan.risk_excluded}")
    print(f"pool ({len(plan.pool)})       : {plan.pool}")
    print(f"sampled/filter  : {stats['sampled']:,} -> {stats['filtered']:,} -> {stats['unique_filtered']:,}")

    if args.show_risk > 0:
        print("\nTop risk numbers:")
        for row in plan.detail_rows[: args.show_risk]:
            print(
                f"  {int(row['number']):2d} risk={float(row['risk']):.4f} "
                f"avg_gap={row['avg_gap']} max_gap={row['max_gap']} current_gap={row['current_gap']} "
                f"recent50={row['recent50']} rare<=10={row['rare_pair_le10']}"
            )

    print("\nRecommendations:")
    for index, combo in enumerate(selected, 1):
        print(f"  {index}게임: [{', '.join(f'{number:2d}' for number in combo)}] sum={sum(combo)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
