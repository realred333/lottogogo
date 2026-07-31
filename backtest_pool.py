#!/usr/bin/env python3
"""Backtest for recommend_pool.py without touching the original backtest.py."""

from __future__ import annotations

import argparse

import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader

from recommend_pool import NUMBER_COLS, generate_recommendations


def _actual_numbers(row: pd.Series) -> tuple[int, ...]:
    return tuple(int(row[column]) for column in NUMBER_COLS)


def _run_one(
    full_history: pd.DataFrame,
    target_round: int,
    *,
    games: int,
    seed: int,
    pool_size: int,
    sample_size: int,
) -> dict[str, object]:
    target_row = full_history[full_history["round"] == target_round]
    if target_row.empty:
        raise ValueError(f"target round not found: {target_round}")

    train = full_history[full_history["round"] < target_round].copy()
    if len(train) < 100:
        raise ValueError("target round requires at least 100 prior rows")

    actual = set(_actual_numbers(target_row.iloc[0]))
    bonus = int(target_row.iloc[0].get("bonus", 0))

    combos, plan, stats = generate_recommendations(
        train,
        games=games,
        seed=seed,
        pool_size=pool_size,
        sample_size=sample_size,
    )
    hits = [len(set(combo) & actual) for combo in combos]
    bonus_hits = [bonus in set(combo) for combo in combos]

    print("=" * 60)
    print(f"{target_round}회 백테스트")
    print(f"actual          : {sorted(actual)} + bonus {bonus}")
    print(f"hard excluded   : {plan.hard_excluded}")
    print(f"risk excluded   : {plan.risk_excluded}")
    print(f"pool ({len(plan.pool)})       : {plan.pool}")
    print(f"sampled/filter  : {stats['sampled']:,} -> {stats['filtered']:,} -> {stats['unique_filtered']:,}")
    print()
    for index, (combo, hit, bonus_hit) in enumerate(zip(combos, hits, bonus_hits), 1):
        matched = sorted(set(combo) & actual)
        bonus_text = " +bonus" if bonus_hit else ""
        print(f"  {index}게임 {list(combo)}: {hit}개 {matched}{bonus_text}")
    print(f"  => best={max(hits) if hits else 0}, avg={sum(hits) / len(hits) if hits else 0:.2f}, bonus_games={sum(bonus_hits)}")
    print()

    return {
        "round": target_round,
        "best": max(hits) if hits else 0,
        "avg": sum(hits) / len(hits) if hits else 0.0,
        "ge2": sum(1 for hit in hits if hit >= 2),
        "ge3": sum(1 for hit in hits if hit >= 3),
        "bonus_games": sum(bonus_hits),
        "unique_filtered": stats["unique_filtered"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest pool-based recommender")
    parser.add_argument("--csv", default="history.csv")
    parser.add_argument("--target-round", type=int, default=None)
    parser.add_argument("--last", type=int, default=None, help="Backtest the latest N known rounds")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=100_000)
    args = parser.parse_args()

    full_history = LottoHistoryLoader().load_and_validate(args.csv).reset_index(drop=True)
    full_history = full_history.sort_values("round").reset_index(drop=True)

    if args.last is not None:
        rounds = full_history["round"].astype(int).tolist()[-args.last :]
    elif args.target_round is not None:
        rounds = [args.target_round]
    else:
        rounds = [int(full_history["round"].max())]

    summaries = [
        _run_one(
            full_history,
            target_round,
            games=args.games,
            seed=args.seed,
            pool_size=args.pool_size,
            sample_size=args.sample_size,
        )
        for target_round in rounds
    ]

    if len(summaries) > 1:
        print("=" * 60)
        print(f"summary ({len(summaries)} rounds x {args.games} games)")
        print(
            f"avg_best={sum(float(row['best']) for row in summaries) / len(summaries):.2f}, "
            f"avg_hit_per_game={sum(float(row['avg']) for row in summaries) / len(summaries):.2f}, "
            f"ge2_games={sum(int(row['ge2']) for row in summaries)}, "
            f"ge3_games={sum(int(row['ge3']) for row in summaries)}, "
            f"bonus_games={sum(int(row['bonus_games']) for row in summaries)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
