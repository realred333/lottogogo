#!/usr/bin/env python3
"""다회차 백테스트: GA vs XGBoost 평균 적중 비교"""

import argparse
import pandas as pd
import recommend
import recommend_ml


def run_multi_backtest(rounds: list[int], num_games: int, seed: int, weights_path: str | None):
    df = pd.read_csv("history.csv")
    cols = ["n1", "n2", "n3", "n4", "n5", "n6"]

    summary = []

    for target_round in rounds:
        actual_row = df[df["round"] == target_round]
        if actual_row.empty:
            continue

        actual = set(int(actual_row.iloc[0][c]) for c in cols)
        bonus = int(actual_row.iloc[0]["bonus"])

        masked = df[df["round"] != target_round]
        tmp_csv = f"/tmp/backtest_{target_round}.csv"
        masked.to_csv(tmp_csv, index=False)

        print(f"\n{'='*60}")
        print(f"▶ {target_round}회  실제: {sorted(actual)}  보너스: {bonus}")
        print(f"{'='*60}")

        results = {}

        print("  [GA 모델]")
        ga_combos = recommend.main(csv_path=tmp_csv, num_games=num_games, seed=seed, weights_path=weights_path)
        results["GA"] = ga_combos

        print("\n  [XGBoost 모델]")
        xgb_combos = recommend_ml.main(csv_path=tmp_csv, num_games=num_games, seed=seed, weights_path=weights_path)
        results["XGBoost"] = xgb_combos

        row = {"round": target_round, "actual": sorted(actual), "bonus": bonus}
        for model_name, combos in results.items():
            hits = [len(actual & set(c)) for c in combos]
            bonus_hits = sum(1 for c in combos if bonus in set(c))
            best = max(hits)
            avg = sum(hits) / len(hits)
            row[f"{model_name}_best"] = best
            row[f"{model_name}_avg"] = round(avg, 2)
            row[f"{model_name}_bonus"] = bonus_hits

            print(f"\n  [{model_name}]")
            for i, (combo, h) in enumerate(zip(combos, hits), 1):
                b = " +보너스" if bonus in set(combo) else ""
                print(f"    {i:2d}게임 {sorted(combo)}: {h}개{b}")
            print(f"    → 최고: {best}개 | 평균: {avg:.2f}개 | 보너스 포함 게임: {bonus_hits}/{num_games}")

        summary.append(row)

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"📊 전체 요약 ({len(summary)}회차 × {num_games}게임)")
    print(f"{'='*60}")
    print(f"  {'회차':>5} | {'실제번호':<28} | {'GA최고':>6} | {'GA평균':>6} | {'XGB최고':>7} | {'XGB평균':>7}")
    print(f"  {'-'*5}-+-{'-'*28}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    ga_bests, ga_avgs = [], []
    xgb_bests, xgb_avgs = [], []

    for r in summary:
        actual_str = str(r["actual"])
        print(f"  {r['round']:>5} | {actual_str:<28} | {r['GA_best']:>6} | {r['GA_avg']:>6} | {r['XGBoost_best']:>7} | {r['XGBoost_avg']:>7}")
        ga_bests.append(r["GA_best"])
        ga_avgs.append(r["GA_avg"])
        xgb_bests.append(r["XGBoost_best"])
        xgb_avgs.append(r["XGBoost_avg"])

    print(f"  {'-'*5}-+-{'-'*28}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
    print(f"  {'평균':>5} | {'':<28} | {sum(ga_bests)/len(ga_bests):>6.2f} | {sum(ga_avgs)/len(ga_avgs):>6.2f} | {sum(xgb_bests)/len(xgb_bests):>7.2f} | {sum(xgb_avgs)/len(xgb_avgs):>7.2f}")
    print()

    winner = "XGBoost" if sum(xgb_avgs) > sum(ga_avgs) else "GA"
    print(f"  🏆 평균 적중 우세: {winner}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", type=int, help="검증할 회차 목록")
    parser.add_argument("--last", type=int, default=10, help="최근 N회차 자동 선택")
    parser.add_argument("--games", type=int, default=10, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="data/optimized_weights.json")
    args = parser.parse_args()

    if args.rounds:
        target_rounds = args.rounds
    else:
        df = pd.read_csv("history.csv")
        target_rounds = sorted(df["round"].tolist(), reverse=True)[1:args.last + 1]  # 최신 제외, 그 다음부터
        print(f"자동 선택 회차: {target_rounds}")

    run_multi_backtest(
        rounds=target_rounds,
        num_games=args.games,
        seed=args.seed,
        weights_path=args.weights,
    )
