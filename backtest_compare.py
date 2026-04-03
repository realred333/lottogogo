#!/usr/bin/env python3
"""백테스트 비교: GA 모델 vs XGBoost 모델
지정 회차를 가리고 두 모델로 예측 후 실제 번호와 비교.
"""

import argparse
import pandas as pd
import recommend
import recommend_ml


def run_backtest(target_round: int, num_games: int, seed: int, weights_path: str | None):
    df = pd.read_csv("history.csv")
    actual_row = df[df["round"] == target_round]
    if actual_row.empty:
        print(f"❌ {target_round}회차 데이터가 없습니다.")
        return

    cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    actual = set(int(actual_row.iloc[0][c]) for c in cols)
    bonus = int(actual_row.iloc[0]["bonus"])

    # target_round 제거한 임시 CSV
    masked = df[df["round"] != target_round]
    tmp_csv = f"/tmp/backtest_{target_round}.csv"
    masked.to_csv(tmp_csv, index=False)

    print("=" * 60)
    print(f"🎯 백테스트 대상: {target_round}회")
    print(f"   실제 당첨번호: {sorted(actual)}  보너스: {bonus}")
    print("=" * 60)

    results = {}

    # ── GA 모델 ──
    print("\n[ GA 모델 ]")
    ga_combos = recommend.main(
        csv_path=tmp_csv,
        num_games=num_games,
        seed=seed,
        weights_path=weights_path,
    )
    results["GA"] = ga_combos

    # ── XGBoost 모델 ──
    print("\n[ XGBoost 모델 ]")
    xgb_combos = recommend_ml.main(
        csv_path=tmp_csv,
        num_games=num_games,
        seed=seed,
        weights_path=weights_path,
    )
    results["XGBoost"] = xgb_combos

    # ── 비교 결과 ──
    print("\n" + "=" * 60)
    print(f"📊 비교 결과  |  실제 당첨: {sorted(actual)}  보너스: {bonus}")
    print("=" * 60)

    for model_name, combos in results.items():
        print(f"\n  [{model_name}]")
        best = 0
        for i, combo in enumerate(combos, 1):
            matched = actual & set(combo)
            bonus_hit = bonus in set(combo)
            hit_str = f"{len(matched)}개 일치 {sorted(matched)}"
            if bonus_hit:
                hit_str += f"  +보너스({bonus})"
            print(f"    {i}게임 {sorted(combo)}: {hit_str}")
            best = max(best, len(matched))
        print(f"    → 최고 일치: {best}개")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA vs XGBoost 백테스트 비교")
    parser.add_argument("--round", type=int, required=True, help="검증할 회차 (가릴 회차)")
    parser.add_argument("--games", type=int, default=5, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--weights", default="data/optimized_weights.json", help="가중치 JSON 경로")
    args = parser.parse_args()

    run_backtest(
        target_round=args.round,
        num_games=args.games,
        seed=args.seed,
        weights_path=args.weights,
    )
