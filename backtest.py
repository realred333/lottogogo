#!/usr/bin/env python3
"""LottoGoGo - 백테스트: 특정 회차 기준으로 추천 번호의 적중률 테스트

추천 파이프라인을 여기서 다시 구현하지 않는다. recommend.generate_recommendations를
그대로 호출하므로, 백테스트는 정의상 실제 추천과 동일한 조건으로 돈다.
(과거에는 이 파일이 파이프라인을 복사해 갖고 있었고, AP 필터·고구간 하향·가중치
주입·다양성 완화가 빠진 채 조용히 갈라져 있었다.)
"""

import sys
from datetime import datetime

import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from recommend import generate_recommendations, load_weights

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# 랜덤 기준선: 6개 중 K개를 무작위로 골랐을 때의 기대 적중 수 = 6 * 6 / 45
RANDOM_EXPECTED_MATCH = 6 * 6 / 45


def count_matches(prediction: tuple[int, ...], actual: tuple[int, ...], bonus: int) -> tuple[int, int]:
    """예측 번호와 실제 당첨번호 비교, (메인 일치 수, 보너스 일치 여부) 반환"""
    pred_set = set(prediction)
    actual_set = set(actual)
    main_matches = len(pred_set & actual_set)
    bonus_match = 1 if bonus in pred_set else 0
    return main_matches, bonus_match


def get_prize_rank(main_matches: int, bonus_match: int) -> str:
    """당첨 등수 반환"""
    if main_matches == 6:
        return "🏆 1등!"
    elif main_matches == 5 and bonus_match:
        return "🥈 2등!"
    elif main_matches == 5:
        return "🥉 3등!"
    elif main_matches == 4:
        return "4등"
    elif main_matches == 3:
        return "5등"
    else:
        return "낙첨"


def backtest(
    csv_path: str,
    target_round: int,
    num_games: int = 5,
    seed: int | None = None,
    weights_path: str | None = None,
    verbose: bool = True,
):
    """특정 회차 기준으로 백테스트 실행.

    target_round까지의 데이터로 학습하고 target_round+1 회차를 예측한다.
    """

    # 시드 설정 (없으면 현재 시간 기반)
    if seed is None:
        seed = int(datetime.now().timestamp()) % 100000

    # 데이터 로드
    loader = LottoHistoryLoader()
    try:
        full_history = loader.load_and_validate(csv_path).reset_index(drop=True)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)

    max_round = int(full_history["round"].max())

    if target_round >= max_round:
        print(f"❌ target_round({target_round})는 최신 회차({max_round})보다 작아야 합니다.")
        sys.exit(1)

    if target_round < 100:
        print(f"❌ target_round({target_round})는 최소 100 이상이어야 합니다. (데이터 부족)")
        sys.exit(1)

    # target_round까지의 데이터만 사용 (미래 데이터 유출 없음)
    history = full_history[full_history["round"] <= target_round].copy()

    # 다음 회차 (정답) 데이터
    next_round_data = full_history[full_history["round"] == target_round + 1].iloc[0]
    actual_numbers = tuple(int(next_round_data[col]) for col in NUMBER_COLS)
    bonus_number = int(next_round_data["bonus"])

    if verbose:
        print("=" * 60)
        print("🧪 LottoGoGo - 백테스트")
        print("=" * 60)
        print(f"📊 학습 데이터: 1회 ~ {target_round}회 ({len(history)}회차)")
        print(f"🎯 예측 대상: {target_round + 1}회차")
        print(f"🎲 Seed: {seed}")
        print()

    weights = load_weights(weights_path, verbose=verbose)

    # 추천 번호 생성 — recommend.py와 완전히 동일한 경로
    if verbose:
        print("⏳ 추천 번호 생성 중...")
    recommendations = generate_recommendations(
        history,
        num_games=num_games,
        seed=seed,
        weights=weights,
        verbose=False,
    )

    hits = [count_matches(combo, actual_numbers, bonus_number) for combo in recommendations]
    main_hits = [main for main, _ in hits]
    best_match = max(main_hits) if main_hits else 0
    avg_match = sum(main_hits) / len(main_hits) if main_hits else 0.0

    if verbose:
        print()
        print("=" * 60)
        print(f"📋 {target_round + 1}회차 예측 결과")
        print("=" * 60)

        actual_str = ", ".join(f"{n:2d}" for n in actual_numbers)
        print(f"\n🎱 실제 당첨번호: [{actual_str}] + 보너스: {bonus_number}\n")

        best_result = "낙첨"
        for i, (combo, (main_matches, bonus_match)) in enumerate(zip(recommendations, hits), 1):
            numbers_str = ", ".join(f"{n:2d}" for n in combo)
            prize = get_prize_rank(main_matches, bonus_match)

            match_indicator = f"({main_matches}개 일치"
            if bonus_match:
                match_indicator += " +보너스"
            match_indicator += ")"

            print(f"  {i}게임: [{numbers_str}] → {match_indicator} {prize}")

            if main_matches == best_match:
                best_result = prize

        print()
        print("=" * 60)
        print(f"✨ 최고 결과: {best_match}개 일치 - {best_result}")
        print(f"📊 평균 일치: {avg_match:.2f}개  (랜덤 기대값 {RANDOM_EXPECTED_MATCH:.2f}개)")
        print(f"   → 랜덤 대비 {avg_match - RANDOM_EXPECTED_MATCH:+.2f}개")
        print("=" * 60)

    return {
        "round": target_round + 1,
        "actual": actual_numbers,
        "bonus": bonus_number,
        "recommendations": recommendations,
        "main_hits": main_hits,
        "best": best_match,
        "avg": avg_match,
    }


def run_range(
    csv_path: str,
    last_n: int,
    num_games: int,
    seed: int | None,
    weights_path: str | None,
) -> None:
    """최근 N개 회차를 연속 백테스트하고 랜덤 기준선과 비교한다."""
    loader = LottoHistoryLoader()
    full_history = loader.load_and_validate(csv_path).reset_index(drop=True)
    rounds = sorted(int(r) for r in full_history["round"].tolist())

    # target_round+1을 예측하므로 마지막 회차는 target이 될 수 없다
    targets = [r for r in rounds if r >= 100][-(last_n + 1):-1]
    if not targets:
        print("❌ 백테스트할 회차가 부족합니다.")
        sys.exit(1)

    print("=" * 60)
    print(f"🧪 연속 백테스트: {targets[0] + 1}회 ~ {targets[-1] + 1}회 ({len(targets)}회차 × {num_games}게임)")
    print("=" * 60)

    summaries = []
    for target_round in targets:
        result = backtest(
            csv_path=csv_path,
            target_round=target_round,
            num_games=num_games,
            seed=seed,
            weights_path=weights_path,
            verbose=False,
        )
        summaries.append(result)
        print(
            f"  {result['round']:>5}회  최고 {result['best']}개  평균 {result['avg']:.2f}개  "
            f"실제 {list(result['actual'])}"
        )

    all_hits = [hit for row in summaries for hit in row["main_hits"]]
    avg = sum(all_hits) / len(all_hits)
    best_overall = max(row["best"] for row in summaries)
    ge3 = sum(1 for hit in all_hits if hit >= 3)
    ge4 = sum(1 for hit in all_hits if hit >= 4)

    print()
    print("=" * 60)
    print(f"📊 전체 요약 ({len(summaries)}회차 × {num_games}게임 = {len(all_hits)}게임)")
    print("=" * 60)
    print(f"  평균 일치   : {avg:.3f}개   (랜덤 기대값 {RANDOM_EXPECTED_MATCH:.3f}개)")
    print(f"  랜덤 대비   : {avg - RANDOM_EXPECTED_MATCH:+.3f}개")
    print(f"  최고 일치   : {best_overall}개")
    print(f"  3개 이상    : {ge3}/{len(all_hits)}게임 ({ge3 / len(all_hits) * 100:.1f}%)")
    print(f"  4개 이상    : {ge4}/{len(all_hits)}게임 ({ge4 / len(all_hits) * 100:.1f}%)")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LottoGoGo 백테스트")
    parser.add_argument("--csv", default="history.csv", help="CSV 파일 경로")
    parser.add_argument("--round", type=int, default=None, help="기준 회차 (이 회차까지 학습, 다음 회차 예측)")
    parser.add_argument("--last", type=int, default=None, help="최근 N개 회차를 연속 백테스트")
    parser.add_argument("--games", type=int, default=5, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--weights", default=None, help="최적화 가중치 JSON 경로 (recommend.py와 동일)")

    args = parser.parse_args()

    if args.last is not None:
        run_range(
            csv_path=args.csv,
            last_n=args.last,
            num_games=args.games,
            seed=args.seed,
            weights_path=args.weights,
        )
        return

    if args.round is None:
        parser.error("--round 또는 --last 중 하나는 필요합니다.")

    backtest(
        csv_path=args.csv,
        target_round=args.round,
        num_games=args.games,
        seed=args.seed,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
