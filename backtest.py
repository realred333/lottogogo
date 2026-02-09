#!/usr/bin/env python3
"""LottoGoGo - 백테스트: 특정 회차 기준으로 추천 번호의 적중률 테스트"""

import sys
from datetime import datetime

import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.engine.score.calculator import BaseScoreCalculator, ScoreEnsembler
from lottogogo.engine.score.booster import BoostCalculator
from lottogogo.engine.score.penalizer import PenaltyCalculator
from lottogogo.engine.score.hmm_scorer import HMMScorer
from lottogogo.engine.score.normalizer import ProbabilityNormalizer
from lottogogo.engine.sampler.monte_carlo import MonteCarloSampler
from lottogogo.engine.filters import (
    FilterPipeline,
    SumFilter,
    ACFilter,
    ZoneFilter,
    TailFilter,
    OddEvenFilter,
    HighLowFilter,
    HistoryFilter,
)
from lottogogo.engine.ranker.scorer import CombinationRanker
from lottogogo.engine.ranker.diversity import DiversitySelector

NUMBER_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

# ============================================================
# 🎰 도박사의 오류 필터링 설정 (recommend.py와 동일)
# ============================================================
RARE_PAIR_COMBINATIONS = [
    (8, 12), (8, 26), (24, 43), (26, 32),
    (1, 22), (3, 25), (4, 30), (6, 33), (8, 9), (11, 40), (23, 41),
    (6, 23), (6, 29), (7, 32), (9, 20), (11, 34), (16, 22),
    (19, 29), (24, 28), (25, 41), (29, 30), (37, 44),
]
EXCLUDE_NUMBERS = {8}
# 이월수(carryover) 제한: 조합에 포함될 최대 이월수 개수 (직전 + 2주전 합계)
MAX_CARRYOVER_IN_COMBO = 2


def contains_rare_pair(combination: tuple[int, ...]) -> bool:
    nums = set(combination)
    for a, b in RARE_PAIR_COMBINATIONS:
        if a in nums and b in nums:
            return True
    return False


def contains_excluded_number(combination: tuple[int, ...]) -> bool:
    return bool(set(combination) & EXCLUDE_NUMBERS)


def exceeds_carryover_limit(combination: tuple[int, ...], carryover_numbers: set[int]) -> bool:
    carryover_count = len(set(combination) & carryover_numbers)
    return carryover_count > MAX_CARRYOVER_IN_COMBO


def generate_recommendations(history: pd.DataFrame, num_games: int = 5, seed: int = 42) -> list[tuple[int, ...]]:
    """주어진 히스토리 기준으로 추천 번호 생성"""
    
    # 과거 당첨번호 추출
    historical_draws = [
        tuple(row[NUMBER_COLS].astype(int).tolist()) 
        for _, row in history.iterrows()
    ]

    # 점수 계산
    base_calc = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
    booster = BoostCalculator(hot_threshold=2, hot_window=5, cold_window=10)
    penalizer = PenaltyCalculator(poisson_window=20, poisson_lambda=0.0, markov_lambda=0.0)
    ensembler = ScoreEnsembler(minimum_score=0.0)

    base_scores = base_calc.calculate_scores(history, recent_n=50)
    boosts, boost_tags = booster.calculate_boosts(history)
    penalties = penalizer.calculate_penalties(history)
    
    # HMM scoring
    hmm_scorer = HMMScorer(hot_boost=0.3, cold_boost=0.15, window=100)
    hmm_boosts, hmm_tags = hmm_scorer.calculate_boosts(history)
    
    # Combine boosts (regular + HMM)
    combined_boosts = {n: boosts.get(n, 0) + hmm_boosts.get(n, 0) for n in range(1, 46)}
    for n, tags in hmm_tags.items():
        if tags:
            boost_tags.setdefault(n, []).extend(tags)
    
    raw_scores = ensembler.combine(base_scores, combined_boosts, penalties)

    # Carryover 번호 (직전 + 2주전)
    carryover_numbers = set(n for n in range(1, 46) if "carryover" in boost_tags.get(n, []))
    carryover2_numbers = set(n for n in range(1, 46) if "carryover2" in boost_tags.get(n, []))
    all_carryover_numbers = carryover_numbers | carryover2_numbers

    # 확률 변환
    probs = ProbabilityNormalizer.to_sampling_probabilities(
        raw_scores, temperature=0.5, min_prob_floor=0.005
    )

    # 조합 생성
    sampler = MonteCarloSampler(sample_size=100000, chunk_size=20000)
    combinations = sampler.sample(probs, seed=seed)

    # 기본 필터링
    pipeline = FilterPipeline([
        SumFilter(min_sum=100, max_sum=175),
        ACFilter(min_ac=7),
        ZoneFilter(max_per_zone=3),
        TailFilter(max_same_tail=2),
        OddEvenFilter(min_odd=2, max_odd=4),
        HighLowFilter(min_high=2, max_high=4),
        HistoryFilter(historical_draws=historical_draws, match_threshold=5),
    ])
    filtered = pipeline.filter_combinations(combinations)

    # 도박사의 오류 필터링
    filtered = [
        combo for combo in filtered
        if not contains_rare_pair(combo) 
        and not contains_excluded_number(combo)
        and not exceeds_carryover_limit(combo, all_carryover_numbers)
    ]

    # 랭킹 및 다양성 적용
    ranker = CombinationRanker()
    ranked = ranker.rank(filtered, raw_scores, top_k=100)
    selector = DiversitySelector(max_overlap=3)
    final = selector.select([r.numbers for r in ranked], output_count=num_games)

    return final


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


def backtest(csv_path: str, target_round: int, num_games: int = 5, seed: int | None = None):
    """특정 회차 기준으로 백테스트 실행"""
    
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
    
    # target_round까지의 데이터만 사용
    history = full_history[full_history["round"] <= target_round].copy()
    
    # 다음 회차 (정답) 데이터
    next_round_data = full_history[full_history["round"] == target_round + 1].iloc[0]
    actual_numbers = tuple(int(next_round_data[col]) for col in NUMBER_COLS)
    bonus_number = int(next_round_data["bonus"])
    
    print("=" * 60)
    print("🧪 LottoGoGo - 백테스트")
    print("=" * 60)
    print(f"📊 학습 데이터: 1회 ~ {target_round}회 ({len(history)}회차)")
    print(f"🎯 예측 대상: {target_round + 1}회차")
    print(f"🎲 Seed: {seed}")
    print()
    
    # 추천 번호 생성
    print("⏳ 추천 번호 생성 중...")
    recommendations = generate_recommendations(history, num_games=num_games, seed=seed)
    
    # 결과 출력
    print()
    print("=" * 60)
    print(f"📋 {target_round + 1}회차 예측 결과")
    print("=" * 60)
    
    actual_str = ", ".join(f"{n:2d}" for n in actual_numbers)
    print(f"\n🎱 실제 당첨번호: [{actual_str}] + 보너스: {bonus_number}\n")
    
    best_match = 0
    best_result = ""
    
    for i, combo in enumerate(recommendations, 1):
        numbers_str = ", ".join(f"{n:2d}" for n in combo)
        main_matches, bonus_match = count_matches(combo, actual_numbers, bonus_number)
        prize = get_prize_rank(main_matches, bonus_match)
        
        # 일치하는 번호 표시
        matched = set(combo) & set(actual_numbers)
        match_indicator = f"({main_matches}개 일치"
        if bonus_match:
            match_indicator += " +보너스"
        match_indicator += ")"
        
        print(f"  {i}게임: [{numbers_str}] → {match_indicator} {prize}")
        
        if main_matches > best_match:
            best_match = main_matches
            best_result = prize
    
    print()
    print("=" * 60)
    print(f"✨ 최고 결과: {best_match}개 일치 - {best_result}")
    print("=" * 60)
    
    return recommendations, actual_numbers, bonus_number


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LottoGoGo 백테스트")
    parser.add_argument("--csv", default="history.csv", help="CSV 파일 경로")
    parser.add_argument("--round", type=int, required=True, help="기준 회차 (이 회차까지 학습, 다음 회차 예측)")
    parser.add_argument("--games", type=int, default=5, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    
    args = parser.parse_args()
    backtest(csv_path=args.csv, target_round=args.round, num_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
