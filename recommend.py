#!/usr/bin/env python3
"""LottoGoGo - 로또 번호 추천기"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.engine.score.calculator import BaseScoreCalculator, ScoreEnsembler
from lottogogo.engine.score.booster import BoostCalculator
from lottogogo.engine.score.penalizer import PenaltyCalculator
from lottogogo.engine.score.hmm_scorer import HMMScorer
from lottogogo.engine.score.normalizer import ProbabilityNormalizer
from lottogogo.engine.sampler.monte_carlo import MonteCarloSampler
from lottogogo.engine.filters import (
    ArithmeticProgressionFilter,
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
# 🎰 도박사의 오류 필터링 설정
# ============================================================
# 역대 가장 적게 나온 두 숫자 조합 (7~10회 출현)
RARE_PAIR_COMBINATIONS = [
    (8, 12),   # 7회 ⭐ 최저
    (8, 26),   # 8회
    (24, 43),  # 8회
    (26, 32),  # 8회
    (1, 22),   # 9회
    (3, 25),   # 9회
    (4, 30),   # 9회
    (6, 33),   # 9회
    (8, 9),    # 9회
    (11, 40),  # 9회
    (23, 41),  # 9회
    (6, 23),   # 10회
    (6, 29),   # 10회
    (7, 32),   # 10회
    (9, 20),   # 10회
    (11, 34),  # 10회
    (16, 22),  # 10회
    (19, 29),  # 10회
    (24, 28),  # 10회
    (25, 41),  # 10회
    (29, 30),  # 10회
    (37, 44),  # 10회
]

# 제외할 숫자 (8은 희귀 조합에 많이 포함됨)
EXCLUDE_NUMBERS = {8}

# 이월수(carryover) 제한: 조합에 포함될 최대 이월수 개수 (직전 + 2주전 합계)
MAX_CARRYOVER_IN_COMBO = 2


def contains_rare_pair(combination: tuple[int, ...]) -> bool:
    """조합이 희귀 쌍을 포함하는지 확인"""
    nums = set(combination)
    for a, b in RARE_PAIR_COMBINATIONS:
        if a in nums and b in nums:
            return True
    return False


def contains_excluded_number(combination: tuple[int, ...]) -> bool:
    """조합이 제외 숫자를 포함하는지 확인"""
    return bool(set(combination) & EXCLUDE_NUMBERS)


def exceeds_carryover_limit(combination: tuple[int, ...], carryover_numbers: set[int]) -> bool:
    """조합이 이월수 제한을 초과하는지 확인"""
    carryover_count = len(set(combination) & carryover_numbers)
    return carryover_count > MAX_CARRYOVER_IN_COMBO


def main(csv_path: str = "history.csv", num_games: int = 5, seed: int | None = None, weights_path: str | None = None):
    """로또 번호 추천 실행"""
    
    # 시드 설정 (없으면 현재 시간 기반)
    if seed is None:
        seed = int(datetime.now().timestamp()) % 100000
    
    print("=" * 60)
    print("🎰 LottoGoGo - 로또 번호 추천기")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎲 Seed: {seed}")
    print()

    # 1. 데이터 로드
    loader = LottoHistoryLoader()
    try:
        history = loader.load_and_validate(csv_path).reset_index(drop=True)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    
    latest_round = int(history["round"].max())
    print(f"📊 데이터: {len(history)}회차 (최신: {latest_round}회)")

    # 1.1 최적화 가중치 로드
    weights = {}
    if weights_path:
        w_path = Path(weights_path)
        if w_path.exists():
            print(f"📂 가중치 로드: {weights_path}")
            try:
                weights_data = json.loads(w_path.read_text(encoding="utf-8"))
                weights = weights_data.get("weights", {})
                print(f"   - Cycle: {weights_data.get('cycle_label', 'unknown')}")
            except Exception as e:
                print(f"⚠️ 가중치 로드 실패 (기본값 사용): {e}")
        else:
            print(f"⚠️ 가중치 파일을 찾을 수 없습니다: {weights_path} (기본값 사용)")

    # 과거 당첨번호 추출
    historical_draws = [
        tuple(row[NUMBER_COLS].astype(int).tolist()) 
        for _, row in history.iterrows()
    ]

    # 2. 점수 계산
    print("\n🧮 점수 계산 중...")
    base_calc = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0)
    
    # 가중치 주입 (있을 경우)
    booster = BoostCalculator(
        hot_threshold=2, 
        hot_window=5, 
        cold_window=10,
        hot_weight=weights.get("hot_weight", 0.4),
        cold_weight=weights.get("cold_weight", 0.15),
        neighbor_weight=weights.get("neighbor_weight", 0.3),
        carryover_weight=weights.get("carryover_weight", 0.4),
        reverse_weight=weights.get("reverse_weight", 0.1)
    )
    
    penalizer = PenaltyCalculator(
        poisson_window=20, 
        poisson_lambda=weights.get("poisson_lambda", 0.0), 
        markov_lambda=weights.get("markov_lambda", 0.0)
    )
    
    ensembler = ScoreEnsembler(minimum_score=0.0)

    base_scores = base_calc.calculate_scores(history, recent_n=50)
    boosts, boost_tags = booster.calculate_boosts(history)
    penalties = penalizer.calculate_penalties(history)
    
    # HMM scoring
    hmm_scorer = HMMScorer(
        hot_boost=weights.get("hmm_hot_boost", 0.3), 
        cold_boost=weights.get("hmm_cold_boost", 0.15), 
        window=100
    )
    hmm_boosts, hmm_tags = hmm_scorer.calculate_boosts(history)
    hmm_summary = hmm_scorer.get_summary()
    
    # Combine boosts (regular + HMM)
    combined_boosts = {n: boosts.get(n, 0) + hmm_boosts.get(n, 0) for n in range(1, 46)}
    for n, tags in hmm_tags.items():
        if tags:
            boost_tags.setdefault(n, []).extend(tags)
    
    raw_scores = ensembler.combine(base_scores, combined_boosts, penalties)

    # Hot/Cold/Carryover 번호
    hot_numbers = [n for n in range(1, 46) if "hot" in boost_tags.get(n, [])]
    cold_numbers = [n for n in range(1, 46) if "cold" in boost_tags.get(n, [])]
    carryover_numbers = set(n for n in range(1, 46) if "carryover" in boost_tags.get(n, []))
    carryover2_numbers = set(n for n in range(1, 46) if "carryover2" in boost_tags.get(n, []))
    
    print(f"   🔥 Hot: {hot_numbers}")
    print(f"   ❄️  Cold: {cold_numbers}")
    print(f"   🔄 Carryover (직전): {sorted(carryover_numbers)}")
    print(f"   🔄 Carryover2 (2주전): {sorted(carryover2_numbers)}")
    print(f"   → 이월수(직전+2주전) 최대 {MAX_CARRYOVER_IN_COMBO}개까지 허용")
    print(f"   🧠 HMM Hot: {hmm_summary['hot'][:10]}{'...' if len(hmm_summary['hot']) > 10 else ''}")
    print(f"   🧠 HMM Cold: {hmm_summary['cold'][:10]}{'...' if len(hmm_summary['cold']) > 10 else ''}")
    
    # 이월수 필터용: 직전 + 2주전 합치기
    all_carryover_numbers = carryover_numbers | carryover2_numbers

    # 상위 확률 번호
    probs = ProbabilityNormalizer.to_sampling_probabilities(
        raw_scores,
        temperature=weights.get("temperature", 0.5),
        min_prob_floor=0.005
    )

    # 40~45 번호 확률 하향 (정규화 후 적용 → 진짜 절반)
    HIGH_ZONE_PENALTY = 0.5
    for num in range(40, 46):
        probs[num] *= HIGH_ZONE_PENALTY
    total_prob = sum(probs.values())
    probs = {n: p / total_prob for n, p in probs.items()}
    
    # Poisson/Markov 개별 페널티 계산
    poisson_penalties = penalizer.calculate_poisson_penalty(history)
    markov_penalties = penalizer.calculate_markov_penalty(history)
    
    # 번호별 점수 상세 출력 (세부 컴포넌트별)
    print(f"\n📊 번호별 점수 상세 (전체 45개):")
    print(f"   {'번호':>4} | {'Base':>5} | {'Hot':>4} | {'Cold':>4} | {'Nbr':>4} | {'Cry':>4} | {'Rev':>4} | {'Poi':>5} | {'Mrk':>5} | {'총점':>5} | {'확률':>5}")
    print(f"   {'-'*4}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    
    sorted_numbers = sorted(range(1, 46), key=lambda n: raw_scores.get(n, 0), reverse=True)
    for num in sorted_numbers:
        tags = boost_tags.get(num, [])
        base = base_scores.get(num, 0)
        hot = booster.hot_weight if "hot" in tags else 0
        cold = booster.cold_weight if "cold" in tags else 0
        neighbor = booster.neighbor_weight if "neighbor" in tags else 0
        carryover = booster.carryover_weight if "carryover" in tags else 0
        reverse = booster.reverse_weight if "reverse" in tags else 0
        poisson = poisson_penalties.get(num, 0)
        markov = markov_penalties.get(num, 0)
        total = raw_scores.get(num, 0)
        prob = probs.get(num, 0)
        print(f"   {num:4d} | {base:5.2f} | {hot:4.2f} | {cold:4.2f} | {neighbor:4.2f} | {carryover:4.2f} | {reverse:4.2f} | {poisson:5.2f} | {markov:5.2f} | {total:5.2f} | {prob:4.1%}")

    # 3. 조합 생성
    print(f"\n🎲 조합 생성 중 (100,000개)...")
    sampler = MonteCarloSampler(sample_size=100000, chunk_size=20000)
    combinations = sampler.sample(probs, seed=seed)

    # 4. 필터링
    print("🔍 필터링 중...")
    pipeline = FilterPipeline([
        SumFilter(min_sum=100, max_sum=175),
        ACFilter(min_ac=7),
        ZoneFilter(max_per_zone=3),
        TailFilter(max_same_tail=2),
        OddEvenFilter(min_odd=2, max_odd=4),
        HighLowFilter(min_high=2, max_high=4),
        HistoryFilter(historical_draws=historical_draws, match_threshold=5),
        ArithmeticProgressionFilter(min_ap_length=5),
    ])
    filtered = pipeline.filter_combinations(combinations)
    pass_rate = len(filtered) / len(combinations) * 100
    print(f"   기본 필터 통과: {len(filtered):,}개 ({pass_rate:.1f}%)")

    # 🎰 도박사의 오류 필터링 (희귀 조합 제외 + 숫자 8 제외 + 이월수 제한)
    print("\n🎰 도박사의 오류 필터링...")
    before_gambler = len(filtered)
    filtered = [
        combo for combo in filtered
        if not contains_rare_pair(combo) 
        and not contains_excluded_number(combo)
        and not exceeds_carryover_limit(combo, all_carryover_numbers)
    ]
    after_gambler = len(filtered)
    removed = before_gambler - after_gambler
    print(f"   제외된 조합: {removed:,}개")
    print(f"   - 희귀 쌍 조합: {len(RARE_PAIR_COMBINATIONS)}개 패턴 제외")
    print(f"   - 제외 숫자: {EXCLUDE_NUMBERS}")
    print(f"   - 이월수 제한: 최대 {MAX_CARRYOVER_IN_COMBO}개 (직전+2주전 합계)")
    print(f"   최종 통과: {len(filtered):,}개")

    # 5. 랭킹 및 다양성 적용
    print("🏆 랭킹 및 다양성 적용...")

    # 샘플링/필터링 과정에서 동일 조합이 다수 포함될 수 있어(top_k가 중복으로 채워짐)
    # 다양성 선택에서 게임 수가 부족해질 수 있다. 랭킹 전 중복 제거로 후보 폭을 확보한다.
    unique_filtered = list(dict.fromkeys(filtered))
    if len(unique_filtered) != len(filtered):
        print(f"   (중복 제거) {len(filtered):,} -> {len(unique_filtered):,}개")
    filtered = unique_filtered

    ranker = CombinationRanker()
    ranked = ranker.rank(filtered, raw_scores)
    candidates = [r.numbers for r in ranked]

    base_overlap = 3
    selector = DiversitySelector(max_overlap=base_overlap)
    final = selector.select(candidates, output_count=num_games)

    # 다양성 조건이 너무 빡빡해 게임 수가 부족하면 점진적으로 완화해 채운다.
    if len(final) < num_games:
        print(f"   ⚠️ 다양성 조건(max_overlap={base_overlap})으로 {len(final)}/{num_games}개만 선택됨. 조건을 완화합니다.")
        for overlap in range(base_overlap + 1, 7):
            final = DiversitySelector(max_overlap=overlap).select(candidates, output_count=num_games)
            if len(final) >= num_games:
                print(f"   -> max_overlap={overlap}로 {len(final)}/{num_games}개 선택")
                break

    # 결과 출력
    print("\n" + "=" * 60)
    if len(final) < num_games:
        print(f"⚠️ 요청 {num_games}게임 중 {len(final)}게임만 생성되었습니다. (후보 부족/조건 과다)")
    print(f"🎯 {latest_round + 1}회 추천 번호 ({len(final)}/{num_games}게임)")
    print("=" * 60)
    
    for i, combo in enumerate(final, 1):
        numbers_str = ", ".join(f"{n:2d}" for n in combo)
        total = sum(combo)
        print(f"\n  {i}게임: [{numbers_str}]")
        print(f"         합계: {total} | 홀수: {sum(1 for n in combo if n % 2)}개")
    
    print("\n" + "=" * 60)
    print("🍀 행운을 빕니다!")
    print("=" * 60)
    
    return final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LottoGoGo 번호 추천기")
    parser.add_argument("--csv", default="history.csv", help="CSV 파일 경로")
    parser.add_argument("--games", type=int, default=5, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--weights", default=None, help="최적화 가중치 JSON 경로")
    
    args = parser.parse_args()
    main(csv_path=args.csv, num_games=args.games, seed=args.seed, weights_path=args.weights)
