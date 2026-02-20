#!/usr/bin/env python3
"""LottoGoGo ML - XGBoost 기반 로또 번호 추천기"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.tuning.xgb_ranker import XGBRanker
from lottogogo.tuning.feature_builder import FeatureBuilder
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
    (8, 12), (8, 26), (24, 43), (26, 32), (1, 22), (3, 25),
    (4, 30), (6, 33), (8, 9), (11, 40), (23, 41), (6, 23),
    (6, 29), (7, 32), (9, 20), (11, 34), (16, 22), (19, 29),
    (24, 28), (25, 41), (29, 30), (37, 44),
]

EXCLUDE_NUMBERS = {8}
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


def save_model(model, latest_round: int, weights_path: str | None = None, model_dir: str = "data") -> None:
    """XGBoost 모델을 저장하고 메타데이터 기록"""
    import pickle
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장 (pickle 사용)
    model_file = model_path / "xgb_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    # 메타데이터 저장 (가중치 경로 포함)
    metadata = {
        "trained_until_round": latest_round,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": str(model_file),
        "weights_path": weights_path,
    }
    metadata_file = model_path / "xgb_model_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def load_model_if_valid(latest_round: int, weights_path: str | None = None, model_dir: str = "data"):
    """저장된 모델이 유효하면 로드, 아니면 None 반환
    
    Args:
        latest_round: 최신 회차
        weights_path: 가중치 파일 경로 (None이면 기본 가중치)
        model_dir: 모델 저장 디렉토리
    """
    import pickle
    
    model_path = Path(model_dir)
    metadata_file = model_path / "xgb_model_metadata.json"
    model_file = model_path / "xgb_model.pkl"
    
    # 메타데이터 파일이 없으면 None
    if not metadata_file.exists() or not model_file.exists():
        return None
    
    try:
        metadata = json.loads(metadata_file.read_text())
        trained_round = metadata.get("trained_until_round", 0)
        trained_weights = metadata.get("weights_path", None)
        
        # 저장된 모델이 최신 데이터로 학습되었고, 같은 가중치를 사용했는지 확인
        if trained_round >= latest_round and trained_weights == weights_path:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            return model, metadata
        else:
            return None
    except Exception:
        return None


def main(csv_path: str = "history.csv", num_games: int = 5, seed: int | None = None, weights_path: str | None = None):
    """XGBoost 기반 로또 번호 추천 실행"""
    
    # 시드 설정
    if seed is None:
        seed = int(datetime.now().timestamp()) % 100000
    
    print("=" * 60)
    print("🤖 LottoGoGo ML - XGBoost 기반 로또 번호 추천기")
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

    # 2. XGBoost 모델 학습 또는 로드
    print("\n🧠 XGBoost 모델 준비 중...")
    
    # 저장된 모델 확인 (가중치 경로 포함)
    cached = load_model_if_valid(latest_round, weights_path)
    
    if cached is not None:
        model, metadata = cached
        trained_round = metadata["trained_until_round"]
        trained_at = metadata["trained_at"]
        print(f"   ✅ 저장된 모델 로드 (학습 회차: {trained_round}회, 학습 시각: {trained_at})")
        print(f"   ⚡ 재학습 생략 (최신 데이터와 일치)")
    else:
        # 모델이 없거나 오래되었으면 새로 학습
        if Path("data/xgb_model_metadata.json").exists():
            old_metadata = json.loads(Path("data/xgb_model_metadata.json").read_text())
            old_round = old_metadata.get("trained_until_round", 0)
            print(f"   🔄 새로운 회차 감지 ({old_round}회 → {latest_round}회)")
        print(f"   🧠 모델 학습 중... (전체 {latest_round}회차 데이터)")
        
        builder = FeatureBuilder(history, weights=weights if weights else None)
        ranker = XGBRanker(builder)
        
        # 전체 데이터로 학습 (가중치 적용)
        X_train, y_train = builder.build((1, latest_round))
        
        spw = FeatureBuilder.scale_pos_weight(y_train)
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": spw,
            "seed": seed,
            "verbosity": 0,
        }
        
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=xgb_params.pop("n_estimators"),
            random_state=xgb_params.pop("seed"),
            **xgb_params,
        )
        model.fit(X_train, y_train)
        
        # 모델 저장 (가중치 경로 포함)
        save_model(model, latest_round, weights_path)
        print(f"   ✅ 모델 학습 완료 및 저장 (data/xgb_model.pkl)")

    # 3. 다음 회차 예측
    print(f"\n🔮 {latest_round + 1}회 번호 확률 예측 중...")
    next_round = latest_round + 1
    
    # FeatureBuilder 인스턴스 생성 (캐시된 모델 사용 시에도 필요)
    if cached is not None:
        builder = FeatureBuilder(history, weights=weights if weights else None)
    
    # 특징 추출 (가중치 적용)
    features = builder._extract_features(history, next_round, weights=weights if weights else None)
    probs_raw = model.predict_proba(features)[:, 1]
    
    # 확률을 딕셔너리로 변환
    probs = {n: float(probs_raw[n - 1]) for n in range(1, 46)}
    
    # 확률 정규화 (샘플링용)
    probs_normalized = ProbabilityNormalizer.to_sampling_probabilities(
        probs, 
        temperature=0.5, 
        min_prob_floor=0.005
    )
    
    # 상위 번호 출력
    sorted_numbers = sorted(range(1, 46), key=lambda n: probs[n], reverse=True)
    print(f"\n📊 예측 확률 상위 15개:")
    print(f"   {'번호':>4} | {'예측 확률':>8} | {'샘플링 확률':>10}")
    print(f"   {'-'*4}-+-{'-'*8}-+-{'-'*10}")
    for num in sorted_numbers[:15]:
        print(f"   {num:4d} | {probs[num]:8.4f} | {probs_normalized[num]:9.2%}")

    # Carryover 번호 추출 (직전 + 2주전)
    last_round = history[history["round"] == latest_round].iloc[0]
    carryover_numbers = set(int(last_round[col]) for col in NUMBER_COLS)
    
    if len(history) >= 2:
        second_last_round = history[history["round"] == latest_round - 1].iloc[0]
        carryover2_numbers = set(int(second_last_round[col]) for col in NUMBER_COLS)
    else:
        carryover2_numbers = set()
    
    all_carryover_numbers = carryover_numbers | carryover2_numbers
    print(f"\n   🔄 Carryover (직전): {sorted(carryover_numbers)}")
    print(f"   🔄 Carryover2 (2주전): {sorted(carryover2_numbers)}")
    print(f"   → 이월수(직전+2주전) 최대 {MAX_CARRYOVER_IN_COMBO}개까지 허용")

    # 4. 조합 생성
    print(f"\n🎲 조합 생성 중 (100,000개)...")
    sampler = MonteCarloSampler(sample_size=100000, chunk_size=20000)
    combinations = sampler.sample(probs_normalized, seed=seed)

    # 5. 필터링
    print("🔍 필터링 중...")
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
    pass_rate = len(filtered) / len(combinations) * 100
    print(f"   기본 필터 통과: {len(filtered):,}개 ({pass_rate:.1f}%)")

    # 🎰 도박사의 오류 필터링
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

    # 6. 랭킹 및 다양성 적용
    print("🏆 랭킹 및 다양성 적용...")
    
    # 중복 제거
    unique_filtered = list(dict.fromkeys(filtered))
    if len(unique_filtered) != len(filtered):
        print(f"   (중복 제거) {len(filtered):,} -> {len(unique_filtered):,}개")
    filtered = unique_filtered

    ranker_obj = CombinationRanker()
    ranked = ranker_obj.rank(filtered, probs)
    candidates = [r.numbers for r in ranked]

    base_overlap = 3
    selector = DiversitySelector(max_overlap=base_overlap)
    final = selector.select(candidates, output_count=num_games)

    # 다양성 조건 완화
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
        avg_prob = np.mean([probs[n] for n in combo])
        print(f"\n  {i}게임: [{numbers_str}]")
        print(f"         합계: {total} | 홀수: {sum(1 for n in combo if n % 2)}개 | 평균 예측확률: {avg_prob:.4f}")
    
    print("\n" + "=" * 60)
    print("🤖 XGBoost 추천 완료! 행운을 빕니다!")
    print("=" * 60)
    
    return final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LottoGoGo ML - XGBoost 번호 추천기")
    parser.add_argument("--csv", default="history.csv", help="CSV 파일 경로")
    parser.add_argument("--games", type=int, default=5, help="추천 게임 수")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--weights", default=None, help="최적화 가중치 JSON 경로")
    
    args = parser.parse_args()
    main(csv_path=args.csv, num_games=args.games, seed=args.seed, weights_path=args.weights)
