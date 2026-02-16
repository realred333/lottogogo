#!/usr/bin/env python3
"""Build static frontend model.json from history.csv and engine logic.

This script runs weekly in GitHub Actions after history.csv incremental updates.
It keeps model generation deterministic for the same history by using a stable seed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from lottogogo.engine.ranker import CombinationRanker
from lottogogo.engine.sampler import MonteCarloSampler
from lottogogo.engine.score import (
    BaseScoreCalculator,
    BoostCalculator,
    PenaltyCalculator,
    ProbabilityNormalizer,
    ScoreEnsembler,
)
from lottogogo.engine.score.hmm_scorer import HMMScorer
from lottogogo.tuning.feature_builder import FeatureBuilder
from lottogogo.tuning.xgb_ranker import XGBRanker
from lottogogo.mvp.service import (
    NUMBER_COLS,
    PRESET_CONFIGS,
    RARE_PAIR_COMBINATIONS,
    PresetConfig,
    contains_rare_pair,
    exceeds_carryover_limit,
)

TAG_LABELS = {
    "hmm_hot": "AI 흐름 강세",
    "hot": "최근 빈도 상승",
    "carryover": "직전 회차 연결",
    "carryover2": "최근 흐름 연장",
    "neighbor": "인접 패턴 확장",
    "reverse": "반전 대칭 후보",
    "hmm_cold": "잠복 구간 반등",
    "cold": "저출현 반등",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frontend model JSON from lotto history.")
    parser.add_argument("--history-csv", default="history.csv", help="Input history CSV path.")
    parser.add_argument("--weights", default="data/optimized_weights.json", help="Path to optimized weights JSON (optional).")
    parser.add_argument("--output-ga", default="data/model_ga.json", help="Output GA model JSON path.")
    parser.add_argument("--output-xgb", default="data/model_xgb.json", help="Output XGBoost model JSON path.")
    parser.add_argument("--output", default=None, help="Legacy single output (deprecated).")
    parser.add_argument("--chunk-size", type=int, default=20_000, help="Monte Carlo sampler chunk size.")
    parser.add_argument(
        "--base-weight",
        type=float,
        default=0.65,
        help="Blend weight for base probabilities vs filtered-frequency probabilities.",
    )
    parser.add_argument(
        "--sample-size-override",
        type=int,
        default=None,
        help="Optional override for Monte Carlo sample size (debug only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. If omitted, derived from latest round.",
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        help="Enable HMM scoring (slower but more accurate).",
    )
    return parser.parse_args()


def normalize_distribution(values: dict[int, float]) -> dict[int, float]:
    total = float(sum(max(0.0, float(value)) for value in values.values()))
    if total <= 0:
        uniform = 1.0 / 45.0
        return {number: uniform for number in range(1, 46)}
    return {number: max(0.0, float(values.get(number, 0.0))) / total for number in range(1, 46)}


def combo_key(numbers: tuple[int, ...] | list[int]) -> str:
    return "-".join(str(int(value)) for value in numbers)


def build_scores(history: pd.DataFrame, weights: dict[str, float] | None = None, use_hmm: bool = False) -> tuple[dict[int, float], dict[int, list[str]], set[int]]:
    if weights is None:
        weights = {}

    base_scores = BaseScoreCalculator(prior_alpha=1.0, prior_beta=1.0).calculate_scores(history, recent_n=50)

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
    boosts, boost_tags = booster.calculate_boosts(history)

    penalties = PenaltyCalculator(
        poisson_window=20, 
        poisson_lambda=weights.get("poisson_lambda", 0.0), 
        markov_lambda=weights.get("markov_lambda", 0.0)
    ).calculate_penalties(history)

    # HMM scoring (optional, slow)
    if use_hmm:
        hmm_boosts, hmm_tags = HMMScorer(
            hot_boost=weights.get("hmm_hot_boost", 0.3), 
            cold_boost=weights.get("hmm_cold_boost", 0.15), 
            window=100
        ).calculate_boosts(history)
    else:
        hmm_boosts, hmm_tags = {}, {}

    combined_boosts = {number: boosts.get(number, 0.0) + hmm_boosts.get(number, 0.0) for number in range(1, 46)}
    for number, tags in hmm_tags.items():
        if tags:
            merged = boost_tags.setdefault(number, [])
            merged.extend(tags)

    raw_scores = ScoreEnsembler(minimum_score=0.0).combine(base_scores, combined_boosts, penalties)

    deduped_tags: dict[int, list[str]] = {}
    for number in range(1, 46):
        deduped_tags[number] = sorted(set(boost_tags.get(number, [])))

    carryover_numbers = {
        number
        for number in range(1, 46)
        if "carryover" in deduped_tags.get(number, []) or "carryover2" in deduped_tags.get(number, [])
    }

    return raw_scores, deduped_tags, carryover_numbers


def apply_filters(
    combinations_in: list[tuple[int, ...]],
    historical_draws: list[tuple[int, ...]],
    carryover_numbers: set[int],
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
            HistoryFilter(historical_draws=historical_draws, match_threshold=5),
        ]
    )

    base_filtered = pipeline.filter_combinations(combinations_in)
    result: list[tuple[int, ...]] = []

    for combination in base_filtered:
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


def build_reasons(config: PresetConfig) -> list[str]:
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


def quantiles_from_scores(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}

    points = np.quantile(scores, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": float(points[0]),
        "p25": float(points[1]),
        "p50": float(points[2]),
        "p75": float(points[3]),
        "p90": float(points[4]),
    }


def build_history_index(draws: list[tuple[int, ...]]) -> tuple[list[str], list[str]]:
    exact = sorted({combo_key(draw) for draw in draws})

    subsets: set[str] = set()
    for draw in draws:
        for subset in combinations(draw, 5):
            subsets.add(combo_key(subset))

    return exact, sorted(subsets)


def preset_seed(base_seed: int, preset_name: str) -> int:
    offset = 11 if preset_name == "A" else 29
    return base_seed * 131 + offset


def calculate_probabilities_xgb(history: pd.DataFrame, weights: dict[str, float] | None = None) -> dict[int, float]:
    """Calculate probabilities using XGBoost model."""
    import xgboost as xgb
    
    latest_round = int(history["round"].max())
    
    # Build features and train XGBoost
    builder = FeatureBuilder(history, weights=weights)
    
    # Train on all historical data
    X_train, y_train = builder.build((1, latest_round))
    spw = FeatureBuilder.scale_pos_weight(y_train)
    
    # Create and train XGBoost model directly
    model = xgb.XGBClassifier(
        scale_pos_weight=spw,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    
    # Predict probabilities for next round
    next_round = latest_round + 1
    features = builder._extract_features(history, next_round, weights=weights)
    probs_raw = model.predict_proba(features)[:, 1]
    
    # Convert to dict
    return {n: float(probs_raw[n-1]) for n in range(1, 46)}


def main() -> int:
    args = parse_args()

    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if not (0.0 <= args.base_weight <= 1.0):
        raise SystemExit("--base-weight must be between 0 and 1")
    if args.sample_size_override is not None and args.sample_size_override <= 0:
        raise SystemExit("--sample-size-override must be > 0")

    history_path = Path(args.history_csv)
    if not history_path.exists():
        raise SystemExit(f"history csv not found: {history_path}")

    loader = LottoHistoryLoader()
    history = loader.load_and_validate(history_path).reset_index(drop=True)
    if history.empty:
        raise SystemExit("history csv is empty")

    historical_draws = [tuple(int(row[column]) for column in NUMBER_COLS) for _, row in history.iterrows()]
    latest_round = int(history["round"].max())
    base_seed = int(args.seed) if args.seed is not None else latest_round

    # Load optimized weights if they exist
    weights: dict[str, float] = {}
    weights_path = Path(args.weights)
    if weights_path.exists():
        print(f"📂 Loading optimized weights from {weights_path}")
        try:
            weights_data = json.loads(weights_path.read_text(encoding="utf-8"))
            weights = weights_data.get("weights", {})
            print(f"   - Cycle: {weights_data.get('cycle_label', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Failed to load weights: {e}")
    else:
        print(f"ℹ️ No optimized weights found at {weights_path}, using defaults.")

    raw_scores, boost_tags, carryover_numbers = build_scores(history, weights=weights, use_hmm=args.use_hmm)

    model_presets: dict[str, dict[str, object]] = {}

    for preset_name, config in PRESET_CONFIGS.items():
        sample_size = args.sample_size_override or config.sample_size

        base_prob_map = ProbabilityNormalizer.to_sampling_probabilities(
            raw_scores,
            temperature=weights.get("temperature", config.temperature),
            min_prob_floor=config.min_prob_floor,
        )
        base_prob_map = normalize_distribution(base_prob_map)

        sampler = MonteCarloSampler(sample_size=sample_size, chunk_size=args.chunk_size)
        seed = preset_seed(base_seed, preset_name)

        sampled = sampler.sample(base_prob_map, seed=seed)
        filtered = apply_filters(sampled, historical_draws, carryover_numbers, config)
        if not filtered:
            filtered = [tuple(sorted(int(value) for value in combo)) for combo in sampled]

        number_counts = Counter(number for combo in filtered for number in combo)
        total_hits = sum(number_counts.values())
        if total_hits <= 0:
            filtered_prob_map = {number: 1.0 / 45.0 for number in range(1, 46)}
        else:
            filtered_prob_map = {number: number_counts.get(number, 0) / total_hits for number in range(1, 46)}

        blended_prob_map = {
            number: args.base_weight * base_prob_map.get(number, 0.0)
            + (1.0 - args.base_weight) * filtered_prob_map.get(number, 0.0)
            for number in range(1, 46)
        }
        blended_prob_map = normalize_distribution(blended_prob_map)

        score_values = np.array(
            [sum(raw_scores.get(number, 0.0) for number in combo) for combo in filtered],
            dtype=np.float64,
        )
        score_q = quantiles_from_scores(score_values)

        acceptance_rate = float(len(filtered)) / float(sample_size)
        estimated_attempts = int(max(8_000, min(60_000, round((max(config.top_k * 3, 800) / max(acceptance_rate, 0.02))))))

        top_numbers = [
            number
            for number, _ in sorted(
                ((number, blended_prob_map[number]) for number in range(1, 46)),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        ]

        model_presets[preset_name] = {
            "filters": {
                "min_sum": config.min_sum,
                "max_sum": config.max_sum,
                "min_ac": config.min_ac,
                "max_per_zone": config.max_per_zone,
                "max_same_tail": config.max_same_tail,
                "min_odd": config.min_odd,
                "max_odd": config.max_odd,
                "min_high": config.min_high,
                "max_high": config.max_high,
                "history_match_threshold": 5,
            },
            "special": {
                "rare_pair_filter": config.rare_pair_filter,
                "excluded_numbers": sorted(int(value) for value in config.excluded_numbers),
                "max_carryover_in_combo": config.max_carryover_in_combo,
            },
            "ranking": {
                "top_k": config.top_k,
                "max_overlap": config.max_overlap,
                "percentile_bias": config.percentile_bias,
            },
            "sampling": {
                "sample_size": int(sample_size),
                "chunk_size": int(args.chunk_size),
                "max_attempts": estimated_attempts,
                "blend_base_weight": float(args.base_weight),
                "weights": [float(blended_prob_map[number]) for number in range(1, 46)],
                "base_probabilities": [float(base_prob_map[number]) for number in range(1, 46)],
            },
            "reasons": build_reasons(config),
            "monte_carlo": {
                "seed": seed,
                "sampled_count": int(sample_size),
                "filtered_count": len(filtered),
                "acceptance_rate": round(acceptance_rate, 6),
                "score_quantiles": {k: round(v, 6) for k, v in score_q.items()},
                "top_numbers": top_numbers,
            },
        }

    exact_keys, five_subset_keys = build_history_index(historical_draws)

    model = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "history_csv": str(history_path.as_posix()),
            "rows": int(len(history)),
            "latest_round": latest_round,
        },
        "tag_labels": TAG_LABELS,
        "numbers": list(range(1, 46)),
        "raw_scores": [float(raw_scores.get(number, 0.0)) for number in range(1, 46)],
        "boost_tags_by_number": [boost_tags.get(number, []) for number in range(1, 46)],
        "signals": {
            "hot_numbers": [number for number in range(1, 46) if "hot" in boost_tags.get(number, [])],
            "cold_numbers": [number for number in range(1, 46) if "cold" in boost_tags.get(number, [])],
            "hmm_hot_numbers": [number for number in range(1, 46) if "hmm_hot" in boost_tags.get(number, [])],
            "hmm_cold_numbers": [number for number in range(1, 46) if "hmm_cold" in boost_tags.get(number, [])],
        },
        "carryover_numbers": sorted(carryover_numbers),
        "rare_pairs": [[left, right] for left, right in sorted(RARE_PAIR_COMBINATIONS)],
        "history": {
            "draws": [list(draw) for draw in historical_draws],
            "exact_keys": exact_keys,
            "five_subset_keys": five_subset_keys,
            "match_threshold": 5,
        },
        "presets": model_presets,
    }

    # Save GA model
    output_ga = Path(args.output_ga)
    output_ga.parent.mkdir(parents=True, exist_ok=True)
    output_ga.write_text(
        json.dumps(model, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    print(f"✅ GA model saved: {output_ga}")
    
    # Generate and save XGBoost model
    print("\n🧠 Generating XGBoost model...")
    try:
        xgb_probs = calculate_probabilities_xgb(history, weights=weights)
        xgb_probs_normalized = normalize_distribution(xgb_probs)
        
        # Create XGBoost model with same structure but different probabilities
        xgb_model = model.copy()
        xgb_model["model_type"] = "xgboost"
        
        # Update sampling weights in all presets
        for preset_name in xgb_model["presets"]:
            xgb_model["presets"][preset_name]["sampling"]["weights"] = [
                float(xgb_probs_normalized[n]) for n in range(1, 46)
            ]
            xgb_model["presets"][preset_name]["sampling"]["base_probabilities"] = [
                float(xgb_probs_normalized[n]) for n in range(1, 46)
            ]
        
        output_xgb = Path(args.output_xgb)
        output_xgb.parent.mkdir(parents=True, exist_ok=True)
        output_xgb.write_text(
            json.dumps(xgb_model, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        print(f"✅ XGBoost model saved: {output_xgb}")
    except Exception as e:
        print(f"⚠️ XGBoost model generation failed: {e}")
        print("   Continuing with GA model only...")
    
    # Legacy single output support
    if args.output:
        legacy_path = Path(args.output)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps(model, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        print(f"✅ Legacy model saved: {legacy_path}")

    print(f"\nhistory rows      : {len(history)}")
    print(f"latest round      : {latest_round}")
    print(f"history exact keys: {len(exact_keys)}")
    print(f"history 5-subsets : {len(five_subset_keys)}")
    for preset_name, preset_data in model_presets.items():
        mc = preset_data["monte_carlo"]
        print(
            f"preset {preset_name} -> sampled={mc['sampled_count']} "
            f"filtered={mc['filtered_count']} acceptance={mc['acceptance_rate']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
