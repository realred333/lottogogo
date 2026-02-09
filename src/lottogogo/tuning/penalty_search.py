"""Parallel grid search for poisson/markov penalty lambdas."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lottogogo.data import LottoHistoryLoader
from lottogogo.engine.backtester import (
    RandomBaselineGenerator,
    WalkForwardBacktester,
    summarize_results,
)
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
from lottogogo.engine.ranker import CombinationRanker, DiversitySelector
from lottogogo.engine.sampler import MonteCarloSampler
from lottogogo.engine.score import (
    BaseScoreCalculator,
    BoostCalculator,
    PenaltyCalculator,
    ProbabilityNormalizer,
    ScoreEnsembler,
)

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
MAX_PENALTY_LAMBDA = 0.5


@dataclass(frozen=True)
class PenaltyTuneConfig:
    """Configuration for penalty lambda grid search."""

    history_path: str = "history.csv"
    start_round: int | None = None
    end_round: int | None = None
    recent_n: int = 50
    output_count: int = 5
    sample_size: int = 3000
    rank_top_k: int = 300
    seed: int = 42
    workers: int = max(1, (os.cpu_count() or 2) - 1)

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    poisson_window: int = 20
    min_prob_floor: float = 0.001
    temperature: float = 1.0
    max_overlap: int = 3

    min_sum: int = 100
    max_sum: int = 175
    min_ac: int = 7
    max_per_zone: int = 3
    max_same_tail: int = 2
    min_odd: int = 2
    max_odd: int = 4
    min_high: int = 2
    max_high: int = 4
    high_start: int = 23
    history_match_threshold: int = 5

    hot_threshold: int = 2
    hot_window: int = 5
    hot_weight: float = 0.4
    cold_window: int = 10
    cold_weight: float = 0.15
    neighbor_weight: float = 0.3
    carryover_weight: float = 0.2
    reverse_weight: float = 0.1


@dataclass(frozen=True)
class PenaltyTuneResult:
    """One grid point result."""

    poisson_lambda: float
    markov_lambda: float
    p_match_ge_3: float
    p_match_ge_4: float
    average_match_count: float
    std_match_count: float
    total_rounds: int
    total_recommendations: int

    def objective_key(self) -> tuple[float, float, float, float]:
        """Primary objective tuple for sorting (higher is better)."""
        return (
            self.p_match_ge_3,
            self.p_match_ge_4,
            self.average_match_count,
            -self.std_match_count,
        )


@dataclass(frozen=True)
class _WorkerInput:
    history: pd.DataFrame
    config: PenaltyTuneConfig
    poisson_lambda: float
    markov_lambda: float


def build_float_grid(start: float, end: float, step: float) -> list[float]:
    """Build an inclusive float grid."""
    if step <= 0:
        raise ValueError("step must be > 0.")
    if end < start:
        raise ValueError("end must be >= start.")
    values = np.arange(start, end + (step * 0.5), step, dtype=np.float64)
    return [float(round(value, 6)) for value in values]


def _validate_lambda_bounds(values: list[float], label: str) -> None:
    if not values:
        raise ValueError(f"{label} grid cannot be empty.")
    if min(values) < 0.0 or max(values) > MAX_PENALTY_LAMBDA:
        raise ValueError(f"{label} values must be within 0.0~{MAX_PENALTY_LAMBDA}.")


def run_penalty_grid_search(
    config: PenaltyTuneConfig,
    *,
    poisson_values: list[float],
    markov_values: list[float],
) -> list[PenaltyTuneResult]:
    """Evaluate all lambda pairs and return sorted results."""
    history = LottoHistoryLoader().load_and_validate(config.history_path)
    tasks = [
        _WorkerInput(
            history=history,
            config=config,
            poisson_lambda=poisson_lambda,
            markov_lambda=markov_lambda,
        )
        for poisson_lambda in poisson_values
        for markov_lambda in markov_values
    ]

    if config.workers <= 1:
        results = [_evaluate_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            results = list(executor.map(_evaluate_worker, tasks))

    results.sort(key=lambda result: result.objective_key(), reverse=True)
    return results


def _evaluate_worker(worker_input: _WorkerInput) -> PenaltyTuneResult:
    config = worker_input.config
    poisson_lambda = worker_input.poisson_lambda
    markov_lambda = worker_input.markov_lambda
    history = worker_input.history

    recommender = _build_recommender(
        config=config,
        poisson_lambda=poisson_lambda,
        markov_lambda=markov_lambda,
    )

    results = WalkForwardBacktester().run(
        history=history,
        recommender=recommender,
        start_round=config.start_round,
        end_round=config.end_round,
        recent_n=None,
        output_count=config.output_count,
        seed=config.seed,
    )
    summary = summarize_results(results)
    return PenaltyTuneResult(
        poisson_lambda=poisson_lambda,
        markov_lambda=markov_lambda,
        p_match_ge_3=float(summary["p_match_ge_3"]),
        p_match_ge_4=float(summary["p_match_ge_4"]),
        average_match_count=float(summary["average_match_count"]),
        std_match_count=float(summary["std_match_count"]),
        total_rounds=int(summary["total_rounds"]),
        total_recommendations=int(summary["total_recommendations"]),
    )


def _build_recommender(
    *,
    config: PenaltyTuneConfig,
    poisson_lambda: float,
    markov_lambda: float,
):
    base_calculator = BaseScoreCalculator(config.prior_alpha, config.prior_beta)
    boost_calculator = BoostCalculator(
        hot_threshold=config.hot_threshold,
        hot_window=config.hot_window,
        hot_weight=config.hot_weight,
        cold_window=config.cold_window,
        cold_weight=config.cold_weight,
        neighbor_weight=config.neighbor_weight,
        carryover_weight=config.carryover_weight,
        reverse_weight=config.reverse_weight,
    )
    penalty_calculator = PenaltyCalculator(
        poisson_window=config.poisson_window,
        poisson_lambda=poisson_lambda,
        markov_lambda=markov_lambda,
    )
    sampler = MonteCarloSampler(sample_size=config.sample_size)
    ranker = CombinationRanker()
    diversity_selector = DiversitySelector(max_overlap=config.max_overlap)
    ensembler = ScoreEnsembler(minimum_score=0.0)

    def recommender(train_df: pd.DataFrame, output_count: int, seed: int | None):
        full_history_df = train_df
        score_history_df = (
            full_history_df.tail(config.recent_n)
            if config.recent_n is not None and config.recent_n > 0
            else full_history_df
        )

        base_scores = base_calculator.calculate_scores(score_history_df)
        boost_scores, _ = boost_calculator.calculate_boosts(score_history_df)
        penalties = penalty_calculator.calculate_penalties(score_history_df)
        raw_scores = ensembler.combine(
            base_scores=base_scores,
            boost_scores=boost_scores,
            penalties=penalties,
        )
        probabilities = ProbabilityNormalizer.to_sampling_probabilities(
            raw_scores=raw_scores,
            temperature=config.temperature,
            min_prob_floor=config.min_prob_floor,
        )

        sampled = sampler.sample(probabilities, sample_size=config.sample_size, seed=seed)
        # History overlap filtering must cover all available rounds, not recent_n window.
        pipeline = _build_filter_pipeline(train_df=full_history_df, config=config)
        filtered = pipeline.filter_combinations(sampled)

        candidate_source = filtered if filtered else sampled
        ranked = ranker.rank(
            combinations=candidate_source,
            raw_scores=raw_scores,
            top_k=config.rank_top_k,
        )
        ranked_combos = [rank.numbers for rank in ranked]
        selected = diversity_selector.select(ranked_combos, output_count=output_count)
        if len(selected) < output_count:
            selected = _fill_shortfall(
                selected=selected,
                ranked_candidates=ranked_combos,
                sampled_candidates=sampled,
                output_count=output_count,
                seed=seed,
            )
        return selected[:output_count]

    return recommender


def _build_filter_pipeline(train_df: pd.DataFrame, config: PenaltyTuneConfig) -> FilterPipeline:
    history_draws = train_df[NUMBER_COLUMNS].astype(int).values.tolist()
    filters = [
        SumFilter(min_sum=config.min_sum, max_sum=config.max_sum),
        ACFilter(min_ac=config.min_ac),
        ZoneFilter(max_per_zone=config.max_per_zone),
        TailFilter(max_same_tail=config.max_same_tail),
        OddEvenFilter(min_odd=config.min_odd, max_odd=config.max_odd),
        HighLowFilter(min_high=config.min_high, max_high=config.max_high, high_start=config.high_start),
        HistoryFilter(historical_draws=history_draws, match_threshold=config.history_match_threshold),
    ]
    return FilterPipeline(filters)


def _fill_shortfall(
    *,
    selected: list[tuple[int, ...]],
    ranked_candidates: list[tuple[int, ...]],
    sampled_candidates: list[tuple[int, ...]],
    output_count: int,
    seed: int | None,
) -> list[tuple[int, ...]]:
    seen = set(selected)
    filled = list(selected)

    for candidate in ranked_candidates:
        if candidate in seen:
            continue
        filled.append(candidate)
        seen.add(candidate)
        if len(filled) >= output_count:
            return filled

    for candidate in sampled_candidates:
        normalized = tuple(sorted(int(value) for value in candidate))
        if normalized in seen:
            continue
        filled.append(normalized)
        seen.add(normalized)
        if len(filled) >= output_count:
            return filled

    baseline_candidates = RandomBaselineGenerator().generate(output_count=output_count * 3, seed=seed)
    for candidate in baseline_candidates:
        if candidate in seen:
            continue
        filled.append(candidate)
        seen.add(candidate)
        if len(filled) >= output_count:
            return filled
    return filled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel tuning for poisson/markov lambdas.")
    parser.add_argument("--history", default="history.csv")
    parser.add_argument("--start-round", type=int, default=None)
    parser.add_argument("--end-round", type=int, default=None)
    parser.add_argument("--recent-n", type=int, default=50)
    parser.add_argument("--output-count", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--rank-top-k", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    parser.add_argument("--poisson-start", type=float, default=0.0)
    parser.add_argument("--poisson-end", type=float, default=0.5)
    parser.add_argument("--poisson-step", type=float, default=0.1)
    parser.add_argument("--markov-start", type=float, default=0.0)
    parser.add_argument("--markov-end", type=float, default=0.5)
    parser.add_argument("--markov-step", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def _format_result_row(rank: int, result: PenaltyTuneResult) -> str:
    return (
        f"{rank:>2}. poisson={result.poisson_lambda:.3f} "
        f"markov={result.markov_lambda:.3f} "
        f"p>=3={result.p_match_ge_3:.4f} "
        f"p>=4={result.p_match_ge_4:.4f} "
        f"avg={result.average_match_count:.4f} "
        f"std={result.std_match_count:.4f}"
    )


def _main() -> None:
    args = _parse_args()
    config = PenaltyTuneConfig(
        history_path=args.history,
        start_round=args.start_round,
        end_round=args.end_round,
        recent_n=args.recent_n,
        output_count=args.output_count,
        sample_size=args.sample_size,
        rank_top_k=args.rank_top_k,
        seed=args.seed,
        workers=args.workers,
    )
    poisson_values = build_float_grid(args.poisson_start, args.poisson_end, args.poisson_step)
    markov_values = build_float_grid(args.markov_start, args.markov_end, args.markov_step)
    _validate_lambda_bounds(poisson_values, "poisson_lambda")
    _validate_lambda_bounds(markov_values, "markov_lambda")

    results = run_penalty_grid_search(
        config=config,
        poisson_values=poisson_values,
        markov_values=markov_values,
    )
    top_n = min(args.top_n, len(results))
    for idx, result in enumerate(results[:top_n], start=1):
        print(_format_result_row(idx, result))

    if args.save_json:
        output = {
            "config": asdict(config),
            "poisson_values": poisson_values,
            "markov_values": markov_values,
            "results": [asdict(result) for result in results],
        }
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
