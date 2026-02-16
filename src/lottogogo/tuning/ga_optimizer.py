"""GA weight optimizer using DEAP.

Optimizes the 10 engine weights via genetic algorithm with
tournament selection, blend crossover, and Gaussian mutation.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools
from tqdm import tqdm

from lottogogo.tuning.fitness import (
    WEIGHT_BOUNDS,
    WEIGHT_BOUNDS_NO_HMM,
    WEIGHT_KEYS,
    FitnessEvaluator,
    FitnessResult,
    random_baseline,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    """Configuration for GA optimization."""

    population_size: int = 50  # Reduced from 100 for 8D search space
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.1  # Initial sigma, will adapt during evolution
    elitism_count: int = 5
    tournament_size: int = 3  # Reduced from 5 for diversity
    seed: int = 42
    jobs: int = 1

    def __post_init__(self) -> None:
        if self.population_size < 10:
            raise ValueError("population_size must be >= 10")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.elitism_count > self.population_size // 2:
            raise ValueError("elitism_count must be <= population_size // 2")


@dataclass
class OptimizationResult:
    """Final result of GA optimization."""

    best_weights: dict[str, float]
    best_fitness: FitnessResult
    baseline_fitness: FitnessResult
    generation_log: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# GA Optimizer
# ---------------------------------------------------------------------------


def _vec_to_weights(vec: list[float], weight_keys: list[str]) -> dict[str, float]:
    """Convert a flat vector to a named weight dict."""
    return {key: vec[i] for i, key in enumerate(weight_keys)}


def _weights_to_vec(weights: dict[str, float], weight_keys: list[str]) -> list[float]:
    """Convert a named weight dict to a flat vector."""
    return [weights[key] for key in weight_keys]


def _clamp_individual(ind: list[float], weight_bounds: dict[str, tuple[float, float]], weight_keys: list[str]) -> list[float]:
    """Clamp each gene to its valid range."""
    for i, key in enumerate(weight_keys):
        lo, hi = weight_bounds[key]
        ind[i] = max(lo, min(hi, ind[i]))
    return ind


class GAOptimizer:
    """Genetic Algorithm optimizer for engine weights."""

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        config: GAConfig | None = None,
        use_hmm: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or GAConfig()
        self.use_hmm = use_hmm
        self.weight_bounds = WEIGHT_BOUNDS if use_hmm else WEIGHT_BOUNDS_NO_HMM
        self.weight_keys = list(self.weight_bounds.keys())
        self._setup_deap()

    def _setup_deap(self) -> None:
        """Register DEAP types and operators."""
        # Avoid re-creating if already registered
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Gene initialization: uniform random within bounds
        def _init_gene(i: int) -> float:
            lo, hi = self.weight_bounds[self.weight_keys[i]]
            return random.uniform(lo, hi)

        def _init_individual() -> Any:
            genes = [_init_gene(i) for i in range(len(self.weight_keys))]
            return creator.Individual(genes)

        self.toolbox.register("individual", _init_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.config.tournament_size
        )
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        # Parallelism is handled in run(); keep DEAP's map as the built-in map.
        self.toolbox.register("map", map)

        self.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0.0,
            sigma=self.config.mutation_sigma,
            indpb=0.2,
        )

    def _evaluate(self, individual: list[float]) -> tuple[float]:
        """Fitness function for DEAP (returns tuple)."""
        return (self._parallel_evaluate(individual),)

    def _parallel_evaluate(self, individual: list[float]) -> float:
        """Helper for parallel map (returns float)."""
        weights = _vec_to_weights(individual, self.weight_keys)
        try:
            return self.evaluator.evaluate(weights).combined_fitness
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return 0.0

    def run(
        self,
        checkpoint_path: Path | None = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """Run the GA optimization.

        Args:
            checkpoint_path: If given, save/load checkpoints here.
            verbose: Print progress to stdout.
        """
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        executor: ThreadPoolExecutor | None = None
        if cfg.jobs > 1:
            try:
                executor = ThreadPoolExecutor(max_workers=cfg.jobs)
                if verbose:
                    print(f"[GA] Parallel evaluation enabled (threads={cfg.jobs})")
            except Exception as e:
                executor = None
                print(f"[GA][WARN] Failed to create thread pool ({e}); using single worker.")

        def _map_eval(items: list[Any]):
            if executor is None:
                return map(self._parallel_evaluate, items)
            return executor.map(self._parallel_evaluate, items)

        # Try loading checkpoint
        start_gen = 0
        pop: list[Any] = []
        gen_log: list[dict[str, Any]] = []

        if checkpoint_path and checkpoint_path.exists():
            cp = self._load_checkpoint(checkpoint_path)
            start_gen = cp["generation"] + 1
            gen_log = cp.get("generation_log", [])
            pop = [creator.Individual(genes) for genes in cp["population"]]
            # Re-evaluate fitness
            for ind in pop:
                ind.fitness.values = self._evaluate(ind)
            if verbose:
                print(f"[GA] Resumed from checkpoint at generation {start_gen}")

        if not pop:
            pop = self.toolbox.population(n=cfg.population_size)
            # Evaluate initial population
            print("[GA] Evaluating initial population...")
            for ind, fit in zip(
                pop,
                tqdm(
                    _map_eval(pop),
                    total=len(pop),
                    desc="  Initial Pop",
                    leave=False,
                ),
            ):
                ind.fitness.values = (fit,)

        best_ever = tools.selBest(pop, 1)[0]
        t0 = time.time()

        try:
            # Main GA Loop
            with tqdm(total=cfg.generations, initial=start_gen, desc="GA Overall") as pbar:
                for gen in range(start_gen, cfg.generations):
                    # Elitism: preserve best individuals
                    elites = tools.selBest(pop, cfg.elitism_count)
                    elites = [self.toolbox.clone(e) for e in elites]

                    # Selection
                    offspring = self.toolbox.select(pop, len(pop) - cfg.elitism_count)
                    offspring = [self.toolbox.clone(o) for o in offspring]

                    # Crossover
                    for i in range(0, len(offspring) - 1, 2):
                        if random.random() < cfg.crossover_rate:
                            self.toolbox.mate(offspring[i], offspring[i + 1])
                            _clamp_individual(offspring[i], self.weight_bounds, self.weight_keys)
                            _clamp_individual(offspring[i + 1], self.weight_bounds, self.weight_keys)
                            del offspring[i].fitness.values
                            del offspring[i + 1].fitness.values

                    # Mutation with adaptive sigma (0.1 → 0.02)
                    progress = gen / cfg.generations
                    adaptive_sigma = 0.1 * (1 - progress) + 0.02 * progress
                    
                    for ind in offspring:
                        if random.random() < cfg.mutation_rate:
                            # Apply Gaussian mutation with adaptive sigma
                            tools.mutGaussian(ind, mu=0.0, sigma=adaptive_sigma, indpb=0.2)
                            _clamp_individual(ind, self.weight_bounds, self.weight_keys)
                            del ind.fitness.values

                    # Evaluate invalidated individuals
                    invalid = [ind for ind in offspring if not ind.fitness.valid]
                    if invalid:
                        for ind, fit in zip(
                            invalid,
                            tqdm(
                                _map_eval(invalid),
                                total=len(invalid),
                                desc="  Evaluating",
                                leave=False,
                            ),
                        ):
                            ind.fitness.values = (fit,)

                    # Replace population = elites + offspring
                    pop = elites + offspring

                    # Stats
                    fits = [ind.fitness.values[0] for ind in pop]
                    gen_best = tools.selBest(pop, 1)[0]
                    gen_entry = {
                        "generation": gen,
                        "best": float(max(fits)),
                        "avg": float(np.mean(fits)),
                        "min": float(min(fits)),
                        "std": float(np.std(fits)),
                    }
                    gen_log.append(gen_entry)

                    if gen_best.fitness.values[0] > best_ever.fitness.values[0]:
                        best_ever = self.toolbox.clone(gen_best)

                    # Checkpoint every 25 generations
                    if checkpoint_path and gen % 25 == 0:
                        self._save_checkpoint(checkpoint_path, gen, pop, gen_log)

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "best": f"{gen_entry['best']:.4f}",
                            "avg": f"{gen_entry['avg']:.4f}",
                        }
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # Build final result
        best_weights = _vec_to_weights(list(best_ever), self.weight_keys)
        best_fitness = self.evaluator.evaluate(best_weights)
        baseline = FitnessResult(
            hit_at_15=random_baseline(15),
            hit_at_20=random_baseline(20),
            mean_rank=TOTAL_NUMBERS / 2,
            train_fitness=random_baseline(15),
            val_fitness=random_baseline(15),
            combined_fitness=random_baseline(15),
        )

        return OptimizationResult(
            best_weights=best_weights,
            best_fitness=best_fitness,
            baseline_fitness=baseline,
            generation_log=gen_log,
        )

    def _save_checkpoint(
        self, path: Path, generation: int, pop: list, gen_log: list
    ) -> None:
        """Save GA state to a JSON checkpoint file."""
        data = {
            "generation": generation,
            "population": [list(ind) for ind in pop],
            "generation_log": gen_log,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_checkpoint(self, path: Path) -> dict:
        """Load GA state from a checkpoint file."""
        return json.loads(path.read_text(encoding="utf-8"))


TOTAL_NUMBERS = 45


# ---------------------------------------------------------------------------
# Result saving / reporting / plotting
# ---------------------------------------------------------------------------


def generate_plot(gen_log: list[dict[str, Any]], path: Path) -> None:
    """Generate fitness convergence plot."""
    gens = [entry["generation"] for entry in gen_log]
    best_fits = [entry["best"] for entry in gen_log]
    avg_fits = [entry["avg"] for entry in gen_log]

    plt.figure(figsize=(10, 6))
    plt.plot(gens, best_fits, label="Best Fitness", color="blue", linewidth=2)
    plt.plot(gens, avg_fits, label="Average Fitness", color="green", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Combined Fitness")
    plt.title("GA Optimization Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_result(result: OptimizationResult, path: Path, cycle_label: str = "ga-weight-optimization-20260215") -> None:
    """Save optimization result to JSON (data/optimized_weights.json)."""
    output = {
        "cycle_label": cycle_label,
        "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "weights": result.best_weights,
        "fitness": {
            "hit_at_15": result.best_fitness.hit_at_15,
            "hit_at_20": result.best_fitness.hit_at_20,
            "mean_rank": result.best_fitness.mean_rank,
            "train_fitness": result.best_fitness.train_fitness,
            "val_fitness": result.best_fitness.val_fitness,
            "combined_fitness": result.best_fitness.combined_fitness,
        },
        "baseline": {
            "hit_at_15": result.baseline_fitness.hit_at_15,
            "hit_at_20": result.baseline_fitness.hit_at_20,
            "mean_rank": result.baseline_fitness.mean_rank,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def print_comparison(result: OptimizationResult) -> None:
    """Print a comparison table between baseline and optimized weights."""
    b = result.baseline_fitness
    o = result.best_fitness

    print("\n" + "=" * 60)
    print("GA Optimization Result — Comparison")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Δ':>10}")
    print("-" * 60)
    print(f"{'hit@15':<20} {b.hit_at_15:>12.4f} {o.hit_at_15:>12.4f} {o.hit_at_15 - b.hit_at_15:>+10.4f}")
    print(f"{'hit@20':<20} {b.hit_at_20:>12.4f} {o.hit_at_20:>12.4f} {o.hit_at_20 - b.hit_at_20:>+10.4f}")
    print(f"{'mean_rank':<20} {b.mean_rank:>12.2f} {o.mean_rank:>12.2f} {o.mean_rank - b.mean_rank:>+10.2f}")
    print(f"{'combined_fitness':<20} {b.combined_fitness:>12.4f} {o.combined_fitness:>12.4f} {o.combined_fitness - b.combined_fitness:>+10.4f}")
    print("-" * 60)

    print("\nOptimized Weights:")
    for key, val in sorted(result.best_weights.items()):
        print(f"  {key:<22} = {val:.6f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: python -m lottogogo.tuning.ga_optimizer"""
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="GA Weight Optimizer for LottoGoGo")
    parser.add_argument("--csv", default="history.csv", help="Path to history CSV")
    parser.add_argument("--train-end", type=int, default=900, help="Last train round")
    parser.add_argument("--val-end", type=int, default=1100, help="Last validation round")
    parser.add_argument("--population", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--cycle-label", default="ga-weight-optimization-20260215", help="Label for this run")
    parser.add_argument("--output", default="data/optimized_weights.json", help="Output JSON path")
    parser.add_argument("--plot", default="data/fitness_history.png", help="Convergence plot path")
    parser.add_argument("--checkpoint", default="data/ga_checkpoint.json", help="Checkpoint path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--use-hmm", action="store_true", help="Enable HMM weights in search space (default: disabled)")
    args = parser.parse_args()

    history = pd.read_csv(args.csv)
    evaluator = FitnessEvaluator(history, args.train_end, args.val_end)
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        jobs=args.jobs,
    )
    optimizer = GAOptimizer(evaluator, config, use_hmm=args.use_hmm)
    result = optimizer.run(
        checkpoint_path=Path(args.checkpoint),
        verbose=not args.quiet,
    )

    save_result(result, Path(args.output), cycle_label=args.cycle_label)
    generate_plot(result.generation_log, Path(args.plot))
    print_comparison(result)
    print(f"[OK] Saved to {args.output}")
    print(f"[OK] Generated plot at {args.plot}")


if __name__ == "__main__":
    main()
