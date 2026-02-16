"""XGBoost ranking model for lottery number prediction.

Trains an XGBoost classifier on engine-derived features and compares
its hit@K performance against the GA-optimized engine.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
except ImportError:
    optuna = None  # type: ignore[assignment]
    TimeSeriesSplit = None  # type: ignore[assignment]

import json
from pathlib import Path

from lottogogo.tuning.feature_builder import (
    FEATURE_NAMES,
    FeatureBuildError,
    FeatureBuilder,
)

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
TOTAL_NUMBERS = 45


class XGBRankerError(Exception):
    """Raised when XGBoost ranking fails."""


@dataclass
class RankerResult:
    """Result of XGBoost ranking evaluation."""

    hit_at_15: float
    hit_at_20: float
    mean_rank: float
    feature_importance: dict[str, float]


class XGBRanker:
    """XGBoost-based ranker for lottery numbers."""

    def __init__(self, feature_builder: FeatureBuilder) -> None:
        if xgb is None:
            raise ImportError(
                "xgboost is required. Install with: uv add xgboost"
            )
        self.feature_builder = feature_builder

    def train_and_evaluate(
        self,
        train_end: int,
        val_end: int,
        xgb_params: dict[str, Any] | None = None,
    ) -> RankerResult:
        """Train XGBoost and evaluate on validation set."""
        history = self.feature_builder.history
        all_rounds = sorted(history["round"].unique())
        val_start = train_end + 1

        # Build training features
        X_train, y_train = self.feature_builder.build((1, train_end))

        # Compute scale_pos_weight
        spw = FeatureBuilder.scale_pos_weight(y_train)

        # Default XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "scale_pos_weight": spw,
            "seed": 42,
            "verbosity": 0,
        }
        if xgb_params:
            params.update(xgb_params)

        # Train
        n_estimators = params.pop("n_estimators", 200)
        seed = params.pop("seed", 42)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            **params,
        )
        model.fit(X_train, y_train)

        # Feature importance
        importance = model.feature_importances_
        feat_imp = {
            name: float(importance[i])
            for i, name in enumerate(FEATURE_NAMES)
        }

        # Evaluate on validation rounds
        val_rounds = [r for r in all_rounds if val_start <= r <= val_end]
        if not val_rounds:
            raise XGBRankerError(f"No validation rounds in [{val_start}, {val_end}]")

        hits_15: list[int] = []
        hits_20: list[int] = []
        ranks: list[float] = []

        for target_round in val_rounds:
            prior = history[history["round"] < target_round]
            if len(prior) < 50:
                continue

            actual_row = history[history["round"] == target_round]
            if actual_row.empty:
                continue
            actual_numbers = set(
                int(actual_row.iloc[0][col]) for col in NUMBER_COLUMNS
            )

            # Extract features for this round
            try:
                features = self.feature_builder._extract_features(prior, target_round)
            except Exception:
                continue

            # Predict probabilities
            probs = model.predict_proba(features)[:, 1]
            # Numbers 1-45, ranked by predicted probability
            ranked_numbers = sorted(
                range(1, 46), key=lambda n: probs[n - 1], reverse=True
            )

            # Hit@K
            top_15 = set(ranked_numbers[:15])
            top_20 = set(ranked_numbers[:20])
            hits_15.append(len(top_15 & actual_numbers))
            hits_20.append(len(top_20 & actual_numbers))

            # Mean rank of actual numbers
            rank_map = {n: i + 1 for i, n in enumerate(ranked_numbers)}
            round_ranks = [rank_map[n] for n in actual_numbers if n in rank_map]
            ranks.append(float(np.mean(round_ranks)) if round_ranks else 23.0)

        if not hits_15:
            raise XGBRankerError("No valid validation rounds")

        return RankerResult(
            hit_at_15=float(np.mean(hits_15)),
            hit_at_20=float(np.mean(hits_20)),
            mean_rank=float(np.mean(ranks)),
            feature_importance=feat_imp,
        )
    
    def tune_hyperparams(
        self,
        train_end: int,
        val_end: int,
        n_trials: int = 50,
        cv_folds: int = 3,
        output_dir: str = "data",
    ) -> dict[str, Any]:
        """Tune XGBoost hyperparameters using Optuna with TimeSeriesSplit CV.
        
        Args:
            train_end: Last training round
            val_end: Last validation round
            n_trials: Number of Optuna trials
            cv_folds: Number of TimeSeriesSplit folds
            output_dir: Directory to save results
            
        Returns:
            Best hyperparameters dict
        """
        if optuna is None:
            raise ImportError("optuna is required. Install with: uv add optuna")
        if TimeSeriesSplit is None:
            raise ImportError("scikit-learn is required for TimeSeriesSplit")
        
        # Build training data
        X_train, y_train = self.feature_builder.build((1, train_end))
        spw = FeatureBuilder.scale_pos_weight(y_train)
        
        # Define Optuna objective
        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "scale_pos_weight": spw,
                "seed": 42,
                "verbosity": 0,
            }
            
            # TimeSeriesSplit CV
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, verbose=False)
                
                # Evaluate on validation fold
                y_pred = model.predict_proba(X_val)[:, 1]
                # Use logloss as optimization metric
                from sklearn.metrics import log_loss
                score = log_loss(y_val, y_pred)
                cv_scores.append(score)
            
            return float(np.mean(cv_scores))
        
        # Run Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params["scale_pos_weight"] = spw
        best_params["seed"] = 42
        best_params["verbosity"] = 0
        
        # Save results (T11)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "xgb_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n✅ Best hyperparameters saved to {output_path / 'xgb_best_params.json'}")
        print(f"Best CV logloss: {study.best_value:.4f}")
        
        return best_params



def print_xgb_comparison(
    xgb_result: RankerResult,
    ga_result: dict[str, float] | None = None,
) -> None:
    """Print XGBoost results with optional GA comparison."""
    print("\n" + "=" * 60)
    print("XGBoost Ranker Result")
    print("=" * 60)
    print(f"{'Metric':<20} {'XGBoost':>12}", end="")
    if ga_result:
        print(f" {'GA':>12} {'Δ':>10}", end="")
    print()
    print("-" * 60)

    metrics = [
        ("hit@15", xgb_result.hit_at_15, ga_result.get("hit_at_15", 0) if ga_result else None),
        ("hit@20", xgb_result.hit_at_20, ga_result.get("hit_at_20", 0) if ga_result else None),
        ("mean_rank", xgb_result.mean_rank, ga_result.get("mean_rank", 0) if ga_result else None),
    ]
    for name, xgb_val, ga_val in metrics:
        print(f"{name:<20} {xgb_val:>12.4f}", end="")
        if ga_val is not None:
            delta = xgb_val - ga_val
            print(f" {ga_val:>12.4f} {delta:>+10.4f}", end="")
        print()

    print("-" * 60)
    print("\nFeature Importance (top 10):")
    sorted_imp = sorted(
        xgb_result.feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    for name, imp in sorted_imp[:10]:
        bar = "█" * int(imp * 50)
        print(f"  {name:<22} {imp:.4f} {bar}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: python -m lottogogo.tuning.xgb_ranker"""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost Ranker for LottoGoGo")
    parser.add_argument("--csv", default="history.csv", help="Path to history CSV")
    parser.add_argument("--train-end", type=int, default=900, help="Last train round")
    parser.add_argument("--val-end", type=int, default=1100, help="Last validation round")
    parser.add_argument(
        "--ga-weights", default=None,
        help="Path to GA optimized_weights.json for comparison",
    )
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--cv-folds", type=int, default=3, help="TimeSeriesSplit CV folds")
    args = parser.parse_args()

    history = pd.read_csv(args.csv)
    builder = FeatureBuilder(history)
    ranker = XGBRanker(builder)
    
    # Run tuning if requested
    if args.tune:
        print(f"\n🔧 Running Optuna hyperparameter tuning ({args.n_trials} trials, {args.cv_folds} CV folds)...")
        best_params = ranker.tune_hyperparams(
            args.train_end, args.val_end,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
        )
        print("\n📊 Training final model with best hyperparameters...")
        result = ranker.train_and_evaluate(args.train_end, args.val_end, xgb_params=best_params)
    else:
        result = ranker.train_and_evaluate(args.train_end, args.val_end)
    
    # Save feature importance (T11)
    output_path = Path("data")
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "xgb_feature_importance.json", "w") as f:
        json.dump(result.feature_importance, f, indent=2)
    print(f"\n✅ Feature importance saved to {output_path / 'xgb_feature_importance.json'}")

    ga_fitness = None
    if args.ga_weights:
        try:
            with open(args.ga_weights) as f:
                data = json.load(f)
            ga_fitness = data.get("fitness", {})
        except Exception as e:
            print(f"[WARN] Could not load GA weights: {e}")

    print_xgb_comparison(result, ga_fitness)


if __name__ == "__main__":
    main()
