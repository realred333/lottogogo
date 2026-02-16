"""Unit tests for tuning/feature_builder.py and tuning/xgb_ranker.py."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from lottogogo.tuning.feature_builder import (
    FEATURE_NAMES,
    FeatureBuildError,
    FeatureBuilder,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_history(n_rounds: int = 120) -> pd.DataFrame:
    rng = random.Random(42)
    rows = []
    for r in range(1, n_rounds + 1):
        nums = sorted(rng.sample(range(1, 46), 6))
        rows.append({
            "round": r,
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "bonus": rng.randint(1, 45),
        })
    return pd.DataFrame(rows)


# ── FeatureBuilder tests ────────────────────────────────────────────────────

def test_feature_builder_rejects_empty():
    with pytest.raises(FeatureBuildError, match="empty"):
        FeatureBuilder(pd.DataFrame())


def test_feature_builder_rejects_missing_columns():
    df = pd.DataFrame({"round": [1], "n1": [1]})
    with pytest.raises(FeatureBuildError, match="Missing"):
        FeatureBuilder(df)


def test_feature_builder_shape():
    """Feature matrix should be (n_valid_rounds * 45, n_features)."""
    history = _make_history(120)
    builder = FeatureBuilder(history)
    X, y = builder.build((60, 80))

    n_rounds = len([r for r in range(60, 81) if r in history["round"].values])
    assert X.shape[0] == n_rounds * 45
    assert X.shape[1] == len(FEATURE_NAMES)
    assert y.shape[0] == X.shape[0]


def test_feature_builder_label_distribution():
    """Each round should have exactly 6 positive labels (6 winning numbers)."""
    history = _make_history(120)
    builder = FeatureBuilder(history)
    X, y = builder.build((60, 80))

    n_rounds = X.shape[0] // 45
    # Total positives = 6 per round
    assert y.sum() == pytest.approx(n_rounds * 6)


def test_feature_builder_labels_are_binary():
    history = _make_history(120)
    builder = FeatureBuilder(history)
    _, y = builder.build((60, 80))
    assert set(np.unique(y)) == {0.0, 1.0}


def test_scale_pos_weight():
    """scale_pos_weight should be ~(45-6)/6 ≈ 6.5 for balanced rounds."""
    y = np.array([1, 1, 1, 1, 1, 1] + [0] * 39)  # 1 round
    spw = FeatureBuilder.scale_pos_weight(y)
    assert spw == pytest.approx(39 / 6)


def test_feature_builder_no_valid_rounds():
    history = _make_history(30)
    builder = FeatureBuilder(history)
    with pytest.raises(FeatureBuildError, match="No valid"):
        builder.build((1, 10), min_train_rounds=50)


# ── XGBRanker tests ─────────────────────────────────────────────────────────

def test_xgb_ranker_import():
    """XGBRanker should be importable."""
    from lottogogo.tuning.xgb_ranker import XGBRanker, RankerResult
    assert XGBRanker is not None
    assert RankerResult is not None


def test_xgb_ranker_train_and_evaluate():
    """XGBRanker should produce valid results on synthetic data."""
    from lottogogo.tuning.xgb_ranker import XGBRanker, RankerResult

    history = _make_history(120)
    builder = FeatureBuilder(history)
    ranker = XGBRanker(builder)
    result = ranker.train_and_evaluate(
        train_end=80,
        val_end=120,
        xgb_params={"n_estimators": 10, "max_depth": 3},
    )

    assert isinstance(result, RankerResult)
    assert result.hit_at_15 >= 0
    assert result.hit_at_20 >= 0
    assert result.mean_rank > 0
    assert len(result.feature_importance) == len(FEATURE_NAMES)


def test_xgb_ranker_feature_importance_sums_roughly_to_one():
    """Feature importance values should sum to approximately 1."""
    from lottogogo.tuning.xgb_ranker import XGBRanker

    history = _make_history(120)
    builder = FeatureBuilder(history)
    ranker = XGBRanker(builder)
    result = ranker.train_and_evaluate(
        train_end=80,
        val_end=120,
        xgb_params={"n_estimators": 10, "max_depth": 3},
    )

    total = sum(result.feature_importance.values())
    assert total == pytest.approx(1.0, abs=0.1)
