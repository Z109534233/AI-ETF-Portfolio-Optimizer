"""
Machine Learning walk-forward / expanding-window cross-validation tests
(Issue #45 item 7).

Covers src/machine_learning.py's walk_forward_validation() /
_fit_and_score_fold() / _choose_n_splits() and their wiring into
run_ml_pipeline(), with deterministic synthetic price series -- no network
access anywhere in this file.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.machine_learning import (
    prepare_ml_dataset, time_series_split, run_ml_pipeline,
    walk_forward_validation, _choose_n_splits, _feasible_fold_count,
    MIN_CV_FOLDS,
)


def _synthetic_prices(n_days=900, seed=7, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    daily_ret = drift + rng.normal(0, vol, n_days)
    prices = 100 * np.cumprod(1 + daily_ret)
    return pd.Series(prices, index=dates)


@pytest.fixture
def _dataset():
    prices = _synthetic_prices()
    X, y, index = prepare_ml_dataset(prices, lookahead=1)
    assert X is not None
    return X, y, index


def test_walk_forward_produces_at_least_two_chronological_folds(_dataset):
    X, y, _ = _dataset
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
    result = walk_forward_validation(X_train, y_train, "Random Forest")
    assert result["available"] is True
    assert result["n_folds"] >= MIN_CV_FOLDS
    assert len(result["folds"]) == result["n_folds"]


def test_folds_are_chronological_expanding_and_non_overlapping(_dataset):
    X, y, _ = _dataset
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
    result = walk_forward_validation(X_train, y_train, "Random Forest")
    folds = result["folds"]
    prev_train_end = None
    prev_n_train = 0
    for f in folds:
        train_start = pd.Timestamp(f["train_start"])
        train_end = pd.Timestamp(f["train_end"])
        val_start = pd.Timestamp(f["val_start"])
        val_end = pd.Timestamp(f["val_end"])
        # train -> validation is chronological and non-overlapping
        assert train_end < val_start
        assert train_start <= train_end
        assert val_start <= val_end
        # expanding: each fold's training window is >= the previous one's
        assert f["n_train"] >= prev_n_train
        prev_n_train = f["n_train"]
        # each fold starts strictly after where the previous one's training ended
        if prev_train_end is not None:
            assert val_start > prev_train_end
        prev_train_end = train_end


def test_final_holdout_never_appears_in_any_cv_fold(_dataset):
    X, y, _ = _dataset
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
    holdout_dates = set(X_test.index)
    result = walk_forward_validation(X_train, y_train, "Random Forest")
    assert result["available"] is True
    train_start_max = X_train.index.max()
    for f in result["folds"]:
        assert pd.Timestamp(f["train_end"]) <= train_start_max
        assert pd.Timestamp(f["val_end"]) <= train_start_max
    # Belt-and-suspenders: walk_forward_validation() is only ever given
    # X_train/y_train, so the holdout dates cannot appear in fold ranges at all.
    fold_dates = set()
    for f in result["folds"]:
        fold_dates.add(f["train_start"])
        fold_dates.add(f["val_end"])
    assert not (fold_dates & {str(d.date()) for d in holdout_dates})


def test_logistic_regression_scaler_is_refit_per_fold(_dataset, monkeypatch):
    """Each fold's StandardScaler must be fit ONLY on that fold's own
    training rows -- never on the full X_train, and never reused across
    folds. Spies on StandardScaler.fit to record what it was called with."""
    X, y, _ = _dataset
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

    seen_fit_sizes = []
    original_fit = StandardScaler.fit

    def _spy_fit(self, data, *a, **k):
        seen_fit_sizes.append(len(data))
        return original_fit(self, data, *a, **k)

    monkeypatch.setattr(StandardScaler, "fit", _spy_fit)

    result = walk_forward_validation(X_train, y_train, "Logistic Regression")
    assert result["available"] is True
    # One scaler.fit() call per fold, each sized to that fold's OWN training
    # rows (n_train) -- never the full X_train, and never repeated/reused.
    expected_sizes = [f["n_train"] for f in result["folds"]]
    assert seen_fit_sizes == expected_sizes
    assert all(size < len(X_train) for size in expected_sizes)


def test_fold_metric_aggregation_is_correct(_dataset):
    X, y, _ = _dataset
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)
    result = walk_forward_validation(X_train, y_train, "Random Forest")
    assert result["available"] is True
    accuracies = [f["accuracy"] for f in result["folds"]]
    assert result["accuracy_mean"] == pytest.approx(float(np.mean(accuracies)))
    assert result["accuracy_std"] == pytest.approx(float(np.std(accuracies, ddof=0)))
    aucs = [f["roc_auc"] for f in result["folds"] if f["roc_auc"] is not None]
    assert result["n_valid_auc_folds"] == len(aucs)
    if aucs:
        assert result["roc_auc_mean"] == pytest.approx(float(np.mean(aucs)))
        assert result["roc_auc_std"] == pytest.approx(float(np.std(aucs, ddof=0)))
    else:
        assert result["roc_auc_mean"] is None
        assert result["roc_auc_std"] is None


def test_insufficient_data_returns_unavailable_not_fake_stats():
    tiny_index = pd.bdate_range("2022-01-01", periods=20)
    X_train = pd.DataFrame({"f1": np.random.rand(20)}, index=tiny_index)
    y_train = pd.Series([0, 1] * 10, index=tiny_index)
    result = walk_forward_validation(X_train, y_train, "Random Forest")
    assert result["available"] is False
    assert "reason" in result and result["reason"]
    assert "folds" not in result


def test_choose_n_splits_reduces_safely_for_small_data():
    # Enough for a couple of folds but nowhere near the target of 5.
    n_splits = _choose_n_splits(45, target_splits=5, min_splits=2)
    assert n_splits is None or 2 <= n_splits <= 5
    if n_splits is not None:
        assert _feasible_fold_count(45, n_splits)

    assert _choose_n_splits(5, target_splits=5, min_splits=2) is None


def test_run_ml_pipeline_attaches_walk_forward_cv():
    prices = _synthetic_prices()
    result = run_ml_pipeline(prices, model_type="Random Forest", test_size=0.2, lookahead=1)
    assert result["error"] is None
    assert "walk_forward_cv" in result
    wf = result["walk_forward_cv"]
    assert wf["available"] in (True, False)
    if wf["available"]:
        # Every fold's validation window must end before the final test
        # window starts (no leakage of the held-out test set into CV).
        test_start = pd.Timestamp(result["test_start"])
        for f in wf["folds"]:
            assert pd.Timestamp(f["val_end"]) < test_start
