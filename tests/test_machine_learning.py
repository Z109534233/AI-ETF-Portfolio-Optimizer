"""
Machine Learning module -- deterministic tests (Issue #18 Stage 5).

Pure pytest, no network access: builds synthetic deterministic price series
so leakage/validation checks never depend on live market data availability.
Covers the Stage 5 checklist: no look-ahead leakage in features/targets,
chronological (non-shuffled) train/test ordering, a simple baseline
comparison computed on the SAME held-out test set, explicit train/test
window + target-definition metadata, and an honest insufficient-data state
instead of fake output.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from src.machine_learning import (
    prepare_ml_dataset, time_series_split, run_ml_pipeline,
    _baseline_majority_class_accuracy,
)
from src.technical_indicators import create_ml_features


def _synthetic_prices(seed: int = 11, n_days: int = 800, drift: float = 0.0004, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    daily_ret = drift + rng.normal(0, vol, n_days)
    return pd.Series(100 * np.cumprod(1 + daily_ret), index=dates)


PRICES = _synthetic_prices()


# ── no look-ahead leakage in features ────────────────────────────────────
def test_features_are_computed_from_past_prices_only():
    """Every feature at date t must be reproducible from prices up to and
    including t -- truncating the price series at t and recomputing the
    last row's features must match the full-series computation exactly.
    A feature that peeked into the future would differ once future prices
    are removed."""
    features_full = create_ml_features(PRICES)
    cutoff = 400
    truncated_prices = PRICES.iloc[: cutoff + 1]
    features_truncated = create_ml_features(truncated_prices)

    last_date = truncated_prices.index[-1]
    row_full = features_full.loc[last_date]
    row_truncated = features_truncated.loc[last_date]
    for col in features_full.columns:
        a, b = row_full[col], row_truncated[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert abs(a - b) < 1e-9, f"feature {col} differs when future prices are removed -- possible look-ahead leakage"


def test_target_label_matches_actual_future_return_sign():
    """The target must be 1 iff the price `lookahead` periods ahead is
    actually higher -- built directly from the raw price series, independent
    of prepare_ml_dataset(), to catch a mislabeled target."""
    X, y, index = prepare_ml_dataset(PRICES, lookahead=1)
    assert X is not None
    for date in index[:20]:
        loc = PRICES.index.get_loc(date)
        if loc + 1 >= len(PRICES):
            continue
        expected = int(PRICES.iloc[loc + 1] > PRICES.iloc[loc])
        assert y.loc[date] == expected, f"target label at {date} does not match actual next-period return sign"


# ── chronological (no shuffling) train/test split ────────────────────────
def test_time_series_split_is_chronological_not_shuffled():
    X, y, index = prepare_ml_dataset(PRICES, lookahead=1)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2)

    assert len(X_train) + len(X_test) == len(X)
    # every train date strictly precedes every test date
    assert X_train.index.max() < X_test.index.min()
    # indices were not reordered/shuffled -- train is a contiguous prefix
    assert list(X_train.index) == list(X.index[: len(X_train)])
    assert list(X_test.index) == list(X.index[len(X_train):])


def test_run_ml_pipeline_reports_chronological_train_before_test_window():
    result = run_ml_pipeline(PRICES, model_type="Random Forest", test_size=0.2)
    assert result["error"] is None
    assert result["train_end"] < result["test_start"]
    assert result["train_start"] < result["train_end"]
    assert result["test_start"] < result["test_end"]


# ── simple baseline comparison ────────────────────────────────────────────
def test_baseline_majority_class_uses_train_distribution_not_test():
    """The baseline must be derived from y_train's class balance, never from
    y_test -- using y_test would leak test-set information into the
    'simple' baseline itself, defeating its purpose as an honest floor."""
    y_train_all_up = pd.Series([1] * 50)
    y_test_all_down = pd.Series([0] * 10)
    # majority class in train is "up" (1); baseline predicts 1 for every
    # test row, which is wrong for all of them since y_test is all 0s.
    acc = _baseline_majority_class_accuracy(y_train_all_up, y_test_all_down)
    assert acc == 0.0

    y_train_mixed = pd.Series([1] * 30 + [0] * 20)  # majority = up
    acc2 = _baseline_majority_class_accuracy(y_train_mixed, y_test_all_down)
    assert acc2 == 0.0  # still predicts "up" for every test row -> 0% correct here


def test_run_ml_pipeline_exposes_baseline_accuracy_alongside_real_metrics():
    result = run_ml_pipeline(PRICES, model_type="Logistic Regression", test_size=0.2)
    assert result["error"] is None
    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    # Sanity check: independently recomputable from the same held-out labels.
    X, y, _ = prepare_ml_dataset(PRICES, lookahead=1)
    _, _, y_train, y_test = time_series_split(X, y, test_size=0.2)
    expected_baseline = _baseline_majority_class_accuracy(y_train, y_test)
    # run_ml_pipeline() rounds to 4 decimals before returning.
    assert abs(result["baseline_accuracy"] - round(expected_baseline, 4)) < 1e-9


# ── honest insufficient-data state ───────────────────────────────────────
def test_insufficient_data_returns_explicit_error_not_fake_output():
    tiny_prices = PRICES.iloc[:30]
    result = run_ml_pipeline(tiny_prices)
    assert result.get("error") is not None
    assert "metrics" not in result or result.get("metrics") is None


def test_prepare_ml_dataset_returns_none_below_minimum_observations():
    tiny_prices = PRICES.iloc[:30]
    X, y, index = prepare_ml_dataset(tiny_prices)
    assert X is None and y is None and index is None


# ── target/window metadata is internally consistent ──────────────────────
def test_lookahead_periods_recorded_matches_argument():
    result = run_ml_pipeline(PRICES, model_type="Random Forest", test_size=0.2, lookahead=1)
    assert result["error"] is None
    assert result["lookahead_periods"] == 1


# ── page-level: disclosure actually renders (real network download) ─────
def test_page_renders_target_window_and_baseline_disclosure():
    """End-to-end AppTest smoke check that the new target-definition /
    data-window / baseline-comparison disclosure actually renders on the
    real page after training, not just that the underlying pipeline
    function returns the right dict keys."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None
    at = AppTest.from_file(os.path.join(REPO_ROOT, "pages/5_Machine_Learning.py"), default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    run_btn = next((b for b in at.button if "Train" in (b.label or "")), None)
    assert run_btn is not None
    run_btn.click()
    at.run()
    assert at.exception == []

    corpus = "\n".join(m.value for m in at.markdown)
    assert "Prediction Target" in corpus
    assert "Out-of-Sample Disclosure" in corpus

    captions = "\n".join(c.value for c in at.caption)
    assert "baseline" in captions.lower()
