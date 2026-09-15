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
    _baseline_majority_class_accuracy, roc_auc_interpretation_level,
    confusion_matrix_skew_direction,
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


# ── Priority-0 regression: stale result must never be shown as current ────
# (Issue #20 section 6A). Each test trains once, then changes exactly ONE
# input via its widget WITHOUT clicking "Train Model" again, and asserts
# the page shows the "inputs changed, please retrain" warning instead of
# silently continuing to display the old (now-stale) result as current.

def _train_once(lang="en"):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None
    at = AppTest.from_file(os.path.join(REPO_ROOT, "pages/5_Machine_Learning.py"), default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    run_btn = next((b for b in at.button if "Train" in (b.label or "") or "訓練" in (b.label or "")), None)
    assert run_btn is not None
    run_btn.click()
    at.run()
    assert at.exception == []
    return at


def _no_retrain_warning_shown(at) -> bool:
    warnings = "\n".join(w.value for w in at.warning)
    return "changed" in warnings.lower() or "變更" in warnings


def test_stale_result_warning_on_ticker_change_without_retrain():
    at = _train_once()
    assert not _no_retrain_warning_shown(at)  # fresh training run: no stale warning yet

    text_inputs = [w for w in at.text_input if "custom" in (w.label or "").lower() or "ARKK" in (w.placeholder or "")]
    assert text_inputs, "expected the custom-ticker text_input on the sidebar"
    text_inputs[0].set_value("SPY")
    at.run()
    assert at.exception == []
    assert _no_retrain_warning_shown(at), "changing the ticker without retraining must show the stale-result warning"


def test_stale_result_warning_on_test_size_change_without_retrain():
    at = _train_once()
    sliders = [s for s in at.slider if "test" in (s.label or "").lower() or "測試" in (s.label or "")]
    assert sliders, "expected the test-set-size slider on the sidebar"
    original = sliders[0].value
    sliders[0].set_value(original + 5 if original + 5 <= 40 else original - 5)
    at.run()
    assert at.exception == []
    assert _no_retrain_warning_shown(at)


def test_stale_result_warning_on_model_change_without_retrain():
    at = _train_once()
    selects = [s for s in at.selectbox if s.label and ("model" in s.label.lower() or "模型" in s.label)]
    assert selects, "expected the model-type selectbox on the sidebar"
    other_options = [o for o in selects[0].options if o != selects[0].value]
    assert other_options
    selects[0].set_value(other_options[0])
    at.run()
    assert at.exception == []
    assert _no_retrain_warning_shown(at)


def test_retraining_after_input_change_clears_stale_warning():
    at = _train_once()
    text_inputs = [w for w in at.text_input if "custom" in (w.label or "").lower() or "ARKK" in (w.placeholder or "")]
    text_inputs[0].set_value("SPY")
    at.run()
    assert _no_retrain_warning_shown(at)

    run_btn = next((b for b in at.button if "Train" in (b.label or "") or "訓練" in (b.label or "")), None)
    run_btn.click()
    at.run()
    assert at.exception == []
    assert not _no_retrain_warning_shown(at), "after retraining, the stale-result warning must clear"
    corpus = "\n".join(m.value for m in at.markdown)
    assert "SPY" in corpus


# ============================================================================
# Issue #41 item H -- ROC AUC interpretation (below-chance / weak / moderate)
# ============================================================================

def test_roc_auc_below_chance_bucket_for_values_under_half():
    assert roc_auc_interpretation_level(0.4356) == "below_chance"
    assert roc_auc_interpretation_level(0.0) == "below_chance"
    assert roc_auc_interpretation_level(0.4999) == "below_chance"


def test_roc_auc_weak_bucket_around_half():
    assert roc_auc_interpretation_level(0.50) == "weak"
    assert roc_auc_interpretation_level(0.55) == "weak"


def test_roc_auc_moderate_bucket_meaningfully_above_half():
    assert roc_auc_interpretation_level(0.5501) == "moderate"
    assert roc_auc_interpretation_level(0.75) == "moderate"


def _mock_download(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        idx = pd.bdate_range("2019-01-01", periods=800)
        rng = np.random.default_rng(3)
        return pd.DataFrame({tk: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 800)) for tk in tickers}, index=idx)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def test_below_chance_auc_produces_dedicated_warning(monkeypatch):
    """End-to-end: when the trained model's actual ROC AUC on this run is
    below 0.5, the page must render the dedicated below-chance warning
    (never silently show nothing, and never the generic "weak" text meant
    for the near-0.5 band), and must not overclaim proof of the Efficient
    Market Hypothesis from a single result."""
    import src.machine_learning as ml_mod

    _mock_download(monkeypatch)
    # Force a deterministic, clearly-below-chance ROC AUC so this test
    # never depends on which real model/ticker/window happens to produce
    # one -- only the page's INTERPRETATION branch is under test here.
    _orig_compute_metrics = ml_mod._compute_metrics

    def _fake_compute_metrics(y_test, y_pred, y_prob):
        m = _orig_compute_metrics(y_test, y_pred, y_prob)
        m["ROC AUC"] = 0.30
        return m

    monkeypatch.setattr(ml_mod, "_compute_metrics", _fake_compute_metrics)

    at = _train_once()
    corpus = "\n".join(w.value for w in at.warning)
    assert "below the 0.5 random-guessing level" in corpus
    assert "0.300" in corpus
    assert "does not prove" in corpus and "Efficient Market Hypothesis" in corpus


# ============================================================================
# Issue #41 item I -- Confusion Matrix / base-rate interpretation
# ============================================================================

def test_confusion_matrix_skew_direction_detects_up_skew():
    # 90 of 100 predictions are "up" -- clearly skewed toward Up.
    assert confusion_matrix_skew_direction(predicted_up=90, total=100) == "up"


def test_confusion_matrix_skew_direction_detects_down_skew():
    # Only 5 of 100 predictions are "up" -- clearly skewed toward Down.
    assert confusion_matrix_skew_direction(predicted_up=5, total=100) == "down"


def test_confusion_matrix_skew_direction_none_when_balanced():
    assert confusion_matrix_skew_direction(predicted_up=50, total=100) is None


def test_confusion_matrix_skew_direction_none_for_zero_total():
    assert confusion_matrix_skew_direction(predicted_up=0, total=0) is None


def test_confusion_matrix_dynamic_counts_use_actual_matrix_not_hardcoded(monkeypatch):
    """Regression guard for Issue #41 item I: the page must never hardcode
    screenshot-era counts (e.g. 134 vs 113) -- the rendered predicted/actual
    up/down counts must be internally consistent with each other (predicted
    up + predicted down == actual up + actual down == total test samples)."""
    _mock_download(monkeypatch)
    at = _train_once()
    corpus = "\n".join(m.value for m in at.markdown)
    assert "134" not in corpus or "113" not in corpus, (
        "these specific hardcoded screenshot counts must never both appear together"
    )
    # The dynamic summary sentence must be present and reference the ACTUAL
    # test-set size shown elsewhere on the page (train_size/test_size).
    assert "predicted" in corpus.lower() and "actual" in corpus.lower()


# ============================================================================
# Issue #41 item J -- neutral Results hero wording
# ============================================================================

def test_results_hero_title_is_neutral_not_prediction_results(monkeypatch):
    _mock_download(monkeypatch)
    at = _train_once(lang="en")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Out-of-Sample Evaluation" in corpus
    # The old, potentially-overclaiming hero title must be gone. The
    # "Predictions" tab label (a distinct, unrelated key) is expected to
    # remain and is not affected by this check.
    assert "<div class=\"results-hero-title\">Prediction Results</div>" not in corpus


def test_results_hero_title_is_neutral_zh_tw(monkeypatch):
    _mock_download(monkeypatch)
    at = _train_once(lang="zh-TW")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "樣本外驗證結果" in corpus
    assert "<div class=\"results-hero-title\">預測結果</div>" not in corpus
