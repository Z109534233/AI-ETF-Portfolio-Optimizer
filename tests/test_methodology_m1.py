"""
M1 -- Portfolio Optimization Methodology: deterministic tests.

Pure pytest `assert` tests (no network, no Streamlit AppTest) covering:
  - src/methodology.py's PORTFOLIO_OPTIMIZATION_METHODOLOGY metadata and
    validate_optimization_result() in isolation
  - src/portfolio_optimizer.py's run_optimization() wiring the validation
    result into every optimization method's output
  - no pre-inception backward fill / no silent ETF dropping
    (src/data_cleaner.py's get_common_date_range + clean_price_data)
  - explicit, surfaced insufficient-history behavior
  - i18n parity (zh-TW / en) for every new opt_methodology_* key
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.methodology import (
    PORTFOLIO_OPTIMIZATION_METHODOLOGY, validate_optimization_result,
)
from src.portfolio_optimizer import run_optimization
from src.data_cleaner import get_common_date_range, clean_price_data
from src.i18n import TRANSLATIONS


def make_synthetic_prices(seed: int = 7, n_days: int = 300) -> pd.DataFrame:
    """4 tickers with distinct drift/volatility and a shared market factor,
    deterministic via a fixed seed -- same style as
    tests/test_portfolio_optimizer.py's fixture, so results are stable and
    require no network access."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    drifts = [0.0007, 0.0002, 0.0005, 0.0001]
    vols = [0.015, 0.009, 0.012, 0.007]
    market = rng.normal(0, 0.010, n_days)
    prices = {}
    for tkr, mu, sigma in zip(tickers, drifts, vols):
        idio = rng.normal(0, sigma, n_days)
        daily_ret = mu + 0.3 * market + idio
        price = 100 * np.cumprod(1 + daily_ret)
        prices[tkr] = price
    return pd.DataFrame(prices, index=dates)


PRICES = make_synthetic_prices()


# ── Methodology metadata ─────────────────────────────────────────────────
def test_methodology_metadata_matches_actual_implementation():
    m = PORTFOLIO_OPTIMIZATION_METHODOLOGY
    assert "252" in m["expected_return"]["annualization"]
    assert "pct_change" in m["expected_return"]["return_type"] or "simple" in m["expected_return"]["return_type"]
    assert "252" in m["covariance"]["annualization"]
    assert "cov()" in m["covariance"]["estimator"]
    assert "Ledoit-Wolf" in m["covariance"]["shrinkage"] and "none" in m["covariance"]["shrinkage"].lower()
    assert "SLSQP" in m["max_sharpe"]["solver"]
    assert "sum(weights) == 1" in m["max_sharpe"]["constraints"]
    assert m["backtest_label"]["type"] == "Fixed-Allocation Historical Backtest"
    assert "walk-forward" in m["backtest_label"]["explanation"]


# ── validate_optimization_result() ───────────────────────────────────────
def test_validation_passes_for_consistent_result():
    weights = {"AAA": 0.5, "BBB": 0.5}
    mean_returns = np.array([0.001, 0.0005])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    w = np.array([0.5, 0.5])
    ret = float(np.dot(w, mean_returns) * 252)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - 0.05) / vol
    result = validate_optimization_result(
        weights, mean_returns, cov, ret, vol, sharpe, risk_free_rate=0.05,
        min_weight=0.0, max_weight=1.0, allow_short=False,
    )
    assert result["is_valid"] is True
    assert result["issues"] == []


def test_validation_flags_weights_not_summing_to_one():
    weights = {"AAA": 0.5, "BBB": 0.3}
    mean_returns = np.array([0.001, 0.0005])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    result = validate_optimization_result(
        weights, mean_returns, cov, 0.1, 0.1, 0.5, risk_free_rate=0.05,
    )
    assert result["is_valid"] is False
    assert any("sum to" in issue for issue in result["issues"])


def test_validation_flags_out_of_bounds_weight():
    weights = {"AAA": 0.9, "BBB": 0.1}
    mean_returns = np.array([0.001, 0.0005])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    w = np.array([0.9, 0.1])
    ret = float(np.dot(w, mean_returns) * 252)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - 0.05) / vol
    result = validate_optimization_result(
        weights, mean_returns, cov, ret, vol, sharpe, risk_free_rate=0.05,
        min_weight=0.0, max_weight=0.5, allow_short=False,
    )
    assert result["is_valid"] is False
    assert any("bounds" in issue for issue in result["issues"])


def test_validation_flags_inconsistent_reported_return():
    weights = {"AAA": 0.5, "BBB": 0.5}
    mean_returns = np.array([0.001, 0.0005])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    w = np.array([0.5, 0.5])
    vol = float(np.sqrt(w @ cov @ w))
    result = validate_optimization_result(
        weights, mean_returns, cov,
        reported_return=999.0, reported_volatility=vol, reported_sharpe=0.0,
        risk_free_rate=0.05,
    )
    assert result["is_valid"] is False
    assert any("expected_return" in issue for issue in result["issues"])


# ── run_optimization() wiring ────────────────────────────────────────────
def test_run_optimization_attaches_validation_for_every_method():
    for method in ("Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility",
                   "Target Return", "Risk Parity"):
        result = run_optimization(PRICES, method=method, risk_free_rate=0.03)
        assert result.get("error") is None, f"{method}: {result.get('error')}"
        validation = result.get("validation")
        assert validation is not None, f"{method}: missing validation key"
        assert validation["is_valid"] is True, f"{method}: {validation['issues']}"


def test_run_optimization_validation_respects_configured_bounds():
    result = run_optimization(
        PRICES, method="Maximum Sharpe Ratio", risk_free_rate=0.03,
        min_weight=0.10, max_weight=0.40,
    )
    assert result.get("error") is None, str(result.get("error"))
    w = np.array(list(result["weights"].values()))
    assert np.all(w >= 0.10 - 1e-6) and np.all(w <= 0.40 + 1e-6)
    assert result["validation"]["is_valid"] is True, result["validation"]["issues"]


# ── No pre-inception backward fill / no silent ETF dropping ─────────────
def test_common_date_range_excludes_pre_inception_dates():
    """A later-inception ETF must push the common start date forward, not
    be backfilled to match the earlier tickers."""
    prices = PRICES.copy()
    late_inception_idx = 100
    prices.loc[prices.index[:late_inception_idx], "DDD"] = np.nan

    common_start, common_end = get_common_date_range(prices)
    assert common_start == prices.index[late_inception_idx]

    sliced = prices.loc[(prices.index >= common_start) & (prices.index <= common_end)]
    cleaned = clean_price_data(sliced)

    # Every originally-selected ticker survives -- none silently dropped.
    assert set(cleaned.columns) == set(prices.columns)
    # No date before the later ETF's real inception is present at all, so
    # there is nothing for ffill/bfill to have backfilled into.
    assert cleaned.index.min() == prices.index[late_inception_idx]
    assert not cleaned["DDD"].isna().any()


def test_fully_missing_ticker_is_not_silently_included():
    """A ticker with zero valid data anywhere must make the common range
    computation fail explicitly (None, None), never silently compute a
    range that ignores it."""
    prices = PRICES.copy()
    prices["EEE"] = np.nan
    common_start, common_end = get_common_date_range(prices)
    assert (common_start, common_end) == (None, None)


# ── Insufficient-history behavior is explicit ────────────────────────────
def test_insufficient_history_is_explicit_not_silent():
    short_prices = PRICES.iloc[:5]
    result = run_optimization(short_prices, method="Maximum Sharpe Ratio")
    # Explicit, human-readable message -- never a silent "successful" result
    # computed from too little data.
    assert result.get("error") is not None
    assert "Insufficient" in result["error"] or "insufficient" in result["error"]
    assert "validation" not in result
    # Still returns a well-formed (fallback) result rather than crashing.
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-6 or not result["weights"]


def test_infeasible_constraints_are_explicit_via_error_code():
    n = len(PRICES.columns)
    result = run_optimization(
        PRICES, method="Maximum Sharpe Ratio", min_weight=0.30, max_weight=1.0,
    )
    assert result.get("error_code") == "infeasible_min_weight"
    assert result.get("error") is not None
    assert str(n) in result["error"]


# ── i18n parity for every new methodology key ────────────────────────────
def test_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "opt_methodology_title", "opt_methodology_subtitle",
        "opt_methodology_return_label", "opt_methodology_return_desc",
        "opt_methodology_covariance_label", "opt_methodology_covariance_desc",
        "opt_methodology_history_label", "opt_methodology_history_value",
        "opt_methodology_optimizer_label", "opt_methodology_optimizer_desc",
        "opt_methodology_short_note_on", "opt_methodology_short_note_off",
        "opt_methodology_backtest_label", "opt_methodology_backtest_value",
        "opt_methodology_backtest_desc",
        "opt_methodology_validation_pass", "opt_methodology_validation_fail",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        # short_note_off is intentionally an empty string in both languages.
        if key.endswith("_short_note_off"):
            continue
        assert TRANSLATIONS["zh-TW"][key] != key
        assert TRANSLATIONS["en"][key] != key
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""
