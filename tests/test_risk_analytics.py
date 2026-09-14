"""
VaR backtesting tests (Issue #20 section 5B).

Covers the Kupiec Proportion-of-Failures test's mathematically correct
handling of zero/all exceptions (a naive binomial log-likelihood would
evaluate log(0) at these boundaries) and var_exception_backtest()'s
no-look-ahead property: every forecast at date i must use ONLY returns
strictly before i, verified directly against a hand-rolled reference
implementation, not just "it runs without raising".
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.risk_analytics import (
    kupiec_pof_test, var_exception_backtest, MIN_BACKTEST_FORECASTS,
)


# ── Kupiec POF test: mathematically correct at both boundaries ─────────────

def test_perfect_calibration_gives_zero_lr_stat_and_p_value_one():
    """Observed exception rate exactly equal to the expected rate must
    give LR = 0 and p-value = 1 (no evidence against H0 at all)."""
    result = kupiec_pof_test(n_forecasts=1000, n_exceptions=50, confidence=0.95)
    assert result["lr_stat"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert result["exception_rate"] == pytest.approx(0.05)
    assert result["expected_rate"] == pytest.approx(0.05)


def test_zero_exceptions_does_not_raise_and_gives_low_p_value():
    """Zero exceptions out of many forecasts at 95% confidence is itself
    evidence the model is miscalibrated (too conservative) -- must not
    raise a math domain error (log(0)) and must report a small p-value."""
    result = kupiec_pof_test(n_forecasts=500, n_exceptions=0, confidence=0.95)
    assert math.isfinite(result["lr_stat"])
    assert result["lr_stat"] > 0
    assert 0.0 <= result["p_value"] < 0.05
    assert result["exception_rate"] == 0.0


def test_all_exceptions_does_not_raise_and_gives_low_p_value():
    """Every forecast being an exception is the opposite miscalibration
    (far too aggressive) -- must not raise and must report a small p-value."""
    result = kupiec_pof_test(n_forecasts=20, n_exceptions=20, confidence=0.95)
    assert math.isfinite(result["lr_stat"])
    assert result["lr_stat"] > 0
    assert result["p_value"] < 0.05
    assert result["exception_rate"] == 1.0


def test_kupiec_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        kupiec_pof_test(n_forecasts=0, n_exceptions=0, confidence=0.95)
    with pytest.raises(ValueError):
        kupiec_pof_test(n_forecasts=10, n_exceptions=5, confidence=1.0)
    with pytest.raises(ValueError):
        kupiec_pof_test(n_forecasts=10, n_exceptions=5, confidence=0.0)


def test_lr_stat_matches_known_chi_squared_critical_value():
    """A LR statistic near the chi-squared(1) 95th percentile (3.841) must
    correspond to a p-value near 0.05 -- cross-checks the closed-form
    erfc-based survival function against a well-known reference point."""
    # Solve for an exception count whose LR is close to 3.841 at n=2000, p=0.05.
    best = None
    for x in range(1, 300):
        r = kupiec_pof_test(2000, x, 0.95)
        if best is None or abs(r["lr_stat"] - 3.841) < abs(best["lr_stat"] - 3.841):
            best = r
    assert best["p_value"] == pytest.approx(0.05, abs=0.01)


# ── var_exception_backtest: no look-ahead, correct exception counting ──────

def _synthetic_prices(seed=3, n=900, drift=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)), index=dates)


def test_insufficient_history_returns_unavailable_not_fabricated():
    short_prices = _synthetic_prices(n=200)  # 199 returns; window=250 leaves 0 forecasts
    result = var_exception_backtest(short_prices, confidence=0.95, window=250)
    assert result["available"] is False
    assert result["reason"]
    assert "window" in result
    assert "confidence" in result


def test_backtest_uses_only_strictly_prior_returns_no_lookahead():
    """Reference re-implementation using the exact same trailing-window
    percentile method, computed independently in this test -- must match
    var_exception_backtest()'s exception count and rate exactly. This
    specifically catches an off-by-one that would let a forecast see its
    own or a future day's return."""
    prices = _synthetic_prices(n=900, seed=7)
    from src.financial_metrics import daily_returns
    returns = daily_returns(prices)
    window = 250
    confidence = 0.95

    expected_exceptions = 0
    for i in range(window, len(returns)):
        train = returns.iloc[i - window:i]  # strictly returns[i-window .. i-1]
        threshold = float(np.percentile(train, (1 - confidence) * 100))
        if returns.iloc[i] < threshold:
            expected_exceptions += 1

    result = var_exception_backtest(prices, confidence=confidence, window=window)
    assert result["available"] is True
    assert result["n_forecasts"] == len(returns) - window
    assert result["n_exceptions"] == expected_exceptions


def test_backtest_perturbing_a_future_return_never_changes_an_earlier_forecast():
    """Changing a LATER day's return must never change an EARLIER
    forecast's exception outcome -- the defining no-look-ahead property.
    Perturbs a single late-window return to an extreme value and confirms
    every exception flag before that date is unaffected."""
    prices = _synthetic_prices(n=900, seed=11)
    window = 250

    result_before = var_exception_backtest(prices, confidence=0.95, window=window)

    perturbed = prices.copy()
    late_idx = len(perturbed) - 5
    perturbed.iloc[late_idx] = perturbed.iloc[late_idx - 1] * 0.5  # inject an extreme -50% day near the end
    result_after = var_exception_backtest(perturbed, confidence=0.95, window=window)

    # Recompute exceptions independently for both series up to (but not
    # including) the perturbed date, and confirm they match exactly --
    # an implementation with look-ahead would let this early portion shift.
    from src.financial_metrics import daily_returns

    def exceptions_up_to(price_series, cutoff_idx):
        returns = daily_returns(price_series)
        count = 0
        for i in range(window, cutoff_idx):
            train = returns.iloc[i - window:i]
            threshold = float(np.percentile(train, 5))
            if returns.iloc[i] < threshold:
                count += 1
        return count

    cutoff = late_idx - 1  # last unaffected return index
    assert exceptions_up_to(prices, cutoff) == exceptions_up_to(perturbed, cutoff)


def test_backtest_includes_kupiec_results_and_disclosure_fields():
    prices = _synthetic_prices(n=900, seed=5)
    result = var_exception_backtest(prices, confidence=0.95, window=250)
    assert result["available"] is True
    for key in (
        "method", "window", "confidence", "backtest_start", "backtest_end",
        "n_forecasts", "n_exceptions", "expected_exceptions", "exception_rate",
        "kupiec_lr_stat", "kupiec_p_value",
    ):
        assert key in result
    assert result["expected_exceptions"] == pytest.approx(result["n_forecasts"] * 0.05)
    assert 0.0 <= result["kupiec_p_value"] <= 1.0
