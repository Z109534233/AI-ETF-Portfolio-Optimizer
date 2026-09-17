"""
Risk Analytics Module (M3)
Traceable, explicitly-labeled risk helpers layered on top of the existing
src/financial_metrics.py primitives (value_at_risk, conditional_var,
maximum_drawdown, portfolio_diagnosis, correlation_matrix) and
src/holdings.py's real holdings service. Nothing here replaces those
functions' math -- this module only adds the metadata (method/confidence/
window/provenance) and explicit "unavailable" states pages/4_Risk_Analytics.py
needs to disclose exactly what each number is and where it comes from.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.financial_metrics import (
    daily_returns, value_at_risk, conditional_var, portfolio_diagnosis,
)
from src.holdings import get_etf_holdings, itemized_holdings, STATUS_UPDATED, STATUS_CACHED

# A historical VaR/CVaR percentile needs a reasonable number of observations
# to not just be sampling noise -- below this, the page must say so
# explicitly rather than show a number computed from (e.g.) 5 days of data.
MIN_VAR_OBSERVATIONS = 30


def historical_var_cvar(prices: pd.Series, confidence: float = 0.95,
                         min_observations: int = MIN_VAR_OBSERVATIONS) -> dict:
    """Historical (empirical percentile) VaR/CVaR with full disclosure of
    method, confidence level, holding period, and lookback window --
    reuses financial_metrics.value_at_risk()/conditional_var() verbatim
    (no new estimator), and adds the sufficient-data check those two
    functions don't do on their own.

    Returns a dict with "available": False and a stated "reason" (never a
    fabricated number) when there are fewer than `min_observations` daily
    returns to estimate from.
    """
    returns = daily_returns(prices) if prices is not None else pd.Series(dtype=float)
    n = len(returns)
    if n < min_observations:
        return {
            "available": False,
            "reason": (
                f"only {n} daily return observation(s) available; at least "
                f"{min_observations} are required for a historical VaR/CVaR estimate"
            ),
            "method": "historical (empirical percentile of realized daily returns)",
            "confidence": confidence,
            "holding_period_days": 1,
        }
    return {
        "available": True,
        "method": "historical (empirical percentile of realized daily returns)",
        "confidence": confidence,
        "holding_period_days": 1,
        "window_start": str(returns.index.min().date()),
        "window_end": str(returns.index.max().date()),
        "n_observations": n,
        "var": value_at_risk(prices, confidence),
        "cvar": conditional_var(prices, confidence),
    }


def concentration_from_weights(weights: dict) -> dict:
    """Concentration/effective-holdings metrics MUST be computed from the
    canonical final portfolio weights dict -- never independently
    recomputed from a different, page-local weight source. This is a thin,
    explicit wrapper around financial_metrics.portfolio_diagnosis() -- the
    SAME function Portfolio Optimizer uses -- so a portfolio's
    concentration reads identically wherever it's shown across the app.
    """
    return portfolio_diagnosis(weights)


def holdings_overlap_matrix(tickers: List[str]) -> Dict[Tuple[str, str], dict]:
    """For every pair of ETFs, compute an Overlap Score -- sum over shared
    itemized holding tickers of min(weight_in_A, weight_in_B) -- the
    standard underlying-holdings overlap measure. This is DELIBERATELY
    distinct from ETF-level return correlation (financial_metrics.
    correlation_matrix): two ETFs can move together (high return
    correlation) with zero shared holdings (e.g. different market
    segments driven by the same macro factor), and two ETFs can share
    most of their holdings yet show lower short-term return correlation if
    their few differing positions dominate volatility. Conflating the two
    would misstate actual diversification.

    Never fabricates a number: a pair where either ETF's holdings are
    unavailable (see src/holdings.py's HoldingsSnapshot.status) gets
    "available": False with a stated reason instead of an invented score.
    """
    snapshots = {tk: get_etf_holdings(tk) for tk in tickers}
    results: Dict[Tuple[str, str], dict] = {}

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            snap_a, snap_b = snapshots[a], snapshots[b]
            items_a = itemized_holdings(snap_a) if snap_a.status in (STATUS_UPDATED, STATUS_CACHED) else []
            items_b = itemized_holdings(snap_b) if snap_b.status in (STATUS_UPDATED, STATUS_CACHED) else []
            key = (a, b)
            if not items_a or not items_b:
                results[key] = {
                    "available": False,
                    "reason": "holdings data unavailable for one or both ETFs",
                }
                continue
            weights_a = {h.holding_ticker: h.weight for h in items_a}
            weights_b = {h.holding_ticker: h.weight for h in items_b}
            shared = set(weights_a) & set(weights_b)
            overlap_score = sum(min(weights_a[t], weights_b[t]) for t in shared)
            results[key] = {
                "available": True,
                "overlap_score": float(overlap_score),
                "shared_holdings_count": len(shared),
            }
    return results


# ============================================================================
# Stress Scenarios -- explicit hypothetical vs historically-sourced provenance
# ============================================================================
# Shock magnitudes are UNCHANGED from the pre-existing values already shown
# on pages/4_Risk_Analytics.py (this module only adds provenance/notes, it
# does not recalibrate or "correct" any historical figure). "historical"
# entries are commonly-cited, approximate broad-market peak-to-trough
# declines for the named event -- not a full historical portfolio replay;
# the page still applies them via the same simple (shock x portfolio beta)
# linear approximation as every "hypothetical" entry, which is disclosed
# separately in the Methodology panel.
STRESS_SCENARIOS = {
    "equity_decline": {
        "i18n_key": "risk_scenario_equity_decline", "shock": -0.30,
        "provenance": "hypothetical",
        "note_i18n_key": "risk_scenario_note_hypothetical",
    },
    "rate_shock": {
        "i18n_key": "risk_scenario_rate_shock", "shock": -0.15,
        "provenance": "hypothetical",
        "note_i18n_key": "risk_scenario_note_hypothetical",
    },
    "high_vol": {
        "i18n_key": "risk_scenario_high_vol", "shock": -0.20,
        "provenance": "hypothetical",
        "note_i18n_key": "risk_scenario_note_hypothetical",
    },
    "defensive": {
        "i18n_key": "risk_scenario_defensive", "shock": 0.05,
        "provenance": "hypothetical",
        "note_i18n_key": "risk_scenario_note_hypothetical",
    },
    "gfc_2008": {
        "i18n_key": "risk_scenario_2008", "shock": -0.50,
        "provenance": "historical",
        "note_i18n_key": "risk_scenario_note_2008",
        # Commonly-cited broad-market peak-to-trough window (Oct 2007 --
        # Mar 2009). Used ONLY to detect whether the page's currently
        # selected start_date/end_date actually covers this event's real
        # price path (Issue #43 item B) -- never fetched automatically.
        "event_start": "2007-10-01", "event_end": "2009-03-01",
    },
    "covid_2020": {
        "i18n_key": "risk_scenario_covid", "shock": -0.34,
        "provenance": "historical",
        "note_i18n_key": "risk_scenario_note_covid",
        "event_start": "2020-02-19", "event_end": "2020-03-23",
    },
    "tech_bubble": {
        "i18n_key": "risk_scenario_tech_bubble", "shock": -0.45,
        "provenance": "historical",
        "note_i18n_key": "risk_scenario_note_tech_bubble",
        "event_start": "2000-03-01", "event_end": "2002-10-01",
    },
}


def scenario_event_covered(scenario: dict, start_date, end_date) -> bool:
    """True when `scenario`'s known historical event window (event_start/
    event_end) is fully contained within [start_date, end_date] -- i.e. the
    page's currently selected dataset actually contains that event's real
    price path. Always True for hypothetical scenarios (no event window to
    check) so callers never have to special-case them. `start_date`/
    `end_date` may be date objects or ISO strings; compared as ISO strings,
    which sort correctly for calendar dates.
    """
    event_start = scenario.get("event_start")
    event_end = scenario.get("event_end")
    if not event_start or not event_end:
        return True
    return str(start_date) <= event_start and str(end_date) >= event_end


# ============================================================================
# VaR Backtesting -- out-of-sample exception count + Kupiec Proportion-of-
# Failures (unconditional coverage) test (Issue #20 section 5B).
# ============================================================================

# A rolling VaR backtest needs enough post-window forecast days for the
# exception rate to mean anything at all -- below this, the page must say
# so explicitly rather than run a Kupiec test on (e.g.) 5 forecasts.
MIN_BACKTEST_FORECASTS = 60


def kupiec_pof_test(n_forecasts: int, n_exceptions: int, confidence: float = 0.95) -> dict:
    """Kupiec (1995) Proportion-of-Failures unconditional coverage test.

    Null hypothesis H0: the true exception probability equals
    `1 - confidence` (i.e. the VaR model is correctly calibrated at this
    confidence level). The likelihood-ratio statistic
        LR_POF = -2 * ln[ L(p) / L(pi_hat) ]
    is asymptotically chi-squared with 1 degree of freedom under H0, where
    `p = 1 - confidence` is the expected exception probability and
    `pi_hat = n_exceptions / n_forecasts` is the observed exception rate.

    Mathematically correct at the boundaries (n_exceptions == 0 or ==
    n_forecasts): the binomial log-likelihood's x*log(pi) / (n-x)*log(1-pi)
    terms are each skipped when their own coefficient (x or n-x) is zero,
    which is the correct calibration of x*log(x) -> 0 as x -> 0 -- so this
    never evaluates log(0) and never raises, and pi_hat's own likelihood
    (the unconstrained MLE) is correctly 0 (i.e. a perfect log-likelihood
    of 0 under log-probability) at the boundary.

    Returns {"lr_stat": float, "p_value": float, "exception_rate": float,
    "expected_rate": float}. The p-value only tells you how compatible the
    OBSERVED exception count is with the stated confidence level under
    H0 -- a high p-value does not "prove" the model is correct, and this
    function makes no pass/fail judgement itself; callers must state the
    null hypothesis explicitly rather than claim the model "passed".
    """
    if n_forecasts <= 0:
        raise ValueError("n_forecasts must be positive")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be strictly between 0 and 1")

    p = 1.0 - confidence
    x = n_exceptions
    n = n_forecasts
    pi_hat = x / n

    def _log_likelihood(prob: float, x: int, n: int) -> float:
        ll = 0.0
        if x > 0:
            ll += x * math.log(prob)
        if n - x > 0:
            ll += (n - x) * math.log(1.0 - prob)
        return ll

    ll_null = _log_likelihood(p, x, n)
    ll_alt = _log_likelihood(pi_hat, x, n)
    lr_stat = max(0.0, -2.0 * (ll_null - ll_alt))

    # 1-df chi-squared survival function via the regularized upper
    # incomplete gamma function Q(1/2, lr/2) -- avoids adding a scipy.stats
    # dependency for a single well-known closed form: for k=1 d.f.,
    # P(X > lr) = erfc(sqrt(lr / 2)).
    p_value = math.erfc(math.sqrt(lr_stat / 2.0))

    return {
        "lr_stat": lr_stat, "p_value": p_value,
        "exception_rate": pi_hat, "expected_rate": p,
    }


def var_exception_backtest(prices: pd.Series, confidence: float = 0.95,
                            window: int = 250,
                            min_forecasts: int = MIN_BACKTEST_FORECASTS) -> dict:
    """Out-of-sample historical VaR exception backtest with NO look-ahead:
    at every forecast date, the VaR threshold is estimated using ONLY the
    trailing `window` daily returns strictly BEFORE that date, then
    compared against that date's actually realized return one step ahead.
    An "exception" is a realized daily return more negative than the
    forecast VaR threshold.

    This is a genuinely rolling/expanding walk-forward evaluation, not a
    single in-sample percentile: the VaR estimate at forecast index i uses
    returns[i-window:i] only, so no forecast ever sees data from its own or
    a later date -- see tests/test_risk_analytics.py's no-look-ahead
    regression test.

    Returns a dict:
      available=False + reason -- when there is not enough return history
        for at least `min_forecasts` forecasts after reserving `window`
        observations to seed the first estimate.
      available=True -- method/window/confidence disclosure, n_forecasts,
        n_exceptions, expected_exceptions (= n_forecasts * (1-confidence)),
        exception_rate, backtest_start/end dates, and the Kupiec POF
        test's lr_stat/p_value/exception_rate/expected_rate (see
        kupiec_pof_test's docstring for what the p-value does and does not
        establish).
    """
    returns = daily_returns(prices) if prices is not None else pd.Series(dtype=float)
    n = len(returns)
    n_possible_forecasts = n - window

    if n_possible_forecasts < min_forecasts:
        return {
            "available": False,
            "reason": (
                f"only {max(n_possible_forecasts, 0)} out-of-sample forecast day(s) possible "
                f"with a {window}-day trailing window ({n} total daily return observations); "
                f"at least {min_forecasts} are required for a meaningful backtest"
            ),
            "method": "historical (rolling trailing window, one-step-ahead)",
            "window": window,
            "confidence": confidence,
        }

    exceptions = 0
    for i in range(window, n):
        train_window = returns.iloc[i - window:i]
        var_threshold = value_at_risk_from_returns(train_window, confidence)
        realized_return = returns.iloc[i]
        if realized_return < var_threshold:
            exceptions += 1

    n_forecasts = n_possible_forecasts
    kupiec = kupiec_pof_test(n_forecasts, exceptions, confidence)

    return {
        "available": True,
        "method": "historical (rolling trailing window, one-step-ahead forecast)",
        "window": window,
        "confidence": confidence,
        "backtest_start": str(returns.index[window].date()),
        "backtest_end": str(returns.index[-1].date()),
        "n_forecasts": n_forecasts,
        "n_exceptions": exceptions,
        "expected_exceptions": n_forecasts * (1.0 - confidence),
        "exception_rate": kupiec["exception_rate"],
        "kupiec_lr_stat": kupiec["lr_stat"],
        "kupiec_p_value": kupiec["p_value"],
    }


def value_at_risk_from_returns(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR percentile computed directly from an already-realized
    daily-return series (rather than a price series) -- the same empirical-
    percentile method as financial_metrics.value_at_risk(), factored out so
    var_exception_backtest() can apply it to a TRAILING WINDOW of returns
    at each forecast date without re-deriving returns from a resliced price
    series each time."""
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))
