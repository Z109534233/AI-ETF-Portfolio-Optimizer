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

from typing import Dict, List, Tuple

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
    },
    "covid_2020": {
        "i18n_key": "risk_scenario_covid", "shock": -0.34,
        "provenance": "historical",
        "note_i18n_key": "risk_scenario_note_covid",
    },
    "tech_bubble": {
        "i18n_key": "risk_scenario_tech_bubble", "shock": -0.45,
        "provenance": "historical",
        "note_i18n_key": "risk_scenario_note_tech_bubble",
    },
}
