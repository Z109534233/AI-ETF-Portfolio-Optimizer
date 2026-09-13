"""
Methodology & Disclosure Module (M1 -> M4)
Single source of truth for the platform's stated calculation methodology,
assumptions, and post-hoc validation helpers -- surfaced through compact
"Methodology & Assumptions" UI panels and exercised by deterministic tests.

Every entry here documents ONLY what the corresponding src/ module actually
does. If the implementation changes, this module must be updated in the
same change -- it must never describe a method the app doesn't implement,
and it must never silently fall out of sync with the real calculation.

No Streamlit import here on purpose: this module is pure Python/NumPy so it
can be imported and unit-tested without a Streamlit runtime, by
src/portfolio_optimizer.py (a calculation module) and by every
pages/*.py file that renders a methodology panel.
"""

import numpy as np

# ============================================================================
# M1 -- Portfolio Optimization Methodology
# ============================================================================
# Describes src/portfolio_optimizer.py + src/financial_metrics.py's
# covariance_matrix() as actually implemented -- see run_optimization(),
# optimize_max_sharpe(), and covariance_matrix() for the code this
# metadata must always match.

PORTFOLIO_OPTIMIZATION_METHODOLOGY = {
    "expected_return": {
        "price_field": "Close (split/dividend-adjusted via yfinance auto_adjust=True)",
        "return_type": "simple daily percentage returns (DataFrame.pct_change)",
        "sampling_frequency": "daily",
        "estimator": "arithmetic mean of daily returns",
        "annualization": "mean daily return * 252 trading days",
        "missing_data_handling": (
            "dates missing for every ticker are dropped; dates missing for "
            "only some tickers are kept (each pair's statistics are computed "
            "from their own overlapping dates). No ETF's price is ever "
            "forward- or back-filled before its own first valid trading date."
        ),
        "common_history_window": (
            "every selected ETF is sliced to the common valid-data window "
            "(the latest first-valid-date to the earliest last-valid-date "
            "across the selection) before optimization, so a later-inception "
            "ETF shortens the analysis window instead of being backfilled."
        ),
    },
    "covariance": {
        "estimator": "sample covariance (pandas DataFrame.cov())",
        "sampling_frequency": "daily",
        "annualization": "daily covariance matrix * 252 trading days",
        "alignment": (
            "pandas .cov() uses pairwise-complete observations per ticker "
            "pair; rows where every ticker is missing are dropped first, but "
            "no ticker is ever excluded from the matrix outright."
        ),
        "nan_handling": (
            "a ticker with zero valid overlapping data produces NaN on its "
            "own diagonal; run_optimization() detects this explicitly and "
            "reports it as an error rather than treating it as zero risk."
        ),
        "psd_stabilization": (
            "a small ridge term (identity * 1e-8) is added to the diagonal "
            "only to keep the matrix numerically solvable during "
            "optimization; this is not shrinkage of the estimator itself "
            "and does not materially change reported risk."
        ),
        "shrinkage": "none -- no Ledoit-Wolf or other shrinkage estimator is applied",
    },
    "max_sharpe": {
        "objective": (
            "maximize (portfolio_return - risk_free_rate) / portfolio_volatility, "
            "implemented as minimizing the negative Sharpe ratio"
        ),
        "solver": "scipy.optimize.minimize, method='SLSQP'",
        "risk_free_rate": (
            "user-configurable annual rate (default 5%), used directly as a "
            "flat annual rate in the Sharpe ratio numerator -- not compounded "
            "or converted to a daily rate"
        ),
        "constraints": (
            "sum(weights) == 1 (fully invested, no cash); per-asset weight "
            "bounded to [min_weight, max_weight], or [-max_weight, max_weight] "
            "if short selling is allowed"
        ),
    },
    "insufficient_history": {
        "policy": (
            "no ETF is ever silently dropped for insufficient history. If "
            "fewer than 2 ETFs have overlapping valid price data, or fewer "
            "than 20 overlapping trading days / 10 overlapping daily returns "
            "exist, the page stops with an explicit error instead of running "
            "optimization on a smaller or fabricated dataset."
        ),
    },
    "backtest_label": {
        "type": "Fixed-Allocation Historical Backtest",
        "explanation": (
            "the single set of weights produced by the CURRENT optimization "
            "run is applied unchanged across the entire historical price "
            "window to compute a hypothetical value path. It is NOT a "
            "walk-forward backtest: weights are never re-optimized at any "
            "point using only data available as of that date, so it does not "
            "reflect how the strategy would actually have been selected and "
            "rebalanced in real time."
        ),
    },
}


# ============================================================================
# M2 -- Simulation & Backtest Methodology
# ============================================================================
# Describes src/simulator.py as actually implemented -- see
# simulate_investment(), historical_backtest(), prepare_historical_prices(),
# and xirr() for the code this metadata must always match.

SIMULATION_METHODOLOGY = {
    "monte_carlo": {
        "purpose": (
            "a projection of a RANGE of plausible future outcomes under "
            "stated assumptions -- never a prediction or guarantee of what "
            "markets will actually do."
        ),
        "distribution": "independent monthly returns drawn from a Normal distribution",
        "timestep": "monthly",
        "n_simulations": "user-configurable (default 1,000; 200-5,000)",
        "contribution_timing": (
            "the monthly contribution is added to the portfolio AFTER that "
            "month's simulated return is applied, at every month of the horizon"
        ),
        "fees": (
            "an annual fee is converted to an equivalent monthly rate and "
            "subtracted directly from the mean monthly return before each "
            "random draw"
        ),
        "inflation_treatment": (
            "nominal paths are simulated first; a separate 'real' "
            "(inflation-adjusted) path is derived by dividing by a "
            "compounded monthly inflation factor -- inflation never alters "
            "the nominal simulation itself"
        ),
        "reported_percentiles": (
            "10th (pessimistic) / 50th (median) / 90th (optimistic) as "
            "headline KPIs, plus a 5/25/50/75/95 percentile table"
        ),
        "assumption_modes": {
            "Portfolio Historical Statistics": (
                "the CURRENT portfolio's own historical expected return/ "
                "volatility from Portfolio Optimizer, used as a forward "
                "assumption -- labeled as a historical estimate, never a "
                "prediction"
            ),
            "Market Scenario": (
                "one of 4 hard-coded hypothetical return/volatility pairs "
                "(Bull/Base/Bear/Sideways) -- labeled as hypothetical, not "
                "a market forecast"
            ),
            "Custom Assumptions": "user-entered return/volatility, used as-is",
        },
    },
    "historical_simulation": {
        "data": (
            "real downloaded ETF prices only -- no synthetic or fabricated "
            "returns are ever used in Historical Simulation mode"
        ),
        "start_date": (
            "constrained to the common valid-data window across every "
            "active ticker (the same get_common_date_range() logic as "
            "Portfolio Optimizer); a user-chosen start earlier than that "
            "window is clamped forward to the common start, never backfilled"
        ),
        "pre_inception_handling": (
            "prepare_historical_prices() only forward-fills genuine "
            "mid-series gaps within the requested window; any date before a "
            "ticker's own first valid price is never included, let alone filled"
        ),
    },
    "rebalancing": {
        "convention": "monthly",
        "mechanics": (
            "shares are held constant (pure mark-to-market) between "
            "rebalance points -- no daily rebalancing. On the first trading "
            "day of every calendar month after the starting month, the "
            "position is marked to market, the monthly contribution (which "
            "may be $0) is added, and the total is reallocated back to the "
            "target weights in the same step."
        ),
    },
    "contributions_and_returns": {
        "total_invested": (
            "initial_investment + sum of every monthly contribution actually "
            "made over the simulated/backtested period"
        ),
        "gain": "final portfolio value - total_invested",
        "fractional_shares": "assumed (a simplification -- not a real brokerage constraint)",
        "xirr": (
            "money-weighted annualized return solved from the ACTUAL dated "
            "cash flows (initial investment + every monthly contribution, "
            "each on its real date, plus the final value as a single "
            "liquidating inflow) via Brent's method; returns None (shown as "
            "unavailable, never a fabricated number) if it cannot be solved "
            "robustly"
        ),
    },
    "backtest_label": {
        "type": "Fixed-Allocation Historical Backtest",
        "look_ahead_bias": (
            "the weights applied throughout the ENTIRE historical window "
            "are the CURRENT Portfolio Optimizer result -- i.e. weights "
            "chosen using information that includes data from later in (or "
            "after) the very window being tested. This is look-ahead bias "
            "by construction, and is never described as walk-forward or as "
            "evidence the strategy would have been selected this way in "
            "real time."
        ),
    },
}


# ============================================================================
# M3 -- Risk Methodology
# ============================================================================
# Describes src/risk_analytics.py + the pre-existing src/financial_metrics.py
# risk functions it wraps (value_at_risk, conditional_var, maximum_drawdown,
# portfolio_diagnosis, correlation_matrix) as actually implemented.

RISK_METHODOLOGY = {
    "var_cvar": {
        "method": "historical -- empirical percentile of realized daily returns (no parametric/Monte Carlo VaR)",
        "confidence": "user-configurable via the confidence level passed to historical_var_cvar() (95% default)",
        "holding_period": "1 trading day (VaR/CVaR are computed on daily returns, not scaled to any longer horizon)",
        "window": "the full selected date range's overlapping daily returns for the current price series",
        "insufficient_data": (
            "if fewer than MIN_VAR_OBSERVATIONS (30) daily returns are "
            "available, the page shows an explicit 'unavailable' state with "
            "the observation count and requirement, never a number computed "
            "from too little data"
        ),
    },
    "max_drawdown": {
        "formula": "min over time of (price - rolling_max(price)) / rolling_max(price)",
        "verification": "computed directly from the SAME price series shown in the growth/backtest chart -- never a separately-sourced or hard-coded figure",
    },
    "concentration": {
        "source": (
            "largest-position, top-N concentration, effective-holdings (1/"
            "Herfindahl), and active-position-count are ALWAYS computed from "
            "the canonical current_portfolio weights dict (portfolio_diagnosis()) "
            "-- the same function and the same weights Portfolio Optimizer uses "
            "-- never from a separately-entered or independently-normalized "
            "weight source"
        ),
    },
    "correlation_vs_overlap": {
        "return_correlation": "Pearson correlation of each ETF pair's daily returns (financial_metrics.correlation_matrix) -- a statement about how the ETFs' PRICES have moved together",
        "holdings_overlap": (
            "sum over shared itemized holding tickers of min(weight_in_A, "
            "weight_in_B) -- a statement about how much of the ETFs' "
            "UNDERLYING PORTFOLIOS are the same securities. High return "
            "correlation does not imply holdings overlap (different segments "
            "can move together macro-economically) and low return "
            "correlation does not imply low overlap (a few differing "
            "positions can dominate short-term volatility) -- the two are "
            "never conflated or substituted for one another."
        ),
        "unavailable_handling": "a pair is marked unavailable (with a stated reason) rather than scored, whenever either ETF's underlying holdings cannot be retrieved",
    },
    "stress_scenarios": {
        "calculation": "portfolio_impact = market_shock * portfolio_beta -- a simple linear approximation, not a full historical portfolio reconstruction or factor-model replay",
        "provenance_labeling": (
            "each scenario is explicitly labeled 'hypothetical' (a stated, "
            "illustrative assumption) or 'historical' (a commonly-cited, "
            "approximate broad-market peak-to-trough decline for a named "
            "past event) -- see src/risk_analytics.py's STRESS_SCENARIOS; "
            "shock magnitudes are pre-existing values, unchanged by this task"
        ),
        "limitation": (
            "even 'historical' scenarios use the SAME linear beta-scaling as "
            "hypothetical ones -- this is never a claim that the current "
            "portfolio would have actually behaved this way during that "
            "event, only an illustrative, beta-scaled approximation"
        ),
    },
}


def validate_optimization_result(weights: dict, mean_returns, cov_matrix,
                                  reported_return: float, reported_volatility: float,
                                  reported_sharpe: float, risk_free_rate: float,
                                  min_weight: float = 0.0, max_weight: float = 1.0,
                                  allow_short: bool = False, tol: float = 1e-4) -> dict:
    """Post-optimization validation (M1 acceptance criteria #4).

    Independently re-derives return/volatility/Sharpe from `weights` +
    `mean_returns` + `cov_matrix` and checks that: (1) weights sum to 1,
    (2) every weight respects the configured bounds, and (3) the reported
    metrics are numerically consistent with the weights actually returned.

    Returns {"is_valid": bool, "issues": [str, ...]} -- an empty `issues`
    list means every check passed. Never raises; a malformed/empty
    `weights` dict simply fails the "sum to 1" check rather than crashing,
    so callers can always safely inspect the result.
    """
    issues = []
    w = np.array(list(weights.values()), dtype=float) if weights else np.array([])

    weight_sum = float(w.sum()) if w.size else 0.0
    if abs(weight_sum - 1.0) > 1e-4:
        issues.append(f"weights sum to {weight_sum:.6f}, expected 1.0")

    lower_bound = -max_weight if allow_short else min_weight
    upper_bound = max_weight
    if w.size and (np.min(w) < lower_bound - 1e-6 or np.max(w) > upper_bound + 1e-6):
        issues.append(
            f"weight(s) fall outside the configured bounds "
            f"[{lower_bound}, {upper_bound}]: min={np.min(w):.6f}, max={np.max(w):.6f}"
        )

    mean_returns_arr = np.asarray(mean_returns, dtype=float)
    cov_arr = np.asarray(cov_matrix, dtype=float)
    if w.size and w.size == mean_returns_arr.size and cov_arr.shape == (w.size, w.size):
        recomputed_return = float(np.dot(w, mean_returns_arr) * 252)
        recomputed_volatility = float(np.sqrt(max(float(w @ cov_arr @ w), 0.0)))
        recomputed_sharpe = (
            (recomputed_return - risk_free_rate) / recomputed_volatility
            if recomputed_volatility > 0 else 0.0
        )

        if abs(recomputed_return - reported_return) > tol:
            issues.append(
                f"reported expected_return {reported_return:.6f} is inconsistent "
                f"with the weights (recomputed {recomputed_return:.6f})"
            )
        if abs(recomputed_volatility - reported_volatility) > tol:
            issues.append(
                f"reported expected_volatility {reported_volatility:.6f} is "
                f"inconsistent with the weights (recomputed {recomputed_volatility:.6f})"
            )
        if abs(recomputed_sharpe - reported_sharpe) > max(tol, 1e-3):
            issues.append(
                f"reported sharpe_ratio {reported_sharpe:.6f} is inconsistent "
                f"with the weights (recomputed {recomputed_sharpe:.6f})"
            )
    else:
        issues.append("weights/mean_returns/cov_matrix shapes do not align; cannot verify metrics")

    return {"is_valid": len(issues) == 0, "issues": issues}
