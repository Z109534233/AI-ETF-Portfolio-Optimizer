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
