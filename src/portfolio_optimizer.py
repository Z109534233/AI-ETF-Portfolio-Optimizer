"""
Portfolio Optimizer Module
Implements mean-variance optimization, risk parity, and Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
from src.financial_metrics import (
    portfolio_return, portfolio_volatility, portfolio_sharpe,
    covariance_matrix, diversification_ratio
)


def equal_weight(tickers: list) -> np.ndarray:
    """Equal weight allocation."""
    n = len(tickers)
    return np.array([1.0 / n] * n)


def validate_weight_constraints(n: int, min_weight: float = 0.0, max_weight: float = 1.0,
                                 allow_short: bool = False, tol: float = 1e-9):
    """Pre-flight feasibility check for min/max per-asset weight bounds against
    `sum(weights) == 1`, run BEFORE any optimizer call so an infeasible
    configuration (e.g. 5 ETFs x 30% minimum > 100%) produces a clear error
    instead of a silent scipy convergence failure downstream.

    Returns (is_feasible, error_code) where error_code is one of
    "infeasible_min_weight" / "infeasible_max_weight" / None.

    Note: max_weight infeasibility (n * max_weight < 1) applies regardless
    of allow_short, since bounds are symmetric (+/-max_weight) either way
    and the achievable sum is still capped at n * max_weight. min_weight
    infeasibility only applies when NOT allow_short, matching the existing
    bounds logic in optimize_max_sharpe()/optimize_min_volatility() below,
    where min_weight is intentionally not applied once short selling is on.
    """
    if n <= 0:
        return True, None
    if n * max_weight < 1.0 - tol:
        return False, "infeasible_max_weight"
    if not allow_short and n * min_weight > 1.0 + tol:
        return False, "infeasible_min_weight"
    return True, None


def _clean_weights(weights: np.ndarray, allow_short: bool, epsilon: float = 1e-6) -> np.ndarray:
    """Zero out weights that are effectively noise (e.g. 1e-16 instead of
    exactly 0, a byproduct of SLSQP's numerical solve) and re-normalize so
    sum(weights) stays exactly 1 -- without this, the allocation table can
    show a meaningless "0.00%" row that isn't really a materially different
    position from zero. Only clips magnitude noise; does not alter any
    weight that is a real, material allocation.
    """
    cleaned = np.where(np.abs(weights) < epsilon, 0.0, weights)
    total = cleaned.sum()
    if abs(total) > epsilon:
        cleaned = cleaned / total
    return cleaned


def optimize_max_sharpe(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        risk_free_rate: float = 0.05,
                        min_weight: float = 0.0, max_weight: float = 1.0,
                        allow_short: bool = False):
    """Maximize Sharpe Ratio using SciPy optimization.

    Returns (weights, success). `success=False` means SLSQP did not
    converge -- `weights` is still a safe equal-weight fallback so callers
    never crash, but callers MUST surface `success` to the user rather than
    silently presenting the fallback as if it were the real optimized
    result (see run_optimization()).
    """
    n = len(mean_returns)
    init_weights = np.array([1.0 / n] * n)

    def neg_sharpe(weights):
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        if vol == 0:
            return 0.0
        return -(ret - risk_free_rate) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if allow_short:
        bounds = tuple((-max_weight, max_weight) for _ in range(n))
    else:
        bounds = tuple((min_weight, max_weight) for _ in range(n))

    result = minimize(
        neg_sharpe, init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success:
        weights = np.array(result.x)
        weights = np.clip(weights, 0 if not allow_short else -max_weight, max_weight)
        total = weights.sum()
        if total != 0:
            weights /= total
        return _clean_weights(weights, allow_short), True
    return init_weights, False


def optimize_min_volatility(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             min_weight: float = 0.0, max_weight: float = 1.0,
                             allow_short: bool = False):
    """Minimize portfolio volatility.

    Returns (weights, success) -- see optimize_max_sharpe() docstring for
    what `success=False` means and why callers must not hide it.
    """
    n = len(mean_returns)
    init_weights = np.array([1.0 / n] * n)

    def port_vol(weights):
        return portfolio_volatility(weights, cov_matrix)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if allow_short:
        bounds = tuple((-max_weight, max_weight) for _ in range(n))
    else:
        bounds = tuple((min_weight, max_weight) for _ in range(n))

    result = minimize(
        port_vol, init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success:
        weights = np.array(result.x)
        weights = np.clip(weights, 0 if not allow_short else -max_weight, max_weight)
        total = weights.sum()
        if total != 0:
            weights /= total
        return _clean_weights(weights, allow_short), True
    return init_weights, False


def optimize_target_return(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                            target_return: float,
                            min_weight: float = 0.0, max_weight: float = 1.0) -> np.ndarray:
    """Minimize volatility subject to a target return constraint."""
    n = len(mean_returns)
    init_weights = np.array([1.0 / n] * n)

    def port_vol(weights):
        return portfolio_volatility(weights, cov_matrix)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w, mean_returns) - target_return}
    ]
    bounds = tuple((min_weight, max_weight) for _ in range(n))

    result = minimize(
        port_vol, init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    if result.success:
        weights = np.array(result.x)
        weights = np.clip(weights, 0, max_weight)
        total = weights.sum()
        if total != 0:
            weights /= total
        return weights
    return init_weights


def optimize_risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Risk Parity: each asset contributes equally to portfolio risk.
    """
    n = cov_matrix.shape[0]
    init_weights = np.array([1.0 / n] * n)

    def risk_parity_objective(weights):
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        if port_vol == 0:
            return 0.0
        marginal_contrib = cov_matrix @ weights / port_vol
        risk_contrib = weights * marginal_contrib
        target_risk = port_vol / n
        return float(np.sum((risk_contrib - target_risk) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.001, 1.0) for _ in range(n))

    result = minimize(
        risk_parity_objective, init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10}
    )

    if result.success:
        weights = np.abs(result.x)
        return weights / weights.sum()
    return init_weights


def monte_carlo_simulation(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                            n_simulations: int = 5000,
                            risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate Monte Carlo portfolio simulations.
    Returns DataFrame with columns: Return, Volatility, Sharpe, Weights.
    """
    n_assets = len(mean_returns)
    results = []

    for _ in range(n_simulations):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        results.append({
            "Return": ret,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Weights": weights.tolist()
        })

    return pd.DataFrame(results)


def compute_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                n_points: int = 50,
                                min_weight: float = 0.0,
                                max_weight: float = 1.0) -> pd.DataFrame:
    """
    Compute the efficient frontier by solving for minimum volatility at each target return.
    """
    min_ret = float(np.min(mean_returns) * 252)
    max_ret = float(np.max(mean_returns) * 252)
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_points = []
    for target in target_returns:
        weights = optimize_target_return(mean_returns, cov_matrix, target, min_weight, max_weight)
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        frontier_points.append({"Return": ret, "Volatility": vol})

    return pd.DataFrame(frontier_points)


def run_optimization(prices_df: pd.DataFrame, method: str,
                     risk_free_rate: float = 0.05,
                     min_weight: float = 0.0, max_weight: float = 1.0,
                     allow_short: bool = False,
                     target_return: float = None) -> dict:
    """
    Main optimization function. Returns weights and portfolio metrics.
    """
    tickers = prices_df.columns.tolist()
    n = len(tickers)

    def _fallback_result(error_message: str, error_code: str = None) -> dict:
        weights = np.array([1.0 / n] * n) if n else np.array([])
        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "method": method,
            "error": error_message,
            "error_code": error_code,
        }

    if prices_df.empty or n == 0:
        return _fallback_result("No price data available for optimization.")

    # Pre-flight feasibility check -- run BEFORE touching price data or the
    # optimizer at all, since this is a pure configuration check (e.g. 5
    # ETFs x 30% minimum > 100% is infeasible regardless of what the price
    # history looks like). Without this, an infeasible config just makes
    # SLSQP fail to converge downstream, which used to be silently papered
    # over with an equal-weight fallback -- see optimize_max_sharpe() /
    # optimize_min_volatility() above for why that was unsafe.
    feasible, infeasible_code = validate_weight_constraints(n, min_weight, max_weight, allow_short)
    if not feasible:
        if infeasible_code == "infeasible_min_weight":
            msg = (f"{n} ETFs x {min_weight:.0%} minimum weight exceeds 100% "
                   "-- lower the minimum weight or select more ETFs.")
        else:
            msg = (f"{n} ETFs x {max_weight:.0%} maximum weight cannot reach 100% "
                   "-- raise the maximum weight or select fewer ETFs.")
        return _fallback_result(msg, error_code=infeasible_code)

    # dropna(how="all") -- not the pandas default how="any" -- so that one
    # ticker missing a single date cannot wipe out that date for every
    # other ticker too (see covariance_matrix() for the full rationale).
    returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")

    if returns_df.empty or len(returns_df) < 10:
        return _fallback_result("Insufficient data for optimization.")

    mean_returns = returns_df.mean().values
    cov = covariance_matrix(prices_df)
    cov_array = cov.values.copy()

    # covariance_matrix() guarantees cov_array.shape == (n, n), so this
    # regularization step can never hit a shape mismatch. It can still
    # contain NaN, though, if a ticker has no valid overlapping return
    # data at all (e.g. it failed to download and slipped through) --
    # check for that explicitly rather than silently doing NaN arithmetic.
    if cov_array.shape != (n, n):
        return _fallback_result(
            f"Covariance matrix shape {cov_array.shape} does not match "
            f"{n} tickers; aborting optimization."
        )

    if not np.isfinite(cov_array).all():
        # A NaN on the DIAGONAL means that specific ticker has no valid
        # data at all (its own variance couldn't be computed) -- that
        # ticker is unambiguously the problem. A NaN only OFF the diagonal
        # means two otherwise-valid tickers simply have no overlapping
        # trading dates between them, which isn't any single ticker's
        # "fault". We distinguish these so the error message names the
        # actual culprit instead of blaming every ticker whenever one is bad.
        diag_nan = ~np.isfinite(np.diag(cov_array))
        if diag_nan.any():
            bad_tickers = [tickers[i] for i, bad in enumerate(diag_nan) if bad]
            return _fallback_result(
                f"No valid price data for: {', '.join(bad_tickers)}. "
                "Remove these tickers or widen the date range."
            )
        return _fallback_result(
            "Some selected ETFs have no overlapping trading dates with each "
            "other. Widen the date range or choose ETFs with more shared "
            "trading history."
        )

    # Regularize covariance matrix to avoid singularity
    cov_array += np.eye(n) * 1e-8

    try:
        optimizer_failed = False
        if method == "Equal Weight":
            weights = _clean_weights(equal_weight(tickers), allow_short=False)
        elif method == "Maximum Sharpe Ratio":
            weights, converged = optimize_max_sharpe(mean_returns, cov_array, risk_free_rate,
                                                      min_weight, max_weight, allow_short)
            optimizer_failed = not converged
        elif method == "Minimum Volatility":
            weights, converged = optimize_min_volatility(mean_returns, cov_array,
                                                          min_weight, max_weight, allow_short)
            optimizer_failed = not converged
        elif method == "Target Return":
            tr = target_return if target_return is not None else float(np.mean(mean_returns) * 252)
            weights = optimize_target_return(mean_returns, cov_array, tr, min_weight, max_weight)
        elif method == "Risk Parity":
            weights = optimize_risk_parity(cov_array)
        else:
            weights = equal_weight(tickers)

        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_array)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        div_ratio = diversification_ratio(weights, cov_array)

        # `optimizer_failed` (Max Sharpe / Min Volatility only): SLSQP did
        # not converge, so `weights` is the safe equal-weight fallback from
        # optimize_max_sharpe()/optimize_min_volatility(). Per Round 2A
        # requirements, this must NEVER be presented silently as if it were
        # the real optimized result -- error/error_code are always set so
        # the page can show a clear, translated warning alongside the
        # (still equal-weight) numbers.
        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": float(ret),
            "expected_volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "diversification_ratio": float(div_ratio),
            "method": method,
            "error": (
                f"{method} optimization did not converge for this data/settings; "
                "showing Equal Weight as a fallback."
            ) if optimizer_failed else None,
            "error_code": "optimizer_failed" if optimizer_failed else None,
        }

    except Exception as e:
        weights = np.array([1.0 / n] * n)
        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "method": method,
            "error": str(e),
            "error_code": "unexpected_error",
        }


def backtest_portfolio(prices_df: pd.DataFrame, weights: dict,
                       initial_investment: float = 10000.0) -> pd.DataFrame:
    """
    Backtest a portfolio with given weights.
    Returns a DataFrame with portfolio value over time.
    """
    tickers = list(weights.keys())
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return pd.DataFrame()

    w_array = np.array([weights[t] for t in available])
    w_array = w_array / w_array.sum()

    prices = prices_df[available].dropna()
    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change().dropna()
    port_returns = returns @ w_array
    portfolio_value = initial_investment * (1 + port_returns).cumprod()

    result = pd.DataFrame({
        "Portfolio Value": portfolio_value,
        "Daily Return": port_returns,
        "Cumulative Return": (1 + port_returns).cumprod() - 1
    })
    return result
