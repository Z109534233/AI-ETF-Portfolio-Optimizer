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


def optimize_max_sharpe(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        risk_free_rate: float = 0.05,
                        min_weight: float = 0.0, max_weight: float = 1.0,
                        allow_short: bool = False) -> np.ndarray:
    """Maximize Sharpe Ratio using SciPy optimization."""
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
        return weights
    return init_weights


def optimize_min_volatility(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             min_weight: float = 0.0, max_weight: float = 1.0,
                             allow_short: bool = False) -> np.ndarray:
    """Minimize portfolio volatility."""
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
        return weights
    return init_weights


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
    returns_df = prices_df.pct_change().dropna()

    if returns_df.empty or len(returns_df) < 10:
        n = len(tickers)
        weights = np.array([1.0 / n] * n)
        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "method": method,
            "error": "Insufficient data for optimization."
        }

    mean_returns = returns_df.mean().values
    cov = covariance_matrix(prices_df)
    cov_array = cov.values.copy()

    # Regularize covariance matrix to avoid singularity
    cov_array += np.eye(len(tickers)) * 1e-8

    try:
        if method == "Equal Weight":
            weights = equal_weight(tickers)
        elif method == "Maximum Sharpe Ratio":
            weights = optimize_max_sharpe(mean_returns, cov_array, risk_free_rate,
                                          min_weight, max_weight, allow_short)
        elif method == "Minimum Volatility":
            weights = optimize_min_volatility(mean_returns, cov_array,
                                              min_weight, max_weight, allow_short)
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

        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": float(ret),
            "expected_volatility": float(vol),
            "sharpe_ratio": float(sharpe),
            "diversification_ratio": float(div_ratio),
            "method": method,
            "error": None
        }

    except Exception as e:
        n = len(tickers)
        weights = np.array([1.0 / n] * n)
        return {
            "weights": dict(zip(tickers, weights)),
            "expected_return": 0.0,
            "expected_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "method": method,
            "error": str(e)
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
