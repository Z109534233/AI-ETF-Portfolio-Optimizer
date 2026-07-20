"""
Financial Metrics Module
Implements standard portfolio and ETF performance metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats


def daily_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily percentage returns."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log daily returns."""
    return np.log(prices / prices.shift(1)).dropna()


def cumulative_return(prices: pd.Series) -> pd.Series:
    """Compute cumulative return series (starts at 0)."""
    returns = daily_returns(prices)
    return (1 + returns).cumprod() - 1


def total_return(prices: pd.Series) -> float:
    """Total return from first to last price."""
    if prices.empty or prices.iloc[0] == 0:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def annualized_return(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return using CAGR formula."""
    if len(prices) < 2:
        return 0.0
    n_years = len(prices) / periods_per_year
    if n_years <= 0 or prices.iloc[0] <= 0:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1


def annualized_volatility(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of daily returns."""
    returns = daily_returns(prices)
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.05, periods_per_year: int = 252) -> float:
    """Sharpe Ratio: (annualized return - risk-free rate) / annualized volatility."""
    ann_ret = annualized_return(prices, periods_per_year)
    ann_vol = annualized_volatility(prices, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(prices: pd.Series, risk_free_rate: float = 0.05, periods_per_year: int = 252) -> float:
    """Sortino Ratio: uses downside deviation instead of total volatility."""
    returns = daily_returns(prices)
    if returns.empty:
        return 0.0
    ann_ret = annualized_return(prices, periods_per_year)
    downside = downside_deviation(prices, risk_free_rate, periods_per_year)
    if downside == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside


def maximum_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown: largest peak-to-trough decline."""
    if prices.empty:
        return 0.0
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Return the full drawdown time series."""
    if prices.empty:
        return pd.Series(dtype=float)
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


def calmar_ratio(prices: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar Ratio: annualized return / |maximum drawdown|."""
    ann_ret = annualized_return(prices, periods_per_year)
    mdd = abs(maximum_drawdown(prices))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def beta(prices: pd.Series, benchmark_prices: pd.Series) -> float:
    """Beta relative to a benchmark."""
    ret = daily_returns(prices)
    bench_ret = daily_returns(benchmark_prices)
    common = ret.index.intersection(bench_ret.index)
    if len(common) < 10:
        return 1.0
    ret_c = ret.loc[common]
    bench_c = bench_ret.loc[common]
    cov = np.cov(ret_c, bench_c)
    if cov[1, 1] == 0:
        return 1.0
    return float(cov[0, 1] / cov[1, 1])


def alpha(prices: pd.Series, benchmark_prices: pd.Series,
          risk_free_rate: float = 0.05, periods_per_year: int = 252) -> float:
    """Jensen's Alpha."""
    b = beta(prices, benchmark_prices)
    ann_ret = annualized_return(prices, periods_per_year)
    bench_ann_ret = annualized_return(benchmark_prices, periods_per_year)
    return ann_ret - (risk_free_rate + b * (bench_ann_ret - risk_free_rate))


def value_at_risk(prices: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (negative number = loss)."""
    returns = daily_returns(prices)
    if returns.empty:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(prices: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean of returns below VaR threshold."""
    returns = daily_returns(prices)
    if returns.empty:
        return 0.0
    var = value_at_risk(prices, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def downside_deviation(prices: pd.Series, risk_free_rate: float = 0.05,
                       periods_per_year: int = 252) -> float:
    """Annualized downside deviation (semi-deviation below risk-free rate)."""
    returns = daily_returns(prices)
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    downside = returns[returns < daily_rf] - daily_rf
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def tracking_error(prices: pd.Series, benchmark_prices: pd.Series,
                   periods_per_year: int = 252) -> float:
    """Annualized tracking error vs benchmark."""
    ret = daily_returns(prices)
    bench_ret = daily_returns(benchmark_prices)
    common = ret.index.intersection(bench_ret.index)
    if len(common) < 5:
        return 0.0
    diff = ret.loc[common] - bench_ret.loc[common]
    return float(diff.std() * np.sqrt(periods_per_year))


def information_ratio(prices: pd.Series, benchmark_prices: pd.Series,
                      periods_per_year: int = 252) -> float:
    """Information Ratio: excess return / tracking error."""
    ann_ret = annualized_return(prices, periods_per_year)
    bench_ann_ret = annualized_return(benchmark_prices, periods_per_year)
    te = tracking_error(prices, benchmark_prices, periods_per_year)
    if te == 0:
        return 0.0
    return (ann_ret - bench_ann_ret) / te


def correlation_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix from price DataFrame."""
    returns = prices_df.pct_change().dropna()
    return returns.corr()


def covariance_matrix(prices_df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Compute annualized covariance matrix from price DataFrame."""
    returns = prices_df.pct_change().dropna()
    return returns.cov() * periods_per_year


def diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Diversification Ratio: weighted average volatility / portfolio volatility.
    Higher = more diversified.
    """
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, individual_vols)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    if port_vol == 0:
        return 1.0
    return float(weighted_avg_vol / port_vol)


def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray,
                     periods_per_year: int = 252) -> float:
    """Expected annualized portfolio return."""
    return float(np.dot(weights, mean_returns) * periods_per_year)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Annualized portfolio volatility (already annualized cov_matrix)."""
    return float(np.sqrt(weights @ cov_matrix @ weights))


def portfolio_sharpe(weights: np.ndarray, mean_returns: np.ndarray,
                     cov_matrix: np.ndarray, risk_free_rate: float = 0.05) -> float:
    """Portfolio Sharpe Ratio."""
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


def compute_all_metrics(prices: pd.Series, benchmark_prices: pd.Series = None,
                        risk_free_rate: float = 0.05) -> dict:
    """Compute a comprehensive set of metrics for a single ETF/portfolio."""
    metrics = {
        "Total Return": f"{total_return(prices):.2%}",
        "Annualized Return": f"{annualized_return(prices):.2%}",
        "Annualized Volatility": f"{annualized_volatility(prices):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(prices, risk_free_rate):.2f}",
        "Sortino Ratio": f"{sortino_ratio(prices, risk_free_rate):.2f}",
        "Maximum Drawdown": f"{maximum_drawdown(prices):.2%}",
        "Calmar Ratio": f"{calmar_ratio(prices):.2f}",
        "VaR (95%)": f"{value_at_risk(prices, 0.95):.2%}",
        "CVaR (95%)": f"{conditional_var(prices, 0.95):.2%}",
        "Downside Deviation": f"{downside_deviation(prices, risk_free_rate):.2%}",
    }
    if benchmark_prices is not None and not benchmark_prices.empty:
        metrics["Beta"] = f"{beta(prices, benchmark_prices):.2f}"
        metrics["Alpha"] = f"{alpha(prices, benchmark_prices, risk_free_rate):.2%}"
        metrics["Tracking Error"] = f"{tracking_error(prices, benchmark_prices):.2%}"
        metrics["Information Ratio"] = f"{information_ratio(prices, benchmark_prices):.2f}"
    return metrics


def monthly_returns_table(prices: pd.Series) -> pd.DataFrame:
    """Create a year x month table of monthly returns."""
    monthly = prices.resample("M").last()
    monthly_ret = monthly.pct_change().dropna()
    df = pd.DataFrame({
        "Year": monthly_ret.index.year,
        "Month": monthly_ret.index.month,
        "Return": monthly_ret.values
    })
    pivot = df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="first")
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]
    return pivot
