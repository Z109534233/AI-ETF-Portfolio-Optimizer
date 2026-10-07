"""Statistical estimation helpers for portfolio construction.

The optimizer uses one common observation panel for every asset so expected
returns and covariance are estimated from the same dates. This avoids
pairwise sample drift when combining markets with different holiday calendars.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PERIODS_PER_YEAR = 252
SUPPORTED_COVARIANCE_ESTIMATORS = ("Ledoit-Wolf", "Sample Covariance")


def common_observation_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a valid observed price for every selected asset."""
    if prices_df is None or prices_df.empty:
        return pd.DataFrame(columns=getattr(prices_df, "columns", None))
    panel = prices_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    panel = panel.loc[:, panel.notna().any(axis=0)]
    return panel.sort_index()


def aligned_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns from a common-date price panel.

    Returns are calculated only after intersecting observed price dates, so
    every row represents the same interval for every asset. Missing market
    holidays are not silently converted into zero returns.
    """
    panel = common_observation_prices(prices_df)
    if panel.empty:
        return pd.DataFrame(columns=getattr(prices_df, "columns", None))
    return panel.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def estimate_covariance(returns_df: pd.DataFrame, estimator: str = "Ledoit-Wolf",
                        periods_per_year: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    """Estimate an annualized covariance matrix on already-aligned returns."""
    if returns_df is None or returns_df.empty:
        cols = getattr(returns_df, "columns", [])
        return pd.DataFrame(index=cols, columns=cols, dtype=float)

    if estimator not in SUPPORTED_COVARIANCE_ESTIMATORS:
        raise ValueError(f"Unsupported covariance estimator: {estimator}")

    cols = list(returns_df.columns)
    values = returns_df.to_numpy(dtype=float)
    if estimator == "Ledoit-Wolf":
        cov = LedoitWolf().fit(values).covariance_ * periods_per_year
        return pd.DataFrame(cov, index=cols, columns=cols)

    return returns_df.cov() * periods_per_year


def estimate_moments(prices_df: pd.DataFrame, estimator: str = "Ledoit-Wolf",
                     periods_per_year: int = PERIODS_PER_YEAR):
    """Return aligned daily returns, daily means, and annualized covariance."""
    returns_df = aligned_returns(prices_df)
    if returns_df.empty:
        return returns_df, np.array([]), pd.DataFrame()
    mean_returns = returns_df.mean().to_numpy(dtype=float)
    cov = estimate_covariance(returns_df, estimator=estimator, periods_per_year=periods_per_year)
    return returns_df, mean_returns, cov
