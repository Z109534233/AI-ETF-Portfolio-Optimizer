"""
Data Cleaner Module
Handles data validation, cleaning, and preprocessing for ETF price data.
"""

import pandas as pd
import numpy as np


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate price data.
    - Remove columns (tickers) with no valid data at all
    - Remove rows where all values are NaN
    - Forward-fill then back-fill missing values
    - Remove duplicate indices
    - Ensure positive prices
    """
    if df.empty:
        return df

    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]

    # Sort by date
    df = df.sort_index()

    # Drop columns (tickers) that have no valid data whatsoever. A column
    # that is 100% NaN (e.g. a ticker that failed to download) would
    # otherwise survive ffill()/bfill() completely unchanged -- there is
    # nothing to fill from -- and silently poison every downstream
    # computation that relies on this DataFrame's column count matching
    # its actual data (e.g. covariance matrices).
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return df

    # Remove rows where all values are NaN
    df = df.dropna(how="all")

    # Forward fill then back fill
    df = df.ffill().bfill()

    # Remove negative or zero prices
    df = df.where(df > 0, other=np.nan).ffill().bfill()

    return df


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Compute returns from price data.
    method: 'simple' for percentage returns, 'log' for log returns
    """
    if prices.empty:
        return pd.DataFrame()

    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()

    return returns


def resample_prices(prices: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    """
    Resample price data to a given frequency.
    frequency: 'D' daily, 'W' weekly, 'ME' monthly end, 'QE' quarterly end, 'YE' yearly end
    """
    if prices.empty:
        return prices

    # Map user-friendly names to pandas offset aliases
    freq_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
        "Yearly": "YE",
        "D": "D",
        "W": "W",
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "ME": "ME",
        "QE": "QE",
        "YE": "YE",
    }
    freq = freq_map.get(frequency, "D")
    try:
        return prices.resample(freq).last().dropna(how="all")
    except Exception:
        return prices


def align_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """Align two DataFrames to share the same index."""
    combined_index = df1.index.intersection(df2.index)
    return df1.loc[combined_index], df2.loc[combined_index]


def normalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Normalize prices to a common base value (e.g., 100)."""
    if prices.empty:
        return prices
    first_valid = prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else 1.0)
    return (prices / first_valid) * base


def remove_outliers(returns: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
    """Remove extreme outliers from return series using z-score method."""
    if returns.empty:
        return returns
    z_scores = (returns - returns.mean()) / returns.std()
    mask = z_scores.abs() <= z_threshold
    return returns.where(mask)


def get_common_date_range(prices: pd.DataFrame) -> tuple:
    """Return (common_start, common_end) -- the widest date range over
    which EVERY column has at least one valid (non-NaN) price.
    common_start = the LATEST "first valid date" across columns (a later
    ETF inception date pushes this later, per market-data reliability
    fix section 7 -- a late-inception ETF is NOT "unavailable", the
    analysis range just needs to start from its inception onward).
    common_end = the EARLIEST "last valid date" across columns.

    Returns (None, None) if `prices` has no columns, or if ANY column has
    ZERO valid data anywhere (a fully-failed ticker) -- callers must treat
    that as "no common range exists," not silently compute a range that
    ignores the missing column entirely (a column of all-NaN produces NaT
    for both its own first/last valid date, which pandas' max()/min()
    skip by default -- so naively taking max()/min() across columns would
    otherwise silently return a "common range" that never actually
    included the missing ticker at all).
    """
    if prices.empty or prices.shape[1] == 0:
        return None, None
    first_valid = []
    last_valid = []
    for col in prices.columns:
        fv = prices[col].first_valid_index()
        lv = prices[col].last_valid_index()
        if fv is None or lv is None:
            return None, None
        first_valid.append(fv)
        last_valid.append(lv)
    start = max(first_valid)
    end = min(last_valid)
    if start > end:
        return None, None
    return start, end


def validate_data_quality(prices: pd.DataFrame) -> dict:
    """Return a data quality report for the price DataFrame."""
    if prices.empty:
        return {}

    report = {}
    for col in prices.columns:
        series = prices[col]
        report[col] = {
            "total_rows": len(series),
            "missing_values": series.isna().sum(),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "start_date": series.dropna().index.min().strftime("%Y-%m-%d") if not series.dropna().empty else "N/A",
            "end_date": series.dropna().index.max().strftime("%Y-%m-%d") if not series.dropna().empty else "N/A",
            "min_price": round(series.min(), 2),
            "max_price": round(series.max(), 2),
        }
    return report
