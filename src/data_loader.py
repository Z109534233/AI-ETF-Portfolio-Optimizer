"""
Data Loader Module
Downloads and caches ETF price data using yfinance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import os


DEFAULT_ETFS = [
    "VOO", "QQQ", "SPY", "VTI", "VT",
    "VXUS", "SCHD", "BND", "TLT", "GLD",
    "IWM", "XLK", "XLF", "XLV", "VNQ"
]


@st.cache_data(ttl=3600, show_spinner=False)
def download_etf_data(tickers: list, start_date: str, end_date: str, price_field: str = "Close") -> pd.DataFrame:
    """
    Download historical price data for a list of ETF tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    Falls back to sample data if download fails.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if raw.empty:
            return _generate_sample_data(tickers, start_date, end_date)

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            ticker = tickers[0]
            if price_field in raw.columns:
                df = raw[[price_field]].rename(columns={price_field: ticker})
            else:
                df = raw[["Close"]].rename(columns={"Close": ticker})
        else:
            if price_field in raw.columns.get_level_values(0):
                df = raw[price_field]
            elif "Close" in raw.columns.get_level_values(0):
                df = raw["Close"]
            else:
                df = raw.iloc[:, :len(tickers)]
                df.columns = tickers[:len(df.columns)]

        df = df.dropna(how="all")
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        st.warning(f"Could not download live data: {e}. Using sample data.")
        return _generate_sample_data(tickers, start_date, end_date)


@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_info(ticker: str) -> dict:
    """Fetch basic ETF information from yfinance."""
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        return {
            "name": info.get("longName", ticker),
            "category": info.get("category", "N/A"),
            "expense_ratio": info.get("annualReportExpenseRatio", None),
            "total_assets": info.get("totalAssets", None),
            "description": info.get("longBusinessSummary", "No description available."),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception:
        return {
            "name": ticker,
            "category": "N/A",
            "expense_ratio": None,
            "total_assets": None,
            "description": "Information unavailable.",
            "currency": "USD",
            "exchange": "N/A",
        }


def validate_ticker(ticker: str) -> bool:
    """Check if a ticker symbol is valid by attempting a short download."""
    try:
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        return not data.empty
    except Exception:
        return False


def _generate_sample_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic sample ETF data for fallback purposes."""
    np.random.seed(42)
    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) == 0:
        dates = pd.bdate_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now()
        )

    base_prices = {
        "VOO": 380.0, "QQQ": 350.0, "SPY": 420.0, "VTI": 210.0,
        "VT": 95.0, "VXUS": 55.0, "SCHD": 75.0, "BND": 74.0,
        "TLT": 95.0, "GLD": 180.0, "IWM": 195.0, "XLK": 165.0,
        "XLF": 38.0, "XLV": 135.0, "VNQ": 88.0,
    }
    annual_returns = {
        "VOO": 0.12, "QQQ": 0.15, "SPY": 0.11, "VTI": 0.12,
        "VT": 0.09, "VXUS": 0.07, "SCHD": 0.10, "BND": 0.03,
        "TLT": 0.02, "GLD": 0.06, "IWM": 0.10, "XLK": 0.18,
        "XLF": 0.09, "XLV": 0.11, "VNQ": 0.08,
    }
    volatilities = {
        "VOO": 0.15, "QQQ": 0.20, "SPY": 0.15, "VTI": 0.15,
        "VT": 0.14, "VXUS": 0.14, "SCHD": 0.13, "BND": 0.05,
        "TLT": 0.12, "GLD": 0.13, "IWM": 0.20, "XLK": 0.22,
        "XLF": 0.18, "XLV": 0.14, "VNQ": 0.18,
    }

    n = len(dates)
    data = {}
    for ticker in tickers:
        base = base_prices.get(ticker, 100.0)
        mu = annual_returns.get(ticker, 0.10) / 252
        sigma = volatilities.get(ticker, 0.15) / np.sqrt(252)
        daily_returns = np.random.normal(mu, sigma, n)
        prices = base * np.cumprod(1 + daily_returns)
        data[ticker] = prices

    df = pd.DataFrame(data, index=dates)
    return df


def load_sample_csv() -> pd.DataFrame:
    """Load sample ETF data from CSV file if available."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_etf_data.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_data(benchmark: str, start_date: str, end_date: str) -> pd.Series:
    """Download benchmark ETF data as a Series."""
    df = download_etf_data([benchmark], start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[:, 0]
