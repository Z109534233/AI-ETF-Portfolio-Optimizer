"""
Technical Indicators Module
Implements common technical analysis indicators for ETF price data.
"""

import pandas as pd
import numpy as np


def sma(prices: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=window, min_periods=1).mean()


def ema(prices: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    Returns values between 0 and 100.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    Returns DataFrame with columns: MACD, Signal, Histogram.
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "Signal": signal_line,
        "Histogram": histogram
    })


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: Middle, Upper, Lower.
    """
    middle = sma(prices, window)
    std = prices.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({
        "Middle": middle,
        "Upper": upper,
        "Lower": lower
    })


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Price momentum: current price / price n periods ago - 1."""
    return prices / prices.shift(period) - 1


def rolling_volatility(prices: pd.Series, window: int = 21,
                       periods_per_year: int = 252) -> pd.Series:
    """Annualized rolling volatility."""
    returns = prices.pct_change()
    return returns.rolling(window=window, min_periods=5).std() * np.sqrt(periods_per_year)


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return volume.rolling(window=window, min_periods=1).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def add_all_indicators(df: pd.DataFrame, price_col: str = "Close",
                       volume_col: str = None) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    Expects a DataFrame with at least a price column.
    """
    result = df.copy()
    prices = result[price_col]

    result["SMA_20"] = sma(prices, 20)
    result["SMA_50"] = sma(prices, 50)
    result["EMA_12"] = ema(prices, 12)
    result["EMA_26"] = ema(prices, 26)
    result["RSI_14"] = rsi(prices, 14)

    macd_df = macd(prices)
    result["MACD"] = macd_df["MACD"]
    result["MACD_Signal"] = macd_df["Signal"]
    result["MACD_Hist"] = macd_df["Histogram"]

    bb = bollinger_bands(prices)
    result["BB_Upper"] = bb["Upper"]
    result["BB_Middle"] = bb["Middle"]
    result["BB_Lower"] = bb["Lower"]

    result["Momentum_10"] = momentum(prices, 10)
    result["Rolling_Vol_21"] = rolling_volatility(prices, 21)

    if volume_col and volume_col in result.columns:
        result["Volume_SMA_20"] = volume_sma(result[volume_col], 20)

    return result


def create_ml_features(prices: pd.Series, volume: pd.Series = None) -> pd.DataFrame:
    """
    Create a feature DataFrame suitable for machine learning.
    Uses lagged returns, moving averages, RSI, MACD, momentum, and rolling volatility.
    """
    features = pd.DataFrame(index=prices.index)

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        features[f"Return_Lag_{lag}"] = prices.pct_change(lag)

    # Moving averages (ratio to current price)
    for window in [5, 10, 20, 50]:
        features[f"SMA_{window}_Ratio"] = prices / sma(prices, window) - 1

    # EMA ratios
    features["EMA_12_Ratio"] = prices / ema(prices, 12) - 1
    features["EMA_26_Ratio"] = prices / ema(prices, 26) - 1

    # RSI
    features["RSI_14"] = rsi(prices, 14) / 100.0

    # MACD
    macd_df = macd(prices)
    features["MACD_Norm"] = macd_df["MACD"] / prices
    features["MACD_Signal_Norm"] = macd_df["Signal"] / prices
    features["MACD_Hist_Norm"] = macd_df["Histogram"] / prices

    # Momentum
    for period in [5, 10, 20]:
        features[f"Momentum_{period}"] = momentum(prices, period)

    # Rolling volatility
    features["Rolling_Vol_10"] = rolling_volatility(prices, 10)
    features["Rolling_Vol_21"] = rolling_volatility(prices, 21)

    # Bollinger Band position
    bb = bollinger_bands(prices)
    bb_range = (bb["Upper"] - bb["Lower"]).replace(0, np.nan)
    features["BB_Position"] = (prices - bb["Lower"]) / bb_range

    # Volume features
    if volume is not None and not volume.empty:
        features["Volume_Change"] = volume.pct_change()
        features["Volume_SMA_Ratio"] = volume / volume_sma(volume, 20) - 1

    return features.replace([np.inf, -np.inf], np.nan)
