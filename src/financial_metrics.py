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


def average_pairwise_correlation(prices_df: pd.DataFrame):
    """Average off-diagonal pairwise return correlation across every column
    of `prices_df` (Issue #41 item E -- the Efficient Frontier's dynamic
    diversification interpretation). Returns None when fewer than 2 columns
    have a computable pairwise correlation (never 0.0, which would be a
    fabricated "no correlation" claim instead of "not enough data")."""
    corr = correlation_matrix(prices_df)
    cols = corr.columns
    pairs = [
        corr.iloc[i, j]
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if pd.notna(corr.iloc[i, j])
    ]
    if not pairs:
        return None
    return float(sum(pairs) / len(pairs))


def correlation_diversification_level(avg_corr: float) -> str:
    """Bucket an average pairwise return correlation into "high"/"moderate"/
    "low" for the Efficient Frontier's diversification interpretation
    (Issue #41 item E): >=0.80 high (feasible risk-return set is
    compressed, diversification benefit from reweighting alone is
    limited), 0.40-0.79 moderate, <0.40 low (broader diversification
    opportunity). Simple PRODUCT UI buckets, not an academic or regulatory
    correlation standard -- same pattern as concentration_level() below."""
    if avg_corr >= 0.80:
        return "high"
    if avg_corr >= 0.40:
        return "moderate"
    return "low"


def covariance_matrix(prices_df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Compute annualized covariance matrix from price DataFrame.

    Two properties are guaranteed regardless of missing/misaligned data:

    1. The result is always reindexed to exactly `prices_df.columns` (both
       rows and columns), so `cov.shape == (n, n)` where
       `n == len(prices_df.columns)`. Callers that regularize the matrix
       (e.g. `cov += np.eye(n) * 1e-8`) can rely on this without checking.
       Any ticker pandas could not compute a variance for appears as a row/
       column of NaN instead of silently shrinking the matrix -- which
       previously caused a numpy shape-mismatch crash when a ticker's data
       was missing or misaligned (see pages/2_Portfolio_Optimizer.py).

    2. Rows are only dropped when *every* ticker is missing on that date
       (`how="all"`), not when *any single* ticker is missing
       (pandas' `dropna()` default, `how="any"`). With `how="any"`, one
       ticker having a single missing/misaligned date -- common with
       partial Yahoo Finance failures, since different tickers can have
       different trading calendars or gaps -- would silently wipe out that
       date's return for every OTHER ticker too, and in the worst case
       collapse the whole return series to zero rows. `DataFrame.cov()`
       already excludes NaNs pairwise per column pair, so pre-dropping
       rows with `how="any"` was unnecessary and actively harmful.
    """
    if prices_df.empty:
        return pd.DataFrame(index=prices_df.columns, columns=prices_df.columns, dtype=float)

    returns = prices_df.pct_change(fill_method=None).dropna(how="all")
    cov = returns.cov() * periods_per_year
    return cov.reindex(index=prices_df.columns, columns=prices_df.columns)


ILL_CONDITIONED_THRESHOLD = 1e10  # raw covariance condition number at/above this -> "severe"
MODERATE_CONDITION_THRESHOLD = 1e6  # at/above this (but below the severe threshold) -> "moderate"
HIGH_AVG_CORRELATION_THRESHOLD = 0.95  # avg pairwise return correlation at/above this -> "severe"
MODERATE_AVG_CORRELATION_THRESHOLD = 0.85  # at/above this (but below the severe threshold) -> "moderate"


def covariance_diagnostics(prices_df: pd.DataFrame) -> dict:
    """Numerical-robustness diagnostics of the RAW covariance matrix
    (Issue #45 item 1) -- computed from covariance_matrix() BEFORE any
    ridge/regularization term is added, so this reflects the actual data,
    not a numerically-patched version of it.

    Returns {"n_assets": int, "rank": int, "is_rank_deficient": bool,
    "condition_number": float or None, "avg_pairwise_correlation": float or
    None}. `condition_number` is None when it cannot be computed (e.g. a
    non-finite matrix) rather than a fabricated inf/nan value. A high
    condition number or near-collinear return correlation means the
    selected assets carry little independent information from each other
    -- this is described as "ill-conditioned"/"near-collinear", never
    unconditionally as "singular", since the matrix may still be technically
    invertible.
    """
    cov = covariance_matrix(prices_df)
    cov_array = cov.values
    n = cov_array.shape[0]
    avg_corr = average_pairwise_correlation(prices_df)
    if n == 0 or not np.isfinite(cov_array).all():
        return {
            "n_assets": n, "rank": 0, "is_rank_deficient": True,
            "condition_number": None, "avg_pairwise_correlation": avg_corr,
        }
    rank = int(np.linalg.matrix_rank(cov_array))
    try:
        cond = float(np.linalg.cond(cov_array))
        if not np.isfinite(cond):
            cond = None
    except np.linalg.LinAlgError:
        cond = None
    return {
        "n_assets": n,
        "rank": rank,
        "is_rank_deficient": rank < n,
        "condition_number": cond,
        "avg_pairwise_correlation": avg_corr,
    }


def covariance_diagnostics_level(diag: dict) -> str:
    """Bucket covariance_diagnostics() output into "severe"/"moderate"/"ok"
    for a Portfolio Optimizer warning/info panel. Deterministic PRODUCT UI
    thresholds (see the *_THRESHOLD constants above), not an academic or
    regulatory numerical-stability standard -- same pattern as
    concentration_level()/correlation_diversification_level() below.

    - "severe": rank-deficient, OR condition number >= ILL_CONDITIONED_THRESHOLD,
      OR average pairwise correlation >= HIGH_AVG_CORRELATION_THRESHOLD.
    - "moderate": condition number >= MODERATE_CONDITION_THRESHOLD or average
      pairwise correlation >= MODERATE_AVG_CORRELATION_THRESHOLD (but not
      "severe").
    - "ok": none of the above.
    """
    if diag.get("is_rank_deficient"):
        return "severe"
    cond = diag.get("condition_number")
    avg_corr = diag.get("avg_pairwise_correlation")
    if (cond is not None and cond >= ILL_CONDITIONED_THRESHOLD) or (
            avg_corr is not None and avg_corr >= HIGH_AVG_CORRELATION_THRESHOLD):
        return "severe"
    if (cond is not None and cond >= MODERATE_CONDITION_THRESHOLD) or (
            avg_corr is not None and avg_corr >= MODERATE_AVG_CORRELATION_THRESHOLD):
        return "moderate"
    return "ok"


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


def monthly_returns_table(prices) -> pd.DataFrame:
    """Create a year x month table of monthly returns.

    Hardened against real-world failures seen on Streamlit Cloud, where
    `requirements.txt` pins `pandas>=2.0.0` with no upper bound, so Cloud
    installs whatever the latest pandas is at deploy time:

    - `prices` must be a pandas Series (or single-column DataFrame, which
      is squeezed to a Series); any other type returns an empty DataFrame
      instead of raising.
    - A non-DatetimeIndex is coerced via `pd.to_datetime()`; a
      timezone-aware index is left as-is (`pd.to_datetime` preserves tz
      info, and `resample()` works natively on tz-aware indexes).
    - Empty input, or less than ~1 month of history, returns an empty
      DataFrame -- there is nothing to compute a monthly return from.
    - pandas >= 2.2 requires the "ME" (month-end) resample alias; the old
      "M" alias is deprecated there and removed entirely in pandas 3.0.
      We use "ME" exclusively (never "M") so this works on whatever
      pandas version Streamlit Cloud actually installs.
    - The resample/pivot logic is wrapped in try/except so any other
      unexpected failure degrades to an empty DataFrame instead of
      crashing the page -- this function is guaranteed to never raise.

    Return format is unchanged: a DataFrame indexed by Year, with one
    column per month ("Jan".."Dec") of monthly returns.
    """
    empty_result = pd.DataFrame()

    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        return empty_result

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            return empty_result
        prices = prices.iloc[:, 0]

    if prices.empty:
        return empty_result

    prices = prices.copy()

    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except Exception:
            return empty_result

    prices = prices.sort_index()

    # Need at least ~1 month of history to compute a single monthly return.
    if prices.index.max() - prices.index.min() < pd.Timedelta(days=28):
        return empty_result

    try:
        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()

        if monthly_ret.empty:
            return empty_result

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
    except Exception:
        return empty_result


# ── Portfolio Diagnosis (Round 2B-3) ─────────────────────────────────────
# Pure structural/concentration diagnostics computed directly from a
# {ticker: weight} dict -- always the SAME canonical weights already used
# by Optimization Results / Allocation Table / Efficient Frontier /
# Backtest, never a separately recalculated portfolio. No optimization
# math lives here.

ACTIVE_POSITION_TOLERANCE = 0.001  # weights <= this are floating-point noise, not real positions


def largest_position(weights: dict):
    """Return (ticker, weight) for the single largest holding, or (None, 0.0) if empty."""
    if not weights:
        return None, 0.0
    ticker = max(weights, key=weights.get)
    return ticker, float(weights[ticker])


def top_n_concentration(weights: dict, n: int = 2) -> float:
    """Sum of the n largest weights (e.g. n=2 -> Top 2 Concentration)."""
    if not weights:
        return 0.0
    sorted_weights = sorted(weights.values(), reverse=True)
    return float(sum(sorted_weights[:n]))


def herfindahl_index(weights: dict) -> float:
    """HHI = sum(w_i^2), the mathematical basis for Effective Holdings below."""
    if not weights:
        return 0.0
    return float(sum(w ** 2 for w in weights.values()))


def effective_number_of_holdings(weights: dict) -> float:
    """Inverse-HHI effective holdings count: 1 / sum(w_i^2).

    5 ETFs at 20% each -> HHI=0.20, effective holdings=5 (fully spread out).
    One ETF near 100% -> HHI approaches 1, effective holdings approaches 1
    regardless of how many ETFs are merely SELECTED. This is what separates
    "5 ETFs chosen" from "5 ETFs actually diversifying the portfolio".
    """
    hhi = herfindahl_index(weights)
    if hhi <= 0:
        return 0.0
    return float(1.0 / hhi)


def effective_number_of_disclosed_holdings(weights: dict) -> float:
    """Effective count within a partially disclosed holdings subset.

    ETF data providers often expose only top holdings, so their raw weights
    can sum to far below 100%. Applying 1 / sum(w^2) directly to those raw
    weights can produce an impossible result larger than the number of
    disclosed holdings. Normalize the positive disclosed weights to 100%
    first, then compute inverse-HHI. The result therefore describes only the
    disclosed subset; it is not the fund-wide effective number of holdings.
    """
    positive = {k: float(v) for k, v in (weights or {}).items() if float(v) > 0}
    total = sum(positive.values())
    if total <= 0:
        return 0.0
    normalized = {k: v / total for k, v in positive.items()}
    return effective_number_of_holdings(normalized)


def active_position_count(weights: dict, tolerance: float = ACTIVE_POSITION_TOLERANCE) -> int:
    """Count of holdings with a materially non-zero weight (> tolerance).

    A weight of 1e-8 left over from optimizer/floating-point noise must
    not be counted as a real position -- see ACTIVE_POSITION_TOLERANCE.
    """
    if not weights:
        return 0
    return sum(1 for w in weights.values() if w > tolerance)


def concentration_level(largest_weight: float) -> str:
    """Bucket the largest single position into "low"/"moderate"/"high".

    These thresholds (<=30% / 30-50% / >50%) are simple PRODUCT diagnostics
    for this app's UI, not a regulatory or academic standard of any kind.
    Returns a canonical (untranslated) bucket key; callers translate it
    for display via i18n.
    """
    if largest_weight <= 0.30:
        return "low"
    if largest_weight <= 0.50:
        return "moderate"
    return "high"


def top2_concentration_status(top2_weight: float) -> str:
    """Bucket the Top-2 Concentration metric into "distributed"/"moderate"/
    "concentrated" -- an optional, compact status label (Round 2B-3 polish).

    Like concentration_level() above, these thresholds (<=50% / 50-75% /
    >75%) are simple PRODUCT interpretation buckets for this app's UI, not
    an academic or regulatory concentration standard. Returns a canonical
    (untranslated) bucket key; callers translate it for display via i18n.
    """
    if top2_weight <= 0.50:
        return "distributed"
    if top2_weight <= 0.75:
        return "moderate"
    return "concentrated"


def diagnosis_case(diag: dict) -> str:
    """Classify a portfolio_diagnosis() result into one of three
    deterministic, rule-based summary cases:

    "concentrated" -- largest position > 50% AND the effective holdings
        count sits substantially below the selected ETF count (< 70% of
        it) -- i.e. the portfolio LOOKS diversified by ETF count but
        isn't, structurally.
    "balanced"     -- largest position is in the "low" concentration
        bucket (<= 30%).
    "moderate"     -- everything else (moderate concentration, or high
        concentration without a meaningful effective-holdings gap, e.g.
        very few ETFs were selected to begin with).

    Purely rule-based on already-calculated weight structure -- no
    generative text, no investment recommendation, no reference to
    strategy name/goal/risk tolerance (see Round 2B-3 spec sections 8, 13, 14).
    """
    selected = diag.get("selected_holdings", 0)
    effective = diag.get("effective_holdings", 0.0)
    largest = diag.get("largest_weight", 0.0)
    has_effective_gap = selected > 0 and effective < selected * 0.7
    if largest > 0.50 and has_effective_gap:
        return "concentrated"
    if diag.get("concentration_level") == "low":
        return "balanced"
    return "moderate"


def portfolio_diagnosis(weights: dict, tolerance: float = ACTIVE_POSITION_TOLERANCE) -> dict:
    """Bundle all Portfolio Diagnosis metrics for a single canonical weights dict."""
    ticker, weight = largest_position(weights)
    diag = {
        "largest_ticker": ticker,
        "largest_weight": weight,
        "top2_concentration": top_n_concentration(weights, 2),
        "hhi": herfindahl_index(weights),
        "effective_holdings": effective_number_of_holdings(weights),
        "selected_holdings": len(weights),
        "active_holdings": active_position_count(weights, tolerance),
        "concentration_level": concentration_level(weight),
    }
    diag["top2_status"] = top2_concentration_status(diag["top2_concentration"])
    diag["case"] = diagnosis_case(diag)
    return diag
