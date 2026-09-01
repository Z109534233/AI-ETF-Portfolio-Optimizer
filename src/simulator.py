"""
Investment Simulator Module
Monte Carlo simulation for long-term investment projections.
"""

import numpy as np
import pandas as pd


MARKET_SCENARIOS = {
    "Bull Market": {"return": 0.15, "volatility": 0.12},
    "Base Case": {"return": 0.10, "volatility": 0.15},
    "Bear Market": {"return": 0.02, "volatility": 0.25},
    "Sideways Market": {"return": 0.04, "volatility": 0.10},
}


def simulate_investment(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_return: float,
    annual_volatility: float,
    inflation_rate: float = 0.025,
    annual_fee: float = 0.001,
    n_simulations: int = 1000,
    seed: int = 42
) -> dict:
    """
    Run Monte Carlo simulation for long-term investment growth.

    Returns a dict with:
    - paths: DataFrame of simulation paths (shape: months x n_simulations)
    - summary: dict with key statistics
    - annual_table: DataFrame with yearly balance summary
    """
    np.random.seed(seed)
    months = years * 12
    monthly_return = (1 + annual_return) ** (1 / 12) - 1
    monthly_vol = annual_volatility / np.sqrt(12)
    monthly_fee = (1 + annual_fee) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1

    # Simulate paths
    paths = np.zeros((months + 1, n_simulations))
    paths[0, :] = initial_investment

    for t in range(1, months + 1):
        random_returns = np.random.normal(monthly_return - monthly_fee, monthly_vol, n_simulations)
        paths[t, :] = paths[t - 1, :] * (1 + random_returns) + monthly_contribution

    # Inflation-adjusted paths
    inflation_factors = np.array([(1 + monthly_inflation) ** t for t in range(months + 1)])
    real_paths = paths / inflation_factors[:, np.newaxis]

    final_values = paths[-1, :]
    real_final_values = real_paths[-1, :]

    # Total contributed capital
    total_contributed = initial_investment + monthly_contribution * months

    summary = {
        "median_final": float(np.median(final_values)),
        "mean_final": float(np.mean(final_values)),
        "optimistic_final": float(np.percentile(final_values, 90)),
        "pessimistic_final": float(np.percentile(final_values, 10)),
        "real_median_final": float(np.median(real_final_values)),
        "total_contributed": total_contributed,
        "median_gain": float(np.median(final_values)) - total_contributed,
        "probability_profit": float(np.mean(final_values > total_contributed)),
        "probability_double": float(np.mean(final_values > 2 * initial_investment)),
    }

    # Annual table (median path)
    median_path_idx = np.argsort(final_values)[n_simulations // 2]
    median_path = paths[:, median_path_idx]

    annual_rows = []
    for yr in range(years + 1):
        month_idx = yr * 12
        contributed = initial_investment + monthly_contribution * month_idx
        value = median_path[month_idx]
        gain = value - contributed
        real_value = value / ((1 + monthly_inflation) ** month_idx)
        annual_rows.append({
            "Year": yr,
            "Portfolio Value": round(value, 2),
            "Total Contributed": round(contributed, 2),
            "Investment Gain": round(gain, 2),
            "Real Value (Inflation-Adj.)": round(real_value, 2),
            "Return %": round((value / contributed - 1) * 100, 2) if contributed > 0 else 0.0,
        })

    annual_table = pd.DataFrame(annual_rows)

    # Create paths DataFrame (sample for performance)
    sample_n = min(n_simulations, 200)
    sample_indices = np.random.choice(n_simulations, sample_n, replace=False)
    date_index = pd.date_range(start="today", periods=months + 1, freq="M")
    paths_df = pd.DataFrame(
        paths[:, sample_indices],
        index=date_index,
        columns=[f"Sim_{i}" for i in range(sample_n)]
    )

    return {
        "paths": paths_df,
        "summary": summary,
        "annual_table": annual_table,
        "all_final_values": final_values,
    }


def compound_growth_projection(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_return: float,
    annual_fee: float = 0.001,
    inflation_rate: float = 0.025
) -> pd.DataFrame:
    """
    Simple deterministic compound growth projection (no randomness).
    """
    months = years * 12
    monthly_return = (1 + annual_return - annual_fee) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1

    rows = []
    balance = initial_investment
    for t in range(months + 1):
        contributed = initial_investment + monthly_contribution * t
        real_balance = balance / ((1 + monthly_inflation) ** t)
        rows.append({
            "Month": t,
            "Year": t / 12,
            "Balance": round(balance, 2),
            "Contributed": round(contributed, 2),
            "Gain": round(balance - contributed, 2),
            "Real Balance": round(real_balance, 2),
        })
        if t < months:
            balance = balance * (1 + monthly_return) + monthly_contribution

    return pd.DataFrame(rows)


def find_common_data_range(prices_df: pd.DataFrame):
    """Return (common_start, common_end) -- the widest date range over
    which EVERY column in `prices_df` has a valid (non-NaN) price.

    common_start = the LATEST "first valid date" across columns (an ETF
    with a later inception date pushes this later). common_end = the
    EARLIEST "last valid date" across columns. Returns (None, None) if
    `prices_df` has no columns or any column is entirely NaN (no common
    range exists at all).
    """
    if prices_df.empty or prices_df.shape[1] == 0:
        return None, None
    first_valid = []
    last_valid = []
    for col in prices_df.columns:
        fv = prices_df[col].first_valid_index()
        lv = prices_df[col].last_valid_index()
        if fv is None or lv is None:
            return None, None
        first_valid.append(fv)
        last_valid.append(lv)
    common_start = max(first_valid)
    common_end = min(last_valid)
    if common_start > common_end:
        return None, None
    return common_start, common_end


def prepare_historical_prices(prices_df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Slice `prices_df` to [start, end] and clean it WITHOUT back-filling
    across periods before an ETF existed (unlike src/data_cleaner.py's
    clean_price_data(), which back-fills -- appropriate for portfolio
    optimization inputs but WRONG for a historical backtest, since it
    would fabricate a flat price history before an ETF's actual inception).

    Only forward-fills genuine mid-series gaps (e.g. a single missing
    trading day from a data-provider hiccup) within the requested range --
    every date in the returned DataFrame is guaranteed within [start, end]
    and every column is guaranteed to have originally had valid data at or
    before that date's neighborhood. Any row that still has a NaN after
    forward-filling (would only happen if a column has no valid data at
    all up to that point) is dropped, sorted ascending by date.
    """
    sliced = prices_df.loc[(prices_df.index >= start) & (prices_df.index <= end)].copy()
    sliced = sliced.sort_index()
    sliced = sliced.ffill()
    sliced = sliced.dropna(how="any")
    return sliced


def xirr(cash_flows) -> float:
    """Money-weighted annualized rate of return for a series of dated cash
    flows: solve r such that sum(amount_i / (1+r)^(days_i/365)) == 0.

    `cash_flows` is an iterable of (date, amount) tuples -- negative
    amounts are money invested (out of pocket), positive amounts are money
    returned (e.g. the final portfolio value, treated as a single
    liquidating cash flow at the end date).

    Returns the annualized rate as a float, or None if it cannot be solved
    robustly (fewer than 2 cash flows, or no sign change found in a wide
    [-99.99%, +1000%] bracket -- this is a real limitation of any
    root-finding approach, not swallowed silently: callers must check for
    None and show a "not available" message rather than a possibly-wrong
    number).
    """
    cash_flows = list(cash_flows)
    if len(cash_flows) < 2:
        return None
    dates_, amounts = zip(*cash_flows)
    if not any(a < 0 for a in amounts) or not any(a > 0 for a in amounts):
        return None  # XIRR is undefined without both an outflow and an inflow
    t0 = dates_[0]
    years_frac = np.array([(d - t0).days / 365.0 for d in dates_])
    amounts = np.array(amounts, dtype=float)

    def npv(r):
        return float(np.sum(amounts / (1.0 + r) ** years_frac))

    try:
        from scipy.optimize import brentq
        lo, hi = -0.9999, 10.0
        f_lo, f_hi = npv(lo), npv(hi)
        if (f_lo > 0) == (f_hi > 0):
            return None  # no sign change in the bracket -- cannot solve robustly
        return float(brentq(npv, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        return None


def historical_backtest(prices_df: pd.DataFrame, weights: dict,
                        initial_investment: float = 10000.0,
                        monthly_contribution: float = 0.0) -> dict:
    """Monthly-rebalanced historical backtest with recurring contributions.

    `prices_df` must already be date-aligned, sorted, and free of NaN in
    the return-relevant columns (see prepare_historical_prices()) --
    columns are the ACTIVE tickers only (zero-weight tickers should be
    excluded by the caller before this is called, per Round 2 spec
    section 4). `weights` maps ticker -> weight for those same active
    tickers (renormalized defensively here in case of floating-point
    drift, but should already sum to ~1).

    Mechanics (see Round 2 spec sections 6-9):
    - At the first date, `initial_investment` is allocated across tickers
      by `weights`, converted to fractional shares (section 10:
      fractional shares are assumed -- a simplification, not a real
      brokerage constraint).
    - Shares are held constant (pure mark-to-market) between rebalance
      points -- no daily rebalancing.
    - On the first trading day of every calendar month AFTER the starting
      month, the position is marked to market, `monthly_contribution` is
      added (may be 0), and the resulting total is reallocated to
      `weights` (i.e. contribution and rebalance happen together, once a
      month -- this is what "monthly rebalancing" means here; a $0
      contribution month still rebalances any drift back to target).

    Returns a dict with:
      "history": DataFrame indexed by date, columns "Portfolio Value",
          "Cumulative Contributions", "Growth Factor" (a time-weighted,
          contribution-neutral compounding index starting at 1.0 -- used
          for Best/Worst Year so a contribution-heavy month is never
          mistaken for investment growth).
      "contribution_dates": list of Timestamps where a contribution/
          rebalance occurred (excludes the initial investment date).
      "summary": dict with final_value, total_invested, gain,
          cumulative_return, max_drawdown, annualized_mwr (float or None),
          best_year / worst_year (each (year:int, return:float) or None),
          start_date, end_date.
    """
    empty = {"history": pd.DataFrame(), "contribution_dates": [], "summary": {}}
    if prices_df.empty or not weights:
        return empty

    tickers = list(prices_df.columns)
    w = np.array([weights.get(tk, 0.0) for tk in tickers], dtype=float)
    total_w = w.sum()
    if total_w <= 0:
        return empty
    w = w / total_w

    dates = prices_df.index
    n = len(dates)
    if n == 0:
        return empty
    prices = prices_df.values.astype(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        return empty

    months = pd.Series(dates).dt.to_period("M")
    first_month = months.iloc[0]
    is_new_month_start = (months.values != np.roll(months.values, 1))
    is_new_month_start[0] = False
    is_new_month_start &= (months.values != first_month)
    contribution_positions = set(np.where(is_new_month_start)[0].tolist())

    shares = (initial_investment * w) / prices[0]
    portfolio_value = np.empty(n)
    cumulative_contrib = np.empty(n)
    growth_factor = np.empty(n)
    portfolio_value[0] = initial_investment
    cumulative_contrib[0] = initial_investment
    growth_factor[0] = 1.0

    running_contrib_total = initial_investment
    last_rebalance_basis = initial_investment
    contribution_dates = []

    for i in range(1, n):
        mtm_value = float(np.dot(shares, prices[i]))
        if i in contribution_positions:
            growth_factor[i] = (
                growth_factor[i - 1] * (mtm_value / last_rebalance_basis)
                if last_rebalance_basis > 0 else growth_factor[i - 1]
            )
            running_contrib_total += monthly_contribution
            new_total = mtm_value + monthly_contribution
            if new_total > 0:
                shares = (new_total * w) / prices[i]
            portfolio_value[i] = new_total
            last_rebalance_basis = new_total
            contribution_dates.append(dates[i])
        else:
            prev_value = portfolio_value[i - 1]
            growth_factor[i] = (
                growth_factor[i - 1] * (mtm_value / prev_value) if prev_value > 0 else growth_factor[i - 1]
            )
            portfolio_value[i] = mtm_value
        cumulative_contrib[i] = running_contrib_total

    history = pd.DataFrame({
        "Portfolio Value": portfolio_value,
        "Cumulative Contributions": cumulative_contrib,
        "Growth Factor": growth_factor,
    }, index=dates)

    final_value = float(portfolio_value[-1])
    total_invested = float(cumulative_contrib[-1])
    gain = final_value - total_invested
    cumulative_return = gain / total_invested if total_invested > 0 else 0.0

    from src.financial_metrics import maximum_drawdown
    mdd = maximum_drawdown(pd.Series(portfolio_value, index=dates))

    cash_flows = [(dates[0], -float(initial_investment))]
    for cd in contribution_dates:
        cash_flows.append((cd, -float(monthly_contribution)))
    cash_flows.append((dates[-1], final_value))
    annualized_mwr = xirr(cash_flows)

    gf_series = pd.Series(growth_factor, index=dates)
    year_end = gf_series.groupby(gf_series.index.year).last()
    year_start = gf_series.groupby(gf_series.index.year).first()
    yearly_returns = {}
    prev_end = None
    for yr in sorted(year_end.index):
        basis = prev_end if prev_end is not None else year_start.loc[yr]
        if basis and basis > 0:
            yearly_returns[int(yr)] = float(year_end.loc[yr] / basis - 1.0)
        prev_end = year_end.loc[yr]
    best_year = max(yearly_returns, key=yearly_returns.get) if yearly_returns else None
    worst_year = min(yearly_returns, key=yearly_returns.get) if yearly_returns else None

    summary = {
        "final_value": final_value,
        "total_invested": total_invested,
        "gain": gain,
        "cumulative_return": cumulative_return,
        "max_drawdown": mdd,
        "annualized_mwr": annualized_mwr,
        "num_contributions": len(contribution_dates),
        "best_year": (best_year, yearly_returns[best_year]) if best_year is not None else None,
        "worst_year": (worst_year, yearly_returns[worst_year]) if worst_year is not None else None,
        "start_date": dates[0],
        "end_date": dates[-1],
    }
    return {"history": history, "contribution_dates": contribution_dates, "summary": summary}


def scenario_comparison(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_fee: float = 0.001
) -> pd.DataFrame:
    """Compare investment outcomes across different market scenarios."""
    rows = []
    for scenario_name, params in MARKET_SCENARIOS.items():
        result = simulate_investment(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            annual_return=params["return"],
            annual_volatility=params["volatility"],
            annual_fee=annual_fee,
            n_simulations=500,
            seed=42
        )
        summary = result["summary"]
        rows.append({
            "Scenario": scenario_name,
            "Expected Return": f"{params['return']:.1%}",
            "Volatility": f"{params['volatility']:.1%}",
            "Median Final Value": f"${summary['median_final']:,.0f}",
            "Optimistic (90th)": f"${summary['optimistic_final']:,.0f}",
            "Pessimistic (10th)": f"${summary['pessimistic_final']:,.0f}",
            "Prob. of Profit": f"{summary['probability_profit']:.1%}",
        })
    return pd.DataFrame(rows)
