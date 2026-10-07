"""Out-of-sample portfolio backtesting utilities.

Weights are estimated only from data available before each rebalance date,
then held over the next period. Portfolio weights drift between rebalances,
and transaction costs are charged on turnover at each subsequent rebalance.
"""

import numpy as np
import pandas as pd

from src.portfolio_optimizer import run_optimization
from src.portfolio_statistics import common_observation_prices


DEFAULT_TRAIN_YEARS = 3
DEFAULT_REBALANCE_MONTHS = 3
DEFAULT_TRANSACTION_COST_BPS = 10.0
MIN_TRAIN_OBSERVATIONS = 252


def _rebalance_dates(index: pd.DatetimeIndex, train_years: int, rebalance_months: int) -> list:
    if len(index) < 2:
        return []
    first_target = index[0] + pd.DateOffset(years=train_years)
    pos = int(index.searchsorted(first_target, side="left"))
    if pos >= len(index):
        return []

    dates = []
    next_target = index[pos]
    while True:
        pos = int(index.searchsorted(next_target, side="left"))
        if pos >= len(index):
            break
        actual = index[pos]
        if not dates or actual > dates[-1]:
            dates.append(actual)
        next_target = actual + pd.DateOffset(months=rebalance_months)
    return dates


def _normalized_weights(weights: dict, columns: list) -> np.ndarray:
    arr = np.array([float(weights.get(c, 0.0)) for c in columns], dtype=float)
    total = arr.sum()
    if not np.isfinite(arr).all() or abs(total) < 1e-12:
        return np.full(len(columns), 1.0 / len(columns))
    return arr / total


def _normalise_risk_free_series(series) -> pd.Series:
    """Return a sorted decimal annual-rate series, or an empty Series."""
    if series is None:
        return pd.Series(dtype=float)
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if series.empty:
        return pd.Series(dtype=float)
    cleaned = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    cleaned.index = pd.to_datetime(cleaned.index)
    return cleaned[~cleaned.index.duplicated(keep="last")].sort_index()


def risk_free_rate_asof(series: pd.Series, date, fallback: float) -> tuple[float, bool]:
    """Return the latest annual risk-free rate observed on/before date.

    The boolean indicates whether a historical observation was available.
    A pre-normalized DatetimeIndex Series is handled without copying/sorting
    on every holding-day lookup.
    """
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return float(fallback), False
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_monotonic_increasing:
        series = _normalise_risk_free_series(series)
    if series.empty:
        return float(fallback), False
    pos = int(series.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    if pos < 0:
        return float(fallback), False
    return float(series.iloc[pos]), True


def historical_excess_sharpe(history: pd.DataFrame,
                             periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio using the historical risk-free series."""
    if history is None or history.empty or "Daily Return" not in history:
        return float("nan")
    returns = pd.to_numeric(history["Daily Return"], errors="coerce")
    if "Risk-Free Rate" in history:
        annual_rf = pd.to_numeric(history["Risk-Free Rate"], errors="coerce")
    else:
        annual_rf = pd.Series(0.0, index=history.index)
    frame = pd.concat([returns.rename("r"), annual_rf.rename("rf")], axis=1).dropna()
    if len(frame) < 2:
        return float("nan")
    vol = float(frame["r"].std(ddof=1) * np.sqrt(periods_per_year))
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    annual_excess = float((frame["r"] - frame["rf"] / periods_per_year).mean() * periods_per_year)
    return annual_excess / vol

def walk_forward_backtest(
    prices_df: pd.DataFrame,
    method: str,
    initial_investment: float = 10000.0,
    risk_free_rate: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
    target_return: float = None,
    covariance_estimator: str = "Ledoit-Wolf",
    train_years: int = DEFAULT_TRAIN_YEARS,
    rebalance_months: int = DEFAULT_REBALANCE_MONTHS,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    min_train_observations: int = MIN_TRAIN_OBSERVATIONS,
    risk_free_rate_series: pd.Series = None,
) -> dict:
    """Run a fixed-length rolling-window walk-forward backtest.

    For each rebalance, the optimizer receives only the trailing training
    window ending before the first held return. The resulting weights are
    held and allowed to drift until the next rebalance. One-way turnover is
    0.5 * sum(abs(new_weight - pre_trade_weight)); the first allocation is
    not charged because the comparison focuses on rebalancing costs.
    """
    columns = list(prices_df.columns)
    historical_rf = _normalise_risk_free_series(risk_free_rate_series)
    panel = common_observation_prices(prices_df[columns]) if columns else pd.DataFrame()
    if panel.empty or len(columns) < 2:
        return {"history": pd.DataFrame(), "rebalance_log": pd.DataFrame(),
                "summary": {"error": "Insufficient common-date price data."}}

    returns = panel.pct_change(fill_method=None).dropna(how="any")
    rebalance_dates = _rebalance_dates(panel.index, train_years, rebalance_months)
    if not rebalance_dates:
        return {"history": pd.DataFrame(), "rebalance_log": pd.DataFrame(),
                "summary": {"error": "Not enough history for the requested training window."}}

    value = float(initial_investment)
    current_weights = None
    history_rows = []
    log_rows = []
    total_turnover = 0.0
    total_cost = 0.0
    failures = 0
    historical_rf_rebalances = 0
    fallback_rf_rebalances = 0
    cost_rate = float(transaction_cost_bps) / 10000.0

    for i, rebalance_date in enumerate(rebalance_dates):
        train_start = rebalance_date - pd.DateOffset(years=train_years)
        train = panel.loc[(panel.index >= train_start) & (panel.index < rebalance_date)]
        if len(train) < min_train_observations:
            continue

        rebalance_rf, rf_is_historical = risk_free_rate_asof(
            historical_rf, rebalance_date, risk_free_rate
        )
        if rf_is_historical:
            historical_rf_rebalances += 1
        else:
            fallback_rf_rebalances += 1

        result = run_optimization(
            train, method=method, risk_free_rate=rebalance_rf,
            min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
            target_return=target_return, covariance_estimator=covariance_estimator,
        )
        target_weights = _normalized_weights(result.get("weights", {}), columns)
        opt_error = result.get("error")
        if opt_error:
            failures += 1

        turnover = 0.0
        transaction_cost = 0.0
        if current_weights is not None:
            turnover = float(0.5 * np.abs(target_weights - current_weights).sum())
            transaction_cost = value * turnover * cost_rate
            value -= transaction_cost
            total_turnover += turnover
            total_cost += transaction_cost
        current_weights = target_weights.copy()

        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_date is None:
            hold = returns.loc[returns.index >= rebalance_date]
        else:
            hold = returns.loc[(returns.index >= rebalance_date) & (returns.index < next_date)]

        log_rows.append({
            "Rebalance Date": rebalance_date,
            "Training Start": train.index.min(),
            "Training End": train.index.max(),
            "Turnover": turnover,
            "Transaction Cost": transaction_cost,
            "Optimizer Error": opt_error,
            "Risk-Free Rate": rebalance_rf,
            "Risk-Free Rate Source": "FRED DGS3MO historical" if rf_is_historical else "fallback assumption",
            "Weights": dict(zip(columns, target_weights.tolist())),
        })

        for dt, row in hold.iterrows():
            asset_returns = row.to_numpy(dtype=float)
            prev_value = value
            gross_return = float(np.dot(current_weights, asset_returns))
            value = value * (1.0 + gross_return)
            if not np.isfinite(value) or value <= 0:
                return {"history": pd.DataFrame(history_rows), "rebalance_log": pd.DataFrame(log_rows),
                        "summary": {"error": "Portfolio value became non-positive or non-finite."}}
            current_weights = current_weights * (1.0 + asset_returns) / (1.0 + gross_return)
            current_weights = current_weights / current_weights.sum()
            day_rf, _ = risk_free_rate_asof(historical_rf, dt, rebalance_rf)
            history_rows.append({
                "Date": dt,
                "Portfolio Value": value,
                "Daily Return": value / prev_value - 1.0,
                "Risk-Free Rate": day_rf,
            })

    history = pd.DataFrame(history_rows)
    if history.empty:
        return {"history": history, "rebalance_log": pd.DataFrame(log_rows),
                "summary": {"error": "No out-of-sample holding period could be evaluated."}}

    history = history.drop_duplicates(subset=["Date"], keep="last").set_index("Date").sort_index()
    history["Cumulative Return"] = history["Portfolio Value"] / float(initial_investment) - 1.0
    summary = {
        "error": None,
        "method": method,
        "train_years": int(train_years),
        "rebalance_months": int(rebalance_months),
        "transaction_cost_bps": float(transaction_cost_bps),
        "n_rebalances": int(len(log_rows)),
        "optimizer_failures": int(failures),
        "total_turnover": float(total_turnover),
        "transaction_cost_amount": float(total_cost),
        "historical_rf_rebalances": int(historical_rf_rebalances),
        "fallback_rf_rebalances": int(fallback_rf_rebalances),
        "risk_free_rate_source": (
            "FRED DGS3MO historical"
            if historical_rf_rebalances > 0 and fallback_rf_rebalances == 0
            else "mixed historical/fallback"
            if historical_rf_rebalances > 0
            else "fallback assumption"
        ),
        "oos_sharpe": float(historical_excess_sharpe(history)),
        "start_date": history.index.min(),
        "end_date": history.index.max(),
    }
    return {"history": history, "rebalance_log": pd.DataFrame(log_rows), "summary": summary}
