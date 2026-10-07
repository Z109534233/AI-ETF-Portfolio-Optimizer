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
) -> dict:
    """Run an expanding calendar walk-forward backtest.

    For each rebalance, the optimizer receives only the trailing training
    window ending before the first held return. The resulting weights are
    held and allowed to drift until the next rebalance. One-way turnover is
    0.5 * sum(abs(new_weight - pre_trade_weight)); the first allocation is
    not charged because the comparison focuses on rebalancing costs.
    """
    columns = list(prices_df.columns)
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
    cost_rate = float(transaction_cost_bps) / 10000.0

    for i, rebalance_date in enumerate(rebalance_dates):
        train_start = rebalance_date - pd.DateOffset(years=train_years)
        train = panel.loc[(panel.index >= train_start) & (panel.index < rebalance_date)]
        if len(train) < min_train_observations:
            continue

        result = run_optimization(
            train, method=method, risk_free_rate=risk_free_rate,
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
            history_rows.append({
                "Date": dt,
                "Portfolio Value": value,
                "Daily Return": value / prev_value - 1.0,
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
        "start_date": history.index.min(),
        "end_date": history.index.max(),
    }
    return {"history": history, "rebalance_log": pd.DataFrame(log_rows), "summary": summary}
