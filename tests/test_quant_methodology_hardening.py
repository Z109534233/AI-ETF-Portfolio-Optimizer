"""Regression tests for the quantitative-methodology hardening pass."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import walk_forward_backtest
from src.machine_learning import prepare_ml_dataset, time_series_split
from src.portfolio_optimizer import monte_carlo_simulation, run_optimization
from src.portfolio_statistics import aligned_returns, estimate_covariance
from src.simulator import simulate_investment


def _prices(seed=7, n_days=1500):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    common = rng.normal(0.00025, 0.006, n_days)
    data = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        r = common + rng.normal(0.0001 + i * 0.00003, 0.007 + i * 0.001, n_days)
        data[ticker] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(data, index=idx)


def test_aligned_returns_use_only_common_observed_dates():
    prices = _prices(n_days=30)
    missing_date = prices.index[10]
    prices.loc[missing_date, "BBB"] = np.nan
    returns = aligned_returns(prices)
    assert missing_date not in returns.index
    assert returns.notna().all().all()


def test_ledoit_wolf_covariance_is_symmetric_and_psd():
    returns = aligned_returns(_prices(n_days=400))
    cov = estimate_covariance(returns, estimator="Ledoit-Wolf")
    assert np.allclose(cov.values, cov.values.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() >= -1e-10


def test_optimizer_reports_common_sample_and_ledoit_wolf():
    prices = _prices(n_days=500)
    prices.loc[prices.index[50], "CCC"] = np.nan
    result = run_optimization(prices, "Minimum Volatility")
    assert result["error"] is None
    assert result["covariance_estimator"] == "Ledoit-Wolf"
    assert result["estimation_observations"] == len(aligned_returns(prices))



def test_risk_parity_failure_is_surfaced(monkeypatch):
    import src.portfolio_optimizer as optimizer

    def _failed_risk_parity(cov):
        n = cov.shape[0]
        return np.full(n, 1.0 / n), False

    monkeypatch.setattr(optimizer, "optimize_risk_parity", _failed_risk_parity)
    result = optimizer.run_optimization(_prices(n_days=500), "Risk Parity")
    assert result["error_code"] == "optimizer_failed"
    assert result["error"]
    assert "Risk Parity" in result["error"]

def test_efficient_frontier_monte_carlo_is_deterministic_by_default():
    mean = np.array([0.0004, 0.0003, 0.0002])
    cov = np.array([[0.02, 0.004, 0.002], [0.004, 0.015, 0.003], [0.002, 0.003, 0.01]])
    a = monte_carlo_simulation(mean, cov, n_simulations=50)
    b = monte_carlo_simulation(mean, cov, n_simulations=50)
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_uses_only_prior_training_data_and_charges_turnover_costs():
    prices = _prices()
    result = walk_forward_backtest(
        prices, "Equal Weight", initial_investment=10000,
        train_years=2, rebalance_months=3, transaction_cost_bps=10,
        min_train_observations=200,
    )
    history = result["history"]
    log = result["rebalance_log"]
    summary = result["summary"]
    assert not history.empty
    assert len(log) >= 2
    assert (pd.to_datetime(log["Training End"]) < pd.to_datetime(log["Rebalance Date"])).all()
    assert summary["total_turnover"] > 0
    assert summary["transaction_cost_amount"] > 0


def test_walk_forward_transaction_cost_reduces_terminal_value():
    prices = _prices()
    free = walk_forward_backtest(
        prices, "Equal Weight", train_years=2, rebalance_months=3,
        transaction_cost_bps=0, min_train_observations=200,
    )["history"]
    costly = walk_forward_backtest(
        prices, "Equal Weight", train_years=2, rebalance_months=3,
        transaction_cost_bps=50, min_train_observations=200,
    )["history"]
    assert costly["Portfolio Value"].iloc[-1] < free["Portfolio Value"].iloc[-1]



def test_ml_dataset_drops_rows_without_future_label():
    idx = pd.bdate_range("2023-01-02", periods=220)
    prices = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    X, y, used_index = prepare_ml_dataset(prices, lookahead=5)
    assert X is not None
    assert used_index.max() <= idx[-6]
    assert y.notna().all()

def test_ml_holdout_embargo_removes_lookahead_rows():
    idx = pd.bdate_range("2024-01-01", periods=100)
    X = pd.DataFrame({"x": np.arange(100)}, index=idx)
    y = pd.Series(np.arange(100) % 2, index=idx)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2, gap=5)
    assert len(X_test) == 20
    assert len(X_train) == 75
    assert X_test.index[0] == idx[80]
    assert X_train.index[-1] == idx[74]
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)


def test_arithmetic_annual_return_uses_arithmetic_monthly_mean():
    result = simulate_investment(
        initial_investment=100.0, monthly_contribution=0.0, years=1,
        annual_return=0.12, annual_volatility=0.0, inflation_rate=0.0,
        annual_fee=0.0, n_simulations=10, seed=42,
        annual_return_is_arithmetic=True,
    )
    expected = 100.0 * (1.01 ** 12)
    assert result["summary"]["median_final"] == pytest.approx(expected)
