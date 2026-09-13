"""
M2 -- Simulation & Backtest Methodology: deterministic tests.

Pure pytest `assert` tests (no network, no Streamlit AppTest) covering:
  - src/methodology.py's SIMULATION_METHODOLOGY metadata
  - src/simulator.py's simulate_investment() actually implements the
    documented Monte Carlo formula (independently re-derived and compared
    value-for-value against the engine's own output, not just spot-checked)
  - the documented inflation treatment (real = nominal / compounded
    monthly inflation, applied AFTER the nominal simulation)
  - the documented "monthly" rebalancing convention in historical_backtest()
  - XIRR against a known closed-form cash-flow example
  - no pre-inception backfill in prepare_historical_prices()
  - i18n parity (zh-TW / en) for every new sim_methodology_* key

Note: tests/test_portfolio_optimizer.py already has extensive HIST-*/SIM-*
coverage of historical_backtest()/xirr()/AppTest page wiring -- these tests
deliberately focus on what that suite does NOT already cover: that the
Monte Carlo engine's actual formula matches what SIMULATION_METHODOLOGY
claims, and the new methodology UI/i18n added in this task.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.methodology import SIMULATION_METHODOLOGY
from src.simulator import (
    simulate_investment, historical_backtest, find_common_data_range,
    prepare_historical_prices, xirr,
)
from src.i18n import TRANSLATIONS


# ── Methodology metadata ─────────────────────────────────────────────────
def test_methodology_metadata_matches_actual_implementation():
    m = SIMULATION_METHODOLOGY
    assert "Normal" in m["monte_carlo"]["distribution"]
    assert m["monte_carlo"]["timestep"] == "monthly"
    assert "AFTER" in m["monte_carlo"]["contribution_timing"]
    assert "subtracted" in m["monte_carlo"]["fees"]
    assert "inflation" in m["monte_carlo"]["inflation_treatment"].lower()
    assert m["rebalancing"]["convention"] == "monthly"
    assert m["backtest_label"]["type"] == "Fixed-Allocation Historical Backtest"
    assert "look-ahead bias" in m["backtest_label"]["look_ahead_bias"]
    assert "Brent" in m["contributions_and_returns"]["xirr"]


# ── Monte Carlo formula matches the documented methodology ──────────────
def test_monte_carlo_formula_matches_engine_output_exactly():
    """Independently re-derive the documented formula (Normal monthly
    returns; fee subtracted from the mean BEFORE the draw; contribution
    added AFTER the return is applied) using the same seed, and confirm it
    produces the EXACT same final values as simulate_investment(). This
    fails the moment the implementation and SIMULATION_METHODOLOGY diverge.
    """
    seed = 123
    n_sims = 30
    years = 2
    initial_investment = 10000.0
    monthly_contribution = 200.0
    annual_return = 0.08
    annual_volatility = 0.15
    annual_fee = 0.01

    result = simulate_investment(
        initial_investment=initial_investment, monthly_contribution=monthly_contribution,
        years=years, annual_return=annual_return, annual_volatility=annual_volatility,
        inflation_rate=0.03, annual_fee=annual_fee, n_simulations=n_sims, seed=seed,
    )

    np.random.seed(seed)
    months = years * 12
    monthly_return = (1 + annual_return) ** (1 / 12) - 1
    monthly_vol = annual_volatility / np.sqrt(12)
    monthly_fee = (1 + annual_fee) ** (1 / 12) - 1
    values = np.full(n_sims, initial_investment)
    for _ in range(months):
        random_returns = np.random.normal(monthly_return - monthly_fee, monthly_vol, n_sims)
        values = values * (1 + random_returns) + monthly_contribution

    assert np.allclose(values, result["all_final_values"], atol=1e-6)

    expected_total_contributed = initial_investment + monthly_contribution * months
    assert abs(result["summary"]["total_contributed"] - expected_total_contributed) < 1e-6


def test_inflation_adjustment_is_a_separate_postprocessing_step():
    """real_median_final must equal median_final divided by the compounded
    monthly inflation factor over the full horizon -- an exact relationship
    since dividing every final value by the SAME scalar preserves the
    median exactly. Confirms inflation is applied to the nominal path
    afterward, never baked into the simulated returns themselves."""
    years = 5
    inflation_rate = 0.03
    result = simulate_investment(
        initial_investment=10000.0, monthly_contribution=100.0, years=years,
        annual_return=0.07, annual_volatility=0.12, inflation_rate=inflation_rate,
        annual_fee=0.001, n_simulations=200, seed=11,
    )
    months = years * 12
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1
    inflation_factor_final = (1 + monthly_inflation) ** months
    expected_real_median = result["summary"]["median_final"] / inflation_factor_final
    assert abs(result["summary"]["real_median_final"] - expected_real_median) < 1e-6
    assert result["summary"]["real_median_final"] <= result["summary"]["median_final"] + 1e-6


# ── Historical backtest rebalances monthly, as documented ────────────────
def test_historical_backtest_rebalances_monthly_as_documented():
    """Constant prices eliminate any price-path ambiguity: with zero
    appreciation, final_value must equal total_invested EXACTLY (gain=0),
    so this simultaneously proves the contribution/rebalance cadence is
    monthly (matching SIMULATION_METHODOLOGY["rebalancing"]) and that
    total_invested/gain are correctly accounted for."""
    dates = pd.bdate_range("2022-01-01", "2022-07-31")
    prices = pd.DataFrame({"A": np.full(len(dates), 100.0), "B": np.full(len(dates), 50.0)}, index=dates)
    result = historical_backtest(
        prices, {"A": 0.6, "B": 0.4}, initial_investment=50000.0, monthly_contribution=1000.0,
    )
    summary = result["summary"]

    # Jan-Jul 2022 = 7 calendar months; the first (Jan) is the initial
    # investment itself, so exactly 6 subsequent monthly rebalance events
    # are expected if the cadence is genuinely monthly.
    assert summary["num_contributions"] == 6

    expected_total_invested = 50000.0 + 1000.0 * 6
    assert abs(summary["total_invested"] - expected_total_invested) < 1e-4
    assert abs(summary["gain"]) < 1e-4
    assert abs(summary["final_value"] - expected_total_invested) < 1e-4
    assert abs(summary["gain"] - (summary["final_value"] - summary["total_invested"])) < 1e-9


# ── XIRR against a known closed-form cash-flow example ───────────────────
def test_xirr_matches_known_closed_form_example():
    t0, t1 = dt.date(2022, 1, 1), dt.date(2023, 1, 1)  # exactly 365 days
    r = xirr([(t0, -1000.0), (t1, 1100.0)])
    assert r is not None
    assert abs(r - 0.10) < 0.01


# ── No pre-inception backfill in prepare_historical_prices() ────────────
def test_prepare_historical_prices_never_backfills_pre_inception():
    dates = pd.bdate_range("2018-01-01", "2020-01-01")
    prices = pd.DataFrame(index=dates)
    prices["OLD"] = np.linspace(50, 100, len(dates))
    prices["NEW"] = np.nan
    new_start_idx = 300
    prices.iloc[new_start_idx:, prices.columns.get_loc("NEW")] = np.linspace(10, 20, len(dates) - new_start_idx)

    common_start, common_end = find_common_data_range(prices)
    assert common_start == dates[new_start_idx]

    prepared = prepare_historical_prices(prices, dates[0], dates[-1])
    assert not prepared.empty
    assert prepared.index.min() == common_start
    assert not prepared.isna().any().any()
    assert set(prepared.columns) == {"OLD", "NEW"}


# ── i18n parity for every new sim_methodology_* key ──────────────────────
def test_simulation_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "sim_methodology_title",
        "sim_methodology_mc_subtitle", "sim_methodology_mc_distribution",
        "sim_methodology_mc_contribution", "sim_methodology_mc_fees_inflation",
        "sim_methodology_mc_purpose",
        "sim_methodology_hist_subtitle", "sim_methodology_hist_rebalancing",
        "sim_methodology_hist_lookahead", "sim_methodology_hist_xirr",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key] != key
        assert TRANSLATIONS["en"][key] != key
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""
