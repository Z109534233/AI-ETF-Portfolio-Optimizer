"""
Investment Simulator -- stale-result regression tests (Issue #20 release-gate
review: the ML page's Priority-0 stale-result fix, applied elsewhere too).

The Monte Carlo ("Future Projection") result previously had no
input-fingerprint guard -- changing a sidebar widget (e.g. switching the
market scenario) without clicking "Run Simulation" again left the OLD
result on screen with no warning that it no longer matched the displayed
inputs. This mirrors the exact bug class fixed for Machine Learning in
Issue #20 section 6A.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.simulator import (
    conservative_portfolio_return, simulate_investment, goal_attainment_analysis,
)


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _run_monte_carlo_once(lang="en"):
    at = _apptest_from_file("pages/3_Investment_Simulator.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
    assert run_btn is not None, "expected the Run Simulation button in Future Projection mode"
    run_btn.click()
    at.run()
    assert at.exception == []
    return at


def _stale_warning_shown(at) -> bool:
    warnings = "\n".join(w.value for w in at.warning)
    return "changed" in warnings.lower() or "變更" in warnings


def test_monte_carlo_no_stale_warning_immediately_after_run():
    at = _run_monte_carlo_once()
    assert not _stale_warning_shown(at)


def test_monte_carlo_stale_warning_on_scenario_change_without_rerun():
    at = _run_monte_carlo_once()
    selects = [s for s in at.selectbox if s.key == "sim_market_scenario_choice"]
    assert selects, "expected the Market Scenario selectbox (default assumption source with no current_portfolio)"
    other_options = [o for o in selects[0].options if o != selects[0].value]
    assert other_options
    selects[0].set_value(other_options[0])
    at.run()
    assert at.exception == []
    assert _stale_warning_shown(at), "changing the market scenario without re-running must show the stale-result warning"


def test_monte_carlo_stale_warning_clears_after_rerun():
    at = _run_monte_carlo_once()
    selects = [s for s in at.selectbox if s.key == "sim_market_scenario_choice"]
    other_options = [o for o in selects[0].options if o != selects[0].value]
    selects[0].set_value(other_options[0])
    at.run()
    assert _stale_warning_shown(at)

    run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
    run_btn.click()
    at.run()
    assert at.exception == []
    assert not _stale_warning_shown(at), "after re-running, the stale-result warning must clear"



def test_conservative_portfolio_return_does_not_reuse_optimizer_expected_return():
    # 50% equity at 6.5% + 50% fixed income at 3.5% = 5.0%.
    value = conservative_portfolio_return({"VOO": 0.50, "BND": 0.50})
    assert value == pytest.approx(0.05)


def test_conservative_portfolio_return_ignores_zero_weight_assets():
    value = conservative_portfolio_return({"VOO": 1.0, "BND": 0.0, "TLT": 0.0})
    assert value == pytest.approx(0.065)



def test_monte_carlo_supports_annual_contributions_for_goal_planner():
    result = simulate_investment(
        initial_investment=100.0,
        monthly_contribution=10.0,
        annual_contribution=120.0,
        years=2,
        annual_return=0.0,
        annual_volatility=0.0,
        annual_fee=0.0,
        inflation_rate=0.0,
        n_simulations=20,
        seed=42,
    )
    # 100 initial + 24*10 monthly + 2*120 annual = 580.
    assert result["summary"]["total_contributed"] == pytest.approx(580.0)
    assert result["summary"]["median_final"] == pytest.approx(580.0)



def test_goal_attainment_default_demo_has_meaningful_nonzero_distribution():
    scenarios = [(0.045, 0.07), (0.065, 0.12), (0.085, 0.17)]
    shares = []
    for annual_return, annual_volatility in scenarios:
        result = goal_attainment_analysis(
            initial_investment=10_000.0,
            monthly_contribution=5_000.0,
            annual_contribution=0.0,
            years=30,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            target_amount=5_000_000.0,
            n_simulations=5_000,
            seed=42,
        )
        shares.append(result["target_share"])

    # Demo defaults should teach distributional differences rather than show
    # three identical-looking 0.0% figures.
    assert all(0.0 < share < 1.0 for share in shares)
    assert shares[0] < shares[1] < shares[2]


def test_goal_attainment_80pct_monthly_amount_reaches_about_80pct_of_same_paths():
    base = goal_attainment_analysis(
        initial_investment=10_000.0,
        monthly_contribution=5_000.0,
        annual_contribution=0.0,
        years=30,
        annual_return=0.065,
        annual_volatility=0.12,
        target_amount=5_000_000.0,
        target_path_share=0.80,
        n_simulations=5_000,
        seed=42,
    )
    required = base["monthly_for_target_path_share"]
    check = goal_attainment_analysis(
        initial_investment=10_000.0,
        monthly_contribution=required,
        annual_contribution=0.0,
        years=30,
        annual_return=0.065,
        annual_volatility=0.12,
        target_amount=5_000_000.0,
        target_path_share=0.80,
        n_simulations=5_000,
        seed=42,
    )
    assert 0.79 <= check["target_share"] <= 0.81


def test_formula_based_reference_contribution_is_not_mislabeled_as_50pct_threshold():
    # The deterministic contribution solver ignores volatility drag/fee, so
    # under a stochastic simulation its target share need not be 50%.
    result = goal_attainment_analysis(
        initial_investment=10_000.0,
        monthly_contribution=5_000.0,
        annual_contribution=0.0,
        years=30,
        annual_return=0.065,
        annual_volatility=0.12,
        target_amount=5_000_000.0,
        reference_monthly_contribution=4_623.93,
        n_simulations=5_000,
        seed=42,
    )
    assert result["reference_target_share"] is not None
    assert result["reference_target_share"] < 0.50
