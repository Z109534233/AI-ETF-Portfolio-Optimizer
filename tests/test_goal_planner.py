"""
Goal Planner -- deterministic tests (Issue #22).

Pure pytest `assert` tests (no network, no Streamlit AppTest) covering
src/goal_planner.py's future-value / annuity math, required-contribution
solver, on_track/below_target/above_target classification, income-mode
withdrawal-rate conversion, edge-case handling, scenario ordering, and
illustrative example-ETF selection against the real src/etf_database.py
universe.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.goal_planner import (
    SCENARIO_ASSUMPTIONS,
    SCENARIO_ORDER,
    WITHDRAWAL_RATE,
    ASSUMPTIONS_DISCLOSURE,
    ON_TRACK_TOLERANCE,
    future_value_of_savings,
    required_monthly_contribution,
    classify_status,
    select_example_etfs,
    build_goal_plan,
)
from src.etf_database import get_etf


# ── future_value_of_savings() ────────────────────────────────────────────
def test_fv_zero_capital_monthly_only_matches_hand_calculation():
    # Zero current capital, no annual contribution: pure ordinary-annuity
    # monthly-compounding future value.
    monthly_contribution = 500.0
    annual_return = 0.06
    n_months = 120  # 10 years

    r_m = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    expected = monthly_contribution * (((1.0 + r_m) ** n_months - 1.0) / r_m)

    actual = future_value_of_savings(0.0, monthly_contribution, annual_return, n_months)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_fv_zero_return_is_simple_addition():
    # r = 0 edge case: no growth, FV = capital + sum of contributions.
    actual = future_value_of_savings(1000.0, 100.0, 0.0, 24, annual_contribution=50.0)
    expected = 1000.0 + 100.0 * 24 + 50.0 * 2
    assert actual == pytest.approx(expected, rel=1e-9)


def test_fv_zero_months_returns_current_capital():
    assert future_value_of_savings(1234.0, 100.0, 0.07, 0) == pytest.approx(1234.0)


def test_fv_lump_sum_only_compounds_correctly():
    current_capital = 10000.0
    annual_return = 0.05
    n_months = 60  # 5 years
    expected = current_capital * (1.0 + annual_return) ** (n_months / 12.0)
    actual = future_value_of_savings(current_capital, 0.0, annual_return, n_months)
    assert actual == pytest.approx(expected, rel=1e-9)


# ── required_monthly_contribution() round-trip ───────────────────────────
@pytest.mark.parametrize("scenario_name", SCENARIO_ORDER)
def test_required_contribution_round_trips_to_target(scenario_name):
    annual_return = SCENARIO_ASSUMPTIONS[scenario_name]["expected_return"]
    current_capital = 5000.0
    annual_contribution = 200.0
    n_months = 25 * 12
    target_amount = 500000.0

    required = required_monthly_contribution(
        current_capital, target_amount, annual_return, n_months, annual_contribution,
    )
    assert required >= 0.0

    projected = future_value_of_savings(
        current_capital, required, annual_return, n_months, annual_contribution,
    )
    # Cent-level tolerance relative to a target in the hundreds of thousands.
    assert projected == pytest.approx(target_amount, abs=0.01)


def test_required_contribution_zero_when_lump_sum_already_exceeds_target():
    # Large current capital alone blows past a modest target: required
    # contribution must clamp at 0.0, never go negative.
    required = required_monthly_contribution(
        current_capital=1_000_000.0,
        target_amount=50_000.0,
        annual_return=0.06,
        n_months=120,
    )
    assert required == 0.0


def test_required_contribution_handles_zero_return():
    # r = 0 edge case for the solver too.
    n_months = 100
    target_amount = 10500.0
    current_capital = 500.0
    required = required_monthly_contribution(current_capital, target_amount, 0.0, n_months)
    projected = future_value_of_savings(current_capital, required, 0.0, n_months)
    assert projected == pytest.approx(target_amount, abs=0.01)


# ── classify_status() ────────────────────────────────────────────────────
def test_classify_status_on_track_exact_match():
    assert classify_status(1000.0, 1000.0) == "on_track"


def test_classify_status_on_track_within_tolerance_band():
    required = 1000.0
    just_inside = required * (1.0 + ON_TRACK_TOLERANCE / 2)
    assert classify_status(just_inside, required) == "on_track"


def test_classify_status_below_target():
    assert classify_status(500.0, 1000.0) == "below_target"


def test_classify_status_above_target():
    assert classify_status(2000.0, 1000.0) == "above_target"


def test_classify_status_zero_required_contribution_is_on_track_or_above():
    assert classify_status(0.0, 0.0) == "on_track"
    assert classify_status(100.0, 0.0) == "above_target"


# ── Income modes / withdrawal rate ───────────────────────────────────────
def test_monthly_income_mode_derives_implied_target_total():
    monthly_income = 3000.0
    plan = build_goal_plan(
        current_age=30, target_age=60, current_capital=0.0,
        monthly_contribution=1000.0, target_mode="monthly_income",
        target_amount=monthly_income, market_preference="United States",
        risk_tolerance="balanced", base_currency="USD",
    )
    assert plan["valid"]
    assert plan["withdrawal_rate"] == WITHDRAWAL_RATE
    expected_target = (monthly_income * 12.0) / WITHDRAWAL_RATE
    assert plan["implied_target_total"] == pytest.approx(expected_target)


def test_annual_income_mode_derives_implied_target_total():
    annual_income = 40000.0
    plan = build_goal_plan(
        current_age=25, target_age=65, current_capital=0.0,
        monthly_contribution=500.0, target_mode="annual_income",
        target_amount=annual_income, market_preference="Taiwan",
        risk_tolerance="conservative", base_currency="TWD",
    )
    assert plan["valid"]
    assert plan["withdrawal_rate"] == WITHDRAWAL_RATE
    expected_target = annual_income / WITHDRAWAL_RATE
    assert plan["implied_target_total"] == pytest.approx(expected_target)


def test_total_value_mode_has_no_withdrawal_rate():
    plan = build_goal_plan(
        current_age=40, target_age=65, current_capital=10000.0,
        monthly_contribution=500.0, target_mode="total_value",
        target_amount=1_000_000.0, market_preference="United Kingdom",
        risk_tolerance="aggressive", base_currency="GBP",
    )
    assert plan["valid"]
    assert plan["withdrawal_rate"] is None
    assert plan["implied_target_total"] == pytest.approx(1_000_000.0)


# ── Edge cases ────────────────────────────────────────────────────────────
def test_current_age_ge_target_age_is_invalid():
    plan = build_goal_plan(
        current_age=60, target_age=60, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=100000.0,
    )
    assert plan["valid"] is False
    assert plan["errors"]
    assert plan["scenarios"] == {}


def test_current_age_greater_than_target_age_is_invalid():
    plan = build_goal_plan(
        current_age=70, target_age=60, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=100000.0,
    )
    assert plan["valid"] is False


def test_non_positive_target_amount_is_invalid():
    plan = build_goal_plan(
        current_age=30, target_age=60, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=0.0,
    )
    assert plan["valid"] is False
    assert any("target_amount" in e for e in plan["errors"])

    plan_neg = build_goal_plan(
        current_age=30, target_age=60, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=-500.0,
    )
    assert plan_neg["valid"] is False


def test_zero_horizon_override_is_invalid():
    plan = build_goal_plan(
        current_age=30, target_age=60, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=100000.0, horizon_years=0,
    )
    assert plan["valid"] is False


def test_horizon_years_override_takes_precedence():
    plan = build_goal_plan(
        current_age=30, target_age=31,  # age diff = 1 year
        current_capital=0.0, monthly_contribution=100.0,
        target_mode="total_value", target_amount=100000.0,
        horizon_years=20,
    )
    assert plan["valid"]
    assert plan["horizon_years"] == pytest.approx(20.0)
    assert plan["horizon_months"] == 240


def test_negative_contribution_is_invalid():
    plan = build_goal_plan(
        current_age=30, target_age=60, current_capital=0.0,
        monthly_contribution=-10.0, target_mode="total_value",
        target_amount=100000.0,
    )
    assert plan["valid"] is False


def test_already_exceeding_target_gives_non_negative_required_contribution():
    plan = build_goal_plan(
        current_age=50, target_age=60, current_capital=5_000_000.0,
        monthly_contribution=1000.0, target_mode="total_value",
        target_amount=100000.0,
    )
    assert plan["valid"]
    for scenario in plan["scenarios"].values():
        assert scenario["required_monthly_contribution"] >= 0.0
        assert scenario["status"] in ("on_track", "above_target")


# ── Scenario presence & ordering ─────────────────────────────────────────
def test_three_scenarios_always_present_with_increasing_return_and_volatility():
    plan = build_goal_plan(
        current_age=35, target_age=65, current_capital=20000.0,
        monthly_contribution=800.0, target_mode="total_value",
        target_amount=750000.0, market_preference="Mixed",
        risk_tolerance="balanced", base_currency="USD",
    )
    assert plan["valid"]
    scenarios = plan["scenarios"]
    assert set(scenarios.keys()) == {"conservative", "balanced", "aggressive"}

    assert (
        scenarios["conservative"]["expected_return"]
        < scenarios["balanced"]["expected_return"]
        < scenarios["aggressive"]["expected_return"]
    )
    assert (
        scenarios["conservative"]["expected_volatility"]
        < scenarios["balanced"]["expected_volatility"]
        < scenarios["aggressive"]["expected_volatility"]
    )

    # Matches the module's declared assumption table exactly.
    for name in SCENARIO_ORDER:
        assert scenarios[name]["expected_return"] == SCENARIO_ASSUMPTIONS[name]["expected_return"]
        assert scenarios[name]["expected_volatility"] == SCENARIO_ASSUMPTIONS[name]["expected_volatility"]


def test_assumptions_disclosure_present_and_mentions_key_caveats():
    plan = build_goal_plan(
        current_age=35, target_age=65, current_capital=0.0,
        monthly_contribution=100.0, target_mode="total_value",
        target_amount=100000.0,
    )
    disclosure = plan["assumptions_disclosure"]
    assert disclosure == ASSUMPTIONS_DISCLOSURE
    lowered = disclosure.lower()
    assert "not" in lowered and ("guarantee" in lowered or "advice" in lowered)
    assert "rebalanc" in lowered


# ── Example ETF selection ────────────────────────────────────────────────
@pytest.mark.parametrize("market", ["Taiwan", "United States", "United Kingdom"])
@pytest.mark.parametrize("risk", ["conservative", "balanced", "aggressive"])
def test_example_etfs_are_genuine_tickers_in_the_database(market, risk):
    result = select_example_etfs(market, risk)
    assert isinstance(result["tickers"], list)
    assert isinstance(result["selection_logic"], str) and result["selection_logic"]
    for ticker in result["tickers"]:
        assert get_etf(ticker) is not None, f"{ticker!r} is not a real ETF in the database"
        # Never fabricate a ticker outside the requested market's universe.
        assert get_etf(ticker).country == market


def test_example_etfs_mixed_market_no_crash_and_sensible_result():
    result = select_example_etfs("Mixed", "balanced")
    assert isinstance(result["tickers"], list)
    assert result["selection_logic"]
    for ticker in result["tickers"]:
        assert get_etf(ticker) is not None
    # At least one of the three curated markets should produce a pick.
    assert len(result["tickers"]) > 0


def test_build_goal_plan_includes_example_etfs_per_scenario():
    plan = build_goal_plan(
        current_age=28, target_age=58, current_capital=1000.0,
        monthly_contribution=300.0, target_mode="total_value",
        target_amount=400000.0, market_preference="Taiwan",
        risk_tolerance="aggressive", base_currency="TWD",
    )
    assert plan["valid"]
    for scenario in plan["scenarios"].values():
        for ticker in scenario["example_etfs"]:
            assert get_etf(ticker) is not None
        assert scenario["selection_logic"]


def test_base_currency_passthrough_no_conversion():
    plan = build_goal_plan(
        current_age=30, target_age=50, current_capital=0.0,
        monthly_contribution=200.0, target_mode="total_value",
        target_amount=200000.0, base_currency="GBP",
    )
    assert plan["valid"]
    assert plan["base_currency"] == "GBP"
