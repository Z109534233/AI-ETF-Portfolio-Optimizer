"""
Goal Planner Module (Issue #22)

A self-contained, deterministic retirement / goal investment-planning
calculator. Every function here is pure Python math: no randomness, no
network I/O, no Streamlit imports, no OpenAI calls -- so the whole module is
trivially unit-testable and produces the exact same output for the exact
same input, every time.

Scope, deliberately narrow:
  - This module does NOT fetch live market data and does NOT estimate
    expected returns from history (see src/financial_metrics.py for that
    kind of thing). The three scenario assumptions below (SCENARIO_ASSUMPTIONS)
    are a hardcoded, clearly-labeled illustrative capital-market-assumption
    table -- NOT a live estimate -- see ASSUMPTIONS_DISCLOSURE.
  - This module does NOT do currency conversion. `base_currency` is passed
    through into the result purely as a display label; a separate FX module
    elsewhere in the repo is responsible for any actual conversion.
  - This module is NOT financial advice. See ASSUMPTIONS_DISCLOSURE.

Illustrative ETF examples (see `select_example_etfs()`) are drawn ONLY from
the real src/etf_database.py universe -- never fabricated.
"""

from typing import List, Optional

from src.etf_database import get_etf, get_tickers_by_country


# ── Scenario assumptions (illustrative long-run capital market assumptions) ─
# These are standard, textbook-style long-run capital-market-assumption
# (CMA) numbers for a diversified equity/bond mix at three risk levels --
# NOT derived from any live feed, NOT a forecast, and NOT specific to any
# one market/currency in this app. They exist purely so the Goal Planner can
# produce a deterministic, reproducible projection; the UI is expected to
# display ASSUMPTIONS_DISCLOSURE alongside any number computed from them.
#
#   conservative: bond-heavy allocation  -> lower expected return, lower vol
#   balanced:     ~60/40 equity/bond mix -> moderate expected return/vol
#   aggressive:   equity-heavy allocation -> higher expected return, higher vol
SCENARIO_ASSUMPTIONS = {
    "conservative": {
        "expected_return": 0.045, "expected_volatility": 0.07,
        "asset_mix": {"Equity": 0.30, "Fixed Income": 0.70},
    },
    "balanced": {
        "expected_return": 0.065, "expected_volatility": 0.12,
        "asset_mix": {"Equity": 0.60, "Fixed Income": 0.40},
    },
    "aggressive": {
        "expected_return": 0.085, "expected_volatility": 0.17,
        "asset_mix": {"Equity": 0.90, "Fixed Income": 0.10},
    },
}

# Order matters for display purposes (low risk -> high risk).
SCENARIO_ORDER = ["conservative", "balanced", "aggressive"]

# The classic "4% rule" safe-withdrawal-rate assumption, used ONLY to convert
# a desired retirement income (monthly or annual) into an implied total
# portfolio value target: implied_target_total = annual_income / WITHDRAWAL_RATE.
# This is a widely-cited historical rule of thumb, not a guarantee, and is
# always surfaced back to the caller via the `withdrawal_rate` result field
# so it is never a silently-baked-in assumption.
WITHDRAWAL_RATE = 0.04

ASSUMPTIONS_DISCLOSURE = (
    "This plan uses educational long-run nominal return and volatility assumptions "
    "chosen by this project to represent bond-heavy, balanced, and equity-heavy "
    "portfolio scenarios; they are not calibrated from a live provider and are not "
    "forecasts for any ETF. The deterministic contribution solver is complemented in "
    "the UI by Monte Carlo simulations using the same scenario assumptions. The selected "
    "base currency is a display label only: no FX conversion is performed, so a TWD goal "
    "paired with foreign-asset examples does not model currency risk. Inflation affects "
    "purchasing power; the stated goal amount itself is treated as a future nominal amount. "
    "Projections assume periodic rebalancing back to the target allocation, with no "
    "rebalancing cost or tax modeled. Income-based targets additionally assume a fixed "
    f"{WITHDRAWAL_RATE:.0%} annual withdrawal rate in retirement (the classic \"4% rule\"). "
    "This tool does not provide personalized financial, tax, or legal advice."
)

VALID_TARGET_MODES = ("total_value", "monthly_income", "annual_income")
VALID_RISK_TOLERANCES = ("conservative", "balanced", "aggressive")
VALID_MARKET_PREFERENCES = ("Taiwan", "United States", "United Kingdom", "Mixed")
VALID_BASE_CURRENCIES = ("TWD", "USD", "GBP")

# Required monthly contribution vs. the user's stated contribution is
# considered "on_track" when it is within this fraction of the required
# figure (documented rule, not an arbitrary silent tolerance): e.g. 1%
# means a stated contribution between 99% and 101% of the required amount
# both count as on_track, rather than flagging trivial rounding differences.
ON_TRACK_TOLERANCE = 0.01


def _annual_rate_to_monthly_rate(annual_rate: float) -> float:
    """Convert an annual compounding rate to the equivalent monthly rate."""
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def future_value_of_savings(
    current_capital: float,
    monthly_contribution: float,
    annual_return: float,
    n_months: int,
    annual_contribution: float = 0.0,
) -> float:
    """Future value of a lump sum plus ordinary-annuity monthly contributions
    (contribution made at the END of each month) plus an optional extra
    lump-sum contribution made once per year (also end-of-year), all
    compounded at `annual_return` (converted internally to the equivalent
    monthly rate for the monthly contribution stream).

    FV = current_capital * (1 + r)^n_years
       + monthly_contribution * [((1+r_m)^n_months - 1) / r_m]      (r_m > 0)
       + annual_contribution  * [((1+r)^n_years - 1) / r]           (r > 0)

    where r_m = (1+r)^(1/12) - 1 is the monthly-compounded equivalent of the
    annual rate `r`. The r == 0 edge case degenerates to simple addition
    (no growth), handled explicitly to avoid a division by zero.
    """
    if n_months <= 0:
        return float(current_capital)

    n_years = n_months / 12.0
    r_m = _annual_rate_to_monthly_rate(annual_return)

    # Lump sum growth.
    fv_lump = current_capital * (1.0 + annual_return) ** n_years

    # Monthly contribution stream (ordinary annuity, end-of-month deposits).
    if r_m == 0:
        fv_monthly = monthly_contribution * n_months
    else:
        fv_monthly = monthly_contribution * (((1.0 + r_m) ** n_months - 1.0) / r_m)

    # Annual extra contribution stream (ordinary annuity, end-of-year deposits).
    if annual_contribution == 0:
        fv_annual = 0.0
    elif annual_return == 0:
        fv_annual = annual_contribution * n_years
    else:
        fv_annual = annual_contribution * (((1.0 + annual_return) ** n_years - 1.0) / annual_return)

    return float(fv_lump + fv_monthly + fv_annual)


def required_monthly_contribution(
    current_capital: float,
    target_amount: float,
    annual_return: float,
    n_months: int,
    annual_contribution: float = 0.0,
) -> float:
    """Solve for the level monthly contribution (end-of-month, ordinary
    annuity) needed so that `future_value_of_savings(...)` lands exactly on
    `target_amount` after `n_months`, given the same current_capital,
    annual_return and annual_contribution assumptions.

    Rearranging the ordinary-annuity FV formula for the monthly-contribution
    term:

        remaining = target_amount - fv_lump - fv_annual
        monthly   = remaining / annuity_factor
        annuity_factor = ((1+r_m)^n_months - 1) / r_m   (or n_months if r_m==0)

    Clamped at 0: if the lump sum and/or annual contribution alone already
    meet or exceed the target, the "required" additional monthly
    contribution is reported as 0.0 (documented as "you could contribute
    less than planned") rather than a nonsensical negative number.
    """
    if n_months <= 0:
        return 0.0

    n_years = n_months / 12.0
    r_m = _annual_rate_to_monthly_rate(annual_return)

    fv_lump = current_capital * (1.0 + annual_return) ** n_years
    if annual_contribution == 0:
        fv_annual = 0.0
    elif annual_return == 0:
        fv_annual = annual_contribution * n_years
    else:
        fv_annual = annual_contribution * (((1.0 + annual_return) ** n_years - 1.0) / annual_return)

    remaining = target_amount - fv_lump - fv_annual

    if remaining <= 0:
        return 0.0

    if r_m == 0:
        annuity_factor = float(n_months)
    else:
        annuity_factor = ((1.0 + r_m) ** n_months - 1.0) / r_m

    if annuity_factor <= 0:
        return 0.0

    return float(remaining / annuity_factor)


def classify_status(
    monthly_contribution: float,
    required_contribution: float,
    tolerance: float = ON_TRACK_TOLERANCE,
) -> str:
    """Classify the user's stated monthly contribution against the
    scenario's required contribution.

    Documented rule: compare `monthly_contribution` to `required_contribution`
    with a +/- `tolerance` band around the required figure (default 1%):
      - within the band                                  -> "on_track"
      - below the band (contributing less than required)  -> "below_target"
      - above the band (contributing more than required)  -> "above_target"

    Special case: if required_contribution is 0 (target already met by the
    lump sum / annual contribution alone), any non-negative stated
    contribution is "on_track" or "above_target" (never "below_target",
    since 0 is already sufficient) -- any stated contribution >= 0 satisfies
    the goal, so we call it on_track unless the user is contributing extra
    (which is a fine outcome, not a failure), in which case "above_target".
    """
    if required_contribution <= 0:
        return "on_track" if monthly_contribution <= 0 else "above_target"

    lower = required_contribution * (1.0 - tolerance)
    upper = required_contribution * (1.0 + tolerance)

    if monthly_contribution < lower:
        return "below_target"
    if monthly_contribution > upper:
        return "above_target"
    return "on_track"


def select_example_etfs(market_preference: str, risk_tolerance: str) -> dict:
    """Deterministically pick 2-4 illustrative ETF tickers from the real
    src/etf_database.py universe for a given market preference and risk
    tolerance. Never fabricates a ticker; if a market's universe has no
    fitting category (e.g. no bond ETF found), that leg is simply omitted
    rather than invented.

    Selection rule (documented, deterministic, and simple by design -- this
    is illustrative example content, not a portfolio recommendation):
      - For a single market: pick one broad-market equity ETF (category
        "Equity", sector "Broad Market" preferred) from that market's
        universe, plus one bond ETF (category "Fixed Income") from the same
        market's universe when one exists. More equity ETFs are shown as
        risk_tolerance increases: conservative -> equity + bond (bond
        first if available); balanced -> equity + bond; aggressive ->
        equity only (skip the bond leg) if a second, distinct equity ETF is
        unavailable, otherwise equity + bond, equity-weighted by ordering.
      - For "Mixed": one representative equity ETF from each of the three
        supported markets (United States, Taiwan, United Kingdom), in that
        order, skipping any market with no equity ETF in the universe.

    Returns {"tickers": [...], "selection_logic": "..."}.
    """
    markets = ("United States", "Taiwan", "United Kingdom")

    def _pick_equity(country: str) -> Optional[str]:
        candidates = [t for t in get_tickers_by_country(country)
                      if (rec := get_etf(t)) and rec.category == "Equity"]
        # Prefer a genuine broad-market fund when one exists.
        broad = [t for t in candidates if get_etf(t).sector == "Broad Market"]
        pool = broad or candidates
        return pool[0] if pool else None

    def _pick_bond(country: str, exclude: Optional[str] = None) -> Optional[str]:
        candidates = [t for t in get_tickers_by_country(country)
                      if (rec := get_etf(t)) and rec.category == "Fixed Income" and t != exclude]
        return candidates[0] if candidates else None

    if market_preference == "Mixed":
        tickers = [t for t in (_pick_equity(m) for m in markets) if t]
        logic = (
            "Mixed market preference: one representative broad-market equity "
            "ETF drawn from each of the United States, Taiwan and United "
            "Kingdom universes (markets without a broad-market equity ETF "
            "in the database are omitted), independent of risk_tolerance."
        )
        return {"tickers": tickers, "selection_logic": logic}

    equity = _pick_equity(market_preference)
    bond = _pick_bond(market_preference, exclude=equity)

    tickers: List[str] = []
    if risk_tolerance == "conservative":
        if bond:
            tickers.append(bond)
        if equity:
            tickers.append(equity)
        logic = (
            f"Conservative: a bond ETF from the {market_preference} universe "
            "(category 'Fixed Income') listed ahead of a broad-market equity "
            "ETF (category 'Equity'), reflecting a more bond-weighted "
            "illustrative allocation."
        )
    elif risk_tolerance == "balanced":
        if equity:
            tickers.append(equity)
        if bond:
            tickers.append(bond)
        logic = (
            f"Balanced: one broad-market equity ETF plus one bond ETF from "
            f"the {market_preference} universe, in roughly equal illustrative "
            "weight."
        )
    else:  # aggressive
        if equity:
            tickers.append(equity)
        if bond:
            tickers.append(bond)
        logic = (
            f"Aggressive: broad-market equity ETF from the {market_preference} "
            "universe shown first (equity-weighted illustrative allocation); "
            "a bond ETF is still listed as a diversifier when one exists in "
            "the same market's universe."
        )

    return {"tickers": tickers, "selection_logic": logic}


def _build_scenario(
    scenario_name: str,
    current_capital: float,
    monthly_contribution: float,
    annual_contribution: float,
    n_months: int,
    target_amount: float,
    market_preference: str,
) -> dict:
    assumptions = SCENARIO_ASSUMPTIONS[scenario_name]
    annual_return = assumptions["expected_return"]
    annual_volatility = assumptions["expected_volatility"]
    asset_mix = dict(assumptions["asset_mix"])

    projected_value = future_value_of_savings(
        current_capital, monthly_contribution, annual_return, n_months, annual_contribution,
    )
    required_contribution = required_monthly_contribution(
        current_capital, target_amount, annual_return, n_months, annual_contribution,
    )
    status = classify_status(monthly_contribution, required_contribution)

    example = select_example_etfs(market_preference, scenario_name)

    return {
        "expected_return": annual_return,
        "expected_volatility": annual_volatility,
        "asset_mix": asset_mix,
        "projected_value": projected_value,
        "required_monthly_contribution": required_contribution,
        "status": status,
        "example_etfs": example["tickers"],
        "selection_logic": example["selection_logic"],
    }


def build_goal_plan(
    current_age: int,
    target_age: int,
    current_capital: float,
    monthly_contribution: float,
    target_mode: str,
    target_amount: float,
    market_preference: str = "Mixed",
    risk_tolerance: str = "balanced",
    base_currency: str = "USD",
    annual_contribution: float = 0.0,
    horizon_years: Optional[float] = None,
) -> dict:
    """Top-level Goal Planner entrypoint.

    Builds a deterministic, three-scenario (conservative / balanced /
    aggressive) retirement or goal-investment plan. See the module
    docstring and ASSUMPTIONS_DISCLOSURE for what is and is not modeled.

    Parameters mirror the issue spec:
      - current_age, target_age: used to derive the investment horizon in
        years (target_age - current_age), unless `horizon_years` is
        explicitly supplied, in which case it always wins.
      - current_capital, monthly_contribution, annual_contribution: the
        user's actual/planned savings behavior, in `base_currency`.
      - target_mode: "total_value", "monthly_income", or "annual_income".
      - target_amount: meaning depends on target_mode (a total portfolio
        value, or a desired monthly/annual retirement income), always in
        `base_currency`.
      - market_preference: "Taiwan" / "United States" / "United Kingdom" /
        "Mixed" -- used ONLY to pick illustrative example ETFs.
      - risk_tolerance: "conservative" / "balanced" / "aggressive" -- does
        NOT restrict which of the three scenarios are computed (all three
        are always returned so the user can compare), it only affects
        which example ETFs are highlighted for... itself, via
        select_example_etfs() per scenario.
      - base_currency: passed through as a display label only; no FX
        conversion happens in this module.

    Returns a single dict (see module docstring / issue spec for full
    shape): {valid, errors, horizon_years, horizon_months, withdrawal_rate,
    implied_target_total, scenarios: {conservative, balanced, aggressive},
    assumptions_disclosure, ...passthrough labels}.
    """
    errors: List[str] = []

    if target_mode not in VALID_TARGET_MODES:
        errors.append(f"Invalid target_mode: {target_mode!r}. Must be one of {VALID_TARGET_MODES}.")
    if risk_tolerance not in VALID_RISK_TOLERANCES:
        errors.append(f"Invalid risk_tolerance: {risk_tolerance!r}. Must be one of {VALID_RISK_TOLERANCES}.")
    if market_preference not in VALID_MARKET_PREFERENCES:
        errors.append(f"Invalid market_preference: {market_preference!r}. Must be one of {VALID_MARKET_PREFERENCES}.")
    if base_currency not in VALID_BASE_CURRENCIES:
        errors.append(f"Invalid base_currency: {base_currency!r}. Must be one of {VALID_BASE_CURRENCIES}.")

    if horizon_years is None:
        computed_horizon = target_age - current_age
    else:
        computed_horizon = horizon_years

    if computed_horizon is None or computed_horizon <= 0:
        errors.append(
            "Investment horizon must be positive: target_age must be greater than "
            "current_age (or pass an explicit positive horizon_years override)."
        )

    if target_amount is None or target_amount <= 0:
        errors.append("target_amount must be a positive number.")

    if current_capital is not None and current_capital < 0:
        errors.append("current_capital cannot be negative.")

    if monthly_contribution is not None and monthly_contribution < 0:
        errors.append("monthly_contribution cannot be negative.")

    if annual_contribution is not None and annual_contribution < 0:
        errors.append("annual_contribution cannot be negative.")

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "horizon_years": None,
            "horizon_months": None,
            "withdrawal_rate": None,
            "implied_target_total": None,
            "scenarios": {},
            "base_currency": base_currency,
            "assumptions_disclosure": ASSUMPTIONS_DISCLOSURE,
        }

    horizon_years_final = float(computed_horizon)
    horizon_months = int(round(horizon_years_final * 12))

    withdrawal_rate: Optional[float] = None
    if target_mode == "monthly_income":
        withdrawal_rate = WITHDRAWAL_RATE
        annual_income_target = target_amount * 12.0
        implied_target_total = annual_income_target / WITHDRAWAL_RATE
    elif target_mode == "annual_income":
        withdrawal_rate = WITHDRAWAL_RATE
        implied_target_total = target_amount / WITHDRAWAL_RATE
    else:  # total_value
        implied_target_total = float(target_amount)

    scenarios = {}
    for scenario_name in SCENARIO_ORDER:
        scenarios[scenario_name] = _build_scenario(
            scenario_name,
            current_capital,
            monthly_contribution,
            annual_contribution,
            horizon_months,
            implied_target_total,
            market_preference,
        )

    return {
        "valid": True,
        "errors": [],
        "horizon_years": horizon_years_final,
        "horizon_months": horizon_months,
        "withdrawal_rate": withdrawal_rate,
        "implied_target_total": implied_target_total,
        "scenarios": scenarios,
        "base_currency": base_currency,
        "assumptions_disclosure": ASSUMPTIONS_DISCLOSURE,
    }
