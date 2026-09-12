"""
Portfolio Optimization Methodology & Model Validation (Round M1)

This module does NOT implement any new financial calculation. It documents,
validates, and packages the methodology the optimizer ALREADY uses (audited
directly from src/portfolio_optimizer.py and src/financial_metrics.py), so
that methodology can be displayed to the user and tested independently of
the page that renders it.

── AUDIT FINDINGS (Round M1 Part 1-3) ───────────────────────────────────────
Expected return:
  - Price field: Yahoo Finance "Close", fetched with yfinance's
    `auto_adjust=True` (src/data_loader.py::_download_single_ticker) --
    this makes "Close" economically equivalent to a dividend/split-adjusted
    close, not a raw unadjusted price.
  - Return definition: SIMPLE return, `prices_df.pct_change(fill_method=None)`
    -- never log returns -- in both run_optimization() and covariance_matrix().
  - Frequency: daily (one observation per trading day in the price history).
  - Estimator: ARITHMETIC mean of daily simple returns
    (`returns_df.mean().values` in run_optimization()), annualized by
    SIMPLE multiplication, not compounding:
        mu_annual = mean(r_t) * 252   (see portfolio_return() in
        src/financial_metrics.py: `np.dot(weights, mean_returns) * 252`)
  - This is an ARITHMETIC annualization, not geometric/CAGR. (Contrast with
    annualized_return() elsewhere in financial_metrics.py, which computes a
    GEOMETRIC CAGR on REALIZED prices for historical performance reporting
    -- e.g. ETF Analysis KPIs, backtest summary rows. That is a genuinely
    different, already-existing calculation used for a different purpose
    (ex-post realized performance) and is NOT changed by this module; the
    optimizer's own forward-looking expected-return ESTIMATE is the
    arithmetic one above.)
  - Missing observations: rows are dropped only when EVERY selected ticker
    is NaN that day (`dropna(how="all")`), never when a single ticker alone
    is missing a date -- this prevents one ticker's gap from wiping out
    every other ticker's return for that date.
  - Same date window for every ETF: YES -- both mean_returns and the
    covariance matrix are computed from the SAME `prices_df` argument,
    which the page (pages/2_Portfolio_Optimizer.py) has already sliced to
    the common overlapping date range (see get_common_date_range() in
    src/data_cleaner.py) before either calculation runs.
  - Zero/invalid returns: not specially filtered beyond the NaN handling
    above -- a genuine zero return (no price change that day) is a valid
    observation and is kept.

Covariance:
  - Input frequency: daily simple returns (identical return series as
    expected returns above -- see covariance_matrix() in
    src/financial_metrics.py).
  - Estimator: pandas' `DataFrame.cov()` -- ordinary SAMPLE covariance
    (Bessel-corrected, ddof=1), using pairwise-complete observations per
    column pair (pandas' documented default). In this application's actual
    runtime pipeline, prices reaching this function have already been
    forward/back-filled within the common date window
    (clean_price_data()), so no NaNs are typically present at this stage
    regardless of the pairwise-vs-listwise distinction.
  - Annualization: `Sigma_annual = Cov(daily returns) * 252` -- exactly the
    simple scaling hypothesized in the round's own brief; confirmed by
    reading the code, not assumed.
  - Numerical stabilization: a small ridge term `+ np.eye(n) * 1e-8` is
    added to the diagonal before optimization (run_optimization() and the
    Efficient Frontier code path) purely to avoid near-singularity: no
    other PSD projection/eigenvalue clipping is performed.
  - Same sample as expected returns: YES (same `prices_df`, same
    `pct_change` call pattern).
  - No shrinkage estimator (Ledoit-Wolf or otherwise) is implemented
    anywhere in this codebase today -- confirmed by a repo-wide search.
    scikit-learn IS already a project dependency (requirements.txt) and
    `sklearn.covariance.LedoitWolf` IS importable in this environment, so a
    future round could wire it in as a selectable estimator -- see
    COVARIANCE_ESTIMATOR_OPTIONS below for the documented (not yet
    selectable) architecture placeholder. Production behavior is
    UNCHANGED this round.

Maximum Sharpe:
  - Objective: minimize the NEGATIVE Sharpe ratio (`neg_sharpe()` in
    optimize_max_sharpe()) -- mathematically identical to maximizing
    Sharpe = (w^T mu - Rf) / sqrt(w^T Sigma w).
  - Solver: scipy.optimize.minimize(method="SLSQP").
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Audited constants (Round M1) -- these describe what the code ALREADY
# does; changing what the code does must come with changing these, never
# the other way around. ──────────────────────────────────────────────────
RETURN_PRICE_FIELD = "Close (auto-adjusted for splits/dividends)"
RETURN_TYPE = "Simple"                       # r_t = P_t / P_(t-1) - 1
RETURN_FREQUENCY = "Daily"
RETURN_ESTIMATOR = "Historical Arithmetic Mean"
RETURN_COMPOUNDING = "Arithmetic (simple multiplication, not compounded)"
ANNUALIZATION_FACTOR = 252
COVARIANCE_ESTIMATOR = "Sample Covariance (pandas .cov(), pairwise-complete, ddof=1)"
COVARIANCE_REGULARIZATION = "Diagonal ridge +1e-8 (numerical conditioning only)"

# Documented for Round M1 Part 4 -- NOT selectable, NOT used in production.
# A future round may expose this as a user-selectable estimator.
COVARIANCE_ESTIMATOR_OPTIONS = ["Sample Covariance", "Ledoit-Wolf Shrinkage (planned, not yet available)"]

# ── Data sufficiency policy (Round M1 Part 8) ────────────────────────────
# MIN_OBSERVATIONS_ABSOLUTE mirrors the hard stop already enforced in
# pages/2_Portfolio_Optimizer.py ("at least 20 trading days required").
# MIN_OBSERVATIONS_RECOMMENDED (~3 trading months) is NEW this round: below
# this, estimation is still performed (never silently blocked) but flagged
# to the user as "usable but limited" rather than presented with the same
# confidence as a longer sample.
MIN_OBSERVATIONS_ABSOLUTE = 20
MIN_OBSERVATIONS_RECOMMENDED = 60

SUFFICIENCY_INSUFFICIENT = "insufficient"
SUFFICIENCY_LIMITED = "limited"
SUFFICIENCY_SUFFICIENT = "sufficient"


def classify_data_sufficiency(n_observations: int,
                               min_absolute: int = MIN_OBSERVATIONS_ABSOLUTE,
                               min_recommended: int = MIN_OBSERVATIONS_RECOMMENDED) -> str:
    """Classify a sample size into one of three tiers (Round M1 Part 8/10):

    "insufficient" -- below the hard minimum; estimation should not proceed.
    "limited"      -- usable, but below the recommended threshold for
                       robust estimation; show a caveat, do not block.
    "sufficient"   -- at or above the recommended threshold.
    """
    if n_observations < min_absolute:
        return SUFFICIENCY_INSUFFICIENT
    if n_observations < min_recommended:
        return SUFFICIENCY_LIMITED
    return SUFFICIENCY_SUFFICIENT


# ── Backtest labeling (Round M1 Part 11/12) ──────────────────────────────
# AUDIT: backtest_portfolio() (src/portfolio_optimizer.py) applies ONE
# fixed weights dict across the ENTIRE historical price series -- the same
# `weights` the optimizer produced from the FULL sample. It does not
# re-estimate or re-optimize at any earlier historical date using only
# information available as of that date. This is a fixed-allocation
# lookback illustration, not a walk-forward / bias-free backtest, and must
# be labeled as such rather than implying the model would have picked this
# allocation at each historical point.
BACKTEST_TYPE_FIXED_ALLOCATION = "Fixed Allocation"


@dataclass
class Constraint:
    """One row of the reusable constraint-methodology table (Round M1
    Part 6). `name`/`meaning`/`purpose` are translation KEYS (i18n), not
    display strings -- the page resolves them with t() at render time so
    this module has no i18n dependency and stays reusable by a future
    global Methodology & Model Validation page."""
    name_key: str
    meaning: str          # a literal math expression (e.g. "sum(w_i) = 1") -- language-neutral, not translated
    current_value: str
    purpose_key: str


def build_constraint_list(min_weight: float, max_weight: float, allow_short: bool,
                           target_return: Optional[float] = None) -> List[Constraint]:
    """The COMPLETE, ACTUAL constraint set for one optimizer run -- never
    describes a constraint that isn't actually passed to scipy. Mirrors
    the exact bounds logic in optimize_max_sharpe()/optimize_min_volatility()/
    optimize_target_return() (src/portfolio_optimizer.py): when allow_short
    is True, per-asset bounds are symmetric (-max_weight, max_weight) and
    min_weight is NOT applied (matching validate_weight_constraints()'s own
    documented behavior); when False, bounds are (min_weight, max_weight)
    and weights are long-only.
    """
    constraints = [
        Constraint("methodology_constraint_fully_invested", "sum(w_i) = 1", "100%",
                   "methodology_constraint_fully_invested_purpose"),
    ]
    if allow_short:
        constraints.append(Constraint(
            "methodology_constraint_bounds_short", f"-{max_weight:.0%} ≤ w_i ≤ {max_weight:.0%}",
            f"±{max_weight:.0%}", "methodology_constraint_bounds_short_purpose",
        ))
    else:
        constraints.append(Constraint(
            "methodology_constraint_long_only", "w_i ≥ 0", "0%",
            "methodology_constraint_long_only_purpose",
        ))
        if min_weight > 0:
            constraints.append(Constraint(
                "methodology_constraint_min_weight", f"w_i ≥ {min_weight:.0%}", f"{min_weight:.0%}",
                "methodology_constraint_min_weight_purpose",
            ))
        constraints.append(Constraint(
            "methodology_constraint_max_weight", f"w_i ≤ {max_weight:.0%}", f"{max_weight:.0%}",
            "methodology_constraint_max_weight_purpose",
        ))
    if target_return is not None:
        constraints.append(Constraint(
            "methodology_constraint_target_return", "w^T μ = target",
            f"{target_return:.2%}", "methodology_constraint_target_return_purpose",
        ))
    constraints.append(Constraint(
        "methodology_constraint_numerical_tolerance", "SLSQP ftol = 1e-9, maxiter = 1000", "1e-9 / 1000",
        "methodology_constraint_numerical_tolerance_purpose",
    ))
    return constraints


@dataclass
class ValidationResult:
    is_valid: bool
    problems: List[str] = field(default_factory=list)  # translation keys, not display strings


def validate_optimization_result(weights: Dict[str, float], expected_return: float,
                                  expected_volatility: float, sharpe_ratio: float,
                                  min_weight: float, max_weight: float, allow_short: bool,
                                  optimizer_success: bool, tol: float = 1e-4) -> ValidationResult:
    """Systematic post-optimization validation (Round M1 Part 7) -- NEW
    this round. Previously, run_optimization() trusted scipy's own
    `result.success` flag alone and then UNCONDITIONALLY renormalized
    weights to sum to 1 (see _clean_weights()); that renormalization step
    means `abs(sum(weights) - 1)` can never actually fail by the time this
    runs, which could otherwise mask a genuinely bad underlying solution.
    This function re-checks every condition explicitly and independently,
    so a portfolio is only ever presented as valid if it demonstrably is.

    Returns a ValidationResult; `problems` is empty iff `is_valid` is True.
    A caller MUST refuse to present the portfolio as valid when
    `is_valid` is False (Round M1: "do NOT silently normalize a seriously
    invalid solution").
    """
    problems: List[str] = []

    if not optimizer_success:
        problems.append("methodology_validation_optimizer_not_converged")

    if weights:
        total = sum(weights.values())
        if abs(total - 1.0) > tol:
            problems.append("methodology_validation_weights_not_normalized")

        lower_bound = -max_weight if allow_short else min_weight
        for w in weights.values():
            if w < lower_bound - tol:
                problems.append("methodology_validation_weight_below_bound")
                break
        for w in weights.values():
            if w > max_weight + tol:
                problems.append("methodology_validation_weight_above_bound")
                break
        if not allow_short:
            for w in weights.values():
                if w < -tol:
                    problems.append("methodology_validation_negative_weight_not_allowed")
                    break
    else:
        problems.append("methodology_validation_no_weights")

    if not np.isfinite(expected_return):
        problems.append("methodology_validation_return_not_finite")
    if not np.isfinite(expected_volatility):
        problems.append("methodology_validation_volatility_not_finite")
    if not np.isfinite(sharpe_ratio):
        problems.append("methodology_validation_sharpe_not_finite")

    # De-duplicate while preserving first-seen order (a single bad weight
    # only needs to be reported once per problem category above, but keep
    # this defensive in case of future edits that might append twice).
    seen = set()
    deduped = [p for p in problems if not (p in seen or seen.add(p))]
    return ValidationResult(is_valid=len(deduped) == 0, problems=deduped)


@dataclass
class MethodologyMetadata:
    """Reusable model-metadata structure (Round M1 Part 18). Every field is
    populated from ACTUAL runtime values passed in by the caller -- this
    class never invents or independently recomputes a number; it only
    packages values the caller already computed. Intended to later feed a
    global Methodology & Model Validation page without duplicating this
    logic there.
    """
    return_estimator: str
    return_type: str
    return_frequency: str
    annualization_factor: int
    covariance_estimator: str
    risk_free_rate: float
    constraints: List[Constraint]
    common_start: Optional[str]
    common_end: Optional[str]
    observation_count: int
    data_sufficiency: str
    backtest_type: str
    optimization_method: str


def build_methodology_metadata(*, risk_free_rate: float, min_weight: float, max_weight: float,
                                allow_short: bool, common_start, common_end, observation_count: int,
                                optimization_method: str, target_return: Optional[float] = None) -> MethodologyMetadata:
    """Assemble the full methodology snapshot for the CURRENT optimizer run
    from values the caller already has in scope -- no network access, no
    recomputation, no defaults masquerading as measured data."""
    return MethodologyMetadata(
        return_estimator=RETURN_ESTIMATOR,
        return_type=RETURN_TYPE,
        return_frequency=RETURN_FREQUENCY,
        annualization_factor=ANNUALIZATION_FACTOR,
        covariance_estimator=COVARIANCE_ESTIMATOR,
        risk_free_rate=risk_free_rate,
        constraints=build_constraint_list(min_weight, max_weight, allow_short, target_return),
        common_start=str(common_start) if common_start is not None else None,
        common_end=str(common_end) if common_end is not None else None,
        observation_count=observation_count,
        data_sufficiency=classify_data_sufficiency(observation_count),
        backtest_type=BACKTEST_TYPE_FIXED_ALLOCATION,
        optimization_method=optimization_method,
    )
