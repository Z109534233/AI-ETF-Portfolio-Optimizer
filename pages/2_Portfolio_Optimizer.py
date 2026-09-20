"""
Page 2: Portfolio Optimizer
Mean-variance optimization, efficient frontier, and portfolio backtesting --
organized as a compact analytics workspace (Overview / Allocation /
Strategy Lab / Backtest & Risk / Save & Actions).

PORTFOLIO OPTIMIZER FULL WORKSPACE UI REDESIGN: no optimization math,
weights, Strategy Comparison/Efficient Frontier/Portfolio Diagnosis
formulas, or backtest calculations changed in this round -- this file only
reorganizes HOW and WHEN existing code renders. The five top-level
workspaces are plain Python if/elif branches gated on one
st.segmented_control's value (NOT st.tabs, which would execute every
branch's body -- including Monte Carlo and the Efficient Frontier solve --
on every rerun regardless of which tab is visible).

current_portfolio (the ONE canonical object Save/Investment Simulator/Risk
Analytics all consume) is built UNCONDITIONALLY, before the workspace
branches, from data that's cheap to compute (the user's own already-run
optimization result, portfolio_diagnosis(), and one backtest pass) -- never
gated behind which workspace happens to be open, and never a second,
independently-computed copy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import html as _html
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data, get_common_date_range
from src.etf_database import to_yahoo_symbol, rename_yahoo_columns, get_etf
from src.fx import convert_prices_to_base_currency
from src.portfolio_optimizer import (
    run_optimization, monte_carlo_simulation, backtest_portfolio,
    compute_efficient_frontier, build_backtest_reference_plan,
    bootstrap_max_sharpe_weight_stability, DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP,
)
from src.financial_metrics import (
    covariance_matrix, annualized_return, annualized_volatility,
    sharpe_ratio, maximum_drawdown, drawdown_series, portfolio_diagnosis,
    correlation_matrix, average_pairwise_correlation, correlation_diversification_level
)
from src.database import save_portfolio, init_database, find_duplicate_portfolio, APP_VERSION
from src.risk_analytics import holdings_overlap_matrix
from src.report_generator import generate_portfolio_report
from src.charts import (
    efficient_frontier_chart, allocation_donut_chart,
    portfolio_growth_chart, drawdown_chart, apply_dark_theme,
    bootstrap_weight_stability_box_chart,
)
from src.utils import (
    load_css, page_header, disclaimer_box, dataframe_to_csv,
    weights_to_dataframe, get_date_range_defaults, metric_card_html
)
from src.demo_portfolio import DIVERSIFIED_DEMO_TICKERS
from src.risk_free_rate import get_cached_risk_free_rate, format_rate_provenance, is_manual_override
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state,
    region_multiselect, region_etf_options_multi, multi_region_etf_multiselect,
    kpi_card, chart_caption, ai_interpret_button,
    setup_summary_pills, results_hero, results_hero_metric,
)
from src.theme import COLORS
from src.i18n import t, t_opt_method, t_country, get_language, OPTIMIZATION_METHOD_KEYS

st.set_page_config(
    page_title="Portfolio Optimizer | AI ETF Portfolio Optimizer",
    page_icon="⚡",
    layout="wide"
)

load_css()
init_database()

page_header(t("opt_title"), t("opt_subtitle"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
# Always visible: Market / ETF Selection / Investment Amount / Investment
# Goal / Risk Tolerance / Investment Horizon / Optimization Method / Build
# button (PRODUCT SPEC section 9). Advanced Constraints and Historical &
# Model Settings are two SEPARATE collapsed panels (sections 10/11) -- split
# out of what used to be one combined "Advanced Settings" expander. Every
# existing variable below (optimization_method, min_weight, max_weight,
# allow_short, target_return_pct, risk_free_rate, start_date, end_date,
# n_simulations, custom_ticker) keeps its exact name and feeds the SAME
# calculation calls further down, unchanged -- only where/how each control
# is presented has moved.
def _shadow_default(name: str, default):
    """Read-or-seed a plain (non-widget) session_state mirror for a sidebar
    control. Needed because Streamlit can drop a widget's own keyed state
    if something earlier in the same script run -- the language selector
    inside render_sidebar_nav(), called first thing on every page --
    triggers st.rerun() before this widget has been (re-)instantiated on
    that particular pass. A plain session_state entry isn't tied to widget
    instantiation, so it survives that and reseeds the widget on the next
    run instead of silently falling back to its hard-coded default.
    """
    shadow_key = f"_{name}_shadow"
    if shadow_key not in st.session_state:
        st.session_state[shadow_key] = default
    return shadow_key, st.session_state[shadow_key]


with st.sidebar:
    render_sidebar_nav()
    render_sidebar_footer()

# ── Build Your Portfolio (Issue #24 visual-acceptance round item B5) ────────
# Moved OUT of the persistent navigation sidebar (which now holds only
# language/nav/footer, above) into a secondary configuration surface on the
# main page -- the sidebar is no longer a mix of "global app navigation" and
# "this one page's long input form". Expanded by default on a fresh session
# (nothing built yet, so the user needs the form immediately visible) and
# collapsed by default once a portfolio already exists (the compact setup
# line + results below take over instead). Note: st.expander's `expanded=`
# is only a per-render default, not a persisted widget value (confirmed:
# st.expander does not populate st.session_state even when given a `key`),
# so a user who manually re-opens this panel after a build may see it
# snap back to this computed default on an unrelated rerun -- a known,
# accepted Streamlit limitation, not a bug in this page's own state logic.
with st.expander(t("opt_edit_settings_title"), expanded=(st.session_state.get("opt_result") is None)):
    st.markdown(f"### {t('opt_build_portfolio_title')}")

    # 1. Countries -- Portfolio Optimizer's OWN true multi-select (Issue
    # #24 item 1), independent of the single-select "selected_region"
    # global state shared by ETF Analysis / Risk Analytics / Machine
    # Learning / AI Advisor. The user picks exactly the 1-3 markets they
    # want (e.g. Taiwan + United States) -- there is no "All Regions"
    # shortcut here.
    selected_regions = region_multiselect()
    etf_options = region_etf_options_multi(selected_regions)

    # 2. ETF Selection -- the union of ETFs from ONLY the selected
    # countries. Removing a country automatically prunes any ETF that
    # belonged exclusively to it (see multi_region_etf_multiselect()'s
    # docstring) while keeping every other already-selected ETF intact.
    # Initial demo default (Issue #45 item 1): a fresh United-States-only
    # session starts from the shared cross-asset demo portfolio (VOO/VXUS/
    # BND/GLD/TLT) instead of whatever sorts first in the raw US ETF
    # universe -- which could otherwise be a VOO/VTI/QQQ/SPY-style set with
    # extreme mutual overlap. Only applies to the ONE-TIME initial seed
    # (see multi_region_etf_multiselect()'s docstring); the user remains
    # completely free to pick any ETF afterward, and any other region
    # combination is unaffected.
    _demo_default = (
        DIVERSIFIED_DEMO_TICKERS if selected_regions == ["United States"] else None
    )
    selected_etfs = multi_region_etf_multiselect(
        selected_regions, etf_options, t("field_select_etfs"),
        help_text=t("opt_select_etfs_help"), n_default=5, default_override=_demo_default,
    )
    if _demo_default and set(selected_etfs) == set(DIVERSIFIED_DEMO_TICKERS):
        st.caption(t("opt_demo_portfolio_caption"))
    with st.expander(t("field_add_custom_etf"), expanded=False):
        _ctk, _ctv = _shadow_default("opt_custom_ticker", "")
        custom_ticker = st.text_input(
            t("field_add_custom_etf"), value=_ctv, placeholder="e.g. ARKK",
            label_visibility="collapsed", key="opt_custom_ticker",
        ).upper().strip()
        st.session_state[_ctk] = custom_ticker
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # 3. Investment Amount -- currency intentionally stays USD-denominated
    # in Round 1. weights_to_dataframe() (src/utils.py), the PDF report,
    # and every "$" metric below are hard-coded to USD throughout the
    # calculation/display pipeline; relabeling this per-market without
    # touching that pipeline would silently mislabel the numbers.
    _ak, _av = _shadow_default("investment_amount", 10000.0)
    investment_amount = st.number_input(
        t("field_investment_amount_usd"), min_value=100.0, max_value=10_000_000.0,
        value=_av, step=500.0, key="investment_amount",
    )
    st.session_state[_ak] = investment_amount

    # 3b. Base Currency (Issue #22 section F -- mixed-market portfolios).
    # Selecting ETFs across multiple markets (e.g. VOO + QQQ + 0050) means
    # their prices are natively denominated in different currencies; this is
    # the ONE base currency every price series is converted into (via
    # src.fx) before returns/covariance/optimization are computed below.
    # Single-currency portfolios that already match this choice pay no FX
    # conversion cost at all -- see the currency-adjusted disclosure in the
    # Methodology panel further down the page.
    _BASE_CURRENCY_OPTIONS = ["USD", "TWD", "GBP"]
    _bck, _bcv = _shadow_default("opt_base_currency", "USD")
    base_currency = st.selectbox(
        t("opt_base_currency_label"), _BASE_CURRENCY_OPTIONS,
        index=_BASE_CURRENCY_OPTIONS.index(_bcv) if _bcv in _BASE_CURRENCY_OPTIONS else 0,
        help=t("opt_base_currency_help"), key="opt_base_currency",
    )
    st.session_state[_bck] = base_currency

    # 4. Investment Goal -- portfolio metadata / user preference only.
    # Does NOT alter the optimizer's math -- later rounds will map it to
    # portfolio interpretation and AI analysis.
    _goal_options = ["Growth", "Balanced", "Income", "Capital Preservation"]
    _goal_labels = {
        "Growth": t("opt_goal_growth"), "Balanced": t("opt_goal_balanced"),
        "Income": t("opt_goal_income"), "Capital Preservation": t("opt_goal_capital_preservation"),
    }
    _gk, _gv = _shadow_default("investment_goal", _goal_options[0])
    investment_goal = st.selectbox(
        t("opt_investment_goal_label"), _goal_options,
        index=_goal_options.index(_gv) if _gv in _goal_options else 0,
        format_func=lambda x: _goal_labels.get(x, x), key="investment_goal",
    )
    st.session_state[_gk] = investment_goal

    # 5. Risk Tolerance -- same scope note as Investment Goal above:
    # metadata only, does not change optimization constraints.
    _risk_options = ["Conservative", "Balanced", "Aggressive"]
    _risk_labels = {
        "Conservative": t("opt_risk_conservative"), "Balanced": t("opt_risk_balanced"),
        "Aggressive": t("opt_risk_aggressive"),
    }
    _rk, _rv = _shadow_default("risk_tolerance", _risk_options[1])
    risk_tolerance = st.radio(
        t("opt_risk_tolerance_label"), _risk_options,
        index=_risk_options.index(_rv) if _rv in _risk_options else 1,
        format_func=lambda x: _risk_labels.get(x, x), key="risk_tolerance",
        horizontal=True,
    )
    st.session_state[_rk] = risk_tolerance

    # 6. Investment Horizon -- same scope note as above. This is the
    # investor's intended holding period, distinct from the Historical Data
    # Range in the collapsed panel below (a calculation input, not a
    # preference) -- the two must not be confused.
    _horizon_options = ["1 Year", "3 Years", "5 Years", "10+ Years"]
    _horizon_labels = {
        "1 Year": t("opt_horizon_1y"), "3 Years": t("opt_horizon_3y"),
        "5 Years": t("opt_horizon_5y"), "10+ Years": t("opt_horizon_10y"),
    }
    _hk, _hv = _shadow_default("investment_horizon", _horizon_options[2])
    investment_horizon = st.selectbox(
        t("opt_investment_horizon_label"), _horizon_options,
        index=_horizon_options.index(_hv) if _hv in _horizon_options else 2,
        format_func=lambda x: _horizon_labels.get(x, x), key="investment_horizon",
    )
    st.session_state[_hk] = investment_horizon

    # Honest scope disclosure (Issue #20 section 3A): Investment Goal, Risk
    # Tolerance, and Investment Horizon above are genuinely metadata/
    # interpretation inputs only -- see the code comments on each widget --
    # they do NOT change the optimizer's math (constraints, expected
    # return/covariance estimation, or the chosen method's objective
    # function below). Stated visibly here so the UI never implies
    # personalization that isn't actually happening.
    st.caption(f"ℹ️ {t('opt_investor_profile_scope_note')}")

    st.markdown("---")

    # ── Optimization Strategy ────────────────────────────────────────────
    # Only the methods actually implemented in src/portfolio_optimizer.py
    # (OPTIMIZATION_METHOD_KEYS already matches run_optimization()'s real
    # branches 1:1) are shown; nothing added.
    st.markdown(f"### {t('opt_strategy_title')}")
    _opt_method_labels = {k: t_opt_method(k) for k in OPTIMIZATION_METHOD_KEYS}
    _mk, _mv = _shadow_default("optimization_method", list(OPTIMIZATION_METHOD_KEYS.keys())[0])
    _method_options = list(OPTIMIZATION_METHOD_KEYS.keys())
    optimization_method = st.selectbox(
        t("opt_method_label"), _method_options,
        index=_method_options.index(_mv) if _mv in _method_options else 0,
        format_func=lambda x: _opt_method_labels.get(x, x), key="optimization_method",
    )
    st.session_state[_mk] = optimization_method

    target_return_pct = None
    if optimization_method == "Target Return":
        _tk, _tv = _shadow_default("opt_target_return_pct", 10.0)
        target_return_pct = st.slider(
            t("opt_target_return_pct"), 1.0, 30.0, _tv, 0.5, key="opt_target_return_pct_slider",
        ) / 100
        st.session_state[_tk] = target_return_pct * 100

    # ── Advanced Constraints (collapsed by default) ──────────────────────
    with st.expander(t("opt_advanced_constraints_title"), expanded=False):
        _minwk, _minwv = _shadow_default("opt_min_weight", 0.0)
        min_weight = st.slider(t("opt_min_weight"), 0.0, 20.0, _minwv, 1.0, key="opt_min_weight_slider") / 100
        st.session_state[_minwk] = min_weight * 100

        _maxwk, _maxwv = _shadow_default("opt_max_weight", 100.0)
        max_weight = st.slider(t("opt_max_weight"), 10.0, 100.0, _maxwv, 5.0, key="opt_max_weight_slider") / 100
        st.session_state[_maxwk] = max_weight * 100

        _ashk, _ashv = _shadow_default("opt_allow_short", False)
        allow_short = st.checkbox(t("opt_allow_short"), value=_ashv, key="opt_allow_short_cb")
        st.session_state[_ashk] = allow_short

    # ── Historical & Model Settings (collapsed by default) ───────────────
    with st.expander(t("opt_historical_model_settings_title"), expanded=False):
        st.markdown(f"**{t('opt_historical_data_range_label')}**")
        default_start, default_end = get_date_range_defaults()
        _sk, _sv = _shadow_default("opt_start_date", default_start)
        start_date = st.date_input(t("field_start_date"), value=_sv, key="opt_start_date")
        st.session_state[_sk] = start_date

        _ek, _ev = _shadow_default("opt_end_date", default_end)
        end_date = st.date_input(t("field_end_date"), value=_ev, key="opt_end_date")
        st.session_state[_ek] = end_date

        _rf_info = get_cached_risk_free_rate()
        _fk, _fv = _shadow_default("opt_risk_free_rate", round(_rf_info["rate"] * 100, 2))
        risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, _fv, 0.25, key="opt_risk_free_rate_slider") / 100
        st.session_state[_fk] = risk_free_rate * 100
        st.caption(format_rate_provenance(_rf_info, get_language(), selected_rate=risk_free_rate))

        _nsk, _nsv = _shadow_default("opt_n_simulations", 5000)
        n_simulations = st.slider(t("opt_mc_simulations"), 1000, 10000, _nsv, 500, key="opt_n_simulations_slider")
        st.session_state[_nsk] = n_simulations

    # ── Primary Action ────────────────────────────────────────────────────
    run_btn = st.button(t("btn_run_optimization"), type="primary", use_container_width=True,
                        key="opt_run_optimization_btn")

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_regions) < 1:
    st.warning(t("msg_select_one_country"))
    st.stop()

if len(selected_etfs) < 2:
    st.warning(t("msg_select_two_etfs"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── Compact "Current Setup" Summary (Issue #24 item 2) ──────────────────────
# Replaces the old wide, equally-weighted horizontal strip (every setting
# shown at the same visual weight as the eventual results) with a single
# muted line -- setup stays visually secondary; results_hero() further down
# is where the eye is meant to land once a portfolio has been built.
_market_word = t("opt_setup_unit_market") if len(selected_regions) == 1 else t("opt_setup_unit_markets")
_etf_word = t("opt_setup_unit_etf") if len(selected_etfs) == 1 else t("opt_setup_unit_etfs")
setup_summary_pills([
    f"{len(selected_regions)} {_market_word}",
    f"{len(selected_etfs)} {_etf_word}",
    base_currency,
    _opt_method_labels.get(optimization_method, optimization_method),
])

# ── State Management ──────────────────────────────────────────────────────────
# `run_inputs` fingerprints everything the optimization result depends on.
# It is used AFTER a result exists, to detect that the sidebar has since
# diverged from whatever produced it (Issue #24 item 5: the "Build Optimized
# Portfolio" button is the ONE explicit trigger, on every render -- including
# the very first one in a fresh session. A fresh session's opt_result is
# None, but that alone must never kick off a download/FX/optimization pass;
# only `run_btn` (checked just below) does. Changing the country/ETF
# selection, date range, or optimization method after a build must
# invalidate the stale result with a visible warning, never silently kick
# off a fresh optimization on its own). Mirrors the same fingerprint-then-
# compare pattern already used by Investment Simulator's Historical
# Simulation (`_hist_fingerprint` / `sim_hist_inputs_changed_rerun`).
run_inputs = (
    tuple(sorted(selected_etfs)), tuple(sorted(selected_regions)), str(start_date), str(end_date),
    optimization_method, round(min_weight, 6), round(max_weight, 6),
    allow_short, target_return_pct, base_currency, round(risk_free_rate, 6),
)

if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "opt_run_inputs" not in st.session_state:
    st.session_state.opt_run_inputs = None

if run_btn:
    with st.spinner(t("msg_running_optimization")):
        # Map display tickers to their actual Yahoo Finance-fetchable symbols
        # (e.g. "0050" -> "0050.TW"); tickers not in the ETF database pass
        # through unchanged, so this has no effect on existing US tickers.
        yahoo_tickers = [to_yahoo_symbol(tk) for tk in selected_etfs]
        raw_prices = download_etf_data(yahoo_tickers, str(start_date), str(end_date))
        if raw_prices.empty:
            error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
            st.stop()

        raw_prices = rename_yahoo_columns(raw_prices)
        raw_prices = raw_prices[[tk for tk in selected_etfs if tk in raw_prices.columns]]

        # ── CRITICAL SAFETY RULE (market-data reliability fix) ───────────
        # download_etf_data() already retries each ticker individually
        # with backoff before giving up (src/data_loader.py), so a ticker
        # missing here means it genuinely failed after retries. A user who
        # selected N ETFs must NEVER unknowingly get an optimization result
        # computed from fewer than N -- so this STOPS instead of silently
        # continuing on whichever subset happened to succeed.
        missing_tickers = [tk for tk in selected_etfs if tk not in raw_prices.columns]
        if missing_tickers:
            _sep = "、" if get_language() == "zh-TW" else ", "
            error_state(
                t("opt_partial_data_title"),
                t("opt_partial_data_desc", tickers=_sep.join(missing_tickers)),
            )
            with st.expander(t("opt_data_retrieval_details_title")):
                for _tk in missing_tickers:
                    st.caption(f"{to_yahoo_symbol(_tk)} — {t('opt_request_failed_after_retry')}")
            st.stop()

        # ── Common-start-date handling ────────────────────────────────
        # A later-inception ETF is NOT unavailable -- it just has less
        # history. Determine the true common valid-data range from the RAW
        # (not yet ffill/bfill-cleaned) prices and slice to it BEFORE
        # clean_price_data() runs, so its back-fill never fabricates a flat
        # price for any ETF before it actually existed.
        common_start, common_end = get_common_date_range(raw_prices)
        if common_start is None:
            error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
            st.stop()
        if common_start.date() > start_date:
            st.info(t("opt_common_start_notice", date=common_start.strftime("%Y-%m-%d")))
        raw_prices = raw_prices.loc[(raw_prices.index >= common_start) & (raw_prices.index <= common_end)]

        prices_df = clean_price_data(raw_prices)
        prices_df = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]

        # ── Mixed-Market Currency Conversion (Issue #22 section F) ───────
        # Convert every column to `base_currency` via real FX series BEFORE
        # computing returns/covariance/Sharpe/optimization -- optimizing
        # mixed-currency raw local-currency prices directly would silently
        # blend incompatible units. Tickers not in the curated ETF database
        # (e.g. a free-text custom US ticker) default to USD, matching this
        # app's existing assumption everywhere else a currency isn't on
        # record. Single-currency selections that already match
        # base_currency skip FX entirely (src.fx never calls the network
        # for an identity conversion).
        _ticker_currency_map = {
            tk: (get_etf(tk).currency if get_etf(tk) else "USD") for tk in prices_df.columns
        }
        _currencies_present = set(_ticker_currency_map.values())
        fx_result = None
        if _currencies_present - {base_currency}:
            with st.spinner(t("opt_fx_converting")):
                fx_result = convert_prices_to_base_currency(
                    prices_df, _ticker_currency_map, base_currency, str(start_date), str(end_date),
                )
            _fx_unavailable = fx_result["unavailable_tickers"]
            if _fx_unavailable:
                _sep = "、" if get_language() == "zh-TW" else ", "
                error_state(
                    t("opt_fx_unavailable_title"),
                    t("opt_fx_unavailable_desc", tickers=_sep.join(_fx_unavailable), base=base_currency),
                )
                st.stop()
            prices_df = fx_result["converted_prices"]
            prices_df = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]
        st.session_state.opt_fx_result = fx_result

        # ── Validate prices_df ──────────────────────────────────────────
        if prices_df.empty or len(prices_df.columns) < 2:
            error_state(
                t("msg_no_price_data_title"),
                "At least 2 ETFs with valid price data are required to run "
                "portfolio optimization. Try different tickers or a wider "
                "date range."
            )
            st.stop()

        if len(prices_df) < 20:
            error_state(
                t("msg_no_price_data_title"),
                f"Only {len(prices_df)} trading day(s) of overlapping data "
                "were found — at least 20 are required for a meaningful "
                "optimization. Widen the date range."
            )
            st.stop()

        # ── Validate returns_df ─────────────────────────────────────────
        returns_df_check = prices_df.pct_change(fill_method=None).dropna(how="all")
        if returns_df_check.empty or len(returns_df_check) < 10:
            error_state(
                t("msg_no_price_data_title"),
                "Not enough overlapping daily returns could be computed "
                "from the downloaded price data. Widen the date range or "
                "choose different ETFs."
            )
            st.stop()

        result = run_optimization(
            prices_df=prices_df,
            method=optimization_method,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            target_return=target_return_pct
        )

        # Infeasible min/max weight constraints are a configuration problem,
        # not a data problem -- stop with a clear validation message rather
        # than showing "results" for a request that can't possibly be
        # satisfied. optimizer_failed (SLSQP genuinely didn't converge for
        # otherwise-feasible settings) stays a warning, since the
        # equal-weight fallback is still a reasonable number to show
        # alongside a clear "this isn't the real optimized result" note.
        _err_code = result.get("error_code")
        if _err_code == "infeasible_min_weight":
            error_state(
                t("opt_error_title"),
                t("opt_error_infeasible_min_weight", n=len(selected_etfs), min=f"{min_weight:.0%}"),
            )
            st.stop()
        elif _err_code == "infeasible_max_weight":
            error_state(
                t("opt_error_title"),
                t("opt_error_infeasible_max_weight", n=len(selected_etfs), max=f"{max_weight:.0%}"),
            )
            st.stop()
        elif _err_code == "optimizer_failed":
            st.warning(t("opt_error_optimizer_failed", method=t_opt_method(optimization_method)))
        elif result.get("error"):
            st.warning(t("opt_note_prefix", error=result["error"]))

        st.session_state.opt_result = result
        st.session_state.prices_df = prices_df
        st.session_state.opt_run_inputs = run_inputs
        # Fresh id/timestamp only when a NEW successful build actually
        # happens here -- stable across simple reruns (e.g. language switch
        # or a workspace change) that don't change run_inputs, so
        # current_portfolio's identity below only changes when the
        # underlying portfolio genuinely does.
        st.session_state.opt_portfolio_id = uuid.uuid4().hex[:12]
        st.session_state.opt_generated_at = datetime.now(timezone.utc).isoformat()
        # A fresh successful build means opt_result is now set, so the
        # settings panel's `expanded=(opt_result is None)` default above
        # will collapse it on the next render -- nothing further to do here.

result = st.session_state.opt_result
prices_df = st.session_state.prices_df

if result is None or prices_df is None or prices_df.empty:
    st.info(t("opt_msg_configure_and_run", action=t("btn_run_optimization")))
    st.stop()

# Stale-result guard (Issue #24 item 5): the sidebar has changed since this
# result was computed. Never silently recompute here -- only the "Build
# Optimized Portfolio" button (checked above) is allowed to trigger a new
# optimization -- so this warns and stops instead of rendering a result
# that no longer matches the currently-selected inputs.
if st.session_state.opt_run_inputs != run_inputs:
    st.warning(t("opt_inputs_changed_rerun", action=t("btn_run_optimization")))
    st.stop()

weights = result["weights"]
exp_ret = result["expected_return"]
exp_vol = result["expected_volatility"]
sharpe = result["sharpe_ratio"]
div_ratio = result.get("diversification_ratio", 1.0)

# ── Cheap, workspace-independent computations ───────────────────────────────
# Portfolio Diagnosis and a single backtest pass over the CHOSEN strategy's
# weights are both pure, cheap functions of already-loaded data (no chart
# rendering, no extra optimizer solves, no Monte Carlo) -- computed
# unconditionally so current_portfolio (below) is always fresh regardless
# of which workspace is open, satisfying PRODUCT SPEC sections 13/14/21.
# Genuinely expensive work (Strategy Comparison's 3 extra optimizer solves,
# Monte Carlo, the Efficient Frontier solve, the equal-weight comparison
# backtest) is deferred into the specific workspace/sub-view that needs it
# -- see the Strategy Lab and Backtest & Risk sections below.
_diag = portfolio_diagnosis(weights)
backtest_df = backtest_portfolio(prices_df, weights, investment_amount)
_cp_max_drawdown = maximum_drawdown(backtest_df["Portfolio Value"]) if not backtest_df.empty else None

# ── Canonical Current Portfolio ─────────────────────────────────────────────
# The ONE session-level object representing the current successfully-built
# portfolio. Downstream pages (Investment Simulator, Risk Analytics) and
# Save & Actions all consume THIS object -- exactly one source of truth,
# built here regardless of which workspace is currently active.
st.session_state.current_portfolio = {
    "portfolio_id": st.session_state.get("opt_portfolio_id"),
    "strategy": optimization_method,
    # "market": kept as a single string for backward compatibility with
    # every existing consumer of saved portfolio/history metadata (AI
    # Advisor caching, Investment Simulator, Portfolio History display,
    # tests) -- a "+"-joined display string for a multi-country portfolio.
    # "markets" (Issue #24 item 1) is the new canonical list, always safe
    # to use for anything that needs the real selected-country set.
    "market": " + ".join(selected_regions),
    "markets": list(selected_regions),
    "tickers": list(weights.keys()),
    "weights": dict(weights),
    "investment_amount": investment_amount,
    "base_currency": base_currency,
    "currency_adjusted": bool(st.session_state.get("opt_fx_result") and st.session_state["opt_fx_result"]["currency_adjusted"]),
    "investment_goal": investment_goal,
    "risk_tolerance": risk_tolerance,
    "investment_horizon": investment_horizon,
    "expected_return": exp_ret,
    "volatility": exp_vol,
    "sharpe_ratio": sharpe,
    "max_drawdown": _cp_max_drawdown,
    "largest_position": {"ticker": _diag["largest_ticker"], "weight": _diag["largest_weight"]},
    "effective_holdings": _diag["effective_holdings"],
    "historical_start_date": str(start_date),
    "historical_end_date": str(end_date),
    "generated_at": st.session_state.get("opt_generated_at"),
}
current_portfolio = st.session_state.current_portfolio

# ── Experiment metadata (Issue #20 section 9C) ──────────────────────────────
# Everything needed to understand/reproduce this specific run, saved as a
# single versioned JSON blob alongside the portfolio (src.database.
# save_portfolio(metadata=...)). Deliberately built from the SAME
# already-computed variables as current_portfolio above -- not
# independently re-derived -- so it can never drift from what the page
# actually used for this optimization.
_experiment_metadata = {
    "schema_version": 1,
    "historical_start_date": str(start_date),
    "historical_end_date": str(end_date),
    "market": " + ".join(selected_regions),
    "markets": list(selected_regions),
    "base_currency": base_currency,
    "currency_adjusted": bool(st.session_state.get("opt_fx_result") and st.session_state["opt_fx_result"]["currency_adjusted"]),
    # Issue #46 review item 2: the slider lets a user move risk_free_rate
    # away from the fetched/fallback _rf_info["rate"] it was initialized
    # to -- when that happens the actual assumption is a manual override,
    # not a FRED-sourced value, and must be recorded as such rather than
    # silently persisting the FRED source/date/status as if it still
    # described risk_free_rate. The FRED reference rate is kept separately
    # (risk_free_rate_fred_reference_rate) so the live/fallback context is
    # never lost even when overridden.
    "risk_free_rate": risk_free_rate,
    "risk_free_rate_is_manual_override": is_manual_override(_rf_info, risk_free_rate),
    "risk_free_rate_fred_reference_rate": _rf_info["rate"],
    "risk_free_rate_source": _rf_info["source"],
    "risk_free_rate_series": _rf_info["series_id"],
    "risk_free_rate_observed_date": _rf_info["observed_date"],
    "risk_free_rate_status": (
        "manual_override" if is_manual_override(_rf_info, risk_free_rate) else _rf_info["status"]
    ),
    "min_weight": min_weight,
    "max_weight": max_weight,
    "allow_short": allow_short,
    "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
    "covariance_estimator": "Sample covariance (historical, annualized)",
    "strategy": optimization_method,
    "asset_universe": list(weights.keys()),
    "generated_at": st.session_state.get("opt_generated_at"),
    "data_as_of": str(end_date),
    "app_version": APP_VERSION,
}

# ── Results Hero (Issue #24 item 3, sharpened in the visual-acceptance
# round item B1) ─────────────────────────────────────────────────────────────
# The FIRST thing visible after a successful build -- a strong, elevated,
# colored header whose subtitle is a compact, dynamically-built one-liner
# of exactly what was built (tickers · markets · horizon · amount), so a
# first-time user can immediately tell "this is the answer" AND what it's
# an answer to, without re-reading the (now collapsed) settings panel above.
_hero_tickers_line = " · ".join(list(weights.keys()))
_hero_regions_line = ", ".join(t_country(r) for r in selected_regions)
_hero_horizon_line = _horizon_labels.get(investment_horizon, investment_horizon)
_hero_amount_line = f"${investment_amount:,.0f} {base_currency}"
results_hero(
    t("opt_results_hero_title"),
    f"{_hero_tickers_line} · {_hero_regions_line} · {_hero_horizon_line} · {_hero_amount_line}",
)

# ── ONE hero metric + a smaller secondary row (Issue #24 visual-acceptance
# round item B1) ─────────────────────────────────────────────────────────────
# Expected Return is THE single visual protagonist of the results area --
# large, isolated typography, nothing else competing with it. Volatility /
# Sharpe / Diversification Ratio are real, decision-useful numbers too, but
# deliberately smaller and grouped in one plain row below -- never a 4th/5th
# equal-weight KPI card. "Method" isn't repeated here at all: it's already
# visible in the compact setup line above and the one-liner just above this.
results_hero_metric(t("metric_expected_annual_return"), f"{exp_ret:.2%}", color=COLORS["success"])
scol1, scol2, scol3 = st.columns(3)
with scol1:
    st.markdown(metric_card_html(t("metric_expected_volatility"), f"{exp_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with scol2:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sharpe:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with scol3:
    st.markdown(metric_card_html(t("metric_diversification_ratio"), f"{div_ratio:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
chart_caption(t("opt_kpi_row_caption"))

# ── Numerical Robustness Warning (Issue #45 item 1) ──────────────────────
# Computed from the RAW covariance matrix BEFORE any regularization (see
# run_optimization()/covariance_diagnostics()) -- a severe reading means
# the selected ETFs carry little independent information from each other,
# so a bounds-hugging (corner) solution reflects that redundancy rather
# than the solver "randomly" choosing a corner. Existing solver
# convergence/error handling (optimizer_failed above) is untouched.
_cov_diag = result.get("covariance_diagnostics")
_cov_diag_level = result.get("covariance_diagnostics_level")
if _cov_diag and _cov_diag_level in ("severe", "moderate"):
    _cond_str = f"{_cov_diag['condition_number']:.2e}" if _cov_diag["condition_number"] is not None else t("opt_covariance_cond_unavailable")
    _avg_corr_str = f"{_cov_diag['avg_pairwise_correlation']:.2%}" if _cov_diag["avg_pairwise_correlation"] is not None else "N/A"
    _cov_msg_kwargs = dict(
        rank=_cov_diag["rank"], n=_cov_diag["n_assets"], cond=_cond_str, avg_corr=_avg_corr_str,
    )
    if _cov_diag_level == "severe":
        st.warning(t("opt_covariance_warning_severe", **_cov_msg_kwargs))
    else:
        st.info(t("opt_covariance_info_moderate", **_cov_msg_kwargs))

# ── Methodology & Assumptions (M1) ──────────────────────────────────────────
# Compact, always-visible (independent of which workspace is open) disclosure
# of the ACTUAL calculation methodology -- see src/methodology.py, the single
# source of truth this panel and tests/test_methodology_m1.py both read from.
# Runtime values (dates, risk-free rate, bounds) are read straight off the
# already-computed prices_df/result/sidebar inputs, never recomputed or
# guessed, so this can never drift from what the KPIs above actually show.
with st.expander(t("opt_methodology_title"), expanded=False):
    st.caption(t("opt_methodology_subtitle"))
    _hist_start = prices_df.index.min().strftime("%Y-%m-%d")
    _hist_end = prices_df.index.max().strftime("%Y-%m-%d")
    _short_note = t("opt_methodology_short_note_on") if allow_short else t("opt_methodology_short_note_off")
    # The SLSQP + sidebar-bounds + risk-free-rate description is only
    # accurate for Maximum Sharpe / Minimum Volatility / Target Return
    # (run_optimization() in src/portfolio_optimizer.py). Equal Weight never
    # calls an optimizer at all (plain 1/N), and Risk Parity does use SLSQP
    # but with its OWN fixed [0.1%, 100%] bounds -- it ignores the sidebar's
    # min/max weight sliders entirely and doesn't use risk_free_rate in its
    # objective. Disclosing the same SLSQP/bounds/rf text for every method
    # would misrepresent both of these, so each gets its own accurate text.
    if optimization_method == "Equal Weight":
        _optimizer_desc = t("opt_methodology_optimizer_desc_equal_weight", method=t_opt_method(optimization_method))
    elif optimization_method == "Risk Parity":
        _optimizer_desc = t("opt_methodology_optimizer_desc_risk_parity", method=t_opt_method(optimization_method))
    else:
        _optimizer_desc = t(
            "opt_methodology_optimizer_desc", method=t_opt_method(optimization_method),
            rf=f"{risk_free_rate:.2%}", min=f"{min_weight:.0%}", max=f"{max_weight:.0%}", short_note=_short_note,
        )
    _fx_result_disclosure = st.session_state.get("opt_fx_result")
    if _fx_result_disclosure:
        _fx_currency_line = t(
            "opt_fx_methodology_value", base=base_currency,
            source=_fx_result_disclosure["fx_source"], method=_fx_result_disclosure["conversion_method"],
            adjusted=(t("opt_fx_yes") if _fx_result_disclosure["currency_adjusted"] else t("opt_fx_no")),
        )
    else:
        _fx_currency_line = t("opt_fx_not_needed", base=base_currency)
    if _cov_diag:
        _cov_cond_str = f"{_cov_diag['condition_number']:.2e}" if _cov_diag["condition_number"] is not None else t("opt_covariance_cond_unavailable")
        _cov_avg_corr_str = f"{_cov_diag['avg_pairwise_correlation']:.2%}" if _cov_diag["avg_pairwise_correlation"] is not None else "N/A"
        _cov_diag_line = t(
            "opt_methodology_covariance_diag_value",
            rank=_cov_diag["rank"], n=_cov_diag["n_assets"], cond=_cov_cond_str, avg_corr=_cov_avg_corr_str,
        )
    else:
        _cov_diag_line = t("opt_covariance_cond_unavailable")
    st.markdown(
        f"- **{t('opt_methodology_return_label')}** — {t('opt_methodology_return_desc')}\n"
        f"- **{t('opt_methodology_covariance_label')}** — {t('opt_methodology_covariance_desc')}\n"
        f"- **{t('opt_methodology_covariance_diag_label')}** — {_cov_diag_line}\n"
        f"- **{t('opt_methodology_history_label')}** — "
        f"{t('opt_methodology_history_value', start=_hist_start, end=_hist_end, days=len(prices_df))}\n"
        f"- **{t('opt_methodology_optimizer_label')}** — {_optimizer_desc}\n"
        f"- **{t('opt_methodology_backtest_label')}** — {t('opt_methodology_backtest_value')}"
        f"{t('opt_methodology_backtest_sep')}{t('opt_methodology_backtest_desc')}\n"
        f"- **{t('opt_methodology_rfr_label')}** — "
        f"{t('opt_methodology_rfr_value', provenance=format_rate_provenance(_rf_info, get_language(), selected_rate=risk_free_rate))}\n"
        f"- **{t('opt_fx_methodology_label')}** — {_fx_currency_line}"
    )
    _validation = result.get("validation")
    if _validation is not None:
        if _validation.get("is_valid"):
            st.success(t("opt_methodology_validation_pass"))
        else:
            st.warning(t("opt_methodology_validation_fail", issues="; ".join(_validation.get("issues", []))))

# ── Bootstrap Weight Stability (Issue #50) ──────────────────────────────────
# On-demand sensitivity analysis, offered ONLY for Maximum Sharpe Ratio --
# this is the only method whose corner (bounds-hugging) solutions are
# materially driven by expected-return estimation error (see
# src/portfolio_optimizer.py's bootstrap_max_sharpe_weight_stability() and
# src/methodology.py's BOOTSTRAP_WEIGHT_STABILITY_METHODOLOGY, the single
# source of truth this panel and tests/test_bootstrap_weight_stability.py
# both read from). Kept strictly behind a button -- 200 extra SLSQP solves
# must never run on an ordinary page rerun -- and cached in session_state
# keyed on every input that fully determines the result (asset selection /
# returns window / risk-free rate / bounds / short flag / seed / n_bootstrap,
# via the same `run_inputs` fingerprint already used to invalidate opt_result),
# so re-opening the expander or an unrelated rerun never silently recomputes it.
if optimization_method == "Maximum Sharpe Ratio":
    with st.expander(t("opt_bootstrap_title"), expanded=False):
        st.caption(t("opt_bootstrap_explanation"))
        st.caption(t("opt_bootstrap_max_sharpe_only_note"))

        if "opt_bootstrap_result" not in st.session_state:
            st.session_state.opt_bootstrap_result = None
        if "opt_bootstrap_inputs" not in st.session_state:
            st.session_state.opt_bootstrap_inputs = None

        _bootstrap_inputs = run_inputs + (DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP)
        if st.button(t("opt_bootstrap_button"), key="opt_bootstrap_run_btn"):
            if st.session_state.opt_bootstrap_inputs != _bootstrap_inputs:
                with st.spinner(t("opt_bootstrap_running")):
                    _bootstrap_returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")
                    st.session_state.opt_bootstrap_result = bootstrap_max_sharpe_weight_stability(
                        _bootstrap_returns_df, risk_free_rate=risk_free_rate,
                        min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
                        n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_BOOTSTRAP_SEED,
                    )
                    st.session_state.opt_bootstrap_inputs = _bootstrap_inputs

        # Only show a result that was actually computed from the CURRENT
        # inputs -- e.g. a stale result from before the user changed the
        # sidebar settings is never displayed as if it still applied.
        _bootstrap_result = (
            st.session_state.opt_bootstrap_result
            if st.session_state.opt_bootstrap_inputs == _bootstrap_inputs else None
        )
        if _bootstrap_result is not None:
            st.caption(t(
                "opt_bootstrap_seed_note",
                seed=_bootstrap_result["seed"], n=_bootstrap_result["n_bootstrap"],
            ))
            st.plotly_chart(
                bootstrap_weight_stability_box_chart(_bootstrap_result["weights_by_ticker"]),
                use_container_width=True, key="opt_bootstrap_box_chart",
            )
            st.markdown(t(
                "opt_bootstrap_success_ratio",
                successful=_bootstrap_result["successful"], attempted=_bootstrap_result["attempted"],
            ))
            _bootstrap_summary = _bootstrap_result["summary"]
            if _bootstrap_summary is None:
                st.warning(t(
                    "opt_bootstrap_insufficient_note",
                    successful=_bootstrap_result["successful"], attempted=_bootstrap_result["attempted"],
                ))
            else:
                _bootstrap_summary_rows = [
                    {
                        t("opt_bootstrap_table_col_etf"): _tkr,
                        t("opt_bootstrap_table_col_median"): f"{_bootstrap_summary[_tkr]['median']:.2%}",
                        t("opt_bootstrap_table_col_p25"): f"{_bootstrap_summary[_tkr]['p25']:.2%}",
                        t("opt_bootstrap_table_col_p75"): f"{_bootstrap_summary[_tkr]['p75']:.2%}",
                        t("opt_bootstrap_table_col_iqr"): f"{_bootstrap_summary[_tkr]['iqr']:.2%}",
                        t("opt_bootstrap_table_col_p5"): f"{_bootstrap_summary[_tkr]['p5']:.2%}",
                        t("opt_bootstrap_table_col_p95"): f"{_bootstrap_summary[_tkr]['p95']:.2%}",
                        t("opt_bootstrap_table_col_min"): f"{_bootstrap_summary[_tkr]['min']:.2%}",
                        t("opt_bootstrap_table_col_max"): f"{_bootstrap_summary[_tkr]['max']:.2%}",
                    }
                    for _tkr in _bootstrap_result["tickers"]
                ]
                st.dataframe(pd.DataFrame(_bootstrap_summary_rows), hide_index=True, use_container_width=True)
                st.markdown(f"**{t('opt_bootstrap_interpretation_title')}**")
                st.caption(t("opt_bootstrap_interpretation_note"))
                st.caption(t("opt_bootstrap_michaud_citation"))

# ── Shared Portfolio Diagnosis rendering helpers (used by both Overview's
# compact snapshot and Backtest & Risk's detailed view -- PRODUCT SPEC
# section 16: never render the FULL diagnosis twice, only a compact
# snapshot in Overview) ──────────────────────────────────────────────────────
_DIAG_LEVEL_KEY = {"low": "opt_diag_concentration_low", "moderate": "opt_diag_concentration_moderate", "high": "opt_diag_concentration_high"}
_DIAG_LEVEL_COLOR = {"low": COLORS["success"], "moderate": COLORS["warning"], "high": COLORS["danger"]}
_DIAG_TOP2_STATUS_KEY = {"distributed": "opt_diag_top2_status_distributed", "moderate": "opt_diag_top2_status_moderate", "concentrated": "opt_diag_top2_status_concentrated"}
_DIAG_TOP2_STATUS_COLOR = {"distributed": COLORS["success"], "moderate": COLORS["warning"], "concentrated": COLORS["danger"]}
_DIAG_SUMMARY_KEY = {"concentrated": "opt_diag_summary_concentrated", "balanced": "opt_diag_summary_balanced", "moderate": "opt_diag_summary_moderate"}


def _diag_label(label: str, tooltip: str) -> str:
    """Metric label + a small native-HTML hover tooltip (`title` attribute
    -- no extra library). Kept to a short "info glyph" so it never
    dominates the card."""
    safe_tooltip = _html.escape(tooltip, quote=True)
    return f'{label}<span title="{safe_tooltip}" style="cursor:help;opacity:0.55;margin-left:4px;font-size:11px;">&#9432;</span>'


def _diag_summary_text():
    _key = _DIAG_SUMMARY_KEY[_diag["case"]]
    if _diag["case"] == "concentrated":
        return t(
            _key, selected_count=_diag["selected_holdings"],
            effective_holdings=f"{_diag['effective_holdings']:.2f}",
            largest_ticker=_diag["largest_ticker"], largest_weight=f"{_diag['largest_weight']:.2%}",
        )
    return t(_key)


def _render_diagnosis_cards():
    """The 5 Portfolio Diagnosis KPI cards -- identical content whether
    shown compactly (Overview) or in the detailed view (Backtest & Risk)."""
    _level_color = _DIAG_LEVEL_COLOR[_diag["concentration_level"]]
    _level_text = t(_DIAG_LEVEL_KEY[_diag["concentration_level"]])
    _top2_status = _diag["top2_status"]
    _top2_status_color = _DIAG_TOP2_STATUS_COLOR[_top2_status]
    _top2_status_text = t(_DIAG_TOP2_STATUS_KEY[_top2_status])

    dcol1, dcol2, dcol3, dcol4, dcol5 = st.columns(5)
    with dcol1:
        st.markdown(kpi_card(
            _diag_label(t("opt_diag_concentration_level_label"), t("opt_diag_tooltip_concentration_level")),
            f'<span style="color:{_level_color};">&#9679;</span> {_level_text}',
            color=_level_color, icon="shield",
        ), unsafe_allow_html=True)
    with dcol2:
        st.markdown(kpi_card(
            _diag_label(t("opt_col_largest_position"), t("opt_diag_tooltip_largest_position")),
            f"{_diag['largest_ticker']} {_diag['largest_weight']:.2%}",
            color=COLORS["primary"], icon="target",
        ), unsafe_allow_html=True)
    with dcol3:
        st.markdown(kpi_card(
            _diag_label(t("opt_diag_top2_concentration"), t("opt_diag_tooltip_top2_concentration")),
            f"{_diag['top2_concentration']:.2%}",
            sub=f'<span style="color:{_top2_status_color};font-weight:600;">{_top2_status_text}</span>',
            color=COLORS["purple"], icon="pie-chart",
        ), unsafe_allow_html=True)
    with dcol4:
        st.markdown(kpi_card(
            _diag_label(t("opt_diag_effective_holdings"), t("opt_diag_tooltip_effective_holdings")),
            f"{_diag['effective_holdings']:.2f} / {_diag['selected_holdings']}",
            color=COLORS["cyan"], icon="layers",
        ), unsafe_allow_html=True)
    with dcol5:
        st.markdown(kpi_card(
            _diag_label(t("opt_diag_active_etfs"), t("opt_diag_tooltip_active_etfs")),
            f"{_diag['active_holdings']} / {_diag['selected_holdings']}",
            color=COLORS["warning"], icon="bar-chart",
        ), unsafe_allow_html=True)


# Real underlying-holdings overlap explanation (Issue #20 section 3B): the
# weight-based diagnosis above ("balanced"/"moderate"/"concentrated") can
# call an equal-weight allocation across e.g. VOO/VTI/QQQ "balanced" by
# TICKER weight alone, while those ETFs' underlying holdings actually
# overlap heavily. This must never be asserted from vague wording -- only
# from the SAME real holdings-overlap data src.risk_analytics.
# holdings_overlap_matrix() already computes for Risk Analytics -- shown as
# an opt-in check (fetches each ETF's underlying holdings on demand) so it
# never runs automatically on every rerun, matching that page's pattern.
def _render_holdings_overlap_check(active_tickers: list) -> None:
    if len(active_tickers) < 2:
        return
    if st.checkbox(t("opt_diag_check_overlap_label"), value=False, key="opt_diag_check_overlap_cb"):
        with st.spinner(t("risk_loading_holdings_overlap")):
            overlap = holdings_overlap_matrix(active_tickers)
        available_scores = [r["overlap_score"] for r in overlap.values() if r["available"]]
        unavailable_pairs = [pair for pair, r in overlap.items() if not r["available"]]
        if not available_scores:
            st.caption(t("opt_diag_overlap_unavailable"))
            return
        avg_overlap = sum(available_scores) / len(available_scores)
        if avg_overlap >= 0.30:
            st.warning(t("opt_diag_overlap_high", overlap=f"{avg_overlap:.0%}"))
        else:
            st.caption(t("opt_diag_overlap_low", overlap=f"{avg_overlap:.0%}"))
        if unavailable_pairs:
            st.caption(t("opt_diag_overlap_partial", n=len(unavailable_pairs)))


# ── Shared Reference-Strategy Computation (Issue #41 items C/D) ─────────────
# Equal Weight / Maximum Sharpe Ratio / Minimum Volatility are the three
# parameter-free reference strategies used by BOTH Allocation's Strategy
# Comparison/Efficient Frontier sub-views AND Backtest & Risk's Historical
# Performance/Drawdown sub-views. Extracted here as the ONE shared
# computation (same run_optimization() + backtest_portfolio() calls on the
# SAME already-loaded prices_df/risk_free_rate/bounds, no re-download) so
# the two workspaces can never duplicate this formula or silently disagree
# with each other. Still only ever called from inside the specific
# sub-view that needs it -- never unconditionally -- so Weights/Overview/
# Save & Actions/Diagnosis never pay for these 3 extra optimizer solves.
_REFERENCE_METHODS = ["Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility"]
_REFERENCE_COLORS = {
    "Equal Weight": COLORS["text_muted"],
    "Maximum Sharpe Ratio": COLORS["warning"],
    "Minimum Volatility": COLORS["purple"],
}


def _compute_reference_strategies() -> dict:
    """Weights/metrics/backtest for each of _REFERENCE_METHODS, computed
    fresh from the current prices_df/risk_free_rate/bounds. Purely
    informational -- never writes to st.session_state.opt_result /
    opt_run_inputs / prices_df, so it cannot alter the user's actual
    selected strategy."""
    results = {}
    for _rm in _REFERENCE_METHODS:
        _rr = run_optimization(
            prices_df=prices_df, method=_rm, risk_free_rate=risk_free_rate,
            min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
        )
        _rw = _rr["weights"]
        _largest_ticker = max(_rw, key=_rw.get) if _rw else None
        _largest_weight = _rw.get(_largest_ticker, 0.0) if _largest_ticker else 0.0
        _rbt = backtest_portfolio(prices_df, _rw, investment_amount)
        _rmdd = maximum_drawdown(_rbt["Portfolio Value"]) if not _rbt.empty else None
        results[_rm] = {
            "weights": _rw, "expected_return": _rr["expected_return"],
            "expected_volatility": _rr["expected_volatility"], "sharpe_ratio": _rr["sharpe_ratio"],
            "largest_ticker": _largest_ticker, "largest_weight": _largest_weight, "max_drawdown": _rmdd,
            "backtest_df": _rbt,
        }
    return results


def _build_backtest_reference_lines(reference_results: dict) -> list:
    """Return [(label, backtest_df, color, dash, width), ...] for Backtest &
    Risk's Historical Performance / Drawdown Analysis charts -- the current
    strategy's own already-computed `backtest_df` plus the UNIQUE reference
    strategies not already shown as the current one (Issue #41 item D: the
    self-comparison bug where a current method identical to the equal-
    weight baseline produced two indistinguishable "Equal Weight" traces).
    Each canonical method name appears in this list at most once, so no
    legend label/trace can ever collide with another.

    - If the current method IS one of _REFERENCE_METHODS, all three
      reference strategies are shown once each, with the current one
      visually emphasized (solid, primary color, thicker, a trailing
      " ★" marker) instead of being duplicated as a separate line.
    - Otherwise (Target Return / Risk Parity), the current strategy is
      shown as its own emphasized line PLUS all three reference
      strategies (4 unique lines total).

    The actual dedup decision (which methods appear, and which is current)
    is delegated to src.portfolio_optimizer.build_backtest_reference_plan()
    -- pure, Streamlit-free logic shared with tests/test_portfolio_optimizer.py
    -- this function only attaches the already-computed DataFrame/styling
    per planned method.
    """
    plan = build_backtest_reference_plan(optimization_method, _REFERENCE_METHODS)
    lines = []
    for _method, _is_current in plan:
        _label = t_opt_method(_method) + (" ★" if _is_current else "")
        _df = backtest_df if _is_current else reference_results[_method]["backtest_df"]
        _color = COLORS["primary"] if _is_current else _REFERENCE_COLORS[_method]
        _dash = None if _is_current else "dash"
        _width = 2.5 if _is_current else 1.5
        lines.append((_label, _df, _color, _dash, _width))
    return lines


# ── Top-Level Workspace Navigation ───────────────────────────────────────────
# st.segmented_control (NOT st.tabs -- see module docstring). Canonical
# English values live in session_state; `_ws_labels` is precomputed ONCE
# per render outside the format_func lambda -- a format_func that calls
# t()/get_language() fresh, inside the lambda, has been confirmed (Global
# ETF Universe round) to corrupt AppTest's widget-state reconciliation
# between reruns, so every segmented_control on this page follows this
# precomputed-dict pattern.
# Reduced from 5 to 4 top-level workspaces (Issue #24 visual-acceptance
# round item B3): "Strategy Lab" is no longer a top-level tab -- its two
# views (Strategy Comparison / Efficient Frontier) now live INSIDE the
# Allocation workspace as a secondary sub-nav, alongside the allocation
# table/donut as a third "Weights" sub-view (see the Allocation branch
# below), since all three are fundamentally about "what should the
# portfolio's weights be", not a separate top-level concern.
_OPT_WORKSPACES = ["Overview", "Allocation", "Backtest & Risk", "Save & Actions"]
_opt_ws_labels = {
    "Overview": t("opt_ws_overview"), "Allocation": t("opt_ws_allocation"),
    "Backtest & Risk": t("opt_ws_backtest_risk"), "Save & Actions": t("opt_ws_save_actions"),
}
# Shadow-state protected (same _shadow_default() pattern as every other
# sidebar control on this page): without it, a language switch that also
# flips the sidebar language selectbox's own stored value out of sync with
# get_language() triggers a fresh st.rerun() from inside language_selector()
# BEFORE this widget is reached on that pass, which can otherwise reset it
# to `default` instead of preserving the user's chosen workspace (PRODUCT
# SPEC section 13: switching workspaces/language must never reset the other).
_wsk, _wsv = _shadow_default("opt_workspace", "Overview")
opt_workspace = st.segmented_control(
    "opt_workspace_nav", _OPT_WORKSPACES, default=_wsv if _wsv in _OPT_WORKSPACES else "Overview",
    format_func=lambda w: _opt_ws_labels.get(w, w), key="opt_workspace", label_visibility="collapsed",
)
st.session_state[_wsk] = opt_workspace or "Overview"
if not opt_workspace:
    opt_workspace = "Overview"

st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# OVERVIEW -- "What portfolio did the optimizer build, and what are the key
# implications?"
# ══════════════════════════════════════════════════════════════════════════
if opt_workspace == "Overview":
    section_header(t("opt_diagnosis_title"), t("opt_diagnosis_subtitle"))
    st.caption(t("opt_diag_current_strategy", method=t_opt_method(optimization_method)))
    _render_diagnosis_cards()
    st.markdown(f"**{t('opt_diag_insight_title')}**  \n{_diag_summary_text()}")

    # Optional compact allocation preview (top 5 by weight, bars -- NOT the
    # full table+donut, which lives in the Allocation workspace).
    section_header(t("opt_overview_allocation_preview_title"))
    _preview_items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
    _preview_html = "".join(
        '<div style="margin-bottom:10px;">'
        '<div style="display:flex;justify-content:space-between;font-size:12.5px;'
        f'color:var(--text);margin-bottom:4px;"><span><b>{tk}</b></span>'
        f'<span style="font-weight:700;">{w:.2%}</span></div>'
        '<div style="background:var(--border);border-radius:999px;height:8px;overflow:hidden;">'
        f'<div style="background:{COLORS["primary"]};width:{min(100.0, w * 100):.2f}%;'
        'height:100%;border-radius:999px;"></div></div></div>'
        for tk, w in _preview_items
    )
    st.markdown(_preview_html, unsafe_allow_html=True)
    st.caption(t("opt_overview_allocation_preview_note"))

# ══════════════════════════════════════════════════════════════════════════
# ALLOCATION -- Weights (table + breakdown) / Compare Strategies / Efficient
# Frontier, as three sub-views of ONE top-level workspace (Issue #24
# visual-acceptance round item B3: "Strategy Lab" is no longer its own
# top-level tab -- folded in here since all three are fundamentally about
# "what should the portfolio's weights be").
# ══════════════════════════════════════════════════════════════════════════
elif opt_workspace == "Allocation":
    _ALLOC_VIEWS = ["Weights", "Comparison", "Frontier"]
    _alloc_labels = {
        "Weights": t("opt_stratlab_nav_weights"), "Comparison": t("opt_stratlab_nav_comparison"),
        "Frontier": t("opt_stratlab_nav_frontier"),
    }
    _avk, _avv = _shadow_default("opt_alloc_view", "Weights")
    alloc_view = st.segmented_control(
        "alloc_nav", _ALLOC_VIEWS, default=_avv if _avv in _ALLOC_VIEWS else "Weights",
        format_func=lambda w: _alloc_labels.get(w, w), key="opt_alloc_view", label_visibility="collapsed",
    ) or "Weights"
    st.session_state[_avk] = alloc_view

    if alloc_view == "Weights":
        section_header(t("opt_allocation_title"))
        col_left, col_right = st.columns([1, 1])

        with col_left:
            with chart_card(t("opt_allocation_table_card"), t("opt_allocation_table_holdings", count=len(weights))):
                alloc_df = weights_to_dataframe(weights, investment_amount)
                st.dataframe(alloc_df[["Ticker", "Weight", "Allocation ($)"]].style.hide(axis="index"),
                             use_container_width=True)
                chart_caption(t("opt_allocation_table_caption"))

        with col_right:
            with chart_card(t("opt_allocation_breakdown_card"), t_opt_method(optimization_method)):
                fig_donut = allocation_donut_chart(weights, "")
                st.plotly_chart(fig_donut, use_container_width=True, key="opt_allocation_donut")
                chart_caption(t("opt_allocation_donut_caption"))
                _alloc_context_text = "; ".join(f"{tk}: {w:.2%}" for tk, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True))
                ai_interpret_button("opt_allocation_ai_interpret", st.session_state, _alloc_context_text)

    else:
        # Strategy Comparison's 3 extra optimizer solves + 3 extra backtests
        # (Round 2B-1) are computed ONLY when the Comparison/Frontier
        # sub-view is active -- Weights (and every other top-level
        # workspace) never triggers this. Shared by BOTH sub-views (the
        # Frontier view needs these same results for its strategy markers),
        # so it's computed once here rather than duplicated per sub-view.
        #
        # Informational decision-support only -- this section NEVER writes to
        # st.session_state.opt_result / opt_run_inputs / prices_df, so it
        # cannot alter the user's actual selected strategy. Reuses the SAME
        # run_optimization() already verified in Round 2A (no duplicated
        # optimization math) on the SAME already-loaded `prices_df` (no
        # re-download, no extra network requests).
        _COMPARISON_METHODS = _REFERENCE_METHODS
        _CONCENTRATION_THRESHOLD = 0.50
        _CARD_DESC_KEYS = {
            "Equal Weight": "opt_card_desc_equal_weight",
            "Maximum Sharpe Ratio": "opt_card_desc_max_sharpe",
            "Minimum Volatility": "opt_card_desc_min_vol",
        }
        # Shared with Backtest & Risk's Historical/Drawdown sub-views (see
        # _compute_reference_strategies() above) so the two workspaces never
        # duplicate this formula or disagree with each other.
        _comparison_results = _compute_reference_strategies()
        # Winners are derived from the ACTUAL calculated values above -- never
        # assumed or hard-coded.
        _best_sharpe_method = max(_comparison_results, key=lambda m: _comparison_results[m]["sharpe_ratio"])
        _lowest_vol_method = min(_comparison_results, key=lambda m: _comparison_results[m]["expected_volatility"])

        if alloc_view == "Comparison":
            section_header(t("opt_strategy_comparison_title"), t("opt_strategy_comparison_subtitle"))
            st.caption(t("opt_strategy_comparison_scope_note"))

            _comp_cols = st.columns(3)
            for _idx, _cm in enumerate(_COMPARISON_METHODS):
                _cdata = _comparison_results[_cm]
                _is_current = (_cm == optimization_method)
                _badges = []
                if _cm == _best_sharpe_method:
                    _badges.append(t("opt_badge_best_sharpe"))
                if _cm == _lowest_vol_method:
                    _badges.append(t("opt_badge_lowest_risk"))
                if _cm == "Equal Weight":
                    _badges.append(t("opt_badge_balanced"))
                if _cdata["largest_weight"] > _CONCENTRATION_THRESHOLD:
                    _badges.append(t("opt_badge_higher_concentration"))
                _badge_html = "".join(
                    '<span style="display:inline-block;background:var(--surface-2);color:var(--text-secondary);'
                    f'font-size:10px;font-weight:600;padding:2px 8px;border-radius:999px;margin:2px 4px 0 0;">{b}</span>'
                    for b in _badges
                )
                _border = "border:1.5px solid var(--primary);" if _is_current else "border:1px solid var(--border);"
                _current_tag = (
                    '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.05em;margin-bottom:4px;">{t("opt_current_strategy_label")}</div>'
                ) if _is_current else ""

                with _comp_cols[_idx]:
                    st.markdown(
                        f'<div style="background:var(--surface);{_border}border-radius:var(--radius-lg);'
                        'padding:14px 16px;margin:4px 0;min-height:196px;">'
                        f'{_current_tag}'
                        f'<div style="color:var(--text);font-weight:800;font-size:14px;margin-bottom:6px;">{_opt_method_labels[_cm]}</div>'
                        f'<div style="color:var(--text-secondary);font-size:11.5px;line-height:1.5;margin-bottom:8px;">{t(_CARD_DESC_KEYS[_cm])}</div>'
                        '<div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:3px;">'
                        f'<span>{t("metric_expected_annual_return")}</span><span style="color:var(--text);font-weight:700;">{_cdata["expected_return"]:.2%}</span></div>'
                        '<div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:3px;">'
                        f'<span>{t("metric_sharpe_ratio")}</span><span style="color:var(--text);font-weight:700;">{_cdata["sharpe_ratio"]:.2f}</span></div>'
                        '<div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-secondary);margin-bottom:8px;">'
                        f'<span>{t("opt_col_largest_position")}</span><span style="color:var(--text);font-weight:700;">{_cdata["largest_ticker"]} {_cdata["largest_weight"]:.2%}</span></div>'
                        f'<div>{_badge_html}</div></div>',
                        unsafe_allow_html=True,
                    )

            with chart_card(t("opt_comparison_table_card")):
                _table_rows = []
                for _cm in _COMPARISON_METHODS:
                    _cdata = _comparison_results[_cm]
                    _strategy_display = _opt_method_labels[_cm] + (" ★" if _cm == optimization_method else "")
                    _row = {
                        t("opt_col_strategy"): _strategy_display,
                        t("metric_expected_annual_return"): f"{_cdata['expected_return']:.2%}",
                        t("metric_expected_volatility"): f"{_cdata['expected_volatility']:.2%}",
                        t("metric_sharpe_ratio"): f"{_cdata['sharpe_ratio']:.2f}",
                        t("opt_col_largest_position"): f"{_cdata['largest_ticker']} {_cdata['largest_weight']:.2%}",
                    }
                    if _cdata["max_drawdown"] is not None:
                        _row[t("metric_maximum_drawdown")] = f"{_cdata['max_drawdown']:.2%}"
                    _table_rows.append(_row)
                _comparison_df = pd.DataFrame(_table_rows)
                st.dataframe(_comparison_df.style.hide(axis="index"), use_container_width=True)
                st.caption(f"★ {t('opt_current_strategy_label')}: {t_opt_method(optimization_method)}")
                chart_caption(t("opt_comparison_table_caption"))
                _comparison_context_text = "; ".join(
                    f"{_opt_method_labels[_cm]}: {t('metric_expected_annual_return')} "
                    f"{_comparison_results[_cm]['expected_return']:.2%}, "
                    f"{t('metric_sharpe_ratio')} {_comparison_results[_cm]['sharpe_ratio']:.2f}, "
                    f"{t('opt_col_largest_position')} {_comparison_results[_cm]['largest_ticker']} "
                    f"{_comparison_results[_cm]['largest_weight']:.2%}"
                    for _cm in _COMPARISON_METHODS
                )
                ai_interpret_button("opt_comparison_ai_interpret", st.session_state, _comparison_context_text)

        else:  # Frontier
            section_header(t("opt_efficient_frontier_title"), t("opt_efficient_frontier_sub", count=f"{n_simulations:,}"))
            n_tickers = len(prices_df.columns)

            with st.spinner(t("msg_running_optimization")):
                returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")
                mean_returns = returns_df.mean().values
                cov_df = covariance_matrix(prices_df)
                cov = cov_df.values.copy()

                if cov.shape != (n_tickers, n_tickers):
                    error_state(
                        t("msg_no_price_data_title"),
                        f"Internal error: covariance matrix shape {cov.shape} does not "
                        f"match {n_tickers} selected ETFs. Please re-run the optimization."
                    )
                    st.stop()

                if not np.isfinite(cov).all():
                    diag_nan = ~np.isfinite(np.diag(cov))
                    if diag_nan.any():
                        bad_tickers = [prices_df.columns[i] for i in range(n_tickers) if diag_nan[i]]
                        error_state(
                            t("msg_no_price_data_title"),
                            f"No valid price data for: {', '.join(bad_tickers)}. "
                            "Remove these tickers or widen the date range, then re-run "
                            "the optimization."
                        )
                    else:
                        error_state(
                            t("msg_no_price_data_title"),
                            "Some selected ETFs have no overlapping trading dates with "
                            "each other. Widen the date range or choose ETFs with more "
                            "shared trading history."
                        )
                    st.stop()

                cov += np.eye(n_tickers) * 1e-8
                mc_df = monte_carlo_simulation(mean_returns, cov, n_simulations, risk_free_rate)

                # ~40 points, within the 30-60 range called for, kept cheap
                # since each point is one small SLSQP solve on already-loaded
                # data (no re-download) -- but still only computed when this
                # specific sub-view is active (PRODUCT SPEC section 15).
                _EF_FRONTIER_POINTS = 40
                frontier_df = compute_efficient_frontier(
                    mean_returns, cov, n_points=_EF_FRONTIER_POINTS,
                    min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
                )

            with chart_card(t("opt_efficient_frontier_card")):
                fig_ef = efficient_frontier_chart(
                    mc_df=mc_df, frontier_df=frontier_df, strategy_results=_comparison_results,
                    strategy_labels=_opt_method_labels, current_method=optimization_method,
                )
                st.plotly_chart(fig_ef, use_container_width=True, key="opt_efficient_frontier")
                if frontier_df is None or len(frontier_df) < 2:
                    st.info(t("opt_frontier_insufficient_points"))
                chart_caption(t("opt_efficient_frontier_caption"))

                # ── Dynamic correlation / diversification insight (Issue #41
                # item E) -- computed from the ACTUAL selected-ETF pairwise
                # return correlations in THIS prices_df, never a hardcoded
                # VOO/VTI/SPY assumption. A near-flat-looking frontier is
                # frequently just a symptom of a highly-correlated selection
                # -- this makes that visible instead of leaving the reader to
                # guess why the curve looks the way it does.
                _corr_df = correlation_matrix(prices_df)
                _avg_corr = average_pairwise_correlation(prices_df)
                if _avg_corr is not None:
                    _corr_pairs = [
                        (_corr_df.columns[i], _corr_df.columns[j], _corr_df.iloc[i, j])
                        for i in range(len(_corr_df.columns))
                        for j in range(i + 1, len(_corr_df.columns))
                        if pd.notna(_corr_df.iloc[i, j])
                    ]
                    _highest_pair = max(_corr_pairs, key=lambda c: c[2])
                    _corr_level = correlation_diversification_level(_avg_corr)
                    if _corr_level == "high":
                        _corr_insight = t(
                            "opt_frontier_correlation_high", avg=f"{_avg_corr:.2f}",
                            a=_highest_pair[0], b=_highest_pair[1], pair_corr=f"{_highest_pair[2]:.2f}",
                        )
                    elif _corr_level == "moderate":
                        _corr_insight = t("opt_frontier_correlation_moderate", avg=f"{_avg_corr:.2f}")
                    else:
                        _corr_insight = t("opt_frontier_correlation_low", avg=f"{_avg_corr:.2f}")
                    st.markdown(f"**{t('opt_frontier_correlation_title')}**  \n{_corr_insight}")

                _ef_context_text = (
                    f"{t('opt_current_strategy_label')}: {t_opt_method(optimization_method)}, "
                    f"{t('metric_expected_annual_return')} {exp_ret:.2%}, "
                    f"{t('metric_expected_volatility')} {exp_vol:.2%}, "
                    f"{t('metric_sharpe_ratio')} {sharpe:.2f}; " +
                    "; ".join(
                        f"{_opt_method_labels[_cm]}: {t('metric_expected_annual_return')} "
                        f"{_comparison_results[_cm]['expected_return']:.2%}, "
                        f"{t('metric_expected_volatility')} {_comparison_results[_cm]['expected_volatility']:.2%}"
                        for _cm in _COMPARISON_METHODS
                    )
                )
                ai_interpret_button("opt_frontier_ai_interpret", st.session_state, _ef_context_text)

            with chart_card(t("opt_how_to_read_title")):
                st.markdown(
                    '<div style="display:flex;flex-wrap:wrap;gap:6px 28px;font-size:12px;'
                    'color:var(--text-secondary);margin-bottom:6px;">'
                    f'<div><b style="color:var(--text);">{t("opt_how_to_read_left_label")}</b> — {t("opt_how_to_read_left_desc")}</div>'
                    f'<div><b style="color:var(--text);">{t("opt_how_to_read_up_label")}</b> — {t("opt_how_to_read_up_desc")}</div>'
                    f'<div><b style="color:var(--text);">{t_opt_method("Maximum Sharpe Ratio")}</b> — {t("opt_how_to_read_max_sharpe_desc")}</div>'
                    f'<div><b style="color:var(--text);">{t_opt_method("Minimum Volatility")}</b> — {t("opt_how_to_read_min_vol_desc")}</div>'
                    f'<div><b style="color:var(--text);">{t_opt_method("Equal Weight")}</b> — {t("opt_how_to_read_equal_weight_desc")}</div>'
                    '</div>'
                    f'<div style="font-size:11px;color:var(--text-muted);">{t("opt_how_to_read_disclaimer")}</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════
# BACKTEST & RISK -- Historical Performance / Drawdown Analysis / Portfolio
# Diagnosis
# ══════════════════════════════════════════════════════════════════════════
elif opt_workspace == "Backtest & Risk":
    _BT_VIEWS = ["Historical", "Drawdown", "Diagnosis"]
    _bt_labels = {
        "Historical": t("opt_backtest_nav_historical"), "Drawdown": t("opt_backtest_nav_drawdown"),
        "Diagnosis": t("opt_backtest_nav_diagnosis"),
    }
    _btk, _btv = _shadow_default("opt_backtest_view", "Historical")
    bt_view = st.segmented_control(
        "bt_nav", _BT_VIEWS, default=_btv if _btv in _BT_VIEWS else "Historical",
        format_func=lambda w: _bt_labels.get(w, w), key="opt_backtest_view", label_visibility="collapsed",
    ) or "Historical"
    st.session_state[_btk] = bt_view

    if bt_view in ("Historical", "Drawdown"):
        # Reference-strategy lines are only needed by these two chart
        # sub-views (not by Diagnosis, not by current_portfolio above) --
        # computed here rather than unconditionally, and shared with
        # Allocation > Comparison/Frontier via _compute_reference_strategies()
        # (Issue #41 item D: no more current-strategy-vs-itself self
        # comparison when the current method IS Equal Weight -- see
        # _build_backtest_reference_lines()'s docstring).
        _bt_reference_results = _compute_reference_strategies()
        _bt_lines = _build_backtest_reference_lines(_bt_reference_results)

        if bt_view == "Historical":
            section_header(t("opt_backtest_title"), t("opt_backtest_sub", method=t_opt_method(optimization_method)))
            st.caption(f"**{t('opt_methodology_backtest_value')}** — {t('opt_methodology_backtest_desc')}")
            if not backtest_df.empty:
                import plotly.graph_objects as go
                with chart_card(t("opt_backtest_card")):
                    fig_bt = go.Figure()
                    for _label, _df, _color, _dash, _width in _bt_lines:
                        if _df.empty:
                            continue
                        _line_style = dict(color=_color, width=_width)
                        if _dash:
                            _line_style["dash"] = _dash
                        fig_bt.add_trace(go.Scatter(
                            x=_df.index, y=_df["Portfolio Value"], name=_label, line=dict(**_line_style),
                        ))
                    fig_bt.update_layout(title=t("chart_portfolio_backtest_comparison"),
                                          xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"),
                                          height=420)
                    st.plotly_chart(apply_dark_theme(fig_bt), use_container_width=True, key="opt_backtest_growth")
                    chart_caption(t("opt_backtest_chart_caption"))
                    st.caption(f"★ {t('opt_current_strategy_label')}")

                bt_metrics = {
                    t("metric_total_return"): f"{backtest_df['Cumulative Return'].iloc[-1]:.2%}",
                    t("metric_annualized_return"): f"{annualized_return(backtest_df['Portfolio Value']):.2%}",
                    t("metric_annualized_volatility"): f"{annualized_volatility(backtest_df['Portfolio Value']):.2%}",
                    t("metric_sharpe_ratio"): f"{sharpe_ratio(backtest_df['Portfolio Value'], risk_free_rate):.2f}",
                    t("metric_maximum_drawdown"): f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}",
                    t("metric_final_value"): f"${backtest_df['Portfolio Value'].iloc[-1]:,.2f}",
                }
                st.markdown(f"**{t('opt_backtest_summary')}**")
                cols = st.columns(len(bt_metrics))
                for i, (k, v) in enumerate(bt_metrics.items()):
                    with cols[i]:
                        st.metric(k, v)
                _backtest_context_text = "; ".join(f"{k}: {v}" for k, v in bt_metrics.items())
                ai_interpret_button("opt_backtest_ai_interpret", st.session_state, _backtest_context_text)

        else:  # Drawdown
            section_header(t("opt_drawdown_comparison_card"))
            if not backtest_df.empty:
                import plotly.graph_objects as go
                with chart_card(t("opt_drawdown_comparison_card")):
                    fig_dd = go.Figure()
                    for _label, _df, _color, _dash, _width in _bt_lines:
                        if _df.empty:
                            continue
                        _dd_series = drawdown_series(_df["Portfolio Value"]) * 100
                        _line_style = dict(color=_color, width=_width)
                        if _dash:
                            _line_style["dash"] = _dash
                        _fill_kwargs = {"fillcolor": "rgba(148,163,184,0.1)"} if _dash else {}
                        fig_dd.add_trace(go.Scatter(
                            x=_dd_series.index, y=_dd_series, fill="tozeroy", name=_label,
                            line=dict(**_line_style), **_fill_kwargs,
                        ))
                    fig_dd.update_layout(title=t("chart_drawdown_comparison_pct"), xaxis_title=t("chart_date"),
                                          yaxis_title=t("chart_drawdown_pct"), height=420)
                    st.plotly_chart(apply_dark_theme(fig_dd), use_container_width=True, key="opt_backtest_drawdown")
                    chart_caption(t("opt_drawdown_chart_caption"))
                    st.caption(f"★ {t('opt_current_strategy_label')}")

                _dd_cols = st.columns(len(_bt_lines))
                _dd_context_parts = []
                for _dd_col, (_label, _df, _color, _dash, _width) in zip(_dd_cols, _bt_lines):
                    with _dd_col:
                        if not _df.empty:
                            _mdd_val = maximum_drawdown(_df["Portfolio Value"])
                            st.metric(f"{_label} {t('metric_maximum_drawdown')}", f"{_mdd_val:.2%}")
                            _dd_context_parts.append(f"{_label} {t('metric_maximum_drawdown')}: {_mdd_val:.2%}")
                _dd_context_text = "; ".join(_dd_context_parts)
                ai_interpret_button("opt_drawdown_ai_interpret", st.session_state, _dd_context_text)

    else:  # Diagnosis (detailed view)
        section_header(t("opt_diagnosis_title"), t("opt_diagnosis_subtitle"))
        st.caption(t("opt_diag_current_strategy", method=t_opt_method(optimization_method)))
        _render_diagnosis_cards()
        st.markdown(f"**{t('opt_diag_insight_title')}**  \n{_diag_summary_text()}")
        st.caption(t("opt_diag_weight_disclaimer"))
        _render_holdings_overlap_check([tk for tk, w in weights.items() if w > 0])

# ══════════════════════════════════════════════════════════════════════════
# SAVE & ACTIONS -- Next Steps + Save / Export
# ══════════════════════════════════════════════════════════════════════════
else:  # opt_workspace == "Save & Actions"
    # Computed once per rerun and reused by both the Quick Save button and
    # the named Save & Export form below (previously each ran its own
    # identical find_duplicate_portfolio() DB read against the same
    # current_portfolio weights/strategy/amount every rerun of this
    # workspace, including reruns triggered by unrelated widgets like
    # typing in the notes field).
    _current_dup_check = find_duplicate_portfolio(
        current_portfolio["weights"], current_portfolio["strategy"],
        current_portfolio["investment_amount"],
    )

    # ── Next Steps (Portfolio Handoff) ──────────────────────────────────────
    # Navigation uses st.switch_page() (the same programmatic-navigation
    # mechanism already used in app.py / src/ui.py), which preserves the
    # whole st.session_state -- including current_portfolio,
    # selected_region, and selected_etfs_<region> -- across the page
    # switch automatically, since it's the same session.
    section_header(t("opt_next_steps_title"))
    # CTA hierarchy (Issue #24 visual-acceptance round item B4): exactly ONE
    # visually dominant solid primary action -- "Simulate this portfolio" --
    # the other two are explicitly secondary/outline (type="secondary",
    # matching this design system's existing outline button style in
    # assets/style.css) so they never compete with it for attention.
    ns_col1, ns_col2, ns_col3 = st.columns(3)
    with ns_col1:
        if st.button(t("opt_next_steps_run_simulation"), use_container_width=True, type="primary", key="opt_next_run_sim"):
            st.switch_page("pages/3_Investment_Simulator.py")
    with ns_col2:
        if st.button(t("opt_next_steps_analyze_risk"), use_container_width=True, type="secondary", key="opt_next_analyze_risk"):
            st.switch_page("pages/4_Risk_Analytics.py")
    with ns_col3:
        if st.button(t("btn_save_portfolio"), use_container_width=True, type="secondary", key="opt_next_quick_save"):
            # One-click save with an auto-generated name (canonical English
            # strategy value). The named/annotated Save & Export form below
            # remains for users who want to customize the name or add notes.
            # Issue #20 section 9D: a one-click "quick save" is exactly the
            # flow most likely to spam identical rows (e.g. double-clicking,
            # or clicking again after just re-viewing a workspace) -- this
            # is the same pattern that produced 140 debug duplicates in the
            # committed demo database, so a quick save never silently
            # repeats an identical one; it points the user at the named
            # Save & Export form instead, which supports an explicit
            # "save anyway" confirmation.
            _dup = _current_dup_check
            if _dup:
                st.warning(t("opt_duplicate_save_warning", name=_dup["name"]))
            else:
                _quick_name = (
                    f"{current_portfolio['strategy'].replace(' ', '_')}_"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )
                _quick_success = save_portfolio(
                    name=_quick_name, weights=current_portfolio["weights"],
                    investment_amount=current_portfolio["investment_amount"],
                    optimization_method=current_portfolio["strategy"],
                    expected_return=current_portfolio["expected_return"],
                    expected_volatility=current_portfolio["volatility"],
                    sharpe_ratio=current_portfolio["sharpe_ratio"], notes="",
                    metadata=_experiment_metadata,
                )
                if _quick_success:
                    st.success(t("opt_portfolio_saved_success", name=_quick_name))
                else:
                    st.error(t("opt_portfolio_save_failed"))

    # ── Save & Download ──────────────────────────────────────────────────
    section_header(t("opt_save_export_title"))
    col1, col2, col3 = st.columns(3)

    with col1:
        portfolio_name = st.text_input(t("field_portfolio_name"), value=f"Portfolio_{current_portfolio['strategy'].replace(' ', '_')}")
        notes = st.text_area(t("field_notes_optional"), height=80)
        _named_dup = _current_dup_check
        _confirm_dup_save = False
        if _named_dup:
            st.warning(t("opt_duplicate_save_warning", name=_named_dup["name"]))
            # Keyed by the specific duplicate's ID (not a fixed key) --
            # Streamlit persists a checkbox's checked state across reruns
            # by key, so a fixed key would let confirming ONE duplicate
            # (portfolio A) silently pre-confirm a LATER, different
            # duplicate collision (portfolio B) in the same session without
            # the user ever re-confirming for B specifically, defeating the
            # whole point of this guard (Issue #20 section 9D).
            _confirm_dup_save = st.checkbox(
                t("opt_confirm_duplicate_save"), key=f"opt_confirm_dup_save_cb_{_named_dup['id']}",
            )
        if st.button(t("btn_save_portfolio"), type="primary", key="opt_save_export_btn"):
            # Sourced from the canonical current_portfolio object (built
            # above, unconditionally) -- not independently recomputed
            # locals -- so this can never drift from what Diagnosis /
            # Efficient Frontier / Strategy Comparison show for the same
            # run. Strategy is stored in English (the canonical value)
            # regardless of which language was active when saved.
            if _named_dup and not _confirm_dup_save:
                st.error(t("opt_duplicate_save_blocked"))
            else:
                success = save_portfolio(
                    name=portfolio_name, weights=current_portfolio["weights"],
                    investment_amount=current_portfolio["investment_amount"],
                    optimization_method=current_portfolio["strategy"],
                    expected_return=current_portfolio["expected_return"],
                    expected_volatility=current_portfolio["volatility"],
                    sharpe_ratio=current_portfolio["sharpe_ratio"], notes=notes,
                    metadata=_experiment_metadata,
                )
                if success:
                    st.success(t("opt_portfolio_saved_success", name=portfolio_name))
                else:
                    st.error(t("opt_portfolio_save_failed"))

    with col2:
        alloc_df_export = weights_to_dataframe(weights, investment_amount)
        alloc_csv = dataframe_to_csv(alloc_df_export)
        st.download_button(
            t("btn_download_allocation_csv"), alloc_csv, "portfolio_allocation.csv", "text/csv",
            use_container_width=True
        )

    with col3:
        try:
            # NOTE: these dict keys are fixed English labels that
            # src/report_generator.py matches against internally to build
            # the PDF's metrics table — they must NOT be translated, or the
            # PDF report would silently lose all metric rows.
            bt_metrics_full = {
                "Expected Annual Return": f"{exp_ret:.2%}",
                "Expected Annual Volatility": f"{exp_vol:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Diversification Ratio": f"{div_ratio:.2f}",
            }
            if not backtest_df.empty:
                bt_metrics_full["Maximum Drawdown"] = f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}"
                bt_metrics_full["Annualized Return (Backtest)"] = f"{annualized_return(backtest_df['Portfolio Value']):.2%}"

            pdf_bytes = generate_portfolio_report(
                portfolio_name=portfolio_name, weights=weights, metrics=bt_metrics_full,
                investment_amount=investment_amount, optimization_method=optimization_method, notes=notes,
            )
            st.download_button(
                t("btn_download_pdf_report"), pdf_bytes, f"{portfolio_name}.pdf", "application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(t("opt_pdf_unavailable", error=e))

disclaimer_box()
render_footer()
