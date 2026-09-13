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
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.portfolio_optimizer import (
    run_optimization, monte_carlo_simulation, backtest_portfolio,
    compute_efficient_frontier
)
from src.financial_metrics import (
    covariance_matrix, annualized_return, annualized_volatility,
    sharpe_ratio, maximum_drawdown, drawdown_series, portfolio_diagnosis
)
from src.database import save_portfolio, init_database
from src.report_generator import generate_portfolio_report
from src.charts import (
    efficient_frontier_chart, allocation_donut_chart,
    portfolio_growth_chart, drawdown_chart, apply_dark_theme
)
from src.utils import (
    load_css, page_header, disclaimer_box, dataframe_to_csv,
    weights_to_dataframe, get_date_range_defaults, metric_card_html
)
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state,
    region_selector, region_etf_options, region_etf_multiselect,
    kpi_card,
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

    # ── Build Your Portfolio ─────────────────────────────────────────────
    st.markdown(f"### {t('opt_build_portfolio_title')}")

    # 1. Market -- shared global state (src/ui.py), same canonical
    # st.session_state["selected_region"] used by ETF Analysis, Risk
    # Analytics, Machine Learning, and AI Advisor. Picking a market here
    # updates those pages too, and vice versa.
    selected_region, ALL_REGIONS_LABEL = region_selector()
    etf_options = region_etf_options(selected_region, ALL_REGIONS_LABEL)

    # 2. ETF Selection -- shared global state, same as above. Invalid
    # tickers from a since-changed market are dropped automatically since
    # region_etf_multiselect() filters against the current `etf_options`.
    selected_etfs = region_etf_multiselect(
        selected_region, etf_options, t("field_select_etfs"),
        help_text=t("opt_select_etfs_help"), n_default=5,
    )
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

        _fk, _fv = _shadow_default("opt_risk_free_rate", 5.0)
        risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, _fv, 0.25, key="opt_risk_free_rate_slider") / 100
        st.session_state[_fk] = risk_free_rate * 100

        _nsk, _nsv = _shadow_default("opt_n_simulations", 5000)
        n_simulations = st.slider(t("opt_mc_simulations"), 1000, 10000, _nsv, 500, key="opt_n_simulations_slider")
        st.session_state[_nsk] = n_simulations

    # ── Primary Action ────────────────────────────────────────────────────
    run_btn = st.button(t("btn_run_optimization"), type="primary", use_container_width=True,
                        key="opt_run_optimization_btn")

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_etfs) < 2:
    st.warning(t("msg_select_two_etfs"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── Compact Page Header: Portfolio Setup Summary ────────────────────────────
# Compact confirmation of "what portfolio am I currently building", shown
# before any optimization results -- not a results section, just an echo of
# the current inputs above. NOT a large hero section (PRODUCT SPEC section 2).
_market_display = t_country(selected_region) if selected_region != ALL_REGIONS_LABEL else selected_region
_setup_rows = [
    (t("opt_setup_label_market"), _market_display),
    (t("opt_setup_label_etfs"), " · ".join(selected_etfs)),
    (t("opt_investment_goal_label"), _goal_labels.get(investment_goal, investment_goal)),
    (t("opt_risk_tolerance_label"), _risk_labels.get(risk_tolerance, risk_tolerance)),
    (t("opt_investment_horizon_label"), _horizon_labels.get(investment_horizon, investment_horizon)),
    (t("opt_setup_label_amount"), f"${investment_amount:,.0f}"),
    (t("opt_setup_label_strategy"), _opt_method_labels.get(optimization_method, optimization_method)),
]
st.markdown(
    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
    'padding:12px 16px;margin:6px 0 16px 0;box-shadow:var(--shadow-sm);">'
    f'<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">{t("opt_setup_summary_title")}</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:20px;">'
    + "".join(
        '<div><div style="color:var(--text-muted);font-size:10px;margin-bottom:2px;">'
        f'{label}</div><div style="color:var(--text);font-weight:700;font-size:13px;">{value}</div></div>'
        for label, value in _setup_rows
    )
    + '</div></div>',
    unsafe_allow_html=True,
)

# ── State Management ──────────────────────────────────────────────────────────
# `run_inputs` fingerprints everything the optimization result depends on.
# Without this, changing the ETF selection, date range, or optimization
# method WITHOUT re-clicking "Run Optimization" would silently keep
# showing stale results computed from a previous, different selection.
# UNCHANGED this round (PRODUCT SPEC section 14: the Build button remains
# the explicit trigger; workspace navigation never re-runs this).
run_inputs = (
    tuple(sorted(selected_etfs)), str(start_date), str(end_date),
    optimization_method, round(min_weight, 6), round(max_weight, 6),
    allow_short, target_return_pct,
)

if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "opt_run_inputs" not in st.session_state:
    st.session_state.opt_run_inputs = None

inputs_changed = run_inputs != st.session_state.opt_run_inputs

if run_btn or inputs_changed or st.session_state.opt_result is None:
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

result = st.session_state.opt_result
prices_df = st.session_state.prices_df

if result is None or prices_df is None or prices_df.empty:
    st.info(t("msg_configure_and_run", action=t("btn_run_optimization")))
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
    "market": selected_region,
    "tickers": list(weights.keys()),
    "weights": dict(weights),
    "investment_amount": investment_amount,
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

# ── Core Result KPI Row (always visible, near the top -- PRODUCT SPEC
# section 3) ─────────────────────────────────────────────────────────────────
kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
with kcol1:
    st.markdown(metric_card_html(t("metric_expected_annual_return"), f"{exp_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with kcol2:
    st.markdown(metric_card_html(t("metric_expected_volatility"), f"{exp_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with kcol3:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sharpe:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with kcol4:
    st.markdown(metric_card_html(t("metric_diversification_ratio"), f"{div_ratio:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with kcol5:
    st.markdown(metric_card_html(t("metric_method"), t_opt_method(optimization_method), color=COLORS["warning"]), unsafe_allow_html=True)

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
    st.markdown(
        f"- **{t('opt_methodology_return_label')}** — {t('opt_methodology_return_desc')}\n"
        f"- **{t('opt_methodology_covariance_label')}** — {t('opt_methodology_covariance_desc')}\n"
        f"- **{t('opt_methodology_history_label')}** — "
        f"{t('opt_methodology_history_value', start=_hist_start, end=_hist_end, days=len(prices_df))}\n"
        f"- **{t('opt_methodology_optimizer_label')}** — {_optimizer_desc}\n"
        f"- **{t('opt_methodology_backtest_label')}** — {t('opt_methodology_backtest_value')}. "
        f"{t('opt_methodology_backtest_desc')}"
    )
    _validation = result.get("validation")
    if _validation is not None:
        if _validation.get("is_valid"):
            st.success(t("opt_methodology_validation_pass"))
        else:
            st.warning(t("opt_methodology_validation_fail", issues="; ".join(_validation.get("issues", []))))

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


# ── Top-Level Workspace Navigation ───────────────────────────────────────────
# st.segmented_control (NOT st.tabs -- see module docstring). Canonical
# English values live in session_state; `_ws_labels` is precomputed ONCE
# per render outside the format_func lambda -- a format_func that calls
# t()/get_language() fresh, inside the lambda, has been confirmed (Global
# ETF Universe round) to corrupt AppTest's widget-state reconciliation
# between reruns, so every segmented_control on this page follows this
# precomputed-dict pattern.
_OPT_WORKSPACES = ["Overview", "Allocation", "Strategy Lab", "Backtest & Risk", "Save & Actions"]
_opt_ws_labels = {
    "Overview": t("opt_ws_overview"), "Allocation": t("opt_ws_allocation"),
    "Strategy Lab": t("opt_ws_strategy_lab"), "Backtest & Risk": t("opt_ws_backtest_risk"),
    "Save & Actions": t("opt_ws_save_actions"),
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
# ALLOCATION -- full allocation table + breakdown
# ══════════════════════════════════════════════════════════════════════════
elif opt_workspace == "Allocation":
    section_header(t("opt_allocation_title"))
    col_left, col_right = st.columns([1, 1])

    with col_left:
        with chart_card(t("opt_allocation_table_card"), t("opt_allocation_table_holdings", count=len(weights))):
            alloc_df = weights_to_dataframe(weights, investment_amount)
            st.dataframe(alloc_df[["Ticker", "Weight", "Allocation ($)"]].style.hide(axis="index"),
                         use_container_width=True)

    with col_right:
        with chart_card(t("opt_allocation_breakdown_card"), t_opt_method(optimization_method)):
            fig_donut = allocation_donut_chart(weights, "")
            st.plotly_chart(fig_donut, use_container_width=True, key="opt_allocation_donut")

# ══════════════════════════════════════════════════════════════════════════
# STRATEGY LAB -- Strategy Comparison / Efficient Frontier
# ══════════════════════════════════════════════════════════════════════════
elif opt_workspace == "Strategy Lab":
    _STRATLAB_VIEWS = ["Comparison", "Frontier"]
    _stratlab_labels = {"Comparison": t("opt_stratlab_nav_comparison"), "Frontier": t("opt_stratlab_nav_frontier")}
    _slk, _slv = _shadow_default("opt_stratlab_view", "Comparison")
    stratlab_view = st.segmented_control(
        "stratlab_nav", _STRATLAB_VIEWS, default=_slv if _slv in _STRATLAB_VIEWS else "Comparison",
        format_func=lambda w: _stratlab_labels.get(w, w), key="opt_stratlab_view", label_visibility="collapsed",
    ) or "Comparison"
    st.session_state[_slk] = stratlab_view

    # Strategy Comparison's 3 extra optimizer solves + 3 extra backtests
    # (Round 2B-1) are computed ONLY when Strategy Lab is the active
    # workspace -- Overview/Allocation/Backtest & Risk/Save & Actions never
    # trigger this. Shared by BOTH Strategy Lab sub-views (the Frontier
    # view needs these same results for its strategy markers), so it's
    # computed once here rather than duplicated per sub-view.
    #
    # Informational decision-support only -- this section NEVER writes to
    # st.session_state.opt_result / opt_run_inputs / prices_df, so it
    # cannot alter the user's actual selected strategy. Reuses the SAME
    # run_optimization() already verified in Round 2A (no duplicated
    # optimization math) on the SAME already-loaded `prices_df` (no
    # re-download, no extra network requests).
    _COMPARISON_METHODS = ["Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility"]
    _CONCENTRATION_THRESHOLD = 0.50
    _CARD_DESC_KEYS = {
        "Equal Weight": "opt_card_desc_equal_weight",
        "Maximum Sharpe Ratio": "opt_card_desc_max_sharpe",
        "Minimum Volatility": "opt_card_desc_min_vol",
    }
    _comparison_results = {}
    for _cm in _COMPARISON_METHODS:
        _cr = run_optimization(
            prices_df=prices_df, method=_cm, risk_free_rate=risk_free_rate,
            min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
        )
        _cw = _cr["weights"]
        _largest_ticker = max(_cw, key=_cw.get) if _cw else None
        _largest_weight = _cw.get(_largest_ticker, 0.0) if _largest_ticker else 0.0
        _cbt = backtest_portfolio(prices_df, _cw, investment_amount)
        _cmdd = maximum_drawdown(_cbt["Portfolio Value"]) if not _cbt.empty else None
        _comparison_results[_cm] = {
            "weights": _cw, "expected_return": _cr["expected_return"],
            "expected_volatility": _cr["expected_volatility"], "sharpe_ratio": _cr["sharpe_ratio"],
            "largest_ticker": _largest_ticker, "largest_weight": _largest_weight, "max_drawdown": _cmdd,
        }
    # Winners are derived from the ACTUAL calculated values above -- never
    # assumed or hard-coded.
    _best_sharpe_method = max(_comparison_results, key=lambda m: _comparison_results[m]["sharpe_ratio"])
    _lowest_vol_method = min(_comparison_results, key=lambda m: _comparison_results[m]["expected_volatility"])

    if stratlab_view == "Comparison":
        section_header(t("opt_strategy_comparison_title"), t("opt_strategy_comparison_subtitle"))

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
        # The equal-weight comparison baseline is only needed by these two
        # chart sub-views (not by Diagnosis, not by current_portfolio
        # above) -- computed here rather than unconditionally.
        equal_weights = {tk: 1.0 / len(weights) for tk in weights.keys()}
        equal_backtest_df = backtest_portfolio(prices_df, equal_weights, investment_amount)

        if bt_view == "Historical":
            section_header(t("opt_backtest_title"), t("opt_backtest_sub", method=t_opt_method(optimization_method)))
            st.caption(f"**{t('opt_methodology_backtest_value')}** — {t('opt_methodology_backtest_desc')}")
            if not backtest_df.empty:
                import plotly.graph_objects as go
                with chart_card(t("opt_backtest_card")):
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=backtest_df.index, y=backtest_df["Portfolio Value"],
                        name=t_opt_method(optimization_method), line=dict(color=COLORS["primary"], width=2.5)
                    ))
                    if not equal_backtest_df.empty:
                        fig_bt.add_trace(go.Scatter(
                            x=equal_backtest_df.index, y=equal_backtest_df["Portfolio Value"],
                            name=t("chart_equal_weight"), line=dict(color=COLORS["text_muted"], width=1.5, dash="dash")
                        ))
                    fig_bt.update_layout(title=t("chart_portfolio_backtest_vs_equal"),
                                          xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"),
                                          height=420)
                    st.plotly_chart(apply_dark_theme(fig_bt), use_container_width=True, key="opt_backtest_growth")

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

        else:  # Drawdown
            section_header(t("opt_drawdown_comparison_card"))
            if not backtest_df.empty:
                import plotly.graph_objects as go
                with chart_card(t("opt_drawdown_comparison_card")):
                    fig_dd = go.Figure()
                    dd = drawdown_series(backtest_df["Portfolio Value"]) * 100
                    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                                 name=t_opt_method(optimization_method), line=dict(color=COLORS["danger"], width=1.5)))
                    if not equal_backtest_df.empty:
                        dd_eq = drawdown_series(equal_backtest_df["Portfolio Value"]) * 100
                        fig_dd.add_trace(go.Scatter(x=dd_eq.index, y=dd_eq, fill="tozeroy",
                                                     name=t("chart_equal_weight"), line=dict(color=COLORS["text_muted"], width=1.5),
                                                     fillcolor="rgba(148,163,184,0.1)"))
                    fig_dd.update_layout(title=t("chart_drawdown_comparison_pct"), xaxis_title=t("chart_date"),
                                          yaxis_title=t("chart_drawdown_pct"), height=420)
                    st.plotly_chart(apply_dark_theme(fig_dd), use_container_width=True, key="opt_backtest_drawdown")

                _dd_mcol1, _dd_mcol2 = st.columns(2)
                with _dd_mcol1:
                    st.metric(f"{t_opt_method(optimization_method)} {t('metric_maximum_drawdown')}",
                               f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}")
                with _dd_mcol2:
                    if not equal_backtest_df.empty:
                        st.metric(f"{t('chart_equal_weight')} {t('metric_maximum_drawdown')}",
                                   f"{maximum_drawdown(equal_backtest_df['Portfolio Value']):.2%}")

    else:  # Diagnosis (detailed view)
        section_header(t("opt_diagnosis_title"), t("opt_diagnosis_subtitle"))
        st.caption(t("opt_diag_current_strategy", method=t_opt_method(optimization_method)))
        _render_diagnosis_cards()
        st.markdown(f"**{t('opt_diag_insight_title')}**  \n{_diag_summary_text()}")
        st.caption(t("opt_diag_weight_disclaimer"))

# ══════════════════════════════════════════════════════════════════════════
# SAVE & ACTIONS -- Next Steps + Save / Export
# ══════════════════════════════════════════════════════════════════════════
else:  # opt_workspace == "Save & Actions"
    # ── Next Steps (Portfolio Handoff) ──────────────────────────────────────
    # Navigation uses st.switch_page() (the same programmatic-navigation
    # mechanism already used in app.py / src/ui.py), which preserves the
    # whole st.session_state -- including current_portfolio,
    # selected_region, and selected_etfs_<region> -- across the page
    # switch automatically, since it's the same session.
    section_header(t("opt_next_steps_title"))
    ns_col1, ns_col2, ns_col3 = st.columns(3)
    with ns_col1:
        if st.button(t("opt_next_steps_run_simulation"), use_container_width=True, key="opt_next_run_sim"):
            st.switch_page("pages/3_Investment_Simulator.py")
    with ns_col2:
        if st.button(t("opt_next_steps_analyze_risk"), use_container_width=True, key="opt_next_analyze_risk"):
            st.switch_page("pages/4_Risk_Analytics.py")
    with ns_col3:
        if st.button(t("btn_save_portfolio"), use_container_width=True, key="opt_next_quick_save"):
            # One-click save with an auto-generated name (canonical English
            # strategy value). The named/annotated Save & Export form below
            # remains for users who want to customize the name or add notes.
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
        if st.button(t("btn_save_portfolio"), type="primary", key="opt_save_export_btn"):
            # Sourced from the canonical current_portfolio object (built
            # above, unconditionally) -- not independently recomputed
            # locals -- so this can never drift from what Diagnosis /
            # Efficient Frontier / Strategy Comparison show for the same
            # run. Strategy is stored in English (the canonical value)
            # regardless of which language was active when saved.
            success = save_portfolio(
                name=portfolio_name, weights=current_portfolio["weights"],
                investment_amount=current_portfolio["investment_amount"],
                optimization_method=current_portfolio["strategy"],
                expected_return=current_portfolio["expected_return"],
                expected_volatility=current_portfolio["volatility"],
                sharpe_ratio=current_portfolio["sharpe_ratio"], notes=notes,
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
