"""
Page 2: Portfolio Optimizer
Mean-variance optimization, efficient frontier, and portfolio backtesting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.portfolio_optimizer import (
    run_optimization, monte_carlo_simulation, backtest_portfolio,
    compute_efficient_frontier
)
from src.financial_metrics import (
    covariance_matrix, annualized_return, annualized_volatility,
    sharpe_ratio, maximum_drawdown, drawdown_series
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
# Round 1 (Portfolio Builder / Input UX): progressive disclosure --
# essential investor-facing inputs (Build Your Portfolio) first, then
# Optimization Strategy, then Advanced Settings (collapsed by default).
# Every existing variable below (optimization_method, min_weight,
# max_weight, allow_short, target_return_pct, risk_free_rate, start_date,
# end_date, n_simulations, custom_ticker) keeps its exact name and is fed
# into the SAME calculation calls further down, unchanged -- only where
# and how each control is presented has moved.
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
    # touching that pipeline would silently mislabel the numbers. See the
    # end-of-round report for the currency-architecture note.
    _ak, _av = _shadow_default("investment_amount", 10000.0)
    investment_amount = st.number_input(
        t("field_investment_amount_usd"), min_value=100.0, max_value=10_000_000.0,
        value=_av, step=500.0, key="investment_amount",
    )
    st.session_state[_ak] = investment_amount

    # 4. Investment Goal -- portfolio metadata / user preference only in
    # Round 1. Does NOT alter the optimizer's math yet -- later rounds will
    # map it to portfolio interpretation and AI analysis.
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

    # 5. Risk Tolerance -- same Round 1 scope note as Investment Goal above:
    # metadata only, does not change optimization constraints yet.
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

    # 6. Investment Horizon -- same Round 1 scope note as above. This is the
    # investor's intended holding period, distinct from the Historical Data
    # Range in Advanced Settings below (a calculation input, not a
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
    # branches 1:1 -- Equal Weight, Maximum Sharpe Ratio, Minimum
    # Volatility, Target Return, Risk Parity) are shown; nothing added.
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

    # ── Advanced Settings (collapsed by default) ─────────────────────────
    # Historical Data Range, Risk-Free Rate, Min/Max ETF Weight, Allow
    # Short Selling, Monte Carlo Simulation Count -- same values,
    # constraints, and calculation behavior as before; only relocated here
    # for progressive disclosure.
    with st.expander(t("opt_advanced_settings_title"), expanded=False):
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

        _minwk, _minwv = _shadow_default("opt_min_weight", 0.0)
        min_weight = st.slider(t("opt_min_weight"), 0.0, 20.0, _minwv, 1.0, key="opt_min_weight_slider") / 100
        st.session_state[_minwk] = min_weight * 100

        _maxwk, _maxwv = _shadow_default("opt_max_weight", 100.0)
        max_weight = st.slider(t("opt_max_weight"), 10.0, 100.0, _maxwv, 5.0, key="opt_max_weight_slider") / 100
        st.session_state[_maxwk] = max_weight * 100

        _ashk, _ashv = _shadow_default("opt_allow_short", False)
        allow_short = st.checkbox(t("opt_allow_short"), value=_ashv, key="opt_allow_short_cb")
        st.session_state[_ashk] = allow_short

        _nsk, _nsv = _shadow_default("opt_n_simulations", 5000)
        n_simulations = st.slider(t("opt_mc_simulations"), 1000, 10000, _nsv, 500, key="opt_n_simulations_slider")
        st.session_state[_nsk] = n_simulations

    # ── Primary Action ────────────────────────────────────────────────────
    run_btn = st.button(t("btn_run_optimization"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_etfs) < 2:
    st.warning(t("msg_select_two_etfs"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── Portfolio Setup Summary ─────────────────────────────────────────────────
# Compact confirmation of "what portfolio am I currently building", shown
# before any optimization results (and before the Build button has
# necessarily been clicked) -- not a results section, just an echo of the
# current inputs above.
_market_display = t_country(selected_region) if selected_region != ALL_REGIONS_LABEL else selected_region
_setup_rows = [
    (t("opt_setup_label_market"), _market_display),
    (t("opt_setup_label_etfs"), " · ".join(selected_etfs)),
    (t("opt_investment_goal_label"), _goal_labels.get(investment_goal, investment_goal)),
    (t("opt_risk_tolerance_label"), _risk_labels.get(risk_tolerance, risk_tolerance)),
    (t("opt_investment_horizon_label"), _horizon_labels.get(investment_horizon, investment_horizon)),
    (t("opt_setup_label_amount"), f"${investment_amount:,.0f}"),
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
# showing stale results computed from a previous, different selection
# (st.button() only returns True on the exact rerun it was clicked in;
# every other widget change also triggers a rerun but leaves run_btn
# False, and the old code only recomputed when run_btn was True or no
# result existed yet). We now recompute whenever the actual inputs change.
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

        prices_df = clean_price_data(raw_prices)
        prices_df = rename_yahoo_columns(prices_df)
        prices_df = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]

        # ── Validate prices_df ──────────────────────────────────────────
        # clean_price_data() now drops any ticker with zero valid data, so
        # a ticker that failed to download (common on Streamlit Cloud when
        # Yahoo Finance rate-limits cloud IPs) is silently absent from
        # prices_df.columns rather than lingering as an all-NaN column.
        # Report exactly which requested tickers are missing.
        missing_tickers = [tk for tk in selected_etfs if tk not in prices_df.columns]
        if missing_tickers:
            st.warning(
                f"No usable price data for: {', '.join(missing_tickers)}. "
                "These tickers were excluded from the optimization."
            )

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
        # how="all" (not the pandas default how="any"): a single ticker
        # missing one date must not wipe out that date for every other
        # ticker too. See covariance_matrix() for the full rationale.
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

        if result.get("error"):
            st.warning(t("opt_note_prefix", error=result["error"]))

        st.session_state.opt_result = result
        st.session_state.prices_df = prices_df
        st.session_state.opt_run_inputs = run_inputs

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

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("opt_results_title"))
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(metric_card_html(t("metric_expected_annual_return"), f"{exp_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_expected_volatility"), f"{exp_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sharpe:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_diversification_ratio"), f"{div_ratio:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html(t("metric_method"), t_opt_method(optimization_method), color=COLORS["warning"]), unsafe_allow_html=True)

# ── Allocation Table & Donut ──────────────────────────────────────────────────
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

# ── Efficient Frontier ────────────────────────────────────────────────────────
section_header(t("opt_efficient_frontier_title"), t("opt_efficient_frontier_sub", count=f"{n_simulations:,}"))

n_tickers = len(prices_df.columns)

with st.spinner(t("msg_running_optimization")):
    # how="all" (not the pandas default how="any"): a single ticker missing
    # one date must not wipe out that date for every other ticker too.
    returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")
    mean_returns = returns_df.mean().values
    cov_df = covariance_matrix(prices_df)
    cov = cov_df.values

    # Defensive validation: covariance_matrix() guarantees cov.shape ==
    # (n_tickers, n_tickers) and run_optimization() already validated this
    # same data upstream, so these checks should never fire in practice --
    # but if prices_df ever reaches here in an unexpected state (e.g. a
    # future code change, or a Yahoo Finance edge case not seen before),
    # we show a clear error instead of crashing on `cov += np.eye(...)`
    # with an opaque numpy shape-mismatch exception.
    if cov.shape != (n_tickers, n_tickers):
        error_state(
            t("msg_no_price_data_title"),
            f"Internal error: covariance matrix shape {cov.shape} does not "
            f"match {n_tickers} selected ETFs. Please re-run the optimization."
        )
        st.stop()

    if not np.isfinite(cov).all():
        # A NaN on the diagonal means that ticker has no valid data at all;
        # a NaN only off the diagonal means two otherwise-fine tickers just
        # don't share any trading dates. Distinguish these so the message
        # names the actual problem ticker(s) instead of every ticker.
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

with chart_card(t("opt_efficient_frontier_card")):
    fig_ef = efficient_frontier_chart(mc_df, weights, None, mean_returns, cov)
    st.plotly_chart(fig_ef, use_container_width=True, key="opt_efficient_frontier")

# ── Backtest ──────────────────────────────────────────────────────────────────
section_header(t("opt_backtest_title"), t("opt_backtest_sub", method=t_opt_method(optimization_method)))

backtest_df = backtest_portfolio(prices_df, weights, investment_amount)
equal_weights = {tk: 1.0 / len(weights) for tk in weights.keys()}
equal_backtest_df = backtest_portfolio(prices_df, equal_weights, investment_amount)

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
                              xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"))
        st.plotly_chart(apply_dark_theme(fig_bt), use_container_width=True, key="opt_backtest_growth")

    # Drawdown comparison
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
        fig_dd.update_layout(title=t("chart_drawdown_comparison_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
        st.plotly_chart(apply_dark_theme(fig_dd), use_container_width=True, key="opt_backtest_drawdown")

    # Backtest metrics
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

# ── Save & Download ───────────────────────────────────────────────────────────
section_header(t("opt_save_export_title"))

col1, col2, col3 = st.columns(3)

with col1:
    portfolio_name = st.text_input(t("field_portfolio_name"), value=f"Portfolio_{optimization_method.replace(' ', '_')}")
    notes = st.text_area(t("field_notes_optional"), height=80)
    if st.button(t("btn_save_portfolio"), type="primary"):
        # NOTE: optimization_method is stored in English (the raw selectbox
        # value) so it stays consistent regardless of which language was
        # active when saved; it is translated only at display time via
        # t_opt_method() wherever it is shown (e.g. Portfolio History page).
        success = save_portfolio(
            name=portfolio_name,
            weights=weights,
            investment_amount=investment_amount,
            optimization_method=optimization_method,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            notes=notes
        )
        if success:
            st.success(t("opt_portfolio_saved_success", name=portfolio_name))
        else:
            st.error(t("opt_portfolio_save_failed"))

with col2:
    alloc_csv = dataframe_to_csv(alloc_df)
    st.download_button(
        t("btn_download_allocation_csv"),
        alloc_csv, "portfolio_allocation.csv", "text/csv",
        use_container_width=True
    )

with col3:
    try:
        # NOTE: these dict keys are fixed English labels that
        # src/report_generator.py matches against internally to build the
        # PDF's metrics table — they must NOT be translated, or the PDF
        # report would silently lose all metric rows.
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
            portfolio_name=portfolio_name,
            weights=weights,
            metrics=bt_metrics_full,
            investment_amount=investment_amount,
            optimization_method=optimization_method,
            notes=notes
        )
        st.download_button(
            t("btn_download_pdf_report"),
            pdf_bytes, f"{portfolio_name}.pdf", "application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(t("opt_pdf_unavailable", error=e))

disclaimer_box()
render_footer()

# ── TEMPORARY DIAGNOSTIC (deployment-mismatch investigation) ────────────────
# Literal string, deliberately NOT routed through t() -- this is the
# ground-truth marker for confirming which commit Streamlit Cloud is
# actually executing. Remove once the deployment mismatch is confirmed.
st.caption("BUILD: OPT-R1-I18N-20260901-A")
st.caption(
    f"I18N TEST: lang={get_language()} | "
    f"strategy={t('opt_strategy_title')} | "
    f"goal={t('opt_goal_growth')} | "
    f"risk={t('opt_risk_balanced')} | "
    f"horizon={t('opt_horizon_5y')}"
)
