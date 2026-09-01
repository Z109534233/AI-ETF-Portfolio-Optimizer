"""
Page 3: Investment Simulator
Future Monte Carlo projection AND real Historical Simulation (a genuine
monthly-rebalanced backtest against actual ETF price history -- see
src/simulator.py's historical_backtest(), not Monte Carlo).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.simulator import (
    simulate_investment, compound_growth_projection, scenario_comparison, MARKET_SCENARIOS,
    historical_backtest, find_common_data_range, prepare_historical_prices,
)
from src.database import save_simulation, init_database
from src.data_loader import download_etf_data
from src.etf_database import to_yahoo_symbol, rename_yahoo_columns
from src.financial_metrics import drawdown_series, ACTIVE_POSITION_TOLERANCE
from src.charts import (
    monte_carlo_paths_chart, future_value_distribution_chart, historical_growth_chart, apply_dark_theme,
)
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, dataframe_to_csv
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header, chart_card,
    render_footer, render_current_portfolio_handoff, error_state,
)
from src.theme import COLORS
from src.i18n import t, t_market_scenario, t_opt_method

# Display-currency symbol per market (Round 2 spec section 20) -- this
# ONLY changes which symbol is shown; every underlying calculation stays
# in whatever currency the ETF's own price data is already denominated in
# (Yahoo Finance returns .TW tickers in TWD, .L in GBX/GBP, etc.). The
# rest of this app (Portfolio Optimizer, and this page's Future Projection
# section) still hard-codes "$" everywhere regardless of market -- a
# pre-existing, cross-cutting assumption this round does not change; see
# the end-of-round report for the full limitation writeup.
_CURRENCY_SYMBOL_BY_MARKET = {"United States": "$", "Taiwan": "NT$", "United Kingdom": "£"}

st.set_page_config(
    page_title="Investment Simulator | AI ETF Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

load_css()
init_database()

page_header(t("sim_title"), t("sim_subtitle"))

# ── Current Portfolio handoff (Round 2B-4) ───────────────────────────────────
# Proof-of-handoff preview only -- the rest of this page remains fully
# self-contained and never requires a current_portfolio to exist; see
# render_current_portfolio_handoff() in src/ui.py.
render_current_portfolio_handoff(
    t("handoff_empty_state_title"), t("handoff_empty_state_body_sim"),
)
current_portfolio = st.session_state.get("current_portfolio")

st.info(t("sim_projection_disclaimer"))


def _shadow_default(name: str, default):
    """Same pattern as pages/2_Portfolio_Optimizer.py's helper of the same
    name (duplicated here, not imported -- a page script runs
    st.set_page_config() and other top-level Streamlit calls on import, so
    importing another page's module is unsafe). Needed because
    render_sidebar_nav()'s language selector can st.rerun() before a
    not-yet-reached widget on this page is instantiated, which would
    otherwise silently drop that widget's state back to its hard-coded
    default -- see Round 1 spec section 15: switching language/tabs/charts
    must not reset simulation inputs.
    """
    shadow_key = f"_{name}_shadow"
    if shadow_key not in st.session_state:
        st.session_state[shadow_key] = default
    return shadow_key, st.session_state[shadow_key]


# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()

    # ── Simulation Mode ───────────────────────────────────────────────────
    # Canonical (untranslated) values stored in session_state, per spec
    # section 15 -- "Future Projection" / "Historical Simulation", never a
    # translated label.
    _mode_options = ["Future Projection", "Historical Simulation"]
    _mode_labels = {"Future Projection": t("sim_mode_future"), "Historical Simulation": t("sim_mode_historical")}
    _mk, _mv = _shadow_default("simulation_mode", _mode_options[0])
    simulation_mode = st.selectbox(
        t("sim_mode_label"), _mode_options,
        index=_mode_options.index(_mv) if _mv in _mode_options else 0,
        format_func=lambda x: _mode_labels.get(x, x), key="simulation_mode",
    )
    st.session_state[_mk] = simulation_mode

    if simulation_mode == "Future Projection":
        # ── Primary Controls ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {t('sim_sidebar_params')}")

        _ik, _iv = _shadow_default("sim_initial_investment_val", 10000.0)
        initial_investment = st.number_input(
            t("sim_initial_investment"), 100.0, 1_000_000.0, _iv, 500.0, key="sim_initial_investment_val",
        )
        st.session_state[_ik] = initial_investment

        _mck, _mcv = _shadow_default("sim_monthly_contribution_val", 500.0)
        monthly_contribution = st.number_input(
            t("sim_monthly_contribution"), 0.0, 50000.0, _mcv, 100.0, key="sim_monthly_contribution_val",
        )
        st.session_state[_mck] = monthly_contribution

        _yk, _yv = _shadow_default("sim_investment_years_val", 20)
        years = st.slider(t("sim_investment_years"), 1, 40, _yv, key="sim_investment_years_val")
        st.session_state[_yk] = years

        # ── Projection Assumptions ───────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {t('sim_assumption_source_label')}")

        _portfolio_stats_available = (
            current_portfolio is not None
            and current_portfolio.get("expected_return") is not None
            and current_portfolio.get("volatility") is not None
        )
        _asrc_options = (
            (["Portfolio Historical Statistics"] if _portfolio_stats_available else [])
            + ["Market Scenario", "Custom Assumptions"]
        )
        _asrc_labels = {
            "Portfolio Historical Statistics": t("sim_assumption_source_portfolio"),
            "Market Scenario": t("sim_market_scenario"),
            "Custom Assumptions": t("sim_assumption_source_custom"),
        }
        _asrc_default = "Portfolio Historical Statistics" if _portfolio_stats_available else "Market Scenario"
        _ask, _asv = _shadow_default("projection_assumption_source", _asrc_default)
        _asrc_index = _asrc_options.index(_asv) if _asv in _asrc_options else 0
        projection_assumption_source = st.selectbox(
            t("sim_assumption_source_label"), _asrc_options, index=_asrc_index,
            format_func=lambda x: _asrc_labels.get(x, x),
            key="projection_assumption_source", label_visibility="collapsed",
        )
        st.session_state[_ask] = projection_assumption_source

        if not _portfolio_stats_available:
            st.caption(t("sim_portfolio_stats_unavailable"))

        if projection_assumption_source == "Portfolio Historical Statistics":
            # These are historical estimates from Portfolio Optimizer used
            # AS simulation assumptions -- never described as a prediction
            # of future returns (spec section 6).
            annual_return = current_portfolio["expected_return"]
            annual_volatility = current_portfolio["volatility"]
            st.caption(
                f"{t('metric_expected_annual_return')}: {annual_return:.2%} | "
                f"{t('metric_expected_volatility')}: {annual_volatility:.2%}"
            )
            st.caption(t("sim_portfolio_stats_source_note"))
        elif projection_assumption_source == "Market Scenario":
            _scenario_labels = {k: t_market_scenario(k) for k in MARKET_SCENARIOS}
            _scenario_options = list(MARKET_SCENARIOS.keys())
            _sk, _sv = _shadow_default("sim_market_scenario_choice", _scenario_options[0])
            scenario = st.selectbox(
                t("sim_market_scenario"), _scenario_options,
                index=_scenario_options.index(_sv) if _sv in _scenario_options else 0,
                format_func=lambda x: _scenario_labels.get(x, x), key="sim_market_scenario_choice",
            )
            st.session_state[_sk] = scenario
            annual_return = MARKET_SCENARIOS[scenario]["return"]
            annual_volatility = MARKET_SCENARIOS[scenario]["volatility"]
            st.caption(
                f"{t('sim_assumed_return_label')}: {annual_return:.1%} | "
                f"{t('sim_assumed_volatility_label')}: {annual_volatility:.1%}"
            )
            st.caption(t("sim_scenario_hypothetical_note"))
        else:  # Custom Assumptions
            _crk, _crv = _shadow_default("sim_custom_return_pct", 10.0)
            _custom_return_pct = st.slider(
                t("sim_expected_annual_return_pct"), -5.0, 30.0, _crv, 0.5, key="sim_custom_return_pct",
            )
            st.session_state[_crk] = _custom_return_pct

            _cvk, _cvv = _shadow_default("sim_custom_vol_pct", 15.0)
            _custom_vol_pct = st.slider(
                t("sim_expected_annual_volatility_pct"), 1.0, 50.0, _cvv, 0.5, key="sim_custom_vol_pct",
            )
            st.session_state[_cvk] = _custom_vol_pct

            annual_return = _custom_return_pct / 100
            annual_volatility = _custom_vol_pct / 100

        # ── Advanced Settings ─────────────────────────────────────────────
        with st.expander(t("opt_advanced_settings_title"), expanded=False):
            _infk, _infv = _shadow_default("sim_inflation_rate_pct_val", 2.5)
            _inflation_pct = st.slider(
                t("sim_inflation_rate_pct"), 0.0, 10.0, _infv, 0.25, key="sim_inflation_rate_pct_val",
            )
            st.session_state[_infk] = _inflation_pct
            inflation_rate = _inflation_pct / 100

            _feek, _feev = _shadow_default("sim_annual_fee_pct_val", 0.1)
            _fee_pct = st.slider(
                t("sim_annual_fee_pct"), 0.0, 3.0, _feev, 0.05, key="sim_annual_fee_pct_val",
            )
            st.session_state[_feek] = _fee_pct
            annual_fee = _fee_pct / 100

            _nsk, _nsv = _shadow_default("sim_number_of_simulations_val", 1000)
            n_simulations = st.slider(
                t("sim_number_of_simulations"), 200, 5000, _nsv, 100, key="sim_number_of_simulations_val",
            )
            st.session_state[_nsk] = n_simulations

        run_btn = st.button(t("btn_run_simulation"), type="primary", use_container_width=True, key="sim_run_btn")

    elif simulation_mode == "Historical Simulation":
        # ── Primary Controls ─────────────────────────────────────────────
        # Initial Investment / Monthly Contribution SHARE the same shadow
        # key as the Future Projection widgets above (different widget
        # `key=` since Streamlit needs a unique id per instantiated
        # widget, but the same underlying value) -- switching modes
        # preserves these amounts, per spec section 19.
        st.markdown("---")
        st.markdown(f"### {t('sim_sidebar_params')}")

        _ik, _iv = _shadow_default("sim_initial_investment_val", 10000.0)
        initial_investment = st.number_input(
            t("sim_initial_investment"), 100.0, 1_000_000.0, _iv, 500.0, key="hist_initial_investment_val",
        )
        st.session_state[_ik] = initial_investment

        _mck, _mcv = _shadow_default("sim_monthly_contribution_val", 500.0)
        monthly_contribution = st.number_input(
            t("sim_monthly_contribution"), 0.0, 50000.0, _mcv, 100.0, key="hist_monthly_contribution_val",
        )
        st.session_state[_mck] = monthly_contribution

        # ── Historical Simulation Period ─────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {t('hist_period_label')}")

        active_tickers = []
        hist_wide_prices = pd.DataFrame()
        hist_common_start = None
        hist_common_end = None
        hist_start_date = None
        hist_end_date = None

        if current_portfolio:
            active_tickers = [
                tk for tk, w in (current_portfolio.get("weights") or {}).items()
                if w > ACTIVE_POSITION_TOLERANCE
            ]

        if not active_tickers:
            st.caption(t("hist_requires_portfolio"))
        else:
            # A single wide-window download (start=2000-01-01) discovers
            # true per-ticker data availability in one shot; it's cached
            # by download_etf_data()'s @st.cache_data, so re-running this
            # page (language switch, other widget changes) doesn't
            # re-fetch. The user's chosen date range below is then just a
            # slice of this same DataFrame -- no second download.
            with st.spinner(t("hist_running")):
                _yahoo_tickers = [to_yahoo_symbol(tk) for tk in active_tickers]
                _raw_wide = download_etf_data(_yahoo_tickers, "2000-01-01", str(date.today()))
            if not _raw_wide.empty:
                hist_wide_prices = rename_yahoo_columns(_raw_wide)
                hist_wide_prices = hist_wide_prices[[tk for tk in active_tickers if tk in hist_wide_prices.columns]]
                hist_common_start, hist_common_end = find_common_data_range(hist_wide_prices)

            if hist_common_start is None:
                st.caption(t("hist_no_common_data"))
            else:
                _default_start = hist_common_start.date()
                _default_end = hist_common_end.date()
                _hsk, _hsv = _shadow_default("hist_start_date_val", _default_start)
                hist_start_date = st.date_input(
                    t("field_start_date"), value=_hsv, max_value=_default_end, key="hist_start_date_widget",
                )
                st.session_state[_hsk] = hist_start_date

                _hek, _hev = _shadow_default("hist_end_date_val", _default_end)
                hist_end_date = st.date_input(
                    t("field_end_date"), value=_hev, max_value=_default_end, key="hist_end_date_widget",
                )
                st.session_state[_hek] = hist_end_date

                if hist_start_date < _default_start:
                    st.caption(t("hist_start_date_constrained_msg"))
                st.caption(t("hist_data_limitation_note"))

        st.caption(f"{t('hist_rebalancing_label')}: {t('hist_rebalancing_monthly')}")

        run_btn_hist = st.button(
            t("btn_run_simulation"), type="primary", use_container_width=True, key="hist_run_btn",
        )

    render_sidebar_footer()

# ── Historical Simulation (Round 2: real backtest, no Monte Carlo) ──────────────
if simulation_mode == "Historical Simulation":
    if not active_tickers or hist_common_start is None or hist_start_date is None:
        if current_portfolio and active_tickers:
            st.info(t("hist_no_common_data"))
        elif current_portfolio:
            st.info(t("hist_requires_portfolio"))
        # else: the top-of-page handoff empty_state already explains this
        disclaimer_box()
        render_footer()
        st.stop()

    # Effective start/end never fall outside the common valid-data range,
    # even if a shadow-restored date from a previous session predates it.
    effective_start = max(hist_start_date, hist_common_start.date())
    effective_end = min(hist_end_date, hist_common_end.date()) if hist_end_date else hist_common_end.date()

    if "hist_result" not in st.session_state:
        st.session_state.hist_result = None

    if run_btn_hist or st.session_state.hist_result is None:
        with st.spinner(t("hist_running")):
            _missing = set(active_tickers) - set(hist_wide_prices.columns)
            if _missing:
                st.warning(f"No usable price data for: {', '.join(sorted(_missing))}. "
                           "These tickers were excluded from the historical simulation.")
            _prepared = prepare_historical_prices(
                hist_wide_prices, pd.Timestamp(effective_start), pd.Timestamp(effective_end),
            )
            active_weights = {
                tk: w for tk, w in current_portfolio["weights"].items() if tk in active_tickers
            }
            _bt = historical_backtest(
                _prepared, active_weights,
                initial_investment=initial_investment, monthly_contribution=monthly_contribution,
            )
        st.session_state.hist_result = _bt
        st.session_state.hist_params = {
            "initial_investment": initial_investment,
            "monthly_contribution": monthly_contribution,
            "strategy": current_portfolio.get("strategy"),
            "market": current_portfolio.get("market"),
            "active_weights": active_weights,
        }

    hist_result = st.session_state.hist_result
    if not hist_result or hist_result.get("history") is None or hist_result["history"].empty:
        error_state(t("msg_no_price_data_title"), t("hist_no_common_data"))
        disclaimer_box()
        render_footer()
        st.stop()

    history = hist_result["history"]
    hist_summary = hist_result["summary"]
    hist_params = st.session_state.hist_params
    _currency_symbol = _CURRENCY_SYMBOL_BY_MARKET.get(hist_params.get("market"), "$")

    # ── Allocation disclaimer (spec section 16: this is a fixed
    # hypothetical allocation from the CURRENT optimizer run, never a
    # claim that the optimizer would have chosen this historically) ──────
    _active_weights_text = " · ".join(
        f"{tk} {w:.2%}" for tk, w in sorted(hist_params["active_weights"].items(), key=lambda kv: -kv[1])
    )
    st.caption(f"{t('hist_allocation_disclaimer')} {_active_weights_text}")

    # ── Backtest Setup ────────────────────────────────────────────────────
    with chart_card(t("hist_backtest_setup_title")):
        _bt_setup_rows = [
            (t("sim_initial_investment"), f"{_currency_symbol}{hist_params['initial_investment']:,.0f}"),
            (t("sim_monthly_contribution"), f"{_currency_symbol}{hist_params['monthly_contribution']:,.0f}"),
            (t("hist_num_contributions"), f"{hist_summary['num_contributions']}"),
            (t("hist_total_invested"), f"{_currency_symbol}{hist_summary['total_invested']:,.0f}"),
            (t("field_start_date"), str(hist_summary["start_date"].date())),
            (t("field_end_date"), str(hist_summary["end_date"].date())),
            (t("hist_rebalancing_label"), t("hist_rebalancing_monthly")),
        ]
        _bt_items_html = "".join(
            f'<div style="min-width:130px;"><div style="font-size:11px;color:{COLORS["text_muted"]};">{label}</div>'
            f'<div style="font-size:13px;color:{COLORS["text"]};font-weight:600;">{value}</div></div>'
            for label, value in _bt_setup_rows
        )
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:10px 28px;">{_bt_items_html}</div>', unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────
    section_header(t("hist_results_title"))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_html(t("hist_total_invested"), f"{_currency_symbol}{hist_summary['total_invested']:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html(t("hist_final_value"), f"{_currency_symbol}{hist_summary['final_value']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
    with col3:
        _gain_color = COLORS["success"] if hist_summary["gain"] >= 0 else COLORS["danger"]
        st.markdown(metric_card_html(t("hist_investment_gain_loss"), f"{_currency_symbol}{hist_summary['gain']:,.0f}", color=_gain_color), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_html(t("hist_cumulative_return"), f"{hist_summary['cumulative_return']:.2%}", color=COLORS["purple"]), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if hist_summary.get("annualized_mwr") is not None:
            st.markdown(metric_card_html(t("hist_annualized_mwr"), f"{hist_summary['annualized_mwr']:.2%}", color=COLORS["warning"]), unsafe_allow_html=True)
        else:
            st.markdown(metric_card_html(t("hist_annualized_mwr"), "—", color=COLORS["text_muted"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html(t("metric_maximum_drawdown"), f"{hist_summary['max_drawdown']:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
    with col3:
        if hist_summary.get("best_year"):
            _by, _byr = hist_summary["best_year"]
            st.markdown(metric_card_html(t("hist_best_year"), f"{_by} ({_byr:.1%})", color=COLORS["success"]), unsafe_allow_html=True)

    if hist_summary.get("annualized_mwr") is None:
        st.caption(t("hist_mwr_unavailable"))
    if hist_summary.get("worst_year"):
        _wy, _wyr = hist_summary["worst_year"]
        st.caption(f"{t('hist_worst_year')}: {_wy} ({_wyr:.1%})")

    # ── Charts ────────────────────────────────────────────────────────────
    with chart_card(t("hist_growth_chart_title")):
        fig_hist_growth = historical_growth_chart(history, currency_symbol=_currency_symbol)
        st.plotly_chart(fig_hist_growth, use_container_width=True, key="hist_growth_chart")

    with chart_card(t("hist_drawdown_chart_title")):
        _hist_dd = drawdown_series(history["Portfolio Value"]) * 100
        fig_hist_dd = go.Figure()
        fig_hist_dd.add_trace(go.Scatter(
            x=_hist_dd.index, y=_hist_dd, fill="tozeroy",
            line=dict(color=COLORS["danger"], width=1.5), name=t("hist_drawdown_chart_title"),
        ))
        fig_hist_dd.update_layout(xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
        st.plotly_chart(apply_dark_theme(fig_hist_dd), use_container_width=True, key="hist_drawdown_chart")

    # ── Result Interpretation (deterministic, not generative) ──────────────
    _hist_interp_key = (
        "hist_summary_positive" if hist_summary["final_value"] >= hist_summary["total_invested"]
        else "hist_summary_negative"
    )
    st.markdown(f"**{t('hist_result_interpretation_title')}**  \n{t(_hist_interp_key)}")
    st.caption(t("hist_data_limitation_note"))

    disclaimer_box()
    render_footer()
    st.stop()

# ── Run Simulation ────────────────────────────────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None

if run_btn or st.session_state.sim_result is None:
    with st.spinner(t("sim_running_monte_carlo")):
        sim_result = simulate_investment(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            inflation_rate=inflation_rate,
            annual_fee=annual_fee,
            n_simulations=n_simulations
        )
        st.session_state.sim_result = sim_result
        st.session_state.sim_params = {
            "initial_investment": initial_investment,
            "monthly_contribution": monthly_contribution,
            "years": years,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "inflation_rate": inflation_rate,
            "annual_fee": annual_fee,
            "n_simulations": n_simulations,
            "assumption_source": projection_assumption_source,
            "portfolio_strategy": current_portfolio.get("strategy") if current_portfolio else None,
        }

sim_result = st.session_state.sim_result
if sim_result is None:
    st.info(t("msg_configure_and_run", action=t("btn_run_simulation")))
    st.stop()

summary = sim_result["summary"]
annual_table = sim_result["annual_table"]
paths_df = sim_result["paths"]
final_values = sim_result["all_final_values"]
total_contributed = summary["total_contributed"]
sim_params = st.session_state.sim_params

# ── Projection Setup (Round 1: assumption transparency) ──────────────────────
# Reflects exactly what drove the CURRENTLY DISPLAYED sim_result (frozen at
# the moment "Run Simulation" was last clicked, via sim_params above) --
# not the live sidebar widget values, which may have changed since without
# a re-run. This is what actually produced the numbers below.
_asrc_display_labels = {
    "Portfolio Historical Statistics": t("sim_assumption_source_portfolio"),
    "Market Scenario": t("sim_market_scenario"),
    "Custom Assumptions": t("sim_assumption_source_custom"),
}
_setup_portfolio_label = (
    t_opt_method(sim_params["portfolio_strategy"]) if sim_params.get("portfolio_strategy")
    else t("sim_projection_setup_no_portfolio")
)
_setup_rows = [
    (t("sim_projection_setup_portfolio_label"), _setup_portfolio_label),
    (t("sim_assumption_source_label"), _asrc_display_labels.get(sim_params.get("assumption_source"), "—")),
    (t("metric_expected_annual_return"), f"{sim_params['annual_return']:.2%}"),
    (t("metric_expected_volatility"), f"{sim_params['annual_volatility']:.2%}"),
    (t("sim_initial_investment"), f"${sim_params['initial_investment']:,.0f}"),
    (t("sim_monthly_contribution"), f"${sim_params['monthly_contribution']:,.0f}"),
    (t("sim_investment_years"), f"{sim_params['years']}"),
    (t("sim_inflation_rate_pct"), f"{sim_params['inflation_rate']:.2%}"),
    (t("sim_annual_fee_pct"), f"{sim_params['annual_fee']:.2%}"),
    (t("sim_number_of_simulations"), f"{sim_params['n_simulations']:,}"),
]
with chart_card(t("sim_projection_setup_title")):
    _setup_items_html = "".join(
        f'<div style="min-width:130px;"><div style="font-size:11px;color:{COLORS["text_muted"]};">{label}</div>'
        f'<div style="font-size:13px;color:{COLORS["text"]};font-weight:600;">{value}</div></div>'
        for label, value in _setup_rows
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:10px 28px;">{_setup_items_html}</div>',
        unsafe_allow_html=True,
    )

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("sim_results_title"), t("sim_results_sub", count=f"{n_simulations:,}", years=years))
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html(t("metric_median_final_value"), f"${summary['median_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_total_contributed"), f"${total_contributed:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_median_investment_gain"), f"${summary['median_gain']:,.0f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_inflation_adjusted_value"), f"${summary['real_median_final']:,.0f}", color=COLORS["warning"]), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(metric_card_html(t("metric_optimistic_90"), f"${summary['optimistic_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_pessimistic_10"), f"${summary['pessimistic_final']:,.0f}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    # Exact definition (see src/simulator.py): mean(final nominal value >
    # total nominal contributions) -- renamed from the ambiguous "Probability
    # of Profit" to state that definition directly (Round 1 spec section 10).
    st.markdown(metric_card_html(t("sim_prob_ending_above_contributions"), f"{summary['probability_profit']:.1%}", color=COLORS["primary"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("sim_projection_charts_title"))
with chart_card(t("sim_simulation_detail_card")):
    tab1, tab2, tab3, tab4 = st.tabs([
        t("sim_tab_monte_carlo"), t("sim_tab_compound_growth"), t("sim_tab_value_distribution"), t("sim_tab_annual_table")
    ])

    with tab1:
        fig_mc = monte_carlo_paths_chart(paths_df, t("chart_monte_carlo_simulation") + f" — {years}")
        st.plotly_chart(fig_mc, use_container_width=True, key="sim_monte_carlo_paths")

    with tab2:
        # Compound growth projection (deterministic)
        growth_df = compound_growth_projection(
            initial_investment, monthly_contribution, years,
            annual_return, annual_fee, inflation_rate
        )
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Balance"],
            name=t("chart_portfolio_balance"), line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Contributed"],
            name=t("chart_total_contributed"), line=dict(color=COLORS["text_muted"], width=2, dash="dash")
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Real Balance"],
            name=t("chart_real_value_inflation_adj"), line=dict(color=COLORS["warning"], width=1.5, dash="dot")
        ))
        fig_growth.update_layout(
            title=t("chart_portfolio_growth_deterministic"),
            xaxis_title=t("chart_years"), yaxis_title=t("chart_value_usd")
        )
        st.plotly_chart(apply_dark_theme(fig_growth), use_container_width=True, key="sim_compound_growth")

        # Contributions vs gains
        fig_bar = go.Figure()
        yearly = growth_df[growth_df["Year"] == growth_df["Year"].astype(int)]
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Contributed"],
                                  name=t("chart_contributed"), marker_color=COLORS["primary"]))
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Gain"].clip(lower=0),
                                  name=t("chart_investment_gain"), marker_color=COLORS["success"]))
        fig_bar.update_layout(title=t("chart_contributions_vs_gains"),
                               xaxis_title=t("chart_year"), yaxis_title=t("chart_value_usd"), barmode="stack")
        st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True, key="sim_contributions_vs_gains")

    with tab3:
        fig_dist = future_value_distribution_chart(final_values, total_contributed)
        st.plotly_chart(fig_dist, use_container_width=True, key="sim_value_distribution")

        # Percentile table
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_col = t("sim_percentile_col")
        pct_data = {
            percentile_col: [f"{p}th" for p in percentiles],
            t("sim_final_value_col"): [f"${np.percentile(final_values, p):,.0f}" for p in percentiles],
            t("sim_gain_col"): [f"${np.percentile(final_values, p) - total_contributed:,.0f}" for p in percentiles],
            t("sim_return_multiple_col"): [f"{np.percentile(final_values, p) / initial_investment:.1f}x" for p in percentiles],
        }
        st.markdown(f"**{t('sim_outcome_percentile_table')}**")
        st.dataframe(pd.DataFrame(pct_data).set_index(percentile_col), use_container_width=True)

    with tab4:
        st.markdown(f"**{t('sim_annual_balance_summary')}**")
        display_table = annual_table.copy()
        for col in ["Portfolio Value", "Total Contributed", "Investment Gain", "Real Value (Inflation-Adj.)"]:
            display_table[col] = display_table[col].apply(lambda x: f"${x:,.0f}")
        display_table["Return %"] = display_table["Return %"].apply(lambda x: f"{x:.1f}%")
        display_table = display_table.rename(columns={
            "Portfolio Value": t("metric_final_value"),
            "Total Contributed": t("chart_total_contributed"),
            "Investment Gain": t("chart_investment_gain"),
            "Real Value (Inflation-Adj.)": t("chart_real_value_inflation_adj"),
            "Return %": t("chart_return"),
            "Year": t("chart_year"),
        })
        st.dataframe(display_table.set_index(t("chart_year")), use_container_width=True)

        csv = dataframe_to_csv(annual_table)
        st.download_button(t("btn_download_annual_table"), csv, "simulation_annual.csv", "text/csv")

# ── Scenario Comparison ───────────────────────────────────────────────────────
section_header(t("sim_scenario_comparison_title"))
with st.spinner(t("msg_running_optimization")):
    scenario_df = scenario_comparison(initial_investment, monthly_contribution, years, annual_fee)
with chart_card(t("sim_scenario_comparison_card")):
    display_scenario_df = scenario_df.copy()
    display_scenario_df["Scenario"] = display_scenario_df["Scenario"].apply(t_market_scenario)
    st.dataframe(display_scenario_df.rename(columns={"Scenario": t("sim_market_scenario")}).set_index(t("sim_market_scenario")),
                 use_container_width=True)

# ── Save Simulation ───────────────────────────────────────────────────────────
section_header(t("sim_save_simulation_title"))
if st.button(t("btn_save_simulation_history")):
    success = save_simulation(
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
        expected_return=annual_return,
        final_value=summary["median_final"]
    )
    if success:
        st.success(t("sim_saved_success"))
    else:
        st.warning(t("sim_save_failed"))

disclaimer_box()
render_footer()
