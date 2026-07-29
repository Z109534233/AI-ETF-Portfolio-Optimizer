"""
Page 4: Risk Analytics
Comprehensive portfolio risk analysis with stress testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.etf_database import to_yahoo_symbol, rename_yahoo_columns, etf_display_label
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    maximum_drawdown, calmar_ratio, beta, alpha, value_at_risk, conditional_var,
    downside_deviation, tracking_error, information_ratio,
    correlation_matrix, covariance_matrix, drawdown_series, diversification_ratio
)
from src.charts import (
    correlation_heatmap, return_distribution_chart, drawdown_chart,
    rolling_metrics_chart, apply_dark_theme, CHART_COLORS
)
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state, style_signed_columns, region_selector
)
from src.theme import COLORS
from src.i18n import t

st.set_page_config(
    page_title="Risk Analytics | AI ETF Portfolio Optimizer",
    page_icon="🛡️",
    layout="wide"
)

load_css()

page_header(t("risk_title"), t("risk_subtitle"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('risk_sidebar_settings')}")

    # ── Region Selector (Global ETF Support, remembered across every page) ──
    selected_region, etf_options = region_selector()

    selected_etfs = st.multiselect(
        t("field_select_etfs"),
        options=etf_options,
        default=etf_options[:4],
        format_func=etf_display_label,
        help=t("risk_select_etfs_help"),
        key=f"risk_multiselect_{selected_region}",
    )

    custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # Portfolio weights
    st.markdown(f"#### {t('risk_weights_label')}")
    weights_input = {}
    if selected_etfs:
        equal_w = 1.0 / len(selected_etfs)
        for ticker in selected_etfs:
            w = st.slider(t("risk_weight_pct", ticker=ticker), 0.0, 100.0, equal_w * 100, 1.0, key=f"w_{ticker}")
            weights_input[ticker] = w / 100.0
        total_w = sum(weights_input.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(t("risk_weights_normalised_warning", total=f"{total_w:.1%}"))
            if total_w > 0:
                weights_input = {k: v / total_w for k, v in weights_input.items()}

    benchmark = st.selectbox(t("field_benchmark"), options=DEFAULT_ETFS, index=2,
                              format_func=etf_display_label)
    risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, 5.0, 0.25) / 100

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("field_start_date"), value=default_start)
    end_date = st.date_input(t("field_end_date"), value=default_end)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf"))
    st.stop()

# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner(t("msg_downloading_market_data")):
    all_tickers = list(set(selected_etfs + [benchmark]))
    # Map each display ticker to its actual Yahoo Finance-fetchable symbol
    # (e.g. "0050" -> "0050.TW"). Tickers not in the ETF database (including
    # the US-only benchmark) pass through unchanged.
    yahoo_tickers = [to_yahoo_symbol(tk) for tk in all_tickers]
    raw_prices = download_etf_data(yahoo_tickers, str(start_date), str(end_date))

if raw_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

prices = clean_price_data(raw_prices)
prices = rename_yahoo_columns(prices)
etf_prices = prices[[tk for tk in selected_etfs if tk in prices.columns]]
bench_prices = prices[benchmark].dropna() if benchmark in prices.columns else None

if etf_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

# Build portfolio price series
weights_arr = np.array([weights_input.get(tk, 0) for tk in etf_prices.columns])
if weights_arr.sum() > 0:
    weights_arr = weights_arr / weights_arr.sum()
returns_df = etf_prices.pct_change().dropna()
port_returns = (returns_df * weights_arr).sum(axis=1)
port_prices = (1 + port_returns).cumprod() * 100

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("risk_portfolio_metrics_title"))

ann_ret = annualized_return(port_prices)
ann_vol = annualized_volatility(port_prices)
sr = sharpe_ratio(port_prices, risk_free_rate)
so_r = sortino_ratio(port_prices, risk_free_rate)
mdd = maximum_drawdown(port_prices)
cal = calmar_ratio(port_prices)
var95 = value_at_risk(port_prices, 0.95)
cvar95 = conditional_var(port_prices, 0.95)
dd_dev = downside_deviation(port_prices, risk_free_rate)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html(t("metric_annualized_return"), f"{ann_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_annualized_volatility"), f"{ann_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sr:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_sortino_ratio"), f"{so_r:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_maximum_drawdown"), f"{mdd:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_calmar_ratio"), f"{cal:.2f}", color=COLORS["warning"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_var_95"), f"{var95:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_cvar_95"), f"{cvar95:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)

# Benchmark metrics
if bench_prices is not None and len(bench_prices) > 10:
    b = beta(port_prices, bench_prices)
    a = alpha(port_prices, bench_prices, risk_free_rate)
    te = tracking_error(port_prices, bench_prices)
    ir = information_ratio(port_prices, bench_prices)

    section_header(t("risk_benchmark_metrics_title"), t("risk_benchmark_metrics_sub", benchmark=benchmark))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_html(t("metric_beta"), f"{b:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html(t("metric_alpha"), f"{a:.2%}", color=COLORS["success"] if a >= 0 else COLORS["danger"]), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html(t("metric_tracking_error"), f"{te:.2%}", color=COLORS["warning"]), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_html(t("metric_information_ratio"), f"{ir:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("risk_charts_title"))
with chart_card(t("risk_detail_card")):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("risk_tab_drawdown"), t("risk_tab_rolling_metrics"), t("risk_tab_return_distribution"),
        t("risk_tab_correlation"), t("risk_tab_risk_contribution")
    ])

    with tab1:
        fig_dd = drawdown_chart(etf_prices)
        st.plotly_chart(fig_dd, use_container_width=True, key="risk_drawdown_all")

        # Portfolio drawdown
        dd_series = drawdown_series(port_prices) * 100
        fig_port_dd = go.Figure()
        fig_port_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill="tozeroy", name=t("chart_portfolio_drawdown"),
            line=dict(color=COLORS["danger"], width=2),
            fillcolor="rgba(248,113,113,0.15)"
        ))
        fig_port_dd.update_layout(title=t("chart_portfolio_drawdown_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
        st.plotly_chart(apply_dark_theme(fig_port_dd), use_container_width=True, key="risk_drawdown_portfolio")

    with tab2:
        col_sel = st.selectbox(t("risk_select_rolling_etf"), etf_prices.columns.tolist(), key="risk_rolling")
        window = st.slider(t("risk_rolling_window_days"), 21, 252, 63, key="risk_window")
        p = etf_prices[col_sel].dropna()
        if len(p) > window:
            fig_roll = rolling_metrics_chart(p, window)
            st.plotly_chart(fig_roll, use_container_width=True, key="risk_rolling_metrics")

        # Rolling beta
        if bench_prices is not None and len(bench_prices) > window:
            ret_etf = p.pct_change().dropna()
            ret_bench = bench_prices.pct_change().dropna()
            common = ret_etf.index.intersection(ret_bench.index)
            if len(common) > window:
                rolling_beta = pd.Series(index=common, dtype=float)
                for i in range(window, len(common)):
                    r_e = ret_etf.loc[common[i - window:i]]
                    r_b = ret_bench.loc[common[i - window:i]]
                    cov_mat = np.cov(r_e, r_b)
                    rolling_beta.iloc[i] = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 1.0
                rolling_beta = rolling_beta.dropna()
                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta,
                                               name=f"{t('chart_beta')} ({window}d)",
                                               line=dict(color=COLORS["primary"], width=2)))
                fig_beta.add_hline(y=1.0, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.6)
                fig_beta.update_layout(title=t("chart_rolling_beta_window", benchmark=benchmark, window=window),
                                        xaxis_title=t("chart_date"), yaxis_title=t("chart_beta"))
                st.plotly_chart(apply_dark_theme(fig_beta), use_container_width=True, key="risk_rolling_beta")

    with tab3:
        fig_dist = return_distribution_chart(etf_prices)
        st.plotly_chart(fig_dist, use_container_width=True, key="risk_return_distribution_all")

        # Portfolio return distribution
        fig_port_dist = go.Figure()
        fig_port_dist.add_trace(go.Histogram(
            x=port_returns * 100, nbinsx=60,
            marker_color=COLORS["primary"], opacity=0.8, name=t("chart_portfolio_returns")
        ))
        fig_port_dist.add_vline(x=float(var95 * 100), line_dash="dash", line_color=COLORS["danger"],
                                 annotation_text=f"{t('metric_var_95')}: {var95:.2%}")
        fig_port_dist.add_vline(x=float(cvar95 * 100), line_dash="dash", line_color=COLORS["warning"],
                                 annotation_text=f"{t('metric_cvar_95')}: {cvar95:.2%}")
        fig_port_dist.update_layout(title=t("chart_portfolio_daily_return_dist"),
                                     xaxis_title=t("chart_daily_return_pct"), yaxis_title=t("chart_frequency"))
        st.plotly_chart(apply_dark_theme(fig_port_dist), use_container_width=True, key="risk_return_distribution_portfolio")

    with tab4:
        if len(etf_prices.columns) >= 2:
            corr = correlation_matrix(etf_prices)
            fig_corr = correlation_heatmap(corr)
            st.plotly_chart(fig_corr, use_container_width=True, key="risk_correlation_heatmap")
        else:
            st.info(t("risk_select_2_correlation"))

    with tab5:
        # Risk contribution
        if len(etf_prices.columns) >= 2:
            cov = covariance_matrix(etf_prices).values
            cov += np.eye(len(etf_prices.columns)) * 1e-8
            port_vol_val = np.sqrt(weights_arr @ cov @ weights_arr)
            if port_vol_val > 0:
                marginal = cov @ weights_arr / port_vol_val
                risk_contrib = weights_arr * marginal
                risk_contrib_pct = risk_contrib / risk_contrib.sum()

                fig_rc = go.Figure(go.Bar(
                    x=etf_prices.columns.tolist(),
                    y=risk_contrib_pct * 100,
                    marker_color=CHART_COLORS[:len(etf_prices.columns)],
                    hovertemplate=f"<b>%{{x}}</b><br>{t('chart_risk_contribution_pct')}: %{{y:.2f}}%<extra></extra>"
                ))
                fig_rc.update_layout(title=t("chart_risk_contribution_by_etf"),
                                      xaxis_title=t("chart_etf"), yaxis_title=t("chart_risk_contribution_pct"))
                st.plotly_chart(apply_dark_theme(fig_rc), use_container_width=True, key="risk_contribution_bar")

# ── Stress Tests ──────────────────────────────────────────────────────────────
section_header(t("risk_stress_test_title"), t("risk_stress_test_caption"))

stress_scenarios = {
    t("risk_scenario_equity_decline"): -0.30,
    t("risk_scenario_rate_shock"): -0.15,
    t("risk_scenario_high_vol"): -0.20,
    t("risk_scenario_defensive"): 0.05,
    t("risk_scenario_2008"): -0.50,
    t("risk_scenario_covid"): -0.34,
    t("risk_scenario_tech_bubble"): -0.45,
}

scenario_col = t("risk_col_scenario")
shock_col = t("risk_col_market_shock")
beta_col = t("risk_col_portfolio_beta")
impact_col = t("risk_col_estimated_impact")
dollar_impact_col = t("risk_col_impact_10k")

stress_rows = []
for scenario_name, market_shock in stress_scenarios.items():
    # Approximate portfolio impact based on beta and weights
    if bench_prices is not None:
        b_val = beta(port_prices, bench_prices)
    else:
        b_val = 1.0
    port_impact = market_shock * b_val
    dollar_impact = port_impact * 10000  # Assume $10,000 portfolio
    stress_rows.append({
        scenario_col: scenario_name,
        shock_col: f"{market_shock:.0%}",
        beta_col: f"{b_val:.2f}",
        impact_col: f"{port_impact:.2%}",
        dollar_impact_col: f"${dollar_impact:,.0f}",
    })

stress_df = pd.DataFrame(stress_rows).set_index(scenario_col)
with chart_card(t("risk_stress_test_impact_card")):
    st.dataframe(
        style_signed_columns(stress_df, [impact_col, dollar_impact_col]),
        use_container_width=True,
    )

disclaimer_box()
render_footer()
