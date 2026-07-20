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

st.set_page_config(
    page_title="Risk Analytics | AI ETF Portfolio Optimizer",
    page_icon="🛡️",
    layout="wide"
)

load_css()

page_header(
    "Risk Analytics",
    "Portfolio risk metrics, drawdown analysis, and stress testing",
    "🛡️"
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Portfolio Settings")

    selected_etfs = st.multiselect(
        "Select ETFs",
        options=DEFAULT_ETFS,
        default=["VOO", "QQQ", "BND", "GLD"],
        help="Select ETFs to analyse"
    )

    custom_ticker = st.text_input("Add Custom Ticker", placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # Portfolio weights
    st.markdown("#### Portfolio Weights")
    weights_input = {}
    if selected_etfs:
        equal_w = 1.0 / len(selected_etfs)
        for ticker in selected_etfs:
            w = st.slider(f"{ticker} Weight (%)", 0.0, 100.0, equal_w * 100, 1.0, key=f"w_{ticker}")
            weights_input[ticker] = w / 100.0
        total_w = sum(weights_input.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.1%}. They will be normalised.")
            if total_w > 0:
                weights_input = {k: v / total_w for k, v in weights_input.items()}

    benchmark = st.selectbox("Benchmark", options=DEFAULT_ETFS, index=2)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning("Please select at least one ETF.")
    st.stop()

# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner("Downloading market data..."):
    all_tickers = list(set(selected_etfs + [benchmark]))
    raw_prices = download_etf_data(all_tickers, str(start_date), str(end_date))

if raw_prices.empty:
    st.error("No price data available.")
    st.stop()

prices = clean_price_data(raw_prices)
etf_prices = prices[[t for t in selected_etfs if t in prices.columns]]
bench_prices = prices[benchmark].dropna() if benchmark in prices.columns else None

if etf_prices.empty:
    st.error("No valid price data for selected ETFs.")
    st.stop()

# Build portfolio price series
weights_arr = np.array([weights_input.get(t, 0) for t in etf_prices.columns])
if weights_arr.sum() > 0:
    weights_arr = weights_arr / weights_arr.sum()
returns_df = etf_prices.pct_change().dropna()
port_returns = (returns_df * weights_arr).sum(axis=1)
port_prices = (1 + port_returns).cumprod() * 100

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("### Portfolio Risk Metrics")

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
    st.markdown(metric_card_html("Annualized Return", f"{ann_ret:.2%}", color="#10B981"), unsafe_allow_html=True)
    st.markdown(metric_card_html("Annualized Volatility", f"{ann_vol:.2%}", color="#EF4444"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Sharpe Ratio", f"{sr:.2f}", color="#3B82F6"), unsafe_allow_html=True)
    st.markdown(metric_card_html("Sortino Ratio", f"{so_r:.2f}", color="#8B5CF6"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Maximum Drawdown", f"{mdd:.2%}", color="#EF4444"), unsafe_allow_html=True)
    st.markdown(metric_card_html("Calmar Ratio", f"{cal:.2f}", color="#F59E0B"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("VaR (95%)", f"{var95:.2%}", color="#EF4444"), unsafe_allow_html=True)
    st.markdown(metric_card_html("CVaR (95%)", f"{cvar95:.2%}", color="#EF4444"), unsafe_allow_html=True)

# Benchmark metrics
if bench_prices is not None and len(bench_prices) > 10:
    b = beta(port_prices, bench_prices)
    a = alpha(port_prices, bench_prices, risk_free_rate)
    te = tracking_error(port_prices, bench_prices)
    ir = information_ratio(port_prices, bench_prices)

    st.markdown("### Benchmark-Relative Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_html("Beta", f"{b:.2f}", color="#3B82F6"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html("Alpha", f"{a:.2%}", color="#10B981" if a >= 0 else "#EF4444"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html("Tracking Error", f"{te:.2%}", color="#F59E0B"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_html("Information Ratio", f"{ir:.2f}", color="#8B5CF6"), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Drawdown", "Rolling Metrics", "Return Distribution", "Correlation", "Risk Contribution"])

with tab1:
    fig_dd = drawdown_chart(etf_prices)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Portfolio drawdown
    dd_series = drawdown_series(port_prices) * 100
    fig_port_dd = go.Figure()
    fig_port_dd.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series,
        fill="tozeroy", name="Portfolio Drawdown",
        line=dict(color="#EF4444", width=2),
        fillcolor="rgba(239,68,68,0.15)"
    ))
    fig_port_dd.update_layout(title="Portfolio Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)")
    st.plotly_chart(apply_dark_theme(fig_port_dd), use_container_width=True)

with tab2:
    col_sel = st.selectbox("Select ETF for rolling metrics", etf_prices.columns.tolist(), key="risk_rolling")
    window = st.slider("Rolling Window (days)", 21, 252, 63, key="risk_window")
    p = etf_prices[col_sel].dropna()
    if len(p) > window:
        fig_roll = rolling_metrics_chart(p, window)
        st.plotly_chart(fig_roll, use_container_width=True)

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
                                           name=f"Rolling Beta ({window}d)",
                                           line=dict(color="#3B82F6", width=2)))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#9CA3AF", opacity=0.6)
            fig_beta.update_layout(title=f"Rolling Beta vs {benchmark} ({window}-Day Window)",
                                    xaxis_title="Date", yaxis_title="Beta")
            st.plotly_chart(apply_dark_theme(fig_beta), use_container_width=True)

with tab3:
    fig_dist = return_distribution_chart(etf_prices)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Portfolio return distribution
    fig_port_dist = go.Figure()
    fig_port_dist.add_trace(go.Histogram(
        x=port_returns * 100, nbinsx=60,
        marker_color="#3B82F6", opacity=0.8, name="Portfolio Returns"
    ))
    fig_port_dist.add_vline(x=float(var95 * 100), line_dash="dash", line_color="#EF4444",
                             annotation_text=f"VaR 95%: {var95:.2%}")
    fig_port_dist.add_vline(x=float(cvar95 * 100), line_dash="dash", line_color="#F59E0B",
                             annotation_text=f"CVaR 95%: {cvar95:.2%}")
    fig_port_dist.update_layout(title="Portfolio Daily Return Distribution",
                                 xaxis_title="Daily Return (%)", yaxis_title="Frequency")
    st.plotly_chart(apply_dark_theme(fig_port_dist), use_container_width=True)

with tab4:
    if len(etf_prices.columns) >= 2:
        corr = correlation_matrix(etf_prices)
        fig_corr = correlation_heatmap(corr)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least 2 ETFs for correlation analysis.")

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
                hovertemplate="<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>"
            ))
            fig_rc.update_layout(title="Risk Contribution by ETF (%)",
                                  xaxis_title="ETF", yaxis_title="Risk Contribution (%)")
            st.plotly_chart(apply_dark_theme(fig_rc), use_container_width=True)

# ── Stress Tests ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Stress Test Scenarios")
st.caption("*Simulated stress test results for educational purposes. Actual market impacts may differ.*")

stress_scenarios = {
    "Equity Market Decline (-30%)": -0.30,
    "Interest Rate Shock (-15%)": -0.15,
    "High Volatility Period (-20%)": -0.20,
    "Defensive Market (+5%)": 0.05,
    "2008 Financial Crisis (-50%)": -0.50,
    "COVID-19 Crash (-34%)": -0.34,
    "Tech Bubble Burst (-45%)": -0.45,
}

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
        "Scenario": scenario_name,
        "Market Shock": f"{market_shock:.0%}",
        "Portfolio Beta": f"{b_val:.2f}",
        "Estimated Portfolio Impact": f"{port_impact:.2%}",
        "Impact on $10,000": f"${dollar_impact:,.0f}",
    })

stress_df = pd.DataFrame(stress_rows)
st.dataframe(stress_df.set_index("Scenario"), use_container_width=True)

disclaimer_box()
