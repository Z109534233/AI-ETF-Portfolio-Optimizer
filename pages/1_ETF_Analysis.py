"""
Page 1: ETF Analysis
Comprehensive ETF price, return, risk, and correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data, compute_returns, normalize_prices
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    maximum_drawdown, calmar_ratio, value_at_risk, conditional_var,
    correlation_matrix, covariance_matrix, monthly_returns_table, compute_all_metrics
)
from src.technical_indicators import sma, ema, rsi, bollinger_bands
from src.charts import (
    price_chart, normalized_price_chart, cumulative_return_chart,
    drawdown_chart, correlation_heatmap, return_distribution_chart,
    risk_return_scatter, rolling_metrics_chart, monthly_heatmap
)
from src.utils import load_css, page_header, disclaimer_box, dataframe_to_csv, get_date_range_defaults

st.set_page_config(
    page_title="ETF Analysis | AI ETF Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

load_css()

page_header(
    "ETF Analysis",
    "Historical price analysis, risk metrics, and correlation insights",
    "📊"
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Analysis Settings")

    selected_etfs = st.multiselect(
        "Select ETFs",
        options=DEFAULT_ETFS,
        default=["VOO", "QQQ", "SPY"],
        help="Select one or more ETFs to analyse"
    )

    custom_ticker = st.text_input("Add Custom Ticker", placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    benchmark = st.selectbox("Benchmark ETF", options=DEFAULT_ETFS, index=2)  # SPY
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100

    st.markdown("---")
    st.markdown("### Analysis Sections")
    show_price = st.checkbox("Price Analysis", value=True)
    show_returns = st.checkbox("Return Analysis", value=True)
    show_risk = st.checkbox("Risk Analysis", value=True)
    show_correlation = st.checkbox("Correlation Analysis", value=True)

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning("Please select at least one ETF from the sidebar.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner("Downloading market data..."):
    all_tickers = list(set(selected_etfs + [benchmark]))
    raw_prices = download_etf_data(
        all_tickers,
        str(start_date),
        str(end_date)
    )

if raw_prices.empty:
    st.error("No price data available. Please check your ticker symbols and date range.")
    st.stop()

prices = clean_price_data(raw_prices)
etf_prices = prices[[t for t in selected_etfs if t in prices.columns]]
bench_prices = prices[benchmark] if benchmark in prices.columns else None

if etf_prices.empty:
    st.error("No valid price data found for the selected ETFs.")
    st.stop()

# ── Summary KPIs ──────────────────────────────────────────────────────────────
st.markdown("### Summary Metrics")
cols = st.columns(len(etf_prices.columns))
for i, ticker in enumerate(etf_prices.columns):
    with cols[i]:
        p = etf_prices[ticker].dropna()
        ann_ret = annualized_return(p)
        ann_vol = annualized_volatility(p)
        sr = sharpe_ratio(p, risk_free_rate)
        mdd = maximum_drawdown(p)
        st.metric(ticker, f"{ann_ret:.2%}", f"Vol: {ann_vol:.2%}")
        st.caption(f"Sharpe: {sr:.2f} | MDD: {mdd:.2%}")

# ── Price Analysis ────────────────────────────────────────────────────────────
if show_price:
    st.markdown("---")
    st.markdown("## Price Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Historical Prices", "Normalized Comparison", "Cumulative Return", "Drawdown"])

    with tab1:
        fig = price_chart(etf_prices, "Historical Adjusted Prices")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = normalized_price_chart(etf_prices)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = cumulative_return_chart(etf_prices)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = drawdown_chart(etf_prices)
        st.plotly_chart(fig, use_container_width=True)

    # Technical indicators for single ETF
    if len(etf_prices.columns) == 1:
        ticker = etf_prices.columns[0]
        p = etf_prices[ticker]
        st.markdown(f"#### Technical Indicators — {ticker}")
        import plotly.graph_objects as go
        from src.charts import apply_dark_theme
        bb = bollinger_bands(p)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=p.index, y=p, name=ticker, line=dict(color="#3B82F6", width=2)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Upper"], name="BB Upper",
                                     line=dict(color="#EF4444", dash="dash", width=1)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Middle"], name="BB Middle",
                                     line=dict(color="#9CA3AF", dash="dot", width=1)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Lower"], name="BB Lower",
                                     line=dict(color="#10B981", dash="dash", width=1),
                                     fill="tonexty", fillcolor="rgba(16,185,129,0.05)"))
        fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(apply_dark_theme(fig_bb), use_container_width=True)

# ── Return Analysis ───────────────────────────────────────────────────────────
if show_returns:
    st.markdown("---")
    st.markdown("## Return Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Monthly Heatmap", "Rolling Metrics", "Annual Performance"])

    with tab1:
        fig = return_distribution_chart(etf_prices)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        for ticker in etf_prices.columns:
            p = etf_prices[ticker].dropna()
            if len(p) > 30:
                monthly_ret = monthly_returns_table(p)
                if not monthly_ret.empty:
                    st.markdown(f"**{ticker} Monthly Returns**")
                    fig = monthly_heatmap(monthly_ret)
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ticker_select = st.selectbox("Select ETF for rolling metrics", etf_prices.columns.tolist(), key="rolling_ticker")
        window = st.slider("Rolling Window (days)", 21, 252, 63)
        p = etf_prices[ticker_select].dropna()
        if len(p) > window:
            fig = rolling_metrics_chart(p, window)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        returns_df = etf_prices.pct_change().dropna()
        annual_returns = returns_df.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
        if not annual_returns.empty:
            import plotly.graph_objects as go
            from src.charts import apply_dark_theme, CHART_COLORS
            fig = go.Figure()
            for i, col in enumerate(annual_returns.columns):
                fig.add_trace(go.Bar(
                    x=annual_returns.index.year,
                    y=annual_returns[col],
                    name=col,
                    marker_color=CHART_COLORS[i % len(CHART_COLORS)]
                ))
            fig.update_layout(title="Annual Performance (%)", xaxis_title="Year",
                               yaxis_title="Annual Return (%)", barmode="group")
            st.plotly_chart(apply_dark_theme(fig), use_container_width=True)

# ── Risk Analysis ─────────────────────────────────────────────────────────────
if show_risk:
    st.markdown("---")
    st.markdown("## Risk Analysis")

    metrics_data = []
    for ticker in etf_prices.columns:
        p = etf_prices[ticker].dropna()
        if len(p) < 10:
            continue
        bench_p = bench_prices.dropna() if bench_prices is not None else None
        all_metrics = compute_all_metrics(p, bench_p, risk_free_rate)
        row = {"Ticker": ticker}
        row.update(all_metrics)
        metrics_data.append(row)

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data).set_index("Ticker")
        st.dataframe(metrics_df.T, use_container_width=True)

    fig = risk_return_scatter(etf_prices)
    st.plotly_chart(fig, use_container_width=True)

# ── Correlation Analysis ──────────────────────────────────────────────────────
if show_correlation:
    st.markdown("---")
    st.markdown("## Correlation Analysis")

    if len(etf_prices.columns) >= 2:
        corr = correlation_matrix(etf_prices)
        fig = correlation_heatmap(corr)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Covariance Matrix (Annualized)**")
        cov = covariance_matrix(etf_prices)
        st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)
    else:
        st.info("Select at least 2 ETFs to view correlation analysis.")

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Download Data")
col1, col2, col3 = st.columns(3)

with col1:
    csv = dataframe_to_csv(etf_prices)
    st.download_button("Download Raw Price Data (CSV)", csv, "etf_prices.csv", "text/csv")

with col2:
    returns_df = etf_prices.pct_change().dropna()
    csv2 = dataframe_to_csv(returns_df)
    st.download_button("Download Daily Returns (CSV)", csv2, "etf_returns.csv", "text/csv")

with col3:
    if metrics_data:
        metrics_export = pd.DataFrame(metrics_data).set_index("Ticker")
        csv3 = dataframe_to_csv(metrics_export)
        st.download_button("Download Metrics (CSV)", csv3, "etf_metrics.csv", "text/csv")

disclaimer_box()
