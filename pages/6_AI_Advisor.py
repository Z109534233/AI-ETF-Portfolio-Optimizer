"""
Page 6: AI Advisor
Educational portfolio explanation using OpenAI or rule-based fallback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, maximum_drawdown,
    sortino_ratio, calmar_ratio, value_at_risk
)
from src.ai_advisor import generate_ai_analysis, DISCLAIMER, get_openai_client
from src.charts import allocation_donut_chart
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer

st.set_page_config(
    page_title="AI Advisor | AI ETF Portfolio Optimizer",
    page_icon="🧠",
    layout="wide"
)

load_css()

page_header(
    "AI Portfolio Advisor",
    "Educational portfolio analysis and explanation"
)

# Check OpenAI availability
client = get_openai_client()
if client is None:
    st.info(
        "**AI Analysis Mode**: OpenAI API key not configured. "
        "The advisor will use rule-based analysis. "
        "To enable AI-powered analysis, add `OPENAI_API_KEY` to your Streamlit secrets."
    )
else:
    st.success("AI-powered analysis is available.")

st.warning(f"**Important**: {DISCLAIMER}")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown("### Portfolio Configuration")

    selected_etfs = st.multiselect(
        "Select ETFs",
        options=DEFAULT_ETFS,
        default=["VOO", "QQQ", "BND", "GLD"],
    )

    custom_ticker = st.text_input("Add Custom Ticker", placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    st.markdown("#### Portfolio Weights")
    weights_input = {}
    if selected_etfs:
        equal_w = 1.0 / len(selected_etfs)
        for ticker in selected_etfs:
            w = st.slider(f"{ticker} (%)", 0.0, 100.0, equal_w * 100, 1.0, key=f"ai_w_{ticker}")
            weights_input[ticker] = w / 100.0
        total_w = sum(weights_input.values())
        if abs(total_w - 1.0) > 0.01 and total_w > 0:
            st.warning(f"Weights sum to {total_w:.1%}. Will be normalised.")
            weights_input = {k: v / total_w for k, v in weights_input.items()}

    investment_amount = st.number_input("Investment Amount ($)", 100.0, 10_000_000.0, 10000.0, 500.0)

    st.markdown("---")
    st.markdown("### Investor Profile")
    investment_objective = st.selectbox(
        "Investment Objective",
        ["Long-term Growth", "Income & Dividends", "Capital Preservation",
         "Balanced Growth & Income", "Aggressive Growth", "Retirement Planning"]
    )
    risk_level = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
    investment_horizon = st.slider("Investment Horizon (Years)", 1, 40, 10)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input("Data Start Date", value=default_start)
    end_date = st.date_input("Data End Date", value=default_end)

    analyse_btn = st.button("Generate Analysis", type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning("Please select at least one ETF.")
    st.stop()

# ── Data Loading & Metrics ────────────────────────────────────────────────────
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if analyse_btn or st.session_state.ai_result is None:
    with st.spinner("Downloading data and computing metrics..."):
        raw_prices = download_etf_data(selected_etfs, str(start_date), str(end_date))
        prices_df = clean_price_data(raw_prices) if not raw_prices.empty else pd.DataFrame()

        # Compute portfolio metrics
        metrics = {}
        if not prices_df.empty:
            etf_prices = prices_df[[t for t in selected_etfs if t in prices_df.columns]]
            if not etf_prices.empty:
                weights_arr = np.array([weights_input.get(t, 0) for t in etf_prices.columns])
                if weights_arr.sum() > 0:
                    weights_arr = weights_arr / weights_arr.sum()
                returns_df = etf_prices.pct_change().dropna()
                port_returns = (returns_df * weights_arr).sum(axis=1)
                port_prices = (1 + port_returns).cumprod() * 100

                metrics = {
                    "Annualized Return": f"{annualized_return(port_prices):.2%}",
                    "Annualized Volatility": f"{annualized_volatility(port_prices):.2%}",
                    "Sharpe Ratio": f"{sharpe_ratio(port_prices, 0.05):.2f}",
                    "Sortino Ratio": f"{sortino_ratio(port_prices, 0.05):.2f}",
                    "Maximum Drawdown": f"{maximum_drawdown(port_prices):.2%}",
                    "Calmar Ratio": f"{calmar_ratio(port_prices):.2f}",
                    "VaR (95%)": f"{value_at_risk(port_prices, 0.95):.2%}",
                    "Number of Holdings": str(len(selected_etfs)),
                    "Investment Horizon": f"{investment_horizon} years",
                    "Risk Tolerance": risk_level,
                    "Investment Objective": investment_objective,
                }

    with st.spinner("Generating portfolio analysis..."):
        analysis_text = generate_ai_analysis(
            portfolio_weights=weights_input,
            metrics=metrics,
            investment_objective=investment_objective,
            risk_level=risk_level,
            investment_horizon=investment_horizon
        )

    st.session_state.ai_result = {
        "analysis": analysis_text,
        "metrics": metrics,
        "weights": weights_input,
    }

result = st.session_state.ai_result
if result is None:
    st.info("Configure your portfolio in the sidebar and click **Generate Analysis**.")
    st.stop()

# ── Display Results ───────────────────────────────────────────────────────────
section_header("Analysis Results")
col_left, col_right = st.columns([2, 1])

with col_left:
    with chart_card("Portfolio Analysis", tag="AI-Generated" if client else "Rule-Based"):
        st.markdown(result["analysis"])

with col_right:
    with chart_card("Portfolio Overview"):
        fig_donut = allocation_donut_chart(result["weights"], "")
        st.plotly_chart(fig_donut, use_container_width=True, key="ai_advisor_allocation_donut")

        if result["metrics"]:
            st.markdown("**Key Metrics**")
            for k, v in result["metrics"].items():
                if k not in ["Number of Holdings", "Investment Horizon", "Risk Tolerance", "Investment Objective"]:
                    st.metric(k, v)

# ── Regenerate ────────────────────────────────────────────────────────────────
section_header("Actions")
col1, col2 = st.columns(2)
with col1:
    if st.button("Regenerate Analysis"):
        st.session_state.ai_result = None
        st.rerun()

with col2:
    if result["analysis"]:
        analysis_bytes = result["analysis"].encode("utf-8")
        st.download_button(
            "Download Analysis (TXT)",
            analysis_bytes,
            "portfolio_analysis.txt",
            "text/plain"
        )

disclaimer_box()
render_footer()
