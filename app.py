"""
AI ETF Portfolio Optimizer
Main landing page — professional FinTech dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio,
    maximum_drawdown, diversification_ratio, covariance_matrix
)
from src.portfolio_optimizer import equal_weight, backtest_portfolio
from src.database import init_database
from src.charts import (
    normalized_price_chart, allocation_donut_chart,
    risk_return_scatter, cumulative_return_chart, apply_dark_theme
)
from src.utils import load_css, disclaimer_box, metric_card_html, ensure_directories

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI ETF Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AI ETF Portfolio Optimizer — Educational FinTech Portfolio Project",
    }
)

# ── Initialisation ────────────────────────────────────────────────────────────
ensure_directories()
init_database()
load_css()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0B1220 0%, #111827 50%, #0B1220 100%);
    border-bottom: 2px solid #1F2937;
    padding: 32px 0 24px 0;
    margin-bottom: 8px;
">
    <h1 style="
        color: #F8FAFC;
        font-size: 36px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    ">
        <span style="color:#3B82F6;">AI</span> ETF Portfolio Optimizer
    </h1>
    <p style="color:#9CA3AF;font-size:16px;margin:0 0 4px 0;">
        AI-Powered ETF Portfolio Analytics and Optimization Platform
    </p>
    <p style="color:#6B7280;font-size:13px;margin:0;">
        Analyse ETF performance, measure portfolio risk, compare allocation strategies,
        simulate long-term investment outcomes, and generate explainable portfolio insights.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px;
    padding: 8px 16px;
    margin: 8px 0 16px 0;
    color: #FCA5A5;
    font-size: 12px;
">
    ⚠️ <strong>Educational Use Only</strong>: This platform is a portfolio project for academic purposes.
    All analysis and projections are for educational demonstration only and do not constitute financial advice.
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <div style="font-size:32px;">📊</div>
        <div style="color:#3B82F6;font-weight:700;font-size:16px;">AI ETF Optimizer</div>
        <div style="color:#6B7280;font-size:11px;">Portfolio Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Settings")

    demo_etfs = st.multiselect(
        "Dashboard ETFs",
        DEFAULT_ETFS,
        default=["VOO", "QQQ", "BND", "GLD"],
        help="Select ETFs for the dashboard preview"
    )

    import datetime
    end_date = datetime.date.today()
    start_date = datetime.date(end_date.year - 3, end_date.month, end_date.day)

    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - 📊 **ETF Analysis** — Price, returns & risk
    - ⚡ **Portfolio Optimizer** — Efficient frontier
    - 📈 **Investment Simulator** — Monte Carlo
    - 🛡️ **Risk Analytics** — Drawdown & VaR
    - 🤖 **Machine Learning** — Direction prediction
    - 🧠 **AI Advisor** — Portfolio explanation
    - 📚 **Portfolio History** — Saved portfolios
    """)

# ── Load Dashboard Data ───────────────────────────────────────────────────────
if not demo_etfs:
    demo_etfs = ["VOO", "QQQ", "BND", "GLD"]

with st.spinner("Loading market data for dashboard..."):
    raw_prices = download_etf_data(demo_etfs, str(start_date), str(end_date))

if raw_prices.empty:
    st.warning("Live market data unavailable. Dashboard is showing sample data.")
    from src.data_loader import _generate_sample_data
    raw_prices = _generate_sample_data(demo_etfs, str(start_date), str(end_date))

prices = clean_price_data(raw_prices)
etf_prices = prices[[t for t in demo_etfs if t in prices.columns]]

# Compute portfolio metrics (equal weight)
if not etf_prices.empty:
    n = len(etf_prices.columns)
    weights_arr = np.array([1.0 / n] * n)
    returns_df = etf_prices.pct_change().dropna()
    port_returns = (returns_df * weights_arr).sum(axis=1)
    port_prices = (1 + port_returns).cumprod() * 10000

    ann_ret = annualized_return(port_prices)
    ann_vol = annualized_volatility(port_prices)
    sr = sharpe_ratio(port_prices, 0.05)
    mdd = maximum_drawdown(port_prices)
    port_value = port_prices.iloc[-1]
    cov = covariance_matrix(etf_prices).values.copy()
    cov += np.eye(n) * 1e-8
    div_r = diversification_ratio(weights_arr, cov)
else:
    ann_ret, ann_vol, sr, mdd, port_value, div_r = 0.10, 0.15, 0.67, -0.12, 10800, 1.25

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("### Portfolio Dashboard")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(metric_card_html("Portfolio Value", f"${port_value:,.0f}", color="#3B82F6"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Annualized Return", f"{ann_ret:.2%}", color="#10B981"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Annualized Volatility", f"{ann_vol:.2%}", color="#EF4444"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Sharpe Ratio", f"{sr:.2f}", color="#3B82F6"), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html("Max Drawdown", f"{mdd:.2%}", color="#EF4444"), unsafe_allow_html=True)
with col6:
    st.markdown(metric_card_html("Diversification Score", f"{div_r:.2f}", color="#8B5CF6"), unsafe_allow_html=True)

# ── Main Charts ───────────────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### ETF Performance Comparison")
    if not etf_prices.empty:
        fig_norm = normalized_price_chart(etf_prices)
        st.plotly_chart(fig_norm, use_container_width=True)

with col_right:
    st.markdown("#### Portfolio Allocation")
    weights_dict = {t: 1.0 / len(etf_prices.columns) for t in etf_prices.columns}
    fig_donut = allocation_donut_chart(weights_dict, "Equal Weight Portfolio")
    st.plotly_chart(fig_donut, use_container_width=True)

col_left2, col_right2 = st.columns([3, 2])

with col_left2:
    st.markdown("#### Portfolio Growth (Equal Weight, $10,000)")
    if not etf_prices.empty:
        fig_cum = cumulative_return_chart(etf_prices)
        st.plotly_chart(fig_cum, use_container_width=True)

with col_right2:
    st.markdown("#### Risk vs Return")
    if not etf_prices.empty:
        fig_rr = risk_return_scatter(etf_prices)
        st.plotly_chart(fig_rr, use_container_width=True)

# ── Feature Overview ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Platform Features")

features = [
    {
        "icon": "📊",
        "title": "ETF Analysis",
        "desc": "Historical price charts, return distributions, risk metrics, correlation heatmaps, and technical indicators for any ETF.",
        "page": "ETF_Analysis",
    },
    {
        "icon": "⚡",
        "title": "Portfolio Optimizer",
        "desc": "Mean-variance optimization with 5 methods: Equal Weight, Max Sharpe, Min Volatility, Target Return, and Risk Parity.",
        "page": "Portfolio_Optimizer",
    },
    {
        "icon": "📈",
        "title": "Investment Simulator",
        "desc": "Monte Carlo simulation for long-term investment projections with inflation adjustment and scenario comparison.",
        "page": "Investment_Simulator",
    },
    {
        "icon": "🛡️",
        "title": "Risk Analytics",
        "desc": "Comprehensive risk metrics including VaR, CVaR, Beta, Alpha, Tracking Error, and stress test scenarios.",
        "page": "Risk_Analytics",
    },
    {
        "icon": "🤖",
        "title": "Machine Learning",
        "desc": "Educational ML demonstration using Logistic Regression and Random Forest for ETF direction prediction.",
        "page": "Machine_Learning",
    },
    {
        "icon": "🧠",
        "title": "AI Advisor",
        "desc": "AI-powered portfolio explanation using OpenAI GPT with rule-based fallback when API key is not configured.",
        "page": "AI_Advisor",
    },
    {
        "icon": "📚",
        "title": "Portfolio History",
        "desc": "Save, view, compare, and manage portfolios stored in a local SQLite database with CSV export.",
        "page": "Portfolio_History",
    },
]

cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="
            background: #111827;
            border: 1px solid #1F2937;
            border-top: 3px solid #3B82F6;
            border-radius: 8px;
            padding: 16px;
            margin: 6px 0;
            height: 140px;
        ">
            <div style="font-size:24px;margin-bottom:6px;">{feature['icon']}</div>
            <div style="color:#F8FAFC;font-weight:700;font-size:14px;margin-bottom:4px;">{feature['title']}</div>
            <div style="color:#9CA3AF;font-size:12px;line-height:1.4;">{feature['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Technology Stack")
tech_cols = st.columns(6)
tech_stack = [
    ("Python 3.11", "🐍"), ("Streamlit", "⚡"), ("Pandas / NumPy", "📊"),
    ("Plotly", "📈"), ("SciPy / Scikit-learn", "🔬"), ("SQLite / SQLAlchemy", "🗄️"),
]
for i, (tech, icon) in enumerate(tech_stack):
    with tech_cols[i]:
        st.markdown(f"""
        <div style="
            background:#111827;border:1px solid #1F2937;border-radius:6px;
            padding:10px;text-align:center;
        ">
            <div style="font-size:20px;">{icon}</div>
            <div style="color:#9CA3AF;font-size:11px;margin-top:4px;">{tech}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
disclaimer_box()
st.markdown("""
<div style="text-align:center;color:#4B5563;font-size:11px;padding:8px 0;">
    AI ETF Portfolio Optimizer — Portfolio Project for UK Master's Programme Applications
    (Business Analytics / Finance Analytics / FinTech / Data Analytics) |
    Built with Python & Streamlit
</div>
""", unsafe_allow_html=True)
