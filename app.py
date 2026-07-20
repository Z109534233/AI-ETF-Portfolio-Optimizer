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
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, hero_section,
    section_header, chart_card, feature_card, render_footer
)
from src.theme import COLORS

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
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

    render_sidebar_footer()

# ── Hero ──────────────────────────────────────────────────────────────────────
hero_section()

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
section_header("Portfolio Dashboard", "Equal-weight preview across your selected ETFs")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(metric_card_html("Portfolio Value", f"${port_value:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Annualized Return", f"{ann_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Annualized Volatility", f"{ann_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Sharpe Ratio", f"{sr:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html("Max Drawdown", f"{mdd:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col6:
    st.markdown(metric_card_html("Diversification Score", f"{div_r:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)

# ── Main Charts ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    with chart_card("ETF Performance Comparison", "Normalized price, base = 100"):
        if not etf_prices.empty:
            fig_norm = normalized_price_chart(etf_prices)
            st.plotly_chart(fig_norm, use_container_width=True, key="home_normalized_price")

with col_right:
    with chart_card("Portfolio Allocation", "Equal weight across holdings"):
        weights_dict = {t: 1.0 / len(etf_prices.columns) for t in etf_prices.columns}
        fig_donut = allocation_donut_chart(weights_dict, "")
        st.plotly_chart(fig_donut, use_container_width=True, key="home_allocation_donut")

col_left2, col_right2 = st.columns([3, 2])

with col_left2:
    with chart_card("Portfolio Growth", "Equal weight, starting value $10,000"):
        if not etf_prices.empty:
            fig_cum = cumulative_return_chart(etf_prices)
            st.plotly_chart(fig_cum, use_container_width=True, key="home_cumulative_return")

with col_right2:
    with chart_card("Risk vs Return", "Annualized volatility vs. return"):
        if not etf_prices.empty:
            fig_rr = risk_return_scatter(etf_prices)
            st.plotly_chart(fig_rr, use_container_width=True, key="home_risk_return")

# ── Feature Overview ──────────────────────────────────────────────────────────
section_header("Platform Features", "Seven analytics modules covering the full portfolio workflow")

features = [
    {"icon": "bar-chart", "title": "ETF Analysis",
     "desc": "Historical price charts, return distributions, risk metrics, correlation heatmaps, and technical indicators for any ETF."},
    {"icon": "target", "title": "Portfolio Optimizer",
     "desc": "Mean-variance optimization with 5 methods: Equal Weight, Max Sharpe, Min Volatility, Target Return, and Risk Parity."},
    {"icon": "trending-up", "title": "Investment Simulator",
     "desc": "Monte Carlo simulation for long-term investment projections with inflation adjustment and scenario comparison."},
    {"icon": "shield", "title": "Risk Analytics",
     "desc": "Comprehensive risk metrics including VaR, CVaR, Beta, Alpha, Tracking Error, and stress test scenarios."},
    {"icon": "activity", "title": "Machine Learning",
     "desc": "Educational ML demonstration using Logistic Regression and Random Forest for ETF direction prediction."},
    {"icon": "layers", "title": "AI Advisor",
     "desc": "AI-powered portfolio explanation using OpenAI GPT with rule-based fallback when API key is not configured."},
    {"icon": "pie-chart", "title": "Portfolio History",
     "desc": "Save, view, compare, and manage portfolios stored in a local SQLite database with CSV export."},
]

cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        st.markdown(feature_card(feature["title"], feature["desc"], feature["icon"]), unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
section_header("Technology Stack")
tech_cols = st.columns(6)
tech_stack = ["Python 3.12", "Streamlit", "Pandas / NumPy", "Plotly", "SciPy / Scikit-learn", "SQLite / SQLAlchemy"]
for i, tech in enumerate(tech_stack):
    with tech_cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:12px 8px;">
            <div style="color:{COLORS['text']};font-size:12px;font-weight:600;">{tech}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
disclaimer_box()
render_footer()
