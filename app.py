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
    section_header, chart_card, feature_card, render_footer,
    process_flow, question_grid, badge
)
from src.theme import COLORS
from src.i18n import t

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
    st.markdown(f"### {t('home_quick_settings')}")

    demo_etfs = st.multiselect(
        t("home_dashboard_etfs_label"),
        DEFAULT_ETFS,
        default=DEFAULT_ETFS[:4],
        help=t("home_dashboard_etfs_help"),
    )

    import datetime
    end_date = datetime.date.today()
    start_date = datetime.date(end_date.year - 3, end_date.month, end_date.day)

    render_sidebar_footer()

# ── Hero ──────────────────────────────────────────────────────────────────────
hero_section()

# ── Supported Markets (Global ETF Support) ───────────────────────────────────
section_header(t("home_supported_markets_title"), t("home_supported_markets_subtitle"))
st.markdown(
    '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
    + badge(t("home_market_us"), "blue")
    + badge(t("home_market_taiwan"), "green")
    + badge(t("home_market_uk"), "amber")
    + '</div>',
    unsafe_allow_html=True,
)

# ── Why Choose This Platform ─────────────────────────────────────────────────
section_header(t("home_why_choose_title"), t("home_why_choose_subtitle"))
why_choose = [
    {"icon": "search", "title_key": "why_etf_analytics_title", "desc_key": "why_etf_analytics_desc"},
    {"icon": "target", "title_key": "why_portfolio_optimization_title", "desc_key": "why_portfolio_optimization_desc"},
    {"icon": "trending-up", "title_key": "why_investment_simulation_title", "desc_key": "why_investment_simulation_desc"},
    {"icon": "cpu", "title_key": "why_ai_insights_title", "desc_key": "why_ai_insights_desc"},
]
why_cols = st.columns(4)
for i, card in enumerate(why_choose):
    with why_cols[i]:
        st.markdown(feature_card(t(card["title_key"]), t(card["desc_key"]), card["icon"]), unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
section_header(t("home_how_it_works_title"), t("home_how_it_works_subtitle"))
process_flow([
    t("step_choose_etfs"),
    t("step_analyze_performance"),
    t("step_optimize_portfolio"),
    t("step_simulate_investment"),
    t("step_ai_insights"),
])

# ── Who Is This Platform For ─────────────────────────────────────────────────
section_header(t("home_target_users_title"), t("home_target_users_subtitle"))
personas = [
    {"icon": "search", "title_key": "persona_beginner_title", "desc_key": "persona_beginner_desc"},
    {"icon": "shield", "title_key": "persona_long_term_title", "desc_key": "persona_long_term_desc"},
    {"icon": "book", "title_key": "persona_student_title", "desc_key": "persona_student_desc"},
]
persona_cols = st.columns(3)
for i, card in enumerate(personas):
    with persona_cols[i]:
        st.markdown(feature_card(t(card["title_key"]), t(card["desc_key"]), card["icon"]), unsafe_allow_html=True)

# ── Common Investment Questions ──────────────────────────────────────────────
section_header(t("home_problem_title"))
question_grid(
    [t(f"problem_q{i}") for i in range(1, 7)],
    t("home_problem_conclusion"),
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Load Dashboard Data ───────────────────────────────────────────────────────
if not demo_etfs:
    demo_etfs = DEFAULT_ETFS[:4]

with st.spinner(t("home_loading_market_data")):
    raw_prices = download_etf_data(demo_etfs, str(start_date), str(end_date))

if raw_prices.empty:
    st.warning(t("home_live_data_unavailable"))
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
section_header(t("home_dashboard_title"), t("home_dashboard_subtitle"))
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(metric_card_html(t("metric_portfolio_value"), f"${port_value:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_annualized_return"), f"{ann_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_annualized_volatility"), f"{ann_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sr:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html(t("metric_maximum_drawdown"), f"{mdd:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col6:
    st.markdown(metric_card_html(t("metric_diversification_score"), f"{div_r:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)

# ── Main Charts ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    with chart_card(t("home_chart_etf_performance_title"), t("home_chart_etf_performance_sub")):
        if not etf_prices.empty:
            fig_norm = normalized_price_chart(etf_prices)
            st.plotly_chart(fig_norm, use_container_width=True, key="home_normalized_price")

with col_right:
    with chart_card(t("home_chart_allocation_title"), t("home_chart_allocation_sub")):
        weights_dict = {tk: 1.0 / len(etf_prices.columns) for tk in etf_prices.columns}
        fig_donut = allocation_donut_chart(weights_dict, "")
        st.plotly_chart(fig_donut, use_container_width=True, key="home_allocation_donut")

col_left2, col_right2 = st.columns([3, 2])

with col_left2:
    with chart_card(t("home_chart_growth_title"), t("home_chart_growth_sub")):
        if not etf_prices.empty:
            fig_cum = cumulative_return_chart(etf_prices)
            st.plotly_chart(fig_cum, use_container_width=True, key="home_cumulative_return")

with col_right2:
    with chart_card(t("home_chart_risk_return_title"), t("home_chart_risk_return_sub")):
        if not etf_prices.empty:
            fig_rr = risk_return_scatter(etf_prices)
            st.plotly_chart(fig_rr, use_container_width=True, key="home_risk_return")

# ── Feature Overview ──────────────────────────────────────────────────────────
section_header(t("home_features_title"), t("home_features_subtitle"))

features = [
    {"icon": "bar-chart", "title_key": "feature_etf_analysis_title", "desc_key": "feature_etf_analysis_desc"},
    {"icon": "target", "title_key": "feature_portfolio_optimizer_title", "desc_key": "feature_portfolio_optimizer_desc"},
    {"icon": "trending-up", "title_key": "feature_investment_simulator_title", "desc_key": "feature_investment_simulator_desc"},
    {"icon": "shield", "title_key": "feature_risk_analytics_title", "desc_key": "feature_risk_analytics_desc"},
    {"icon": "activity", "title_key": "feature_machine_learning_title", "desc_key": "feature_machine_learning_desc"},
    {"icon": "layers", "title_key": "feature_ai_advisor_title", "desc_key": "feature_ai_advisor_desc"},
    {"icon": "pie-chart", "title_key": "feature_portfolio_history_title", "desc_key": "feature_portfolio_history_desc"},
]

cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        st.markdown(feature_card(t(feature["title_key"]), t(feature["desc_key"]), feature["icon"]), unsafe_allow_html=True)

# ── Tech Stack ────────────────────────────────────────────────────────────────
section_header(t("home_tech_stack_title"))
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
