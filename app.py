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
from src.etf_database import get_all_tickers, get_countries
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
    process_flow, question_grid, badge, NAV_ITEMS
)
from src.theme import COLORS
from src.i18n import t, get_language

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

# ── Platform Statistics (4 cards: ETF / Markets / Modules / Languages;
# counts read from the actual registered data instead of being hardcoded).
# The "Markets" card description already lists US/Taiwan/UK, so a separate
# Supported Markets section would just repeat that information. ────────────
_platform_lang = get_language()
if _platform_lang == "zh-TW":
    _platform_stats_title = "平台統計"
    _platform_stats = [
        (str(len(get_all_tickers())), "支援 ETF 數", "涵蓋美國、台灣、英國市場"),
        (str(len(get_countries())), "涵蓋市場", "美國、台灣、英國"),
        (str(len(NAV_ITEMS)), "分析模組", "從 ETF 分析到市場情報"),
        ("2", "支援語言", "繁體中文、English"),
    ]
else:
    _platform_stats_title = "Platform Statistics"
    _platform_stats = [
        (str(len(get_all_tickers())), "Supported ETFs", "Across US, Taiwan, and UK markets"),
        (str(len(get_countries())), "Markets", "United States, Taiwan, United Kingdom"),
        (str(len(NAV_ITEMS)), "Analysis Modules", "From ETF analysis to market intelligence"),
        ("2", "Languages", "Traditional Chinese, English"),
    ]

section_header(_platform_stats_title)
stat_cols = st.columns(4)
for _col, (_value, _title, _desc) in zip(stat_cols, _platform_stats):
    with _col:
        st.markdown(
            '<div class="kpi-card">'
            f'<div class="kpi-value" style="margin-bottom:4px;">{_value}</div>'
            f'<div style="color:var(--text);font-weight:700;font-size:13px;margin-bottom:6px;">{_title}</div>'
            f'<div style="color:var(--text-secondary);font-size:11.5px;line-height:1.4;">{_desc}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Why Choose This Platform (merged with the former Feature Overview
# section -- one 8-card grid, one card per module, no duplicated content). ──
_why_lang = get_language()
if _why_lang == "zh-TW":
    _why_title, _why_subtitle = "為什麼選擇這個平台", "八大核心模組，涵蓋分析到決策的完整流程。"
    why_choose = [
        {"icon": "newspaper", "title": "AI 市場情報", "desc": "即時新聞、事件分類與 AI 市場摘要。"},
        {"icon": "bar-chart", "title": "ETF 分析", "desc": "跨市場 ETF 價格、報酬與風險指標分析。"},
        {"icon": "target", "title": "投資組合最佳化", "desc": "五種方法找出最佳風險調整後配置。"},
        {"icon": "trending-up", "title": "投資模擬", "desc": "蒙地卡羅模擬長期投資成長情境。"},
        {"icon": "shield", "title": "風險分析", "desc": "VaR、CVaR、貝塔值與壓力測試分析。"},
        {"icon": "cpu", "title": "機器學習預測", "desc": "數據驅動的 ETF 漲跌方向預測模型。"},
        {"icon": "layers", "title": "AI 投資分析", "desc": "AI 生成投資組合說明與建議。"},
        {"icon": "pie-chart", "title": "投資組合紀錄", "desc": "儲存、比較與管理你的投資組合紀錄。"},
    ]
else:
    _why_title, _why_subtitle = "Why Choose This Platform", "Eight core modules spanning the full journey from analysis to decision."
    why_choose = [
        {"icon": "newspaper", "title": "AI Market Intelligence", "desc": "Real-time news, event tagging, AI summaries."},
        {"icon": "bar-chart", "title": "ETF Analysis", "desc": "Cross-market ETF price, return, and risk analysis."},
        {"icon": "target", "title": "Portfolio Optimization", "desc": "Five methods to find the optimal risk-adjusted mix."},
        {"icon": "trending-up", "title": "Investment Simulator", "desc": "Monte Carlo projections for long-term growth."},
        {"icon": "shield", "title": "Risk Analytics", "desc": "VaR, CVaR, Beta, and stress-test scenarios."},
        {"icon": "cpu", "title": "Machine Learning Forecast", "desc": "Data-driven ETF direction prediction models."},
        {"icon": "layers", "title": "AI Advisor", "desc": "AI-generated portfolio explanations and insights."},
        {"icon": "pie-chart", "title": "Portfolio History", "desc": "Save, compare, and manage your portfolio records."},
    ]
section_header(_why_title, _why_subtitle)
for _row_start in (0, 4):
    _why_cols = st.columns(4)
    for _col, _card in zip(_why_cols, why_choose[_row_start:_row_start + 4]):
        with _col:
            st.markdown(feature_card(_card["title"], _card["desc"], _card["icon"]), unsafe_allow_html=True)

# ── How It Works (vertical flow-card timeline, Apple/Bloomberg style) ────────
section_header(t("home_how_it_works_title"), t("home_how_it_works_subtitle"))
_how_it_works_steps = (
    ["① 選擇 ETF", "② 分析績效", "③ 最佳化投資組合", "④ 了解市場", "⑤ 做出更好的決策"]
    if get_language() == "zh-TW" else
    ["① Select ETFs", "② Analyze Performance", "③ Optimize Portfolio", "④ Understand Market", "⑤ Make Better Decisions"]
)
_flow_rows = []
for _i, _step in enumerate(_how_it_works_steps):
    _flow_rows.append(
        '<div style="display:flex;align-items:center;gap:16px;width:100%;max-width:480px;'
        'background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
        'padding:16px 22px;box-shadow:var(--shadow-sm);">'
        f'<div style="color:var(--text);font-weight:700;font-size:15px;letter-spacing:-0.01em;">{_step}</div>'
        '</div>'
    )
    if _i < len(_how_it_works_steps) - 1:
        _flow_rows.append('<div style="color:var(--text-muted);font-size:20px;line-height:1;">&#8595;</div>')
st.markdown(
    '<div style="display:flex;flex-direction:column;align-items:center;gap:10px;margin:8px 0 20px 0;">'
    + "".join(_flow_rows) +
    '</div>',
    unsafe_allow_html=True,
)

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

# ── Technology Stack ──────────────────────────────────────────────────────────
section_header(t("home_tech_stack_title"))
tech_stack = ["Python", "Streamlit", "Plotly", "Pandas", "NumPy", "Machine Learning", "SQLite"]
tech_cols = st.columns(len(tech_stack))
for _tech_col, _tech in zip(tech_cols, tech_stack):
    with _tech_col:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:12px 8px;">
            <div style="color:{COLORS['text']};font-size:12px;font-weight:600;">{_tech}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Final CTA (Apple/Stripe-style closing banner) ────────────────────────────
_cta_lang = get_language()
_cta_heading = "準備好最佳化你的投資組合了嗎？" if _cta_lang == "zh-TW" else "Ready to optimize your portfolio?"
_cta_button = "開始分析" if _cta_lang == "zh-TW" else "Start Analysis"
st.markdown(
    '<div class="hero" style="text-align:center;padding:36px 24px;margin-top:24px;">'
    f'<h2 style="color:var(--text);font-size:26px;font-weight:800;letter-spacing:-0.02em;margin:0;">{_cta_heading}</h2>'
    '</div>',
    unsafe_allow_html=True,
)
_cta_cols = st.columns([1, 1, 1])
with _cta_cols[1]:
    st.markdown('<div class="hero-cta-row">', unsafe_allow_html=True)
    if st.button(_cta_button, type="primary", use_container_width=True, key="final_cta_start_analysis"):
        st.switch_page("pages/1_ETF_Analysis.py")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
disclaimer_box()
render_footer()
_footer_credit = (
    "由 Hidey 打造 · NTUB · Version 1.0 · 2026" if _cta_lang == "zh-TW"
    else "Built by Hidey · NTUB · Version 1.0 · 2026"
)
st.markdown(
    f'<div style="text-align:center;color:var(--text-muted);font-size:11px;margin-top:6px;">{_footer_credit}</div>',
    unsafe_allow_html=True,
)
