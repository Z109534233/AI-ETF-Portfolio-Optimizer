"""
AI ETF Portfolio Optimizer
Main landing page — professional FinTech dashboard.
"""
import streamlit as st
import numpy as np
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.etf_database import get_all_tickers, get_countries
from src.data_cleaner import clean_price_data
from src.financial_metrics import annualized_return, annualized_volatility, sharpe_ratio
from src.database import init_database
from src.utils import load_css, disclaimer_box, ensure_directories
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, ticker_strip, hero_section,
    section_header, feature_card, render_footer, persona_row, faq_accordion,
    stat_strip, tech_stack_strip, section_surface, NAV_ITEMS
)
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

    # Shadow value protects against the same rerun-before-instantiation
    # issue as pages/1_ETF_Analysis.py: render_sidebar_nav() (called just
    # above) renders the language selector, whose internal st.rerun() can
    # otherwise drop this widget's own keyed state before it's reached.
    if "_home_selected_etfs_shadow" not in st.session_state:
        st.session_state["_home_selected_etfs_shadow"] = DEFAULT_ETFS[:4]
    demo_etfs = st.multiselect(
        t("home_dashboard_etfs_label"),
        DEFAULT_ETFS,
        default=st.session_state["_home_selected_etfs_shadow"],
        help=t("home_dashboard_etfs_help"),
        key="home_selected_etfs",
    )
    st.session_state["_home_selected_etfs_shadow"] = demo_etfs

    import datetime
    end_date = datetime.date.today()
    start_date = datetime.date(end_date.year - 3, end_date.month, end_date.day)

    render_sidebar_footer()

# ── Demo Portfolio Data (single fetch shared by the ticker strip + hero
# composition/metrics below -- Issue #31 item 1/2: no second, redundant
# network request, and no second dashboard preview later on the page). ──────
if not demo_etfs:
    demo_etfs = DEFAULT_ETFS[:4]

with st.spinner(t("home_loading_market_data")):
    raw_prices = download_etf_data(demo_etfs, str(start_date), str(end_date))

_using_sample_data = raw_prices.empty
if _using_sample_data:
    st.warning(t("home_live_data_unavailable"))
    from src.data_loader import _generate_sample_data
    raw_prices = _generate_sample_data(demo_etfs, str(start_date), str(end_date))

prices = clean_price_data(raw_prices)
etf_prices = prices[[tk for tk in demo_etfs if tk in prices.columns]]

if not etf_prices.empty:
    n = len(etf_prices.columns)
    weights_arr = np.array([1.0 / n] * n)
    returns_df = etf_prices.pct_change().dropna()
    port_returns = (returns_df * weights_arr).sum(axis=1)
    port_prices = (1 + port_returns).cumprod() * 10000

    ann_ret = annualized_return(port_prices)
    ann_vol = annualized_volatility(port_prices)
    sr = sharpe_ratio(port_prices, 0.05)

    _hero_composition = [(tk, f"{100.0 / n:.0f}%") for tk in etf_prices.columns]
    _ticker_items = []
    if len(etf_prices) >= 2:
        _latest_change = etf_prices.iloc[-1] / etf_prices.iloc[-2] - 1
        _ticker_items = [(tk, float(_latest_change[tk]) * 100) for tk in etf_prices.columns]
else:
    # No usable data at all, even after the sample-data fallback above --
    # these are fixed placeholder figures, not a computation of anything.
    _using_sample_data = True
    ann_ret, ann_vol, sr = 0.10, 0.15, 0.67
    _hero_composition = [(tk, f"{100.0 / len(demo_etfs):.0f}%") for tk in demo_etfs] if demo_etfs else []
    _ticker_items = []

_hero_metrics = [f"{ann_ret:.1%}", f"{ann_vol:.1%}", f"{sr:.2f}"]

# ── Ticker Strip + Hero ──────────────────────────────────────────────────────
ticker_strip(_ticker_items, is_sample=_using_sample_data)
hero_section(composition=_hero_composition, metrics=_hero_metrics, is_sample=_using_sample_data)

# ── Platform Statistics: low-emphasis numeric strip, not a 4-card grid
# (Issue #31 item 8). Counts read from the actual registered data instead
# of being hardcoded. ─────────────────────────────────────────────────────
_platform_lang = get_language()
if _platform_lang == "zh-TW":
    _platform_stats_title = "平台統計"
    _platform_stats = [
        (str(len(get_all_tickers())), "支援 ETF 數"),
        (str(len(get_countries())), "涵蓋市場"),
        (str(len(NAV_ITEMS)), "分析模組"),
        ("2", "支援語言"),
    ]
else:
    _platform_stats_title = "Platform Statistics"
    _platform_stats = [
        (str(len(get_all_tickers())), "Supported ETFs"),
        (str(len(get_countries())), "Markets"),
        (str(len(NAV_ITEMS)), "Analysis Modules"),
        ("2", "Languages"),
    ]

section_header(_platform_stats_title)
stat_strip(_platform_stats)

# Supported-ETF-count provenance (Issue #20 section 1B): the count above is
# real (len(get_all_tickers())), but must not be read as a claim that every
# listed ticker has guaranteed, always-available price data.
if _platform_lang == "zh-TW":
    st.caption(
        "ℹ️ ETF 清單彙整自平台內建的美國（NASDAQ／NYSE 上市清單）、"
        "台灣（證交所 TWSE／櫃買中心 TPEX）與英國市場 ETF 名單；"
        "實際可取得的價格資料仍取決於市場資料來源的覆蓋範圍。"
    )
else:
    st.caption(
        "ℹ️ The ETF list is compiled from the platform's built-in US "
        "(NASDAQ/NYSE listing files), Taiwan (TWSE/TPEX), and UK ETF "
        "universes; actual price data availability still depends on "
        "market-data source coverage."
        )

# ── Why Choose This Platform (the ONE primary card grid on Home, Issue #31
# item 3) -- merged with the former Feature Overview section, one 8-card
# grid, one card per module, no duplicated content. Carries the in-page
# jump target for the hero's "Explore Features" secondary CTA. ──────────────
_why_lang = get_language()
if _why_lang == "zh-TW":
    _why_title, _why_subtitle = "為什麼選擇這個平台", "八大核心模組，涵蓋分析到決策的完整流程"
    why_choose = [
        {"icon": "newspaper", "title": "市場情報", "desc": "即時新聞、事件分類與市場摘要（部分摘要由 AI 輔助生成）"},
        {"icon": "bar-chart", "title": "ETF 分析", "desc": "跨市場 ETF 價格、報酬與風險指標分析"},
        {"icon": "target", "title": "投資組合最佳化", "desc": "五種方法找出最佳風險調整後配置"},
        {"icon": "trending-up", "title": "投資模擬", "desc": "蒙地卡羅模擬長期投資成長情境"},
        {"icon": "shield", "title": "風險分析", "desc": "VaR、CVaR、貝塔值與壓力測試分析"},
        {"icon": "cpu", "title": "機器學習預測", "desc": "數據驅動的 ETF 漲跌方向預測模型"},
        {"icon": "layers", "title": "投資組合分析助手", "desc": "以自然語言說明既有量化分析結果，AI 不產生任何數字"},
        {"icon": "pie-chart", "title": "投資組合紀錄", "desc": "儲存、比較與管理你的投資組合紀錄"},
    ]
else:
    _why_title, _why_subtitle = "Why Choose This Platform", "Eight core modules spanning the full journey from analysis to decision."
    why_choose = [
        {"icon": "newspaper", "title": "Market Intelligence", "desc": "Real-time news, event tagging, and market summaries (some summaries are AI-assisted)."},
        {"icon": "bar-chart", "title": "ETF Analysis", "desc": "Cross-market ETF price, return, and risk analysis."},
        {"icon": "target", "title": "Portfolio Optimization", "desc": "Five methods to find the optimal risk-adjusted mix."},
        {"icon": "trending-up", "title": "Investment Simulator", "desc": "Monte Carlo projections for long-term growth."},
        {"icon": "shield", "title": "Risk Analytics", "desc": "VaR, CVaR, Beta, and stress-test scenarios."},
        {"icon": "cpu", "title": "Machine Learning Forecast", "desc": "Data-driven ETF direction prediction models."},
        {"icon": "layers", "title": "AI Portfolio Analyst", "desc": "Explains existing quantitative results in plain language -- the AI never generates the numbers."},
        {"icon": "pie-chart", "title": "Portfolio History", "desc": "Save, compare, and manage your portfolio records."},
    ]
section_header(_why_title, _why_subtitle, anchor_id="why-choose-anchor")
for _row_start in (0, 4):
    _why_cols = st.columns(4)
    for _col, _card in zip(_why_cols, why_choose[_row_start:_row_start + 4]):
        with _col:
            st.markdown(feature_card(_card["title"], _card["desc"], _card["icon"]), unsafe_allow_html=True)

# ── How It Works (horizontal step row on desktop, full width; collapses to
# a vertical stack on mobile via the site's existing stHorizontalBlock
# wrap rule in assets/style.css). Wrapped in a slightly lifted section
# surface for background rhythm (Issue #31 item 4). ──────────────────────
with section_surface():
    section_header(t("home_how_it_works_title"), t("home_how_it_works_subtitle"))
    _how_it_works_steps = (
        ["① 選 ETF", "② 分析績效", "③ 最佳化", "④ 市場分析", "⑤ 做出決策"]
        if get_language() == "zh-TW" else
        ["① Select ETFs", "② Analyze", "③ Optimize", "④ Market Analysis", "⑤ Decide"]
    )
    _how_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    _step_idx = 0
    for _ci, _col in enumerate(_how_cols):
        with _col:
            if _ci % 2 == 0:
                st.markdown(
                    '<div style="display:flex;align-items:center;justify-content:center;height:90px;'
                    'background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'box-shadow:var(--shadow-sm);padding:8px 12px;text-align:center;">'
                    f'<div style="color:var(--text);font-weight:700;font-size:14px;letter-spacing:-0.01em;">{_how_it_works_steps[_step_idx]}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                _step_idx += 1
            else:
                st.markdown(
                    '<div style="display:flex;align-items:center;justify-content:center;height:90px;">'
                    '<span class="how-it-works-arrow" style="color:var(--text-muted);font-size:20px;line-height:1;">&#8594;</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

# ── Who Is This Platform For -- lightweight persona row, not full feature
# cards (Issue #31 item 3). ──────────────────────────────────────────────────
section_header(t("home_target_users_title"), t("home_target_users_subtitle"))
persona_row([
    {"icon": "search", "title": t("persona_beginner_title"), "desc": t("persona_beginner_desc")},
    {"icon": "shield", "title": t("persona_long_term_title"), "desc": t("persona_long_term_desc")},
    {"icon": "book", "title": t("persona_student_title"), "desc": t("persona_student_desc")},
])

# ── Common Investment Questions -- compact accordion/expander FAQ, not six
# identical blue boxes (Issue #31 item 3). Wrapped in a section surface for
# rhythm, alternating with the plain-background Who It's For above it. ──────
with section_surface():
    section_header(t("home_problem_title"))
    faq_accordion([(t(f"problem_q{i}"), t(f"problem_a{i}")) for i in range(1, 7)])
    st.caption(t("home_problem_conclusion"))

# ── Technology Stack (low-key pills near the footer, Issue #31 item 7) +
# Footer. "Start Analysis" appears exactly once on Home, in the hero --
# no repeated final CTA banner (Issue #31 item 6). ──────────────────────────
_cta_lang = get_language()
tech_stack_strip(
    ["Python", "Streamlit", "Plotly", "Pandas", "NumPy", "Machine Learning", "SQLite"],
    label=t("home_tech_stack_title"),
)

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
