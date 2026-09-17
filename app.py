"""
AI ETF Portfolio Optimizer
Main landing page — professional FinTech dashboard.
"""
import streamlit as st
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import download_etf_data_with_status, DEFAULT_ETFS
from src.etf_database import get_all_tickers, get_countries
from src.data_cleaner import clean_price_data
from src.portfolio_optimizer import run_optimization
from src.database import init_database
from src.utils import load_css, disclaimer_box, ensure_directories, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, ticker_strip, hero_section,
    section_header, capability_hierarchy_grid, render_footer, persona_row, faq_accordion,
    stat_strip, tech_stack_strip, section_surface, process_flow, NAV_ITEMS
)
from src.i18n import t, get_language
from src.demo_portfolio import DIVERSIFIED_DEMO_TICKERS, resolve_demo_tickers
from src.risk_free_rate import get_cached_risk_free_rate, format_rate_provenance
from src.etf_coverage import load_coverage_snapshot, coverage_display_stat, coverage_unavailable_caption

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
    #
    # Default demo selection (Issue #45 item 1): the shared cross-asset
    # demo portfolio (VOO/VXUS/BND/GLD/TLT) when every one of its tickers
    # is actually available, instead of whichever 4 tickers sort first in
    # DEFAULT_ETFS (which could be a VOO/VTI/QQQ/SPY-style set with
    # extreme mutual overlap) -- see src.demo_portfolio. This is only the
    # ONE-TIME initial default; the user remains free to pick any ETF.
    if "_home_selected_etfs_shadow" not in st.session_state:
        st.session_state["_home_selected_etfs_shadow"] = resolve_demo_tickers(DEFAULT_ETFS)
    demo_etfs = st.multiselect(
        t("home_dashboard_etfs_label"),
        DEFAULT_ETFS,
        default=st.session_state["_home_selected_etfs_shadow"],
        help=t("home_dashboard_etfs_help"),
        key="home_selected_etfs",
    )
    st.session_state["_home_selected_etfs_shadow"] = demo_etfs

    # SAME default date window as Portfolio Optimizer (Issue #45 item 2) --
    # no second, independently-chosen lookback window.
    start_date, end_date = get_date_range_defaults()

    render_sidebar_footer()

# ── Demo Portfolio Data (single fetch shared by the ticker strip + hero
# composition/metrics below -- Issue #31 item 1/2: no second, redundant
# network request, and no second dashboard preview later on the page). ──────
if not demo_etfs:
    demo_etfs = resolve_demo_tickers(DEFAULT_ETFS)

with st.spinner(t("home_loading_market_data")):
    # download_etf_data_with_status() (not download_etf_data()) is required
    # here: a full-fallback simulated DataFrame is never empty, so an
    # `.empty` check alone cannot detect it -- Issue #46 review flagged this
    # as a real sample-data-disclosure gap (simulated numbers could render
    # as if live).
    raw_prices, _using_sample_data = download_etf_data_with_status(demo_etfs, str(start_date), str(end_date))

if raw_prices.empty and not _using_sample_data:
    # Defensive-only: download_etf_data_with_status() only returns an empty,
    # non-sample DataFrame when `tickers` itself was empty, which demo_etfs
    # never is here (see the `if not demo_etfs` guard above) -- kept so this
    # can never crash on downstream indexing into an empty DataFrame.
    from src.data_loader import _generate_sample_data
    raw_prices = _generate_sample_data(demo_etfs, str(start_date), str(end_date))
    _using_sample_data = True

if _using_sample_data:
    st.warning(t("home_live_data_unavailable"))

prices = clean_price_data(raw_prices)
etf_prices = prices[[tk for tk in demo_etfs if tk in prices.columns]]

# SAME live/default risk-free-rate source as every other page (Issue #45
# item 3) -- never a second, independently hard-coded 5% assumption.
_home_rf_info = get_cached_risk_free_rate()
_home_rf_rate = _home_rf_info["rate"]

if not etf_prices.empty:
    # SAME run_optimization(..., method="Equal Weight") engine Portfolio
    # Optimizer uses (Issue #45 item 2) -- Home no longer maintains a
    # second, independently-computed CAGR-based formula for these three
    # headline numbers, so identical inputs can never produce different
    # numbers between Home and the Optimizer.
    _demo_result = run_optimization(etf_prices, method="Equal Weight", risk_free_rate=_home_rf_rate)
    ann_ret = _demo_result["expected_return"]
    ann_vol = _demo_result["expected_volatility"]
    sr = _demo_result["sharpe_ratio"]
    _demo_weights = _demo_result["weights"]

    _hero_composition = [(tk, f"{_demo_weights.get(tk, 0.0) * 100:.0f}%") for tk in etf_prices.columns]
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

# ── Equal-Weight Demo disclosure (Issue #45 items 1/2/6): date range,
# risk-free-rate source/as-of, cross-asset rationale, and the expected-
# return estimator's fragility caveat -- all in compact text directly below
# the hero preview, never asserted only in a hidden methodology page. ──────
_demo_lang = get_language()
_demo_rf_caption = format_rate_provenance(_home_rf_info, _demo_lang)
if _demo_lang == "zh-TW":
    st.caption(
        f"📅 等權重示範（Equal-Weight Demo）｜資料期間：{start_date} 至 {end_date} ｜"
        f"無風險利率：{_demo_rf_caption}"
    )
    if not _using_sample_data and set(etf_prices.columns) == set(DIVERSIFIED_DEMO_TICKERS):
        st.caption(
            "跨資產示範組合：美股（VOO）、國際股票（VXUS）、美國綜合債券（BND）、"
            "黃金（GLD）與長天期美國公債（TLT），刻意涵蓋五種不同資產類別以示範跨資產"
            "分散效果，並非投資建議。"
        )
    st.caption(
        "⚠️ 預期報酬為歷史每日報酬算術平均值年化（×252），估計誤差較大，是本平台"
        "最佳化引擎中最敏感的假設之一，僅供示範，並非未來報酬的預測。"
    )
else:
    st.caption(
        f"📅 Equal-Weight Demo | Date range: {start_date} to {end_date} | "
        f"Risk-free rate: {_demo_rf_caption}"
    )
    if not _using_sample_data and set(etf_prices.columns) == set(DIVERSIFIED_DEMO_TICKERS):
        st.caption(
            "Cross-asset demonstration portfolio: US equities (VOO), international "
            "equities (VXUS), US aggregate bonds (BND), gold (GLD), and long-term US "
            "Treasuries (TLT) -- five distinct asset classes chosen to demonstrate "
            "diversification mechanics across asset classes; this is not investment advice."
        )
    st.caption(
        "⚠️ Expected return is the historical arithmetic mean daily return annualized "
        "(x252) -- a high-estimation-error, sensitive assumption in this platform's "
        "optimization engine, shown here for demonstration only, not a forecast."
    )

# ── Platform Statistics: low-emphasis numeric strip, not a 4-card grid
# (Issue #31 item 8). Counts read from the actual registered data instead
# of being hardcoded. ─────────────────────────────────────────────────────
_platform_lang = get_language()
# "Registered ETF Universe" (Issue #45 item 8), NOT "Supported ETFs" --
# len(get_all_tickers()) is the platform's registered/searchable universe
# size, never a proven, verified count of tickers with actual live price
# coverage (that separate, low-emphasis stat is added below ONLY when a
# real audit snapshot exists -- see src.etf_coverage).
if _platform_lang == "zh-TW":
    _platform_stats_title = "平台統計"
    _platform_stats = [
        (str(len(get_all_tickers())), "ETF 清單規模"),
        (str(len(get_countries())), "涵蓋市場"),
        (str(len(NAV_ITEMS)), "分析模組"),
        ("2", "支援語言"),
    ]
else:
    _platform_stats_title = "Platform Statistics"
    _platform_stats = [
        (str(len(get_all_tickers())), "Registered ETF Universe"),
        (str(len(get_countries())), "Markets"),
        (str(len(NAV_ITEMS)), "Analysis Modules"),
        ("2", "Languages"),
    ]

section_header(_platform_stats_title)
stat_strip(_platform_stats)

# Registered-universe-size provenance (Issue #20 section 1B, reworded for
# Issue #45 item 8): the count above is real (len(get_all_tickers())), but
# is explicitly the SEARCHABLE/REGISTERED universe size, not a guarantee of
# actual live price-data coverage for every listed ticker.
if _platform_lang == "zh-TW":
    st.caption(
        "ℹ️ 此為平台可搜尋／已登錄的 ETF 清單規模，彙整自美國（NASDAQ／NYSE 上市清單）、"
        "台灣（證交所 TWSE／櫃買中心 TPEX）與英國市場 ETF 名單；"
        "並非保證每檔 ETF 都有可取得的即時價格資料，實際可取得的價格資料仍取決於市場資料來源的覆蓋範圍。"
    )
else:
    st.caption(
        "ℹ️ This is the platform's searchable/registered ETF universe size, "
        "compiled from the built-in US (NASDAQ/NYSE listing files), Taiwan "
        "(TWSE/TPEX), and UK ETF universes -- it is NOT a guarantee that "
        "every listed ticker has verified, available live price data; "
        "actual price data availability still depends on market-data source coverage."
        )

# ── Verified price-coverage stat (Issue #45 item 8) -- shown ONLY when a
# real audit snapshot exists (scripts/audit_etf_price_coverage.py); never a
# fabricated or estimated coverage count. ────────────────────────────────
_coverage_snapshot = load_coverage_snapshot()
_coverage_stat = coverage_display_stat(_coverage_snapshot, _platform_lang)
if _coverage_stat:
    _coverage_value, _coverage_label = _coverage_stat
    st.caption(f"🔎 {_coverage_label}: {_coverage_value}")
else:
    st.caption(f"🔎 {coverage_unavailable_caption(_platform_lang)}")

# ── Why Choose This Platform (the ONE primary capability section on Home,
# Issue #31 item 3 / Issue #33 hierarchy pass) -- one featured capability
# (Portfolio Optimization) plus a borderless grid of seven smaller
# supporting items, instead of eight identically-weighted outlined cards.
# Carries the in-page jump target for the hero's "Explore Features"
# secondary CTA. ──────────────────────────────────────────────────────────
_why_lang = get_language()
if _why_lang == "zh-TW":
    _why_title, _why_subtitle = "為什麼選擇這個平台", "八大核心模組，涵蓋分析到決策的完整流程"
    why_featured = {"icon": "target", "title": "投資組合最佳化", "desc": "五種方法找出最佳風險調整後配置，是本平台的核心分析引擎"}
    why_items = [
        {"icon": "newspaper", "title": "市場情報", "desc": "即時新聞、事件分類與市場摘要（部分摘要由 AI 輔助生成）"},
        {"icon": "bar-chart", "title": "ETF 分析", "desc": "跨市場 ETF 價格、報酬與風險指標分析"},
        {"icon": "trending-up", "title": "投資模擬", "desc": "蒙地卡羅模擬長期投資成長情境"},
        {"icon": "shield", "title": "風險分析", "desc": "VaR、CVaR、貝塔值與壓力測試分析"},
        {"icon": "cpu", "title": "機器學習預測", "desc": "數據驅動的 ETF 漲跌方向預測模型"},
        {"icon": "layers", "title": "投資組合分析助手（規則式，可選 AI 輔助）", "desc": "以自然語言說明既有量化分析結果；預設為規則式分析，AI 為選用的輔助說明，不產生任何數字"},
        {"icon": "pie-chart", "title": "投資組合紀錄", "desc": "儲存、比較與管理你的投資組合紀錄"},
    ]
else:
    _why_title, _why_subtitle = "Why Choose This Platform", "Eight core modules spanning the full journey from analysis to decision."
    why_featured = {"icon": "target", "title": "Portfolio Optimization", "desc": "Five methods to find the optimal risk-adjusted mix -- the platform's core analysis engine."}
    why_items = [
        {"icon": "newspaper", "title": "Market Intelligence", "desc": "Real-time news, event tagging, and market summaries (some summaries are AI-assisted)."},
        {"icon": "bar-chart", "title": "ETF Analysis", "desc": "Cross-market ETF price, return, and risk analysis."},
        {"icon": "trending-up", "title": "Investment Simulator", "desc": "Monte Carlo projections for long-term growth."},
        {"icon": "shield", "title": "Risk Analytics", "desc": "VaR, CVaR, Beta, and stress-test scenarios."},
        {"icon": "cpu", "title": "Machine Learning Forecast", "desc": "Data-driven ETF direction prediction models."},
        {"icon": "layers", "title": "Portfolio Analyst (rule-based, optional AI assistance)", "desc": "Explains existing quantitative results in plain language; rule-based by default, with optional AI-assisted explanation -- it never generates the numbers."},
        {"icon": "pie-chart", "title": "Portfolio History", "desc": "Save, compare, and manage your portfolio records."},
    ]
section_header(_why_title, _why_subtitle, anchor_id="why-choose-anchor")
capability_hierarchy_grid(why_featured, why_items)

# ── How It Works -- a compact process strip, not a boxed-card band (Issue
# #37): no section_surface() wrapper (that bordered/lifted-background band
# plus five 90px boxed steps read as one oversized slab), just the title/
# subtitle followed by process_flow()'s single-line numbered pill row so the
# whole section stays visually subordinate to "Why Choose" above it. ────────
section_header(t("home_how_it_works_title"), t("home_how_it_works_subtitle"))
_how_it_works_steps = (
    ["選 ETF", "分析績效", "最佳化", "市場分析", "做出決策"]
    if get_language() == "zh-TW" else
    ["Select ETFs", "Analyze", "Optimize", "Market Analysis", "Decide"]
)
process_flow(_how_it_works_steps)

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
