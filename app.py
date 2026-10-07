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
    section_header, capability_hierarchy_grid, render_footer,
    stat_strip, tech_stack_strip, process_flow, NAV_ITEMS
)
from src.i18n import t, get_language
from src.demo_portfolio import DIVERSIFIED_DEMO_TICKERS, resolve_demo_tickers
from src.risk_free_rate import get_cached_risk_free_rate, format_rate_provenance
from src.etf_coverage import load_coverage_snapshot, coverage_display_stat

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

# Render feedback before the first potentially slow market-data call. This
# cannot mask a platform-level cold start before Streamlit itself is serving,
# but it prevents the app from looking blank while market data is fetched.
if "_home_first_load_notice_seen" not in st.session_state:
    st.info(t("home_first_load_note"))
    st.session_state["_home_first_load_notice_seen"] = True

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

# ── Platform Scope ───────────────────────────────────────────────────────────
# Keep the top-level counts factual and low-emphasis. The ETF number is the
# registered/searchable universe, not a claim that every ticker has verified
# live market-data coverage.
_platform_lang = get_language()
_feature_module_count = len(NAV_ITEMS) - 1
if _platform_lang == "zh-TW":
    _platform_stats_title = "平台範圍"
    _platform_stats = [
        (str(len(get_all_tickers())), "已登錄 ETF"),
        (str(len(get_countries())), "涵蓋市場"),
        (str(_feature_module_count), "分析模組"),
        ("2", "介面語言"),
    ]
else:
    _platform_stats_title = "Platform Scope"
    _platform_stats = [
        (str(len(get_all_tickers())), "Registered ETFs"),
        (str(len(get_countries())), "Markets"),
        (str(_feature_module_count), "Analysis Modules"),
        ("2", "Interface Languages"),
    ]

section_header(_platform_stats_title)
stat_strip(_platform_stats)

if _platform_lang == "zh-TW":
    st.caption(
        "ETF 數量代表平台可搜尋／已登錄的投資標的；實際市場資料可用性仍取決於資料來源。"
    )
else:
    st.caption(
        "ETF count refers to the platform's searchable/registered universe; "
        "actual market-data availability depends on the upstream data source."
    )

# Show a verified coverage statistic only when an audit snapshot exists. A
# missing audit is documented in methodology rather than surfaced as a
# prominent unfinished-product warning on the admissions landing page.
_coverage_snapshot = load_coverage_snapshot()
_coverage_stat = coverage_display_stat(_coverage_snapshot, _platform_lang)
if _coverage_stat:
    _coverage_value, _coverage_label = _coverage_stat
    st.caption(f"🔎 {_coverage_label}: {_coverage_value}")

# ── Quantitative Analytics Modules ───────────────────────────────────────────
_modules_lang = get_language()
if _modules_lang == "zh-TW":
    _modules_title = "量化分析模組"
    _modules_subtitle = "以投資組合建構、風險衡量、模擬與模型驗證為核心的分析流程"
    modules_featured = {
        "icon": "target",
        "title": "投資組合最佳化",
        "desc": "以等權重、最大夏普、最低波動、目標報酬與風險平價等方法比較風險調整後配置。",
    }
    module_items = [
        {"icon": "shield", "title": "風險分析", "desc": "VaR、CVaR、Beta、最大回撤與壓力測試。"},
        {"icon": "trending-up", "title": "蒙地卡羅模擬", "desc": "量化長期財富路徑與情境不確定性。"},
        {"icon": "bar-chart", "title": "ETF 分析", "desc": "比較報酬、波動、相關性與跨市場 ETF 特徵。"},
        {"icon": "cpu", "title": "機器學習", "desc": "以時間序列切分評估 ETF 漲跌方向模型。"},
        {"icon": "newspaper", "title": "市場情報", "desc": "整合市場新聞、來源時間與規則式影響摘要。"},
        {"icon": "layers", "title": "投資組合解讀", "desc": "以規則式方法解讀既有量化結果，並可選擇 AI 輔助敘述。"},
        {"icon": "pie-chart", "title": "投資組合紀錄", "desc": "儲存與比較分析結果，支援後續檢視與匯出。"},
    ]
else:
    _modules_title = "Quantitative Analytics Modules"
    _modules_subtitle = "Portfolio construction, risk measurement, simulation, and model validation in one analytical workflow."
    modules_featured = {
        "icon": "target",
        "title": "Portfolio Optimization",
        "desc": "Compare Equal Weight, Maximum Sharpe, Minimum Volatility, Target Return, and Risk Parity allocations.",
    }
    module_items = [
        {"icon": "shield", "title": "Risk Analytics", "desc": "VaR, CVaR, Beta, maximum drawdown, and stress testing."},
        {"icon": "trending-up", "title": "Monte Carlo Simulation", "desc": "Quantify long-horizon wealth paths and scenario uncertainty."},
        {"icon": "bar-chart", "title": "ETF Analysis", "desc": "Compare return, volatility, correlation, and cross-market ETF characteristics."},
        {"icon": "cpu", "title": "Machine Learning", "desc": "Evaluate ETF direction models with time-aware validation."},
        {"icon": "newspaper", "title": "Market Intelligence", "desc": "Aggregate market news with source timestamps and rule-based impact summaries."},
        {"icon": "layers", "title": "Portfolio Interpretation", "desc": "Explain computed quantitative results with rule-based logic and optional AI-assisted narrative."},
        {"icon": "pie-chart", "title": "Portfolio History", "desc": "Save and compare analytical outputs for later review and export."},
    ]

section_header(_modules_title, _modules_subtitle, anchor_id="analytics-modules-anchor")
capability_hierarchy_grid(modules_featured, module_items)

# ── Analytical Workflow ──────────────────────────────────────────────────────
if get_language() == "zh-TW":
    _workflow_title = "分析流程"
    _workflow_subtitle = "從資料與假設，到估計、最佳化、風險衡量與驗證"
    _workflow_steps = ["資料與假設", "報酬／風險估計", "投資組合建構", "風險評估", "模型驗證"]
else:
    _workflow_title = "Analytical Workflow"
    _workflow_subtitle = "From data and assumptions to estimation, portfolio construction, risk evaluation, and validation."
    _workflow_steps = ["Data & Assumptions", "Estimate", "Construct", "Evaluate Risk", "Validate"]

section_header(_workflow_title, _workflow_subtitle)
process_flow(_workflow_steps)

# ── Validation & Methodology ─────────────────────────────────────────────────
_validation_lang = get_language()
if _validation_lang == "zh-TW":
    _validation_title = "驗證與方法"
    _validation_subtitle = "將分析結果與模型假設、抽樣不確定性及樣本外表現分開檢視"
    validation_featured = {
        "icon": "target",
        "title": "模型驗證",
        "desc": "使用 walk-forward／時間序列切分與樣本外指標評估預測模型，避免只呈現訓練期結果。",
    }
    validation_items = [
        {"icon": "activity", "title": "Bootstrap 不確定性", "desc": "以重抽樣呈現估計值的穩定性與不確定性。"},
        {"icon": "shield", "title": "壓力測試", "desc": "在不利市場情境下檢查投資組合的下檔風險。"},
        {"icon": "layers", "title": "資料來源揭露", "desc": "區分即時資料、示範資料與資料來源的 as-of 資訊。"},
        {"icon": "book", "title": "方法限制", "desc": "明確揭露預期報酬、共變異數與模型假設的限制。"},
    ]
else:
    _validation_title = "Validation & Methodology"
    _validation_subtitle = "Separate analytical outputs from model assumptions, sampling uncertainty, and out-of-sample performance."
    validation_featured = {
        "icon": "target",
        "title": "Model Validation",
        "desc": "Use walk-forward/time-series splits and out-of-sample metrics so predictive results are not judged only in-sample.",
    }
    validation_items = [
        {"icon": "activity", "title": "Bootstrap Uncertainty", "desc": "Use resampling to assess the stability and uncertainty of estimates."},
        {"icon": "shield", "title": "Stress Testing", "desc": "Evaluate downside exposure under adverse market scenarios."},
        {"icon": "layers", "title": "Data Provenance", "desc": "Distinguish live data, demonstration fallbacks, and source as-of information."},
        {"icon": "book", "title": "Method Limitations", "desc": "Document assumptions in expected returns, covariance estimation, and predictive models."},
    ]

section_header(_validation_title, _validation_subtitle)
capability_hierarchy_grid(validation_featured, validation_items)

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
    "由 TZU-HSIN TSENG 開發 · NTUB · 2026" if _cta_lang == "zh-TW"
    else "Developed by TZU-HSIN TSENG · NTUB · 2026"
)
st.markdown(
    f'<div style="text-align:center;color:var(--text-muted);font-size:11px;margin-top:6px;">{_footer_credit}</div>',
    unsafe_allow_html=True,
)
