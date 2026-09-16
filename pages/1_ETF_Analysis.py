"""
Page 1: ETF Analysis
Professional analytics workspace -- Overview / Performance / Risk /
Holdings & Exposure / Compare / Deep Analysis.

ETF ANALYSIS FULL-PAGE WORKSPACE REDESIGN: every calculation on this page
is unchanged from before this round -- this file only reorganizes HOW and
WHEN that existing code renders. The six top-level workspaces are plain
Python if/elif branches gated on one st.segmented_control's value (NOT
st.tabs, which would execute every tab's body -- including every expensive
chart -- on every single rerun regardless of which tab is visible). Only
the active workspace's branch ever runs, so switching workspaces never
re-downloads market data (already cached by download_etf_data()) and never
computes/renders a chart the user isn't currently looking at.

A single "Focus ETF" (one of the currently selected tickers) drives the
compact header, KPI row, Overview, Performance, Risk, Holdings, and Deep
Analysis workspaces -- matching how the redesign brief describes those
areas ("ONE compact ETF identity header", singular "Ticker"/"Fund Name").
Compare is the one workspace that operates over the FULL multi-ETF
selection, since multi-ETF comparison is its entire purpose.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data
from src.data_cleaner import clean_price_data, compute_returns, normalize_prices
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns, get_etf
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    maximum_drawdown, calmar_ratio, value_at_risk, conditional_var,
    correlation_matrix, covariance_matrix, monthly_returns_table, compute_all_metrics
)
from src.technical_indicators import sma, ema, rsi, bollinger_bands, momentum
from src.charts import (
    price_chart, normalized_price_chart, cumulative_return_chart,
    drawdown_chart, correlation_heatmap, return_distribution_chart,
    risk_return_scatter, rolling_metrics_chart, monthly_heatmap
)
from src.utils import load_css, page_header, disclaimer_box, dataframe_to_csv, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state, kpi_card, chart_caption, ai_interpret_button,
    region_selector, region_etf_options, region_etf_multiselect, region_benchmark_selector,
    hero_metric_panel, insight_panel,
)
from src.i18n import (
    t, t_country, get_language, t_portfolio_view, t_trend_signal,
    t_etf_risk_level, t_etf_return_label, t_etf_verdict_return,
    t_investment_horizon, t_suitable_investor, t_compare_metric,
)
from src.etf_signals import (
    compute_quant_signals, trend_signal_from_return, recent_trend_return,
    generate_etf_interpretation, has_sufficient_history,
    risk_level_from_vol, expected_return_label_from_ann_ret,
)
from src.methodology import validate_etf_analysis_window

st.set_page_config(
    page_title="ETF Analysis | AI ETF Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

load_css()

page_header(t("etf_analysis_title"), t("etf_analysis_subtitle"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
# Redesign section 11: always-visible = Market + ETF Search + ETF Selection;
# Advanced ETF Filters and Analysis Settings both collapse by default.
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('etf_sidebar_settings')}")

    # region_selector() / region_etf_multiselect() (src/ui.py) are the SAME
    # shared helpers used by Portfolio Optimizer, Risk Analytics and AI
    # Advisor: they read and write one canonical
    # st.session_state["selected_region"] / ["selected_etfs_<region>"], so
    # picking a market or ETF here is immediately reflected on those other
    # pages too. extra_filters=True/collapse_filters=True are opt-in flags
    # (default False everywhere else) so this is the ONLY page whose
    # Advanced ETF Filters panel collapses and gains the Underlying
    # Market/Trading Currency filters -- Portfolio Optimizer / Risk
    # Analytics / AI Advisor render byte-for-byte as before.
    selected_region, ALL_REGIONS_LABEL = region_selector()
    etf_options = region_etf_options(selected_region, ALL_REGIONS_LABEL)
    selected_etfs = region_etf_multiselect(
        selected_region, etf_options, t("field_select_etfs"),
        help_text=t("etf_select_etfs_help"), n_default=3,
        extra_filters=True, collapse_filters=True,
    )

    custom_ticker = st.text_input(
        t("field_add_custom_ticker"), placeholder="e.g. ARKK", key="selected_custom_ticker",
    ).upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # ── Analysis Settings (collapsed by default) ──────────────────────────
    with st.expander(t("etf_analysis_settings_label"), expanded=False):
        default_start, default_end = get_date_range_defaults()
        if "_selected_start_date_shadow" not in st.session_state:
            st.session_state["_selected_start_date_shadow"] = default_start
        if "_selected_end_date_shadow" not in st.session_state:
            st.session_state["_selected_end_date_shadow"] = default_end

        start_date = st.date_input(
            t("field_start_date"), value=st.session_state["_selected_start_date_shadow"], key="selected_start_date",
        )
        st.session_state["_selected_start_date_shadow"] = start_date

        end_date = st.date_input(
            t("field_end_date"), value=st.session_state["_selected_end_date_shadow"], key="selected_end_date",
        )
        st.session_state["_selected_end_date_shadow"] = end_date

        # Market-aware benchmark selector (Global ETF Universe + Benchmark
        # Architecture round): region-keyed, so a benchmark can never be a
        # ticker unavailable in the current market -- see src/ui.py.
        benchmark = region_benchmark_selector(
            selected_region, etf_options, t("field_benchmark_etf"), help_text=t("etf_benchmark_help"),
        )
        risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, 5.0, 0.25) / 100

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf_sidebar"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── Data Loading (unchanged -- cached by download_etf_data(); switching
# workspaces below never re-triggers this block, since it's a plain script
# top-level statement that runs once per rerun regardless of which
# workspace branch executes afterward) ─────────────────────────────────────
with st.spinner(t("msg_downloading_market_data")):
    all_tickers = list(set(selected_etfs + [benchmark]))
    yahoo_tickers = [to_yahoo_symbol(tk) for tk in all_tickers]
    raw_prices = download_etf_data(
        yahoo_tickers,
        str(start_date),
        str(end_date)
    )

if raw_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

prices = clean_price_data(raw_prices)
prices = rename_yahoo_columns(prices)  # e.g. "0050.TW" -> "0050", "VOO" -> "VOO" (no-op)
etf_prices = prices[[tk for tk in selected_etfs if tk in prices.columns]]
bench_prices = prices[benchmark] if benchmark in prices.columns else None

if etf_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

# download_etf_data() already emits a generic warning naming any
# Yahoo-suffixed symbol that failed to download, but that is not the same
# as telling the user which of THEIR selected ETFs were consequently
# dropped from this page's comparisons/charts -- unlike Portfolio
# Optimizer (which hard-stops on any partial failure), this page tolerates
# a partial result so the surviving ETFs stay usable, but that must never
# be silent about which of the user's own choices got excluded.
_missing_selected_etfs = [tk for tk in selected_etfs if tk not in etf_prices.columns]
if _missing_selected_etfs:
    st.warning(t("etf_partial_data_warning", tickers=", ".join(_missing_selected_etfs)))

_lang = get_language()

# ── Focus ETF (drives Header / KPI row / Overview / Performance / Risk /
# Holdings / Deep Analysis; Compare uses the full multi-ETF selection) ──────
# Issue #29 visual-acceptance round item 1: the ticker already appears once
# in the compact header immediately below, so the switcher itself must not
# repeat it in its closed/default state -- a plain st.selectbox always
# shows the current selection as its own visible label, which would repeat
# the ticker a second time right above the header. A compact top-right
# st.popover keeps the exact same underlying selectbox (same key, same
# session state, same AppTest `at.selectbox` surface) but only reveals the
# current ticker once the user actually opens it.
if "_etf_analysis_focus_shadow" not in st.session_state:
    st.session_state["_etf_analysis_focus_shadow"] = etf_prices.columns[0]
_focus_default = st.session_state["_etf_analysis_focus_shadow"]
if _focus_default not in etf_prices.columns:
    _focus_default = etf_prices.columns[0]
_focus_tickers_list = etf_prices.columns.tolist()

_, _focus_switch_col = st.columns([5, 1])
with _focus_switch_col:
    with st.popover(t("etf_switch_etf_button"), use_container_width=True):
        _focus_ticker = st.selectbox(
            t("etf_focus_selector_label"), _focus_tickers_list, index=_focus_tickers_list.index(_focus_default),
            key="etf_analysis_focus_ticker",
        )
st.session_state["_etf_analysis_focus_shadow"] = _focus_ticker
_focus_record = get_etf(_focus_ticker)
_focus_p = etf_prices[_focus_ticker].dropna()

# ── Compact ETF Header ───────────────────────────────────────────────────────
_header_name = _focus_ticker
if _focus_record:
    _header_name = (_focus_record.display_name_zh if (_lang == "zh-TW" and _focus_record.display_name_zh)
                     else _focus_record.name)
_header_fields = [(t("etf_header_listing_market"), t_country(_focus_record.country) if _focus_record else "—"),
                   (t("etf_header_asset_class"), _focus_record.category if _focus_record else "—")]
if _focus_record and _focus_record.issuer:
    _header_fields.append((t("etf_header_issuer"), _focus_record.issuer))
_header_fields.append((t("etf_header_benchmark"), benchmark))
_header_chips = "".join(
    f'<div style="margin-right:22px;"><div style="color:var(--text-muted);font-size:10px;'
    f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">{label}</div>'
    f'<div style="color:var(--text);font-weight:700;font-size:13.5px;">{value}</div></div>'
    for label, value in _header_fields
)
_header_date = _focus_p.index[-1].strftime("%Y-%m-%d") if not _focus_p.empty else None
_header_date_html = (
    f'<div style="color:var(--text-muted);font-size:11px;margin-top:6px;">{t("etf_data_as_of_price", date=_header_date)}</div>'
    if _header_date else ""
)
st.markdown(
    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
    'padding:14px 18px;margin-bottom:10px;box-shadow:var(--shadow-sm);">'
    '<div style="display:flex;align-items:baseline;flex-wrap:wrap;margin-bottom:8px;">'
    f'<div style="color:var(--text);font-weight:800;font-size:20px;margin-right:10px;">{_focus_ticker}</div>'
    f'<div style="color:var(--text-secondary);font-size:13.5px;">{_header_name}</div>'
    '</div>'
    f'<div style="display:flex;flex-wrap:wrap;">{_header_chips}</div>'
    f'{_header_date_html}'
    '</div>',
    unsafe_allow_html=True,
)

# ── Methodology & Assumptions (M5) ──────────────────────────────────────────
# Compact, always-visible (independent of which workspace is open) disclosure
# of the ACTUAL calculation methodology for this page -- see
# src/methodology.py's ETF_ANALYSIS_METHODOLOGY, the single source of truth
# this panel and tests/test_methodology_m5.py both read from. Every runtime
# value (dates, observation count, benchmark, risk-free rate) is read
# straight off the already-computed _focus_p/benchmark/risk_free_rate above,
# never recomputed or guessed, so this can never drift from what the KPIs
# elsewhere on the page actually show.
with st.expander(t("etf_methodology_title"), expanded=False):
    st.caption(t("etf_methodology_subtitle"))
    _etf_meth_start = _focus_p.index.min().strftime("%Y-%m-%d") if not _focus_p.empty else "—"
    _etf_meth_end = _focus_p.index.max().strftime("%Y-%m-%d") if not _focus_p.empty else "—"
    st.markdown(
        f"- **{t('etf_methodology_history_label')}** — "
        f"{t('etf_methodology_history_value', ticker=_focus_ticker, start=_etf_meth_start, end=_etf_meth_end, days=len(_focus_p))}\n"
        f"- **{t('etf_methodology_annualization_label')}** — {t('etf_methodology_annualization_value')}\n"
        f"- **{t('etf_methodology_benchmark_label')}** — {benchmark}\n"
        f"- **{t('etf_methodology_rfr_label')}** — {t('etf_methodology_rfr_value', rf=f'{risk_free_rate:.2%}')}\n"
        f"- **{t('etf_methodology_rfr_usage_label')}** — {t('etf_methodology_rfr_usage_value')}\n"
        f"- **{t('etf_methodology_data_source_label')}** — {t('etf_methodology_data_source_value')}\n"
        f"- **{t('etf_methodology_limitation_label')}** — {t('etf_methodology_limitation_value')}"
    )
    _etf_meth_validation = validate_etf_analysis_window(len(_focus_p))
    if _etf_meth_validation["is_valid"]:
        st.success(t("etf_methodology_validation_pass"))
    else:
        st.warning(t("etf_methodology_validation_fail", issues="; ".join(_etf_meth_validation["issues"])))


# ── Shared rule-based analysis helpers (no external LLM) -- EXACT same
# formulas as before this round, now wrapped as functions so each
# workspace can call them for exactly the ticker(s) it needs instead of
# every ticker on every rerun. ─────────────────────────────────────────────
# Issue #40: trend color is kept, but the 🟢🟡🔴 decorative emoji is not
# rendered anywhere -- restrained finance UI, not status-light styling.
_SUM_TREND_META = {
    "Bullish": "var(--success)", "Neutral": "var(--warning)", "Bearish": "var(--danger)",
}


def _ai_summary_insights(lang, s_ret_period, s_ret_ann, s_vol, s_sharpe, s_mdd, s_price, s_ma_short, s_ma_long, s_mom, s_score):
    cands = []
    if s_ret_period > 0.15:
        cands.append((s_ret_period, "區間累積報酬表現強勁" if lang == "zh-TW" else "Cumulative return over the period is strong"))
    elif s_ret_period < 0:
        cands.append((-s_ret_period, "區間累積報酬為負" if lang == "zh-TW" else "Cumulative return over the period is negative"))

    if s_ret_ann > 0.15:
        cands.append((s_ret_ann, "年化報酬率優於市場平均" if lang == "zh-TW" else "Annualized return is above the market average"))
    elif s_ret_ann < 0:
        cands.append((-s_ret_ann, "年化報酬率低於預期" if lang == "zh-TW" else "Annualized return is below expectations"))

    if s_vol > 0.25:
        if s_score >= 40:
            cands.append((s_vol, "波動增加但仍維持健康趨勢" if lang == "zh-TW" else "Volatility has increased but the trend remains healthy"))
        else:
            cands.append((s_vol, "波動度偏高，風險上升" if lang == "zh-TW" else "Volatility is elevated, raising risk"))
    elif s_vol < 0.12:
        cands.append((0.12 - s_vol, "波動度偏低，價格走勢穩定" if lang == "zh-TW" else "Volatility is low, price action is stable"))

    if s_sharpe > 1.2:
        cands.append((s_sharpe / 2, "Sharpe Ratio 高於平均，風險調整後報酬優異" if lang == "zh-TW" else "Sharpe Ratio is above average, an excellent risk-adjusted return"))
    elif s_sharpe < 0.3:
        cands.append((0.3 - s_sharpe, "Sharpe Ratio 偏低，風險調整後報酬不佳" if lang == "zh-TW" else "Sharpe Ratio is low, a weak risk-adjusted return"))

    if s_mdd < -0.25:
        cands.append((-s_mdd, "最大回撤較深，需留意下檔風險" if lang == "zh-TW" else "Maximum drawdown is deep, downside risk should be noted"))
    elif s_mdd > -0.10:
        cands.append((1 + s_mdd, "最大回撤控制良好" if lang == "zh-TW" else "Maximum drawdown is well contained"))

    if s_price > s_ma_short > s_ma_long:
        cands.append((s_price / s_ma_long - 1, "站上短期與長期均線，趨勢偏多" if lang == "zh-TW" else "Price is above both short- and long-term moving averages"))
    elif s_price < s_ma_short < s_ma_long:
        cands.append((s_ma_long / s_price - 1, "跌破短期與長期均線，趨勢偏空" if lang == "zh-TW" else "Price is below both short- and long-term moving averages"))

    if s_mom > 0.05:
        cands.append((s_mom, f"{t_compare_metric('Momentum')}正在增強" if lang == "zh-TW" else f"{t_compare_metric('Momentum')} is strengthening"))
    elif s_mom < -0.05:
        cands.append((-s_mom, f"{t_compare_metric('Momentum')}正在減弱" if lang == "zh-TW" else f"{t_compare_metric('Momentum')} is weakening"))

    cands.sort(key=lambda c: c[0], reverse=True)
    result = [c[1] for c in cands[:4]]
    if len(result) < 3:
        _fillers = (
            ["整體風險與報酬維持平衡", "短期訊號尚不明確，建議持續觀察", "各項指標未見極端訊號"]
            if lang == "zh-TW" else
            ["Overall risk and return remain balanced", "Short-term signals are not yet decisive, worth continued monitoring", "No extreme signal across the tracked indicators"]
        )
        for f in _fillers:
            if len(result) >= 3:
                break
            if f not in result:
                result.append(f)
    return result


def _ai_summary_entry(ticker, lang):
    """Everything the ETF Analytical Summary card / ETF Ranking / ETF
    Compare Score / Investment Verdict need for one ticker -- delegates the
    actual score/trend/portfolio-view computation to
    src.etf_signals.compute_quant_signals(), the ONE canonical formula used
    everywhere on this page (Issue #20 section 2A: before this, three
    independently-tuned formulas could show a different Trend/Score for the
    exact same ticker). Callable per-ticker so Overview (focus ticker only)
    and Compare (all selected tickers) can each compute only what they use."""
    p = etf_prices[ticker].dropna()
    s_ret_period = p.iloc[-1] / p.iloc[0] - 1 if len(p) > 1 else 0.0
    signals = compute_quant_signals(p, risk_free_rate)
    insights = _ai_summary_insights(
        lang, s_ret_period, signals["ret_ann"], signals["vol"], signals["sharpe"], signals["mdd"],
        signals["price"], signals["ma_short"], signals["ma_long"], signals["mom"], signals["score"],
    )
    return {**signals, "insights": insights}


def _render_ai_summary_card(ticker, entry):
    color = _SUM_TREND_META[entry["trend"]]
    insights_html = "".join(
        f'<div style="color:var(--text-secondary);font-size:12px;line-height:1.6;">• {ins}</div>'
        for ins in entry["insights"]
    )
    st.markdown(
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
        'padding:16px 18px;margin:6px 0;box-shadow:var(--shadow-sm);">'
        f'<div style="color:var(--text);font-weight:800;font-size:16px;margin-bottom:4px;">{ticker}</div>'
        f'<div style="color:{color};font-weight:700;font-size:13px;margin-bottom:10px;">{t_trend_signal(entry["trend"])}</div>'
        f'<div style="color:var(--text-secondary);font-size:11px;margin-bottom:2px;">{t("etf_quant_score_label")}</div>'
        f'<div style="color:var(--text);font-weight:800;font-size:20px;margin-bottom:8px;">{entry["score"]}</div>'
        '<div style="display:flex;justify-content:space-between;color:var(--text-secondary);font-size:11.5px;margin-bottom:10px;">'
        f'<span>{t("etf_signal_agreement_label")}: {entry["signal_agreement"]}%</span>'
        f'<span>{t("etf_portfolio_view_label")}: {t_portfolio_view(entry["portfolio_view"])}</span></div>'
        f'<div style="color:var(--text-muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{t("etf_quant_insights_label")}</div>'
        f'{insights_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def _ai_interpretation(key_findings, investment_insight: str, risk_reminder: str) -> None:
    """Render a 3-part rule-based 'Analysis Notes' block below a chart: Key
    Findings, Investment Insight, Risk Reminder. Reused across workspaces
    -- each call site computes its own content from that chart's actual
    data. `key_findings` is a string or a list of 1-2 strings.

    Deliberately NOT labeled "AI Interpretation" -- this content is 100%
    deterministic/threshold-based, not an LLM call. The genuine OpenAI-
    backed "AI Interpretation" feature lives in _render_ai_interpretation_
    section() below (Issue #20 section 10: never label deterministic
    content as AI-generated)."""
    if isinstance(key_findings, str):
        key_findings = [key_findings]
    _kf_html = "".join(
        f'<div style="color:var(--text);font-size:12.5px;line-height:1.6;">• {kf}</div>'
        for kf in key_findings
    )
    st.markdown(
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);'
        'padding:12px 16px;margin:6px 0 14px 0;">'
        f'<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">{t("etf_analysis_notes_title")}</div>'
        f'<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">{t("etf_key_findings_label")}</div>'
        f'{_kf_html}'
        f'<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:8px 0 3px 0;">{t("etf_investment_insight_label")}</div>'
        f'<div style="color:var(--success);font-size:12.5px;line-height:1.6;">{investment_insight}</div>'
        f'<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:8px 0 3px 0;">{t("etf_risk_reminder_label")}</div>'
        f'<div style="color:var(--warning);font-size:12.5px;line-height:1.6;">{risk_reminder}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_ai_interpretation_section(ticker: str, entry: dict) -> None:
    """Overview workspace's 'Key Observations' block (Issue #20 section 2):
    explains that Trend Signal / Quant Score / Portfolio View are distinct
    constructs that can disagree, then offers a genuine OpenAI-backed "AI
    Interpretation" of THIS ticker's already-computed numbers (`entry`,
    from _ai_summary_entry() / src.etf_signals.compute_quant_signals()).

    User-triggered by a button (never fired automatically on rerun). The
    OpenAI call is only ever asked to explain the numbers in `entry` -- it
    cannot recompute or invent a Trend/Score/Portfolio View -- and results
    are cached by src.openai_service fingerprint in session_state so an
    unrelated widget rerun never re-spends the call. Falls back to a
    one-line rule-based observation, clearly labeled Rule-Based, if OpenAI
    is not configured or the call fails.
    """
    st.caption(t("etf_signal_semantics_note"))
    st.markdown(f"**{t('etf_ai_interpretation_title')}**")
    st.caption(t("etf_ai_interpretation_desc"))

    _req_key = f"_etf_ai_interp_requested_{ticker}"
    if st.button(t("etf_ai_interpretation_btn"), key=f"etf_ai_interp_btn_{ticker}"):
        st.session_state[_req_key] = True

    if not st.session_state.get(_req_key):
        return

    _window_start = _focus_p.index[0].strftime("%Y-%m-%d") if not _focus_p.empty else None
    _window_end = _focus_p.index[-1].strftime("%Y-%m-%d") if not _focus_p.empty else None
    result = generate_etf_interpretation(
        ticker, _lang, _window_start, _window_end, entry, session_state=st.session_state,
    )
    badge = t("ai_tag_generated") if result["source"] == "ai" else t("ai_tag_rule_based")
    with chart_card(t("etf_ai_interpretation_title"), tag=badge):
        if result["source"] == "ai":
            st.markdown(result["text"])
        else:
            st.caption(t("etf_ai_interpretation_unavailable"))
            st.markdown(f"• {entry['insights'][0]}" if entry["insights"] else "—")


# ── Top-Level Workspace Navigation ───────────────────────────────────────────
# st.segmented_control (NOT st.tabs -- see module docstring). Canonical
# English values are stored in session_state; `_ws_labels` is precomputed
# ONCE per render outside the format_func lambda -- a format_func that
# calls t()/get_language() fresh, inside the lambda, has been confirmed
# (Global ETF Universe round) to corrupt AppTest's widget-state
# reconciliation between reruns, so every segmented_control on this page
# follows this same precomputed-dict pattern.
def _shadow_default(name: str, default):
    """Read-or-seed a plain (non-widget) session_state mirror for a
    segmented_control's `default=`. Needed for the same reason every other
    shadow-stated control in this app needs it (see pages/2_Portfolio_
    Optimizer.py's identical helper): the sidebar's language selector can
    trigger st.rerun() before this widget is (re-)instantiated on that
    pass, which -- without this -- can silently reset it to `default`
    instead of preserving the user's chosen view."""
    shadow_key = f"_{name}_shadow"
    if shadow_key not in st.session_state:
        st.session_state[shadow_key] = default
    return shadow_key, st.session_state[shadow_key]


_WORKSPACES = ["Overview", "Performance", "Risk", "Holdings", "Compare", "Deep Analysis"]
_ws_labels = {
    "Overview": t("etf_ws_overview"), "Performance": t("etf_ws_performance"), "Risk": t("etf_ws_risk"),
    "Holdings": t("etf_ws_holdings"), "Compare": t("etf_ws_compare"), "Deep Analysis": t("etf_ws_deep_analysis"),
}
_wsk, _wsv = _shadow_default("etf_analysis_workspace", "Overview")

# Issue #29 visual-acceptance round item 2: the reference hierarchy is
# identity header -> hero Annualized Return + rating badge -> 3 secondary
# metrics -> workspace nav -> insight/content, i.e. Overview's hero must
# render ABOVE the nav widget instead of inside the "if workspace ==
# Overview" branch below it. The nav widget's OWN value can't be read until
# after it's instantiated, so `_pending_workspace` predicts it from
# session_state first -- Streamlit already updates a widget's session_state
# entry for its key the instant the user interacts with it, before the
# script reruns, so this reads the up-to-date post-click value on every
# render except the very first (where it falls back to the shadow default).
_pending_workspace = st.session_state.get("etf_analysis_workspace", _wsv)
if _pending_workspace not in _WORKSPACES:
    _pending_workspace = "Overview"


def _render_overview_hero():
    """Compute the focus ticker's rule-based entry and render the Overview
    hero (Annualized Return + rating badge + 3 secondary metrics). Split out
    so it can run once, pre-nav, for the common case where the upcoming
    workspace is predicted correctly, with a same-function fallback call
    post-nav for the rare case (nav deselect -> Overview fallback) where the
    prediction above was wrong -- never a silent NameError either way."""
    if not has_sufficient_history(_focus_p):
        st.warning(t("etf_thin_history_warning", tickers=_focus_ticker))
    entry = _ai_summary_entry(_focus_ticker, _lang)
    color = _SUM_TREND_META[entry["trend"]]

    hero_metric_panel(
        t("etf_kpi_ann_return"), f"{entry['ret_ann']:.2%}", primary_color="var(--success)",
        badge={
            "label": f"{t_trend_signal(entry['trend'])} · {t('etf_score_badge_label')} {entry['score']}",
            "color": color,
            "meta": [(t("etf_signal_agreement_label"), f"{entry['signal_agreement']}%")],
        },
        secondary=[
            (t("etf_kpi_ann_vol"), f"{entry['vol']:.2%}", "var(--warning)"),
            (t("etf_kpi_sharpe"), f"{entry['sharpe']:.2f}", "var(--primary)"),
            (t("etf_kpi_mdd"), f"{entry['mdd']:.2%}", "var(--danger)"),
        ],
    )
    return entry, color


_focus_entry, _ov_color = (None, None)
if _pending_workspace == "Overview":
    _focus_entry, _ov_color = _render_overview_hero()

workspace = st.segmented_control(
    "workspace_nav", _WORKSPACES, default=_pending_workspace,
    format_func=lambda w: _ws_labels.get(w, w), key="etf_analysis_workspace", label_visibility="collapsed",
)
if not workspace:
    workspace = "Overview"
st.session_state[_wsk] = workspace

st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# OVERVIEW -- "What is this ETF?"
# ══════════════════════════════════════════════════════════════════════════
if workspace == "Overview":
    # The hero (Annualized Return + rating badge + secondary metrics) was
    # already rendered ABOVE the workspace nav -- see _render_overview_hero()
    # / _pending_workspace above -- except in the rare nav-deselect fallback
    # case, where it's computed and rendered here instead. No extra
    # "ETF Smart Summary" heading sits above the hero: the hero itself IS
    # the summary, and the insight block below is ONE flat panel (left
    # accent + heading + bullets + a compact Portfolio View footer line)
    # replacing the old nested Result -> Smart Summary -> Ticker -> Trend ->
    # Quant Score -> Insights card stack.
    if _focus_entry is None:
        _focus_entry, _ov_color = _render_overview_hero()

    # Issue #39 other-detail-3: this ordinary, deterministic "Quantitative
    # Insight" block is informational, not a warning -- it must NOT inherit
    # _ov_color (the Trend Signal's amber/red/green), or a plain Neutral
    # trend makes a routine explanation look like a warning. insight_panel()
    # already defaults accent_color to var(--primary) when none is passed.
    insight_panel(
        t("etf_quant_insights_label"), _focus_entry["insights"],
        footer=f"{t('etf_portfolio_view_label')}: {t_portfolio_view(_focus_entry['portfolio_view'])}",
    )

    with chart_card(t("etf_overview_chart_title")):
        fig = price_chart(etf_prices[[_focus_ticker]])
        st.plotly_chart(fig, use_container_width=True, key="overview_price_chart")
        chart_caption(t("etf_overview_chart_caption"))

    section_header(t("etf_overview_interpretation_title"))
    _render_ai_interpretation_section(_focus_ticker, _focus_entry)

# ══════════════════════════════════════════════════════════════════════════
# PERFORMANCE -- "How has it performed?"
# ══════════════════════════════════════════════════════════════════════════
elif workspace == "Performance":
    _PERF_VIEWS = ["Price", "Returns", "Annual"]
    _perf_labels = {"Price": t("etf_perf_nav_price"), "Returns": t("etf_perf_nav_returns"), "Annual": t("etf_perf_nav_annual")}
    _pvk, _pvv = _shadow_default("etf_perf_view", "Price")
    perf_view = st.segmented_control(
        "perf_nav", _PERF_VIEWS, default=_pvv if _pvv in _PERF_VIEWS else "Price",
        format_func=lambda w: _perf_labels.get(w, w), key="etf_perf_view", label_visibility="collapsed",
    ) or "Price"
    st.session_state[_pvk] = perf_view

    if perf_view == "Price":
        # Issue #39 priority 1: this used to be a THIRD nested
        # segmented_control (Historical / Normalized / Cumulative) stacked
        # under the Workspace and Performance-subview controls above,
        # reading as tab -> tab -> tab. A compact st.selectbox reads as a
        # single chart-type chooser scoped to the price chart below it,
        # not another tab row -- so it's placed in a narrow column and
        # keeps its label visible rather than mimicking the segmented
        # controls' collapsed-label underline style.
        _PRICE_VIEWS = ["Historical", "Normalized", "Cumulative"]
        _price_labels = {"Historical": t("etf_tab_historical"), "Normalized": t("etf_tab_normalized"), "Cumulative": t("etf_tab_cumulative")}
        _prk, _prv = _shadow_default("etf_price_view", "Historical")
        _price_view_default = _prv if _prv in _PRICE_VIEWS else "Historical"
        _price_sel_col, _ = st.columns([1, 2])
        with _price_sel_col:
            price_view = st.selectbox(
                t("etf_chart_type_label"), _PRICE_VIEWS, index=_PRICE_VIEWS.index(_price_view_default),
                format_func=lambda w: _price_labels.get(w, w), key="etf_price_view",
            )
        st.session_state[_prk] = price_view

        with chart_card(t("etf_price_charts_card"), tag=f"{len(etf_prices.columns)} ETFs"):
            if price_view == "Historical":
                fig = price_chart(etf_prices)
                st.plotly_chart(fig, use_container_width=True, key="etf_price_historical")
                chart_caption(t("etf_price_historical_caption"))
                _chg = {}
                for c in etf_prices.columns:
                    s = etf_prices[c].dropna()
                    if len(s) > 1:
                        _chg[c] = s.iloc[-1] / s.iloc[0] - 1
                if _chg:
                    _best, _worst = max(_chg, key=_chg.get), min(_chg, key=_chg.get)
                    _avg_chg = sum(_chg.values()) / len(_chg)
                    _bullish = _avg_chg > 0
                    if _lang == "zh-TW":
                        kf = [f"{_best} 期間漲幅最大（{_chg[_best]:+.1%}）"]
                        if _worst != _best:
                            kf.append(f"{_worst} 期間表現最弱（{_chg[_worst]:+.1%}）")
                        insight = "目前仍維持多頭趨勢，適合以長期持有的角度觀察後續表現" if _bullish else "目前呈現空頭趨勢，短線進場需更加謹慎，建議等待訊號轉強"
                        risk = "價格走勢可能反轉，過去的漲跌不代表未來一定延續"
                    else:
                        kf = [f"{_best} gained the most over the period ({_chg[_best]:+.1%})"]
                        if _worst != _best:
                            kf.append(f"{_worst} was the weakest performer ({_chg[_worst]:+.1%})")
                        insight = "The trend remains bullish, worth holding with a long-term view" if _bullish else "The trend is currently bearish, short-term entries warrant extra caution until it turns"
                        risk = "Price trends can reverse -- past performance is no guarantee of what comes next"
                    _ai_interpretation(kf, insight, risk)

            elif price_view == "Normalized":
                fig = normalized_price_chart(etf_prices)
                st.plotly_chart(fig, use_container_width=True, key="etf_price_normalized")
                chart_caption(t("etf_price_normalized_caption"))
                _norm_end = {}
                for c in etf_prices.columns:
                    s = etf_prices[c].dropna()
                    if len(s) > 1:
                        _norm_end[c] = s.iloc[-1] / s.iloc[0] * 100
                if _norm_end:
                    _best, _worst = max(_norm_end, key=_norm_end.get), min(_norm_end, key=_norm_end.get)
                    if _worst != _best:
                        _gap = _norm_end[_best] - _norm_end[_worst]
                        _wide_gap = _gap > 15
                        if _lang == "zh-TW":
                            kf = [
                                f"以相同基準比較，{_best} 相對表現最佳（指數 {_norm_end[_best]:.1f}）",
                                f"{_worst} 相對表現最弱（指數 {_norm_end[_worst]:.1f}），差距約 {_gap:.1f} 個指數點",
                            ]
                            insight = f"領先幅度明顯，新增資金可優先考慮 {_best}" if _wide_gap else f"{_best} 與 {_worst} 表現差距不大，可依個人配置偏好選擇"
                            risk = f"相對強弱可能反轉，{_worst} 落後不代表長期基本面較差"
                        else:
                            kf = [
                                f"On a normalized basis, {_best} is the relative leader (index {_norm_end[_best]:.1f})",
                                f"{_worst} is the relative laggard (index {_norm_end[_worst]:.1f}), a gap of about {_gap:.1f} index points",
                            ]
                            insight = f"The lead is significant -- new allocations could favor {_best}" if _wide_gap else f"{_best} and {_worst} are fairly close, so the choice can follow personal allocation preference"
                            risk = f"Relative strength can flip -- {_worst} trailing now doesn't imply weaker long-term fundamentals"
                    else:
                        kf = ["僅選取單一 ETF，無相對比較對象"] if _lang == "zh-TW" else ["Only one ETF is selected, so there's no relative comparison"]
                        insight = "建議加入至少一檔其他 ETF 以評估相對強弱" if _lang == "zh-TW" else "Consider adding at least one more ETF to gauge relative strength"
                        risk = "單一標的無法分散非系統性風險" if _lang == "zh-TW" else "A single holding carries undiversified idiosyncratic risk"
                    _ai_interpretation(kf, insight, risk)

            else:  # Cumulative
                fig = cumulative_return_chart(etf_prices)
                st.plotly_chart(fig, use_container_width=True, key="etf_price_cumulative")
                chart_caption(t("etf_price_cumulative_caption"))
                _cum = {}
                for c in etf_prices.columns:
                    s = etf_prices[c].dropna()
                    if len(s) > 1:
                        _cum[c] = (s.iloc[-1] / s.iloc[0] - 1) * 100
                if _cum:
                    _best, _worst = max(_cum, key=_cum.get), min(_cum, key=_cum.get)
                    _pos = sum(1 for v in _cum.values() if v > 0)
                    _majority_positive = _pos >= len(_cum) / 2
                    if _lang == "zh-TW":
                        kf = [f"{_best} 累積報酬最高（{_cum[_best]:+.1f}%）"]
                        if _worst != _best:
                            kf.append(f"{_worst} 累積報酬最低（{_cum[_worst]:+.1f}%）")
                        insight = f"{_pos}/{len(_cum)} 檔標的期間內維持正向複利成長，整體配置方向正確" if _majority_positive else f"僅 {_pos}/{len(_cum)} 檔標的為正報酬，建議重新檢視配置權重"
                        risk = "累積報酬可能因單一區間的大幅回檔而快速侵蝕，不代表未來持續複利"
                    else:
                        kf = [f"{_best} has the highest cumulative return ({_cum[_best]:+.1f}%)"]
                        if _worst != _best:
                            kf.append(f"{_worst} has the lowest cumulative return ({_cum[_worst]:+.1f}%)")
                        insight = f"{_pos}/{len(_cum)} holdings have compounded positively over the period, supporting the current allocation" if _majority_positive else f"Only {_pos}/{len(_cum)} holdings are positive -- worth revisiting the allocation weights"
                        risk = "Cumulative gains can erode quickly in a sharp drawdown -- past compounding doesn't guarantee it continues"
                    _ai_interpretation(kf, insight, risk)

    elif perf_view == "Returns":
        _RET_VIEWS = ["Distribution", "Heatmap", "Rolling"]
        _ret_labels = {"Distribution": t("etf_tab_distribution"), "Heatmap": t("etf_tab_monthly_heatmap"), "Rolling": t("etf_tab_rolling_metrics")}
        _rvk, _rvv = _shadow_default("etf_ret_view", "Distribution")
        ret_view = st.segmented_control(
            "ret_nav", _RET_VIEWS, default=_rvv if _rvv in _RET_VIEWS else "Distribution",
            format_func=lambda w: _ret_labels.get(w, w), key="etf_ret_view", label_visibility="collapsed",
        ) or "Distribution"
        st.session_state[_rvk] = ret_view

        with chart_card(t("etf_return_metrics_card")):
            if ret_view == "Distribution":
                fig = return_distribution_chart(etf_prices)
                st.plotly_chart(fig, use_container_width=True, key="etf_return_distribution")
                chart_caption(t("etf_return_distribution_caption"))
                _daily = etf_prices.pct_change().dropna()
                if not _daily.empty:
                    _pooled = _daily.values.flatten()
                    _pooled = _pooled[~np.isnan(_pooled)]
                    if len(_pooled) > 5:
                        _mean_r = _pooled.mean()
                        _std_r = _pooled.std()
                        _p5 = np.percentile(_pooled, 5)
                        _skew = pd.Series(_pooled).skew()
                        _skew_positive = _skew > 0.1
                        _skew_negative = _skew < -0.1
                        if _lang == "zh-TW":
                            kf = [
                                f"平均單日報酬率約為 {_mean_r:.3%}，波動度（標準差）約為 {_std_r:.2%}",
                                "分布呈現右偏（正報酬機會較大）" if _skew_positive else ("分布呈現左偏（極端虧損風險較高）" if _skew_negative else "分布大致對稱，無明顯偏態"),
                            ]
                            insight = "右偏結構對長期持有者相對有利，適合以時間換取複利機會" if _skew_positive else ("左偏結構代表需特別留意黑天鵝式重挫，配置時應保留緩衝" if _skew_negative else "報酬分布均衡，可依常見資產配置原則決定部位大小")
                            risk = f"左尾風險（5% 分位數）約為 {_p5:.2%}，代表極端下跌情境仍可能發生"
                        else:
                            kf = [
                                f"Average daily return is about {_mean_r:.3%}, with volatility (std dev) around {_std_r:.2%}",
                                "Distribution is right-skewed (more upside potential)" if _skew_positive else ("Distribution is left-skewed (higher extreme-loss risk)" if _skew_negative else "Distribution is roughly symmetric, no strong skew"),
                            ]
                            insight = "The right-skewed shape favors patient, long-term holders looking to compound over time" if _skew_positive else ("The left-skewed shape means occasional sharp losses are more likely -- keep a buffer when sizing positions" if _skew_negative else "Returns are fairly balanced, so standard position-sizing guidelines should apply")
                            risk = f"Left-tail risk (5th percentile) is about {_p5:.2%} -- extreme downside days can still happen"
                        _ai_interpretation(kf, insight, risk)

            elif ret_view == "Heatmap":
                for ticker in etf_prices.columns:
                    p = etf_prices[ticker].dropna()
                    if len(p) > 30:
                        monthly_ret = monthly_returns_table(p)
                        if not monthly_ret.empty:
                            st.markdown(f"**{t('etf_monthly_returns_for', ticker=ticker)}**")
                            fig = monthly_heatmap(monthly_ret)
                            st.plotly_chart(fig, use_container_width=True, key=f"etf_monthly_heatmap_{ticker}")
                            chart_caption(t("etf_monthly_heatmap_caption"))
                            _month_avg = monthly_ret.mean(axis=0, skipna=True)
                            _best_month, _worst_month = _month_avg.idxmax(), _month_avg.idxmin()
                            _pos_rate = (monthly_ret > 0).sum(axis=0) / monthly_ret.notna().sum(axis=0)
                            _seasonal = _pos_rate[(_pos_rate >= 0.75) | (_pos_rate <= 0.25)]
                            _has_seasonality = len(_seasonal) > 0 and monthly_ret.shape[0] >= 2
                            if _lang == "zh-TW":
                                kf = [
                                    f"{_best_month} 平均表現最佳（平均 {_month_avg[_best_month]:+.2%}）",
                                    f"{_worst_month} 平均表現最差（平均 {_month_avg[_worst_month]:+.2%}）",
                                ]
                                insight = f"{'、'.join(_seasonal.index)} 呈現較明顯的季節性傾向，可作為調整進出場時機的參考之一" if _has_seasonality else "未觀察到明顯的季節性規律，以月份作為進出場依據效益有限"
                                risk = "季節性型態基於歷史統計，樣本有限且不保證重演，不應作為唯一決策依據"
                            else:
                                kf = [
                                    f"{_best_month} performs best on average ({_month_avg[_best_month]:+.2%})",
                                    f"{_worst_month} performs worst on average ({_month_avg[_worst_month]:+.2%})",
                                ]
                                insight = f"{', '.join(_seasonal.index)} show a notable seasonal tendency, which could be one input for timing entries and exits" if _has_seasonality else "No clear seasonal pattern observed -- timing trades by calendar month is unlikely to add much value here"
                                risk = "Seasonal patterns are based on limited historical samples and aren't guaranteed to repeat -- don't rely on this alone"
                            _ai_interpretation(kf, insight, risk)

            else:  # Rolling
                ticker_select = st.selectbox(t("etf_select_rolling_etf"), etf_prices.columns.tolist(), key="rolling_ticker")
                window = st.slider(t("etf_rolling_window_days"), 21, 252, 63)
                p = etf_prices[ticker_select].dropna()
                if len(p) > window:
                    fig = rolling_metrics_chart(p, window)
                    st.plotly_chart(fig, use_container_width=True, key="etf_rolling_metrics")
                    chart_caption(t("etf_rolling_metrics_caption"))
                    _ret_series = p.pct_change().dropna()
                    _rolling_ret_valid = (_ret_series.rolling(window).mean() * 252).dropna()
                    if len(_rolling_ret_valid) > 5:
                        _recent = _rolling_ret_valid.iloc[-1]
                        _prior = _rolling_ret_valid.iloc[-min(window, len(_rolling_ret_valid))]
                        _full_mean = _ret_series.mean() * 252
                        _mom_last = momentum(p, 10).iloc[-1]
                        _mom_recent = _mom_last if pd.notna(_mom_last) else None
                        _strengthening = _recent > _prior
                        _breakout = _recent > _full_mean
                        _mom_up = _mom_recent is not None and _mom_recent > 0
                        if _lang == "zh-TW":
                            kf = [
                                "近期滾動報酬呈上升趨勢，動能轉強" if _strengthening else "近期滾動報酬呈下降趨勢，動能轉弱",
                                "目前滾動報酬已高於長期平均，屬於突破訊號" if _breakout else "目前滾動報酬仍低於長期平均，尚未突破",
                            ]
                            insight = "趨勢與動能同步轉強，可能是相對有利的進場時機" if (_strengthening and _breakout) else "訊號尚未一致轉強，建議等待更明確的突破確認再加碼"
                            risk = (f"10 日動能{'持續增加' if _mom_up else '轉為收斂或下滑'}（{_mom_recent:+.2%}），但動能類指標容易反轉，且滾動指標本身落後於即時價格" if _mom_recent is not None else "動能資料不足，且滾動指標本身落後於即時價格")
                        else:
                            kf = [
                                "Recent rolling return is trending up, momentum is strengthening" if _strengthening else "Recent rolling return is trending down, momentum is weakening",
                                "Rolling return is currently above the long-term average, a breakout signal" if _breakout else "Rolling return is still below the long-term average, no breakout yet",
                            ]
                            insight = "Trend and momentum are strengthening together, potentially a favorable entry window" if (_strengthening and _breakout) else "Signals aren't fully aligned yet -- worth waiting for a clearer breakout before adding"
                            risk = (f"10-day Momentum is {'increasing' if _mom_up else 'flattening or declining'} ({_mom_recent:+.2%}), but momentum indicators can reverse quickly and rolling metrics lag real-time price" if _mom_recent is not None else "Not enough data for Momentum, and rolling metrics lag real-time price")
                        _ai_interpretation(kf, insight, risk)

    else:  # Annual
        with chart_card(t("etf_return_metrics_card")):
            returns_df = etf_prices.pct_change().dropna()
            annual_returns = returns_df.resample("A").apply(lambda x: (1 + x).prod() - 1) * 100
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
                fig.update_layout(title=t("chart_annual_performance_pct"), xaxis_title=t("chart_year"),
                                   yaxis_title=t("chart_annual_return_pct"), barmode="group", height=420)
                st.plotly_chart(apply_dark_theme(fig), use_container_width=True, key="etf_annual_performance")
                chart_caption(t("etf_annual_performance_caption"))
                _yearly_avg = annual_returns.mean(axis=1)
                _best_year, _worst_year = _yearly_avg.idxmax().year, _yearly_avg.idxmin().year
                if len(_yearly_avg) >= 2:
                    _half = len(_yearly_avg) // 2
                    _first_half, _second_half = _yearly_avg.iloc[:_half].mean(), _yearly_avg.iloc[_half:].mean()
                    _trend_up = _second_half > _first_half
                else:
                    _trend_up = None
                _year_spread = _yearly_avg.max() - _yearly_avg.min()
                if _lang == "zh-TW":
                    kf = [
                        f"{_best_year} 年平均表現最佳（{_yearly_avg.max():+.1f}%）",
                        f"{_worst_year} 年平均表現最差（{_yearly_avg.min():+.1f}%）",
                    ]
                    if _trend_up is None:
                        insight = "資料年數過短，尚無法判斷長期趨勢，建議搭配更長期的數據評估"
                    else:
                        insight = "長期趨勢偏向轉強，支持持續採用目前的長期配置方向" if _trend_up else "長期趨勢偏向轉弱，建議重新檢視長期投資邏輯是否仍然成立"
                    risk = f"年度報酬落差達 {_year_spread:.1f} 個百分點，顯示年與年之間的波動不小，需有承受單一年度虧損的心理準備"
                else:
                    kf = [
                        f"{_best_year} was the best year on average ({_yearly_avg.max():+.1f}%)",
                        f"{_worst_year} was the worst year on average ({_yearly_avg.min():+.1f}%)",
                    ]
                    if _trend_up is None:
                        insight = "Not enough years of data to judge the long-term trend -- worth revisiting with a longer history"
                    else:
                        insight = "Long-term trend is strengthening, supporting the current long-term allocation approach" if _trend_up else "Long-term trend is weakening -- worth re-examining whether the long-term thesis still holds"
                    risk = f"The spread between the best and worst year is {_year_spread:.1f} percentage points -- year-to-year swings can be sizable, so be prepared for down years"
                _ai_interpretation(kf, insight, risk)

# ══════════════════════════════════════════════════════════════════════════
# RISK -- "How risky has it been?"
# ══════════════════════════════════════════════════════════════════════════
elif workspace == "Risk":
    _r_vol = annualized_volatility(_focus_p)
    _r_mdd = maximum_drawdown(_focus_p)
    _r_sharpe = sharpe_ratio(_focus_p, risk_free_rate)
    _r_var = value_at_risk(_focus_p)
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    with rcol1:
        st.markdown(kpi_card(t("etf_kpi_ann_vol"), f"{_r_vol:.2%}", color="var(--warning)", icon="activity"), unsafe_allow_html=True)
    with rcol2:
        st.markdown(kpi_card(t("etf_kpi_mdd"), f"{_r_mdd:.2%}", color="var(--danger)", icon="trending-down"), unsafe_allow_html=True)
    with rcol3:
        st.markdown(kpi_card(t("etf_kpi_sharpe"), f"{_r_sharpe:.2f}", color="var(--primary)", icon="target"), unsafe_allow_html=True)
    with rcol4:
        st.markdown(kpi_card("VaR (95%)", f"{_r_var:.2%}", color="var(--purple)", icon="shield"), unsafe_allow_html=True)

    _RISK_VIEWS = ["Drawdown", "Scatter"]
    _risk_labels = {"Drawdown": t("etf_risk_nav_drawdown"), "Scatter": t("etf_risk_nav_scatter")}
    _rik, _riv = _shadow_default("etf_risk_view", "Drawdown")
    risk_view = st.segmented_control(
        "risk_nav", _RISK_VIEWS, default=_riv if _riv in _RISK_VIEWS else "Drawdown",
        format_func=lambda w: _risk_labels.get(w, w), key="etf_risk_view", label_visibility="collapsed",
    ) or "Drawdown"
    st.session_state[_rik] = risk_view

    if risk_view == "Drawdown":
        with chart_card(t("etf_price_charts_card")):
            fig = drawdown_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_drawdown")
            chart_caption(t("etf_price_drawdown_caption"))
            _dd, _cur_dd = {}, {}
            for c in etf_prices.columns:
                s = etf_prices[c].dropna()
                if len(s) > 1:
                    roll_max = s.cummax()
                    dd = (s - roll_max) / roll_max
                    _dd[c] = dd.min() * 100
                    _cur_dd[c] = dd.iloc[-1] * 100
            if _dd:
                _deepest = min(_dd, key=_dd.get)
                _still_down = [c for c, v in _cur_dd.items() if v < -1]
                _dd_s = etf_prices[_deepest].dropna()
                _dd_roll_max = _dd_s.cummax()
                _dd_series = (_dd_s - _dd_roll_max) / _dd_roll_max
                _trough_idx = _dd_series.idxmin()
                _peak_before = _dd_roll_max.loc[_trough_idx]
                _after_trough = _dd_s.loc[_trough_idx:]
                _recovered_pts = _after_trough[_after_trough >= _peak_before]
                if len(_recovered_pts) > 1:
                    _recovery_days = (_recovered_pts.index[1] - _trough_idx).days
                    if _recovery_days <= 60:
                        _recovery_zh, _recovery_en = "快", "fast"
                    elif _recovery_days <= 180:
                        _recovery_zh, _recovery_en = "中等", "moderate"
                    else:
                        _recovery_zh, _recovery_en = "慢", "slow"
                else:
                    _recovery_zh, _recovery_en = "尚未恢復", "not yet recovered"

                _dd_pct = abs(_dd[_deepest])
                if _dd_pct < 10:
                    _risk_zh, _risk_en = "偏低", "relatively low"
                elif _dd_pct < 25:
                    _risk_zh, _risk_en = "中等", "medium"
                else:
                    _risk_zh, _risk_en = "偏高", "elevated"

                if _lang == "zh-TW":
                    kf = [f"目前最大回撤為 {_dd_pct:.1f}%（{_deepest}）"]
                    kf.append(f"目前仍處於回撤中：{'、'.join(_still_down)}" if _still_down else "所有標的目前皆已從最大回撤中恢復")
                    insight = f"歷史恢復速度{_recovery_zh}，顯示波動後仍具備修復能力" if _recovery_zh != "尚未恢復" else "尚未從最深回撤恢復，建議持續觀察後續走勢"
                    risk = f"風險屬於{_risk_zh}，實際投資仍應搭配自身風險承受度評估"
                else:
                    kf = [f"Current maximum drawdown is {_dd_pct:.1f}% ({_deepest})"]
                    kf.append(f"Currently still in drawdown: {', '.join(_still_down)}" if _still_down else "All selected ETFs have recovered from their max drawdown")
                    insight = f"Historical recovery speed is {_recovery_en}, showing it can bounce back after a drop" if _recovery_en != "not yet recovered" else "Hasn't recovered from its deepest drawdown yet -- worth continued monitoring"
                    risk = f"Risk is {_risk_en} -- weigh this against your own risk tolerance before investing"
                _ai_interpretation(kf, insight, risk)
    else:
        with chart_card(t("etf_risk_vs_return_card")):
            fig = risk_return_scatter(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_risk_return_scatter")
            chart_caption(t("etf_risk_return_scatter_caption"))
            _rr = {}
            for c in etf_prices.columns:
                s = etf_prices[c].dropna()
                if len(s) > 5:
                    _rr[c] = (annualized_return(s), annualized_volatility(s))
            if _rr:
                _best_ratio = max(_rr, key=lambda k: _rr[k][0] / _rr[k][1] if _rr[k][1] else 0)
                _highest_risk = max(_rr, key=lambda k: _rr[k][1])
                if len(_rr) > 1:
                    if _lang == "zh-TW":
                        kf = [f"{_best_ratio} 的風險報酬比相對最佳", f"{_highest_risk} 波動度最高，風險相對集中"]
                        insight = f"就目前選取的標的而言，{_best_ratio} 的風險調整後表現較值得優先考慮" if _best_ratio != _highest_risk else f"{_best_ratio} 兼具最高波動與最佳比率，屬於高風險高報酬型標的，配置比重應審慎拿捏"
                        risk = f"{_highest_risk} 的波動度最高，短線震盪可能較大，配置比重不宜過度集中"
                    else:
                        kf = [f"{_best_ratio} offers the best return-to-risk ratio", f"{_highest_risk} carries the highest volatility, concentrating risk"]
                        insight = f"Among the current selection, {_best_ratio} looks worth prioritizing on a risk-adjusted basis" if _best_ratio != _highest_risk else f"{_best_ratio} combines the highest volatility with the best ratio -- a high-risk, high-return profile that needs careful position sizing"
                        risk = f"{_highest_risk} carries the highest volatility -- short-term swings could be larger, so avoid over-concentrating in it"
                else:
                    kf = ["僅單一標的，無法比較風險報酬分布"] if _lang == "zh-TW" else ["Only one ETF is selected, so there's no risk-return spread to compare"]
                    insight = "建議加入至少一檔其他 ETF 以評估相對風險報酬位置" if _lang == "zh-TW" else "Consider adding at least one more ETF to gauge its relative risk-return position"
                    risk = "單一標的的風險完全取決於該檔本身，缺乏分散效果" if _lang == "zh-TW" else "A single holding's risk is fully tied to that one ETF, with no diversification benefit"
                _ai_interpretation(kf, insight, risk)
                # Risk-return positioning directly informs which ETF to
                # size up/down, so this is one of the more decision-relevant
                # charts on the page -- context_text reuses ONLY the kf/
                # insight/risk strings already rendered above by
                # _ai_interpretation(), never a fresh computation.
                ai_interpret_button(
                    "etf_risk_return_scatter_ai_interpret", st.session_state,
                    " ".join(kf) + " " + insight + " " + risk,
                )

# ══════════════════════════════════════════════════════════════════════════
# HOLDINGS & EXPOSURE -- "What does it actually own?"
# ══════════════════════════════════════════════════════════════════════════
elif workspace == "Holdings":
    import html as _hld_html
    from src.holdings import (
        get_etf_holdings, itemized_holdings, total_disclosed_weight, search_holdings,
        STATUS_UPDATED, STATUS_CACHED, STATUS_NOT_SUPPORTED,
    )
    from src.financial_metrics import largest_position, top_n_concentration, effective_number_of_holdings
    from src.etf_database import get_related_tickers

    _hld_snapshot = get_etf_holdings(_focus_ticker)

    _HOLD_VIEWS = ["Overview", "Top", "All", "Search"]
    _hold_labels = {
        "Overview": t("etf_holdings_nav_overview"), "Top": t("etf_holdings_nav_top"),
        "All": t("etf_holdings_nav_all"), "Search": t("etf_holdings_nav_search"),
    }
    _hvk, _hvv = _shadow_default("etf_holdings_view", "Overview")
    hold_view = st.segmented_control(
        "hold_nav", _HOLD_VIEWS, default=_hvv if _hvv in _HOLD_VIEWS else "Overview",
        format_func=lambda w: _hold_labels.get(w, w), key="etf_holdings_view", label_visibility="collapsed",
    ) or "Overview"
    st.session_state[_hvk] = hold_view

    # Identity / date / source -- shown on every sub-view (cheap captions,
    # never omitted per PRODUCT SPEC: never present weights without a date).
    if _hld_snapshot.issuer or _hld_snapshot.exchange:
        st.caption(t("etf_holdings_identity_line", issuer=_hld_snapshot.issuer or "—", exchange=_hld_snapshot.exchange or "—"))
    if _hld_snapshot.isin:
        st.caption(t("etf_holdings_identity_isin", isin=_hld_snapshot.isin))
    _hld_related = get_related_tickers(_focus_ticker)
    if _hld_related:
        st.caption(t("etf_holdings_related_lines", tickers=", ".join(_hld_related)))
    if _hld_snapshot.data_date:
        st.caption(t("etf_holdings_data_as_of", date=_hld_snapshot.data_date))
    st.caption(
        f"{t('etf_holdings_source_label', source=_hld_snapshot.source)} · "
        f"[{t('etf_holdings_view_source')}]({_hld_snapshot.source_url})"
    )

    if _hld_snapshot.status == STATUS_CACHED:
        st.info(t("etf_holdings_status_cached"))
    elif _hld_snapshot.status == STATUS_NOT_SUPPORTED:
        error_state(t("etf_holdings_status_not_supported_title"), t("etf_holdings_status_not_supported_desc"))
    elif not _hld_snapshot.holdings:
        error_state(t("etf_holdings_status_unavailable_title"), t("etf_holdings_status_unavailable_desc"))

    if _hld_snapshot.holdings:
        _hld_items = itemized_holdings(_hld_snapshot)
        _hld_weights = {h.holding_ticker: h.weight for h in _hld_items}

        def _hld_label(label: str, tooltip: str) -> str:
            safe = _hld_html.escape(tooltip, quote=True)
            return f'{label}<span title="{safe}" style="cursor:help;opacity:0.55;margin-left:4px;font-size:11px;">&#9432;</span>'

        if hold_view == "Overview":
            _hld_largest_ticker, _hld_largest_weight = largest_position(_hld_weights)
            _hld_top5 = top_n_concentration(_hld_weights, 5)
            _hld_top10 = top_n_concentration(_hld_weights, 10)
            _hld_effective = effective_number_of_holdings(_hld_weights)

            from src.theme import COLORS as _HLD_C
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.markdown(kpi_card(
                    _hld_label(t("etf_holdings_metric_largest"), t("etf_holdings_tooltip_largest")),
                    f"{_hld_largest_ticker} {_hld_largest_weight:.2%}" if _hld_largest_ticker else "—",
                    color=_HLD_C["primary"], icon="target",
                ), unsafe_allow_html=True)
            with mcol2:
                st.markdown(kpi_card(
                    _hld_label(t("etf_holdings_metric_top5"), t("etf_holdings_tooltip_top5")),
                    f"{_hld_top5:.2%}", color=_HLD_C["purple"], icon="pie-chart",
                ), unsafe_allow_html=True)
            with mcol3:
                st.markdown(kpi_card(
                    _hld_label(t("etf_holdings_metric_top10"), t("etf_holdings_tooltip_top10")),
                    f"{_hld_top10:.2%}", color=_HLD_C["cyan"], icon="pie-chart",
                ), unsafe_allow_html=True)
            with mcol4:
                st.markdown(kpi_card(
                    _hld_label(t("etf_holdings_metric_count"), t("etf_holdings_tooltip_count")),
                    str(len(_hld_items)), color=_HLD_C["warning"], icon="layers",
                ), unsafe_allow_html=True)
            with mcol5:
                st.markdown(kpi_card(
                    _hld_label(t("etf_holdings_metric_effective"), t("etf_holdings_tooltip_effective")),
                    f"{_hld_effective:.2f}" if _hld_effective else "—",
                    color=_HLD_C["success"], icon="layers",
                ), unsafe_allow_html=True)

            _hld_coverage = total_disclosed_weight(_hld_snapshot)
            st.markdown(
                f"**{t('etf_holdings_coverage_label')}: {_hld_coverage:.1%}**  \n"
                f"<span style='color:var(--text-muted);font-size:12px;'>{t('etf_holdings_coverage_note')}</span>",
                unsafe_allow_html=True,
            )

        elif hold_view == "Top":
            from src.theme import COLORS as _HLD_C
            _hld_top10_rows = _hld_items[:10]
            _hld_bar_html = "".join(
                '<div style="margin-bottom:10px;">'
                '<div style="display:flex;justify-content:space-between;font-size:12.5px;'
                f'color:var(--text);margin-bottom:4px;"><span><b>{h.holding_ticker}</b> {h.holding_name}</span>'
                f'<span style="font-weight:700;">{h.weight:.2%}</span></div>'
                '<div style="background:var(--border);border-radius:999px;height:8px;overflow:hidden;">'
                f'<div style="background:{_HLD_C["primary"]};width:{min(100.0, h.weight * 100):.2f}%;'
                'height:100%;border-radius:999px;"></div></div></div>'
                for h in _hld_top10_rows
            )
            st.markdown(_hld_bar_html or f"<div>{t('etf_holdings_empty_no_holdings')}</div>", unsafe_allow_html=True)

        elif hold_view == "All":
            _hld_asset_type_key = {
                "Equity": "etf_holdings_asset_type_equity", "Cash": "etf_holdings_asset_type_cash",
                "Bond": "etf_holdings_asset_type_bond", "Other": "etf_holdings_asset_type_other",
                "Preferred": "etf_holdings_asset_type_preferred", "Convertible": "etf_holdings_asset_type_convertible",
                "Futures": "etf_holdings_asset_type_futures", "ETF": "etf_holdings_asset_type_etf",
                "Derivative": "etf_holdings_asset_type_derivative",
            }
            _hld_table_rows = [
                {
                    t("etf_holdings_col_ticker"): h.holding_ticker,
                    t("etf_holdings_col_name"): (
                        f"{h.holding_name} ({t('etf_holdings_aggregate_note')})" if h.is_aggregate else h.holding_name
                    ),
                    t("etf_holdings_col_weight"): h.weight,
                    t("etf_holdings_col_asset_type"): t(_hld_asset_type_key.get(h.asset_type, "etf_holdings_asset_type_other")),
                }
                for h in sorted(_hld_snapshot.holdings, key=lambda h: h.weight, reverse=True)
            ]
            _hld_df = pd.DataFrame(_hld_table_rows)
            # Fixed-height, internally scrollable (PRODUCT SPEC section 7/15:
            # large holdings tables must scroll inside their own container,
            # never stretch the page).
            st.dataframe(
                _hld_df, use_container_width=True, hide_index=True, height=420,
                column_config={t("etf_holdings_col_weight"): st.column_config.NumberColumn(format="percent")},
            )
            chart_caption(t("etf_holdings_all_table_caption"))

        else:  # Search
            _hld_query = st.text_input(
                t("etf_holdings_search_label"), placeholder=t("etf_holdings_search_placeholder"),
                key="holdings_search",
            )
            _hld_matches = search_holdings(_hld_snapshot, _hld_query)
            if _hld_query and not _hld_matches:
                st.info(t("etf_holdings_search_no_match", query=_hld_query))
            elif _hld_query:
                _hld_asset_type_key = {
                    "Equity": "etf_holdings_asset_type_equity", "Cash": "etf_holdings_asset_type_cash",
                    "Bond": "etf_holdings_asset_type_bond", "Other": "etf_holdings_asset_type_other",
                    "Preferred": "etf_holdings_asset_type_preferred", "Convertible": "etf_holdings_asset_type_convertible",
                    "Futures": "etf_holdings_asset_type_futures", "ETF": "etf_holdings_asset_type_etf",
                    "Derivative": "etf_holdings_asset_type_derivative",
                }
                _hld_table_rows = [
                    {
                        t("etf_holdings_col_ticker"): h.holding_ticker,
                        t("etf_holdings_col_name"): h.holding_name,
                        t("etf_holdings_col_weight"): h.weight,
                        t("etf_holdings_col_asset_type"): t(_hld_asset_type_key.get(h.asset_type, "etf_holdings_asset_type_other")),
                    }
                    for h in sorted(_hld_matches, key=lambda h: h.weight, reverse=True)
                ]
                st.dataframe(
                    pd.DataFrame(_hld_table_rows), use_container_width=True, hide_index=True,
                    column_config={t("etf_holdings_col_weight"): st.column_config.NumberColumn(format="percent")},
                )
                chart_caption(t("etf_holdings_search_table_caption"))

# ══════════════════════════════════════════════════════════════════════════
# COMPARE -- "How does it compare with other ETFs?"
# ══════════════════════════════════════════════════════════════════════════
elif workspace == "Compare":
    if len(etf_prices.columns) < 2:
        st.info(t("etf_compare_empty_state"))
    else:
        _CMP_VIEWS = ["Rankings", "Correlation"]
        _cmp_view_labels = {"Rankings": t("etf_compare_nav_rankings"), "Correlation": t("etf_compare_nav_correlation")}
        _cvk, _cvv = _shadow_default("etf_compare_view", "Rankings")
        cmp_view = st.segmented_control(
            "cmp_nav", _CMP_VIEWS, default=_cvv if _cvv in _CMP_VIEWS else "Rankings",
            format_func=lambda w: _cmp_view_labels.get(w, w), key="etf_compare_view", label_visibility="collapsed",
        ) or "Rankings"
        st.session_state[_cvk] = cmp_view

        st.caption(t("etf_compare_normalized_xref"))
        _ai_summary_data = {tk: _ai_summary_entry(tk, _lang) for tk in etf_prices.columns}
        # Issue #20 release-gate review: the pre-redesign "ETF Compare
        # Score" table used to silently EXCLUDE any ticker with fewer than
        # 10 valid price points (too little history for a numerically
        # stable Quant Score/Sharpe/moving-average signal). Consolidating
        # onto one shared _ai_summary_data dict (section 2A) means every
        # selected ticker now gets a score no matter how little history it
        # has -- which is more consistent across Summary/Ranking/Compare
        # Score, but must disclose the limitation rather than silently
        # present a thin-history score at the same confidence as a
        # fully-populated one.
        _thin_history_tickers = [
            tk for tk in etf_prices.columns if not has_sufficient_history(etf_prices[tk].dropna())
        ]
        if _thin_history_tickers:
            st.warning(t("etf_thin_history_warning", tickers=", ".join(_thin_history_tickers)))

        if cmp_view == "Rankings":
            # ── ETF Analytical Summary (all selected tickers) ───────────────
            section_header(t("etf_analytical_summary_title"))
            st.caption(t("etf_signal_semantics_note"))
            _sum_cols = st.columns(len(etf_prices.columns))
            for i, ticker in enumerate(etf_prices.columns):
                with _sum_cols[i]:
                    _render_ai_summary_card(ticker, _ai_summary_data[ticker])

            # ── ETF Ranking -- the ONE canonical ranking/comparison table
            # (Issue #39 priority 2). Before this, a separate "ETF Compare
            # Score" table below duplicated the same ETF/Score/Trend/Risk/
            # Portfolio View fields from a SEPARATE risk-threshold formula
            # (0.12/0.25 vs this table's 0.15/0.28), so the same ETF could
            # show a different Risk Level in each table. Now every row's
            # risk_level_from_vol()/expected_return_label_from_ann_ret()
            # call reads the same _ai_summary_data entry already shown in
            # ETF Analytical Summary above, so nothing here can disagree
            # with itself. ───────────────────────────────────────────────
            section_header(t("etf_ranking_title"))
            _ranked = sorted(_ai_summary_data.items(), key=lambda kv: kv[1]["score"], reverse=True)
            _rank_row_html = []
            for _idx, (_r_ticker, _r_data) in enumerate(_ranked):
                _r_trend_color = _SUM_TREND_META[_r_data["trend"]]
                _r_risk = t_etf_risk_level(risk_level_from_vol(_r_data["vol"]))
                _r_return_label = t_etf_return_label(expected_return_label_from_ann_ret(_r_data["ret_ann"]))
                # Issue #39 other-detail-1: no medal emoji -- rank is a
                # plain number, the top row gets a restrained soft-primary
                # highlight instead.
                _r_top = _idx == 0
                _r_row_style = "background:var(--primary-soft);" if _r_top else ""
                _r_lead_style = "color:var(--primary);font-weight:800;" if _r_top else "color:var(--text);font-weight:800;"
                _rank_row_html.append(
                    f'<tr style="{_r_row_style}">'
                    f'<td style="padding:9px 12px;{_r_lead_style}border-bottom:1px solid var(--border);">{_idx + 1}</td>'
                    f'<td style="padding:9px 12px;{_r_lead_style}border-bottom:1px solid var(--border);">{_r_ticker}</td>'
                    f'<td style="padding:9px 12px;color:var(--text);font-weight:700;border-bottom:1px solid var(--border);">{_r_data["score"]}</td>'
                    f'<td style="padding:9px 12px;color:{_r_trend_color};font-weight:700;border-bottom:1px solid var(--border);">{t_trend_signal(_r_data["trend"])}</td>'
                    f'<td style="padding:9px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{_r_risk}</td>'
                    f'<td style="padding:9px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{_r_return_label}</td>'
                    f'<td style="padding:9px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{t_portfolio_view(_r_data["portfolio_view"])}</td>'
                    '</tr>'
                )
            _rank_headers = [
                t("etf_rank_col_rank"), t("etf_rank_col_etf"), t("etf_rank_col_score"),
                t("etf_rank_col_trend"), t("etf_rank_col_risk"), t("metric_expected_return"), t("etf_rank_col_view"),
            ]
            _rank_header_html = "".join(
                f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;'
                f'letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{h}</th>'
                for h in _rank_headers
            )
            _rank_col, _why_col = st.columns([3, 2])
            with _rank_col:
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'padding:4px 8px;overflow-x:auto;box-shadow:var(--shadow-sm);">'
                    '<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr>{_rank_header_html}</tr></thead>'
                    f'<tbody>{"".join(_rank_row_html)}</tbody>'
                    '</table></div>',
                    unsafe_allow_html=True,
                )
            with _why_col:
                _top_ticker, _top_data = _ranked[0]
                _others = [d for tkr, d in _ranked if tkr != _top_ticker]
                _why_reasons = []
                if _top_data["sharpe"] >= max(o["sharpe"] for o in _others):
                    _why_reasons.append("Sharpe Ratio")
                if _top_data["mom"] >= max(o["mom"] for o in _others):
                    _why_reasons.append(t_compare_metric("Momentum"))
                if _top_data["ret_ann"] >= max(o["ret_ann"] for o in _others):
                    _why_reasons.append("長期報酬" if _lang == "zh-TW" else "long-term return")
                if _top_data["vol"] <= min(o["vol"] for o in _others):
                    _why_reasons.append("波動控制" if _lang == "zh-TW" else "volatility control")

                if _why_reasons:
                    _top3 = _why_reasons[:3]

                    def _join_with_last(parts, sep, last_sep):
                        if len(parts) == 1:
                            return parts[0]
                        return sep.join(parts[:-1]) + last_sep + parts[-1]

                    if _lang == "zh-TW":
                        _phrases = [(f"最佳 {r}" if r == "波動控制" else f"最高 {r}") for r in _top3]
                        _joined = _join_with_last(_phrases, "、", " 與")
                        _why_text = f"{_top_ticker} 在目前所有 ETF 中擁有{_joined}，因此目前排名第一。"
                    else:
                        _joined = _join_with_last(_top3, ", ", " and ")
                        _why_text = f"{_top_ticker} has the best {_joined} among all currently selected ETFs, putting it in first place."
                else:
                    _why_text = (
                        f"{_top_ticker} 並未在單一指標中領先，但整體風險與報酬表現最為均衡，綜合評分因此排名第一。"
                        if _lang == "zh-TW" else
                        f"{_top_ticker} doesn't lead on any single metric, but its overall balance of risk and return gives it the highest combined score."
                    )
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'padding:16px 18px;height:100%;box-shadow:var(--shadow-sm);">'
                    '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
                    + ("為什麼第一？" if _lang == "zh-TW" else "Why #1?") + '</div>'
                    f'<div style="color:var(--text-secondary);font-size:12.5px;line-height:1.7;">{_why_text}</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── Compare Mode (head-to-head, 2 ETFs) ────────────────────────
            section_header(t("etf_compare_mode_title"))
            _vs_options = etf_prices.columns.tolist()
            _vs_c1, _vs_c2 = st.columns(2)
            with _vs_c1:
                vs_ticker_a = st.selectbox("ETF A", _vs_options, index=0, key="compare_mode_ticker_a")
            with _vs_c2:
                vs_ticker_b = st.selectbox("ETF B", [o for o in _vs_options if o != vs_ticker_a], index=0, key="compare_mode_ticker_b")

            _vs_pa = etf_prices[vs_ticker_a].dropna()
            _vs_pb = etf_prices[vs_ticker_b].dropna()
            if len(_vs_pa) > 10 and len(_vs_pb) > 10:
                _vs_a = {
                    "Return": annualized_return(_vs_pa), "Risk": value_at_risk(_vs_pa),
                    "Sharpe": sharpe_ratio(_vs_pa, risk_free_rate), "Volatility": annualized_volatility(_vs_pa),
                    "Drawdown": maximum_drawdown(_vs_pa), "Momentum": momentum(_vs_pa, 10).iloc[-1],
                }
                _vs_b = {
                    "Return": annualized_return(_vs_pb), "Risk": value_at_risk(_vs_pb),
                    "Sharpe": sharpe_ratio(_vs_pb, risk_free_rate), "Volatility": annualized_volatility(_vs_pb),
                    "Drawdown": maximum_drawdown(_vs_pb), "Momentum": momentum(_vs_pb, 10).iloc[-1],
                }
                for _k in ("Return", "Risk", "Sharpe", "Volatility", "Drawdown", "Momentum"):
                    if pd.isna(_vs_a[_k]):
                        _vs_a[_k] = 0.0
                    if pd.isna(_vs_b[_k]):
                        _vs_b[_k] = 0.0

                _vs_higher_wins = {"Return", "Risk", "Sharpe", "Drawdown", "Momentum"}
                _vs_fmt = {
                    "Return": lambda v: f"{v:+.2%}", "Risk": lambda v: f"{v:.2%}",
                    "Sharpe": lambda v: f"{v:.2f}", "Volatility": lambda v: f"{v:.2%}",
                    "Drawdown": lambda v: f"{v:.2%}", "Momentum": lambda v: f"{v:+.2%}",
                }
                _vs_wins = {vs_ticker_a: [], vs_ticker_b: []}
                _vs_row_html = []
                for _metric in ("Return", "Risk", "Sharpe", "Volatility", "Drawdown", "Momentum"):
                    _va, _vb = _vs_a[_metric], _vs_b[_metric]
                    if _metric in _vs_higher_wins:
                        _winner = vs_ticker_a if _va > _vb else (vs_ticker_b if _vb > _va else None)
                    else:
                        _winner = vs_ticker_a if _va < _vb else (vs_ticker_b if _vb < _va else None)
                    if _winner:
                        _vs_wins[_winner].append(_metric)
                    # Issue #39 other-detail-2: no repeated trophy per row --
                    # the winner cell is plain bold/success-colored ticker
                    # text (or an em dash for a tie on that one metric).
                    _winner_html = f'<strong style="color:var(--success);">{_winner}</strong>' if _winner else "—"
                    _vs_row_html.append(
                        '<tr>'
                        f'<td style="padding:9px 12px;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid var(--border);">{t_compare_metric(_metric)}</td>'
                        f'<td style="padding:9px 12px;color:var(--text);border-bottom:1px solid var(--border);">{_vs_fmt[_metric](_va)}</td>'
                        f'<td style="padding:9px 12px;color:var(--text);border-bottom:1px solid var(--border);">{_vs_fmt[_metric](_vb)}</td>'
                        f'<td style="padding:9px 12px;border-bottom:1px solid var(--border);">{_winner_html}</td>'
                        '</tr>'
                    )
                _vs_header_html = (
                    f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{t("etf_compare_col_metric")}</th>'
                    f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{vs_ticker_a}</th>'
                    f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{vs_ticker_b}</th>'
                    f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{t("etf_compare_col_winner")}</th>'
                )
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'padding:4px 8px;overflow-x:auto;box-shadow:var(--shadow-sm);margin-bottom:10px;">'
                    '<table style="width:100%;border-collapse:collapse;">'
                    f'<thead><tr>{_vs_header_html}</tr></thead>'
                    f'<tbody>{"".join(_vs_row_html)}</tbody>'
                    '</table></div>',
                    unsafe_allow_html=True,
                )
                _vs_wins_a, _vs_wins_b = len(_vs_wins[vs_ticker_a]), len(_vs_wins[vs_ticker_b])
                _vs_overall = None if _vs_wins_a == _vs_wins_b else (vs_ticker_a if _vs_wins_a > _vs_wins_b else vs_ticker_b)
                _vs_risk_metrics = {"Risk", "Volatility", "Drawdown"}
                _vs_risk_wins_a = len([m for m in _vs_wins[vs_ticker_a] if m in _vs_risk_metrics])
                _vs_risk_wins_b = len([m for m in _vs_wins[vs_ticker_b] if m in _vs_risk_metrics])
                _vs_risk_winner = None if _vs_risk_wins_a == _vs_risk_wins_b else (vs_ticker_a if _vs_risk_wins_a > _vs_risk_wins_b else vs_ticker_b)

                # Issue #39 other-detail-2: no trophy on the Overall Winner
                # label either -- a clean, bold ticker in the success color
                # (or a neutral "Tie" label) is enough emphasis on its own.
                _overall_label = _vs_overall if _vs_overall else ("平手" if _lang == "zh-TW" else "Tie")
                _overall_color = "var(--success)" if _vs_overall else "var(--text)"
                st.markdown(
                    '<div style="margin:4px 0 4px 0;">'
                    '<div style="color:var(--text-muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
                    + ("整體贏家" if _lang == "zh-TW" else "Overall Winner") + '</div>'
                    f'<div style="color:{_overall_color};font-weight:800;font-size:22px;">{_overall_label}</div></div>',
                    unsafe_allow_html=True,
                )
                # One concise "N of 6 metrics" summary in place of the old
                # per-row repeated trophy -- distinct from the Comparison
                # Notes explanation below, which explains WHY, not just the
                # tally.
                if _vs_wins_a == 6 or _vs_wins_b == 6:
                    _vs_sweep_etf = vs_ticker_a if _vs_wins_a == 6 else vs_ticker_b
                    st.caption(t("etf_compare_summary_sweep", etf=_vs_sweep_etf, total=6))
                else:
                    st.caption(t(
                        "etf_compare_summary_split",
                        etf_a=vs_ticker_a, count_a=_vs_wins_a, etf_b=vs_ticker_b, count_b=_vs_wins_b, total=6,
                    ))

                _cn = {"Return": "報酬率", "Risk": "風險（VaR）", "Sharpe": "夏普比率",
                       "Volatility": "波動度", "Drawdown": "最大回撤", "Momentum": "動能"}
                if _vs_overall is None:
                    explanation = (
                        f"{vs_ticker_a} 與 {vs_ticker_b} 整體表現相近，各項指標互有領先，難分軒輊，建議依個人風險偏好選擇。"
                        if _lang == "zh-TW" else
                        f"{vs_ticker_a} and {vs_ticker_b} are closely matched overall, each leading on different metrics -- the choice comes down to personal risk preference."
                    )
                elif _vs_risk_winner is None or _vs_risk_winner == _vs_overall:
                    _reason_metrics = _vs_wins[_vs_overall]
                    if _lang == "zh-TW":
                        _reason_str = "、".join(_cn[m] for m in _reason_metrics)
                        explanation = f"{_vs_overall} 在{_reason_str}上表現優於{(vs_ticker_b if _vs_overall == vs_ticker_a else vs_ticker_a)}，因此整體勝出。"
                    else:
                        _reason_str = ", ".join(_reason_metrics)
                        explanation = f"{_vs_overall} outperforms {(vs_ticker_b if _vs_overall == vs_ticker_a else vs_ticker_a)} on {_reason_str}, making it the overall winner."
                else:
                    _growth_reasons = [m for m in _vs_wins[_vs_overall] if m not in _vs_risk_metrics] or _vs_wins[_vs_overall]
                    _risk_reasons = [m for m in _vs_wins[_vs_risk_winner] if m in _vs_risk_metrics]
                    if _lang == "zh-TW":
                        _growth_str = "、".join(_cn[m] for m in _growth_reasons)
                        _risk_str = "、".join(_cn[m] for m in _risk_reasons)
                        explanation = (
                            f"{_vs_overall} 在{_growth_str}上表現較佳，整體勝出；"
                            f"不過 {_vs_risk_winner} 在{_risk_str}上風險較低，更適合保守型投資人。"
                        )
                    else:
                        _growth_str = ", ".join(_growth_reasons)
                        _risk_str = ", ".join(_risk_reasons)
                        explanation = (
                            f"{_vs_overall} leads on {_growth_str} and wins overall; "
                            f"however, {_vs_risk_winner} carries lower risk on {_risk_str}, making it more suitable for conservative investors."
                        )
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);padding:12px 16px;">'
                    '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">'
                    + t("etf_comparison_notes_title") + '</div>'
                    f'<div style="color:var(--text-secondary);font-size:12.5px;line-height:1.7;">{explanation}</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Investment Verdict (page-level synthesis over ALL selected
            # tickers) -- closing section of Compare. ───────────────────────
            section_header(t("etf_investment_verdict_title"))
            _verdict_tickers = list(etf_prices.columns)
            _v_scores = [_ai_summary_data[t]["score"] for t in _verdict_tickers]
            _v_vols = [_ai_summary_data[t]["vol"] for t in _verdict_tickers]
            _v_rets = [_ai_summary_data[t]["ret_ann"] for t in _verdict_tickers]
            _v_overall_score = int(round(sum(_v_scores) / len(_v_scores)))
            _v_avg_vol = sum(_v_vols) / len(_v_vols)
            _v_avg_ret = sum(_v_rets) / len(_v_rets)

            if _v_overall_score >= 80:
                _v_trend = "Strong Bullish"
            elif _v_overall_score >= 60:
                _v_trend = "Bullish"
            elif _v_overall_score >= 40:
                _v_trend = "Neutral"
            elif _v_overall_score >= 20:
                _v_trend = "Bearish"
            else:
                _v_trend = "Strong Bearish"

            if _v_avg_vol < 0.10:
                _v_risk = "Low"
            elif _v_avg_vol < 0.18:
                _v_risk = "Medium"
            elif _v_avg_vol < 0.26:
                _v_risk = "Medium High"
            elif _v_avg_vol < 0.35:
                _v_risk = "High"
            else:
                _v_risk = "Very High"

            if _v_avg_ret < 0.08:
                _v_exp_return = "Low"
            elif _v_avg_ret < 0.15:
                _v_exp_return = "Medium"
            elif _v_avg_ret < 0.25:
                _v_exp_return = "High"
            else:
                _v_exp_return = "Very High"

            if _v_avg_vol < 0.15:
                _v_horizon = "Short-to-Medium Term"
            elif _v_avg_vol < 0.28:
                _v_horizon = "Medium-to-Long Term"
            else:
                _v_horizon = "Long Term"

            if _v_risk in ("High", "Very High") and _v_exp_return in ("High", "Very High"):
                _v_suitable = "Growth Investors"
            elif _v_risk == "Low" and _v_exp_return in ("Low", "Medium"):
                _v_suitable = "Conservative Investors"
            else:
                _v_suitable = "Balanced Investors"

            _V_TREND_ZH = {"Strong Bullish": "強力多頭", "Bullish": "多頭", "Neutral": "中性", "Bearish": "空頭", "Strong Bearish": "強力空頭"}
            _V_TREND_COLOR = {"Strong Bullish": "var(--success)", "Bullish": "var(--success)", "Neutral": "var(--warning)", "Bearish": "var(--danger)", "Strong Bearish": "var(--danger)"}
            _v_trend_display = _V_TREND_ZH[_v_trend] if _lang == "zh-TW" else _v_trend
            _v_trend_color = _V_TREND_COLOR[_v_trend]

            _V_MARKET_CLAUSE = {
                "Strong Bullish": ("目前市場整體呈現強勁多頭格局", "The market is currently showing a strong bullish trend overall"),
                "Bullish": ("目前市場仍維持多頭", "The market is currently maintaining a bullish trend"),
                "Neutral": ("目前市場呈現盤整格局", "The market is currently consolidating"),
                "Bearish": ("目前市場呈現空頭格局", "The market is currently in a bearish trend"),
                "Strong Bearish": ("目前市場呈現強勁空頭格局", "The market is currently in a strong bearish trend"),
            }
            _clauses = [_V_MARKET_CLAUSE[_v_trend][0 if _lang == "zh-TW" else 1]]

            if len(_verdict_tickers) == 1:
                _only = _verdict_tickers[0]
                if _lang == "zh-TW":
                    _clauses.append(f"{_only} 目前是唯一納入分析的標的，整體評分為 {_ai_summary_data[_only]['score']} 分")
                else:
                    _clauses.append(f"{_only} is the only ETF currently included in the analysis, with an overall score of {_ai_summary_data[_only]['score']}")
            else:
                _assigned = set()
                _growth_leader = max(_verdict_tickers, key=lambda t: _ai_summary_data[t]["ret_ann"])
                _assigned.add(_growth_leader)
                _clauses.append(f"{_growth_leader} 擁有最佳長期成長能力" if _lang == "zh-TW" else f"{_growth_leader} offers the best long-term growth potential")

                _remaining = [t for t in _verdict_tickers if t not in _assigned]
                if len(_verdict_tickers) >= 3 and _remaining:
                    _core_holding = max(_remaining, key=lambda t: _ai_summary_data[t]["sharpe"])
                    _assigned.add(_core_holding)
                    _clauses.append(f"{_core_holding} 適合作為核心持股" if _lang == "zh-TW" else f"{_core_holding} is well suited as a core holding")

                    _remaining2 = [t for t in _verdict_tickers if t not in _assigned]
                    if _remaining2:
                        _corr = etf_prices[_verdict_tickers].pct_change().dropna().corr()
                        _diversifier = min(_remaining2, key=lambda t: _corr[t].drop(t).mean() if t in _corr.columns else 0)
                        _clauses.append(f"{_diversifier} 適合分散投資" if _lang == "zh-TW" else f"{_diversifier} is well suited for diversification")
                elif _remaining:
                    _other = _remaining[0]
                    _clauses.append(f"{_other} 可作為分散配置的補充" if _lang == "zh-TW" else f"{_other} can serve as a diversifying complement")

            v_recommendation = ("，<br>".join(_clauses) + "。") if _lang == "zh-TW" else (",<br>".join(_clauses) + ".")

            _v_col1, _v_col2 = st.columns([2, 3])
            with _v_col1:
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'padding:18px 20px;box-shadow:var(--shadow-sm);height:100%;">'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;">'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("整體評分" if _lang == "zh-TW" else "Overall Rating") + '</div>'
                    f'<div style="color:var(--text);font-weight:800;font-size:22px;">{_v_overall_score}</div></div>'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("趨勢" if _lang == "zh-TW" else "Trend") + '</div>'
                    f'<div style="color:{_v_trend_color};font-weight:700;font-size:15px;">{_v_trend_display}</div></div>'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("風險" if _lang == "zh-TW" else "Risk") + '</div>'
                    f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{t_etf_risk_level(_v_risk)}</div></div>'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("預期報酬" if _lang == "zh-TW" else "Expected Return") + '</div>'
                    f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{t_etf_verdict_return(_v_exp_return)}</div></div>'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("投資期間" if _lang == "zh-TW" else "Investment Horizon") + '</div>'
                    f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{t_investment_horizon(_v_horizon)}</div></div>'
                    '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
                    + ("適合對象" if _lang == "zh-TW" else "Suitable For") + '</div>'
                    f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{t_suitable_investor(_v_suitable)}</div></div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            with _v_col2:
                st.markdown(
                    '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                    'padding:18px 20px;box-shadow:var(--shadow-sm);height:100%;">'
                    '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
                    + t("etf_verdict_notes_title") + '</div>'
                    f'<div style="color:var(--text-secondary);font-size:13px;line-height:1.8;">{v_recommendation}</div></div>',
                    unsafe_allow_html=True,
                )

        else:  # Correlation
            # ── ETF DNA ──────────────────────────────────────────────────
            section_header(t("etf_dna_title"))
            _DNA_DIMENSIONS = [
                (t("etf_dna_growth"), "var(--success)"), (t("etf_dna_risk"), "var(--danger)"),
                (t("etf_dna_momentum"), "var(--primary)"), (t("etf_dna_diversification"), "var(--purple)"),
                (t("etf_dna_liquidity"), "var(--cyan)"),
            ]
            _bench_returns = bench_prices.dropna().pct_change().dropna() if bench_prices is not None else None
            dna_cols = st.columns(len(etf_prices.columns))
            for i, ticker in enumerate(etf_prices.columns):
                with dna_cols[i]:
                    p = etf_prices[ticker].dropna()
                    d_ret = annualized_return(p)
                    d_vol = annualized_volatility(p)
                    d_mom_last = momentum(p, 10).iloc[-1]
                    d_mom = d_mom_last if pd.notna(d_mom_last) else 0.0
                    d_returns = p.pct_change().dropna()
                    if _bench_returns is not None and len(d_returns) > 5:
                        _aligned = pd.concat([d_returns, _bench_returns], axis=1).dropna()
                        d_corr = _aligned.iloc[:, 0].corr(_aligned.iloc[:, 1]) if len(_aligned) > 5 else None
                    else:
                        d_corr = None
                    d_zero_frac = (d_returns == 0).sum() / len(d_returns) if len(d_returns) > 0 else 0.0

                    d_growth = int(round(max(0, min(100, 50 + d_ret * 150))))
                    d_risk = int(round(max(0, min(100, d_vol / 0.40 * 100))))
                    d_momentum = int(round(max(0, min(100, 50 + d_mom * 300))))
                    d_diversification = int(round(max(0, min(100, d_corr * 100)))) if d_corr is not None else 50
                    d_liquidity = int(round(max(0, min(100, 100 - d_zero_frac * 400))))
                    d_values = [d_growth, d_risk, d_momentum, d_diversification, d_liquidity]

                    bars_html = "".join(
                        '<div style="margin-bottom:10px;">'
                        '<div style="display:flex;justify-content:space-between;font-size:11.5px;'
                        f'color:var(--text-secondary);margin-bottom:4px;"><span>{label}</span>'
                        f'<span style="color:var(--text);font-weight:700;">{value}</span></div>'
                        '<div style="background:var(--border);border-radius:999px;height:6px;overflow:hidden;">'
                        f'<div style="background:{color};width:{value}%;height:100%;border-radius:999px;"></div>'
                        '</div></div>'
                        for (label, color), value in zip(_DNA_DIMENSIONS, d_values)
                    )
                    st.markdown(
                        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
                        'padding:16px 18px;margin:6px 0;box-shadow:var(--shadow-sm);">'
                        f'<div style="color:var(--text);font-weight:800;font-size:15px;margin-bottom:12px;">{ticker}</div>'
                        f'{bars_html}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Correlation + Covariance ─────────────────────────────────
            section_header(t("etf_correlation_analysis_title"), t("etf_correlation_analysis_subtitle"))
            with chart_card(t("etf_correlation_heatmap_card")):
                corr = correlation_matrix(etf_prices)
                fig = correlation_heatmap(corr)
                st.plotly_chart(fig, use_container_width=True, key="etf_correlation_heatmap")
                chart_caption(t("etf_correlation_heatmap_caption"))
                _cols_c = corr.columns.tolist()
                _pairs = [(_cols_c[a], _cols_c[b], corr.iloc[a, b]) for a in range(len(_cols_c)) for b in range(a + 1, len(_cols_c))]
                if _pairs:
                    _highest = max(_pairs, key=lambda x: x[2])
                    _lowest = min(_pairs, key=lambda x: x[2])
                    _avg_corr = sum(pp[2] for pp in _pairs) / len(_pairs)
                    _avg_corr_by_ticker = {tk: corr[tk].drop(tk).mean() for tk in _cols_c if tk in corr.columns}
                    _least_diversifying = max(_avg_corr_by_ticker, key=_avg_corr_by_ticker.get)
                    _limited_benefit = _avg_corr_by_ticker[_least_diversifying] > 0.7

                    if _highest[2] > 0.7:
                        _highest_zh, _highest_en = "高度相關", "highly correlated"
                    elif _highest[2] > 0.4:
                        _highest_zh, _highest_en = "中度相關", "moderately correlated"
                    else:
                        _highest_zh, _highest_en = "相關性偏低（即使是相關性最高的一組）", "only mildly correlated (even as the highest pair in this set)"

                    if _lang == "zh-TW":
                        kf = [f"{_highest[0]} 與 {_highest[1]} {_highest_zh}（{_highest[2]:.2f}）"]
                        kf.append(f"{_lowest[0]} 與 {_lowest[1]} 相關性最低（{_lowest[2]:.2f}），分散效果較佳")
                        insight = (
                            f"加入 {_least_diversifying} 對分散效果有限，可考慮以相關性較低的標的替代以提升分散度"
                            if _limited_benefit else f"目前組合平均相關係數約為 {_avg_corr:.2f}，整體仍具備一定的分散化效益"
                        )
                        risk = "持股間相關性偏高時，市場下跌會同步拖累多檔標的，實際分散效果可能低於預期" if _limited_benefit else "相關係數會隨市場狀態變動，壓力時期（如系統性風險事件）相關性經常會上升，分散效果可能不如平時"
                    else:
                        kf = [f"{_highest[0]} and {_highest[1]} are {_highest_en} ({_highest[2]:.2f})"]
                        kf.append(f"{_lowest[0]} and {_lowest[1]} are the least correlated pair ({_lowest[2]:.2f}), offering better diversification")
                        insight = (
                            f"Adding {_least_diversifying} offers limited diversification benefit -- a lower-correlation ETF could add more diversification value"
                            if _limited_benefit else f"The portfolio's average correlation is about {_avg_corr:.2f}, still providing meaningful diversification benefit"
                        )
                        risk = "When holdings are highly correlated, a market downturn tends to drag several of them down together -- actual diversification may be less than it appears" if _limited_benefit else "Correlations shift with market conditions -- they often rise during systemic stress events, so diversification benefits can shrink exactly when they're needed most"
                    _ai_interpretation(kf, insight, risk)
                    # Correlation directly drives diversification decisions
                    # (which ETF to add/drop for the least overlap), so this
                    # is one of the more decision-relevant charts here.
                    # context_text reuses ONLY the kf/insight/risk strings
                    # already rendered above by _ai_interpretation().
                    ai_interpret_button(
                        "etf_correlation_heatmap_ai_interpret", st.session_state,
                        " ".join(kf) + " " + insight + " " + risk,
                    )

            with chart_card(t("etf_covariance_matrix_card"), t("etf_covariance_matrix_sub")):
                cov = covariance_matrix(etf_prices)
                st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)
                chart_caption(t("etf_covariance_matrix_caption"))

# ══════════════════════════════════════════════════════════════════════════
# DEEP ANALYSIS -- "What do the advanced quantitative indicators show?"
# ══════════════════════════════════════════════════════════════════════════
else:  # workspace == "Deep Analysis"
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
        with chart_card(t("etf_risk_metrics_table_card")):
            metrics_df = pd.DataFrame(metrics_data).set_index("Ticker")
            st.dataframe(metrics_df.T, use_container_width=True)
            chart_caption(t("etf_deep_metrics_table_caption"))
            # Full quantitative comparison table -- one of the more
            # decision-relevant views on the page. context_text is built
            # ONLY from the Annualized Return / Sharpe / Max Drawdown
            # columns already rendered in metrics_df.T above, never a
            # fresh computation.
            _deep_metrics_context = "; ".join(
                f"{_tk}: Annualized Return {_row.get('Annualized Return', '—')}, "
                f"Sharpe Ratio {_row.get('Sharpe Ratio', '—')}, "
                f"Maximum Drawdown {_row.get('Maximum Drawdown', '—')}"
                for _tk, _row in metrics_df.to_dict(orient="index").items()
            )
            ai_interpret_button(
                "etf_deep_metrics_table_ai_interpret", st.session_state, _deep_metrics_context,
            )

    with chart_card(t("etf_technical_indicators_title", ticker=_focus_ticker), t("etf_technical_indicators_sub")):
        import plotly.graph_objects as go
        from src.charts import apply_dark_theme
        from src.theme import COLORS as _C
        p = _focus_p
        bb = bollinger_bands(p)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=p.index, y=p, name=_focus_ticker, line=dict(color=_C["primary"], width=2)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Upper"], name=t("chart_bb_upper"),
                                     line=dict(color=_C["danger"], dash="dash", width=1)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Middle"], name=t("chart_bb_middle"),
                                     line=dict(color=_C["text_muted"], dash="dot", width=1)))
        fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Lower"], name=t("chart_bb_lower"),
                                     line=dict(color=_C["success"], dash="dash", width=1),
                                     fill="tonexty", fillcolor="rgba(52,211,153,0.05)"))
        fig_bb.update_layout(title=t("chart_bollinger_bands"), xaxis_title=t("chart_date"), yaxis_title=t("chart_price"), height=420)
        st.plotly_chart(apply_dark_theme(fig_bb), use_container_width=True, key="etf_bollinger_bands")
        chart_caption(t("etf_bollinger_bands_caption"))
        _bb_last = bb.iloc[-1]
        _last_price = p.iloc[-1]
        _band_width = (_bb_last["Upper"] - _bb_last["Lower"]) / _bb_last["Middle"] * 100 if _bb_last["Middle"] else 0
        _pos_pct = (_last_price - _bb_last["Lower"]) / (_bb_last["Upper"] - _bb_last["Lower"]) * 100 if _bb_last["Upper"] != _bb_last["Lower"] else 50
        _expanding = _band_width > 8
        if _lang == "zh-TW":
            if _pos_pct >= 85:
                kf = ["價格接近上軌，短線有過熱疑慮"]
                insight = "短線可能面臨獲利了結賣壓，追高需留意進場時機"
            elif _pos_pct <= 15:
                kf = ["價格接近下軌，短線可能超賣"]
                insight = "若基本面未變，短線超賣有機會出現反彈"
            else:
                kf = ["價格位於通道中段，未見極端訊號"]
                insight = "目前無明顯短線訊號，可持續觀察待突破再行動"
            kf.append(f"目前價格位於通道約 {_pos_pct:.0f}% 位置，通道寬度約 {_band_width:.1f}%")
            risk = "波動正在擴張，區間可能加大" if _expanding else "波動相對收斂，但盤整後仍可能出現方向性突破"
        else:
            if _pos_pct >= 85:
                kf = ["Price is near the upper band, short-term overbought risk"]
                insight = "May face short-term profit-taking pressure -- be mindful of entry timing when chasing strength"
            elif _pos_pct <= 15:
                kf = ["Price is near the lower band, possibly oversold"]
                insight = "If fundamentals are unchanged, an oversold bounce is possible in the short term"
            else:
                kf = ["Price sits mid-channel, no extreme signal"]
                insight = "No clear short-term signal right now -- worth waiting for a breakout before acting"
            kf.append(f"Current price is at about {_pos_pct:.0f}% of the band width, band width is about {_band_width:.1f}%")
            risk = "Volatility is expanding, so the range could widen further" if _expanding else "Volatility is relatively contained, but a directional breakout can still follow a quiet period"
        _ai_interpretation(kf, insight, risk)

# ── Downloads (always available, independent of active workspace -- cheap
# CSV export, not gated by the lazy-rendering rule which targets expensive
# chart rendering) ───────────────────────────────────────────────────────────
section_header(t("etf_download_data_title"))
col1, col2, col3 = st.columns(3)

with col1:
    csv = dataframe_to_csv(etf_prices)
    st.download_button(t("btn_download_raw_data"), csv, "etf_prices.csv", "text/csv")

with col2:
    returns_df = etf_prices.pct_change().dropna()
    csv2 = dataframe_to_csv(returns_df)
    st.download_button(t("btn_download_daily_returns"), csv2, "etf_returns.csv", "text/csv")

with col3:
    _dl_metrics_data = []
    for ticker in etf_prices.columns:
        p = etf_prices[ticker].dropna()
        if len(p) < 10:
            continue
        bench_p = bench_prices.dropna() if bench_prices is not None else None
        row = {"Ticker": ticker}
        row.update(compute_all_metrics(p, bench_p, risk_free_rate))
        _dl_metrics_data.append(row)
    if _dl_metrics_data:
        metrics_export = pd.DataFrame(_dl_metrics_data).set_index("Ticker")
        csv3 = dataframe_to_csv(metrics_export)
        st.download_button(t("btn_download_metrics"), csv3, "etf_metrics.csv", "text/csv")

disclaimer_box()
render_footer()
