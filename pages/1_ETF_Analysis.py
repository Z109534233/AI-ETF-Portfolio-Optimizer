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
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
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
    chart_card, render_footer, error_state
)
from src.i18n import t, t_country, get_language

st.set_page_config(
    page_title="ETF Analysis | AI ETF Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

load_css()

page_header(t("etf_analysis_title"), t("etf_analysis_subtitle"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('etf_sidebar_settings')}")

    # ── Region Selector (Global ETF Support) ─────────────────────────────
    # "United States" preserves the exact original ETF list (DEFAULT_ETFS)
    # so existing behavior is unchanged unless the user explicitly picks a
    # different region. Switching regions never breaks the custom-ticker
    # field below, which still accepts any free-text symbol.
    #
    # key="selected_region" makes st.session_state the primary source of
    # truth for this widget so ordinary reruns (tabs, chart type,
    # checkboxes) never reset it. On top of that, "_selected_region_shadow"
    # is a plain (non-widget) session_state entry that mirrors the value
    # after every render. This extra shadow is needed because Streamlit can
    # drop a widget's own keyed state if something earlier in the same
    # script run -- here, the language selector inside render_sidebar_nav(),
    # which sits above this widget -- calls st.rerun() before this widget
    # has been (re-)instantiated on that particular pass. A plain
    # session_state entry isn't tied to widget instantiation, so it
    # survives that and lets the widget recover on the next run instead of
    # silently falling back to index=1.
    ALL_REGIONS_LABEL = t("field_all_regions")
    region_options = [ALL_REGIONS_LABEL] + get_countries()
    # Pre-resolve labels once (within a valid script context) rather than
    # passing a format_func that reads st.session_state on every invocation.
    _region_labels = {c: t_country(c) for c in get_countries()}

    if "_selected_region_shadow" not in st.session_state:
        st.session_state["_selected_region_shadow"] = region_options[1]
    _region_shadow = st.session_state["_selected_region_shadow"]
    _region_index = region_options.index(_region_shadow) if _region_shadow in region_options else 1

    selected_region = st.selectbox(
        t("field_select_region"), region_options, index=_region_index,
        format_func=lambda x: ALL_REGIONS_LABEL if x == ALL_REGIONS_LABEL else _region_labels.get(x, x),
        key="selected_region",
    )
    st.session_state["_selected_region_shadow"] = selected_region

    if selected_region == ALL_REGIONS_LABEL:
        etf_options = DEFAULT_ETFS + [tk for c in get_countries() for tk in get_tickers_by_country(c) if tk not in DEFAULT_ETFS]
    elif selected_region == "United States":
        etf_options = DEFAULT_ETFS
    else:
        etf_options = get_tickers_by_country(selected_region)

    # One multiselect key per region (not a single flat "selected_etfs" key)
    # is intentional: it's what auto-validates the ETF selection when the
    # region changes (requirement 5) -- switching to Taiwan can never show
    # stale US tickers because it's a distinct widget with its own state,
    # defaulting fresh the first time that region is visited. Now that
    # `selected_region` above is stable across reruns, this key is stable
    # too. "_selected_etfs_shadow" (per region) gives the same
    # rerun-before-instantiation protection described above: it seeds
    # `default=` from the last known selection for this region instead of
    # a hardcoded default whenever this key's own state gets dropped.
    if "_selected_etfs_shadow" not in st.session_state:
        st.session_state["_selected_etfs_shadow"] = {}
    _etfs_shadow_map = st.session_state["_selected_etfs_shadow"]
    _etfs_default = [tk for tk in _etfs_shadow_map.get(selected_region, []) if tk in etf_options]
    if not _etfs_default:
        _etfs_default = etf_options[:3]

    selected_etfs = st.multiselect(
        t("field_select_etfs"),
        options=etf_options,
        default=_etfs_default,
        help=t("etf_select_etfs_help"),
        key=f"etf_analysis_multiselect_{selected_region}",
    )
    _etfs_shadow_map[selected_region] = selected_etfs
    st.session_state["_selected_etfs_shadow"] = _etfs_shadow_map

    custom_ticker = st.text_input(
        t("field_add_custom_ticker"), placeholder="e.g. ARKK", key="selected_custom_ticker",
    ).upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # Same shadow-state protection as the region/ETF filters above -- date
    # range is also part of the analysis scope ("period filter") and is
    # equally exposed to the rerun-before-instantiation issue.
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

    benchmark = st.selectbox(t("field_benchmark_etf"), options=DEFAULT_ETFS, index=2)  # SPY
    risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, 5.0, 0.25) / 100

    st.markdown("---")
    st.markdown(f"### {t('etf_sections_label')}")
    show_price = st.checkbox(t("etf_show_price"), value=True)
    show_returns = st.checkbox(t("etf_show_returns"), value=True)
    show_risk = st.checkbox(t("etf_show_risk"), value=True)
    show_correlation = st.checkbox(t("etf_show_correlation"), value=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf_sidebar"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner(t("msg_downloading_market_data")):
    all_tickers = list(set(selected_etfs + [benchmark]))
    # Map each display ticker to its actual Yahoo Finance-fetchable symbol
    # (e.g. "0050" -> "0050.TW"). Tickers not in the ETF database (including
    # any custom/free-text ticker) pass through unchanged, so this has no
    # effect on existing US-ticker behavior.
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

# ── AI ETF Summary (rule-based, no external LLM -- a new, independent
# section, separate from AI Insights / ETF Compare Score / ETF DNA /
# Investment Verdict / Compare Mode further down; none of those are read
# from or modified. Very first content section, above everything else,
# including ETF Compare Score). ──────────────────────────────────────────────
_sum_lang = get_language()
section_header(
    "AI ETF Summary",
    "根據目前資料自動產生" if _sum_lang == "zh-TW" else
    "Automatically generated from current data",
)

_SUM_TREND_META = {
    "Strong Bullish": ("🟢", "var(--success)"),
    "Bullish": ("🟢", "var(--success)"),
    "Neutral": ("🟡", "var(--warning)"),
    "Bearish": ("🔴", "var(--danger)"),
    "Strong Bearish": ("🔴", "var(--danger)"),
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
        cands.append((s_mom, "Momentum 正在增強" if lang == "zh-TW" else "Momentum is strengthening"))
    elif s_mom < -0.05:
        cands.append((-s_mom, "Momentum 正在減弱" if lang == "zh-TW" else "Momentum is weakening"))

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


_ai_summary_data = {}  # stashed per ticker below, reused by ETF Ranking so its
                       # "AI Score" always matches the card shown here exactly.

summary_cols = st.columns(len(etf_prices.columns))
for i, ticker in enumerate(etf_prices.columns):
    with summary_cols[i]:
        p = etf_prices[ticker].dropna()
        s_ret_period = p.iloc[-1] / p.iloc[0] - 1 if len(p) > 1 else 0.0
        s_ret_ann = annualized_return(p)
        s_vol = annualized_volatility(p)
        s_sharpe = sharpe_ratio(p, risk_free_rate)
        s_mdd = maximum_drawdown(p)
        s_ma_short = sma(p, 20).iloc[-1]
        s_ma_long = sma(p, 50).iloc[-1]
        s_mom_last = momentum(p, 10).iloc[-1]
        s_mom = s_mom_last if pd.notna(s_mom_last) else 0.0
        s_price = p.iloc[-1]

        s_score = 50.0
        s_score += max(-22, min(22, s_ret_ann * 140))
        s_score += max(-18, min(18, s_sharpe * 11))
        s_score += max(-12, min(12, s_mom * 200))
        s_score -= max(-8, min(22, (s_vol - 0.15) * 90))
        s_score -= max(0, min(22, abs(s_mdd) * 55))
        s_score = int(round(max(0, min(100, s_score))))

        if s_score >= 85:
            s_trend = "Strong Bullish"
        elif s_score >= 65:
            s_trend = "Bullish"
        elif s_score >= 40:
            s_trend = "Neutral"
        elif s_score >= 20:
            s_trend = "Bearish"
        else:
            s_trend = "Strong Bearish"

        if s_score >= 90:
            s_rec = "Buy"
        elif s_score >= 75:
            s_rec = "Hold"
        elif s_score >= 60:
            s_rec = "Watch"
        else:
            s_rec = "Reduce"

        s_signs = [
            1 if s_ret_ann > 0 else (-1 if s_ret_ann < 0 else 0),
            1 if s_sharpe > 0 else (-1 if s_sharpe < 0 else 0),
            1 if s_mom > 0 else (-1 if s_mom < 0 else 0),
            1 if s_price > s_ma_long else (-1 if s_price < s_ma_long else 0),
        ]
        s_overall_sign = 1 if s_score >= 50 else -1
        s_agreement = sum(1 for sgn in s_signs if sgn == s_overall_sign) / len(s_signs)
        s_confidence = round(55 + s_agreement * 40)
        if s_vol > 0.30:
            s_confidence = max(50, s_confidence - 5)

        s_insights = _ai_summary_insights(
            _sum_lang, s_ret_period, s_ret_ann, s_vol, s_sharpe, s_mdd,
            s_price, s_ma_short, s_ma_long, s_mom, s_score,
        )
        _ai_summary_data[ticker] = {
            "score": s_score, "trend": s_trend, "rec": s_rec,
            "ret_ann": s_ret_ann, "vol": s_vol, "sharpe": s_sharpe, "mom": s_mom,
        }
        s_emoji, s_color = _SUM_TREND_META[s_trend]
        s_insights_html = "".join(
            f'<div style="color:var(--text-secondary);font-size:12px;line-height:1.6;">• {ins}</div>'
            for ins in s_insights
        )
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:16px 18px;margin:6px 0;box-shadow:var(--shadow-sm);">'
            f'<div style="color:var(--text);font-weight:800;font-size:16px;margin-bottom:4px;">{ticker}</div>'
            f'<div style="color:{s_color};font-weight:700;font-size:13px;margin-bottom:10px;">{s_emoji} {s_trend}</div>'
            '<div style="color:var(--text-secondary);font-size:11px;margin-bottom:2px;">AI Score</div>'
            f'<div style="color:var(--text);font-weight:800;font-size:20px;margin-bottom:8px;">{s_score}</div>'
            '<div style="display:flex;justify-content:space-between;color:var(--text-secondary);font-size:11.5px;margin-bottom:10px;">'
            f'<span>Confidence: {s_confidence}%</span><span>Recommendation: {s_rec}</span></div>'
            '<div style="color:var(--text-muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">AI Insights</div>'
            f'{s_insights_html}'
            '</div>',
            unsafe_allow_html=True,
        )

# ── ETF Ranking (reuses the exact AI Score/Trend/Recommendation stashed by
# the AI ETF Summary loop just above, so the numbers always match; adds a
# Risk Level per ticker and a dynamically-generated "Why #1?" explanation
# comparing the top-ranked ETF's raw metrics against the rest of the
# currently selected set). Positioned directly below AI ETF Summary. ────────
section_header(
    "ETF Ranking",
    "依 AI Score 排序目前所有選取的 ETF" if _sum_lang == "zh-TW" else
    "All currently selected ETFs, ranked by AI Score",
)

_RANK_MEDALS = ["🥇", "🥈", "🥉"]

_ranked = sorted(_ai_summary_data.items(), key=lambda kv: kv[1]["score"], reverse=True)

if _ranked:
    _rank_row_html = []
    for _idx, (_r_ticker, _r_data) in enumerate(_ranked):
        _r_rank = _RANK_MEDALS[_idx] if _idx < 3 else f"#{_idx + 1}"
        _r_trend_emoji, _r_trend_color = _SUM_TREND_META[_r_data["trend"]]
        if _r_data["vol"] < 0.15:
            _r_risk = "低風險" if _sum_lang == "zh-TW" else "Low Risk"
        elif _r_data["vol"] < 0.28:
            _r_risk = "中風險" if _sum_lang == "zh-TW" else "Medium Risk"
        else:
            _r_risk = "高風險" if _sum_lang == "zh-TW" else "High Risk"
        _rank_row_html.append(
            '<tr>'
            f'<td style="padding:9px 12px;color:var(--text);font-weight:800;border-bottom:1px solid var(--border);">{_r_rank}</td>'
            f'<td style="padding:9px 12px;color:var(--text);font-weight:800;border-bottom:1px solid var(--border);">{_r_ticker}</td>'
            f'<td style="padding:9px 12px;color:var(--text);font-weight:700;border-bottom:1px solid var(--border);">{_r_data["score"]}</td>'
            f'<td style="padding:9px 12px;color:{_r_trend_color};font-weight:700;border-bottom:1px solid var(--border);">{_r_trend_emoji} {_r_data["trend"]}</td>'
            f'<td style="padding:9px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{_r_data["rec"]}</td>'
            f'<td style="padding:9px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{_r_risk}</td>'
            '</tr>'
        )

    _rank_headers = (
        ["排名", "ETF", "AI Score", "Trend", "Recommendation", "Risk Level"] if _sum_lang == "zh-TW" else
        ["Rank", "ETF", "AI Score", "Trend", "Recommendation", "Risk Level"]
    )
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
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    with _why_col:
        _top_ticker, _top_data = _ranked[0]
        _others = [d for tkr, d in _ranked if tkr != _top_ticker]
        _why_reasons = []
        if _others:
            if _top_data["sharpe"] >= max(o["sharpe"] for o in _others):
                _why_reasons.append("Sharpe Ratio")
            if _top_data["mom"] >= max(o["mom"] for o in _others):
                _why_reasons.append("Momentum")
            if _top_data["ret_ann"] >= max(o["ret_ann"] for o in _others):
                _why_reasons.append("長期報酬" if _sum_lang == "zh-TW" else "long-term return")
            if _top_data["vol"] <= min(o["vol"] for o in _others):
                _why_reasons.append("波動控制" if _sum_lang == "zh-TW" else "volatility control")

        if not _others:
            _why_text = (
                f"僅選取一檔 ETF（{_top_ticker}），暫無其他標的可供排名比較。"
                if _sum_lang == "zh-TW" else
                f"Only one ETF ({_top_ticker}) is selected, so there's nothing else to rank it against."
            )
        elif _why_reasons:
            _top3 = _why_reasons[:3]

            def _join_with_last(parts, sep, last_sep):
                if len(parts) == 1:
                    return parts[0]
                return sep.join(parts[:-1]) + last_sep + parts[-1]

            if _sum_lang == "zh-TW":
                _phrases = [(f"最佳 {r}" if r == "波動控制" else f"最高 {r}") for r in _top3]
                _joined = _join_with_last(_phrases, "、", " 與")
                _why_text = f"{_top_ticker} 在目前所有 ETF 中擁有{_joined}，因此目前排名第一。"
            else:
                _joined = _join_with_last(_top3, ", ", " and ")
                _why_text = f"{_top_ticker} has the best {_joined} among all currently selected ETFs, putting it in first place."
        else:
            _why_text = (
                f"{_top_ticker} 並未在單一指標中領先，但整體風險與報酬表現最為均衡，綜合評分因此排名第一。"
                if _sum_lang == "zh-TW" else
                f"{_top_ticker} doesn't lead on any single metric, but its overall balance of risk and return gives it the highest combined score."
            )

        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:16px 18px;height:100%;box-shadow:var(--shadow-sm);">'
            '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
            + (f"為什麼 {_top_ticker} 第一？" if _sum_lang == "zh-TW" else f"Why #1: {_top_ticker}?") + '</div>'
            f'<div style="color:var(--text-secondary);font-size:12.5px;line-height:1.7;">{_why_text}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── ETF Compare Score (rule-based, no external LLM -- independent scoring
# from the AI Insights section further down the page; combines Return,
# Sharpe, Volatility, Maximum Drawdown, and Momentum into one 0-100 score
# per ETF, ranked highest first). Placed at the very top of the page. ────────
_cmp_lang = get_language()
section_header(
    "ETF Compare Score",
    "根據 Return、Sharpe、Volatility、Maximum Drawdown 與 Momentum 綜合計算"
    if _cmp_lang == "zh-TW" else
    "Calculated from Return, Sharpe, Volatility, Maximum Drawdown, and Momentum",
)

_CMP_TREND_META = {
    "Bullish": ("🟢", "var(--success)"),
    "Neutral": ("🟡", "var(--warning)"),
    "Bearish": ("🔴", "var(--danger)"),
}

_cmp_rows = []
for ticker in etf_prices.columns:
    p = etf_prices[ticker].dropna()
    if len(p) < 10:
        continue
    c_ret = annualized_return(p)
    c_vol = annualized_volatility(p)
    c_sr = sharpe_ratio(p, risk_free_rate)
    c_mdd = maximum_drawdown(p)
    c_mom_last = momentum(p, 10).iloc[-1]
    c_mom = c_mom_last if pd.notna(c_mom_last) else 0.0

    c_score = 50.0
    c_score += max(-20, min(20, c_ret * 100))
    c_score += max(-15, min(15, c_sr * 10))
    c_score += max(-10, min(10, c_mom * 100))
    c_score -= max(-10, min(25, (c_vol - 0.15) * 100))
    c_score -= max(0, min(25, abs(c_mdd) * 60))
    c_score = int(round(max(0, min(100, c_score))))

    if c_score >= 65:
        c_trend = "Bullish"
    elif c_score <= 35:
        c_trend = "Bearish"
    else:
        c_trend = "Neutral"

    if c_score >= 75:
        c_rec = "Buy"
    elif c_score >= 45:
        c_rec = "Hold"
    else:
        c_rec = "Reduce"

    if c_vol < 0.12:
        c_risk = "Low"
    elif c_vol < 0.25:
        c_risk = "Medium"
    else:
        c_risk = "High"

    if c_ret < 0:
        c_return_label = "Poor"
    elif c_ret < 0.08:
        c_return_label = "Fair"
    elif c_ret < 0.15:
        c_return_label = "Good"
    elif c_ret < 0.25:
        c_return_label = "Very Good"
    else:
        c_return_label = "Excellent"

    _cmp_rows.append({
        "ticker": ticker, "score": c_score, "trend": c_trend,
        "risk": c_risk, "return_label": c_return_label, "rec": c_rec,
    })

_cmp_rows.sort(key=lambda r: r["score"], reverse=True)

if _cmp_rows:
    _cmp_headers = ["ETF", "Overall Score", "Trend", "Risk", "Return", "Recommendation"]
    _cmp_header_html = "".join(
        f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{h}</th>'
        for h in _cmp_headers
    )
    _cmp_row_html = []
    for r in _cmp_rows:
        emoji, color = _CMP_TREND_META[r["trend"]]
        _cmp_row_html.append(
            '<tr>'
            f'<td style="padding:10px 12px;color:var(--text);font-weight:800;border-bottom:1px solid var(--border);">{r["ticker"]}</td>'
            f'<td style="padding:10px 12px;color:var(--text);font-weight:800;font-size:15px;border-bottom:1px solid var(--border);">{r["score"]}</td>'
            f'<td style="padding:10px 12px;color:{color};font-weight:700;border-bottom:1px solid var(--border);">{emoji} {r["trend"]}</td>'
            f'<td style="padding:10px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{r["risk"]}</td>'
            f'<td style="padding:10px 12px;color:var(--text-secondary);border-bottom:1px solid var(--border);">{r["return_label"]}</td>'
            f'<td style="padding:10px 12px;color:var(--text);font-weight:700;border-bottom:1px solid var(--border);">{r["rec"]}</td>'
            '</tr>'
        )
    st.markdown(
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
        'padding:4px 8px;overflow-x:auto;box-shadow:var(--shadow-sm);margin-bottom:8px;">'
        '<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{_cmp_header_html}</tr></thead>'
        f'<tbody>{"".join(_cmp_row_html)}</tbody>'
        '</table>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Compare Mode (rule-based, no external LLM -- independent from ETF
# Compare Score above; a head-to-head of exactly two ETFs picked from the
# currently loaded selection, one metric per row with a per-row Winner,
# an Overall Winner, and a dynamically-generated AI Explanation). ────────────
_vs_lang = get_language()
section_header(
    "Compare Mode",
    "選擇兩支 ETF 進行逐項比較" if _vs_lang == "zh-TW" else
    "Pick two ETFs for a head-to-head comparison",
)

if len(etf_prices.columns) >= 2:
    _vs_options = etf_prices.columns.tolist()
    _vs_c1, _vs_c2 = st.columns(2)
    with _vs_c1:
        vs_ticker_a = st.selectbox(
            "ETF A", _vs_options, index=0, key="compare_mode_ticker_a",
        )
    with _vs_c2:
        vs_ticker_b = st.selectbox(
            "ETF B", [o for o in _vs_options if o != vs_ticker_a],
            index=0, key="compare_mode_ticker_b",
        )

    _vs_pa = etf_prices[vs_ticker_a].dropna()
    _vs_pb = etf_prices[vs_ticker_b].dropna()

    if len(_vs_pa) > 10 and len(_vs_pb) > 10:
        _vs_a = {
            "Return": annualized_return(_vs_pa),
            "Risk": value_at_risk(_vs_pa),
            "Sharpe": sharpe_ratio(_vs_pa, risk_free_rate),
            "Volatility": annualized_volatility(_vs_pa),
            "Drawdown": maximum_drawdown(_vs_pa),
            "Momentum": momentum(_vs_pa, 10).iloc[-1],
        }
        _vs_b = {
            "Return": annualized_return(_vs_pb),
            "Risk": value_at_risk(_vs_pb),
            "Sharpe": sharpe_ratio(_vs_pb, risk_free_rate),
            "Volatility": annualized_volatility(_vs_pb),
            "Drawdown": maximum_drawdown(_vs_pb),
            "Momentum": momentum(_vs_pb, 10).iloc[-1],
        }
        for _k in ("Return", "Risk", "Sharpe", "Volatility", "Drawdown", "Momentum"):
            if pd.isna(_vs_a[_k]):
                _vs_a[_k] = 0.0
            if pd.isna(_vs_b[_k]):
                _vs_b[_k] = 0.0

        # Higher-is-better rows: Return, Risk (VaR: less negative = safer),
        # Sharpe, Drawdown (less negative = smaller decline), Momentum.
        # Lower-is-better rows: Volatility (less dispersion = the "safer" pick).
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
            _winner_html = f"🏆 {_winner}" if _winner else ("—")
            _vs_row_html.append(
                '<tr>'
                f'<td style="padding:9px 12px;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.04em;border-bottom:1px solid var(--border);">{_metric}</td>'
                f'<td style="padding:9px 12px;color:var(--text);border-bottom:1px solid var(--border);">{_vs_fmt[_metric](_va)}</td>'
                f'<td style="padding:9px 12px;color:var(--text);border-bottom:1px solid var(--border);">{_vs_fmt[_metric](_vb)}</td>'
                f'<td style="padding:9px 12px;color:var(--primary);font-weight:700;border-bottom:1px solid var(--border);">{_winner_html}</td>'
                '</tr>'
            )

        _vs_header_html = (
            '<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">Metric</th>'
            f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{vs_ticker_a}</th>'
            f'<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">{vs_ticker_b}</th>'
            '<th style="text-align:left;color:var(--text-muted);font-size:11px;text-transform:uppercase;letter-spacing:0.05em;padding:8px 12px;border-bottom:1px solid var(--border);">Winner</th>'
        )
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:4px 8px;overflow-x:auto;box-shadow:var(--shadow-sm);margin-bottom:10px;">'
            '<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr>{_vs_header_html}</tr></thead>'
            f'<tbody>{"".join(_vs_row_html)}</tbody>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

        _vs_wins_a, _vs_wins_b = len(_vs_wins[vs_ticker_a]), len(_vs_wins[vs_ticker_b])
        if _vs_wins_a == _vs_wins_b:
            _vs_overall = None
        else:
            _vs_overall = vs_ticker_a if _vs_wins_a > _vs_wins_b else vs_ticker_b

        _vs_risk_metrics = {"Risk", "Volatility", "Drawdown"}
        _vs_risk_wins_a = len([m for m in _vs_wins[vs_ticker_a] if m in _vs_risk_metrics])
        _vs_risk_wins_b = len([m for m in _vs_wins[vs_ticker_b] if m in _vs_risk_metrics])
        if _vs_risk_wins_a == _vs_risk_wins_b:
            _vs_risk_winner = None
        else:
            _vs_risk_winner = vs_ticker_a if _vs_risk_wins_a > _vs_risk_wins_b else vs_ticker_b

        _overall_label = "🏆 " + _vs_overall if _vs_overall else ("平手" if _vs_lang == "zh-TW" else "Tie")
        st.markdown(
            '<div style="margin:4px 0 10px 0;">'
            '<div style="color:var(--text-muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">'
            + ("整體贏家" if _vs_lang == "zh-TW" else "Overall Winner") + '</div>'
            f'<div style="color:var(--text);font-weight:800;font-size:22px;">{_overall_label}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        _cn = {"Return": "報酬率", "Risk": "風險（VaR）", "Sharpe": "夏普比率",
               "Volatility": "波動度", "Drawdown": "最大回撤", "Momentum": "動能"}
        if _vs_overall is None:
            explanation = (
                f"{vs_ticker_a} 與 {vs_ticker_b} 整體表現相近，各項指標互有領先，難分軒輊，建議依個人風險偏好選擇。"
                if _vs_lang == "zh-TW" else
                f"{vs_ticker_a} and {vs_ticker_b} are closely matched overall, each leading on different metrics -- the choice comes down to personal risk preference."
            )
        elif _vs_risk_winner is None or _vs_risk_winner == _vs_overall:
            _reason_metrics = _vs_wins[_vs_overall]
            if _vs_lang == "zh-TW":
                _reason_str = "、".join(_cn[m] for m in _reason_metrics)
                explanation = f"{_vs_overall} 在{_reason_str}上表現優於{(vs_ticker_b if _vs_overall == vs_ticker_a else vs_ticker_a)}，因此整體勝出。"
            else:
                _reason_str = ", ".join(_reason_metrics)
                explanation = f"{_vs_overall} outperforms {(vs_ticker_b if _vs_overall == vs_ticker_a else vs_ticker_a)} on {_reason_str}, making it the overall winner."
        else:
            _growth_reasons = [m for m in _vs_wins[_vs_overall] if m not in _vs_risk_metrics] or _vs_wins[_vs_overall]
            _risk_reasons = [m for m in _vs_wins[_vs_risk_winner] if m in _vs_risk_metrics]
            if _vs_lang == "zh-TW":
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
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);'
            'padding:12px 16px;">'
            '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">'
            + ("AI 解釋" if _vs_lang == "zh-TW" else "AI Explanation") + '</div>'
            f'<div style="color:var(--text-secondary);font-size:12.5px;line-height:1.7;">{explanation}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
else:
    st.info(t("msg_select_2_correlation"))

# ── Summary KPIs ──────────────────────────────────────────────────────────────
section_header(t("etf_summary_metrics"))
cols = st.columns(len(etf_prices.columns))
for i, ticker in enumerate(etf_prices.columns):
    with cols[i]:
        p = etf_prices[ticker].dropna()
        ann_ret = annualized_return(p)
        ann_vol = annualized_volatility(p)
        sr = sharpe_ratio(p, risk_free_rate)
        mdd = maximum_drawdown(p)
        st.metric(ticker, f"{ann_ret:.2%}", t("etf_vol_caption", vol=f"{ann_vol:.2%}"))
        st.caption(t("etf_sharpe_mdd_caption", sharpe=f"{sr:.2f}", mdd=f"{mdd:.2%}"))

# ── AI Insights (rule-based, no external LLM -- reuses the Return /
# Volatility / Sharpe Ratio already computed above, plus a fresh Moving
# Average + Momentum read per ETF, in the same deterministic scoring style
# already used on the Market Intelligence page). One card per selected ETF. ──
_ai_lang = get_language()
section_header(
    "AI Insights",
    "根據 Return、Volatility、Sharpe Ratio、Moving Average 與 Momentum 自動產生"
    if _ai_lang == "zh-TW" else
    "Automatically generated from Return, Volatility, Sharpe Ratio, Moving Average, and Momentum",
)

_AI_TREND_META = {
    "Bullish": ("🟢", "var(--success)"),
    "Neutral": ("🟡", "var(--warning)"),
    "Bearish": ("🔴", "var(--danger)"),
}


def _ai_insights(lang, ann_ret, ann_vol, sr, mom, price_now, ma_short, ma_long, score):
    cands = []
    if mom > 0.05:
        cands.append((mom, "Momentum 持續增強" if lang == "zh-TW" else "Momentum continues to strengthen"))
    elif mom < -0.05:
        cands.append((-mom, "Momentum 明顯轉弱" if lang == "zh-TW" else "Momentum is weakening"))
    else:
        cands.append((0.01, "短期動能持平" if lang == "zh-TW" else "Short-term momentum is flat"))

    if ann_ret > 0.15:
        cands.append((ann_ret, "報酬率高於市場平均" if lang == "zh-TW" else "Return is above the market average"))
    elif ann_ret < 0:
        cands.append((-ann_ret, "報酬率低於預期" if lang == "zh-TW" else "Return is below expectations"))

    if sr > 1.2:
        cands.append((sr / 3, "風險調整後報酬表現優異" if lang == "zh-TW" else "Risk-adjusted return is excellent"))
    elif sr < 0.3:
        cands.append((0.3 - sr, "風險調整後報酬偏弱" if lang == "zh-TW" else "Risk-adjusted return is weak"))

    if ann_vol > 0.25:
        if score >= 50:
            cands.append((ann_vol, "波動增加但仍維持健康趨勢" if lang == "zh-TW" else "Volatility has increased but the trend remains healthy"))
        else:
            cands.append((ann_vol, "波動偏高，風險上升" if lang == "zh-TW" else "Volatility is elevated, raising risk"))
    elif ann_vol < 0.12:
        cands.append((0.12 - ann_vol, "波動度偏低，走勢穩定" if lang == "zh-TW" else "Volatility is low, price action is stable"))

    if price_now > ma_short > ma_long:
        cands.append((price_now / ma_long - 1, "站上短期與長期均線，趨勢偏多" if lang == "zh-TW" else "Price is above both short- and long-term moving averages"))
    elif price_now < ma_short < ma_long:
        cands.append((ma_long / price_now - 1, "跌破短期與長期均線，趨勢偏空" if lang == "zh-TW" else "Price is below both short- and long-term moving averages"))
    else:
        cands.append((0.01, "均線呈現盤整格局" if lang == "zh-TW" else "Moving averages show a consolidating pattern"))

    cands.sort(key=lambda c: c[0], reverse=True)
    return [c[1] for c in cands[:3]]


def _ai_interpretation(key_findings, investment_insight: str, risk_reminder: str) -> None:
    """Render a 3-part 'AI Interpretation' block below a chart: Key
    Findings, Investment Insight, Risk Reminder. Reused by every chart on
    this page -- each call site computes its own content from that chart's
    actual data and is written to answer "so what", not just restate
    numbers; nothing here is fixed text. `key_findings` is a string or a
    list of 1-2 strings."""
    if isinstance(key_findings, str):
        key_findings = [key_findings]
    _kf_html = "".join(
        f'<div style="color:var(--text);font-size:12.5px;line-height:1.6;">• {kf}</div>'
        for kf in key_findings
    )
    st.markdown(
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);'
        'padding:12px 16px;margin:6px 0 14px 0;">'
        '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">AI Interpretation</div>'
        '<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">Key Findings</div>'
        f'{_kf_html}'
        '<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:8px 0 3px 0;">Investment Insight</div>'
        f'<div style="color:var(--success);font-size:12.5px;line-height:1.6;">{investment_insight}</div>'
        '<div style="color:var(--text-muted);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin:8px 0 3px 0;">Risk Reminder</div>'
        f'<div style="color:var(--warning);font-size:12.5px;line-height:1.6;">{risk_reminder}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


ai_cols = st.columns(len(etf_prices.columns))
for i, ticker in enumerate(etf_prices.columns):
    with ai_cols[i]:
        p = etf_prices[ticker].dropna()
        ann_ret = annualized_return(p)
        ann_vol = annualized_volatility(p)
        sr = sharpe_ratio(p, risk_free_rate)
        ma_short = sma(p, 20).iloc[-1]
        ma_long = sma(p, 50).iloc[-1]
        mom_last = momentum(p, 10).iloc[-1]
        mom = mom_last if pd.notna(mom_last) else 0.0
        price_now = p.iloc[-1]

        score = 50.0
        score += max(-20, min(20, ann_ret * 100))
        score += max(-15, min(15, sr * 10))
        score += max(-15, min(15, mom * 100))
        if price_now > ma_short > ma_long:
            score += 10
        elif price_now < ma_short < ma_long:
            score -= 10
        score = int(round(max(0, min(100, score))))

        if score >= 65:
            trend = "Bullish"
        elif score <= 35:
            trend = "Bearish"
        else:
            trend = "Neutral"

        if score >= 75:
            recommendation = "Buy"
        elif score >= 45:
            recommendation = "Hold"
        else:
            recommendation = "Reduce"

        signs = [
            1 if ann_ret > 0 else (-1 if ann_ret < 0 else 0),
            1 if sr > 0 else (-1 if sr < 0 else 0),
            1 if mom > 0 else (-1 if mom < 0 else 0),
            1 if price_now > ma_long else (-1 if price_now < ma_long else 0),
        ]
        overall_sign = 1 if score >= 50 else -1
        agreement = sum(1 for s in signs if s == overall_sign) / len(signs)
        confidence = round(55 + agreement * 40)
        if ann_vol > 0.30:
            confidence = max(50, confidence - 5)

        insights = _ai_insights(_ai_lang, ann_ret, ann_vol, sr, mom, price_now, ma_short, ma_long, score)
        emoji, color = _AI_TREND_META[trend]
        insights_html = "".join(
            f'<div style="color:var(--text-secondary);font-size:12px;line-height:1.6;">• {ins}</div>'
            for ins in insights
        )
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:16px 18px;margin:6px 0;box-shadow:var(--shadow-sm);">'
            f'<div style="color:var(--text);font-weight:800;font-size:16px;margin-bottom:4px;">{ticker}</div>'
            f'<div style="color:{color};font-weight:700;font-size:13px;margin-bottom:10px;">{emoji} {trend}</div>'
            '<div style="color:var(--text-secondary);font-size:11px;margin-bottom:2px;">AI Score</div>'
            f'<div style="color:var(--text);font-weight:800;font-size:20px;margin-bottom:8px;">{score}</div>'
            '<div style="display:flex;justify-content:space-between;color:var(--text-secondary);font-size:11.5px;margin-bottom:10px;">'
            f'<span>Confidence: {confidence}%</span><span>Recommendation: {recommendation}</span></div>'
            '<div style="color:var(--text-muted);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">AI Insights</div>'
            f'{insights_html}'
            '</div>',
            unsafe_allow_html=True,
        )

# ── ETF DNA (rule-based, no external LLM -- independent from AI Insights /
# Compare Score above; five 0-100 dimensions per ETF, each freshly computed
# from the currently loaded price data, rendered as horizontal progress
# bars in the same dark-card style used elsewhere on this page). ────────────
_dna_lang = get_language()
section_header(
    "ETF DNA",
    "根據目前資料自動計算的五個維度" if _dna_lang == "zh-TW" else
    "Five dimensions automatically calculated from current data",
)

_DNA_DIMENSIONS = [
    ("Growth", "var(--success)"),
    ("Risk", "var(--danger)"),
    ("Momentum", "var(--primary)"),
    ("Diversification", "var(--purple)"),
    ("Liquidity", "var(--cyan)"),
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
            f'{bars_html}'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Price Analysis ────────────────────────────────────────────────────────────
if show_price:
    section_header(t("etf_price_analysis_title"), t("etf_price_analysis_subtitle"))
    with chart_card(t("etf_price_charts_card"), tag=f"{len(etf_prices.columns)} ETFs"):
        tab1, tab2, tab3, tab4 = st.tabs([
            t("etf_tab_historical"), t("etf_tab_normalized"), t("etf_tab_cumulative"), t("etf_tab_drawdown")
        ])

        with tab1:
            fig = price_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_historical")
            _chg = {}
            for c in etf_prices.columns:
                s = etf_prices[c].dropna()
                if len(s) > 1:
                    _chg[c] = s.iloc[-1] / s.iloc[0] - 1
            if _chg:
                _best, _worst = max(_chg, key=_chg.get), min(_chg, key=_chg.get)
                _avg_chg = sum(_chg.values()) / len(_chg)
                _bullish = _avg_chg > 0
                if _ai_lang == "zh-TW":
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

        with tab2:
            fig = normalized_price_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_normalized")
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
                    if _ai_lang == "zh-TW":
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
                    kf = ["僅選取單一 ETF，無相對比較對象"] if _ai_lang == "zh-TW" else ["Only one ETF is selected, so there's no relative comparison"]
                    insight = "建議加入至少一檔其他 ETF 以評估相對強弱" if _ai_lang == "zh-TW" else "Consider adding at least one more ETF to gauge relative strength"
                    risk = "單一標的無法分散非系統性風險" if _ai_lang == "zh-TW" else "A single holding carries undiversified idiosyncratic risk"
                _ai_interpretation(kf, insight, risk)

        with tab3:
            fig = cumulative_return_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_cumulative")
            _cum = {}
            for c in etf_prices.columns:
                s = etf_prices[c].dropna()
                if len(s) > 1:
                    _cum[c] = (s.iloc[-1] / s.iloc[0] - 1) * 100
            if _cum:
                _best, _worst = max(_cum, key=_cum.get), min(_cum, key=_cum.get)
                _pos = sum(1 for v in _cum.values() if v > 0)
                _majority_positive = _pos >= len(_cum) / 2
                if _ai_lang == "zh-TW":
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

        with tab4:
            fig = drawdown_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_drawdown")
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

                # Recovery speed for the deepest-drawdown ticker: trading
                # days from its trough back to the prior peak (or "not yet
                # recovered" if the price hasn't gotten back there).
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

                if _ai_lang == "zh-TW":
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

    # Technical indicators for single ETF
    if len(etf_prices.columns) == 1:
        ticker = etf_prices.columns[0]
        p = etf_prices[ticker]
        import plotly.graph_objects as go
        from src.charts import apply_dark_theme
        from src.theme import COLORS as _C
        with chart_card(t("etf_technical_indicators_title", ticker=ticker), t("etf_technical_indicators_sub")):
            bb = bollinger_bands(p)
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=p.index, y=p, name=ticker, line=dict(color=_C["primary"], width=2)))
            fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Upper"], name=t("chart_bb_upper"),
                                         line=dict(color=_C["danger"], dash="dash", width=1)))
            fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Middle"], name=t("chart_bb_middle"),
                                         line=dict(color=_C["text_muted"], dash="dot", width=1)))
            fig_bb.add_trace(go.Scatter(x=bb.index, y=bb["Lower"], name=t("chart_bb_lower"),
                                         line=dict(color=_C["success"], dash="dash", width=1),
                                         fill="tonexty", fillcolor="rgba(52,211,153,0.05)"))
            fig_bb.update_layout(title=t("chart_bollinger_bands"), xaxis_title=t("chart_date"), yaxis_title=t("chart_price"))
            st.plotly_chart(apply_dark_theme(fig_bb), use_container_width=True, key="etf_bollinger_bands")
            _bb_last = bb.iloc[-1]
            _last_price = p.iloc[-1]
            _band_width = (_bb_last["Upper"] - _bb_last["Lower"]) / _bb_last["Middle"] * 100 if _bb_last["Middle"] else 0
            _pos_pct = (_last_price - _bb_last["Lower"]) / (_bb_last["Upper"] - _bb_last["Lower"]) * 100 if _bb_last["Upper"] != _bb_last["Lower"] else 50
            _expanding = _band_width > 8
            if _ai_lang == "zh-TW":
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

# ── Return Analysis ───────────────────────────────────────────────────────────
if show_returns:
    section_header(t("etf_return_analysis_title"), t("etf_return_analysis_subtitle"))
    with chart_card(t("etf_return_metrics_card")):
        tab1, tab2, tab3, tab4 = st.tabs([
            t("etf_tab_distribution"), t("etf_tab_monthly_heatmap"), t("etf_tab_rolling_metrics"), t("etf_tab_annual_performance")
        ])

        with tab1:
            fig = return_distribution_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_return_distribution")
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
                    if _ai_lang == "zh-TW":
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

        with tab2:
            for ticker in etf_prices.columns:
                p = etf_prices[ticker].dropna()
                if len(p) > 30:
                    monthly_ret = monthly_returns_table(p)
                    if not monthly_ret.empty:
                        st.markdown(f"**{t('etf_monthly_returns_for', ticker=ticker)}**")
                        fig = monthly_heatmap(monthly_ret)
                        st.plotly_chart(fig, use_container_width=True, key=f"etf_monthly_heatmap_{ticker}")
                        _month_avg = monthly_ret.mean(axis=0, skipna=True)
                        _best_month, _worst_month = _month_avg.idxmax(), _month_avg.idxmin()
                        _pos_rate = (monthly_ret > 0).sum(axis=0) / monthly_ret.notna().sum(axis=0)
                        _seasonal = _pos_rate[(_pos_rate >= 0.75) | (_pos_rate <= 0.25)]
                        _has_seasonality = len(_seasonal) > 0 and monthly_ret.shape[0] >= 2
                        if _ai_lang == "zh-TW":
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

        with tab3:
            ticker_select = st.selectbox(t("etf_select_rolling_etf"), etf_prices.columns.tolist(), key="rolling_ticker")
            window = st.slider(t("etf_rolling_window_days"), 21, 252, 63)
            p = etf_prices[ticker_select].dropna()
            if len(p) > window:
                fig = rolling_metrics_chart(p, window)
                st.plotly_chart(fig, use_container_width=True, key="etf_rolling_metrics")
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
                    if _ai_lang == "zh-TW":
                        kf = [
                            "近期滾動報酬呈上升趨勢，動能轉強" if _strengthening else "近期滾動報酬呈下降趨勢，動能轉弱",
                            "目前滾動報酬已高於長期平均，屬於突破訊號" if _breakout else "目前滾動報酬仍低於長期平均，尚未突破",
                        ]
                        insight = "趨勢與動能同步轉強，可能是相對有利的進場時機" if (_strengthening and _breakout) else "訊號尚未一致轉強，建議等待更明確的突破確認再加碼"
                        risk = (f"10 日 Momentum {'持續增加' if _mom_up else '轉為收斂或下滑'}（{_mom_recent:+.2%}），但動能類指標容易反轉，且滾動指標本身落後於即時價格" if _mom_recent is not None else "Momentum 資料不足，且滾動指標本身落後於即時價格")
                    else:
                        kf = [
                            "Recent rolling return is trending up, momentum is strengthening" if _strengthening else "Recent rolling return is trending down, momentum is weakening",
                            "Rolling return is currently above the long-term average, a breakout signal" if _breakout else "Rolling return is still below the long-term average, no breakout yet",
                        ]
                        insight = "Trend and momentum are strengthening together, potentially a favorable entry window" if (_strengthening and _breakout) else "Signals aren't fully aligned yet -- worth waiting for a clearer breakout before adding"
                        risk = (f"10-day Momentum is {'increasing' if _mom_up else 'flattening or declining'} ({_mom_recent:+.2%}), but momentum indicators can reverse quickly and rolling metrics lag real-time price" if _mom_recent is not None else "Not enough data for Momentum, and rolling metrics lag real-time price")
                    _ai_interpretation(kf, insight, risk)

        with tab4:
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
                                   yaxis_title=t("chart_annual_return_pct"), barmode="group")
                st.plotly_chart(apply_dark_theme(fig), use_container_width=True, key="etf_annual_performance")
                _yearly_avg = annual_returns.mean(axis=1)
                _best_year, _worst_year = _yearly_avg.idxmax().year, _yearly_avg.idxmin().year
                if len(_yearly_avg) >= 2:
                    _half = len(_yearly_avg) // 2
                    _first_half, _second_half = _yearly_avg.iloc[:_half].mean(), _yearly_avg.iloc[_half:].mean()
                    _trend_up = _second_half > _first_half
                else:
                    _trend_up = None
                _year_spread = _yearly_avg.max() - _yearly_avg.min()
                if _ai_lang == "zh-TW":
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

# ── Risk Analysis ─────────────────────────────────────────────────────────────
if show_risk:
    section_header(t("etf_risk_analysis_title"), t("etf_risk_analysis_subtitle"))

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

    with chart_card(t("etf_risk_vs_return_card")):
        fig = risk_return_scatter(etf_prices)
        st.plotly_chart(fig, use_container_width=True, key="etf_risk_return_scatter")
        _rr = {}
        for c in etf_prices.columns:
            s = etf_prices[c].dropna()
            if len(s) > 5:
                _rr[c] = (annualized_return(s), annualized_volatility(s))
        if _rr:
            _best_ratio = max(_rr, key=lambda k: _rr[k][0] / _rr[k][1] if _rr[k][1] else 0)
            _highest_risk = max(_rr, key=lambda k: _rr[k][1])
            if len(_rr) > 1:
                if _ai_lang == "zh-TW":
                    kf = [f"{_best_ratio} 的風險報酬比相對最佳", f"{_highest_risk} 波動度最高，風險相對集中"]
                    insight = f"就目前選取的標的而言，{_best_ratio} 的風險調整後表現較值得優先考慮" if _best_ratio != _highest_risk else f"{_best_ratio} 兼具最高波動與最佳比率，屬於高風險高報酬型標的，配置比重應審慎拿捏"
                    risk = f"{_highest_risk} 的波動度最高，短線震盪可能較大，配置比重不宜過度集中"
                else:
                    kf = [f"{_best_ratio} offers the best return-to-risk ratio", f"{_highest_risk} carries the highest volatility, concentrating risk"]
                    insight = f"Among the current selection, {_best_ratio} looks worth prioritizing on a risk-adjusted basis" if _best_ratio != _highest_risk else f"{_best_ratio} combines the highest volatility with the best ratio -- a high-risk, high-return profile that needs careful position sizing"
                    risk = f"{_highest_risk} carries the highest volatility -- short-term swings could be larger, so avoid over-concentrating in it"
            else:
                kf = ["僅單一標的，無法比較風險報酬分布"] if _ai_lang == "zh-TW" else ["Only one ETF is selected, so there's no risk-return spread to compare"]
                insight = "建議加入至少一檔其他 ETF 以評估相對風險報酬位置" if _ai_lang == "zh-TW" else "Consider adding at least one more ETF to gauge its relative risk-return position"
                risk = "單一標的的風險完全取決於該檔本身，缺乏分散效果" if _ai_lang == "zh-TW" else "A single holding's risk is fully tied to that one ETF, with no diversification benefit"
            _ai_interpretation(kf, insight, risk)

# ── Correlation Analysis ──────────────────────────────────────────────────────
if show_correlation:
    section_header(t("etf_correlation_analysis_title"), t("etf_correlation_analysis_subtitle"))

    if len(etf_prices.columns) >= 2:
        with chart_card(t("etf_correlation_heatmap_card")):
            corr = correlation_matrix(etf_prices)
            fig = correlation_heatmap(corr)
            st.plotly_chart(fig, use_container_width=True, key="etf_correlation_heatmap")
            _cols_c = corr.columns.tolist()
            _pairs = [
                (_cols_c[a], _cols_c[b], corr.iloc[a, b])
                for a in range(len(_cols_c)) for b in range(a + 1, len(_cols_c))
            ]
            if _pairs:
                _highest = max(_pairs, key=lambda x: x[2])
                _lowest = min(_pairs, key=lambda x: x[2])
                _avg_corr = sum(pp[2] for pp in _pairs) / len(_pairs)

                # Which single ticker adds the LEAST diversification: the one
                # with the highest average correlation to every other ticker.
                _avg_corr_by_ticker = {
                    tk: corr[tk].drop(tk).mean() for tk in _cols_c if tk in corr.columns
                }
                _least_diversifying = max(_avg_corr_by_ticker, key=_avg_corr_by_ticker.get)
                _limited_benefit = _avg_corr_by_ticker[_least_diversifying] > 0.7

                if _highest[2] > 0.7:
                    _highest_zh, _highest_en = "高度相關", "highly correlated"
                elif _highest[2] > 0.4:
                    _highest_zh, _highest_en = "中度相關", "moderately correlated"
                else:
                    _highest_zh, _highest_en = "相關性偏低（即使是相關性最高的一組）", "only mildly correlated (even as the highest pair in this set)"

                if _ai_lang == "zh-TW":
                    kf = [f"{_highest[0]} 與 {_highest[1]} {_highest_zh}（{_highest[2]:.2f}）"]
                    kf.append(f"{_lowest[0]} 與 {_lowest[1]} 相關性最低（{_lowest[2]:.2f}），分散效果較佳")
                    insight = (
                        f"加入 {_least_diversifying} 對分散效果有限，可考慮以相關性較低的標的替代以提升分散度"
                        if _limited_benefit else
                        f"目前組合平均相關係數約為 {_avg_corr:.2f}，整體仍具備一定的分散化效益"
                    )
                    risk = "持股間相關性偏高時，市場下跌會同步拖累多檔標的，實際分散效果可能低於預期" if _limited_benefit else "相關係數會隨市場狀態變動，壓力時期（如系統性風險事件）相關性經常會上升，分散效果可能不如平時"
                else:
                    kf = [f"{_highest[0]} and {_highest[1]} are {_highest_en} ({_highest[2]:.2f})"]
                    kf.append(f"{_lowest[0]} and {_lowest[1]} are the least correlated pair ({_lowest[2]:.2f}), offering better diversification")
                    insight = (
                        f"Adding {_least_diversifying} offers limited diversification benefit -- a lower-correlation ETF could add more diversification value"
                        if _limited_benefit else
                        f"The portfolio's average correlation is about {_avg_corr:.2f}, still providing meaningful diversification benefit"
                    )
                    risk = "When holdings are highly correlated, a market downturn tends to drag several of them down together -- actual diversification may be less than it appears" if _limited_benefit else "Correlations shift with market conditions -- they often rise during systemic stress events, so diversification benefits can shrink exactly when they're needed most"
                _ai_interpretation(kf, insight, risk)

        with chart_card(t("etf_covariance_matrix_card"), t("etf_covariance_matrix_sub")):
            cov = covariance_matrix(etf_prices)
            st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)
    else:
        st.info(t("msg_select_2_correlation"))

# ── Downloads ─────────────────────────────────────────────────────────────────
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
    if metrics_data:
        metrics_export = pd.DataFrame(metrics_data).set_index("Ticker")
        csv3 = dataframe_to_csv(metrics_export)
        st.download_button(t("btn_download_metrics"), csv3, "etf_metrics.csv", "text/csv")

# ── Investment Verdict (rule-based, no external LLM -- a single page-level
# synthesis that AGGREGATES the AI Score/Trend/raw metrics already stashed
# in _ai_summary_data by the AI ETF Summary section above, per "整合整頁分析
# 結果", rather than an isolated per-ETF calculation. Replaces the previous
# per-ETF-card version of this section (explicit user choice). Bottom-most
# content section, before the shared disclaimer/footer. ──────────────────────
_verdict_lang = get_language()
section_header(
    "Investment Verdict",
    "整合整頁分析結果的最終投資結論" if _verdict_lang == "zh-TW" else
    "A single conclusion synthesizing the whole page's analysis",
)

_verdict_tickers = [t for t in etf_prices.columns if t in _ai_summary_data]

if _verdict_tickers:
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
    _v_trend_display = _V_TREND_ZH[_v_trend] if _verdict_lang == "zh-TW" else _v_trend
    _v_trend_color = _V_TREND_COLOR[_v_trend]

    # ── AI Final Recommendation: one market-trend clause, then up to three
    # per-ETF role clauses (growth leader / core holding / diversifier),
    # each assigned to a distinct ticker so no ETF is cited twice. ──────────
    _V_MARKET_CLAUSE = {
        "Strong Bullish": ("目前市場整體呈現強勁多頭格局", "The market is currently showing a strong bullish trend overall"),
        "Bullish": ("目前市場仍維持多頭", "The market is currently maintaining a bullish trend"),
        "Neutral": ("目前市場呈現盤整格局", "The market is currently consolidating"),
        "Bearish": ("目前市場呈現空頭格局", "The market is currently in a bearish trend"),
        "Strong Bearish": ("目前市場呈現強勁空頭格局", "The market is currently in a strong bearish trend"),
    }
    _clauses = [_V_MARKET_CLAUSE[_v_trend][0 if _verdict_lang == "zh-TW" else 1]]

    if len(_verdict_tickers) == 1:
        _only = _verdict_tickers[0]
        if _verdict_lang == "zh-TW":
            _clauses.append(f"{_only} 目前是唯一納入分析的標的，整體評分為 {_ai_summary_data[_only]['score']} 分")
        else:
            _clauses.append(f"{_only} is the only ETF currently included in the analysis, with an overall score of {_ai_summary_data[_only]['score']}")
    else:
        _assigned = set()
        _growth_leader = max(_verdict_tickers, key=lambda t: _ai_summary_data[t]["ret_ann"])
        _assigned.add(_growth_leader)
        _clauses.append(f"{_growth_leader} 擁有最佳長期成長能力" if _verdict_lang == "zh-TW" else f"{_growth_leader} offers the best long-term growth potential")

        _remaining = [t for t in _verdict_tickers if t not in _assigned]
        if len(_verdict_tickers) >= 3 and _remaining:
            _core_holding = max(_remaining, key=lambda t: _ai_summary_data[t]["sharpe"])
            _assigned.add(_core_holding)
            _clauses.append(f"{_core_holding} 適合作為核心持股" if _verdict_lang == "zh-TW" else f"{_core_holding} is well suited as a core holding")

            _remaining2 = [t for t in _verdict_tickers if t not in _assigned]
            if _remaining2:
                _corr = etf_prices[_verdict_tickers].pct_change().dropna().corr()
                _diversifier = min(_remaining2, key=lambda t: _corr[t].drop(t).mean() if t in _corr.columns else 0)
                _clauses.append(f"{_diversifier} 適合分散投資" if _verdict_lang == "zh-TW" else f"{_diversifier} is well suited for diversification")
        elif _remaining:
            _other = _remaining[0]
            _clauses.append(f"{_other} 可作為分散配置的補充" if _verdict_lang == "zh-TW" else f"{_other} can serve as a diversifying complement")

    v_recommendation = ("，<br>".join(_clauses) + "。") if _verdict_lang == "zh-TW" else (",<br>".join(_clauses) + ".")

    _v_col1, _v_col2 = st.columns([2, 3])
    with _v_col1:
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:18px 20px;box-shadow:var(--shadow-sm);height:100%;">'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;">'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("整體評分" if _verdict_lang == "zh-TW" else "Overall Rating") + '</div>'
            f'<div style="color:var(--text);font-weight:800;font-size:22px;">{_v_overall_score}</div></div>'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("趨勢" if _verdict_lang == "zh-TW" else "Trend") + '</div>'
            f'<div style="color:{_v_trend_color};font-weight:700;font-size:15px;">{_v_trend_display}</div></div>'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("風險" if _verdict_lang == "zh-TW" else "Risk") + '</div>'
            f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{_v_risk}</div></div>'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("預期報酬" if _verdict_lang == "zh-TW" else "Expected Return") + '</div>'
            f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{_v_exp_return}</div></div>'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("投資期間" if _verdict_lang == "zh-TW" else "Investment Horizon") + '</div>'
            f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{_v_horizon}</div></div>'
            '<div><div style="color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;">'
            + ("適合對象" if _verdict_lang == "zh-TW" else "Suitable For") + '</div>'
            f'<div style="color:var(--text-secondary);font-weight:700;font-size:15px;">{_v_suitable}</div></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with _v_col2:
        st.markdown(
            '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);'
            'padding:18px 20px;box-shadow:var(--shadow-sm);height:100%;">'
            '<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">'
            + ("AI 最終建議" if _verdict_lang == "zh-TW" else "AI Final Recommendation") + '</div>'
            f'<div style="color:var(--text-secondary);font-size:13px;line-height:1.8;">{v_recommendation}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

disclaimer_box()
render_footer()
