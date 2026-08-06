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
    ALL_REGIONS_LABEL = t("field_all_regions")
    region_options = [ALL_REGIONS_LABEL] + get_countries()
    # Pre-resolve labels once (within a valid script context) rather than
    # passing a format_func that reads st.session_state on every invocation.
    _region_labels = {c: t_country(c) for c in get_countries()}
    selected_region = st.selectbox(
        t("field_select_region"), region_options, index=1,
        format_func=lambda x: ALL_REGIONS_LABEL if x == ALL_REGIONS_LABEL else _region_labels.get(x, x),
    )

    if selected_region == ALL_REGIONS_LABEL:
        etf_options = DEFAULT_ETFS + [tk for c in get_countries() for tk in get_tickers_by_country(c) if tk not in DEFAULT_ETFS]
    elif selected_region == "United States":
        etf_options = DEFAULT_ETFS
    else:
        etf_options = get_tickers_by_country(selected_region)

    selected_etfs = st.multiselect(
        t("field_select_etfs"),
        options=etf_options,
        default=etf_options[:3],
        help=t("etf_select_etfs_help"),
        key=f"etf_analysis_multiselect_{selected_region}",
    )

    custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("field_start_date"), value=default_start)
    end_date = st.date_input(t("field_end_date"), value=default_end)

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


def _ai_interpretation(lines: list) -> None:
    """Render an 'AI Interpretation' bullet block below a chart. Reused by
    every chart on this page -- each call site computes its own `lines`
    from that chart's actual data, nothing here is fixed text."""
    label = "AI 解讀" if _ai_lang == "zh-TW" else "AI Interpretation"
    items_html = "".join(
        f'<div style="color:var(--text-secondary);font-size:12px;line-height:1.7;">• {ln}</div>'
        for ln in lines
    )
    st.markdown(
        '<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);'
        'padding:12px 16px;margin:6px 0 14px 0;">'
        f'<div style="color:var(--primary);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">{label}</div>'
        f'{items_html}'
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
                lines = []
                if _ai_lang == "zh-TW":
                    lines.append(f"{_best} 期間漲幅最大（{_chg[_best]:+.1%}）")
                    if _worst != _best:
                        lines.append(f"{_worst} 期間表現最弱（{_chg[_worst]:+.1%}）")
                    lines.append(f"整體平均價格變化為 {sum(_chg.values()) / len(_chg):+.1%}")
                else:
                    lines.append(f"{_best} gained the most over the period ({_chg[_best]:+.1%})")
                    if _worst != _best:
                        lines.append(f"{_worst} was the weakest performer ({_chg[_worst]:+.1%})")
                    lines.append(f"Average price change across selected ETFs is {sum(_chg.values()) / len(_chg):+.1%}")
                _ai_interpretation(lines)

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
                lines = []
                if _ai_lang == "zh-TW":
                    lines.append(f"以相同基準比較，{_best} 相對表現最佳（指數 {_norm_end[_best]:.1f}）")
                    if _worst != _best:
                        lines.append(f"{_worst} 相對表現最弱（指數 {_norm_end[_worst]:.1f}）")
                        lines.append(f"領先與落後標的差距約 {_norm_end[_best] - _norm_end[_worst]:.1f} 個指數點")
                    else:
                        lines.append("僅選取單一 ETF，無相對比較對象")
                else:
                    lines.append(f"On a normalized basis, {_best} is the relative leader (index {_norm_end[_best]:.1f})")
                    if _worst != _best:
                        lines.append(f"{_worst} is the relative laggard (index {_norm_end[_worst]:.1f})")
                        lines.append(f"Gap between leader and laggard is about {_norm_end[_best] - _norm_end[_worst]:.1f} index points")
                    else:
                        lines.append("Only one ETF selected, no relative comparison available")
                _ai_interpretation(lines)

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
                lines = []
                if _ai_lang == "zh-TW":
                    lines.append(f"{_best} 累積報酬最高（{_cum[_best]:+.1f}%）")
                    if _worst != _best:
                        lines.append(f"{_worst} 累積報酬最低（{_cum[_worst]:+.1f}%）")
                    lines.append(f"{_pos}/{len(_cum)} 檔 ETF 期間累積報酬為正")
                else:
                    lines.append(f"{_best} has the highest cumulative return ({_cum[_best]:+.1f}%)")
                    if _worst != _best:
                        lines.append(f"{_worst} has the lowest cumulative return ({_cum[_worst]:+.1f}%)")
                    lines.append(f"{_pos}/{len(_cum)} selected ETFs have a positive cumulative return")
                _ai_interpretation(lines)

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
                lines = []
                if _ai_lang == "zh-TW":
                    lines.append(f"{_deepest} 歷史最大回撤最深（{_dd[_deepest]:.1f}%）")
                    lines.append(f"目前仍處於回撤中：{'、'.join(_still_down)}" if _still_down else "所有標的目前皆已從最大回撤中恢復")
                    lines.append(f"平均最大回撤約為 {sum(_dd.values()) / len(_dd):.1f}%")
                else:
                    lines.append(f"{_deepest} has the deepest historical drawdown ({_dd[_deepest]:.1f}%)")
                    lines.append(f"Currently still in drawdown: {', '.join(_still_down)}" if _still_down else "All selected ETFs have recovered from their max drawdown")
                    lines.append(f"Average max drawdown is about {sum(_dd.values()) / len(_dd):.1f}%")
                _ai_interpretation(lines)

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
            lines = []
            if _ai_lang == "zh-TW":
                if _pos_pct >= 85:
                    lines.append("價格接近上軌，短線有過熱疑慮")
                elif _pos_pct <= 15:
                    lines.append("價格接近下軌，短線可能超賣")
                else:
                    lines.append("價格位於通道中段，未見極端訊號")
                lines.append(f"目前價格位於通道約 {_pos_pct:.0f}% 位置")
                lines.append(f"通道寬度約 {_band_width:.1f}%，{'波動擴張中' if _band_width > 8 else '波動相對收斂'}")
            else:
                if _pos_pct >= 85:
                    lines.append("Price is near the upper band, short-term overbought risk")
                elif _pos_pct <= 15:
                    lines.append("Price is near the lower band, possibly oversold")
                else:
                    lines.append("Price sits mid-channel, no extreme signal")
                lines.append(f"Current price is at about {_pos_pct:.0f}% of the band width")
                lines.append(f"Band width is about {_band_width:.1f}%, {'volatility is expanding' if _band_width > 8 else 'volatility is relatively contained'}")
            _ai_interpretation(lines)

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
                    lines = []
                    if _ai_lang == "zh-TW":
                        lines.append(f"平均單日報酬率約為 {_mean_r:.3%}")
                        lines.append(f"左尾風險（5% 分位數）約為 {_p5:.2%}，代表極端下跌情境")
                        lines.append(f"報酬波動度（標準差）約為 {_std_r:.2%}")
                        lines.append("分布呈現右偏（正報酬機會較大）" if _skew > 0.1 else ("分布呈現左偏（極端虧損風險較高）" if _skew < -0.1 else "分布大致對稱，無明顯偏態"))
                    else:
                        lines.append(f"Average daily return is about {_mean_r:.3%}")
                        lines.append(f"Left-tail risk (5th percentile) is about {_p5:.2%}, representing extreme downside scenarios")
                        lines.append(f"Return volatility (std dev) is about {_std_r:.2%}")
                        lines.append("Distribution is right-skewed (more upside potential)" if _skew > 0.1 else ("Distribution is left-skewed (higher extreme-loss risk)" if _skew < -0.1 else "Distribution is roughly symmetric, no strong skew"))
                    _ai_interpretation(lines)

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
                        lines = []
                        if _ai_lang == "zh-TW":
                            lines.append(f"{_best_month} 平均表現最佳（平均 {_month_avg[_best_month]:+.2%}）")
                            lines.append(f"{_worst_month} 平均表現最差（平均 {_month_avg[_worst_month]:+.2%}）")
                            if len(_seasonal) > 0 and monthly_ret.shape[0] >= 2:
                                lines.append(f"{'、'.join(_seasonal.index)} 呈現較明顯的季節性傾向")
                            else:
                                lines.append("未觀察到明顯的季節性規律")
                        else:
                            lines.append(f"{_best_month} performs best on average ({_month_avg[_best_month]:+.2%})")
                            lines.append(f"{_worst_month} performs worst on average ({_month_avg[_worst_month]:+.2%})")
                            if len(_seasonal) > 0 and monthly_ret.shape[0] >= 2:
                                lines.append(f"{', '.join(_seasonal.index)} show a notable seasonal tendency")
                            else:
                                lines.append("No clear seasonal pattern observed")
                        _ai_interpretation(lines)

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
                    lines = []
                    if _ai_lang == "zh-TW":
                        lines.append("近期滾動報酬呈上升趨勢，動能轉強" if _recent > _prior else "近期滾動報酬呈下降趨勢，動能轉弱")
                        lines.append("目前滾動報酬已高於長期平均，屬於突破訊號" if _recent > _full_mean else "目前滾動報酬仍低於長期平均，尚未突破")
                        lines.append(f"10 日 Momentum {'持續增加' if _mom_recent is not None and _mom_recent > 0 else '轉為收斂或下滑'}（{_mom_recent:+.2%}）" if _mom_recent is not None else "Momentum 資料不足")
                    else:
                        lines.append("Recent rolling return is trending up, momentum is strengthening" if _recent > _prior else "Recent rolling return is trending down, momentum is weakening")
                        lines.append("Rolling return is currently above the long-term average, a breakout signal" if _recent > _full_mean else "Rolling return is still below the long-term average, no breakout yet")
                        lines.append(f"10-day Momentum is {'increasing' if _mom_recent is not None and _mom_recent > 0 else 'flattening or declining'} ({_mom_recent:+.2%})" if _mom_recent is not None else "Not enough data for Momentum")
                    _ai_interpretation(lines)

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
                lines = []
                if len(_yearly_avg) >= 2:
                    _half = len(_yearly_avg) // 2
                    _first_half, _second_half = _yearly_avg.iloc[:_half].mean(), _yearly_avg.iloc[_half:].mean()
                    _trend_up = _second_half > _first_half
                else:
                    _trend_up = None
                if _ai_lang == "zh-TW":
                    lines.append(f"{_best_year} 年平均表現最佳（{_yearly_avg.max():+.1f}%）")
                    lines.append(f"{_worst_year} 年平均表現最差（{_yearly_avg.min():+.1f}%）")
                    lines.append(("長期趨勢偏向轉強" if _trend_up else "長期趨勢偏向轉弱") if _trend_up is not None else "資料年數過短，尚無法判斷長期趨勢")
                else:
                    lines.append(f"{_best_year} was the best year on average ({_yearly_avg.max():+.1f}%)")
                    lines.append(f"{_worst_year} was the worst year on average ({_yearly_avg.min():+.1f}%)")
                    lines.append(("Long-term trend is strengthening" if _trend_up else "Long-term trend is weakening") if _trend_up is not None else "Not enough years of data to judge the long-term trend")
                _ai_interpretation(lines)

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
            lines = []
            if _ai_lang == "zh-TW":
                lines.append(f"{_best_ratio} 的風險報酬比相對最佳")
                lines.append(f"{_highest_risk} 波動度最高，風險相對集中")
                lines.append("報酬與風險大致呈正向關係" if len(_rr) > 1 else "僅單一標的，無法比較風險報酬分布")
            else:
                lines.append(f"{_best_ratio} offers the best return-to-risk ratio")
                lines.append(f"{_highest_risk} carries the highest volatility, concentrating risk")
                lines.append("Higher return broadly tracks with higher risk here" if len(_rr) > 1 else "Only one ETF selected, no risk-return spread to compare")
            _ai_interpretation(lines)

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
                lines = []
                if _ai_lang == "zh-TW":
                    lines.append(f"{_highest[0]} 與 {_highest[1]} 相關性最高（{_highest[2]:.2f}）")
                    lines.append(f"{_lowest[0]} 與 {_lowest[1]} 相關性最低（{_lowest[2]:.2f}），分散效果較佳")
                    lines.append(f"平均相關係數約為 {_avg_corr:.2f}，{'分散化效益有限' if _avg_corr > 0.7 else '具備一定的分散化效益'}")
                else:
                    lines.append(f"{_highest[0]} and {_highest[1]} are the most correlated pair ({_highest[2]:.2f})")
                    lines.append(f"{_lowest[0]} and {_lowest[1]} are the least correlated pair ({_lowest[2]:.2f}), offering better diversification")
                    lines.append(f"Average correlation is about {_avg_corr:.2f}, {'limiting diversification benefit' if _avg_corr > 0.7 else 'providing meaningful diversification benefit'}")
                _ai_interpretation(lines)

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
