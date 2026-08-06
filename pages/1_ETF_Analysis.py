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

disclaimer_box()
render_footer()
