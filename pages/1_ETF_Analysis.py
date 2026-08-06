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

        with tab2:
            fig = normalized_price_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_normalized")

        with tab3:
            fig = cumulative_return_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_cumulative")

        with tab4:
            fig = drawdown_chart(etf_prices)
            st.plotly_chart(fig, use_container_width=True, key="etf_price_drawdown")

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

        with tab2:
            for ticker in etf_prices.columns:
                p = etf_prices[ticker].dropna()
                if len(p) > 30:
                    monthly_ret = monthly_returns_table(p)
                    if not monthly_ret.empty:
                        st.markdown(f"**{t('etf_monthly_returns_for', ticker=ticker)}**")
                        fig = monthly_heatmap(monthly_ret)
                        st.plotly_chart(fig, use_container_width=True, key=f"etf_monthly_heatmap_{ticker}")

        with tab3:
            ticker_select = st.selectbox(t("etf_select_rolling_etf"), etf_prices.columns.tolist(), key="rolling_ticker")
            window = st.slider(t("etf_rolling_window_days"), 21, 252, 63)
            p = etf_prices[ticker_select].dropna()
            if len(p) > window:
                fig = rolling_metrics_chart(p, window)
                st.plotly_chart(fig, use_container_width=True, key="etf_rolling_metrics")

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

# ── Correlation Analysis ──────────────────────────────────────────────────────
if show_correlation:
    section_header(t("etf_correlation_analysis_title"), t("etf_correlation_analysis_subtitle"))

    if len(etf_prices.columns) >= 2:
        with chart_card(t("etf_correlation_heatmap_card")):
            corr = correlation_matrix(etf_prices)
            fig = correlation_heatmap(corr)
            st.plotly_chart(fig, use_container_width=True, key="etf_correlation_heatmap")

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
