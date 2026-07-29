"""
Page 6: AI Advisor
Educational portfolio explanation using OpenAI or rule-based fallback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, maximum_drawdown,
    sortino_ratio, calmar_ratio, value_at_risk
)
from src.ai_advisor import generate_ai_analysis, DISCLAIMER, get_openai_client
from src.charts import allocation_donut_chart
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer
from src.i18n import (
    t, t_investment_objective, t_risk_level, t_country,
    INVESTMENT_OBJECTIVE_KEYS, RISK_LEVEL_KEYS
)

st.set_page_config(
    page_title="AI Advisor | AI ETF Portfolio Optimizer",
    page_icon="🧠",
    layout="wide"
)

load_css()

page_header(t("ai_title"), t("ai_subtitle"))

# Check OpenAI availability
client = get_openai_client()
if client is None:
    st.info(t("ai_mode_info"))
else:
    st.success(t("ai_mode_success"))

st.warning(t("ai_disclaimer_banner"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('ai_sidebar_config')}")

    # ── Region Selector (Global ETF Support) ─────────────────────────────
    # "United States" preserves the exact original ETF list (DEFAULT_ETFS)
    # so existing behavior is unchanged unless the user explicitly picks a
    # different region.
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
        default=etf_options[:4],
        key=f"ai_multiselect_{selected_region}",
    )

    custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    st.markdown(f"#### {t('ai_weights_label')}")
    weights_input = {}
    if selected_etfs:
        equal_w = 1.0 / len(selected_etfs)
        for ticker in selected_etfs:
            w = st.slider(t("ai_weight_pct", ticker=ticker), 0.0, 100.0, equal_w * 100, 1.0, key=f"ai_w_{ticker}")
            weights_input[ticker] = w / 100.0
        total_w = sum(weights_input.values())
        if abs(total_w - 1.0) > 0.01 and total_w > 0:
            st.warning(t("ai_weights_normalised_warning", total=f"{total_w:.1%}"))
            weights_input = {k: v / total_w for k, v in weights_input.items()}

    investment_amount = st.number_input(t("field_investment_amount_usd"), 100.0, 10_000_000.0, 10000.0, 500.0)

    st.markdown("---")
    st.markdown(f"### {t('ai_investor_profile')}")
    # NOTE: format_func displays the translated label while the selectbox
    # still returns the raw English value (required by ai_advisor.py's
    # rule-based generator, which has no dependency on the exact string —
    # we immediately translate to display text below for use everywhere else).
    # Labels are pre-resolved once (within a valid script context) rather
    # than passing a format_func that reads st.session_state on every call.
    _objective_labels = {k: t_investment_objective(k) for k in INVESTMENT_OBJECTIVE_KEYS}
    _risk_labels = {k: t_risk_level(k) for k in RISK_LEVEL_KEYS}
    investment_objective = st.selectbox(
        t("ai_investment_objective_label"),
        list(INVESTMENT_OBJECTIVE_KEYS.keys()),
        format_func=lambda x: _objective_labels.get(x, x),
    )
    risk_level = st.selectbox(
        t("ai_risk_tolerance_label"),
        list(RISK_LEVEL_KEYS.keys()),
        format_func=lambda x: _risk_labels.get(x, x),
    )
    investment_objective = t_investment_objective(investment_objective)
    risk_level = t_risk_level(risk_level)
    investment_horizon = st.slider(t("ai_investment_horizon_years"), 1, 40, 10)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("ai_data_start_date"), value=default_start)
    end_date = st.date_input(t("ai_data_end_date"), value=default_end)

    analyse_btn = st.button(t("btn_generate_analysis"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf"))
    st.stop()

# ── Data Loading & Metrics ────────────────────────────────────────────────────
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if analyse_btn or st.session_state.ai_result is None:
    with st.spinner(t("msg_downloading_market_data")):
        yahoo_tickers = [to_yahoo_symbol(tk) for tk in selected_etfs]
        raw_prices = download_etf_data(yahoo_tickers, str(start_date), str(end_date))
        prices_df = rename_yahoo_columns(clean_price_data(raw_prices)) if not raw_prices.empty else pd.DataFrame()

        # Compute portfolio metrics
        metrics = {}
        if not prices_df.empty:
            etf_prices = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]
            if not etf_prices.empty:
                weights_arr = np.array([weights_input.get(tk, 0) for tk in etf_prices.columns])
                if weights_arr.sum() > 0:
                    weights_arr = weights_arr / weights_arr.sum()
                returns_df = etf_prices.pct_change().dropna()
                port_returns = (returns_df * weights_arr).sum(axis=1)
                port_prices = (1 + port_returns).cumprod() * 100

                metrics = {
                    t("metric_annualized_return"): f"{annualized_return(port_prices):.2%}",
                    t("metric_annualized_volatility"): f"{annualized_volatility(port_prices):.2%}",
                    t("metric_sharpe_ratio"): f"{sharpe_ratio(port_prices, 0.05):.2f}",
                    t("metric_sortino_ratio"): f"{sortino_ratio(port_prices, 0.05):.2f}",
                    t("metric_maximum_drawdown"): f"{maximum_drawdown(port_prices):.2%}",
                    t("metric_calmar_ratio"): f"{calmar_ratio(port_prices):.2f}",
                    t("metric_var_95"): f"{value_at_risk(port_prices, 0.95):.2%}",
                    t("metric_number_of_holdings"): str(len(selected_etfs)),
                    t("metric_investment_horizon"): f"{investment_horizon}",
                    t("metric_risk_tolerance"): risk_level,
                    t("metric_investment_objective"): investment_objective,
                }

    with st.spinner(t("msg_generating_report")):
        analysis_text = generate_ai_analysis(
            portfolio_weights=weights_input,
            metrics=metrics,
            investment_objective=investment_objective,
            risk_level=risk_level,
            investment_horizon=investment_horizon
        )

    st.session_state.ai_result = {
        "analysis": analysis_text,
        "metrics": metrics,
        "weights": weights_input,
    }

result = st.session_state.ai_result
if result is None:
    st.info(t("ai_configure_and_generate"))
    st.stop()

# ── Display Results ───────────────────────────────────────────────────────────
section_header(t("ai_analysis_results_title"))
col_left, col_right = st.columns([2, 1])

_excluded_metric_keys = {
    t("metric_number_of_holdings"), t("metric_investment_horizon"),
    t("metric_risk_tolerance"), t("metric_investment_objective"),
}

with col_left:
    with chart_card(t("ai_portfolio_analysis_card"), tag=t("ai_tag_generated") if client else t("ai_tag_rule_based")):
        st.markdown(result["analysis"])

with col_right:
    with chart_card(t("ai_portfolio_overview_card")):
        fig_donut = allocation_donut_chart(result["weights"], "")
        st.plotly_chart(fig_donut, use_container_width=True, key="ai_advisor_allocation_donut")

        if result["metrics"]:
            st.markdown(f"**{t('ai_key_metrics')}**")
            for k, v in result["metrics"].items():
                if k not in _excluded_metric_keys:
                    st.metric(k, v)

# ── Regenerate ────────────────────────────────────────────────────────────────
section_header(t("ai_actions_title"))
col1, col2 = st.columns(2)
with col1:
    if st.button(t("btn_regenerate_analysis")):
        st.session_state.ai_result = None
        st.rerun()

with col2:
    if result["analysis"]:
        analysis_bytes = result["analysis"].encode("utf-8")
        st.download_button(
            t("btn_download_analysis_txt"),
            analysis_bytes,
            "portfolio_analysis.txt",
            "text/plain"
        )

disclaimer_box()
render_footer()
