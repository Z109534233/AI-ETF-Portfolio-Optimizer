"""
Page 6: AI Portfolio Analyst (Issue #20 section 7 -- renamed from "AI Advisor" /
generic "AI Investment Analysis" naming)
Synthesis layer (M6 -- issue #18 Stage 6): combines the canonical current
portfolio, risk, Investment Simulator, Machine Learning, and Market
Intelligence outputs already computed elsewhere in the app into one
educational narrative, via src.ai_advisor.build_advisor_context() /
generate_advisor_narrative(). Falls back to a self-contained custom
hypothetical portfolio when no canonical portfolio exists yet.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data
from src.data_cleaner import clean_price_data
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.financial_metrics import annualized_return, annualized_volatility, sharpe_ratio, maximum_drawdown
from src.ai_advisor import build_advisor_context, generate_advisor_narrative
from src.openai_service import is_configured as openai_is_configured
from src.news import fetch_market_news
from src.charts import allocation_donut_chart
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer,
    region_selector, region_etf_options, region_etf_multiselect,
    render_current_portfolio_handoff,
)
from src.i18n import (
    t, t_investment_objective, t_risk_level, t_country, get_language,
    INVESTMENT_OBJECTIVE_KEYS, RISK_LEVEL_KEYS
)

st.set_page_config(
    page_title="AI Portfolio Analyst | AI ETF Portfolio Optimizer",
    page_icon="🧠",
    layout="wide"
)

load_css()

page_header(t("ai_title"), t("ai_subtitle"))

# ── Current Portfolio handoff ────────────────────────────────────────────────
# Same canonical st.session_state["current_portfolio"] object used by
# Investment Simulator / Risk Analytics / Portfolio History (Round 2B-4).
render_current_portfolio_handoff(
    t("handoff_empty_state_title"), t("handoff_empty_state_body_ai"),
)
current_portfolio = st.session_state.get("current_portfolio")

if not openai_is_configured():
    st.info(t("ai_mode_info"))
else:
    st.success(t("ai_mode_success"))

st.warning(t("ai_disclaimer_banner"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('ai_sidebar_config')}")

    if current_portfolio:
        use_custom = st.checkbox(t("ai_use_custom_portfolio"), value=False, key="ai_use_custom")
    else:
        st.info(t("ai_no_current_portfolio_hint"))
        use_custom = True

    selected_etfs, weights_input, start_date, end_date = [], {}, None, None
    if use_custom:
        selected_region, ALL_REGIONS_LABEL = region_selector()
        etf_options = region_etf_options(selected_region, ALL_REGIONS_LABEL)
        selected_etfs = region_etf_multiselect(
            selected_region, etf_options, t("field_select_etfs"), n_default=4,
        )
        custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
        if custom_ticker and custom_ticker not in selected_etfs:
            selected_etfs.append(custom_ticker)

        st.markdown(f"#### {t('ai_weights_label')}")
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
        default_start, default_end = get_date_range_defaults()
        start_date = st.date_input(t("ai_data_start_date"), value=default_start)
        end_date = st.date_input(t("ai_data_end_date"), value=default_end)
    else:
        selected_etfs = current_portfolio["tickers"]
        weights_input = current_portfolio["weights"]
        investment_amount = current_portfolio.get("investment_amount")
        # historical_start_date/end_date can be None (e.g. a portfolio
        # reloaded from Portfolio History -- the saved-portfolio schema has
        # no date columns, see pages/7_Portfolio_History.py's
        # _set_as_current_portfolio()); fall back to the same default
        # window Investment Simulator/Risk Analytics use rather than
        # passing "None" through to the price downloader.
        default_start, default_end = get_date_range_defaults()
        start_date = current_portfolio.get("historical_start_date") or default_start
        end_date = current_portfolio.get("historical_end_date") or default_end

    st.markdown("---")
    st.markdown(f"### {t('ai_investor_profile')}")
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

    analyse_btn = st.button(t("btn_generate_analysis"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf"))
    st.stop()

st.caption(t("ai_source_mode_custom") if use_custom else t("ai_source_mode_current"))

# ── Data Loading & Context Assembly ────────────────────────────────────────────
# Stale-result guard (same fix as Issue #20 section 6A applied to Machine
# Learning): the generated narrative must be bound to every input that
# changes what it actually explains, so changing the ETF/weight selection,
# investment amount, custom-vs-current toggle, investor-profile inputs, or
# the active UI language without clicking "Generate Analysis" again shows a
# warning instead of silently keeping the old narrative on screen next to
# new sidebar values -- language matters because the cached narrative text
# itself is only ever generated in ONE language (whichever was active at
# generation time), so a language switch alone must also trigger this guard
# (Issue #20 release-gate review: previously omitted, so switching zh-TW/EN
# after generating a narrative left it on screen in the old language).
_ai_fingerprint = (
    tuple(sorted(selected_etfs)), tuple(sorted(weights_input.items())),
    round(investment_amount, 2) if investment_amount is not None else None,
    use_custom, str(start_date), str(end_date),
    investment_objective, risk_level, investment_horizon, get_language(),
)

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "ai_fingerprint" not in st.session_state:
    st.session_state.ai_fingerprint = None

if analyse_btn or st.session_state.ai_result is None:
    with st.spinner(t("msg_downloading_market_data")):
        yahoo_tickers = [to_yahoo_symbol(tk) for tk in selected_etfs]
        raw_prices = download_etf_data(yahoo_tickers, str(start_date), str(end_date))
        prices_df = rename_yahoo_columns(clean_price_data(raw_prices)) if not raw_prices.empty else pd.DataFrame()

        portfolio_prices = None
        if not prices_df.empty:
            etf_prices = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]
            if not etf_prices.empty:
                w_arr = np.array([weights_input.get(tk, 0) for tk in etf_prices.columns])
                if w_arr.sum() > 0:
                    w_arr = w_arr / w_arr.sum()
                returns_df = etf_prices.pct_change().dropna()
                port_returns = (returns_df * w_arr).sum(axis=1)
                portfolio_prices = (1 + port_returns).cumprod() * 100

        if use_custom:
            if portfolio_prices is not None and len(portfolio_prices) > 1:
                portfolio_for_context = {
                    "strategy": "Custom",
                    "market": st.session_state.get("selected_region"),
                    "tickers": selected_etfs,
                    "weights": weights_input,
                    "investment_amount": investment_amount,
                    "expected_return": annualized_return(portfolio_prices),
                    "volatility": annualized_volatility(portfolio_prices),
                    "sharpe_ratio": sharpe_ratio(portfolio_prices, 0.05),
                    "max_drawdown": maximum_drawdown(portfolio_prices),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                portfolio_for_context = None
            portfolio_source = "custom"
        else:
            portfolio_for_context = current_portfolio
            portfolio_source = "current"

        news_items = []
        try:
            news_items = fetch_market_news()
        except Exception:
            news_items = []

        context = build_advisor_context(
            portfolio=portfolio_for_context,
            portfolio_source=portfolio_source,
            portfolio_prices=portfolio_prices,
            sim_result=st.session_state.get("sim_result"),
            sim_params=st.session_state.get("sim_params"),
            hist_result=st.session_state.get("hist_result"),
            hist_params=st.session_state.get("hist_params"),
            ml_result=st.session_state.get("ml_result"),
            ml_ticker=st.session_state.get("ml_ticker"),
            news_items=news_items,
        )

    with st.spinner(t("msg_generating_report")):
        narrative = generate_advisor_narrative(
            context,
            investment_objective=investment_objective,
            risk_level=risk_level,
            investment_horizon=investment_horizon,
            session_state=st.session_state,
        )

    st.session_state.ai_result = {
        "analysis": narrative["text"], "source": narrative["source"], "context": context,
    }
    st.session_state.ai_fingerprint = _ai_fingerprint

result = st.session_state.ai_result
if result is None or not result["context"]["portfolio"]["available"]:
    st.info(t("ai_configure_and_generate"))
    st.stop()

if st.session_state.ai_fingerprint is not None and st.session_state.ai_fingerprint != _ai_fingerprint:
    st.warning(t("ai_inputs_changed_regenerate"))
    st.stop()

context = result["context"]
port_ctx = context["portfolio"]

# ── Display Results ───────────────────────────────────────────────────────────
section_header(t("ai_analysis_results_title"))
col_left, col_right = st.columns([2, 1])

with col_left:
    with chart_card(t("ai_portfolio_analysis_card"), tag=t("ai_tag_generated") if result.get("source") == "ai" else t("ai_tag_rule_based")):
        st.markdown(f"**{t('ai_narrative_title')}**")
        st.markdown(result["analysis"])

with col_right:
    with chart_card(t("ai_portfolio_overview_card")):
        fig_donut = allocation_donut_chart(port_ctx["weights"], "")
        st.plotly_chart(fig_donut, use_container_width=True, key="ai_advisor_allocation_donut")

# ── Deterministic Data (computed, not AI-generated) ─────────────────────────
section_header(t("ai_deterministic_data_title"))
st.caption(f"{t('ai_as_of_label')}: {context['as_of']}")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(t("metric_annualized_return"), f"{port_ctx['expected_return']:.2%}" if port_ctx["expected_return"] is not None else "N/A")
with kpi2:
    st.metric(t("metric_annualized_volatility"), f"{port_ctx['volatility']:.2%}" if port_ctx["volatility"] is not None else "N/A")
with kpi3:
    st.metric(t("metric_sharpe_ratio"), f"{port_ctx['sharpe_ratio']:.2f}" if port_ctx["sharpe_ratio"] is not None else "N/A")
with kpi4:
    st.metric(t("metric_maximum_drawdown"), f"{port_ctx['max_drawdown']:.2%}" if port_ctx["max_drawdown"] is not None else "N/A")

risk_ctx = context["risk"]
if risk_ctx.get("available"):
    c = risk_ctx["concentration"]
    kpi5, kpi6 = st.columns(2)
    with kpi5:
        st.metric(t("opt_diag_effective_holdings"), f"{c['effective_holdings']:.1f}")
    with kpi6:
        vc = risk_ctx["var_cvar"]
        st.metric(t("metric_var_95"), f"{vc['var']:.2%}" if vc.get("available") else "N/A")

with st.expander(t("ai_section_risk"), expanded=False):
    if risk_ctx.get("available"):
        c = risk_ctx["concentration"]
        st.markdown(t(
            "ai_risk_concentration_line", ticker=c["largest_ticker"], weight=f"{c['largest_weight']:.2%}",
            effective_holdings=f"{c['effective_holdings']:.1f}",
        ))
        vc = risk_ctx["var_cvar"]
        if vc.get("available"):
            st.markdown(t(
                "ai_risk_var_line", confidence=f"{vc['confidence']:.0%}", holding_period=vc["holding_period_days"],
                var=f"{vc['var']:.2%}", cvar=f"{vc['cvar']:.2%}", n_obs=vc["n_observations"],
                window_start=vc["window_start"], window_end=vc["window_end"],
            ))
        else:
            st.markdown(t("ai_risk_var_unavailable", reason=vc.get("reason", "")))
    else:
        st.markdown(t("ai_section_unavailable", reason=risk_ctx.get("reason", "")))

with st.expander(t("ai_section_simulator"), expanded=False):
    sim_ctx = context["simulator"]
    fp = sim_ctx["future_projection"]
    if fp.get("available"):
        s = fp["summary"]
        st.markdown(t(
            "ai_sim_future_line", years=fp["years"], median=f"${s.get('median_final', 0):,.0f}",
            prob=f"{s.get('probability_profit', 0):.1%}", source=fp.get("assumption_source", ""),
        ))
    else:
        st.markdown(t("ai_sim_future_unavailable", reason=fp.get("reason", "")))
    hs = sim_ctx["historical_simulation"]
    if hs.get("available"):
        s = hs["summary"]
        st.markdown(t(
            "ai_sim_historical_line", final=f"${s.get('final_value', 0):,.0f}",
            gain=f"${s.get('gain', 0):,.0f}", mwr=f"{s.get('annualized_mwr', 0):.2%}",
        ))
    else:
        st.markdown(t("ai_sim_historical_unavailable", reason=hs.get("reason", "")))

with st.expander(t("ai_section_ml"), expanded=False):
    ml_ctx = context["ml"]
    if ml_ctx.get("available"):
        beats = t("ai_ml_beats") if ml_ctx["beats_baseline"] else t("ai_ml_below")
        st.markdown(t(
            "ai_ml_line", ticker=ml_ctx["ticker"], model=ml_ctx["model_name"],
            accuracy=f"{ml_ctx['accuracy']:.1%}" if ml_ctx["accuracy"] is not None else "N/A",
            baseline=f"{ml_ctx['baseline_accuracy']:.1%}" if ml_ctx["baseline_accuracy"] is not None else "N/A",
            beats=beats, test_start=ml_ctx["test_start"], test_end=ml_ctx["test_end"],
        ))
    else:
        st.markdown(t("ai_section_unavailable", reason=ml_ctx.get("reason", "")))

with st.expander(t("ai_section_news"), expanded=False):
    news_ctx = context["news"]
    if news_ctx.get("available"):
        st.markdown(t("ai_news_line", count=news_ctx["headline_count"], relevant=news_ctx["relevant_holdings_count"]))
        if news_ctx.get("portfolio_impact_text"):
            st.markdown(f"- {news_ctx['portfolio_impact_text']}")
    else:
        st.markdown(t("ai_section_unavailable", reason=news_ctx.get("reason", "")))

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
