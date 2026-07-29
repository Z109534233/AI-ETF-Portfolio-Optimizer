"""
Page 8: AI Market Intelligence Center
A market intelligence dashboard (not a news site): today's index snapshot,
breaking headlines, an AI/rule-based market summary, ETFs today's news may
affect, aggregate headline sentiment, a placeholder economic calendar, and
(when a portfolio has been saved) a portfolio-impact analysis.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import load_all_portfolios, init_database
from src.news import fetch_market_news
from src.market_intelligence import (
    fetch_market_indices, fetch_fear_greed_index, get_affected_etfs,
    calculate_market_sentiment, get_economic_calendar,
    generate_market_summary, analyze_portfolio_impact,
    calculate_affected_markets,
)
from src.ai_advisor import get_openai_client
from src.charts import sentiment_donut_chart, allocation_donut_chart
from src.theme import COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, news_card, status_card, star_rating_html,
    market_impact_card, empty_state, error_state,
)
from src.i18n import t

st.set_page_config(
    page_title="Market Intelligence | AI ETF Portfolio Optimizer",
    page_icon="📰",
    layout="wide",
)

load_css()
init_database()

page_header(t("mi_title"), t("mi_subtitle"))

with st.sidebar:
    render_sidebar_nav()
    render_sidebar_footer()

IMPACT_VARIANT = {"Positive": "green", "Negative": "red", "Neutral": "neutral"}

# ── Fetch all data up front; every helper below is defensive and returns a
# safe empty/placeholder value on failure, but this outer guard ensures the
# page can never crash even on an unexpected error. ─────────────────────────
try:
    news_items = fetch_market_news(limit=10)
    indices = fetch_market_indices()
    fear_greed = fetch_fear_greed_index()
    affected_etfs = get_affected_etfs(news_items)
    sentiment = calculate_market_sentiment(news_items)
    calendar_events = get_economic_calendar()
    affected_markets = calculate_affected_markets(news_items)
    data_load_failed = False
except Exception:
    news_items, indices, fear_greed = [], {}, {"available": False, "label": t("mi_fear_greed")}
    affected_etfs, sentiment, calendar_events = [], calculate_market_sentiment([]), []
    affected_markets = []
    data_load_failed = True

if data_load_failed:
    error_state(t("mi_no_news_available"), t("mi_summary_no_news"))

# ── Section 1: Today's Market Overview ───────────────────────────────────────
section_header(t("mi_section_overview_title"))

col1, col2, col3, col4, col5, col6 = st.columns(6)
index_cols = {"sp500": col1, "nasdaq": col2, "dow": col3, "russell": col4, "vix": col5}

for key, col in index_cols.items():
    info = indices.get(key, {"label": t(f"mi_{key}"), "available": False})
    with col:
        if info.get("available"):
            change_pct = info["change_pct"]
            delta = f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"
            color = COLORS["warning"] if key == "vix" else COLORS["primary"]
            st.markdown(
                metric_card_html(info["label"], f"{info['price']:,.2f}", delta, color=color),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                metric_card_html(info["label"], t("mi_index_unavailable"), color=COLORS["text_muted"]),
                unsafe_allow_html=True,
            )

with col6:
    st.markdown(
        metric_card_html(fear_greed["label"], t("mi_placeholder_value"), t("mi_placeholder_note"),
                          color=COLORS["text_muted"]),
        unsafe_allow_html=True,
    )

# ── Section 1b: Affected Markets ─────────────────────────────────────────────
section_header(t("mi_section_affected_markets_title"), t("mi_section_affected_markets_subtitle"))

if affected_markets:
    market_cols = st.columns(len(affected_markets))
    for col, market in zip(market_cols, affected_markets):
        with col:
            st.markdown(
                market_impact_card(
                    market["market"], t("mi_impact_level_caption"),
                    star_rating_html(market["stars"]), market["impact_label"],
                    t("mi_affected_by_caption"), market["affected_by"],
                ),
                unsafe_allow_html=True,
            )
else:
    empty_state(t("mi_no_news_available"), t("mi_section_affected_markets_subtitle"), icon="layers")

# ── Section 2: Breaking Market News ──────────────────────────────────────────
section_header(t("mi_section_news_title"), t("mi_section_news_subtitle"))

if not news_items:
    empty_state(t("mi_no_news_available"), t("mi_section_news_subtitle"), icon="newspaper")
else:
    cards_html = "".join(
        news_card(
            title=item["title"],
            time_str=item["published"].strftime("%Y-%m-%d %H:%M") if item["published"] else "—",
            source=item["publisher"],
            impact_label=t(f"mi_impact_{item['impact'].lower()}_label"),
            impact_variant=IMPACT_VARIANT.get(item["impact"], "neutral"),
            url=item["link"] or None,
        )
        for item in news_items
    )
    st.markdown(cards_html, unsafe_allow_html=True)

# ── Section 3: AI Market Summary ─────────────────────────────────────────────
section_header(t("mi_section_summary_title"))

summary_text = generate_market_summary(news_items, sentiment, affected_etfs)
ai_client = get_openai_client()
with chart_card(t("mi_section_summary_title"), tag=t("ai_tag_generated") if ai_client else t("ai_tag_rule_based")):
    st.markdown(summary_text)

# ── Section 4: Global ETFs (Affected ETFs across US / Taiwan / UK) ───────────
section_header(t("mi_section_global_etfs_title"), t("mi_section_global_etfs_subtitle"))

if affected_etfs:
    etf_cols = st.columns(len(affected_etfs))
    for col, etf in zip(etf_cols, affected_etfs):
        with col:
            st.markdown(
                status_card(etf["ticker"], etf["country"], etf["impact_label"],
                            IMPACT_VARIANT.get(etf["impact"], "neutral"),
                            star_rating_html(etf["stars"])),
                unsafe_allow_html=True,
            )
else:
    empty_state(t("mi_no_news_available"), t("mi_section_global_etfs_subtitle"), icon="layers")

# ── Section 5: Market Sentiment ──────────────────────────────────────────────
section_header(t("mi_section_sentiment_title"))

with chart_card(t("mi_section_sentiment_title")):
    if sentiment["available"]:
        col_chart, col_bars = st.columns([1, 1])
        with col_chart:
            fig = sentiment_donut_chart(sentiment["bullish_pct"], sentiment["neutral_pct"], sentiment["bearish_pct"])
            st.plotly_chart(fig, use_container_width=True, key="mi_sentiment_donut")
        with col_bars:
            st.caption(f"{t('mi_sentiment_bullish_pct_label')}: {sentiment['bullish_pct']:.1f}%")
            st.progress(min(int(sentiment["bullish_pct"]), 100))
            st.caption(f"{t('mi_sentiment_neutral_pct_label')}: {sentiment['neutral_pct']:.1f}%")
            st.progress(min(int(sentiment["neutral_pct"]), 100))
            st.caption(f"{t('mi_sentiment_bearish_pct_label')}: {sentiment['bearish_pct']:.1f}%")
            st.progress(min(int(sentiment["bearish_pct"]), 100))
    else:
        st.info(t("mi_sentiment_no_data"))

# ── Section 6: Economic Calendar ─────────────────────────────────────────────
section_header(t("mi_section_calendar_title"), t("mi_section_calendar_subtitle"))

with chart_card(t("mi_section_calendar_title")):
    calendar_df = pd.DataFrame([
        {
            t("mi_cal_col_event"): e["event"],
            t("mi_cal_col_when"): e["when"],
            t("mi_cal_col_importance"): e["importance"],
        }
        for e in calendar_events
    ])
    st.dataframe(calendar_df, use_container_width=True, hide_index=True)

# ── Section 7: Portfolio Impact ──────────────────────────────────────────────
section_header(t("mi_section_portfolio_impact_title"))

portfolios = load_all_portfolios()
if not portfolios:
    empty_state(t("hist_no_portfolios_title"), t("mi_portfolio_no_data"), icon="layers")
else:
    latest = portfolios[0]
    impact_text = analyze_portfolio_impact(latest["holdings"], affected_etfs)
    col_text, col_chart = st.columns([2, 1])
    with col_text:
        with chart_card(latest["name"], t("mi_portfolio_using", name=latest["name"])):
            st.markdown(impact_text)
    with col_chart:
        if latest["holdings"]:
            with chart_card(t("hist_allocation_breakdown_card")):
                fig = allocation_donut_chart(latest["holdings"], "")
                st.plotly_chart(fig, use_container_width=True, key="mi_portfolio_allocation_donut")

# ── Section 8: Educational Disclaimer ────────────────────────────────────────
disclaimer_box(t("mi_disclaimer"))
render_footer()
