"""
Page 8: AI Global Market Intelligence Center
A global market intelligence dashboard (not a news site, not an ETF quote
page): AI analyzes today's breaking events, scores overall market impact,
identifies affected markets and ETFs (US / Taiwan / UK UCITS), explains why
each event matters, summarizes market mood, and -- when a portfolio has
been saved -- analyzes how today's events relate to it. Decision support
only: no price prediction, no buy/sell recommendation.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import load_all_portfolios, init_database
from src.news_service import fetch_breaking_events
from src.impact_engine import calculate_market_impact, calculate_affected_markets
from src.global_etf import get_global_etf_universe, calculate_etf_impacts, calculate_portfolio_exposure
from src.market_intelligence import (
    fetch_market_indices, fetch_fear_greed_index, calculate_market_sentiment,
    determine_market_mood, get_economic_calendar, generate_market_summary,
    generate_why_it_matters, generate_portfolio_analysis,
)
from src.ai_advisor import get_openai_client
from src.charts import allocation_donut_chart
from src.theme import COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, news_card, status_card, star_rating_html,
    impact_score_card, empty_state, error_state,
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

DIRECTION_VARIANT = {"Positive": "green", "Negative": "red", "Neutral": "neutral"}
CATEGORY_VARIANT = {
    "monetary_policy": "blue", "trade": "amber", "earnings": "green",
    "macro_data": "neutral", "commodities": "amber", "corporate": "neutral",
    "general": "neutral",
}
STAR_VARIANT = {5: "amber", 4: "amber", 3: "blue", 2: "neutral", 1: "neutral", 0: "neutral"}

# ── Fetch all data up front; every helper below is defensive and returns a
# safe empty/placeholder value on failure, but this outer guard ensures the
# page can never crash even on an unexpected error. ─────────────────────────
try:
    events = fetch_breaking_events(limit=10)
    indices = fetch_market_indices()
    fear_greed = fetch_fear_greed_index()
    affected_markets = calculate_affected_markets(events)
    affected_etfs = calculate_etf_impacts(events)
    impact = calculate_market_impact(events)
    sentiment = calculate_market_sentiment(events)
    mood = determine_market_mood(sentiment)
    calendar_events = get_economic_calendar()
    data_load_failed = False
except Exception:
    events, indices, fear_greed = [], {}, {"available": False, "label": t("mi_fear_greed")}
    affected_markets, affected_etfs, calendar_events = [], [], []
    impact = calculate_market_impact([])
    sentiment = calculate_market_sentiment([])
    mood = determine_market_mood(sentiment)
    data_load_failed = True

if data_load_failed:
    error_state(t("mi_no_market_data"), t("mi_no_market_data"))

ai_client = get_openai_client()
ai_tag = t("ai_tag_generated") if ai_client else t("ai_tag_rule_based")

# ── Section 1: Today's Market ────────────────────────────────────────────────
section_header(t("mi_section_today_market_title"))

def _render_index_card(col, key: str) -> None:
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
                metric_card_html(info["label"], t("mi_no_market_data"), color=COLORS["text_muted"]),
                unsafe_allow_html=True,
            )


row1_keys = ["sp500", "nasdaq", "dow", "russell", "vix"]
row2_keys = ["twii", "ftse", "nikkei", "stoxx50"]  # 5th slot below is Fear & Greed

cols_row1 = st.columns(5)
for col, key in zip(cols_row1, row1_keys):
    _render_index_card(col, key)

cols_row2 = st.columns(5)
for col, key in zip(cols_row2, row2_keys):
    _render_index_card(col, key)

with cols_row2[4]:
    st.markdown(
        metric_card_html(fear_greed["label"], t("mi_placeholder_value"), t("mi_placeholder_note"),
                          color=COLORS["text_muted"]),
        unsafe_allow_html=True,
    )

# ── Section 2: Breaking Market Events ────────────────────────────────────────
section_header(t("mi_section_breaking_events_title"), t("mi_section_breaking_events_subtitle"))

if not events:
    empty_state(t("mi_no_market_data"), t("mi_section_breaking_events_subtitle"), icon="newspaper")
else:
    cards_html = "".join(
        news_card(
            title=event.title,
            time_str=event.published.strftime("%Y-%m-%d %H:%M") if event.published else "—",
            source=event.source,
            impact_label=t(f"mi_category_{event.category}"),
            impact_variant=CATEGORY_VARIANT.get(event.category, "neutral"),
            url=event.link or None,
        )
        for event in events
    )
    st.markdown(cards_html, unsafe_allow_html=True)

# ── Section 3: AI Global Market Impact Score ─────────────────────────────────
section_header(t("mi_section_impact_score_title"))
st.markdown(
    impact_score_card(impact["score"], impact["stars"], impact["label"], impact["explanation"]),
    unsafe_allow_html=True,
)

# ── Section 4: Affected Markets ──────────────────────────────────────────────
section_header(t("mi_section_affected_markets_title"), t("mi_section_affected_markets_subtitle"))

if affected_markets:
    market_cols = st.columns(len(affected_markets))
    for col, market in zip(market_cols, affected_markets):
        with col:
            st.markdown(
                status_card(market.market, "", market.label, STAR_VARIANT.get(market.stars, "neutral"),
                            star_rating_html(market.stars)),
                unsafe_allow_html=True,
            )
else:
    empty_state(t("mi_no_market_data"), t("mi_section_affected_markets_subtitle"), icon="layers")

# ── Section 5: Affected ETFs (Global ETF Coverage) ───────────────────────────
section_header(t("mi_section_affected_etfs_title"), t("mi_section_affected_etfs_subtitle"))

universe = get_global_etf_universe()
etf_by_ticker = {etf.ticker: etf for etf in affected_etfs}
REGION_ORDER = ["us", "taiwan", "uk"]

for region_key in REGION_ORDER:
    tickers = universe.get(region_key, [])
    if not tickers:
        continue
    st.markdown(f"**{t(f'mi_region_{region_key}')}**")
    region_cols = st.columns(len(tickers))
    for col, ticker in zip(region_cols, tickers):
        etf = etf_by_ticker.get(ticker)
        with col:
            if etf:
                st.markdown(
                    status_card(etf.ticker, etf.sector, etf.direction_label,
                                DIRECTION_VARIANT.get(etf.direction, "neutral"),
                                star_rating_html(etf.stars)),
                    unsafe_allow_html=True,
                )

# ── Section 6: AI Why It Matters ─────────────────────────────────────────────
section_header(t("mi_section_why_matters_title"))
why_text = generate_why_it_matters(events, affected_markets, affected_etfs)
with chart_card(t("mi_section_why_matters_title"), tag=ai_tag):
    st.markdown(why_text)

# ── Section 7: AI Market Summary (+ Overall Market Mood) ─────────────────────
section_header(t("mi_section_market_summary_title"))
summary_text = generate_market_summary(events, sentiment, affected_etfs)
with chart_card(t("mi_section_market_summary_title"), tag=ai_tag):
    st.markdown(
        f'<span class="badge badge-{mood["variant"]}">{t("mi_section_mood_label")}: {mood["label"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(summary_text)

# ── Section 8: Portfolio Impact ──────────────────────────────────────────────
section_header(t("mi_section_portfolio_impact_title"))

portfolios = load_all_portfolios()
if not portfolios:
    empty_state(t("hist_no_portfolios_title"), t("mi_portfolio_no_data"), icon="layers")
else:
    latest = portfolios[0]
    exposure = calculate_portfolio_exposure(latest["holdings"])
    impact_text = generate_portfolio_analysis(latest["holdings"], affected_etfs)

    col_text, col_chart = st.columns([2, 1])
    with col_text:
        with chart_card(latest["name"], t("mi_portfolio_using", name=latest["name"]), tag=ai_tag):
            if exposure:
                st.markdown(f"**{t('mi_portfolio_exposure_title')}**")
                for sector, weight in exposure.items():
                    st.caption(f"{sector}: {weight:.0%}")
                    st.progress(min(int(weight * 100), 100))
            st.markdown(impact_text)
    with col_chart:
        if latest["holdings"]:
            with chart_card(t("hist_allocation_breakdown_card")):
                fig = allocation_donut_chart(latest["holdings"], "")
                st.plotly_chart(fig, use_container_width=True, key="mi_portfolio_allocation_donut")

# ── Section 9: Economic Calendar ─────────────────────────────────────────────
section_header(t("mi_section_calendar_title"), t("mi_section_calendar_subtitle"))

cal_col1, cal_col2, cal_col3 = st.columns(3)
CAL_BUCKETS = [("today", "mi_cal_today", cal_col1), ("this_week", "mi_cal_this_week", cal_col2),
               ("upcoming", "mi_cal_upcoming", cal_col3)]

for when_key, label_key, col in CAL_BUCKETS:
    bucket_events = [e for e in calendar_events if e["when_key"] == when_key]
    with col:
        with chart_card(t(label_key)):
            if bucket_events:
                df = pd.DataFrame([
                    {t("mi_cal_col_event"): e["event"], t("mi_cal_col_importance"): e["importance"]}
                    for e in bucket_events
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption(t("mi_no_market_data"))

# ── Section 10: Educational Disclaimer ───────────────────────────────────────
disclaimer_box(t("mi_disclaimer"))
render_footer()
