"""
Page 8: Market Intelligence Center (Issue #20 section 10 -- renamed from "AI
Market Intelligence Center"; classification/impact/sentiment scoring here is
rule-based, only the "Today's Market Summary" card is genuinely OpenAI-backed)
A market intelligence dashboard (not a news site): today's index snapshot,
breaking headlines, an AI/rule-based market summary, ETFs today's news may
affect, aggregate headline sentiment, a placeholder economic calendar, and
(when a portfolio has been saved) a portfolio-impact analysis.

Issue #22 sections H/I: reorganized into tabs (Overview / News / Portfolio
Impact / Sentiment / Methodology) to cut scrolling, and audited for
decorative emoji that added no navigational/semantic value (flags, mood
emoji, and a "hot headline" fire icon were dropped -- star ratings and
status badges already carry that meaning).
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import load_all_portfolios, init_database
from src.news import fetch_market_news
from src.market_intelligence import (
    fetch_market_indices, fetch_fear_greed_index, get_affected_etfs,
    calculate_market_sentiment, calculate_ai_market_sentiment, get_economic_calendar,
    generate_market_summary, analyze_portfolio_impact,
    calculate_affected_markets, generate_today_ai_summary, calculate_market_impact,
    get_todays_major_events, generate_etf_card_data, get_news_card_metadata,
    generate_todays_market_action, stars_to_impact_label,
)
from src.charts import sentiment_donut_chart, allocation_donut_chart
from src.theme import COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, news_card, status_card, star_rating_html,
    market_impact_card, ai_sentiment_card, empty_state, error_state,
    chart_caption, ai_interpret_button,
)
from src.i18n import t, get_language, t_opt_method

st.set_page_config(
    page_title="Market Intelligence | AI ETF Portfolio Optimizer",
    page_icon="\U0001F4F0",
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
    ai_sentiment = calculate_ai_market_sentiment(news_items)
    calendar_events = get_economic_calendar()
    affected_markets = calculate_affected_markets(news_items)
    today_ai_summary = generate_today_ai_summary(news_items, indices)
    market_impact = calculate_market_impact(news_items)
    major_events = get_todays_major_events(news_items, limit=5)
    etf_cards = generate_etf_card_data(news_items)
    news_card_meta = get_news_card_metadata(news_items)
    todays_market_action = generate_todays_market_action(news_items)
    data_load_failed = False
except Exception:
    news_items, indices, fear_greed = [], {}, {"available": False, "label": t("mi_fear_greed")}
    affected_etfs, sentiment, calendar_events = [], calculate_market_sentiment([]), []
    ai_sentiment = calculate_ai_market_sentiment([])
    affected_markets = []
    today_ai_summary = generate_today_ai_summary([], {})
    market_impact = calculate_market_impact([])
    major_events = []
    etf_cards = generate_etf_card_data([])
    news_card_meta = get_news_card_metadata([])
    todays_market_action = generate_todays_market_action([])
    data_load_failed = True

if data_load_failed:
    error_state(t("mi_no_news_available"), t("mi_summary_no_news"))

_mi_lang = get_language()
_major_events_title = "今日重大事件" if _mi_lang == "zh-TW" else "Today's Major Events"
_sentiment_caption = "市場情緒" if _mi_lang == "zh-TW" else "Market Sentiment"
_markets_caption = "受影響市場" if _mi_lang == "zh-TW" else "Affected Markets"
_etfs_caption = "受影響 ETF" if _mi_lang == "zh-TW" else "Affected ETFs"
_confidence_caption = "情緒傾向強度" if _mi_lang == "zh-TW" else "Sentiment Skew"
_impact_caption = t("mi_impact_caption")
_watchlist_title = "今日觀察清單" if _mi_lang == "zh-TW" else "Today's Watchlist"
_reasons_caption = "原因" if _mi_lang == "zh-TW" else "Reasons"
_etf_market_caption = "市場" if _mi_lang == "zh-TW" else "Market"
_etf_reasons_caption = "為什麼？" if _mi_lang == "zh-TW" else "Why?"
_top_drivers_caption = "Top Drivers"

tab_overview, tab_news, tab_impact, tab_sentiment, tab_methodology = st.tabs([
    t("mi_tab_overview"), t("mi_tab_news"), t("mi_tab_portfolio_impact"),
    t("mi_tab_sentiment"), t("mi_tab_methodology"),
])

# ══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_overview:
    # ── Today's Major Events (top 3-5 headlines by Market Impact Score) ──
    section_header(_major_events_title)
    if major_events:
        event_cols = st.columns(len(major_events))
        for col, event in zip(event_cols, major_events):
            markets_html = "".join(f'<div class="affected-by-item">{m}</div>' for m in event["affected_markets"])
            markets_block = (
                f'<div class="affected-by-caption">{_markets_caption}</div><div class="affected-by-list">{markets_html}</div>'
                if event["affected_markets"] else ""
            )
            etfs_html = "".join(f'<div class="affected-by-item">{tk}</div>' for tk in event["affected_etfs"])
            etfs_block = (
                f'<div class="affected-by-caption">{_etfs_caption}</div><div class="affected-by-list">{etfs_html}</div>'
                if event["affected_etfs"] else ""
            )
            with col:
                st.markdown(
                    '<div class="status-card market-impact-card">'
                    f'<div class="status-card-ticker">{event["headline"]}</div>'
                    f'<div class="status-card-sector">{event.get("publisher", "")}'
                    f'{(" · " + event.get("published_text", "")) if event.get("published_text") else ""}</div>'
                    f'<div class="affected-by-caption">{_impact_caption}</div>'
                    f'<div class="status-card-stars">{star_rating_html(event["stars"])}</div>'
                    f'<div class="market-impact-label">{event["category"]}</div>'
                    f'<div class="status-card-sector">{_sentiment_caption}: '
                    f'<span class="badge badge-{event["sentiment_variant"]}">{event["sentiment_label"]}</span></div>'
                    f'{markets_block}'
                    f'{etfs_block}'
                    '</div>',
                    unsafe_allow_html=True,
                )
    else:
        empty_state(t("mi_no_news_available"), _major_events_title, icon="activity")

    # ── Today's AI Summary (left) + Today's Watchlist (right) ────────────
    section_header(today_ai_summary["title"])
    col_summary, col_watchlist = st.columns([2.5, 1])
    with col_summary:
        with chart_card(today_ai_summary["title"], tag=t("ai_tag_rule_based")):
            for _section in today_ai_summary["sections"]:
                st.markdown(f"**{_section['heading']}**")
                st.markdown(_section["text"])
            if today_ai_summary["disclaimer"]:
                st.caption(today_ai_summary["disclaimer"])

        # ── Today's Market Action (template-generated, no LLM call) ──────
        with chart_card(todays_market_action["title"], tag=t("ai_tag_rule_based")):
            for _action_item in todays_market_action["items"]:
                st.markdown(f"• {_action_item}")
    with col_watchlist:
        with st.container(border=True):
            st.markdown(f"**{_watchlist_title}**")
            if major_events:
                for _event in major_events:
                    st.markdown(_event["headline"])
                    st.markdown(
                        f'{star_rating_html(_event["stars"])} '
                        f'<span class="badge badge-neutral">{_event["category"]}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(t("mi_no_news_available"))

    # ── Market Impact Score ───────────────────────────────────────────────
    # ONE primary scale (Issue #43 item F): the numeric score/100 stays the
    # primary value, and the secondary text is a localized QUALITATIVE
    # impact level (e.g. "High Impact" / "高度影響") derived from the exact
    # same star count already used elsewhere on this page -- never a second
    # 1-5 star rating duplicating the same market-impact concept. Stars
    # remain fine on the per-news/per-market cards elsewhere on this page.
    _mi_score_title = "市場影響分數" if _mi_lang == "zh-TW" else "Market Impact Score"
    _mi_impact_level_label = stars_to_impact_label(market_impact["stars"])
    _mi_score_color = (
        COLORS["danger"] if market_impact["stars"] >= 4
        else COLORS["warning"] if market_impact["stars"] == 3
        else COLORS["text_muted"]
    )
    st.markdown(
        metric_card_html(_mi_score_title, f"{market_impact['score']}/100", _mi_impact_level_label, color=_mi_score_color),
        unsafe_allow_html=True,
    )

    # ── Today's Market Overview ────────────────────────────────────────────
    section_header(t("mi_section_overview_title"))

    # Fear & Greed has no verified/keyless data source connected yet
    # (fetch_fear_greed_index() always returns available=False) -- render
    # it as a KPI card ONLY when a real source is actually available, so it
    # never looks like a broken/empty live metric next to VIX (Issue #43
    # item G). The 5 real indices always render their own column,
    # individually showing "Data unavailable" only on a genuine fetch
    # failure -- a different, temporary condition from Fear & Greed's
    # structural "not connected" state.
    _mi_show_fear_greed = bool(fear_greed.get("available"))
    _mi_index_keys = ["sp500", "nasdaq", "dow", "russell", "vix"]
    _mi_cols = st.columns(len(_mi_index_keys) + (1 if _mi_show_fear_greed else 0))
    index_cols = dict(zip(_mi_index_keys, _mi_cols))

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

    if _mi_show_fear_greed:
        with _mi_cols[-1]:
            st.markdown(
                metric_card_html(fear_greed["label"], str(fear_greed.get("value", "—")),
                                  fear_greed.get("category"), color=COLORS["primary"]),
                unsafe_allow_html=True,
            )
    else:
        st.caption(t("mi_fear_greed_not_connected"))

    # ── Affected Markets ──────────────────────────────────────────────────
    section_header(t("mi_section_affected_markets_title"), t("mi_section_affected_markets_subtitle"))

    if affected_markets:
        market_cols = st.columns(len(affected_markets))
        for col, market in zip(market_cols, affected_markets):
            reasons_html = "".join(f'<span class="badge badge-neutral">{r}</span> ' for r in market["reasons"])
            reasons_block = (
                f'<div class="affected-by-caption">{_reasons_caption}</div><div>{reasons_html}</div>'
                if market["reasons"] else ""
            )
            with col:
                st.markdown(
                    '<div class="status-card market-impact-card">'
                    f'<div class="status-card-ticker">{market["market"]}</div>'
                    f'<div class="status-card-stars">{star_rating_html(market["stars"])}</div>'
                    f'<div class="market-impact-label">{market["impact_label"]}</div>'
                    f'{reasons_block}'
                    '</div>',
                    unsafe_allow_html=True,
                )
    else:
        empty_state(t("mi_no_news_available"), t("mi_section_affected_markets_subtitle"), icon="layers")

    # ── Global ETFs (Affected ETFs across US / Taiwan / UK) ───────────────
    section_header(t("mi_section_global_etfs_title"), t("mi_section_global_etfs_subtitle"))

    if etf_cards:
        etf_cols = st.columns(len(etf_cards))
        for col, etf in zip(etf_cols, etf_cards):
            reasons_html = "".join(f'<span class="badge badge-neutral">{r}</span> ' for r in etf["reasons"])
            reasons_block = (
                f'<div class="affected-by-caption">{_etf_reasons_caption}</div><div>{reasons_html}</div>'
                if etf["reasons"] else ""
            )
            drivers_html = "".join(f'<span class="badge badge-neutral">{d}</span> ' for d in etf["top_drivers"])
            drivers_block = (
                f'<div class="affected-by-caption">{_top_drivers_caption}</div><div>{drivers_html}</div>'
                if etf["top_drivers"] else ""
            )
            with col:
                st.markdown(
                    '<div class="status-card market-impact-card">'
                    f'<div class="status-card-ticker">{etf["ticker"]}</div>'
                    f'<div class="affected-by-caption">{_impact_caption}</div>'
                    f'<div class="status-card-stars">{star_rating_html(etf["stars"])}</div>'
                    f'<div class="status-card-sector">{_etf_market_caption}: '
                    f'<span class="badge badge-{etf["sentiment_variant"]}">{etf["sentiment_label"]}</span></div>'
                    f'{reasons_block}'
                    f'{drivers_block}'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"{_confidence_caption}: {etf['confidence']}%")
                st.progress(min(int(etf["confidence"]), 100))
    else:
        empty_state(t("mi_no_news_available"), t("mi_section_global_etfs_subtitle"), icon="layers")

    # ── Economic Calendar ──────────────────────────────────────────────────
    section_header(t("mi_section_calendar_title"), t("mi_section_calendar_subtitle"))

    with chart_card(t("mi_section_calendar_title")):
        if calendar_events:
            calendar_df = pd.DataFrame([
                {
                    t("mi_cal_col_event"): e["event"],
                    t("mi_cal_col_when"): e["when"],
                    t("mi_cal_col_importance"): e["importance"],
                }
                for e in calendar_events
            ])
            st.dataframe(calendar_df, use_container_width=True, hide_index=True)
            chart_caption(t("mi_caption_calendar"))
        else:
            st.info(t("mi_calendar_no_verified_data"))

# ══════════════════════════════════════════════════════════════════════════
# NEWS
# ══════════════════════════════════════════════════════════════════════════
with tab_news:
    section_header(t("mi_section_news_title"), t("mi_section_news_subtitle"))

    if not news_items:
        empty_state(t("mi_no_news_available"), t("mi_section_news_subtitle"), icon="newspaper")
    else:
        for item, meta in zip(news_items, news_card_meta):
            time_str = item["published"].strftime("%Y-%m-%d %H:%M") if item["published"] else "—"
            title_html = f'<a href="{item["link"]}" target="_blank" rel="noopener noreferrer">{item["title"]}</a>' if item["link"] else item["title"]
            st.markdown(
                '<div class="news-card">'
                f'<div class="news-card-title">{title_html}</div>'
                f'<div class="news-card-meta"><span>{time_str}</span><span>&middot;</span><span>{item["publisher"]}</span></div>'
                f'<div class="status-card-sector">{meta["category"]}</div>'
                f'<div class="affected-by-caption">{_impact_caption}</div>'
                f'<div class="status-card-stars">{star_rating_html(meta["stars"])}</div>'
                '<div class="news-card-footer">'
                f'<span class="badge badge-{meta["sentiment_variant"]}">{meta["sentiment_label"]}</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{_confidence_caption}: {meta['confidence']}%")
            st.progress(min(int(meta["confidence"]), 100))

    # ── AI Market Summary ──────────────────────────────────────────────────
    section_header(t("mi_section_summary_title"))

    summary_result = generate_market_summary(
        news_items, sentiment, affected_etfs,
        session_state=st.session_state, market_indices=indices,
    )
    with chart_card(t("mi_section_summary_title"), tag=t("ai_tag_generated") if summary_result["source"] == "ai" else t("ai_tag_rule_based")):
        st.markdown(summary_result["text"])

# ══════════════════════════════════════════════════════════════════════════
# PORTFOLIO IMPACT
# ══════════════════════════════════════════════════════════════════════════
with tab_impact:
    # Prefers the ONE canonical st.session_state["current_portfolio"] (built
    # in Portfolio Optimizer -- same object AI Advisor/Investment Simulator/
    # Risk Analytics consume) over a saved-database record, so this section
    # reflects what the user is actually looking at right now rather than a
    # possibly stale or never-updated saved portfolio (Issue #18 Stage 7).
    # Falls back to the most recently saved portfolio only when no current
    # portfolio exists yet, preserving the pre-existing standalone behavior.
    section_header(t("mi_section_portfolio_impact_title"))

    current_portfolio_mi = st.session_state.get("current_portfolio")
    if current_portfolio_mi and current_portfolio_mi.get("weights"):
        mi_portfolio_name = t_opt_method(current_portfolio_mi.get("strategy", ""))
        mi_portfolio_holdings = {
            tk: w for tk, w in current_portfolio_mi["weights"].items() if float(w) >= 0.005
        }
        mi_portfolio_caption = t("mi_portfolio_using_current", name=mi_portfolio_name)
    else:
        # Public deployment fallback: never present an arbitrary database row
        # written by another visitor as "your" portfolio. If this session has
        # no active optimizer portfolio, use only a curated synthetic demo and
        # label it explicitly as shared demo data.
        portfolios = load_all_portfolios()
        demo_portfolios = [
            p for p in portfolios if bool((p.get("metadata") or {}).get("synthetic_demo"))
        ]
        if demo_portfolios:
            demo = demo_portfolios[0]
            mi_portfolio_name = demo["name"]
            mi_portfolio_holdings = {
                tk: w for tk, w in demo["holdings"].items() if float(w) >= 0.005
            }
            mi_portfolio_caption = t("mi_portfolio_using_shared_demo", name=mi_portfolio_name)
        else:
            mi_portfolio_name = None
            mi_portfolio_holdings = None
            mi_portfolio_caption = None

    if not mi_portfolio_holdings:
        empty_state(t("hist_no_portfolios_title"), t("mi_portfolio_no_data"), icon="layers")
    else:
        impact_text = analyze_portfolio_impact(mi_portfolio_holdings, affected_etfs)
        col_text, col_chart = st.columns([2, 1])
        with col_text:
            with chart_card(mi_portfolio_name, mi_portfolio_caption):
                st.markdown(impact_text)
        with col_chart:
            with chart_card(t("hist_allocation_breakdown_card")):
                fig = allocation_donut_chart(mi_portfolio_holdings, "")
                st.plotly_chart(fig, use_container_width=True, key="mi_portfolio_allocation_donut")
                chart_caption(t("mi_caption_allocation_donut"))
                # context_text built ONLY from values already displayed in
                # this donut -- the portfolio name/strategy and each
                # holding's own allocation share (Issue #24 item 4 rule E:
                # explain what's already computed, never predict/recommend).
                _allocation_context = "Portfolio: {}. Allocation by ticker: {}.".format(
                    mi_portfolio_name,
                    ", ".join(
                        f"{tk} {w:.1%}"
                        for tk, w in sorted(mi_portfolio_holdings.items(), key=lambda kv: kv[1], reverse=True)
                    ),
                )
                ai_interpret_button("mi_allocation_donut_interpret", st.session_state, _allocation_context)

# ══════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ══════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    # ── AI Market Sentiment (rule-based engine, not a headline count) ─────
    section_header(t("mi_section_ai_sentiment_title"), t("mi_section_ai_sentiment_subtitle"))

    if news_items:
        st.markdown(
            ai_sentiment_card(
                "", ai_sentiment["label"], ai_sentiment["variant"],
                t("mi_ai_sentiment_confidence_label"), ai_sentiment["confidence"],
                t("mi_ai_sentiment_drivers_label"), ai_sentiment["drivers"],
                t("mi_ai_sentiment_updated_label"), datetime.now().strftime("%H:%M"),
            ),
            unsafe_allow_html=True,
        )
        st.progress(min(int(ai_sentiment["confidence"]), 100))
    else:
        empty_state(t("mi_no_news_available"), t("mi_ai_sentiment_no_data"), icon="activity")

    # ── News Sentiment Analysis ────────────────────────────────────────────
    section_header(t("mi_section_sentiment_title"), t("mi_section_sentiment_subtitle"))

    with chart_card(t("mi_section_sentiment_title")):
        if sentiment["available"]:
            col_chart, col_bars = st.columns([1, 1])
            with col_chart:
                fig = sentiment_donut_chart(sentiment["bullish_pct"], sentiment["neutral_pct"], sentiment["bearish_pct"])
                st.plotly_chart(fig, use_container_width=True, key="mi_sentiment_donut")
                chart_caption(t("mi_caption_sentiment_donut"))
                # context_text built ONLY from the already-displayed bullish/
                # neutral/bearish percentages -- explains the existing
                # breakdown, never predicts a market move (Issue #24 item 4
                # rule E).
                _sentiment_context = (
                    f"Today's news sentiment breakdown: {sentiment['bullish_pct']:.1f}% bullish, "
                    f"{sentiment['neutral_pct']:.1f}% neutral, {sentiment['bearish_pct']:.1f}% bearish."
                )
                ai_interpret_button("mi_sentiment_donut_interpret", st.session_state, _sentiment_context)
            with col_bars:
                st.caption(f"{t('mi_sentiment_bullish_pct_label')}: {sentiment['bullish_pct']:.1f}%")
                st.progress(min(int(sentiment["bullish_pct"]), 100))
                st.caption(f"{t('mi_sentiment_neutral_pct_label')}: {sentiment['neutral_pct']:.1f}%")
                st.progress(min(int(sentiment["neutral_pct"]), 100))
                st.caption(f"{t('mi_sentiment_bearish_pct_label')}: {sentiment['bearish_pct']:.1f}%")
                st.progress(min(int(sentiment["bearish_pct"]), 100))
        else:
            st.info(t("mi_sentiment_no_data"))

# ══════════════════════════════════════════════════════════════════════════
# METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════
with tab_methodology:
    # Compact disclosure of the ACTUAL pipeline and its validation status --
    # see src/methodology.py's MARKET_INTELLIGENCE_METHODOLOGY, the single
    # source of truth this panel and tests/test_methodology_m4.py both read
    # from.
    st.caption(t("mi_methodology_subtitle"))
    st.markdown(
        f"- **{t('mi_methodology_pipeline_label')}** — {t('mi_methodology_pipeline_desc')}\n"
        f"- **{t('mi_methodology_ai_vs_rule_label')}** — {t('mi_methodology_ai_vs_rule_desc')}\n"
        f"- **{t('mi_methodology_measures_label')}** — {t('mi_methodology_measures_desc')}\n"
        f"- **{t('mi_methodology_markets_label')}** — {t('mi_methodology_markets_desc')}\n"
        f"- **{t('mi_methodology_source_label')}** — {t('mi_methodology_source_desc')}"
    )
    st.warning(f"**{t('mi_methodology_validation_label')}** — {t('mi_methodology_validation_desc')}")

# ── Educational Disclaimer ────────────────────────────────────────────────
disclaimer_box(t("mi_disclaimer"))
render_footer()
