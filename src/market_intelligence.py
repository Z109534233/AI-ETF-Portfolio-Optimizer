"""
Market Intelligence Module
Orchestration layer for the AI Global Market Intelligence Center: fetches
global index quotes, computes overall market mood, provides a placeholder
economic calendar, and generates the two AI-facing narratives (market
summary and portfolio impact analysis). Impact scoring itself lives in
impact_engine.py / global_etf.py; this module consumes their output.

Every public function here is defensive: on any data or API failure it
returns a safe empty/placeholder value instead of raising, so the page can
never crash.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

from src.ai_advisor import get_openai_client
from src.global_etf import calculate_portfolio_exposure
from src.i18n import t, get_language

# ── Market Indices ───────────────────────────────────────────────────────────
# Includes the reserved international indices (FTSE 100 / Nikkei 225 / Euro
# Stoxx 50) as real yfinance lookups -- not fake placeholders -- so they
# work as soon as the deployment environment has outbound internet access,
# even if the current sandbox does not.
INDEX_TICKERS = {
    "sp500": ("mi_sp500", "^GSPC"),
    "nasdaq": ("mi_nasdaq", "^IXIC"),
    "dow": ("mi_dow", "^DJI"),
    "russell": ("mi_russell", "^RUT"),
    "vix": ("mi_vix", "^VIX"),
    "twii": ("mi_twii", "^TWII"),
    "ftse": ("mi_ftse", "^FTSE"),
    "nikkei": ("mi_nikkei", "^N225"),
    "stoxx50": ("mi_stoxx50", "^STOXX50E"),
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_indices() -> dict:
    """
    Fetch the latest price and day-over-day change for each tracked index
    via yfinance. Returns a dict keyed by index id, each value a dict with
    "label", "available" (bool), and -- when available -- "price",
    "change", "change_pct". A per-index failure is isolated: one bad
    ticker never prevents the others from rendering.
    """
    result = {}
    for key, (label_key, symbol) in INDEX_TICKERS.items():
        label = t(label_key)
        try:
            hist = yf.download(symbol, period="5d", progress=False, auto_adjust=True, threads=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            closes = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)

            if len(closes) < 2:
                result[key] = {"label": label, "available": False}
                continue

            last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            change = last - prev
            change_pct = (change / prev * 100) if prev else 0.0
            result[key] = {
                "label": label, "available": True,
                "price": last, "change": change, "change_pct": change_pct,
            }
        except Exception:
            result[key] = {"label": label, "available": False}
    return result


def fetch_fear_greed_index() -> dict:
    """
    Placeholder for the Fear & Greed Index. No free, keyless data source is
    currently wired in, so this explicitly returns "unavailable" rather than
    fabricating a number -- the page renders a clearly-labelled placeholder
    KPI card instead.
    """
    return {"available": False, "label": t("mi_fear_greed")}


# ── Market Sentiment / Mood ──────────────────────────────────────────────────
def calculate_market_sentiment(events: list) -> dict:
    """
    Aggregate Bullish/Neutral/Bearish percentages across today's events
    (using each EventImpact's .impact field). Returns "available": False
    (all percentages at 0) when there is no news to analyze, so the page
    can show a clean empty state instead of a fake split.
    """
    if not events:
        return {
            "available": False,
            "bullish_pct": 0.0, "neutral_pct": 0.0, "bearish_pct": 0.0,
            "bullish_count": 0, "neutral_count": 0, "bearish_count": 0,
        }

    from collections import Counter
    counts = Counter(e.impact for e in events)
    total = len(events)
    bullish, bearish, neutral = counts.get("Positive", 0), counts.get("Negative", 0), counts.get("Neutral", 0)

    return {
        "available": True,
        "bullish_pct": round(bullish / total * 100, 1),
        "neutral_pct": round(neutral / total * 100, 1),
        "bearish_pct": round(bearish / total * 100, 1),
        "bullish_count": bullish, "neutral_count": neutral, "bearish_count": bearish,
    }


def determine_market_mood(sentiment: dict) -> dict:
    """Reduce a sentiment split to a single Bullish/Neutral/Bearish mood badge."""
    if not sentiment.get("available"):
        return {"mood": "Neutral", "label": t("mi_mood_neutral"), "variant": "neutral"}

    scored = [
        ("Bullish", sentiment["bullish_pct"], "mi_mood_bullish", "green"),
        ("Neutral", sentiment["neutral_pct"], "mi_mood_neutral", "neutral"),
        ("Bearish", sentiment["bearish_pct"], "mi_mood_bearish", "red"),
    ]
    mood, _pct, label_key, variant = max(scored, key=lambda x: x[1])
    return {"mood": mood, "label": t(label_key), "variant": variant}


# ── Economic Calendar ────────────────────────────────────────────────────────
def get_economic_calendar() -> list:
    """
    Static, clearly-labelled placeholder economic calendar (Fed Meeting, CPI,
    PPI, GDP, NFP, FOMC) grouped into Today / This Week / Upcoming via
    "when_key". Intentionally simple and easy to swap for a live data source
    (e.g. FRED, Trading Economics) later without changing the page that
    renders it.
    """
    return [
        {"event": "US CPI (Consumer Price Index)", "when_key": "today", "importance": t("mi_cal_high")},
        {"event": "Fed Chair Public Remarks", "when_key": "today", "importance": t("mi_cal_medium")},
        {"event": "FOMC / Fed Interest Rate Decision", "when_key": "this_week", "importance": t("mi_cal_high")},
        {"event": "PPI (Producer Price Index)", "when_key": "this_week", "importance": t("mi_cal_medium")},
        {"event": "Non-Farm Payrolls (NFP)", "when_key": "upcoming", "importance": t("mi_cal_high")},
        {"event": "GDP Growth Rate", "when_key": "upcoming", "importance": t("mi_cal_medium")},
    ]


# ── AI Market Summary (reserved for OpenAI / Claude / Gemini) ───────────────
def generate_market_summary(events: list, sentiment: dict, affected_etfs: list) -> str:
    """
    Generate a ~100-200 word "Today's Market Summary" from current events.
    Uses OpenAI when an API key is configured, otherwise falls back to a
    rule-based summary built from the same data. Reserved entry point for
    a future Claude/Gemini model. Never raises -- always returns
    display-safe text, including when there is no news at all.
    """
    if not events:
        return t("mi_summary_no_news")

    client = get_openai_client()
    if client is not None:
        try:
            return _generate_ai_summary(client, events, sentiment, affected_etfs)
        except Exception:
            pass  # fall through to the rule-based summary below

    return _generate_rule_based_summary(events, sentiment, affected_etfs)


def _generate_ai_summary(client, events: list, sentiment: dict, affected_etfs: list) -> str:
    headlines = "\n".join(f"- {e.title}" for e in events[:8])
    affected_str = ", ".join(f"{e.ticker} ({e.sector}: {e.direction_label})" for e in affected_etfs) or "N/A"
    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文)."
        if get_language() == "zh-TW" else "Respond entirely in English."
    )
    prompt = f"""You are an educational market analyst assistant. Based on today's events below, write a "Today's Market Summary" of about 100-200 words covering: the main market events today, overall market sentiment, which sectors are affected, and which ETFs may be affected. Do not predict future prices or give personalised investment advice -- this is decision support, not a trade recommendation. {language_instruction}

Events:
{headlines}

Sentiment split: {sentiment['bullish_pct']}% bullish, {sentiment['neutral_pct']}% neutral, {sentiment['bearish_pct']}% bearish.
Potentially affected ETFs: {affected_str}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional, educational market analyst. Never give personalised investment advice or price predictions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
        temperature=0.6,
    )
    return response.choices[0].message.content


def _generate_rule_based_summary(events: list, sentiment: dict, affected_etfs: list) -> str:
    top_titles = [e.title for e in events[:3]]

    if sentiment["bullish_pct"] > sentiment["bearish_pct"] + 10:
        tilt = t("mi_tilt_bullish")
    elif sentiment["bearish_pct"] > sentiment["bullish_pct"] + 10:
        tilt = t("mi_tilt_bearish")
    else:
        tilt = t("mi_tilt_mixed")

    positive_etfs = [e.ticker for e in affected_etfs if e.direction == "Positive"]
    negative_etfs = [e.ticker for e in affected_etfs if e.direction == "Negative"]

    parts = [t("mi_summary_intro", headline=top_titles[0] if top_titles else "")]
    if len(top_titles) > 1:
        parts.append(t("mi_summary_more_headlines", headlines="; ".join(top_titles[1:])))
    parts.append(t(
        "mi_summary_sentiment", tilt=tilt,
        bullish=sentiment["bullish_pct"], bearish=sentiment["bearish_pct"]
    ))
    if positive_etfs:
        parts.append(t("mi_summary_positive_etfs", tickers=", ".join(positive_etfs)))
    if negative_etfs:
        parts.append(t("mi_summary_negative_etfs", tickers=", ".join(negative_etfs)))
    parts.append(t("mi_summary_disclaimer_note"))

    return " ".join(parts)


# ── AI Why It Matters (reserved for OpenAI / Claude / Gemini) ───────────────
CATEGORY_EXPLANATION_KEY = {
    "monetary_policy": "mi_why_monetary_policy",
    "trade": "mi_why_trade",
    "earnings": "mi_why_earnings",
    "macro_data": "mi_why_macro_data",
    "commodities": "mi_why_commodities",
    "corporate": "mi_why_corporate",
    "general": "mi_why_general",
}


def generate_why_it_matters(events: list, affected_markets: list, affected_etfs: list) -> str:
    """
    AI "Why It Matters" causal explanation -- reserved entry point for a
    future LLM-based reasoning model (OpenAI/Claude/Gemini). Builds a short
    cause -> sector -> ETF chain from the highest-weight event plus the
    currently keyword-matched markets/ETFs. Explains topical relevance
    only -- never predicts price direction or gives investment advice.
    """
    if not events:
        return t("mi_summary_no_news")

    top_event = max(events, key=lambda e: e.weight)
    category_sentence = t(CATEGORY_EXPLANATION_KEY.get(top_event.category, "mi_why_general"))

    relevant_etfs = [e.ticker for e in affected_etfs if e.direction != "Neutral"][:3]
    top_markets = [m.market for m in affected_markets[:2]]

    parts = [f"{top_event.title}.", category_sentence]
    if top_markets:
        parts.append(t("mi_why_market_note", markets=", ".join(top_markets)))
    if relevant_etfs:
        parts.append(t("mi_why_etf_note", tickers=", ".join(relevant_etfs)))
    parts.append(t("mi_why_disclaimer"))
    return " ".join(parts)


# ── AI Portfolio Analysis (reserved for OpenAI / Claude / Gemini) ───────────
def generate_portfolio_analysis(holdings: dict, affected_etfs: list) -> str:
    """
    Given a saved portfolio's holdings ({ticker: weight}) and the output of
    global_etf.calculate_etf_impacts(), produce a short rule-based narrative
    describing the portfolio's dominant sector exposure and which held
    tickers today's events may be relevant to. Reserved entry point for a
    future LLM model. Analyzes possible relevance only -- never predicts
    prices or returns.
    """
    if not holdings:
        return t("mi_portfolio_no_data")

    impact_by_ticker = {e.ticker: e for e in affected_etfs}
    exposure = calculate_portfolio_exposure(holdings)
    if not exposure:
        return t("mi_portfolio_no_data")

    top_sector, top_weight = next(iter(exposure.items()))
    lines = [t("mi_portfolio_exposure", sector=top_sector, weight=f"{top_weight:.0%}")]

    relevant = [
        (tk, impact_by_ticker[tk]) for tk in holdings
        if tk in impact_by_ticker and impact_by_ticker[tk].direction != "Neutral"
    ]
    if relevant:
        for tk, info in relevant:
            verb = t("mi_impact_positive_verb") if info.direction == "Positive" else t("mi_impact_negative_verb")
            lines.append(t("mi_portfolio_news_line", ticker=tk, verb=verb))
    else:
        lines.append(t("mi_portfolio_no_relevant_news"))

    return " ".join(lines)
