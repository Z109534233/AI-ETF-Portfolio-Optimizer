"""
News Service Module
Fetches recent market-moving headlines via yfinance's built-in news feed
(no API key required), categorizes and weights each one, and classifies its
headline sentiment. Feeds the "Breaking Market Events" section and every
downstream impact-scoring function (impact_engine, global_etf).
"""

from datetime import datetime
import streamlit as st
import yfinance as yf

from src.market_models import EventImpact

# Representative broad-market tickers whose .news feed gives a cross-section
# of "what's moving markets today" without needing a paid news API.
NEWS_SOURCE_TICKERS = ["SPY", "QQQ", "DIA", "^GSPC"]

POSITIVE_KEYWORDS = [
    "surge", "rally", "soar", "jump", "gain", "record high", "rebound",
    "climb", "bullish", "upgrade", "outperform", "beat", "optimis", "recovery",
    "boost", "rises", "rose", "strong demand",
]
NEGATIVE_KEYWORDS = [
    "plunge", "crash", "drop", "sell-off", "selloff", "tumble", "slump",
    "fear", "recession", "downgrade", "miss", "bearish", "warns", "slide",
    "losses", "volatility spikes", "falls", "fell", "sinks", "worries", "concern",
]

# category key -> headline keywords used to classify each breaking event.
CATEGORY_KEYWORDS = {
    "monetary_policy": ["fed", "interest rate", "rate decision", "fomc", "central bank", "boe", "ecb", "boj", "rate hike", "rate cut"],
    "trade": ["tariff", "trade war", "export", "import", "sanction"],
    "earnings": ["earnings", "quarterly results", "revenue", "eps", "guidance"],
    "macro_data": ["cpi", "ppi", "gdp", "non-farm", "payroll", "pmi", "inflation", "unemployment", "jobs report"],
    "commodities": ["oil", "crude", "gold", "commodity", "opec"],
    "corporate": ["merger", "acquisition", "ipo", "lawsuit", "ceo", "buyback"],
}

# categories/terms that represent unusually market-moving events get a
# higher "weight" (a simple magnitude proxy consumed by impact_engine).
HIGH_WEIGHT_CATEGORIES = {"monetary_policy", "trade"}
HIGH_WEIGHT_TERMS = ["tariff", "rate decision", "war", "crisis", "crash", "surge", "record", "emergency"]
MEDIUM_WEIGHT_CATEGORIES = {"earnings", "macro_data"}


def classify_headline_sentiment(title: str) -> str:
    """Lightweight keyword-based sentiment classifier for a news headline.

    This is a transparent, dependency-free fallback (not a real NLP model)
    used when no sentiment-analysis API is configured. Returns one of
    "Positive", "Negative", "Neutral".
    """
    if not title:
        return "Neutral"
    low = title.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in low)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in low)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def categorize_event(title: str) -> str:
    """Classify a headline into a broad event category (internal key)."""
    low = title.lower()
    for key, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return key
    return "general"


def event_weight(title: str, category: str) -> int:
    """
    A simple 1-3 magnitude proxy for how market-moving an event likely is.
    Reserved as an easy hook for a future model to replace with a real
    materiality score.
    """
    low = title.lower()
    if category in HIGH_WEIGHT_CATEGORIES or any(term in low for term in HIGH_WEIGHT_TERMS):
        return 3
    if category in MEDIUM_WEIGHT_CATEGORIES:
        return 2
    return 1


@st.cache_data(ttl=900, show_spinner=False)
def fetch_breaking_events(limit: int = 10) -> list:
    """
    Fetch recent market headlines using yfinance's built-in `.news` feed
    (no API key required). Aggregates across a few broad-market tickers,
    de-duplicates by title, and sorts by publish time (newest first).

    Returns a list of EventImpact objects. Never raises -- returns an empty
    list if no data is available (offline, rate-limited, etc.), so the page
    can render a clean "No market data available" state instead of crashing.
    """
    seen_titles = set()
    events = []

    for symbol in NEWS_SOURCE_TICKERS:
        try:
            raw_news = yf.Ticker(symbol).news or []
        except Exception:
            continue

        for entry in raw_news:
            if not isinstance(entry, dict):
                continue

            # yfinance's news item shape has changed across versions: newer
            # releases nest the real fields inside a "content" sub-dict.
            content = entry.get("content") if isinstance(entry.get("content"), dict) else {}

            title = (content.get("title") or entry.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            link = ""
            canonical = content.get("canonicalUrl")
            if isinstance(canonical, dict):
                link = canonical.get("url", "")
            if not link:
                link = entry.get("link", "")

            publisher = ""
            provider = content.get("provider")
            if isinstance(provider, dict):
                publisher = provider.get("displayName", "")
            if not publisher:
                publisher = entry.get("publisher", "Yahoo Finance")

            pub_time = None
            ts = entry.get("providerPublishTime")
            if ts:
                try:
                    pub_time = datetime.fromtimestamp(ts)
                except Exception:
                    pub_time = None
            if pub_time is None:
                pub_date_str = content.get("pubDate")
                if pub_date_str:
                    try:
                        pub_time = datetime.strptime(pub_date_str[:19], "%Y-%m-%dT%H:%M:%S")
                    except Exception:
                        pub_time = None

            category = categorize_event(title)
            events.append(EventImpact(
                title=title,
                category=category,
                source=publisher,
                published=pub_time,
                impact=classify_headline_sentiment(title),
                link=link,
                weight=event_weight(title, category),
            ))

    events.sort(key=lambda e: e.published or datetime.min, reverse=True)
    return events[:limit]
