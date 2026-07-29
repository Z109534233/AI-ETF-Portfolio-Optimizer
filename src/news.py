"""
Market News Module
Fetches recent market-moving headlines via yfinance's built-in news feed
(no API key required) and provides a lightweight keyword-based sentiment
classifier shared across the Market Intelligence page.
"""

from datetime import datetime
import streamlit as st
import yfinance as yf

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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_news(limit: int = 10) -> list:
    """
    Fetch recent market headlines using yfinance's built-in `.news` feed
    (no API key required). Aggregates across a few broad-market tickers,
    de-duplicates by title, and sorts by publish time (newest first).

    Returns a list of dicts: title, publisher, link, published (datetime or
    None), impact ("Positive"/"Negative"/"Neutral"). Never raises -- returns
    an empty list if no data is available (offline, rate-limited, etc.), so
    the page can render a clean "No market news available" state instead of
    crashing.
    """
    seen_titles = set()
    items = []

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

            items.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": pub_time,
                "impact": classify_headline_sentiment(title),
            })

    items.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return items[:limit]
