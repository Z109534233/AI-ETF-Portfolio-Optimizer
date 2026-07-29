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

# Deliberately broad: real headlines rarely use textbook words like
# "bullish"/"bearish" outright, so the keyword list leans on the concrete,
# recurring vocabulary of financial reporting (rate decisions, earnings,
# trade policy, inflation prints, geopolitical events, ...) rather than a
# handful of generic adjectives. This is what keeps most real headlines out
# of a meaningless default "Neutral" bucket.
POSITIVE_KEYWORDS = [
    "surge", "rally", "soar", "jump", "gain", "record high", "rebound",
    "climb", "bullish", "upgrade", "outperform", "beat", "beats", "optimis",
    "recovery", "boost", "rises", "rose", "strong demand", "rate cut",
    "cuts rate", "cuts rates", "cut rates", "lowers rate", "lower rates",
    "cuts interest rate", "cut interest rate", "lowers interest rate",
    "strong earnings", "earnings beat", "beat estimates", "beats estimates",
    "beat expectations", "beats expectations", "record profit", "record earnings",
    "expands", "accelerates", "stronger than expected", "tops estimates",
    "raises guidance", "raises forecast", "buyback", "stimulus", "easing",
    "cools", "cooling inflation", "eases", "trade deal", "deal reached",
    "ceasefire", "cease-fire", "peace talks",
]
NEGATIVE_KEYWORDS = [
    "plunge", "crash", "drop", "sell-off", "selloff", "tumble", "slump",
    "fear", "recession", "downgrade", "miss", "misses", "bearish", "warns",
    "slide", "losses", "volatility spikes", "falls", "fell", "sinks",
    "worries", "concern", "tariff", "tariffs", "inflation higher",
    "inflation rises", "inflation surges", "inflation jumps",
    "hotter than expected", "geopolitical", "conflict", "war", "sanctions",
    "invasion", "attack", "shortage", "layoffs", "job cuts", "cuts jobs",
    "default", "bankruptcy", "lawsuit", "investigation", "probe",
    "widens deficit", "misses estimates", "misses expectations",
    "weaker than expected", "cuts guidance", "cuts forecast", "rate hike",
    "hikes rate", "hikes rates", "raises rates", "escalates", "strike",
    "hikes interest rate", "raises interest rate",
]

# Headlines matching these are informational/scheduling in nature -- they
# don't carry a clear bullish/bearish signal even when a keyword above
# happens to match a fragment of them (e.g. a "Fed Chair Speech" headline
# mentioning "rate" isn't itself bullish or bearish). These are routed to
# the "Market Updates" bucket instead of Breaking Events.
UPDATE_KEYWORDS = [
    "fed speech", "fed chair", "chair powell", "powell speaks", "to speak",
    "ecb meeting", "fomc meeting", "fomc minutes", "meeting minutes",
    "economic calendar", "earnings call", "market open", "market close",
    "trading halt", "press conference", "scheduled", "annual meeting",
    "investor day", "conference call", "webcast", "to report earnings",
    "set to report", "quarterly report due",
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


def is_market_update(title: str) -> bool:
    """True for informational/scheduling headlines (Fed speeches, meeting
    minutes, market open/close, earnings-call scheduling, ...) that don't
    carry a directional signal of their own."""
    if not title:
        return False
    low = title.lower()
    return any(kw in low for kw in UPDATE_KEYWORDS)


def classify_news_bucket(title: str) -> str:
    """
    Sort a headline into exactly one of three display buckets:
      - "bullish" / "bearish": shown in Breaking Events
      - "update": shown in Market Updates instead -- covers both genuinely
        informational headlines (Fed speeches, calendar events, company
        announcements, market open/close) AND headlines with no clear
        keyword signal either way. Breaking Events never shows a meaningless
        "Neutral" tag; ambiguous headlines are treated as updates instead.
    """
    if is_market_update(title):
        return "update"
    sentiment = classify_headline_sentiment(title)
    if sentiment == "Positive":
        return "bullish"
    if sentiment == "Negative":
        return "bearish"
    return "update"


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
                "bucket": classify_news_bucket(title),
            })

    items.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return items[:limit]
