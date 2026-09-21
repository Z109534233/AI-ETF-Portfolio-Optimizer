"""
Market Intelligence Module
Aggregates market indices, headline sentiment, an AI/rule-based market
summary, a placeholder economic calendar, and portfolio-impact analysis for
pages/8_Market_Intelligence.py. Every public function is defensive: on any
data or API failure it returns a safe empty/placeholder value instead of
raising, so the page can never crash.
"""

from collections import Counter

import pandas as pd
import streamlit as st
import yfinance as yf

from src.openai_service import cached_generate, fingerprint, generate_text
from src.etf_database import get_etf, get_countries, get_tickers_by_country
from src.i18n import t, get_language, t_country, t_sector

# ── Market Indices ───────────────────────────────────────────────────────────
INDEX_TICKERS = {
    "sp500": ("mi_sp500", "^GSPC"),
    "nasdaq": ("mi_nasdaq", "^IXIC"),
    "dow": ("mi_dow", "^DJI"),
    "russell": ("mi_russell", "^RUT"),
    "vix": ("mi_vix", "^VIX"),
}

# sector (as stored on ETFRecord.sector, via src/etf_database.py) -> headline
# keywords used to match it. Keyed by *sector*, not by ticker, so any ETF
# added to etf_database.py -- in any country -- is automatically covered
# here as long as its sector matches one of these; unmapped sectors just
# default to "Neutral" (no crash, no hardcoded per-ticker/per-country list).
SECTOR_KEYWORDS = {
    "Technology": ["tech", "technology", "nasdaq", "chip", "semiconductor", "ai "],
    "Broad Market": ["s&p", "wall street", "stocks", "market"],
    "Gold": ["gold", "bullion", "precious metal"],
    "Bond": ["bond", "treasury", "yield", "rate", "fed", "interest rate"],
    "Dividend": ["dividend", "income", "payout"],
    "Financials": ["bank", "financial", "rate hike"],
    "Healthcare": ["health", "pharma", "biotech"],
    "Real Estate": ["real estate", "housing", "reit"],
    "Small Cap": ["small cap", "russell"],
}


def _default_global_watchlist() -> list:
    """
    Pick one representative ticker per supported country (never hardcoded
    to a specific country name) so the "Global ETFs" watch-list grows
    automatically as new markets are registered in src/etf_database.py.
    """
    watchlist = []
    for country in get_countries():
        tickers = get_tickers_by_country(country)
        if tickers:
            watchlist.append(tickers[0])
    return watchlist


# Default watch-list shown in the "Global ETFs" section: one ETF per
# supported country (built dynamically from etf_database.py, see above).
DEFAULT_WATCHLIST = _default_global_watchlist()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_indices() -> dict:
    """
    Fetch the latest price and day-over-day change for the major indices
    (S&P 500, NASDAQ, Dow Jones, Russell 2000, VIX) via yfinance.

    Returns a dict keyed by index id, each value a dict with "label",
    "available" (bool), and -- when available -- "price", "change",
    "change_pct". A per-index failure is isolated: one bad ticker never
    prevents the others from rendering.
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


def derive_market_direction(indices: dict) -> dict:
    """Summarize broad-equity direction from the live index snapshot.

    S&P 500, NASDAQ, Dow and Russell 2000 are the primary direction inputs;
    VIX and headline sentiment are deliberately excluded from this price
    direction. The result is descriptive of the fetched snapshot only.
    """
    keys = ("sp500", "nasdaq", "dow", "russell")
    changes = [
        float(indices[key]["change_pct"])
        for key in keys
        if indices.get(key, {}).get("available")
        and indices[key].get("change_pct") is not None
    ]
    if not changes:
        return {
            "available": False, "direction": "Unknown", "average_change_pct": None,
            "positive_count": 0, "negative_count": 0, "count": 0,
        }

    positive = sum(change > 0 for change in changes)
    negative = sum(change < 0 for change in changes)
    average = sum(changes) / len(changes)
    majority_needed = max(1, (len(changes) // 2) + 1)

    if positive >= majority_needed and average > 0:
        direction = "Higher"
    elif negative >= majority_needed and average < 0:
        direction = "Lower"
    else:
        direction = "Mixed"

    return {
        "available": True,
        "direction": direction,
        "average_change_pct": average,
        "positive_count": positive,
        "negative_count": negative,
        "count": len(changes),
    }


def derive_vix_regime(indices: dict) -> dict:
    """Classify VIX level and daily direction from the live index snapshot."""
    vix = indices.get("vix", {})
    if not vix.get("available") or vix.get("price") is None:
        return {"available": False, "level": "Unknown", "trend": "Unknown"}

    price = float(vix["price"])
    change_pct = float(vix.get("change_pct") or 0.0)
    if price < 15:
        level = "Low"
    elif price < 20:
        level = "Moderate"
    elif price < 30:
        level = "Elevated"
    else:
        level = "High"

    if change_pct > 0:
        trend = "Rising"
    elif change_pct < 0:
        trend = "Falling"
    else:
        trend = "Flat"

    return {
        "available": True,
        "price": price,
        "change_pct": change_pct,
        "level": level,
        "trend": trend,
    }


def sector_label(sector_key: str) -> str:
    """Translate an internal sector key (e.g. 'technology') to display text."""
    return t(f"mi_sector_{sector_key}")


def impact_label(impact: str) -> str:
    """Translate an internal impact value ('Positive'/'Negative'/'Neutral')."""
    return t(f"mi_impact_{impact.lower()}_label")


def _stars_for_impact(impact: str) -> int:
    """
    A simple, transparent star rating derived from an ETF's headline impact:
    Positive/Negative both indicate the ETF is topically relevant to today's
    news (4 stars); Neutral (no matching headline) is the baseline (2 stars).
    Not a predictive score -- purely a visual indicator of relevance.
    """
    return 4 if impact in ("Positive", "Negative") else 2


def stars_to_impact_label(stars: int) -> str:
    """Translate a 0-5 star rating to its display label, e.g. 'High Impact'
    (never just a bare star count)."""
    return t(f"mi_impact_stars_{stars}")


# event type -> headline keywords used to *classify what kind of event* a
# headline is (NOT which country it names). This is deliberately separate
# from "does the headline mention Taiwan" -- a headline never has to name a
# market for that market to be affected by it (e.g. a US tariff headline
# that never says "Taiwan" still hits Taiwan hard via its semiconductor
# supply chain, see EVENT_MARKET_IMPACT below).
EVENT_TYPE_KEYWORDS = {
    "tariff": ["tariff", "tariffs", "trade war", "export ban", "import ban", "trade restriction"],
    "interest_rate": ["interest rate", "rate cut", "rate hike", "fed decision", "fomc",
                       "rate decision", "central bank", "cuts rate", "raises rate", "hikes rate"],
    "semiconductor": ["semiconductor", "chip ", "chips", "chipmaker", "foundry", "wafer"],
    "oil": ["oil", "crude", "opec", "energy price", "energy prices"],
    "inflation": ["inflation", "cpi", "ppi", "consumer price", "producer price"],
    "employment": ["jobs report", "unemployment", "payroll", "non-farm", "labor market",
                    "job cuts", "layoffs"],
    "gdp": ["gdp", "economic growth", "recession", "economic output"],
    "pmi": ["pmi", "manufacturing index", "purchasing managers"],
    "geopolitical": ["geopolitical", "conflict", "war", "sanctions", "invasion"],
    "earnings": ["earnings", "quarterly results", "eps", "revenue beat", "profit report",
                 "beats estimates", "misses estimates", "beats expectations", "misses expectations"],
}

# event type -> how much weight a single headline of that type carries in
# the AI Market Sentiment engine below. Major macro events (Fed, tariffs,
# geopolitics, ...) are deliberately weighted far above routine company
# news, so one Fed headline can outweigh several minor stories instead of
# every headline just being counted equally.
EVENT_IMPORTANCE = {
    "interest_rate": 5,   # Fed / central bank decisions
    "tariff": 5,
    "geopolitical": 5,
    "inflation": 4,
    "gdp": 4,
    "employment": 4,
    "semiconductor": 4,
    "oil": 3,
    "earnings": 3,
    "pmi": 3,
}
# Headlines that don't match any known event type (generic company/market
# chatter) get this baseline weight instead.
DEFAULT_EVENT_WEIGHT = 1

# event type -> {market: impact weight, 1 (Very Low) - 5 (Very High)}. This
# is the actual "Event -> Impact Mapping": a market's rating comes from
# what KIND of event happened, not whether the headline named that market.
# E.g. "tariff" events hit Taiwan hard (semiconductor supply chain) even
# though most tariff headlines only ever mention the US. Markets not listed
# for an event type simply aren't pushed by that event. Keyed by the same
# country names registered in src/etf_database.py -- adding a market there
# plus an entry per relevant event type here is all it takes to extend
# coverage; nothing else needs to change.
EVENT_MARKET_IMPACT = {
    "tariff": {"United States": 5, "Taiwan": 5, "United Kingdom": 2},
    "interest_rate": {"United States": 5, "United Kingdom": 4, "Taiwan": 3},
    "semiconductor": {"Taiwan": 5, "United States": 4, "United Kingdom": 1},
    "oil": {"United States": 4, "United Kingdom": 4, "Taiwan": 2},
    "inflation": {"United States": 5, "United Kingdom": 4, "Taiwan": 3},
    "employment": {"United States": 5, "United Kingdom": 2, "Taiwan": 2},
    "gdp": {"United States": 4, "Taiwan": 3, "United Kingdom": 3},
    "pmi": {"Taiwan": 4, "United States": 3, "United Kingdom": 2},
    "geopolitical": {"United States": 4, "Taiwan": 4, "United Kingdom": 3},
}


def classify_event_types(title: str) -> list:
    """Classify a headline into zero or more event-type keys (a headline
    can be more than one type at once, e.g. "tariffs on semiconductors" is
    both "tariff" and "semiconductor")."""
    if not title:
        return []
    low = title.lower()
    return [event for event, keywords in EVENT_TYPE_KEYWORDS.items() if any(kw in low for kw in keywords)]


# Event Classification: a headline -> exactly one high-level category, for
# a different purpose than EVENT_TYPE_KEYWORDS/classify_event_types() above
# (which returns zero or more internal keys used by Affected Markets / AI
# Market Sentiment). Checked in order below, first match wins -- more
# specific/company-identifying keywords are listed before broader ones, so
# e.g. "Nvidia AI chips" resolves to Technology (a named-company keyword)
# rather than the broader Semiconductor or AI categories, and "TSMC
# earnings" resolves to Semiconductor via the company name even though the
# headline never says the word "semiconductor".
EVENT_CLASSIFICATION_KEYWORDS = [
    ("Cryptocurrency", ["bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain",
                         "btc", "coinbase", "binance"]),
    ("Semiconductor", ["semiconductor", "tsmc", "chipmaker", "foundry", "wafer", "asml"]),
    ("Technology", ["nvidia", "apple", "microsoft", "google", "meta", "amazon",
                     "big tech", "tech stocks", "ai chip", "ai chips", "software"]),
    ("AI", ["artificial intelligence", "generative ai", "ai regulation", "ai adoption",
            "ai boom", "chatgpt", "openai"]),
    ("Interest Rate", ["interest rate", "rate cut", "rate hike", "rate decision",
                        "fed decision", "fed chair", "federal reserve", "fomc",
                        "central bank", "cuts rate", "raises rate", "hikes rate"]),
    ("Inflation", ["inflation", "cpi", "ppi", "consumer price", "producer price"]),
    ("Trade Policy", ["tariff", "tariffs", "trade war", "trade policy", "export ban",
                       "import ban", "trade restriction"]),
    ("Energy", ["oil", "crude", "opec", "energy price", "energy prices", "gas price", "natural gas"]),
    ("Geopolitics", ["geopolitical", "conflict", "war", "sanctions", "invasion",
                      "israel", "iran", "ukraine", "russia"]),
    ("Banking", ["bank", "banking", "lender", "deposit", "loan default"]),
    ("Economy", ["gdp", "economic growth", "recession", "economic output", "employment",
                 "jobs report", "unemployment", "payroll", "pmi", "manufacturing index"]),
]


def classify_event(headline: str) -> str:
    """
    Classify a single headline into exactly one high-level event category:
    Interest Rate, Inflation, Trade Policy, Technology, Semiconductor,
    Energy, Geopolitics, Economy, AI, Banking, Cryptocurrency, or "Other"
    when nothing matches. Checked in the fixed priority order of
    EVENT_CLASSIFICATION_KEYWORDS above (first match wins), so a headline
    that could plausibly fit more than one category resolves consistently.

    Single-word keywords (e.g. "war", "bank", "oil") are matched as whole
    words only, not as a bare substring -- otherwise a word like "award"
    would falsely match "war". Multi-word phrases (e.g. "cuts rate",
    "central bank") are matched as a substring of the full headline, since
    splitting them into separate words would lose the phrase's meaning.
    A transparent rule-based heuristic, not a prediction.
    """
    if not headline:
        return "Other"
    low = headline.lower()
    words = set(low.replace("-", " ").split())
    for category, keywords in EVENT_CLASSIFICATION_KEYWORDS:
        for kw in keywords:
            matched = (kw in low) if " " in kw else (kw in words)
            if matched:
                return category
    return "Other"


# ── Market Impact Score ──────────────────────────────────────────────────────
# category (from classify_event() above) -> how market-moving that *type*
# of event generally is, 0-100. This replaces "did the headline mention an
# ETF/market by name" with "how important is the underlying event" as the
# basis for the star rating.
EVENT_IMPORTANCE_SCORE = {
    "Interest Rate": 100,
    "Trade Policy": 100,
    "Geopolitics": 95,
    "Inflation": 85,
    "Economy": 80,
    "Semiconductor": 80,
    "AI": 65,
    "Technology": 60,
    "Energy": 60,
    "Banking": 55,
    "Cryptocurrency": 40,
    "Other": 20,
}

# category -> {country: impact weight, 1 (Low) - 5 (Very High)}. Component
# score is the strongest (max) weight among the countries our ETF universe
# covers, scaled to 0-100 (weight * 20). Categories not listed (e.g.
# "Other") fall back to a 20-point baseline.
EVENT_AFFECTED_MARKETS = {
    "Interest Rate": {"United States": 5, "United Kingdom": 4, "Taiwan": 3},
    "Trade Policy": {"United States": 5, "Taiwan": 5, "United Kingdom": 2},
    "Semiconductor": {"Taiwan": 5, "United States": 4, "United Kingdom": 1},
    "Technology": {"United States": 4, "Taiwan": 2, "United Kingdom": 2},
    "AI": {"United States": 4, "Taiwan": 3, "United Kingdom": 2},
    "Energy": {"United States": 4, "United Kingdom": 4, "Taiwan": 2},
    "Inflation": {"United States": 5, "United Kingdom": 4, "Taiwan": 3},
    "Geopolitics": {"United States": 4, "Taiwan": 4, "United Kingdom": 3},
    "Economy": {"United States": 4, "Taiwan": 3, "United Kingdom": 3},
    "Banking": {"United States": 4, "United Kingdom": 4, "Taiwan": 2},
    "Cryptocurrency": {"United States": 3, "United Kingdom": 2, "Taiwan": 2},
}

# category -> how broadly that event type tends to reach across industries
# / sectors (0-100), e.g. an interest-rate decision touches nearly every
# sector at once, while a single crypto headline stays fairly contained.
EVENT_INDUSTRY_BREADTH = {
    "Interest Rate": 90,
    "Inflation": 75,
    "Trade Policy": 80,
    "Semiconductor": 75,
    "Economy": 70,
    "Technology": 70,
    "AI": 65,
    "Banking": 60,
    "Energy": 60,
    "Geopolitics": 55,
    "Cryptocurrency": 30,
    "Other": 15,
}


COMMENTARY_MARKERS = (
    "opinion", "commentary", "column", "interview", "outlook",
    " says ", " argues ", " believes ", " thinks ", " predicts ",
    " expects ", " warns ", " view:", " views ",
)


def _headline_content_factor(item: dict) -> float:
    """Down-weight commentary/attribution headlines versus reported events.

    A headline mentioning the Fed is not automatically a Fed decision. Titles
    framed as opinion/commentary or a person's attributed view receive a 0.65
    factor. The factor is intentionally transparent and title-only because the
    news feed does not provide a reliable article-type taxonomy.
    """
    title = f" {str(item.get('title', '')).lower()} "
    return 0.65 if any(marker in title for marker in COMMENTARY_MARKERS) else 1.0


def _market_impact_score_to_stars(score: int) -> int:
    """0-100 Market Impact Score -> 1-5 star rating (never 0 stars for a
    valid score; 0 stars is reserved for "no news to score" below)."""
    if score >= 90:
        return 5
    if score >= 70:
        return 4
    if score >= 50:
        return 3
    if score >= 30:
        return 2
    return 1


def calculate_market_impact(news_items: list) -> dict:
    """Return a transparent 0-100 rule-based market-impact score.

    The base score blends event importance, affected-market breadth,
    affected-industry breadth and theme frequency. A final content-type
    factor prevents commentary/opinion headlines from receiving the same
    impact score as a reported policy decision merely because both contain
    words such as "Federal Reserve".

    If at least one headline in the dominant category is a direct/reporting
    headline, the category keeps a 1.0 factor. If every dominant-category
    headline is commentary/attributed opinion, the score is multiplied by
    0.65. This measures headline importance/relevance, not market direction.
    """
    empty_breakdown = {
        "event_importance": 0,
        "affected_markets": 0,
        "affected_industry": 0,
        "news_frequency": 0,
        "content_type_factor": 0.0,
    }
    if not news_items:
        return {
            "score": 0, "stars": 0, "star_label": "☆☆☆☆☆",
            "category": None, "breakdown": empty_breakdown,
        }

    categories = [classify_event(item["title"]) for item in news_items]
    category_counts = Counter(categories)
    present = [c for c in category_counts if c != "Other"] or list(category_counts)
    dominant = max(
        present,
        key=lambda c: (EVENT_IMPORTANCE_SCORE.get(c, 20), category_counts[c]),
    )

    event_importance = EVENT_IMPORTANCE_SCORE.get(dominant, 20)
    market_weights = EVENT_AFFECTED_MARKETS.get(dominant, {})
    affected_markets = max(market_weights.values()) * 20 if market_weights else 20
    affected_industry = EVENT_INDUSTRY_BREADTH.get(dominant, 15)
    news_frequency = min(
        100, round(category_counts[dominant] / len(news_items) * 100)
    )

    dominant_items = [
        item for item in news_items if classify_event(item["title"]) == dominant
    ]
    content_type_factor = max(
        (_headline_content_factor(item) for item in dominant_items),
        default=1.0,
    )

    base_score = (
        event_importance * 0.40
        + affected_markets * 0.25
        + affected_industry * 0.20
        + news_frequency * 0.15
    )
    score = round(base_score * content_type_factor)
    score = max(0, min(100, score))
    stars = _market_impact_score_to_stars(score)

    return {
        "score": score,
        "stars": stars,
        "star_label": "★" * stars + "☆" * (5 - stars),
        "category": dominant,
        "breakdown": {
            "event_importance": event_importance,
            "affected_markets": affected_markets,
            "affected_industry": affected_industry,
            "news_frequency": news_frequency,
            "content_type_factor": content_type_factor,
        },
    }


# ── ETF Impact (Event -> Event Type -> ETF Type -> Impact) ──────────────────
# ticker -> its ETF Type tag(s). An ETF can carry more than one tag (e.g.
# SOXX is both a Technology fund and a pure-play Semiconductor fund), in
# which case its rating is the strongest of its tags' ratings for the
# day's dominant event. VOO (US-domiciled) and VUSA (UCITS/European
# feeder) both track the S&P 500 but are kept as distinct tags, since a
# UCITS feeder fund is one step removed from direct US market pricing.
ETF_TYPE_MAP = {
    "BND": ["Bond"],
    "TLT": ["Bond"],
    "QQQ": ["Technology"],
    "SOXX": ["Technology", "Semiconductor"],
    "0050": ["Semiconductor", "Broad Market"],
    "VOO": ["Broad Market"],
    "VUSA": ["Broad Market (UCITS)"],
}

# The default watch-list for calculate_etf_impact() when no tickers list is
# supplied -- every ticker currently registered in ETF_TYPE_MAP above.
DEFAULT_ETF_IMPACT_WATCHLIST = list(ETF_TYPE_MAP.keys())

# event category (from classify_event()) -> {ETF type: impact weight, 1
# (Low) - 5 (Very High)}. ETF types not listed for a category fall back to
# a 1-star baseline. This is the "ETF Type -> Impact" stage of the
# pipeline; which ETF types apply to a given ticker comes from
# ETF_TYPE_MAP above.
EVENT_TYPE_TO_ETF_TYPE_IMPACT = {
    "Interest Rate": {"Bond": 5, "Technology": 3, "Semiconductor": 3,
                       "Broad Market": 3, "Broad Market (UCITS)": 3},
    "Inflation": {"Bond": 4, "Technology": 3, "Semiconductor": 3,
                  "Broad Market": 3, "Broad Market (UCITS)": 3},
    "Trade Policy": {"Semiconductor": 4, "Technology": 4, "Broad Market": 4,
                      "Broad Market (UCITS)": 3, "Bond": 1},
    "Technology": {"Technology": 5, "Semiconductor": 4, "Broad Market": 3,
                   "Broad Market (UCITS)": 2, "Bond": 1},
    "Semiconductor": {"Semiconductor": 5, "Technology": 4, "Broad Market": 3,
                       "Broad Market (UCITS)": 2, "Bond": 1},
    "AI": {"Technology": 5, "Semiconductor": 4, "Broad Market": 3,
           "Broad Market (UCITS)": 2, "Bond": 1},
    "Energy": {"Broad Market": 2, "Broad Market (UCITS)": 2, "Bond": 1,
               "Technology": 1, "Semiconductor": 1},
    "Geopolitics": {"Bond": 3, "Broad Market": 3, "Broad Market (UCITS)": 3,
                     "Technology": 2, "Semiconductor": 2},
    "Economy": {"Broad Market": 3, "Broad Market (UCITS)": 3, "Bond": 2,
                "Technology": 2, "Semiconductor": 2},
    "Banking": {"Broad Market": 2, "Broad Market (UCITS)": 2, "Bond": 2,
                "Technology": 1, "Semiconductor": 1},
    "Cryptocurrency": {"Technology": 1, "Semiconductor": 1, "Broad Market": 1,
                        "Broad Market (UCITS)": 1, "Bond": 1},
    "Other": {},
}


def calculate_etf_impact(news_items: list, tickers: list = None) -> list:
    """
    Event -> Event Type -> ETF Type -> Impact pipeline for ETF star
    ratings. Replaces "did a headline mention this ETF/market by keyword"
    with a rule-based chain:
      1. Event -> Event Type: reuse calculate_market_impact()'s dominant
         category for today's news (classify_event() per headline, then
         the single most important category present wins) -- the exact
         same "what actually happened today" logic already used for the
         Market Impact Score, not a second, different classification.
      2. Event Type -> ETF Type -> Impact: EVENT_TYPE_TO_ETF_TYPE_IMPACT
         maps that category to how strongly each ETF TYPE (Bond,
         Technology, Semiconductor, Broad Market, Broad Market (UCITS)) is
         affected.
      3. ETF Type -> ETF: each watch-list ticker's own type tag(s)
         (ETF_TYPE_MAP) are looked up against that mapping; a ticker with
         more than one tag (e.g. SOXX = Technology + Semiconductor) takes
         the strongest applicable rating.
    A transparent rule-based heuristic, not a prediction.

    Returns a list of dicts: ticker, etf_types (the ticker's type tag(s)),
    stars (1-5), event_category (the dominant event category driving the
    rating for today, or None if there was no news to classify).
    """
    tickers = tickers or DEFAULT_ETF_IMPACT_WATCHLIST
    category = calculate_market_impact(news_items)["category"]
    type_impact = EVENT_TYPE_TO_ETF_TYPE_IMPACT.get(category, {})

    results = []
    for ticker in tickers:
        etf_types = ETF_TYPE_MAP.get(ticker, [])
        stars = max((type_impact.get(t, 1) for t in etf_types), default=1)
        results.append({
            "ticker": ticker,
            "etf_types": etf_types,
            "stars": stars,
            "event_category": category,
        })
    return results


_MOOD_KEY = {"Bullish": "mi_mood_bullish", "Neutral": "mi_mood_neutral", "Bearish": "mi_mood_bearish"}
_MOOD_VARIANT = {"Bullish": "green", "Neutral": "neutral", "Bearish": "red"}


def calculate_ai_market_sentiment(news_items: list) -> dict:
    """
    AI Market Sentiment -- a Rule-based Market Sentiment Engine, distinct
    from calculate_market_sentiment() (which stays untouched and simply
    counts headline keyword hits for the "News Sentiment Analysis" chart).
    This does NOT just count how many headlines are positive/negative. It
    combines three things for every headline:
      1. Event type (classify_event_types) -- Fed/rate decisions, tariffs,
         inflation, GDP, employment, geopolitics, oil, semiconductors,
         earnings, ...
      2. Event importance (EVENT_IMPORTANCE) -- a Fed decision or tariff
         headline counts far more than a routine company story
         (DEFAULT_EVENT_WEIGHT for anything unclassified)
      3. Each headline's directional read (its precomputed "impact" field,
         Positive/Negative/Neutral from src/news.py's keyword classifier)
    A single major macro headline can therefore outweigh several minor
    ones, unlike a flat headline count. Still a transparent heuristic, not
    a real ML/NLP model or a price prediction -- reserved as the entry
    point for a future LLM-based version.

    Returns: mood ("Bullish"/"Neutral"/"Bearish", raw), label (translated),
    variant (badge color), confidence (0-100), drivers (up to 3 headline
    titles that most influenced the result, highest-weight first).
    """
    if not news_items:
        return {"mood": "Neutral", "label": t("mi_mood_neutral"), "variant": "neutral",
                "confidence": 0, "drivers": []}

    weighted_sum = 0.0
    total_weight = 0.0
    drivers = []

    for item in news_items:
        event_types = classify_event_types(item["title"])
        weight = max((EVENT_IMPORTANCE.get(et, DEFAULT_EVENT_WEIGHT) for et in event_types),
                     default=DEFAULT_EVENT_WEIGHT)
        signed = {"Positive": 1, "Negative": -1, "Neutral": 0}.get(item.get("impact"), 0)
        weighted_sum += signed * weight
        total_weight += weight
        if event_types:
            drivers.append((item["title"], weight))

    avg = weighted_sum / total_weight if total_weight else 0.0

    if avg > 0.15:
        mood = "Bullish"
    elif avg < -0.15:
        mood = "Bearish"
    else:
        mood = "Neutral"

    # Confidence blends how one-sided the weighted signal is (abs(avg),
    # 0-1) into a 30-95% display range -- never a fake 0% or 100%.
    confidence = int(max(30, min(95, round(50 + abs(avg) * 50))))

    drivers.sort(key=lambda d: d[1], reverse=True)
    top_drivers = [d[0] for d in drivers[:3]]

    return {
        "mood": mood,
        "label": t(_MOOD_KEY[mood]),
        "variant": _MOOD_VARIANT[mood],
        "confidence": confidence,
        "drivers": top_drivers,
    }


def calculate_regional_market_sentiment(news_items: list) -> list:
    """
    Market Sentiment broken down per market (US / Taiwan / UK, via every
    country registered in src/etf_database.py) plus one "Global" row --
    each with a Bullish/Neutral/Bearish mood AND a Confidence percentage,
    not just the mood alone. This is a separate function from
    calculate_market_sentiment() (the existing headline-count donut chart
    powering "News Sentiment Analysis", left untouched) and from
    calculate_ai_market_sentiment() (a single global mood with no
    Confidence-per-market breakdown, also untouched).

    For each market, only headlines whose classified event type(s)
    (classify_event_types()) are known to affect that market
    (EVENT_MARKET_IMPACT) count toward it, weighted by that event type's
    impact weight on the market; each headline's own directional read
    (Positive/Negative/Neutral, from src/news.py's keyword classifier)
    pulls the weighted average toward Bullish or Bearish -- the same
    weighted-average approach as calculate_ai_market_sentiment(), just
    scoped per market instead of once globally. "Global" uses the
    strongest event-type weight found in ANY market per headline, i.e. how
    big the story is anywhere, not a sum across markets. Confidence blends
    how one-sided each market's weighted signal is into a 50-95% display
    range (never a fake 0% or 100%); markets with no relevant headlines
    today get "Neutral" at 0% confidence.

    Returns a list of dicts (one per market, in src/etf_database.py's
    registration order, then "Global" last): market (raw country name, or
    "Global"), market_label (translated), mood ("Bullish"/"Neutral"/
    "Bearish", raw), label (translated), variant (badge color), confidence
    (0-100).
    """
    markets = get_countries()

    def _weighted_score(weight_lookup):
        weighted_sum = 0.0
        total_weight = 0.0
        for item in news_items:
            weight = max((weight_lookup(et) for et in classify_event_types(item["title"])), default=0)
            if weight <= 0:
                continue
            signed = {"Positive": 1, "Negative": -1, "Neutral": 0}.get(item.get("impact"), 0)
            weighted_sum += signed * weight
            total_weight += weight
        return weighted_sum, total_weight

    def _mood_and_confidence(weighted_sum, total_weight):
        avg = weighted_sum / total_weight if total_weight else 0.0
        if avg > 0.15:
            mood = "Bullish"
        elif avg < -0.15:
            mood = "Bearish"
        else:
            mood = "Neutral"
        confidence = int(max(30, min(95, round(50 + abs(avg) * 50)))) if total_weight else 0
        return mood, confidence

    results = []
    for market in markets:
        weighted_sum, total_weight = _weighted_score(lambda et, m=market: EVENT_MARKET_IMPACT.get(et, {}).get(m, 0))
        mood, confidence = _mood_and_confidence(weighted_sum, total_weight)
        results.append({
            "market": market,
            "market_label": t_country(market),
            "mood": mood,
            "label": t(_MOOD_KEY[mood]),
            "variant": _MOOD_VARIANT[mood],
            "confidence": confidence,
        })

    weighted_sum, total_weight = _weighted_score(
        lambda et: max(EVENT_MARKET_IMPACT.get(et, {}).values(), default=0)
    )
    mood, confidence = _mood_and_confidence(weighted_sum, total_weight)
    results.append({
        "market": "Global",
        "market_label": "Global",
        "mood": mood,
        "label": t(_MOOD_KEY[mood]),
        "variant": _MOOD_VARIANT[mood],
        "confidence": confidence,
    })

    return results


def calculate_affected_markets(news_items: list) -> list:
    """
    Determine how strongly today's headlines may be affecting each
    supported market, via Event Type -> Market Impact (EVENT_MARKET_IMPACT)
    -- NOT whether the headline literally names that market. A headline is
    first classified into its event type(s) (classify_event_types), then
    each event type's mapped impact weight is applied to the markets it's
    known to affect. A market's rating is the strongest (max) impact weight
    reached across today's headlines; markets untouched by any classified
    event default to the "Very Low Impact" baseline so the section always
    renders consistently. This measures topical relevance/impact severity
    only -- it does not predict market direction.

    Returns a list of dicts: market (translated country name), country (raw
    country key, e.g. for a flag-emoji lookup), stars (1-5), impact_label
    (translated, e.g. "High Impact" -- never bare stars), affected_by (up
    to 2 headline titles whose event type drove the rating), reasons (up
    to 3 short display tags -- classify_event() run on each of this
    market's driving headlines, deduplicated, "Other" excluded -- for a
    "Reasons" tag display distinct from the full-headline affected_by
    list; does not change stars/impact_label, purely additive).
    """
    countries = get_countries()
    market_stars = {c: 0 for c in countries}
    market_events = {c: [] for c in countries}

    for item in news_items:
        for event_type in classify_event_types(item["title"]):
            for market, weight in EVENT_MARKET_IMPACT.get(event_type, {}).items():
                if market not in market_stars:
                    continue
                market_stars[market] = max(market_stars[market], weight)
                if item["title"] not in market_events[market]:
                    market_events[market].append(item["title"])

    results = []
    for country in countries:
        stars = market_stars[country] or 1  # baseline: no classified event today -> Very Low Impact
        driving_headlines = market_events[country]
        reasons = []
        for title in driving_headlines:
            tag = classify_event(title)
            if tag != "Other" and tag not in reasons:
                reasons.append(tag)
        results.append({
            "market": t_country(country),
            "country": country,
            "stars": stars,
            "impact_label": stars_to_impact_label(stars),
            "affected_by": driving_headlines[:2],
            "reasons": reasons[:3],
        })
    results.sort(key=lambda m: m["stars"], reverse=True)
    return results


def get_affected_etfs(news_items: list, tickers: list = None) -> list:
    """
    Heuristically determine which watch-list ETFs today's headlines might
    affect, by matching each ETF's sector keywords (SECTOR_KEYWORDS, keyed by
    the ETF's sector from src/etf_database.py) against headline titles. A
    matched ETF's impact is the majority vote of classify_headline_sentiment()
    across its matching headlines; ETFs with no matching headline default to
    "Neutral". This is a transparent keyword heuristic, not a prediction, and
    works for any country registered in etf_database.py -- nothing here is
    hardcoded to a specific market.

    Returns a list of dicts: ticker, sector (translated), country (translated),
    impact (raw Positive/Negative/Neutral, for logic), impact_label
    (translated), stars (0-5, for display only).
    """
    tickers = tickers or DEFAULT_WATCHLIST
    results = []
    for ticker in tickers:
        record = get_etf(ticker)
        sector_raw = record.sector if record else None
        country_raw = record.country if record else None
        keywords = SECTOR_KEYWORDS.get(sector_raw, [])

        matches = [n for n in news_items if any(kw in n["title"].lower() for kw in keywords)]
        if matches:
            counts = Counter(n["impact"] for n in matches)
            impact = counts.most_common(1)[0][0]
        else:
            impact = "Neutral"

        results.append({
            "ticker": ticker,
            "sector": t_sector(sector_raw) if sector_raw else t("mi_sector_other"),
            "country": t_country(country_raw) if country_raw else t("hist_region_unknown"),
            "impact": impact,
            "impact_label": impact_label(impact),
            "stars": _stars_for_impact(impact),
        })
    return results


def calculate_market_sentiment(news_items: list) -> dict:
    """
    Aggregate Bullish/Neutral/Bearish percentages across today's headlines
    using the same keyword classifier as get_affected_etfs(). Returns
    "available": False (with all percentages at 0) when there is no news to
    analyze, so the page can show a clean empty state instead of a fake split.
    """
    if not news_items:
        return {
            "available": False,
            "bullish_pct": 0.0, "neutral_pct": 0.0, "bearish_pct": 0.0,
            "bullish_count": 0, "neutral_count": 0, "bearish_count": 0,
        }

    counts = Counter(n["impact"] for n in news_items)
    total = len(news_items)
    bullish, bearish, neutral = counts.get("Positive", 0), counts.get("Negative", 0), counts.get("Neutral", 0)

    return {
        "available": True,
        "bullish_pct": round(bullish / total * 100, 1),
        "neutral_pct": round(neutral / total * 100, 1),
        "bearish_pct": round(bearish / total * 100, 1),
        "bullish_count": bullish, "neutral_count": neutral, "bearish_count": bearish,
    }


def get_economic_calendar() -> list:
    """Illustrative macro-event list; no verified live schedule is connected.

    The page may use this to explain what kinds of releases matter, but no
    item is described as "this week" or "upcoming" because the app has not
    verified an actual release/FOMC calendar.
    """
    sample_when = t("mi_cal_example_timing")
    return [
        {"event": "CPI (Consumer Price Index)", "when": sample_when, "importance": t("mi_cal_high")},
        {"event": "FOMC / Fed Interest Rate Decision", "when": sample_when, "importance": t("mi_cal_high")},
        {"event": "PPI (Producer Price Index)", "when": sample_when, "importance": t("mi_cal_medium")},
        {"event": "Non-Farm Payrolls (NFP)", "when": sample_when, "importance": t("mi_cal_high")},
        {"event": "GDP Growth Rate", "when": sample_when, "importance": t("mi_cal_medium")},
        {"event": "Fed Chair Speech", "when": sample_when, "importance": t("mi_cal_medium")},
    ]


_MARKET_SUMMARY_SYSTEM_INSTRUCTIONS = (
    "You are a professional, educational market analyst. Summarize ONLY the "
    "headlines and classifications given to you -- never invent a fact, "
    "statistic, or number not present in the input. Never give personalised "
    "investment advice or price predictions."
)


def generate_market_summary(news_items: list, sentiment: dict, affected_etfs: list,
                             session_state=None, indices: dict = None) -> dict:
    """
    Generate a ~100-200 word "Today's Market Summary" from current headlines
    and the app's OWN deterministic classifications (sentiment split,
    affected ETFs -- both computed elsewhere and passed in, never
    recomputed or invented by the model). Uses the central OpenAI Responses
    API service (src.openai_service) when a key is configured, otherwise a
    rule-based summary built from the exact same data. Never raises --
    always returns a display-safe dict, including when there is no news.

    Returns {"text": str, "source": "ai" | "rule_based"}. When
    `session_state` is given, the OpenAI call is cached by a fingerprint of
    the headlines/classifications so an unrelated rerun never re-spends a
    call for the same day's data.
    """
    if not news_items:
        return {"text": t("mi_summary_no_news"), "source": "rule_based"}

    prompt = _market_summary_prompt(news_items, sentiment, affected_etfs, indices=indices)

    if session_state is not None:
        fp = fingerprint(
            get_language(),
            tuple(n["title"] for n in news_items[:8]),
            sentiment.get("bullish_pct"), sentiment.get("neutral_pct"), sentiment.get("bearish_pct"),
            tuple(e["ticker"] for e in affected_etfs),
            tuple(
                (key, indices.get(key, {}).get("price"), indices.get(key, {}).get("change_pct"))
                for key in ("sp500", "nasdaq", "dow", "russell", "vix")
            ) if indices else (),
        )
        result = cached_generate(session_state, "_mi_openai_summary_cache", fp,
                                  _MARKET_SUMMARY_SYSTEM_INSTRUCTIONS, prompt, max_output_tokens=350)
    else:
        result = generate_text(_MARKET_SUMMARY_SYSTEM_INSTRUCTIONS, prompt, max_output_tokens=350)

    if result["available"]:
        return {"text": result["text"], "source": "ai"}
    return {
        "text": _generate_rule_based_summary(
            news_items, sentiment, affected_etfs, indices=indices
        ),
        "source": "rule_based",
    }


def _market_summary_prompt(
    news_items: list,
    sentiment: dict,
    affected_etfs: list,
    indices: dict = None,
) -> str:
    headlines = "\n".join(f"- {n['title']}" for n in news_items[:8])
    affected_str = ", ".join(
        f"{e['ticker']} ({e['sector']}: {e['impact_label']})"
        for e in affected_etfs
    ) or "N/A"
    market_state = derive_market_direction(indices or {})
    vix_state = derive_vix_regime(indices or {})
    market_snapshot = (
        f"Broad-equity direction: {market_state['direction']}; "
        f"average change: {market_state['average_change_pct']:.2f}%; "
        f"{market_state['positive_count']}/{market_state['count']} indices positive."
        if market_state["available"]
        else "Broad-equity direction: unavailable."
    )
    vix_snapshot = (
        f"VIX: {vix_state['price']:.2f}, {vix_state['change_pct']:+.2f}%, "
        f"level={vix_state['level']}, trend={vix_state['trend']}."
        if vix_state["available"]
        else "VIX: unavailable."
    )
    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文). Headlines "
        "are English-language source material and must stay source-faithful "
        "-- never translate or rewrite a headline's title as if it were the "
        "original; when you quote or reference a headline's title verbatim, "
        "wrap it in Chinese quotation marks 「」."
        if get_language() == "zh-TW" else "Respond entirely in English."
    )
    return f"""Based on the app-computed data below, write a Today's Market Summary
of about 100-200 words. MARKET DIRECTION MUST FOLLOW THE ACTUAL BROAD-EQUITY
INDEX SNAPSHOT when it is available. Headline sentiment is only supplementary
context and must not override or contradict the observed index direction.
Any volatility statement must follow the supplied VIX level/trend; do not call
volatility elevated when VIX is low/falling. Do not infer an upcoming Fed
decision or economic release from the illustrative calendar. Do not predict
future prices or give personalised investment advice. {language_instruction}

Market price snapshot:
- {market_snapshot}
- {vix_snapshot}

Headlines:
{headlines}

Headline sentiment split (supplementary): {sentiment['bullish_pct']}% bullish,
{sentiment['neutral_pct']}% neutral, {sentiment['bearish_pct']}% bearish.
Potentially affected ETFs: {affected_str}
"""


def _generate_rule_based_summary(
    news_items: list,
    sentiment: dict,
    affected_etfs: list,
    indices: dict = None,
) -> str:
    top_titles = [n["title"] for n in news_items[:3]]
    market_state = derive_market_direction(indices or {})
    vix_state = derive_vix_regime(indices or {})

    if market_state["available"]:
        direction_key = {
            "Higher": "mi_price_direction_higher",
            "Lower": "mi_price_direction_lower",
            "Mixed": "mi_price_direction_mixed",
        }[market_state["direction"]]
        parts = [t(
            "mi_summary_price_direction",
            direction=t(direction_key),
            positive=market_state["positive_count"],
            total=market_state["count"],
            average=market_state["average_change_pct"],
        )]
    else:
        parts = [t("mi_summary_price_unavailable")]

    if top_titles:
        parts.append(t("mi_summary_intro", headline=top_titles[0]))
    if len(top_titles) > 1:
        if get_language() == "zh-TW":
            quoted_headlines = "、".join(f"「{h}」" for h in top_titles[1:])
        else:
            quoted_headlines = "; ".join(f"\"{h}\"" for h in top_titles[1:])
        parts.append(t("mi_summary_more_headlines", headlines=quoted_headlines))

    if sentiment["bullish_pct"] > sentiment["bearish_pct"] + 10:
        tilt = t("mi_tilt_bullish")
    elif sentiment["bearish_pct"] > sentiment["bullish_pct"] + 10:
        tilt = t("mi_tilt_bearish")
    else:
        tilt = t("mi_tilt_mixed")
    parts.append(t(
        "mi_summary_sentiment_supplement",
        tilt=tilt,
        bullish=sentiment["bullish_pct"],
        bearish=sentiment["bearish_pct"],
    ))

    if vix_state["available"]:
        parts.append(t(
            "mi_summary_vix_state",
            level=t(f"mi_vix_level_{vix_state['level'].lower()}"),
            trend=t(f"mi_vix_trend_{vix_state['trend'].lower()}"),
            value=vix_state["price"],
            change=vix_state["change_pct"],
        ))

    positive_etfs = [
        e["ticker"] for e in affected_etfs if e["impact"] == "Positive"
    ]
    negative_etfs = [
        e["ticker"] for e in affected_etfs if e["impact"] == "Negative"
    ]
    if positive_etfs:
        parts.append(t("mi_summary_positive_etfs", tickers=", ".join(positive_etfs)))
    if negative_etfs:
        parts.append(t("mi_summary_negative_etfs", tickers=", ".join(negative_etfs)))
    parts.append(t("mi_summary_disclaimer_note"))
    return " ".join(parts)


def analyze_portfolio_impact(holdings: dict, affected_etfs: list) -> str:
    """
    Given a saved portfolio's holdings ({ticker: weight}) and the output of
    get_affected_etfs(), produce a short rule-based narrative describing the
    portfolio's dominant sector exposure and which held tickers today's news
    may be relevant to. This analyzes possible relevance only -- it never
    predicts prices or returns.
    """
    if not holdings:
        return t("mi_portfolio_no_data")

    impact_by_ticker = {e["ticker"]: e for e in affected_etfs}
    sector_weight = {}
    for ticker, weight in holdings.items():
        record = get_etf(ticker)
        label = t_sector(record.sector) if record else t("mi_sector_other")
        sector_weight[label] = sector_weight.get(label, 0) + weight

    if not sector_weight:
        return t("mi_portfolio_no_data")

    top_sector, top_weight = max(sector_weight.items(), key=lambda x: x[1])
    lines = [t("mi_portfolio_exposure", sector=top_sector, weight=f"{top_weight:.0%}")]

    relevant = [
        (tk, impact_by_ticker[tk]) for tk in holdings
        if tk in impact_by_ticker and impact_by_ticker[tk]["impact"] != "Neutral"
    ]
    if relevant:
        for tk, info in relevant:
            verb = t("mi_impact_positive_verb") if info["impact"] == "Positive" else t("mi_impact_negative_verb")
            lines.append(t("mi_portfolio_news_line", ticker=tk, verb=verb))
    else:
        lines.append(t("mi_portfolio_no_relevant_news"))

    return " ".join(lines)


# category (from classify_event()) -> a short display phrase per language,
# used only by generate_today_ai_summary() below to compose a natural
# sentence. Kept local to this function's neighborhood (not added to
# src/i18n.py) since each entry is a short narrative fragment specific to
# this one summary sentence, not general app UI copy.
_CATEGORY_PHRASE = {
    "Interest Rate": ("interest rate", "利率"),
    "Inflation": ("inflation", "通膨"),
    "Trade Policy": ("trade policy", "貿易政策"),
    "Technology": ("technology", "科技"),
    "Semiconductor": ("semiconductor", "半導體"),
    "Energy": ("energy", "能源"),
    "Geopolitics": ("geopolitical", "地緣政治"),
    "Economy": ("economic", "總體經濟"),
    "AI": ("AI", "AI"),
    "Banking": ("banking", "銀行業"),
    "Cryptocurrency": ("cryptocurrency", "加密貨幣"),
    "Other": ("general market", "一般市場"),
}
_MOOD_PHRASE_EN = {"Bullish": "trending higher", "Bearish": "under pressure", "Neutral": "holding steady"}
_MOOD_PHRASE_ZH = {"Bullish": "走揚", "Bearish": "承壓", "Neutral": "持穩"}


# ETF Type -> a short "region + type" descriptive phrase per language,
# used only by generate_today_ai_summary()'s paragraph 2 ("Why markets
# moved today") to compare the day's most-affected ETF type against the
# calmer Broad Market baseline, e.g. "Taiwan semiconductor" vs "US broad
# market". Kept local here, not added to src/i18n.py, for the same reason
# as the other narrative-fragment dicts above.
_ETF_TYPE_REGION_EN = {
    "Bond": "US bond",
    "Technology": "US technology",
    "Semiconductor": "Taiwan semiconductor",
    "Broad Market": "US broad market",
    "Broad Market (UCITS)": "UCITS broad market",
}
_ETF_TYPE_REGION_ZH = {
    "Bond": "美國債券型",
    "Technology": "美國科技型",
    "Semiconductor": "台灣半導體",
    "Broad Market": "美國大盤型",
    "Broad Market (UCITS)": "UCITS 大盤型",
}

# mood -> a forward-looking "Investment Insight" phrase per language, used
# only by generate_today_ai_summary()'s paragraph 3.
_INSIGHT_PHRASE_EN = {
    "Bearish": "Bond ETFs may benefit if investors move towards defensive assets.",
    "Bullish": "Growth and technology ETFs may continue to benefit if this momentum holds.",
    "Neutral": "With mixed signals today, diversified broad-market exposure may be the steadier choice.",
}
_INSIGHT_PHRASE_ZH = {
    "Bearish": "若投資人轉向防禦性資產，債券型 ETF 可能因此受惠。",
    "Bullish": "若這股動能持續，成長型與科技型 ETF 可能持續受惠。",
    "Neutral": "今天訊號較為分歧，分散配置的大盤型 ETF 可能是相對穩健的選擇。",
}


def generate_today_ai_summary(news_items: list, indices: dict = None) -> dict:
    """Price-first top-of-page rule-based market summary.

    Broad-equity direction comes from the live S&P 500/NASDAQ/Dow/Russell
    snapshot whenever available. Headline sentiment is supplementary only.
    VIX language is derived from its actual level and daily direction.
    """
    lang = get_language()
    title = "今日市場總覽" if lang == "zh-TW" else "Today's Market Overview"

    if not news_items and not derive_market_direction(indices or {})["available"]:
        no_data = ("目前沒有足夠的市場價格與新聞資料可產生今日摘要。" if lang == "zh-TW"
                   else "Not enough market-price or news data is available to generate today's summary.")
        heading = "今日摘要" if lang == "zh-TW" else "Summary"
        return {"title": title, "sections": [{"heading": heading, "text": no_data}], "disclaimer": ""}

    impact = calculate_market_impact(news_items) if news_items else {"category": None}
    category = impact.get("category") or "Other"
    category_en, category_zh = _CATEGORY_PHRASE.get(category, _CATEGORY_PHRASE["Other"])
    market_state = derive_market_direction(indices or {})
    vix_state = derive_vix_regime(indices or {})

    if market_state["available"]:
        if lang == "zh-TW":
            direction = {"Higher": "走揚", "Lower": "承壓", "Mixed": "漲跌互見"}[market_state["direction"]]
            overview_text = (
                f"今天主要股指{direction}：{market_state['positive_count']}/{market_state['count']} "
                f"個可用股指上漲，平均變動 {market_state['average_change_pct']:+.2f}%。"
            )
        else:
            direction = {"Higher": "moved higher", "Lower": "moved lower", "Mixed": "were mixed"}[market_state["direction"]]
            overview_text = (
                f"Major equity indices {direction}: {market_state['positive_count']}/{market_state['count']} "
                f"available indices rose, with an average move of {market_state['average_change_pct']:+.2f}%."
            )
    else:
        if lang == "zh-TW":
            overview_text = f"主要股指即時方向目前無法取得；新聞主題集中於{category_zh}。"
        else:
            overview_text = f"Live broad-equity direction is unavailable; headlines are currently focused on {category_en}."

    if vix_state["available"]:
        if lang == "zh-TW":
            vix_text = (
                f"VIX 為 {vix_state['price']:.2f}，屬於"
                f"{t('mi_vix_level_' + vix_state['level'].lower())}水準，"
                f"當日{t('mi_vix_trend_' + vix_state['trend'].lower())} "
                f"({vix_state['change_pct']:+.2f}%)。"
            )
        else:
            vix_text = (
                f"VIX is {vix_state['price']:.2f}, a "
                f"{t('mi_vix_level_' + vix_state['level'].lower())} level, and is "
                f"{t('mi_vix_trend_' + vix_state['trend'].lower())} "
                f"({vix_state['change_pct']:+.2f}%) today."
            )
    else:
        vix_text = ("VIX 即時資料目前無法取得。" if lang == "zh-TW"
                    else "Live VIX data is currently unavailable.")

    if news_items:
        sentiment_rows = calculate_regional_market_sentiment(news_items)
        global_sentiment = next((row for row in sentiment_rows if row["market"] == "Global"), None)
        mood = global_sentiment["mood"] if global_sentiment else "Neutral"
        if lang == "zh-TW":
            news_text = (
                f"新聞文字情緒為{t(_MOOD_KEY[mood])}，僅作為價格走勢之外的補充訊號；"
                f"目前主要新聞主題為{category_zh}。"
            )
        else:
            news_text = (
                f"Headline sentiment is {t(_MOOD_KEY[mood])} and is treated only as "
                f"supplementary context to observed price action; the leading news theme is {category_en}."
            )
    else:
        news_text = ("目前沒有足夠新聞可提供補充情緒訊號。" if lang == "zh-TW"
                     else "Not enough news is available for a supplementary sentiment read.")

    headings = (
        ("今日市場概況", "波動度觀察", "新聞補充訊號")
        if lang == "zh-TW"
        else ("Today's Market Overview", "Volatility Check", "Headline Context")
    )
    disclaimer = (
        "此摘要以實際指數與 VIX 快照為主要依據，新聞情緒僅作補充；規則式生成，僅供教育用途。"
        if lang == "zh-TW"
        else "This rule-based summary prioritizes observed index/VIX data; headline sentiment is supplementary and the content is educational only."
    )
    return {
        "title": title,
        "sections": [
            {"heading": headings[0], "text": overview_text},
            {"heading": headings[1], "text": vix_text},
            {"heading": headings[2], "text": news_text},
        ],
        "disclaimer": disclaimer,
    }


# ETF Type (from ETF_TYPE_MAP) -> a "why this matters to this ETF" phrase
# per language, used only by generate_explanation() below. Kept local
# here (not added to src/i18n.py) since these are narrative fragments for
# one specific explanation sentence, not general app UI copy.
_ETF_TYPE_IMPACT_PHRASE_EN = {
    "Bond": "bond ETFs are expected to see notable price movement.",
    "Technology": "technology ETFs may experience increased volatility.",
    "Semiconductor": "semiconductor ETFs are expected to face higher pressure.",
    "Broad Market": "broad market ETFs may see moderate, portfolio-wide impact.",
    "Broad Market (UCITS)": "UCITS broad-market ETFs may see a smaller, delayed reaction.",
}
_ETF_TYPE_IMPACT_PHRASE_ZH = {
    "Bond": "債券型 ETF 預期會有明顯的價格波動。",
    "Technology": "科技型 ETF 可能出現較大的波動。",
    "Semiconductor": "半導體 ETF 預期將面臨較高的壓力。",
    "Broad Market": "大盤型 ETF 可能受到中度、全面性的影響。",
    "Broad Market (UCITS)": "UCITS 大盤型 ETF 反應可能較小、較延遲。",
}

# event category (from classify_event()) -> a broader "ripple effect"
# phrase per language, used only by generate_explanation() below.
_CATEGORY_RIPPLE_EN = {
    "Interest Rate": "Rate-sensitive sectors across the board may see notable price swings.",
    "Inflation": "Consumer-facing and rate-sensitive sectors may see added pressure.",
    "Trade Policy": "Technology stocks may experience increased volatility.",
    "Semiconductor": "Technology stocks may experience increased volatility.",
    "Technology": "Broader tech-sector sentiment may shift alongside this news.",
    "AI": "AI-related stocks across sectors may see heightened trading activity.",
    "Energy": "Energy-sensitive sectors and transportation stocks may see cost pressure.",
    "Geopolitics": "Risk-off sentiment may spread across global equities.",
    "Economy": "Broad market sentiment may shift with the economic outlook.",
    "Banking": "Financial-sector stocks may see added scrutiny.",
    "Cryptocurrency": "Risk-asset sentiment may see a knock-on effect.",
    "Other": "Market reaction is likely to stay limited and short-lived.",
}
_CATEGORY_RIPPLE_ZH = {
    "Interest Rate": "所有利率敏感類股都可能出現明顯波動。",
    "Inflation": "消費相關與利率敏感類股可能承受額外壓力。",
    "Trade Policy": "科技股可能出現較大的波動。",
    "Semiconductor": "科技股可能出現較大的波動。",
    "Technology": "整體科技類股情緒可能隨這則新聞轉變。",
    "AI": "各產業中與 AI 相關的股票可能出現交易量增加的情況。",
    "Energy": "能源敏感產業與運輸類股可能面臨成本壓力。",
    "Geopolitics": "避險情緒可能擴散至全球股市。",
    "Economy": "整體市場情緒可能隨經濟展望改變。",
    "Banking": "金融類股可能受到更多關注與檢視。",
    "Cryptocurrency": "風險性資產的情緒可能受到連帶影響。",
    "Other": "市場反應預期將維持有限且短暫。",
}


def generate_explanation(news_items: list, ticker: str) -> dict:
    """
    "Why?" explanation for a single ETF's calculate_etf_impact() star
    rating, e.g.:
        ★★★★★
        Because:
        - Trump announced tariffs on semiconductors.
        - Taiwan semiconductor ETFs are expected to face higher pressure.
        - Technology stocks may experience increased volatility.
    Reuses the exact same Event -> Event Type -> ETF Type -> Impact chain
    as calculate_etf_impact() (does not re-derive the rating a second,
    different way) -- the star rating, dominant event category, and the
    ETF's own type tag(s) all come from calculate_etf_impact()/
    ETF_TYPE_MAP. Up to three reasons are generated:
      1. What happened -- the actual headline that drove today's dominant
         event category (the first one found; falls back to a generic
         line when none is available).
      2. Why it matters to this specific ETF, phrased around its own ETF
         Type tag(s) (_ETF_TYPE_IMPACT_PHRASE_EN/ZH), and the ETF's
         country when it's a known ticker (via get_etf(), e.g. "Taiwan
         semiconductor ETFs...").
      3. A broader ripple-effect statement for the event category as a
         whole (_CATEGORY_RIPPLE_EN/ZH).
    Template-based only, no LLM call. A transparent rule-based heuristic,
    not investment advice.

    Returns a dict: ticker, stars, star_label (e.g. "★★★★★"), reasons
    (list of up to 3 display-safe strings).
    """
    lang = get_language()
    etf_result = calculate_etf_impact(news_items, [ticker])[0]
    stars = etf_result["stars"]
    category = etf_result["event_category"]
    star_label = "★" * stars + "☆" * (5 - stars)

    if not category:
        no_data = "目前沒有足夠的新聞資料可以說明原因。" if lang == "zh-TW" else "Not enough news data is available to explain this rating."
        return {"ticker": ticker, "stars": stars, "star_label": star_label, "reasons": [no_data]}

    driving_headline = next(
        (item["title"] for item in news_items if classify_event(item["title"]) == category), None
    )

    reasons = []
    if driving_headline:
        reasons.append(driving_headline)

    record = get_etf(ticker)
    country_prefix = ""
    if record:
        country_prefix = (t_country(record.country) if lang == "zh-TW" else record.country) + (" " if lang != "zh-TW" else "")

    etf_types = etf_result["etf_types"]
    if etf_types:
        phrase_map = _ETF_TYPE_IMPACT_PHRASE_ZH if lang == "zh-TW" else _ETF_TYPE_IMPACT_PHRASE_EN
        top_type = max(etf_types, key=lambda et: EVENT_TYPE_TO_ETF_TYPE_IMPACT.get(category, {}).get(et, 0))
        type_phrase = phrase_map.get(top_type)
        if type_phrase:
            if lang == "zh-TW":
                reasons.append(f"{country_prefix}{type_phrase}")
            else:
                reasons.append(f"{country_prefix}{type_phrase[0].upper()}{type_phrase[1:]}")

    ripple_map = _CATEGORY_RIPPLE_ZH if lang == "zh-TW" else _CATEGORY_RIPPLE_EN
    ripple = ripple_map.get(category)
    if ripple:
        reasons.append(ripple)

    if not reasons:
        reasons.append("目前沒有足夠的新聞資料可以說明原因。" if lang == "zh-TW" else "Not enough news data is available to explain this rating.")

    return {"ticker": ticker, "stars": stars, "star_label": star_label, "reasons": reasons}


def get_todays_major_events(news_items: list, limit: int = 5) -> list:
    """
    "Today's Major Events": ranks today's individual headlines by their
    own Market Impact Score and returns the top `limit` (3-5 typically).
    Each headline is scored by treating it as if it were the day's only
    headline -- calculate_market_impact([item]), calculate_etf_impact([item]),
    and calculate_regional_market_sentiment([item]) are each called on a
    single-headline list, reusing exactly the same pipeline already used
    for the aggregate "today" versions rather than re-deriving scoring a
    second, different way. Headlines that classify_event() maps to
    "Other" (nothing recognizable) are excluded -- they aren't "major
    events". "Affected Markets" reuses classify_event_types()/
    EVENT_MARKET_IMPACT (the same data calculate_affected_markets() uses)
    to find which of our supported countries this headline's event
    type(s) hit at a "Moderate Impact" (weight >= 3) level or higher.
    "Affected ETFs" is every DEFAULT_ETF_IMPACT_WATCHLIST ticker whose
    calculate_etf_impact() rating for this single headline is 4+ stars.

    Returns a list of dicts, sorted by score descending: headline, score
    (0-100), stars (1-5), star_label (e.g. "★★★★★"), category, sentiment
    (raw mood), sentiment_label (translated), sentiment_variant (badge
    color), affected_markets (list of translated country names, may be
    empty), affected_etfs (list of tickers, may be empty).
    """
    events = []
    for item in news_items:
        impact = calculate_market_impact([item])
        category = impact["category"]
        if not category or category == "Other":
            continue

        sentiment_rows = calculate_regional_market_sentiment([item])
        global_row = next((s for s in sentiment_rows if s["market"] == "Global"), None)

        etf_rows = calculate_etf_impact([item], DEFAULT_ETF_IMPACT_WATCHLIST)
        affected_etfs = [e["ticker"] for e in etf_rows if e["stars"] >= 4]

        event_types = classify_event_types(item["title"])
        affected_markets = [
            t_country(m) for m in get_countries()
            if any(EVENT_MARKET_IMPACT.get(et, {}).get(m, 0) >= 3 for et in event_types)
        ]

        events.append({
            "headline": item["title"],
            "score": impact["score"],
            "stars": impact["stars"],
            "star_label": "★" * impact["stars"] + "☆" * (5 - impact["stars"]),
            "category": category,
            "sentiment": global_row["mood"] if global_row else "Neutral",
            "sentiment_label": global_row["label"] if global_row else t(_MOOD_KEY["Neutral"]),
            "sentiment_variant": global_row["variant"] if global_row else "neutral",
            "affected_markets": affected_markets,
            "affected_etfs": affected_etfs,
            "publisher": item.get("publisher") or "—",
            "published": item.get("published"),
            "link": item.get("link"),
            "content_type_factor": impact["breakdown"].get("content_type_factor", 1.0),
        })

    events.sort(key=lambda e: e["score"], reverse=True)
    return events[:limit]


# keyword (as it appears in EVENT_CLASSIFICATION_KEYWORDS) -> a clean
# display form, used only by generate_etf_card_data()'s Top Drivers tags
# (e.g. "nvidia" -> "NVIDIA", not a bare lowercase word). Keywords not
# listed here fall back to Title Case.
_KEYWORD_DISPLAY_NAME = {
    "nvidia": "NVIDIA", "tsmc": "TSMC", "asml": "ASML", "apple": "Apple",
    "microsoft": "Microsoft", "google": "Google", "meta": "Meta", "amazon": "Amazon",
    "fomc": "FOMC", "fed decision": "Fed", "fed chair": "Fed", "federal reserve": "Fed",
    "central bank": "Central Bank", "chatgpt": "ChatGPT", "openai": "OpenAI",
    "cpi": "CPI", "ppi": "PPI", "gdp": "GDP", "pmi": "PMI", "opec": "OPEC",
    "bitcoin": "Bitcoin", "ethereum": "Ethereum", "btc": "BTC",
}


def _matched_keywords(headline: str, category: str) -> list:
    """Which literal keyword(s) from EVENT_CLASSIFICATION_KEYWORDS[category]
    actually appear in this headline, formatted for display (e.g. "nvidia"
    -> "NVIDIA") -- used to surface a specific News-level driver tag
    instead of just the category name. The keyword that's just the
    category's own name in disguise (e.g. "semiconductor" for category
    "Semiconductor") is skipped, since generate_etf_card_data() already
    adds the category itself as a separate tag -- keeping it here would
    let a generic word crowd out more specific ones like a company name.
    Not exported; internal to Top Drivers generation below."""
    low = headline.lower()
    words = set(low.replace("-", " ").split())
    keywords = dict(EVENT_CLASSIFICATION_KEYWORDS).get(category, [])
    matched = []
    for kw in keywords:
        if kw == category.lower():
            continue
        hit = (kw in low) if " " in kw else (kw in words)
        if hit:
            matched.append(_KEYWORD_DISPLAY_NAME.get(kw, kw.title()))
    return matched


def generate_etf_card_data(news_items: list, tickers: list = None) -> list:
    """
    ETF Card data for the redesigned "Global ETFs" display: each ticker's
    Impact (stars, reusing calculate_etf_impact() exactly -- not
    recomputed a second, different way), Market (a Bullish/Neutral/Bearish
    sentiment specific to THIS ETF, not one single day-wide mood), Reasons
    (up to 3 short Event Type tags), and Top Drivers (up to 3 tags built
    from Event Type + ETF Type + News, see below) -- so two ETFs driven by
    different headlines can show different sentiment even on the same day.

    For each ticker, a headline counts as a "tag" source when its
    classify_event() category has an EVENT_TYPE_TO_ETF_TYPE_IMPACT weight
    of 4+ ("High" or "Very High") for at least one of the ticker's own ETF
    Type tag(s) (ETF_TYPE_MAP) -- a higher bar than calculate_etf_impact()
    uses for stars, specifically so a ticker isn't "relevant" to nearly
    everything just because its type (e.g. Broad Market) picks up a
    moderate weight from most categories; that would make every ETF's
    Market sentiment converge on the same mood. Among those headlines,
    the single one with the strongest weight for this ticker drives Market
    sentiment (its own directional "impact" field, Positive/Negative/
    Neutral -> Bullish/Bearish/Neutral) -- not a majority vote, so two
    conflicting headlines don't get diluted into an arbitrary tie-break.
    Reasons are the distinct classify_event() categories among all of
    those headlines (deduplicated). A ticker with none today gets Neutral
    sentiment and an empty reasons list.

    Top Drivers pools three tag sources across those same relevant
    headlines, most specific first: (1) News -- the literal keyword(s)
    that matched in the headline text (_matched_keywords(), e.g. "NVIDIA",
    "TSMC", "Fed"), (2) ETF Type -- the ticker's own type tag(s) (e.g.
    "Technology", "Semiconductor"), (3) Event Type -- the classify_event()
    category itself (e.g. "AI", "Inflation"). Deduplicated (case-
    insensitive), capped at 3.

    Returns a list of dicts: ticker, stars, star_label, etf_types,
    sentiment (raw mood), sentiment_label (translated), sentiment_variant
    (badge color), confidence (0-100, how one-sided the driving headline's
    signal is -- same 30-95% blended range used by calculate_ai_market_
    sentiment()/calculate_regional_market_sentiment(), never a fake 0% or
    100%), reasons (list of up to 3 Event Type tags), top_drivers (list of
    up to 3 tags, see above).
    """
    tickers = tickers or DEFAULT_ETF_IMPACT_WATCHLIST
    impact_rows = {row["ticker"]: row for row in calculate_etf_impact(news_items, tickers)}

    results = []
    for ticker in tickers:
        etf_types = ETF_TYPE_MAP.get(ticker, [])
        best_weight = 0
        best_item = None
        reasons = []
        driver_tags = []
        driver_seen = set()

        def _add_driver(tag):
            key = tag.lower()
            if key not in driver_seen:
                driver_seen.add(key)
                driver_tags.append(tag)

        for item in news_items:
            category = classify_event(item["title"])
            if category == "Other":
                continue
            weight = max((EVENT_TYPE_TO_ETF_TYPE_IMPACT.get(category, {}).get(et, 0) for et in etf_types), default=0)
            if weight >= 4:
                if category not in reasons:
                    reasons.append(category)
                for kw in _matched_keywords(item["title"], category):
                    _add_driver(kw)
                _add_driver(category)
                if weight > best_weight:
                    best_weight = weight
                    best_item = item

        for et in etf_types:
            _add_driver(et)
        top_drivers = driver_tags[:3]

        if best_item:
            mood = {"Positive": "Bullish", "Negative": "Bearish"}.get(best_item["impact"], "Neutral")
            signed = {"Positive": 1, "Negative": -1}.get(best_item["impact"], 0)
            avg = signed * (best_weight / 5)
            confidence = int(max(30, min(95, round(50 + abs(avg) * 50))))
        else:
            mood = "Neutral"
            confidence = 30

        stars = impact_rows.get(ticker, {}).get("stars", 1)
        results.append({
            "ticker": ticker,
            "stars": stars,
            "star_label": "★" * stars + "☆" * (5 - stars),
            "etf_types": etf_types,
            "sentiment": mood,
            "sentiment_label": t(_MOOD_KEY[mood]),
            "sentiment_variant": _MOOD_VARIANT[mood],
            "confidence": confidence,
            "reasons": reasons[:3],
            "top_drivers": top_drivers,
        })
    return results


def get_news_card_metadata(news_items: list) -> list:
    """
    Per-headline metadata for the redesigned "Breaking Market News" cards:
    Event Type, Impact (stars), and Sentiment. Unlike
    get_todays_major_events(), nothing is filtered out or limited to a top
    N -- every entry in news_items gets a corresponding row here, in the
    same order, since every news card needs to display something.

    Event Type is classify_event(title). Impact reuses
    calculate_market_impact() on a single-headline list (the same "treat
    this headline as if it were the day's only news" pattern already used
    by get_todays_major_events()/generate_explanation(), not a new scoring
    method). Sentiment is the headline's own precomputed "impact" field
    (Positive/Negative/Neutral, from src/news.py's keyword classifier)
    translated to Bullish/Neutral/Bearish via the same _MOOD_KEY/
    _MOOD_VARIANT labels used everywhere else in this module -- for a
    single headline, its own directional word choice is a more direct
    signal than a weighted multi-headline average.

    Returns a list of dicts (same order/length as news_items): category
    (event type), stars (1-5), star_label, sentiment (raw mood),
    sentiment_label (translated), sentiment_variant (badge color),
    confidence (0-100 -- a single headline is either clearly Positive/
    Negative or Neutral, so this is a fixed 80% for a clear read and 45%
    for Neutral/ambiguous, rather than a fabricated precise number).
    """
    results = []
    for item in news_items:
        category = classify_event(item["title"])
        stars = calculate_market_impact([item])["stars"]
        mood = {"Positive": "Bullish", "Negative": "Bearish"}.get(item.get("impact"), "Neutral")
        confidence = 80 if mood != "Neutral" else 45
        results.append({
            "category": category,
            "stars": stars,
            "star_label": "★" * stars + "☆" * (5 - stars),
            "sentiment": mood,
            "sentiment_label": t(_MOOD_KEY[mood]),
            "sentiment_variant": _MOOD_VARIANT[mood],
            "confidence": confidence,
        })
    return results


# category -> a short "why to watch" reason phrase per language, used only
# by generate_todays_market_action() below.
_ACTION_REASON_EN = {
    "Interest Rate": "shifting interest rate expectations",
    "Inflation": "persistent inflation pressure",
    "Trade Policy": "increasing trade uncertainty",
    "Technology": "renewed technology sector momentum",
    "Semiconductor": "increasing trade uncertainty",
    "Energy": "volatile energy prices",
    "Geopolitics": "escalating geopolitical tension",
    "Economy": "shifting economic growth signals",
    "AI": "accelerating AI-driven demand",
    "Banking": "renewed scrutiny on the banking sector",
    "Cryptocurrency": "volatile crypto market conditions",
    "Other": "today's market developments",
}
_ACTION_REASON_ZH = {
    "Interest Rate": "利率預期出現變化",
    "Inflation": "通膨壓力持續",
    "Trade Policy": "貿易不確定性升高",
    "Technology": "科技類股動能回溫",
    "Semiconductor": "貿易不確定性升高",
    "Energy": "能源價格波動",
    "Geopolitics": "地緣政治緊張升溫",
    "Economy": "經濟成長訊號轉變",
    "AI": "AI 需求加速成長",
    "Banking": "銀行業受到更多關注",
    "Cryptocurrency": "加密貨幣市場波動",
    "Other": "今日市場動態",
}

# category -> a current topic to monitor. These phrases deliberately avoid
# claiming that a decision/release is "upcoming": the app's economic calendar
# is illustrative and is not a verified Fed/release schedule.
_ACTION_TOPIC_EN = {
    "Interest Rate": "Rate-policy headlines remain important, but this app does not have a verified Fed meeting schedule connected.",
    "Inflation": "Inflation developments remain an important macro theme; no release timing is inferred from the illustrative calendar.",
    "Trade Policy": "Trade-policy developments remain a key headline theme.",
    "Technology": "Technology-sector earnings and company updates remain key headline themes.",
    "Semiconductor": "Semiconductor earnings, demand and export-policy developments remain key headline themes.",
    "Energy": "Energy supply and price developments remain key headline themes.",
    "Geopolitics": "Geopolitical developments remain a key source of headline risk.",
    "Economy": "Economic-growth and labor-market developments remain key headline themes.",
    "AI": "AI-industry company and policy developments remain key headline themes.",
    "Banking": "Banking-sector developments remain a key headline theme.",
    "Cryptocurrency": "Crypto market and regulatory developments remain key headline themes.",
    "Other": "Further market-moving news remains worth monitoring.",
}
_ACTION_TOPIC_ZH = {
    "Interest Rate": "利率政策新聞仍值得追蹤；目前系統未串接已驗證的 Fed 會議時程，因此不宣稱有「即將公布」的決策。",
    "Inflation": "通膨相關發展仍是重要總經主題；系統不會從範例行事曆推定實際發布時間。",
    "Trade Policy": "貿易政策後續發展仍是重要新聞主題。",
    "Technology": "科技產業財報與公司動態仍是重要新聞主題。",
    "Semiconductor": "半導體財報、需求與出口政策動態仍是重要新聞主題。",
    "Energy": "能源供需與價格動態仍是重要新聞主題。",
    "Geopolitics": "地緣政治發展仍是重要的新聞風險來源。",
    "Economy": "經濟成長與就業市場動態仍是重要新聞主題。",
    "AI": "AI 產業公司與政策動態仍是重要新聞主題。",
    "Banking": "銀行業後續發展仍是重要新聞主題。",
    "Cryptocurrency": "加密貨幣市場與監管動態仍是重要新聞主題。",
    "Other": "後續可能影響市場的新聞仍值得追蹤。",
}


# ETF Type -> a short lowercase sector word per language, used only by
# generate_todays_market_action()'s first bullet (e.g. "semiconductor-
# related ETFs"). Deliberately without the country/region prefix used by
# _ETF_TYPE_REGION_EN/ZH elsewhere, since this bullet reads more naturally
# without it.
_ETF_TYPE_SECTOR_WORD_EN = {
    "Bond": "bond", "Technology": "technology", "Semiconductor": "semiconductor",
    "Broad Market": "broad market", "Broad Market (UCITS)": "broad market",
}
_ETF_TYPE_SECTOR_WORD_ZH = {
    "Bond": "債券", "Technology": "科技", "Semiconductor": "半導體",
    "Broad Market": "大盤", "Broad Market (UCITS)": "大盤",
}


def generate_todays_market_action(news_items: list, indices: dict = None) -> dict:
    """Rule-based watchlist grounded in current news plus actual VIX state.

    No item infers a future Fed/release date from the illustrative calendar.
    Volatility language comes from VIX when available, not headline mood.
    """
    lang = get_language()
    title = "今日觀察重點" if lang == "zh-TW" else "Today's Watchlist"

    if not news_items:
        no_data = ("目前沒有足夠的新聞資料可產生今日觀察重點。" if lang == "zh-TW"
                   else "Not enough news data is available to generate today's watchlist.")
        return {"title": title, "items": [no_data]}

    top_events = get_todays_major_events(news_items, limit=1)
    category = top_events[0]["category"] if top_events else "Other"
    reason_map = _ACTION_REASON_ZH if lang == "zh-TW" else _ACTION_REASON_EN
    topic_map = _ACTION_TOPIC_ZH if lang == "zh-TW" else _ACTION_TOPIC_EN
    sector_word_map = _ETF_TYPE_SECTOR_WORD_ZH if lang == "zh-TW" else _ETF_TYPE_SECTOR_WORD_EN

    type_impact = EVENT_TYPE_TO_ETF_TYPE_IMPACT.get(category, {})
    top_type = max(type_impact, key=lambda et: type_impact[et], default=None)
    sector_word = sector_word_map.get(top_type, ("相關" if lang == "zh-TW" else "related"))
    reason = reason_map.get(category, reason_map["Other"])

    if lang == "zh-TW":
        item1 = f"因{reason}，{sector_word}相關 ETF 的新聞敏感度較高。"
    else:
        item1 = f"{sector_word.capitalize()}-related ETFs have higher headline sensitivity because of {reason}."

    item2 = topic_map.get(category, topic_map["Other"])

    vix = derive_vix_regime(indices or {})
    if vix["available"]:
        if lang == "zh-TW":
            item3 = (
                f"VIX {vix['price']:.2f}，屬於{t('mi_vix_level_' + vix['level'].lower())}水準，"
                f"目前{t('mi_vix_trend_' + vix['trend'].lower())}；波動度描述以此實際資料為準。"
            )
        else:
            item3 = (
                f"VIX is {vix['price']:.2f}, a {t('mi_vix_level_' + vix['level'].lower())} level, "
                f"and is {t('mi_vix_trend_' + vix['trend'].lower())}; volatility wording follows this observed data."
            )
    else:
        item3 = ("VIX 即時資料無法取得，因此不對目前波動度高低下判斷。" if lang == "zh-TW"
                 else "Live VIX data is unavailable, so no claim is made about the current volatility regime.")

    return {"title": title, "items": [item1, item2, item3]}

