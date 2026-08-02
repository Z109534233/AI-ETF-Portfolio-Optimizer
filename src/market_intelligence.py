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

from src.ai_advisor import get_openai_client
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
    """
    Market Impact Score: a 0-100 score for today's news, built from *event
    importance* rather than "did a headline mention this ETF/market by
    name". Every headline is classified via classify_event() into one
    category, the single most important category present today is treated
    as the day's dominant event (ties broken by how often that category
    repeats), and the score is a weighted blend of:
      - Event Importance   (40%) -- EVENT_IMPORTANCE_SCORE for the
        dominant category
      - Affected Markets    (25%) -- EVENT_AFFECTED_MARKETS, how strongly
        that category is known to hit our supported markets
      - Affected Industry   (20%) -- EVENT_INDUSTRY_BREADTH, how broadly
        that category tends to reach across industries/sectors
      - News Frequency      (15%) -- what share of today's headlines are
        about that same dominant category (a repeated theme signals a
        bigger, more pervasive story than a single one-off mention)
    A transparent rule-based heuristic, not a prediction of market
    direction.

    Returns a dict: score (0-100), stars (0-5; 0 only when there is no
    news to score), star_label (e.g. "★★★★☆"), category (the dominant
    event category driving the score, or None), breakdown (dict of the
    four component scores, each already on a 0-100 scale, for
    transparency).
    """
    empty_breakdown = {"event_importance": 0, "affected_markets": 0, "affected_industry": 0, "news_frequency": 0}
    if not news_items:
        return {"score": 0, "stars": 0, "star_label": "☆☆☆☆☆", "category": None, "breakdown": empty_breakdown}

    categories = [classify_event(item["title"]) for item in news_items]
    category_counts = Counter(categories)

    present = [c for c in category_counts if c != "Other"] or list(category_counts)
    dominant = max(present, key=lambda c: (EVENT_IMPORTANCE_SCORE.get(c, 20), category_counts[c]))

    event_importance = EVENT_IMPORTANCE_SCORE.get(dominant, 20)

    market_weights = EVENT_AFFECTED_MARKETS.get(dominant, {})
    affected_markets = max(market_weights.values()) * 20 if market_weights else 20

    affected_industry = EVENT_INDUSTRY_BREADTH.get(dominant, 15)

    news_frequency = min(100, round(category_counts[dominant] / len(news_items) * 100))

    score = round(
        event_importance * 0.40
        + affected_markets * 0.25
        + affected_industry * 0.20
        + news_frequency * 0.15
    )
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

    Returns a list of dicts: market (translated country name), stars (1-5),
    impact_label (translated, e.g. "High Impact" -- never bare stars),
    affected_by (up to 2 headline titles whose event type drove the rating).
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
        results.append({
            "market": t_country(country),
            "stars": stars,
            "impact_label": stars_to_impact_label(stars),
            "affected_by": market_events[country][:2],
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
    """
    Static, clearly-labelled placeholder economic calendar (Fed Meeting, CPI,
    PPI, GDP, NFP, FOMC). Intentionally simple and easy to swap for a live
    data source (e.g. FRED, Trading Economics) later without changing the
    page that renders it.
    """
    return [
        {"event": "CPI (Consumer Price Index)", "when": t("mi_cal_this_week"), "importance": t("mi_cal_high")},
        {"event": "FOMC / Fed Interest Rate Decision", "when": t("mi_cal_this_week"), "importance": t("mi_cal_high")},
        {"event": "PPI (Producer Price Index)", "when": t("mi_cal_this_week"), "importance": t("mi_cal_medium")},
        {"event": "Non-Farm Payrolls (NFP)", "when": t("mi_cal_upcoming"), "importance": t("mi_cal_high")},
        {"event": "GDP Growth Rate", "when": t("mi_cal_upcoming"), "importance": t("mi_cal_medium")},
        {"event": "Fed Chair Speech", "when": t("mi_cal_upcoming"), "importance": t("mi_cal_medium")},
    ]


def generate_market_summary(news_items: list, sentiment: dict, affected_etfs: list) -> str:
    """
    Generate a ~100-200 word "Today's Market Summary" from current headlines.
    Uses OpenAI when an API key is configured, otherwise falls back to a
    rule-based summary built from the same data. Never raises -- always
    returns display-safe text, including when there is no news at all.
    """
    if not news_items:
        return t("mi_summary_no_news")

    client = get_openai_client()
    if client is not None:
        try:
            return _generate_ai_summary(client, news_items, sentiment, affected_etfs)
        except Exception:
            pass  # fall through to the rule-based summary below

    return _generate_rule_based_summary(news_items, sentiment, affected_etfs)


def _generate_ai_summary(client, news_items: list, sentiment: dict, affected_etfs: list) -> str:
    headlines = "\n".join(f"- {n['title']}" for n in news_items[:8])
    affected_str = ", ".join(f"{e['ticker']} ({e['sector']}: {e['impact_label']})" for e in affected_etfs) or "N/A"
    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文)."
        if get_language() == "zh-TW" else "Respond entirely in English."
    )
    prompt = f"""You are an educational market analyst assistant. Based on today's headlines below, write a "Today's Market Summary" of about 100-200 words covering: the main market events today, overall market sentiment, which sectors are affected, and which ETFs may be affected. Do not predict future prices or give personalised investment advice. {language_instruction}

Headlines:
{headlines}

Headline sentiment split: {sentiment['bullish_pct']}% bullish, {sentiment['neutral_pct']}% neutral, {sentiment['bearish_pct']}% bearish.
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


def _generate_rule_based_summary(news_items: list, sentiment: dict, affected_etfs: list) -> str:
    top_titles = [n["title"] for n in news_items[:3]]

    if sentiment["bullish_pct"] > sentiment["bearish_pct"] + 10:
        tilt = t("mi_tilt_bullish")
    elif sentiment["bearish_pct"] > sentiment["bullish_pct"] + 10:
        tilt = t("mi_tilt_bearish")
    else:
        tilt = t("mi_tilt_mixed")

    positive_etfs = [e["ticker"] for e in affected_etfs if e["impact"] == "Positive"]
    negative_etfs = [e["ticker"] for e in affected_etfs if e["impact"] == "Negative"]

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


def generate_today_ai_summary(news_items: list) -> dict:
    """
    "Today's AI Summary" -- a short template-generated paragraph meant for
    the very top of the Market Intelligence page (above "Today's Market
    Overview"), synthesizing:
      - today's news volume
      - event classification (classify_event() via calculate_market_impact())
      - market sentiment (calculate_regional_market_sentiment()'s "Global" row)
      - ETF impact (calculate_etf_impact(), the most-affected watch-list ticker)
    into one narrative, e.g. "Markets are under pressure today because...".
    Template-based only -- no OpenAI/LLM call -- reserved as the entry
    point for a future LLM-generated version, matching the same
    "transparent heuristic, not a prediction" philosophy as the rest of
    this module. This is a separate, independent feature from
    generate_market_summary() (the existing, untouched "AI Market Summary"
    section further down the page, which calls OpenAI when configured);
    neither reuses the other, and generate_market_summary() is unchanged.

    Language-aware (get_language()) without adding new src/i18n.py keys --
    the sentence needs per-language phrasing throughout, not one swappable
    word, so both language versions are written out directly here.

    Returns a dict: title (translated section title), summary (a
    display-safe ~3-4 sentence string). Never raises -- returns a neutral
    "not enough data" summary when there is no news.
    """
    lang = get_language()
    title = "今日 AI 摘要" if lang == "zh-TW" else "Today's AI Summary"

    if not news_items:
        no_data = ("目前沒有足夠的新聞資料可產生今日摘要。" if lang == "zh-TW"
                   else "Not enough news data is available to generate today's summary.")
        return {"title": title, "summary": no_data}

    impact = calculate_market_impact(news_items)
    sentiment = calculate_regional_market_sentiment(news_items)
    global_sentiment = next((s for s in sentiment if s["market"] == "Global"), None)
    etf_impact = calculate_etf_impact(news_items)
    top_etf = max(etf_impact, key=lambda e: e["stars"]) if etf_impact else None

    category = impact["category"] or "Other"
    category_en, category_zh = _CATEGORY_PHRASE.get(category, _CATEGORY_PHRASE["Other"])
    mood = global_sentiment["mood"] if global_sentiment else "Neutral"
    confidence = global_sentiment["confidence"] if global_sentiment else 0

    if lang == "zh-TW":
        sentences = [
            f"今天市場{_MOOD_PHRASE_ZH.get(mood, '持穩')}，主要受到{category_zh}相關新聞影響，"
            f"全球情緒判讀為{t(_MOOD_KEY[mood])}（信心指數 {confidence}%）。",
            f"今日市場影響分數為 {impact['score']}/100（{impact['stars']} 顆星）。",
        ]
        if top_etf:
            etf_type_label = "、".join(top_etf["etf_types"]) or "未分類"
            sentences.append(f"觀察清單中反應最明顯的是 {top_etf['ticker']}（{etf_type_label}），影響程度 {top_etf['stars']} 顆星。")
        sentences.append("此摘要為規則式生成，僅供教育用途，不構成投資建議或價格預測。")
    else:
        sentences = [
            f"Markets are {_MOOD_PHRASE_EN.get(mood, 'holding steady')} today because of {category_en} news, "
            f"with global sentiment reading {t(_MOOD_KEY[mood])} ({confidence}% confidence).",
            f"Today's Market Impact Score is {impact['score']}/100 ({impact['stars']} stars).",
        ]
        if top_etf:
            etf_type_label = "/".join(top_etf["etf_types"]) or "Unclassified"
            sentences.append(f"{top_etf['ticker']} ({etf_type_label}) shows the strongest reaction potential today at {top_etf['stars']} stars.")
        sentences.append("This is a rule-based summary for educational purposes only -- not investment advice or a price prediction.")

    return {"title": title, "summary": " ".join(sentences)}
