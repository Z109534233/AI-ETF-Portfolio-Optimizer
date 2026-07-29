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
}

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
