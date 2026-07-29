"""
Global ETF Coverage Module
Defines the multi-region ETF universe (US / Taiwan / UK UCITS) shown in the
"Affected ETFs" section, computes per-ETF impact ratings by matching each
ETF's sector/region keywords against today's breaking events, and computes a
saved portfolio's sector-exposure breakdown. Not limited to US equities --
this is the "Global ETF Coverage" layer of the platform.
"""

from collections import Counter
from typing import List

from src.i18n import t
from src.impact_engine import score_to_stars
from src.market_models import ETFImpact

# ticker -> (region_key, sector_key)
ETF_UNIVERSE = {
    # US ETFs
    "VOO": ("us", "broad_market"),
    "QQQ": ("us", "technology"),
    "VTI": ("us", "broad_market"),
    "SPY": ("us", "broad_market"),
    "SCHD": ("us", "dividend"),
    "GLD": ("us", "gold"),
    "BND": ("us", "bond"),

    # Taiwan ETFs
    "0050": ("taiwan", "broad_market"),
    "0056": ("taiwan", "dividend"),
    "006208": ("taiwan", "broad_market"),
    "00878": ("taiwan", "dividend"),
    "00919": ("taiwan", "dividend"),
    "00929": ("taiwan", "dividend"),

    # UK UCITS ETFs
    "VUSA": ("uk", "broad_market"),
    "VUAG": ("uk", "technology"),
    "CSPX": ("uk", "broad_market"),
    "EQQQ": ("uk", "technology"),
    "FUSD": ("uk", "dividend"),
}

REGION_COUNTRY_KEY = {"us": "united_states", "taiwan": "taiwan", "uk": "united_kingdom"}

SECTOR_KEYWORDS = {
    "technology": ["tech", "technology", "nasdaq", "chip", "semiconductor", "ai "],
    "broad_market": ["s&p", "wall street", "stocks", "market", "index"],
    "gold": ["gold", "bullion", "precious metal"],
    "bond": ["bond", "treasury", "yield", "rate", "fed", "interest rate"],
    "dividend": ["dividend", "income", "payout"],
}

REGION_KEYWORDS = {
    "us": ["u.s.", "united states", "fed", "washington", "nasdaq", "dow", "s&p"],
    "taiwan": ["taiwan", "tsmc", "taipei"],
    "uk": ["uk", "britain", "london", "bank of england", "ftse"],
}


def get_global_etf_universe() -> dict:
    """Return the ETF universe grouped by region: {region_key: [tickers]}."""
    grouped = {}
    for ticker, (region, _sector) in ETF_UNIVERSE.items():
        grouped.setdefault(region, []).append(ticker)
    return grouped


def calculate_etf_impacts(events: list, tickers: list = None) -> List[ETFImpact]:
    """
    Rate each ETF's likely relevance to today's events by matching its
    sector AND region keywords against event titles. This is a transparent
    keyword heuristic -- reserved for a future per-ETF ML model -- and only
    flags topical relevance; it never predicts price direction or magnitude.
    """
    tickers = tickers or list(ETF_UNIVERSE.keys())
    results = []
    for ticker in tickers:
        region, sector_key = ETF_UNIVERSE.get(ticker, ("us", "other"))
        keywords = SECTOR_KEYWORDS.get(sector_key, []) + REGION_KEYWORDS.get(region, [])
        matches = [e for e in events if any(kw in e.title.lower() for kw in keywords)]

        if matches:
            counts = Counter(e.impact for e in matches)
            direction = counts.most_common(1)[0][0]
        else:
            direction = "Neutral"

        score = min(100, len(matches) * 25) if matches else 10
        stars = score_to_stars(score) or 1

        results.append(ETFImpact(
            ticker=ticker,
            country=t(f"mi_market_{REGION_COUNTRY_KEY.get(region, 'united_states')}"),
            sector=t(f"mi_sector_{sector_key}"),
            direction=direction,
            direction_label=t(f"mi_impact_{direction.lower()}_label"),
            stars=stars,
            impact_score=score,
        ))
    return results


def calculate_portfolio_exposure(holdings: dict) -> dict:
    """
    Aggregate a saved portfolio's holdings into sector-exposure percentages
    (translated sector label -> weight), sorted from highest to lowest.
    """
    exposure = {}
    for ticker, weight in holdings.items():
        _region, sector_key = ETF_UNIVERSE.get(ticker, ("us", "other"))
        label = t(f"mi_sector_{sector_key}")
        exposure[label] = exposure.get(label, 0) + weight
    return dict(sorted(exposure.items(), key=lambda x: x[1], reverse=True))
