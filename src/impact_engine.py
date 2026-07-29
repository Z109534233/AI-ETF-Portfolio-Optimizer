"""
Impact Engine
Core scoring logic for the AI Global Market Intelligence Center. Every
function here is a transparent, deterministic heuristic built from today's
event volume/category/sentiment -- NOT a predictive model, price forecast,
or trading signal. Each is written as a single, swappable entry point so a
real ML/LLM model can be dropped in later (e.g. inside calculate_market_impact)
without any page code needing to change.
"""

from typing import List

from src.i18n import t
from src.market_models import EventImpact, MarketImpact

# market_key -> headline keywords used to detect relevance to that market.
MARKET_KEYWORDS = {
    "united_states": ["fed", "u.s.", "united states", "nasdaq", "dow", "s&p", "white house", "tariff", "washington"],
    "taiwan": ["taiwan", "tsmc", "taipei"],
    "united_kingdom": ["uk", "britain", "bank of england", "london", "ftse"],
    "japan": ["japan", "nikkei", "bank of japan", "tokyo"],
    "europe": ["europe", "ecb", "eurozone", "euro area", "stoxx"],
    "china": ["china", "beijing", "pboc", "yuan"],
}


def score_to_stars(score: int) -> int:
    """Map a 0-100 impact score to a 0-5 star rating."""
    if score >= 90:
        return 5
    if score >= 70:
        return 4
    if score >= 50:
        return 3
    if score >= 30:
        return 2
    if score > 0:
        return 1
    return 0


def stars_to_label(stars: int) -> str:
    """Translate a 0-5 star rating to its display label (e.g. 'Very High')."""
    return t(f"mi_impact_stars_{stars}")


def star_rating_string(stars: int, max_stars: int = 5) -> str:
    """Plain-text star rating, e.g. '★★★☆☆' -- for non-HTML contexts."""
    stars = max(0, min(stars, max_stars))
    return "★" * stars + "☆" * (max_stars - stars)


def calculate_market_impact(events: List[EventImpact]) -> dict:
    """
    AI Global Market Impact Score -- the headline metric of the Market
    Intelligence Center. Reserved as the entry point for a future ML/LLM
    market-impact model; the current implementation is a transparent
    rule-based heuristic combining event volume, category weight, and
    directional (Positive/Negative) sentiment mix. It measures how much
    market-moving activity is happening today -- NOT which direction
    prices will move, and is not investment advice.

    Returns {"score": 0-100, "stars": 0-5, "label": str, "explanation": str}.
    """
    if not events:
        return {
            "score": 0, "stars": 0,
            "label": t("mi_impact_stars_0"),
            "explanation": t("mi_impact_no_events"),
        }

    high_weight = sum(1 for e in events if e.weight == 3)
    directional = sum(1 for e in events if e.impact != "Neutral")
    score = min(100, 35 + high_weight * 12 + directional * 5)
    stars = score_to_stars(score)
    label = stars_to_label(stars)

    top_categories = []
    seen = set()
    for e in sorted(events, key=lambda x: x.weight, reverse=True):
        if e.category not in seen:
            seen.add(e.category)
            top_categories.append(t(f"mi_category_{e.category}"))
        if len(top_categories) >= 2:
            break

    explanation = t(
        "mi_impact_score_explanation",
        level=label.lower(),
        count=len(events),
        categories=", ".join(top_categories) if top_categories else t("mi_category_general"),
    )
    return {"score": score, "stars": stars, "label": label, "explanation": explanation}


def calculate_affected_markets(events: List[EventImpact]) -> List[MarketImpact]:
    """
    Determine which markets today's events may be relevant to, by matching
    each market's keyword set against event titles -- this is computed
    dynamically from MARKET_KEYWORDS and the supplied events, never a
    hardcoded table of scores. Always returns the full watch-list of
    markets (with a "Very Low" baseline when there is no matching news) so
    the section renders consistently, sorted by score descending.
    """
    results = []
    for market_key, keywords in MARKET_KEYWORDS.items():
        matches = [e for e in events if any(kw in e.title.lower() for kw in keywords)]
        high_weight_matches = sum(1 for e in matches if e.weight == 3)
        score = min(100, 15 + len(matches) * 20 + high_weight_matches * 10) if matches else 15
        stars = score_to_stars(score)
        results.append(MarketImpact(
            market=t(f"mi_market_{market_key}"),
            market_key=market_key,
            score=score,
            stars=stars,
            label=stars_to_label(stars),
        ))
    results.sort(key=lambda m: m.score, reverse=True)
    return results
