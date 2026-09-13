"""
M4 -- Market Intelligence & Sentiment Validation: deterministic tests.

Pure pytest `assert` tests (no network -- all inputs are synthetic
news_items lists, never a live yfinance/OpenAI call) covering:
  - src/methodology.py's MARKET_INTELLIGENCE_METHODOLOGY metadata
  - bullish/bearish signals are never forced when the signal doesn't
    clearly warrant a direction (calculate_ai_market_sentiment)
  - Market Impact Score is independent of headline sentiment direction,
    proving impact and confidence are genuinely different measurements
    (not the same number under two names)
  - affected-market mapping is evidence-based (event-type -> market
    weight), not keyword name-matching against the market's own name --
    a headline that never says "Taiwan" still rates Taiwan's impact
  - i18n parity (zh-TW / en) for every new mi_methodology_* key
  - src/news.py's classify_headline_sentiment() keyword classifier
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methodology import MARKET_INTELLIGENCE_METHODOLOGY
from src.market_intelligence import (
    calculate_ai_market_sentiment, calculate_market_impact, calculate_affected_markets,
)
from src.news import classify_headline_sentiment
from src.etf_database import get_countries
from src.i18n import TRANSLATIONS


# ── Methodology metadata ─────────────────────────────────────────────────
def test_methodology_metadata_matches_actual_implementation():
    m = MARKET_INTELLIGENCE_METHODOLOGY
    assert len(m["pipeline"]["stages"]) >= 5
    assert "deterministic" in m["pipeline"]["deterministic_vs_ai"].lower()
    assert "Neutral" in m["bullish_bearish_neutral"]["policy"]
    assert "impact" in m["measures"] and "confidence" in m["measures"] and "relevance" in m["measures"]
    assert "gating" in m["measures"]["relevance"].lower()
    assert "NOT" in m["measures"]["impact"] and "NOT" in m["measures"]["confidence"]
    assert "Taiwan" in m["affected_markets"]["limitation"] or "UK" in m["affected_markets"]["limitation"]
    assert "NOT externally validated" in m["validation"]["status"]
    assert "precision" in m["validation"]["no_fabricated_metrics"].lower()


# ── Bullish/Bearish/Neutral: never forced when unwarranted ──────────────
def test_sentiment_is_neutral_when_signal_is_balanced():
    """Equal-and-opposite weighted headlines must cancel out to Neutral,
    not be arbitrarily tie-broken into a direction."""
    news_items = [
        {"title": "Central bank raises interest rate significantly", "impact": "Negative"},
        {"title": "Central bank announces surprise interest rate cut", "impact": "Positive"},
    ]
    result = calculate_ai_market_sentiment(news_items)
    assert result["mood"] == "Neutral"


def test_sentiment_is_neutral_with_no_news():
    result = calculate_ai_market_sentiment([])
    assert result["mood"] == "Neutral"
    assert result["confidence"] == 0


def test_sentiment_follows_a_one_sided_signal():
    bullish_news = [{"title": "Central bank announces surprise interest rate cut", "impact": "Positive"}]
    bearish_news = [{"title": "Central bank raises interest rate significantly", "impact": "Negative"}]
    assert calculate_ai_market_sentiment(bullish_news)["mood"] == "Bullish"
    assert calculate_ai_market_sentiment(bearish_news)["mood"] == "Bearish"


# ── Impact vs. Confidence are genuinely independent measurements ────────
def test_market_impact_score_is_independent_of_sentiment_direction():
    """The SAME headline (same event category, same frequency) scored with
    a Negative vs. a Positive sentiment tag must produce the IDENTICAL
    Market Impact Score/stars -- calculate_market_impact() never looks at
    the "impact" (sentiment) field. Confidence/mood, computed by a
    different function, must differ instead. This is the concrete proof
    that impact and confidence are not the same measurement in disguise.
    """
    headline = "Central bank raises interest rate significantly"
    negative_news = [{"title": headline, "impact": "Negative"}]
    positive_news = [{"title": headline, "impact": "Positive"}]

    impact_neg = calculate_market_impact(negative_news)
    impact_pos = calculate_market_impact(positive_news)
    assert impact_neg["score"] == impact_pos["score"]
    assert impact_neg["stars"] == impact_pos["stars"]
    assert impact_neg["category"] == impact_pos["category"] == "Interest Rate"

    sentiment_neg = calculate_ai_market_sentiment(negative_news)
    sentiment_pos = calculate_ai_market_sentiment(positive_news)
    assert sentiment_neg["mood"] != sentiment_pos["mood"]


# ── Affected-market mapping is evidence-based, not name-matching ────────
def test_affected_markets_are_evidence_based_not_keyword_name_matching():
    """A headline that never mentions "Taiwan" by name must still rate
    Taiwan's impact highly when its classified event type (tariff /
    semiconductor) is known to affect Taiwan -- proving the mapping is
    driven by EVENT_MARKET_IMPACT, not by literal country-name matching.
    """
    headline = "New export ban announced on chip technology"
    assert "taiwan" not in headline.lower()
    news_items = [{"title": headline, "impact": "Negative"}]

    results = calculate_affected_markets(news_items)
    assert "Taiwan" in get_countries()
    taiwan_result = next(r for r in results if r["country"] == "Taiwan")
    assert taiwan_result["stars"] == 5


def test_affected_markets_baseline_when_no_relevant_news():
    """A market untouched by any classified event still renders (baseline
    'Very Low Impact', stars=1) rather than being omitted or crashing."""
    results = calculate_affected_markets([])
    assert len(results) == len(get_countries())
    assert all(r["stars"] == 1 for r in results)


# ── src/news.py's headline sentiment classifier ──────────────────────────
def test_classify_headline_sentiment_keyword_matching():
    assert classify_headline_sentiment("Stocks surge to record high on rate cut hopes") == "Positive"
    assert classify_headline_sentiment("Markets plunge as recession fears grow") == "Negative"
    assert classify_headline_sentiment("Company reports quarterly results") == "Neutral"
    assert classify_headline_sentiment("") == "Neutral"
    assert classify_headline_sentiment(None) == "Neutral"


# ── i18n parity for every new methodology/validation key ────────────────
def test_market_intelligence_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "mi_methodology_title", "mi_methodology_subtitle",
        "mi_methodology_pipeline_label", "mi_methodology_pipeline_desc",
        "mi_methodology_ai_vs_rule_label", "mi_methodology_ai_vs_rule_desc",
        "mi_methodology_measures_label", "mi_methodology_measures_desc",
        "mi_methodology_markets_label", "mi_methodology_markets_desc",
        "mi_methodology_validation_label", "mi_methodology_validation_desc",
        "mi_methodology_source_label", "mi_methodology_source_desc",
        "mi_impact_caption",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key] != key
        assert TRANSLATIONS["en"][key] != key
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""
