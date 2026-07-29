"""
Market Data Models
Lightweight dataclasses shared across the AI Global Market Intelligence
Center modules (news_service, impact_engine, global_etf, market_intelligence).
Keeping these as plain dataclasses -- rather than passing raw dicts around --
gives every downstream function (and any future AI model swapped into
impact_engine/market_intelligence) a single, typed contract to depend on.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EventImpact:
    """A single breaking market event (headline) and its classification."""
    title: str
    category: str                      # internal key, e.g. "monetary_policy", "trade"
    source: str
    published: Optional[datetime]
    impact: str                        # "Positive" / "Negative" / "Neutral"
    link: str = ""
    weight: int = 1                    # 1-3 magnitude proxy, used by impact_engine


@dataclass
class MarketImpact:
    """How strongly today's events may be affecting a given country/market."""
    market: str                        # translated display name, e.g. "United States"
    market_key: str                    # internal key, e.g. "united_states"
    score: int                         # 0-100
    stars: int                         # 0-5
    label: str                         # translated label, e.g. "Very High"


@dataclass
class ETFImpact:
    """How strongly today's events may be relevant to a given ETF."""
    ticker: str
    country: str                       # translated display name
    sector: str                        # translated display name
    direction: str                     # raw "Positive" / "Negative" / "Neutral"
    direction_label: str               # translated
    stars: int                         # 0-5, reserved for a future per-ETF model
    impact_score: int                  # 0-100, reserved for a future per-ETF model
