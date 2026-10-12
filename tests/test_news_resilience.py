"""Deterministic tests for the resilient recent-news layer."""

from datetime import datetime, timedelta, timezone

import pytest

import src.news as news_mod
from src.market_intelligence import calculate_market_impact, calculate_market_sentiment


def _item(title, impact="Neutral", age_days=0.0):
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    published = now - timedelta(days=age_days)
    item = {
        "title": title,
        "publisher": "Example",
        "link": "",
        "published": published,
        "impact": impact,
        "source_symbol": "SPY",
    }
    return news_mod.enrich_recency(item, now=now)


def test_two_day_half_life_and_seven_day_residual_weight():
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    two_days_old = now - timedelta(days=2)
    seven_days_old = now - timedelta(days=7)
    assert news_mod.recency_weight(two_days_old, now=now) == pytest.approx(0.5)
    assert news_mod.recency_weight(seven_days_old, now=now) == pytest.approx(0.088388, rel=1e-4)


def test_recent_headline_has_more_sentiment_weight_than_old_headline():
    news = [
        _item("Stocks rally after rate cut", "Positive", age_days=0),
        _item("Markets plunge on recession fears", "Negative", age_days=6),
    ]
    result = calculate_market_sentiment(news)
    assert result["bullish_pct"] > result["bearish_pct"]


def test_market_impact_decays_for_same_event_as_it_ages():
    fresh = [_item("Federal Reserve cuts interest rates", "Positive", age_days=0)]
    old = [_item("Federal Reserve cuts interest rates", "Positive", age_days=7)]
    assert calculate_market_impact(fresh)["score"] > calculate_market_impact(old)["score"]


def test_archive_fallback_is_distinct_from_source_error(monkeypatch):
    archived = [_item("Archived market headline", "Neutral", age_days=1)]
    monkeypatch.setattr(news_mod, "_fetch_live_market_news", lambda: {
        "items": [], "successful_sources": 0, "failed_sources": {"SPY": "blocked"},
    })
    monkeypatch.setattr(news_mod, "_load_archived_news", lambda lookback_days=7, limit=50: archived)
    monkeypatch.setattr(news_mod, "_archive_news_items", lambda items: 0)
    result = news_mod.fetch_market_news_with_status(limit=10, lookback_days=7)
    assert result["status"] == "archive_fallback"
    assert len(result["items"]) == 1


def test_source_error_when_live_and_archive_are_both_unavailable(monkeypatch):
    monkeypatch.setattr(news_mod, "_fetch_live_market_news", lambda: {
        "items": [], "successful_sources": 0, "failed_sources": {"SPY": "blocked"},
    })
    monkeypatch.setattr(news_mod, "_load_archived_news", lambda lookback_days=7, limit=50: [])
    result = news_mod.fetch_market_news_with_status(limit=10, lookback_days=7)
    assert result["status"] == "source_error"
    assert result["items"] == []


def test_relative_age_label_is_localized():
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    published = now - timedelta(days=3)
    assert news_mod.relative_age_label(published, "en", now=now) == "3d ago"
    assert news_mod.relative_age_label(published, "zh-TW", now=now) == "3 天前"


def test_archive_recent_when_source_is_reachable_but_returns_no_new_items(monkeypatch):
    archived = [_item("Recently archived market headline", "Neutral", age_days=1)]
    monkeypatch.setattr(news_mod, "_fetch_live_market_news", lambda: {
        "items": [], "successful_sources": 4, "failed_sources": {},
    })
    monkeypatch.setattr(
        news_mod, "_load_archived_news",
        lambda lookback_days=7, limit=50: archived,
    )
    result = news_mod.fetch_market_news_with_status(limit=10, lookback_days=7)
    assert result["status"] == "archive_recent"
    assert len(result["items"]) == 1
