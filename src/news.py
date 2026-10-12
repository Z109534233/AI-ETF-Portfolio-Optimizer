"""
Market News Module

Yahoo Finance is treated as an opportunistic live source, not a guaranteed
availability contract. Successfully observed headlines are archived in SQLite
and reused for a short recent-news window when the live source is unavailable.

All publication timestamps are normalized to UTC. Recent-news analytics attach
an exponential time-decay weight with a 2-day half-life, so older headlines can
remain relevant without carrying the same influence as fresh information.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import math

import streamlit as st
import yfinance as yf

from src.database import MarketNewsArchive, get_session

NEWS_SOURCE_TICKERS = ["SPY", "QQQ", "DIA", "^GSPC"]
NEWS_CACHE_TTL_SECONDS = 900
NEWS_LOOKBACK_DAYS = 7
NEWS_HALF_LIFE_DAYS = 2.0

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
    "hikes interest rate", "raises interest rate", "rising yields",
    "yields rise", "yields climb", "caution", "cautious",
]


def classify_headline_sentiment(title: str) -> str:
    """Transparent keyword-based Positive/Negative/Neutral classifier."""
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


def _parse_publish_time(entry: dict, content: dict):
    """Return a timezone-aware UTC datetime when the source provides one."""
    ts = entry.get("providerPublishTime")
    if ts:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pass

    pub_date_str = content.get("pubDate")
    if pub_date_str:
        try:
            parsed = datetime.fromisoformat(str(pub_date_str).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except (TypeError, ValueError):
            pass
    return None


def _utc_naive(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _utc_aware(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _dedupe_key(item: dict) -> str:
    raw = "|".join([
        str(item.get("link") or "").strip(),
        str(item.get("title") or "").strip().lower(),
        str(_utc_naive(item.get("published")) or ""),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def recency_weight(published, now=None, half_life_days: float = NEWS_HALF_LIFE_DAYS) -> float:
    """Exponential decay: 1 today, ~0.5 after 2 days, ~0.09 after 7 days."""
    if published is None:
        return 0.5
    now = _utc_aware(now or datetime.now(timezone.utc))
    published = _utc_aware(published)
    age_days = max(0.0, (now - published).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / max(float(half_life_days), 1e-9)))


def enrich_recency(item: dict, now=None) -> dict:
    result = dict(item)
    published = _utc_aware(result.get("published"))
    now = _utc_aware(now or datetime.now(timezone.utc))
    age_days = None
    if published is not None:
        age_days = max(0.0, (now - published).total_seconds() / 86400.0)
    result["published"] = published
    result["age_days"] = age_days
    result["recency_weight"] = recency_weight(published, now=now)
    return result


def relative_age_label(published, lang: str = "en", now=None) -> str:
    published = _utc_aware(published)
    if published is None:
        return "時間未提供" if lang == "zh-TW" else "time unavailable"
    now = _utc_aware(now or datetime.now(timezone.utc))
    seconds = max(0.0, (now - published).total_seconds())
    hours = int(seconds // 3600)
    days = int(seconds // 86400)
    if days >= 1:
        return f"{days} 天前" if lang == "zh-TW" else f"{days}d ago"
    if hours >= 1:
        return f"{hours} 小時前" if lang == "zh-TW" else f"{hours}h ago"
    minutes = max(1, int(seconds // 60))
    return f"{minutes} 分鐘前" if lang == "zh-TW" else f"{minutes}m ago"


@st.cache_data(ttl=NEWS_CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_live_market_news(limit_per_source: int = 10) -> dict:
    """Fetch Yahoo news once per cache window and preserve source status."""
    seen_titles = set()
    items = []
    failures = {}
    successful_sources = 0

    for symbol in NEWS_SOURCE_TICKERS:
        try:
            raw_news = yf.Ticker(symbol).news or []
            successful_sources += 1
        except Exception as exc:
            failures[symbol] = f"{type(exc).__name__}: {exc}"
            logging.warning("Yahoo news fetch failed for %s: %s", symbol, failures[symbol])
            continue

        for entry in raw_news[:limit_per_source]:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content") if isinstance(entry.get("content"), dict) else {}
            title = (content.get("title") or entry.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            canonical = content.get("canonicalUrl")
            link = canonical.get("url", "") if isinstance(canonical, dict) else ""
            if not link:
                link = entry.get("link", "")

            provider = content.get("provider")
            publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
            if not publisher:
                publisher = entry.get("publisher", "Yahoo Finance")

            items.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": _parse_publish_time(entry, content),
                "impact": classify_headline_sentiment(title),
                "source_symbol": symbol,
            })

    items.sort(
        key=lambda x: _utc_naive(x.get("published")) or datetime.min,
        reverse=True,
    )
    return {
        "items": items,
        "successful_sources": successful_sources,
        "failed_sources": failures,
    }


def _archive_news_items(items: list) -> int:
    if not items:
        return 0
    session = get_session()
    if session is None:
        return 0

    stored = 0
    now = datetime.utcnow()
    try:
        for item in items:
            key = _dedupe_key(item)
            row = session.query(MarketNewsArchive).filter(
                MarketNewsArchive.dedupe_key == key
            ).first()
            if row is None:
                row = MarketNewsArchive(
                    dedupe_key=key,
                    title=item.get("title") or "",
                    publisher=item.get("publisher"),
                    link=item.get("link"),
                    published_at=_utc_naive(item.get("published")),
                    impact=item.get("impact") or "Neutral",
                    source_symbol=item.get("source_symbol"),
                    first_seen_at=now,
                    last_seen_at=now,
                )
                session.add(row)
                stored += 1
            else:
                row.last_seen_at = now
                row.publisher = item.get("publisher") or row.publisher
                row.link = item.get("link") or row.link
                row.published_at = _utc_naive(item.get("published")) or row.published_at
                row.impact = item.get("impact") or row.impact
                row.source_symbol = item.get("source_symbol") or row.source_symbol
        session.commit()
        return stored
    except Exception as exc:
        session.rollback()
        logging.warning("News archive write failed: %s", exc)
        return 0
    finally:
        session.close()


def _load_archived_news(lookback_days: int = NEWS_LOOKBACK_DAYS, limit: int = 50) -> list:
    session = get_session()
    if session is None:
        return []
    cutoff = datetime.utcnow() - timedelta(days=int(lookback_days))
    try:
        rows = (
            session.query(MarketNewsArchive)
            .filter(
                (
                    (MarketNewsArchive.published_at.isnot(None))
                    & (MarketNewsArchive.published_at >= cutoff)
                )
                | (
                    (MarketNewsArchive.published_at.is_(None))
                    & (MarketNewsArchive.first_seen_at >= cutoff)
                )
            )
            .order_by(
                MarketNewsArchive.published_at.desc(),
                MarketNewsArchive.first_seen_at.desc(),
            )
            .limit(int(limit))
            .all()
        )
        return [
            {
                "title": row.title,
                "publisher": row.publisher or "Yahoo Finance",
                "link": row.link or "",
                "published": _utc_aware(row.published_at),
                "impact": row.impact or classify_headline_sentiment(row.title),
                "source_symbol": row.source_symbol,
                "archived": True,
            }
            for row in rows
        ]
    except Exception as exc:
        logging.warning("News archive read failed: %s", exc)
        return []
    finally:
        session.close()


def fetch_market_news_with_status(limit: int = 25,
                                  lookback_days: int = NEWS_LOOKBACK_DAYS) -> dict:
    """Return a recent-news feed plus explicit live-source/archive status."""
    live = _fetch_live_market_news()
    live_items = live["items"]
    if live_items:
        _archive_news_items(live_items)

    archived = _load_archived_news(lookback_days=lookback_days, limit=max(limit * 3, 50))
    if archived:
        items = archived[:limit]
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
        items = [
            item for item in live_items
            if item.get("published") is None or _utc_aware(item["published"]) >= cutoff
        ][:limit]

    items = [enrich_recency(item) for item in items]
    items.sort(
        key=lambda x: _utc_naive(x.get("published")) or datetime.min,
        reverse=True,
    )

    failures = live["failed_sources"]
    successful_sources = int(live["successful_sources"])
    if live_items:
        status = "partial_live" if failures else "live"
    elif failures and items:
        status = "archive_fallback"
    elif failures and not items:
        status = "source_error"
    elif successful_sources > 0 and items:
        status = "archive_recent"
    else:
        status = "no_recent_news"

    return {
        "items": items[:limit],
        "status": status,
        "live_item_count": len(live_items),
        "archive_item_count": len(archived),
        "successful_sources": successful_sources,
        "failed_sources": failures,
        "lookback_days": int(lookback_days),
        "half_life_days": NEWS_HALF_LIFE_DAYS,
    }


def fetch_market_news(limit: int = 10) -> list:
    """Backward-compatible list-only wrapper used by existing callers."""
    return fetch_market_news_with_status(limit=limit)["items"]
