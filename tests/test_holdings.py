"""
ETF Holdings & Exposure -- source-adapter regression tests (visual-acceptance
follow-up on PR #28: the public yfinance `funds_data` adapter must be tried
FIRST, the private quoteSummary adapter only as a fallback, and no path may
ever fabricate a holding or weight).

Every test mocks `src.holdings._cached_fetch_public` / `_cached_fetch_raw`
directly (the two module-level names `get_etf_holdings()` actually calls) --
no real network access, no dependency on yfinance actually reaching Yahoo.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest

import src.holdings as holdings_mod
from src.holdings import (
    get_etf_holdings, itemized_holdings, total_disclosed_weight,
    STATUS_UPDATED, STATUS_CACHED, STATUS_UNAVAILABLE, STATUS_NOT_SUPPORTED,
)


@pytest.fixture(autouse=True)
def _isolated_last_good_cache(monkeypatch):
    """The process-local last-known-good cache is a plain module dict, not
    request-scoped -- isolate every test from every other test's writes to
    it (and from anything a prior live call in this process may have
    populated for "VOO")."""
    monkeypatch.setattr(holdings_mod, "_LAST_GOOD_SNAPSHOT", {})


def _public_top_holdings_df():
    return pd.DataFrame(
        {"Name": ["NVIDIA Corp", "Apple Inc"], "Holding Percent": [0.0755, 0.0704]},
        index=pd.Index(["NVDA", "AAPL"], name="Symbol"),
    )


def _public_asset_classes():
    return {"cashPosition": 0.006, "stockPosition": 0.994, "bondPosition": 0.0,
            "otherPosition": 0.0, "preferredPosition": 0.0, "convertiblePosition": 0.0}


def _private_quotesummary_result():
    return {
        "topHoldings": {
            "holdings": [
                {"symbol": "NVDA", "holdingName": "NVIDIA Corp", "holdingPercent": {"raw": 0.0755}},
                {"symbol": "AAPL", "holdingName": "Apple Inc", "holdingPercent": {"raw": 0.0704}},
            ],
            "cashPosition": {"raw": 0.006},
        }
    }


def test_public_adapter_success_normalizes_holdings(monkeypatch):
    """The primary (public funds_data) adapter succeeding should populate a
    fully-normalized, real snapshot -- itemized rows sorted by weight, plus
    the disclosed aggregate cash bucket -- and never touch the fallback."""
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public",
                         lambda sym: (_public_top_holdings_df(), _public_asset_classes(), True))

    def _fallback_should_not_be_called(sym):
        raise AssertionError("fallback adapter must not be called when the public adapter succeeds")
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw", _fallback_should_not_be_called)

    snap = get_etf_holdings("VOO")

    assert snap.status == STATUS_UPDATED
    assert snap.data_date is not None
    itemized = itemized_holdings(snap)
    assert [h.holding_ticker for h in itemized] == ["NVDA", "AAPL"]
    assert itemized[0].weight == pytest.approx(0.0755)
    assert itemized[0].holding_name == "NVIDIA Corp"
    assert all(not h.is_aggregate for h in itemized)

    aggregates = [h for h in snap.holdings if h.is_aggregate]
    assert len(aggregates) == 1
    assert aggregates[0].holding_ticker == "CASH"
    assert aggregates[0].weight == pytest.approx(0.006)

    # Zero-weight buckets in asset_classes (stockPosition is excluded by
    # design -- see _AGGREGATE_BUCKETS -- and bond/other/preferred/
    # convertible are all 0.0 here) must never appear as fabricated rows.
    assert total_disclosed_weight(snap) == pytest.approx(0.0755 + 0.0704 + 0.006)


def test_public_adapter_empty_falls_back_to_private_adapter(monkeypatch):
    """When the public adapter reaches Yahoo but has nothing usable (e.g. a
    yfinance release/response shape it can't parse), the private
    quoteSummary adapter must still be tried, and its real data shown."""
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public", lambda sym: (None, None, True))
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw",
                         lambda sym: (_private_quotesummary_result(), True))

    snap = get_etf_holdings("VOO")

    assert snap.status == STATUS_UPDATED
    itemized = itemized_holdings(snap)
    assert [h.holding_ticker for h in itemized] == ["NVDA", "AAPL"]
    assert itemized[0].weight == pytest.approx(0.0755)


def test_source_failure_with_no_prior_snapshot_is_unavailable(monkeypatch):
    """Both adapters failing outright (network error / rate limit -- request
    never reached the source) with no last-known-good on record must show
    the honest 'unavailable' status, never a fabricated result."""
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public", lambda sym: (None, None, False))
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw", lambda sym: (None, False))

    snap = get_etf_holdings("VOO")

    assert snap.status == STATUS_UNAVAILABLE
    assert snap.holdings == []
    assert snap.data_date is None


def test_source_reached_but_not_a_fund_is_not_supported(monkeypatch):
    """Both adapters reaching Yahoo successfully but returning nothing (the
    instrument genuinely isn't a fund Yahoo discloses holdings for) must be
    distinguished from a network failure -- Status D, not Status C."""
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public", lambda sym: (None, None, True))
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw", lambda sym: (None, True))

    snap = get_etf_holdings("VOO")

    assert snap.status == STATUS_NOT_SUPPORTED
    assert snap.holdings == []


def test_cached_last_known_good_used_when_source_later_fails(monkeypatch):
    """A successful call populates the last-known-good cache; a LATER call
    where both adapters fail must fall back to that real, previously-
    fetched snapshot (status 'cached'), not show 'unavailable' for data
    that was genuinely available moments ago."""
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public",
                         lambda sym: (_public_top_holdings_df(), _public_asset_classes(), True))
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw", lambda sym: (None, False))

    first = get_etf_holdings("VOO")
    assert first.status == STATUS_UPDATED

    # Source now fails entirely on a later call.
    monkeypatch.setattr(holdings_mod, "_cached_fetch_public", lambda sym: (None, None, False))

    second = get_etf_holdings("VOO")
    assert second.status == STATUS_CACHED
    assert [h.holding_ticker for h in itemized_holdings(second)] == \
        [h.holding_ticker for h in itemized_holdings(first)]
    assert second.data_date == first.data_date


def test_public_adapter_never_fabricates_zero_or_missing_buckets(monkeypatch):
    """asset_classes buckets that are absent or exactly 0 must be omitted
    entirely, never rendered as a fabricated zero-weight holding row."""
    monkeypatch.setattr(
        holdings_mod, "_cached_fetch_public",
        lambda sym: (pd.DataFrame(columns=["Name", "Holding Percent"]),
                     {"cashPosition": 0.0, "bondPosition": None}, True),
    )
    monkeypatch.setattr(holdings_mod, "_cached_fetch_raw", lambda sym: (None, True))

    snap = get_etf_holdings("VOO")

    assert snap.holdings == []
    assert snap.status == STATUS_NOT_SUPPORTED
