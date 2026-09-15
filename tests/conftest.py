"""
Shared pytest fixtures for the whole test suite.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest


@pytest.fixture(autouse=True)
def _no_live_holdings_primary_adapter_by_default(monkeypatch):
    """src.holdings.get_etf_holdings() tries a primary (public yfinance
    funds_data) adapter before its private-quoteSummary fallback. Tests
    across the suite mock the FALLBACK adapter (`_fetch_yahoo_topholdings_raw`)
    with deterministic fixtures and never expect a real network call -- so
    by default, force the primary adapter to report "unreached" here,
    session-wide, so every one of those existing fallback-only mocks keeps
    exercising the fallback path deterministically instead of silently
    racing a live Yahoo Finance request. Any test that specifically wants to
    exercise the primary adapter overrides `_cached_fetch_public` itself
    (see tests/test_holdings.py), which simply takes precedence for that
    test since it's the same underlying monkeypatch fixture instance.
    """
    import src.holdings as _holdings_mod
    monkeypatch.setattr(_holdings_mod, "_cached_fetch_public", lambda yahoo_symbol: (None, None, False))
