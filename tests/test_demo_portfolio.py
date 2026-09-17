"""
Shared cross-asset demo portfolio constant tests (Issue #45 item 1).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.demo_portfolio import (
    DIVERSIFIED_DEMO_TICKERS, demo_portfolio_available, resolve_demo_tickers,
)


def test_demo_tickers_are_the_required_diversified_set():
    assert DIVERSIFIED_DEMO_TICKERS == ["VOO", "VXUS", "BND", "GLD", "TLT"]


def test_demo_portfolio_available_true_when_all_present():
    universe = ["QQQ", "VOO", "VXUS", "BND", "GLD", "TLT", "SPY"]
    assert demo_portfolio_available(universe) is True


def test_demo_portfolio_available_false_when_any_missing():
    universe = ["VOO", "VXUS", "BND", "GLD"]  # missing TLT
    assert demo_portfolio_available(universe) is False


def test_resolve_demo_tickers_returns_demo_set_when_available():
    universe = ["QQQ", "SPY", "VOO", "VXUS", "BND", "GLD", "TLT", "VTI"]
    result = resolve_demo_tickers(universe)
    assert result == ["VOO", "VXUS", "BND", "GLD", "TLT"]


def test_resolve_demo_tickers_falls_back_when_demo_set_unavailable():
    universe = ["0050", "0056", "006208"]  # Taiwan-only universe, no demo tickers
    result = resolve_demo_tickers(universe, fallback_n=2)
    assert result == ["0050", "0056"]
    # Never fabricates a ticker that isn't actually in the universe.
    assert all(tk in universe for tk in result)


def test_resolve_demo_tickers_never_fabricates_a_ticker_outside_universe():
    universe = ["VOO", "VXUS"]  # partial overlap only
    result = resolve_demo_tickers(universe, fallback_n=5)
    assert all(tk in universe for tk in result)
