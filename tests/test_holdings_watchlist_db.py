"""
Current Holdings / Watchlist database layer -- deterministic tests (Issue #22
section D). Every test runs against an isolated, throwaway SQLite file (same
`isolated_db` fixture pattern as tests/test_portfolio_history.py) -- never the
real committed database/portfolio.db.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

import src.database as dbmod


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test_portfolio.db"))
    dbmod.init_database()
    return dbmod


def test_add_and_load_user_holding(isolated_db):
    ok = isolated_db.add_user_holding("user:alice", "VOO", 10, 380.0, "USD", "2024-01-15")
    assert ok is True
    rows = isolated_db.load_user_holdings("user:alice")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "VOO"
    assert rows[0]["quantity"] == 10
    assert rows[0]["average_cost"] == 380.0
    assert rows[0]["currency"] == "USD"
    assert rows[0]["purchase_date"] == "2024-01-15"


def test_holdings_are_isolated_per_user(isolated_db):
    isolated_db.add_user_holding("user:alice", "VOO", 10, 380.0, "USD")
    isolated_db.add_user_holding("user:bob", "0050", 100, 150.0, "TWD")
    assert [r["ticker"] for r in isolated_db.load_user_holdings("user:alice")] == ["VOO"]
    assert [r["ticker"] for r in isolated_db.load_user_holdings("user:bob")] == ["0050"]
    assert isolated_db.load_user_holdings("demo") == []


def test_delete_user_holding_scoped_to_owner(isolated_db):
    isolated_db.add_user_holding("user:alice", "VOO", 10, 380.0, "USD")
    holding_id = isolated_db.load_user_holdings("user:alice")[0]["id"]

    # Bob cannot delete Alice's holding even if he knows/guesses its id.
    assert isolated_db.delete_user_holding("user:bob", holding_id) is False
    assert len(isolated_db.load_user_holdings("user:alice")) == 1

    assert isolated_db.delete_user_holding("user:alice", holding_id) is True
    assert isolated_db.load_user_holdings("user:alice") == []


def test_load_user_holdings_empty_returns_empty_list(isolated_db):
    assert isolated_db.load_user_holdings("nobody") == []


def test_add_watchlist_item_and_load(isolated_db):
    assert isolated_db.add_watchlist_item("user:alice", "QQQ") is True
    items = isolated_db.load_watchlist("user:alice")
    assert len(items) == 1
    assert items[0]["ticker"] == "QQQ"


def test_add_watchlist_item_no_duplicate(isolated_db):
    isolated_db.add_watchlist_item("user:alice", "QQQ")
    isolated_db.add_watchlist_item("user:alice", "QQQ")
    items = isolated_db.load_watchlist("user:alice")
    assert len(items) == 1


def test_watchlist_isolated_per_user(isolated_db):
    isolated_db.add_watchlist_item("user:alice", "QQQ")
    isolated_db.add_watchlist_item("user:bob", "0050")
    assert [i["ticker"] for i in isolated_db.load_watchlist("user:alice")] == ["QQQ"]
    assert [i["ticker"] for i in isolated_db.load_watchlist("user:bob")] == ["0050"]


def test_remove_watchlist_item_scoped_to_owner(isolated_db):
    isolated_db.add_watchlist_item("user:alice", "QQQ")
    item_id = isolated_db.load_watchlist("user:alice")[0]["id"]

    assert isolated_db.remove_watchlist_item("user:bob", item_id) is False
    assert len(isolated_db.load_watchlist("user:alice")) == 1

    assert isolated_db.remove_watchlist_item("user:alice", item_id) is True
    assert isolated_db.load_watchlist("user:alice") == []


def test_portfolio_table_has_backward_compatible_user_id_column(isolated_db):
    """A portfolio saved without specifying user_id (every pre-existing
    caller) must still load fine, with user_id absent/None rather than a
    crash -- see _ensure_user_id_column()'s docstring."""
    ok = isolated_db.save_portfolio(
        name="Legacy Save", weights={"VOO": 1.0}, investment_amount=10000.0,
        optimization_method="Equal Weight", expected_return=0.1,
        expected_volatility=0.15, sharpe_ratio=0.6,
    )
    assert ok is True
    portfolios = isolated_db.load_all_portfolios(raise_on_error=True)
    assert len(portfolios) == 1
