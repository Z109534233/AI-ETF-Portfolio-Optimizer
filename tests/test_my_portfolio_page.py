"""
My Portfolio page -- Goal Planner / Current Holdings / Watchlist / Daily
Brief tab behavior (Issue #22). Complements tests/test_portfolio_history.py
(which covers the Portfolio History tab's pre-existing behavior unchanged).

Every test runs against an isolated, throwaway SQLite file (same
`isolated_db` fixture pattern as tests/test_portfolio_history.py) and never
hits the network: `src.data_loader._download_single_ticker` is monkeypatched
so ticker validation / price lookups are deterministic and fast.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest

import src.database as dbmod


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test_portfolio.db"))
    dbmod.init_database()
    return dbmod


@pytest.fixture()
def no_network(monkeypatch):
    """No real ticker ever has live price data in this test process --
    _is_known_or_live_ticker() falls back to `get_etf(ticker) is not None`
    for any ticker in the curated universe (e.g. VOO, QQQ), and to False
    for anything else (e.g. a deliberately bogus ticker)."""
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "_download_single_ticker", lambda *a, **k: None)
    import pages  # noqa: F401  (ensure package import path is set up)


def _run_my_portfolio(lang="en"):
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    return at


# ── Guest/session isolation ───────────────────────────────────────────────
def test_public_page_has_no_auth_not_configured_message(isolated_db, no_network):
    at = _run_my_portfolio()
    corpus = "\n".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert "not configured for this deployment" not in corpus
    assert "shared demo data" not in corpus
    assert "Guest mode" in corpus


def test_holdings_are_isolated_between_fresh_browser_sessions(isolated_db, no_network):
    first = _run_my_portfolio()
    first.text_input(key="ch_new_ticker").set_value("VOO")
    first.number_input(key="ch_new_qty").set_value(10.0)
    first.button(key="ch_add_btn").click()
    first.run()
    assert [h["ticker"] for h in first.session_state["_guest_session_holdings"]] == ["VOO"]

    second = _run_my_portfolio()
    assert second.session_state["_guest_session_holdings"] == []
    assert isolated_db.load_user_holdings("demo") == []


def test_goal_planner_zh_selection_logic_has_no_english_leak(isolated_db, no_network):
    at = _run_my_portfolio(lang="zh-TW")
    corpus = "\n".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert "Conservative:" not in corpus
    assert "category 'Fixed Income'" not in corpus
    assert "非選股結果" in corpus


# ── Goal Planner ─────────────────────────────────────────────────────────
def test_goal_planner_demo_defaults_produce_useful_showcase_inputs(isolated_db, no_network):
    at = _run_my_portfolio()
    assert at.number_input(key="gp_monthly_contribution").value == 5000.0
    assert at.number_input(key="gp_target_amount").value == 5_000_000.0


def test_goal_planner_uses_responsive_scenario_grid_and_nominal_label(isolated_db, no_network):
    at = _run_my_portfolio(lang="zh-TW")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "gp-scenario-grid" in corpus
    assert "@media (max-width: 900px)" in corpus
    assert "達標路徑比例（名目）" in corpus
    assert "80% 路徑月投入" in corpus


def test_goal_planner_renders_three_scenarios_by_default(isolated_db, no_network):
    at = _run_my_portfolio()
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Conservative" in corpus
    assert "Balanced" in corpus
    assert "Aggressive" in corpus
    # Default inputs (age 35 -> 65, i.e. 30-year horizon) must be valid.
    assert "Can't Build a Plan Yet" not in corpus


def test_goal_planner_invalid_horizon_shows_error(isolated_db, no_network):
    at = _run_my_portfolio()
    at.number_input(key="gp_current_age").set_value(70)
    at.number_input(key="gp_target_age").set_value(65)
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Can't Build a Plan Yet" in corpus


def test_goal_planner_income_mode_shows_withdrawal_rate(isolated_db, no_network):
    at = _run_my_portfolio()
    at.selectbox(key="gp_target_mode").set_value("annual_income")
    at.run()
    assert at.exception == []
    corpus = "\n".join(c.value for c in at.caption)
    assert "4%" in corpus


# ── Current Holdings ─────────────────────────────────────────────────────
def test_add_holding_with_known_ticker_succeeds(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("VOO")
    at.number_input(key="ch_new_qty").set_value(10.0)
    at.number_input(key="ch_new_cost").set_value(380.0)
    at.button(key="ch_add_btn").click()
    at.run()
    assert at.exception == []
    success_texts = "\n".join(s.value for s in at.success)
    assert "VOO" in success_texts

    holdings = at.session_state["_guest_session_holdings"]
    assert len(holdings) == 1
    assert holdings[0]["ticker"] == "VOO"
    assert holdings[0]["quantity"] == 10.0
    # Public visitors must never write holdings into the shared SQLite DB.
    assert isolated_db.load_user_holdings("demo") == []


def test_add_holding_with_invalid_ticker_shows_error_and_does_not_persist(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("NOTAREALTICKERXYZ")
    at.number_input(key="ch_new_qty").set_value(5.0)
    at.button(key="ch_add_btn").click()
    at.run()
    assert at.exception == []
    error_texts = "\n".join(e.value for e in at.error)
    assert "NOTAREALTICKERXYZ" in error_texts
    assert at.session_state["_guest_session_holdings"] == []
    assert isolated_db.load_user_holdings("demo") == []


def test_add_holding_with_zero_quantity_rejected(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("VOO")
    at.number_input(key="ch_new_qty").set_value(0.0)
    at.button(key="ch_add_btn").click()
    at.run()
    assert at.exception == []
    assert at.session_state["_guest_session_holdings"] == []
    assert isolated_db.load_user_holdings("demo") == []


def test_holding_price_unavailable_shown_without_fabricated_value(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("VOO")
    at.number_input(key="ch_new_qty").set_value(10.0)
    at.number_input(key="ch_new_cost").set_value(380.0)
    at.button(key="ch_add_btn").click()
    at.run()
    dataframes = [df.value for df in at.dataframe if "VOO" in str(getattr(df.value, "values", ""))]
    assert dataframes, "holdings table not found"
    assert "Unavailable" in str(dataframes[0].values)


def test_remove_holding(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("VOO")
    at.number_input(key="ch_new_qty").set_value(10.0)
    at.number_input(key="ch_new_cost").set_value(380.0)
    at.button(key="ch_add_btn").click()
    at.run()
    at.button(key="ch_remove_btn").click()
    at.run()
    assert at.exception == []
    assert at.session_state["_guest_session_holdings"] == []
    assert isolated_db.load_user_holdings("demo") == []


# ── Watchlist ─────────────────────────────────────────────────────────────
def test_add_watchlist_item_with_known_ticker(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="wl_new_ticker").set_value("QQQ")
    at.button(key="wl_add_btn").click()
    at.run()
    assert at.exception == []
    assert [w["ticker"] for w in at.session_state["_guest_session_watchlist"]] == ["QQQ"]
    assert isolated_db.load_watchlist("demo") == []


def test_add_watchlist_item_duplicate_shows_info_not_error(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="wl_new_ticker").set_value("QQQ")
    at.button(key="wl_add_btn").click()
    at.run()
    at.text_input(key="wl_new_ticker").set_value("QQQ")
    at.button(key="wl_add_btn").click()
    at.run()
    assert at.exception == []
    assert len(at.session_state["_guest_session_watchlist"]) == 1
    assert isolated_db.load_watchlist("demo") == []


def test_remove_watchlist_item(isolated_db, no_network):
    at = _run_my_portfolio()
    at.text_input(key="wl_new_ticker").set_value("QQQ")
    at.button(key="wl_add_btn").click()
    at.run()
    at.button(key="wl_remove_btn").click()
    at.run()
    assert at.exception == []
    assert at.session_state["_guest_session_watchlist"] == []
    assert isolated_db.load_watchlist("demo") == []


# ── Daily Brief ───────────────────────────────────────────────────────────
def test_daily_brief_empty_state_when_no_holdings_or_watchlist(isolated_db, no_network):
    at = _run_my_portfolio()
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Add a ticker to Current Holdings or Watchlist" in corpus


def test_daily_brief_renders_with_holdings(isolated_db, no_network, monkeypatch):
    import src.news as news_mod
    monkeypatch.setattr(news_mod, "fetch_market_news", lambda limit=10: [])
    at = _run_my_portfolio()
    at.text_input(key="ch_new_ticker").set_value("VOO")
    at.number_input(key="ch_new_qty").set_value(10.0)
    at.number_input(key="ch_new_cost").set_value(380.0)
    at.button(key="ch_add_btn").click()
    at.run()
    corpus = "\n".join(m.value for m in at.markdown)
    assert "No major change detected" in corpus or "No major change" in corpus
