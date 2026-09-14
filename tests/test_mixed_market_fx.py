"""
Mixed-market Portfolio Optimizer + base-currency FX conversion --
integration tests (Issue #22 section F).

src/fx.py's own unit tests (tests/test_fx.py) already cover the conversion
math/edge-cases in isolation. This file instead drives
pages/2_Portfolio_Optimizer.py end-to-end (via Streamlit AppTest) with a
mocked price download (no network) and a mocked src.fx.convert_prices_to_base_currency
(no network) to confirm the PAGE actually wires FX conversion into the
optimization pipeline: it calls the conversion when currencies differ, stops
with a clear error rather than silently mixing currencies when FX data is
unavailable, and discloses the currency treatment in the Methodology panel.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _synthetic_prices(tickers, n_days=60, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    data = {}
    for i, tk in enumerate(tickers):
        rets = rng.normal(0.0005 + 0.0001 * i, 0.01, n_days)
        data[tk] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=dates)


def _mock_download_by_display_ticker(prices: pd.DataFrame):
    """pages/2_Portfolio_Optimizer.py requests data using Yahoo-mapped
    symbols (src.etf_database.to_yahoo_symbol(), e.g. "0050" -> "0050.TW")
    and only converts columns back to display tickers afterward
    (rename_yahoo_columns()) -- a fake download keyed by plain display
    ticker would silently return empty for every non-US ticker. This
    builds the fake in terms of the SAME Yahoo-symbol mapping the real
    pipeline uses, so a mixed-market (US + Taiwan) selection round-trips
    exactly like it would against a real download.
    """
    from src.etf_database import to_yahoo_symbol
    yahoo_to_display = {to_yahoo_symbol(tk): tk for tk in prices.columns}

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        cols = {yt: prices[yahoo_to_display[yt]] for yt in tickers if yt in yahoo_to_display}
        return pd.DataFrame(cols, index=prices.index) if cols else pd.DataFrame()

    return _fake_download


@pytest.fixture()
def mixed_market_setup(monkeypatch):
    """VOO (USD) + 0050 (TWD): a genuinely mixed-currency selection using
    real, already-registered ETF universe tickers (src.etf_database), with
    both the price download AND the FX conversion mocked so this test needs
    no network access and is fully deterministic."""
    import src.data_loader as data_loader_mod
    prices = _synthetic_prices(["VOO", "0050"])
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _mock_download_by_display_ticker(prices))
    return prices


def _seed_mixed_region_state(at):
    """Pre-seed session_state so the page starts already scoped to "All
    Regions" with VOO + 0050 selected -- the same direct session_state
    pre-seeding pattern tests/test_portfolio_optimizer.py uses to simulate
    the shared global-market-state mechanism (src/ui.py's region_selector()/
    region_etf_multiselect()) without needing multiple widget-interaction
    reruns to converge (the widgets are backed by a region-keyed shadow
    dict, e.g. "selected_etfs_All Regions", not just their own key).
    """
    all_regions_label = "All Regions"
    at.session_state["language"] = "en"
    at.session_state["selected_region"] = all_regions_label
    at.session_state["_selected_region_shadow"] = all_regions_label
    at.session_state["_selected_etfs_shadow"] = {all_regions_label: ["VOO", "0050"]}
    return all_regions_label


def test_mixed_currency_selection_triggers_fx_conversion(mixed_market_setup, monkeypatch):
    import src.fx as fx_mod

    converted = mixed_market_setup.copy()
    converted["0050"] = converted["0050"] / 30.0  # pretend TWD->USD at ~30

    def _fake_convert(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        assert ticker_currency_map["0050"] == "TWD"
        assert ticker_currency_map["VOO"] == "USD"
        assert base_currency == "USD"
        return {
            "converted_prices": converted[list(prices_df.columns)],
            "unavailable_tickers": [],
            "fx_source": "TEST-FX-SOURCE",
            "conversion_method": "TEST-CONVERSION-METHOD",
            "base_currency": base_currency,
            "currency_adjusted": True,
        }

    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    _seed_mixed_region_state(at)
    at.run()
    at.button(key="opt_run_optimization_btn").click()
    at.run()

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert set(cp["tickers"]) == {"VOO", "0050"}
    assert cp["base_currency"] == "USD"
    assert cp["currency_adjusted"] is True

    corpus = "\n".join(m.value for m in at.markdown)
    assert "TEST-FX-SOURCE" in corpus
    assert "TEST-CONVERSION-METHOD" in corpus


def test_fx_unavailable_stops_instead_of_mixing_currencies(mixed_market_setup, monkeypatch):
    import src.fx as fx_mod

    def _fake_convert_unavailable(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        return {
            "converted_prices": prices_df[["VOO"]],
            "unavailable_tickers": ["0050"],
            "fx_source": "TEST-FX-SOURCE",
            "conversion_method": "TEST-CONVERSION-METHOD",
            "base_currency": base_currency,
            "currency_adjusted": True,
        }

    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert_unavailable)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    _seed_mixed_region_state(at)
    at.run()
    at.button(key="opt_run_optimization_btn").click()
    at.run()

    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "FX Data Unavailable" in corpus
    assert "0050" in corpus
    # Must not have silently produced a "successful" mixed-currency result.
    assert "current_portfolio" not in at.session_state


def test_single_currency_selection_skips_fx_entirely(monkeypatch):
    """VOO + QQQ are both USD -- src.fx.convert_prices_to_base_currency must
    never even be called (no wasted work, and nothing to disclose beyond
    "not needed")."""
    import src.data_loader as data_loader_mod
    import src.fx as fx_mod

    prices = _synthetic_prices(["VOO", "QQQ"])

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return prices[[tk for tk in tickers if tk in prices.columns]].copy()

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)

    called = {"n": 0}

    def _fake_convert(*a, **k):
        called["n"] += 1
        raise AssertionError("convert_prices_to_base_currency should not be called for a single-currency selection")

    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    at.multiselect(key="selected_etfs_United States").set_value(["VOO", "QQQ"])
    at.run()
    at.button(key="opt_run_optimization_btn").click()
    at.run()

    assert at.exception == []
    assert called["n"] == 0
    cp = at.session_state["current_portfolio"]
    assert cp["currency_adjusted"] is False
    corpus = "\n".join(m.value for m in at.markdown)
    assert "no FX conversion was needed" in corpus
