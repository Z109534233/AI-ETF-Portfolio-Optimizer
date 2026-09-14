"""
Portfolio Optimizer -- true multi-country selector (Issue #24 item 1).

Drives pages/2_Portfolio_Optimizer.py end-to-end via Streamlit AppTest with
a mocked price download (no network) to confirm the new country
multiselect (src/ui.py's region_multiselect() / region_etf_options_multi()
/ multi_region_etf_multiselect()) actually lets a user pick any
combination of United States / Taiwan / United Kingdom, that the ETF
universe is the union of ONLY the selected countries, that removing a
country prunes only that country's ETFs from the active selection, and
that current_portfolio / experiment metadata carry the selected countries
correctly. Mixed-currency FX wiring itself (conversion math, "FX
unavailable" stop) is already covered end-to-end by
tests/test_mixed_market_fx.py -- this file adds the equivalent coverage
specifically for 3-way (not just 2-way) country combinations plus the
country-removal safety behavior that file doesn't exercise.
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


def _synthetic_prices(tickers, n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    data = {}
    for i, tk in enumerate(tickers):
        rets = rng.normal(0.0005 + 0.0001 * i, 0.01, n_days)
        data[tk] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=dates)


def _mock_download_by_display_ticker(prices: pd.DataFrame):
    """Same pattern as tests/test_mixed_market_fx.py: pages/2_Portfolio_
    Optimizer.py requests data via Yahoo-mapped symbols
    (src.etf_database.to_yahoo_symbol()), so the fake download must be
    keyed the same way."""
    from src.etf_database import to_yahoo_symbol
    yahoo_to_display = {to_yahoo_symbol(tk): tk for tk in prices.columns}

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        cols = {yt: prices[yahoo_to_display[yt]] for yt in tickers if yt in yahoo_to_display}
        return pd.DataFrame(cols, index=prices.index) if cols else pd.DataFrame()

    return _fake_download


def _find_run_button(at):
    for b in at.button:
        if b.key == "opt_run_optimization_btn":
            return b
    return next(iter(at.button), None)


def _select_countries_and_etfs(at, countries, tickers):
    """Set the new multi-country picker + ETF multiselect, matching on the
    widget's own value space (AppTest's multiselect.options/.value are the
    format_func-rendered "TICKER — Name" strings, not raw tickers -- see
    tests/test_portfolio_optimizer.py's TWU-cross/GEU-E tests for the same
    gotcha)."""
    region_w = next(w for w in at.multiselect if w.key == "selected_regions")
    region_w.set_value(list(countries))
    at.run()

    ms = next(w for w in at.multiselect if w.key and w.key.startswith("selected_etfs_"))
    by_ticker = {opt.split(" — ")[0]: opt for opt in ms.options}
    missing = [tk for tk in tickers if tk not in by_ticker]
    assert not missing, f"tickers not offered for {countries}: {missing} (offered: {sorted(by_ticker)})"
    ms.set_value([by_ticker[tk] for tk in tickers])
    at.run()
    return ms


def _build(at, countries, tickers, base_currency=None):
    _select_countries_and_etfs(at, countries, tickers)
    if base_currency is not None:
        bc_w = next(w for w in at.selectbox if w.key == "opt_base_currency")
        bc_w.set_value(base_currency)
        at.run()
    run_btn = _find_run_button(at)
    run_btn.click()
    at.run()
    return at


@pytest.fixture()
def mock_download(monkeypatch):
    def _apply(tickers, seed=7):
        import src.data_loader as data_loader_mod
        prices = _synthetic_prices(tickers, seed=seed)
        monkeypatch.setattr(data_loader_mod, "download_etf_data", _mock_download_by_display_ticker(prices))
        return prices
    return _apply


# ── Single-country selections ────────────────────────────────────────────
def test_us_only_selection(mock_download):
    mock_download(["VOO", "QQQ"])
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["United States"], ["VOO", "QQQ"])

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert cp["markets"] == ["United States"]
    assert cp["market"] == "United States"
    assert set(cp["tickers"]) == {"VOO", "QQQ"}
    assert cp["currency_adjusted"] is False


def test_taiwan_only_selection(mock_download):
    mock_download(["0050", "006208"])
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["Taiwan"], ["0050", "006208"])

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert cp["markets"] == ["Taiwan"]
    assert cp["market"] == "Taiwan"
    assert set(cp["tickers"]) == {"0050", "006208"}


# ── Two-country combinations ─────────────────────────────────────────────
def test_us_plus_taiwan_selection(mock_download, monkeypatch):
    mock_download(["VOO", "0050"])
    import src.fx as fx_mod

    def _fake_convert(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        assert ticker_currency_map["0050"] == "TWD"
        assert ticker_currency_map["VOO"] == "USD"
        converted = prices_df.copy()
        return {
            "converted_prices": converted, "unavailable_tickers": [],
            "fx_source": "TEST-FX", "conversion_method": "TEST-METHOD",
            "base_currency": base_currency, "currency_adjusted": True,
        }
    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["United States", "Taiwan"], ["VOO", "0050"], base_currency="USD")

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert cp["markets"] == ["United States", "Taiwan"]
    assert cp["market"] == "United States + Taiwan"
    assert set(cp["tickers"]) == {"VOO", "0050"}
    assert cp["currency_adjusted"] is True
    assert cp["base_currency"] == "USD"


def test_taiwan_plus_uk_selection(mock_download, monkeypatch):
    mock_download(["0050", "VUSA"])
    import src.fx as fx_mod

    def _fake_convert(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        assert ticker_currency_map["0050"] == "TWD"
        assert ticker_currency_map["VUSA"] == "GBP"
        return {
            "converted_prices": prices_df.copy(), "unavailable_tickers": [],
            "fx_source": "TEST-FX", "conversion_method": "TEST-METHOD",
            "base_currency": base_currency, "currency_adjusted": True,
        }
    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["Taiwan", "United Kingdom"], ["0050", "VUSA"], base_currency="TWD")

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert cp["markets"] == ["Taiwan", "United Kingdom"]
    assert set(cp["tickers"]) == {"0050", "VUSA"}
    assert cp["currency_adjusted"] is True
    assert cp["base_currency"] == "TWD"


# ── All three countries ──────────────────────────────────────────────────
def test_all_three_countries_selection(mock_download, monkeypatch):
    mock_download(["VOO", "0050", "VUSA"])
    import src.fx as fx_mod

    def _fake_convert(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        return {
            "converted_prices": prices_df.copy(), "unavailable_tickers": [],
            "fx_source": "TEST-FX", "conversion_method": "TEST-METHOD",
            "base_currency": base_currency, "currency_adjusted": True,
        }
    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["United States", "Taiwan", "United Kingdom"], ["VOO", "0050", "VUSA"], base_currency="USD")

    assert at.exception == []
    cp = at.session_state["current_portfolio"]
    assert cp["markets"] == ["United States", "Taiwan", "United Kingdom"]
    assert cp["market"] == "United States + Taiwan + United Kingdom"
    assert set(cp["tickers"]) == {"VOO", "0050", "VUSA"}


# ── Removing a selected country ──────────────────────────────────────────
def test_removing_a_country_prunes_only_its_etfs(mock_download):
    mock_download(["VOO", "0050", "VUSA"])
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    ms = _select_countries_and_etfs(
        at, ["United States", "Taiwan", "United Kingdom"], ["VOO", "0050", "VUSA"],
    )
    before = {opt.split(" — ")[0] for opt in ms.value}
    assert before == {"VOO", "0050", "VUSA"}

    # Drop United Kingdom -- only VUSA (UK-only) should disappear from the
    # active selection; VOO/0050 (still-active countries) must survive.
    region_w = next(w for w in at.multiselect if w.key == "selected_regions")
    region_w.set_value(["United States", "Taiwan"])
    at.run()

    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)
    ms2 = next(w for w in at.multiselect if w.key and w.key.startswith("selected_etfs_"))
    after = {opt.split(" — ")[0] for opt in ms2.value}
    assert after == {"VOO", "0050"}, after
    # And the now-removed country's ETF is no longer even offered.
    offered = {opt.split(" — ")[0] for opt in ms2.options}
    assert "VUSA" not in offered, offered


def test_at_least_one_country_required():
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    region_w = next(w for w in at.multiselect if w.key == "selected_regions")
    region_w.set_value([])
    at.run()

    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)
    corpus = "\n".join(w.value for w in at.warning) if hasattr(at, "warning") else ""
    if not corpus:
        corpus = "\n".join(m.value for m in at.markdown)
    assert "country" in corpus.lower() or "國家" in corpus


# ── Mixed-currency failure state (multi-country) ─────────────────────────
def test_mixed_currency_three_country_fx_unavailable_stops(mock_download, monkeypatch):
    mock_download(["VOO", "0050", "VUSA"])
    import src.fx as fx_mod

    def _fake_convert_unavailable(prices_df, ticker_currency_map, base_currency, start_date, end_date):
        return {
            "converted_prices": prices_df[["VOO"]], "unavailable_tickers": ["0050", "VUSA"],
            "fx_source": "TEST-FX", "conversion_method": "TEST-METHOD",
            "base_currency": base_currency, "currency_adjusted": True,
        }
    monkeypatch.setattr(fx_mod, "convert_prices_to_base_currency", _fake_convert_unavailable)

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _build(at, ["United States", "Taiwan", "United Kingdom"], ["VOO", "0050", "VUSA"], base_currency="USD")

    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "FX Data Unavailable" in corpus
    assert "0050" in corpus and "VUSA" in corpus
    # Must not have silently produced a "successful" mixed-currency result.
    assert "current_portfolio" not in at.session_state
