"""
Sample-data disclosure regression tests (Issue #46 review item 1).

Before this fix, app.py decided whether to show its sample-data warning by
checking `raw_prices.empty` on download_etf_data()'s return value. But
download_etf_data() falls back to a fully simulated (non-empty) DataFrame
when every requested ticker fails to download -- so `.empty` was always
False in that path, and simulated numbers could render on Home as if they
were live. src.data_loader.download_etf_data_with_status() now returns an
explicit (DataFrame, is_sample_data) tuple, and app.py uses it.

No real network call: the download functions are monkeypatched throughout.
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
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


# ── src.data_loader.download_etf_data_with_status() unit tests ──────────────

def test_download_with_status_flags_sample_data_when_every_ticker_fails(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "_download_single_ticker", lambda *a, **k: None)
    data_loader_mod.download_etf_data_with_status.clear()

    df, is_sample = data_loader_mod.download_etf_data_with_status(
        ["AAA", "BBB"], "2024-01-01", "2024-06-01",
    )
    assert is_sample is True
    # Regression check: the whole point is that this fallback DataFrame is
    # NOT empty, so a caller relying on `.empty` would (wrongly) conclude
    # this is live data.
    assert not df.empty
    assert set(df.columns) == {"AAA", "BBB"}


def test_download_with_status_reports_live_when_at_least_one_ticker_succeeds(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_single(ticker, start_date, end_date, price_field="Close"):
        if ticker == "AAA":
            return pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2024-01-01", periods=3))
        return None

    monkeypatch.setattr(data_loader_mod, "_download_single_ticker", _fake_single)
    data_loader_mod.download_etf_data_with_status.clear()

    df, is_sample = data_loader_mod.download_etf_data_with_status(
        ["AAA", "BBB"], "2024-01-01", "2024-01-05",
    )
    assert is_sample is False
    assert "AAA" in df.columns
    assert "BBB" not in df.columns  # excluded, not simulated in its place


def test_download_etf_data_stays_backward_compatible_returning_just_the_dataframe(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "_download_single_ticker", lambda *a, **k: None)
    data_loader_mod.download_etf_data.clear()
    data_loader_mod.download_etf_data_with_status.clear()

    df = data_loader_mod.download_etf_data(["AAA"], "2024-01-01", "2024-06-01")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


# ── app.py (Home) disclosure ─────────────────────────────────────────────

def test_home_discloses_sample_data_when_every_live_fetch_fails(monkeypatch):
    """The actual bug: app.py must warn AND tag the hero/ticker strip as
    sample data when the loader had to fall back, even though the
    fallback DataFrame it receives is not empty."""
    import src.data_loader as data_loader_mod

    def _fake_download_with_status(tickers, start_date, end_date, price_field="Close"):
        return data_loader_mod._generate_sample_data(tickers, start_date, end_date), True

    monkeypatch.setattr(data_loader_mod, "download_etf_data_with_status", _fake_download_with_status)

    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    warnings = "\n".join(w.value for w in at.warning)
    assert "sample data" in warnings.lower()

    # The stylesheet itself always DEFINES the .hero-preview-sample-tag CSS
    # class (loaded on every page regardless of data source) -- assert the
    # actual rendered <span>, not just the class-name substring, or this
    # would pass even when the tag is never applied to anything.
    markdown_html = "\n".join(m.value for m in at.markdown)
    assert '<span class="hero-preview-sample-tag">' in markdown_html


def test_home_does_not_disclose_sample_data_when_live_fetch_succeeds(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_download_with_status(tickers, start_date, end_date, price_field="Close"):
        dates = pd.bdate_range(start_date, end_date)
        data = {}
        for tk in tickers:
            rng = np.random.default_rng(abs(hash(tk)) % (2**32))
            data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(dates)))
        return pd.DataFrame(data, index=dates), False

    monkeypatch.setattr(data_loader_mod, "download_etf_data_with_status", _fake_download_with_status)

    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    warnings = "\n".join(w.value for w in at.warning)
    assert "sample data" not in warnings.lower()

    markdown_html = "\n".join(m.value for m in at.markdown)
    assert '<span class="hero-preview-sample-tag">' not in markdown_html
