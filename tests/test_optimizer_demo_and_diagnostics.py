"""
Portfolio Optimizer demo-default + covariance-diagnostics UI tests
(Issue #45 item 1): a fresh United-States-only session starts from the
shared cross-asset demo portfolio, and a severely ill-conditioned /
near-collinear ETF selection surfaces a visible warning instead of
silently optimizing (or claiming SLSQP "randomly" chose a corner).

No real network call: src.data_loader.download_etf_data is monkeypatched.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.demo_portfolio import DIVERSIFIED_DEMO_TICKERS


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _independent_fake_download(tickers, start_date, end_date, price_field="Close"):
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for tk in tickers:
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    return pd.DataFrame(data, index=dates)


def _collinear_fake_download(tickers, start_date, end_date, price_field="Close"):
    """Every requested ticker gets the EXACT SAME price series -- perfectly
    collinear returns, guaranteed rank-deficient raw covariance."""
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(123)
    base = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    return pd.DataFrame({tk: base for tk in tickers}, index=dates)


@pytest.fixture()
def independent_prices(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _independent_fake_download)


@pytest.fixture()
def collinear_prices(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _collinear_fake_download)


def test_fresh_us_session_defaults_to_diversified_demo_tickers(independent_prices):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    ms = next(m for m in at.multiselect if m.key == "selected_etfs_portfolio_United States")
    assert set(ms.value) == set(DIVERSIFIED_DEMO_TICKERS)


def test_demo_rationale_caption_shown_for_default_selection(independent_prices):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    captions = "\n".join(c.value for c in at.caption)
    assert "Cross-asset demonstration portfolio" in captions
    assert "not investment advice" in captions


def test_demo_caption_disappears_once_user_changes_selection(independent_prices):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    ms = next(m for m in at.multiselect if m.key == "selected_etfs_portfolio_United States")
    ms.set_value(["VOO", "QQQ"])
    at.run()
    assert at.exception == []
    captions = "\n".join(c.value for c in at.caption)
    assert "Cross-asset demonstration portfolio" not in captions


def test_severe_covariance_warning_shown_for_collinear_selection(collinear_prices):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    ms = next(m for m in at.multiselect if m.key == "selected_etfs_portfolio_United States")
    ms.set_value(["VOO", "VTI", "SPY"])
    at.run()

    run_btn = next(b for b in at.button if (b.label or "") == "Build Optimized Portfolio")
    run_btn.click()
    at.run()
    assert at.exception == []

    warnings_corpus = "\n".join(w.value for w in at.warning)
    assert "highly collinear" in warnings_corpus or "ill-conditioned" in warnings_corpus
    # Must use measured wording, never an unconditional "singular" claim.
    assert "singular" not in warnings_corpus.lower()
    # May discuss "randomly" only to explicitly NEGATE it -- must never
    # assert as fact that SLSQP randomly picked a corner.
    lower = warnings_corpus.lower()
    if "randomly" in lower:
        assert "not the solver \"randomly\"" in lower or "not \"randomly\"" in lower or "never" in lower


def test_no_covariance_warning_for_genuinely_diversified_selection(independent_prices):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    run_btn = next(b for b in at.button if (b.label or "") == "Build Optimized Portfolio")
    run_btn.click()
    at.run()
    assert at.exception == []
    warnings_corpus = "\n".join(w.value for w in at.warning)
    assert "highly collinear" not in warnings_corpus
