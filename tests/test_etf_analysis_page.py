"""
Regression test for pages/1_ETF_Analysis.py's Overview thin-history
disclosure (Issue #20 release-gate review, automated PR reviewer finding).

The Compare workspace already warns when a selected ticker has fewer than
src.etf_signals.MIN_RELIABLE_HISTORY_POINTS valid price points (its Quant
Score/Trend Signal/Portfolio View are numerically unstable below that), but
the Overview workspace -- which every fresh page load lands on -- called
the exact same compute_quant_signals()-backed card renderer with no such
check. A short-history focus ticker could show a seemingly normal score
with no caveat. Fixed by mirroring Compare's has_sufficient_history() +
etf_thin_history_warning check in the Overview branch.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest


def _short_price_frame(tickers, n=5):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({tk: [100.0 + i for i in range(n)] for tk in tickers}, index=idx)


@pytest.fixture
def _mock_thin_history_download(monkeypatch):
    """Every ticker download returns only `n` valid rows -- far under
    MIN_RELIABLE_HISTORY_POINTS -- regardless of which tickers/dates the
    page's sidebar defaults actually request."""
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _short_price_frame(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def test_overview_shows_thin_history_warning_for_short_history_focus_ticker(_mock_thin_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    warnings = "\n".join(w.value for w in at.warning)
    assert "fewer than 10 trading days" in warnings, (
        "Overview (the default workspace on a fresh page load) must disclose "
        "thin history for its focus ticker exactly like the Compare workspace does"
    )
