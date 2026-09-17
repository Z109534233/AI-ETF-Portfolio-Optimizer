"""
Cross-page risk-free-rate wiring tests (Issue #45 item 3): Risk Analytics'
sidebar slider defaults to the live/fallback rate with its provenance
shown, and AI Advisor's custom-portfolio Sharpe ratio uses the same fetched
rate instead of a second, independently hard-coded 5%/0.05.

No real network call: src.data_loader.download_etf_data and
src.risk_free_rate.get_cached_risk_free_rate are monkeypatched. No test
here hard-codes today's actual live rate -- a fixed mock value is used.
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


def _fake_download(tickers, start_date, end_date, price_field="Close"):
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for tk in tickers:
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    return pd.DataFrame(data, index=dates)


FIXED_RF = {
    "rate": 0.0333, "observed_date": "2026-08-01", "series_id": "DGS3MO",
    "source": "FRED", "status": "live", "reason": None,
}


@pytest.fixture()
def mocked_inputs(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: FIXED_RF)


def test_risk_analytics_slider_defaults_to_mocked_live_rate(mocked_inputs):
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    slider = next(s for s in at.slider if s.key == "risk_free_rate_slider")
    assert slider.value == pytest.approx(3.33, abs=0.01)

    captions = "\n".join(c.value for c in at.caption)
    assert "3.33%" in captions
    assert "FRED" in captions and "DGS3MO" in captions and "2026-08-01" in captions


def test_ai_advisor_custom_portfolio_sharpe_uses_fetched_rate_not_hardcoded_005(mocked_inputs):
    at = _apptest_from_file("pages/6_AI_Advisor.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    gen_btn = next((b for b in at.button if (b.label or "") == "Generate AI Analysis"), None)
    assert gen_btn is not None
    gen_btn.click()
    at.run()
    assert at.exception == []

    portfolio_ctx = at.session_state["ai_result"]["context"]["portfolio"]
    assert portfolio_ctx["available"] is True

    # Recompute what the Sharpe ratio SHOULD be with the mocked rf vs. the
    # old hardcoded 0.05, using the actual annualized_return/volatility the
    # page itself stored -- if the page still hardcoded 0.05, this equality
    # would fail whenever the mocked rate differs from 0.05 (it does: 0.0333).
    from src.financial_metrics import sharpe_ratio
    # We can't re-derive the exact price series here, but we CAN assert the
    # stored Sharpe is consistent with (return - mocked_rf) / vol -- the
    # formula sharpe_ratio() itself uses -- rather than with 0.05.
    ret = portfolio_ctx["expected_return"]
    vol = portfolio_ctx["volatility"]
    stored_sharpe = portfolio_ctx["sharpe_ratio"]
    if vol:
        expected_with_mocked_rf = (ret - FIXED_RF["rate"]) / vol
        expected_with_old_hardcode = (ret - 0.05) / vol
        assert stored_sharpe == pytest.approx(expected_with_mocked_rf, abs=1e-9)
        if abs(FIXED_RF["rate"] - 0.05) > 1e-6:
            assert stored_sharpe != pytest.approx(expected_with_old_hardcode, abs=1e-9)
