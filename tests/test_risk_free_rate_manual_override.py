"""
Manual risk-free-rate override tests (Issue #46 review item 2).

The risk-free-rate slider on ETF Analysis / Portfolio Optimizer / Risk
Analytics initializes to the fetched/fallback FRED DGS3MO rate, but a user
can move it. Before this fix, the caption and Portfolio Optimizer's
persisted experiment metadata kept describing that moved value as if it
were still FRED-sourced. These tests assert the caption/metadata now
distinguish a "live default" from a "manual override" whenever the
selected rate differs from the fetched DGS3MO rate.

No real network call: src.data_loader.download_etf_data and
src.risk_free_rate.get_cached_risk_free_rate are monkeypatched.
"""

import os
import sys
from unittest.mock import patch

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
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(dates)))
    return pd.DataFrame(data, index=dates)


# Fetched DGS3MO != the slider value these tests move to (0.08, 0.07, 0.0)
# below -- exercising the actual "fetched rate differs from the selected
# slider value" scenario the review asked for.
FIXED_RF = {
    "rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
    "source": "FRED", "status": "live", "reason": None,
}


@pytest.fixture()
def mocked_inputs(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: FIXED_RF)


def test_risk_analytics_slider_untouched_shows_fred_caption_only(mocked_inputs):
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    captions = "\n".join(c.value for c in at.caption)
    assert "FRED" in captions and "DGS3MO" in captions
    assert "override" not in captions.lower()


def test_risk_analytics_slider_moved_discloses_manual_override(mocked_inputs):
    """Regression for Issue #46 review item 2: moving the slider away from
    the fetched DGS3MO rate must be captioned as a manual override, never
    as if it were still the live FRED value."""
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    slider = next(s for s in at.slider if s.key == "risk_free_rate_slider")
    assert slider.value == pytest.approx(4.0, abs=0.15)  # untouched default (4.11 rounds to 0.25 steps)

    slider.set_value(7.0)  # well away from the mocked fetched 4.11%
    at.run()
    assert at.exception == []

    captions = "\n".join(c.value for c in at.caption)
    assert "7.00%" in captions
    assert "override" in captions.lower() or "manual" in captions.lower()
    # The FRED reference rate/date must still be visible for context.
    assert "4.11%" in captions and "FRED" in captions and "DGS3MO" in captions


def test_etf_analysis_slider_moved_discloses_manual_override(mocked_inputs):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    slider = next((s for s in at.slider if s.key == "etf_risk_free_rate_slider"), None)
    assert slider is not None
    slider.set_value(0.0)  # well away from the mocked fetched 4.11%
    at.run()
    assert at.exception == []

    captions = "\n".join(c.value for c in at.caption)
    assert "0.00%" in captions
    assert "override" in captions.lower() or "manual" in captions.lower()


def _switch_to_save_workspace(at):
    ws = next(w for w in at.segmented_control if w.key == "opt_workspace")
    ws.set_value("Save & Actions")
    at.run()


def test_portfolio_optimizer_persists_manual_override_metadata(mocked_inputs):
    """The saved experiment metadata must record
    risk_free_rate_is_manual_override=True and a "manual_override" status
    once the slider has been moved away from the fetched rate BEFORE
    building the portfolio -- while retaining the FRED reference rate
    separately for context (Issue #46 review item 2)."""
    import src.database as db_mod

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    slider = next(s for s in at.slider if s.key == "opt_risk_free_rate_slider")
    slider.set_value(8.0)  # well away from the mocked fetched 4.11%
    at.run()
    assert at.exception == []

    run_btn = next(b for b in at.button if b.key == "opt_run_optimization_btn")
    run_btn.click()
    at.run()
    assert at.exception == []

    _switch_to_save_workspace(at)
    save_btn = next(b for b in at.button if b.key == "opt_save_export_btn")
    with patch.object(db_mod, "save_portfolio", return_value=True) as mock_save, \
         patch.object(db_mod, "find_duplicate_portfolio", return_value=None):
        save_btn.click()
        at.run()

    assert mock_save.called
    metadata = mock_save.call_args.kwargs["metadata"]
    assert metadata["risk_free_rate"] == pytest.approx(0.08)
    assert metadata["risk_free_rate_is_manual_override"] is True
    assert metadata["risk_free_rate_status"] == "manual_override"
    assert metadata["risk_free_rate_fred_reference_rate"] == pytest.approx(FIXED_RF["rate"])
    assert metadata["risk_free_rate_source"] == "FRED"


def test_portfolio_optimizer_untouched_slider_metadata_still_fred_sourced(mocked_inputs):
    """Without touching the slider, persisted metadata must still describe
    the assumption as FRED live-sourced, not a manual override -- this is
    the control case for the previous test."""
    import src.database as db_mod

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    run_btn = next(b for b in at.button if b.key == "opt_run_optimization_btn")
    run_btn.click()
    at.run()
    assert at.exception == []

    _switch_to_save_workspace(at)
    save_btn = next(b for b in at.button if b.key == "opt_save_export_btn")
    with patch.object(db_mod, "save_portfolio", return_value=True) as mock_save, \
         patch.object(db_mod, "find_duplicate_portfolio", return_value=None):
        save_btn.click()
        at.run()

    assert mock_save.called
    metadata = mock_save.call_args.kwargs["metadata"]
    assert metadata["risk_free_rate_is_manual_override"] is False
    assert metadata["risk_free_rate_status"] == "live"
    assert metadata["risk_free_rate_fred_reference_rate"] == pytest.approx(FIXED_RF["rate"])
