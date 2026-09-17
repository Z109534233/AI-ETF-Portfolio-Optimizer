"""
Home vs. Portfolio Optimizer metric-consistency test (Issue #45 item 2).

For the SAME mocked price matrix, SAME mocked risk-free rate, and SAME
default date window, Home's Equal-Weight Demo headline numbers (Expected
Return / Volatility / Sharpe) must exactly match Portfolio Optimizer's own
Equal Weight result -- both now go through the identical
src.portfolio_optimizer.run_optimization(..., method="Equal Weight") engine
rather than two independently-computed formulas.

No real network call: src.data_loader.download_etf_data and
src.risk_free_rate.get_cached_risk_free_rate are both monkeypatched.
"""

import os
import re
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


def _fake_download(tickers, start_date, end_date, price_field="Close"):
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for tk in tickers:
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(dates)))
    return pd.DataFrame(data, index=dates)


def _fake_download_with_status(tickers, start_date, end_date, price_field="Close"):
    return _fake_download(tickers, start_date, end_date, price_field), False


FIXED_RF = {
    "rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
    "source": "FRED", "status": "live", "reason": None,
}


@pytest.fixture()
def mocked_inputs(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    # app.py (Home) reads download_etf_data_with_status() specifically (see
    # Issue #46 review item 1 -- it needs the explicit is_sample_data flag
    # download_etf_data() alone can't provide), so this must be mocked too
    # or Home falls through to a real (network-dependent) call.
    monkeypatch.setattr(data_loader_mod, "download_etf_data_with_status", _fake_download_with_status)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: FIXED_RF)


def _extract_values(html_blobs, css_class):
    pattern = re.compile(rf'class="{css_class}"[^>]*>([^<]+)<')
    values = []
    for blob in html_blobs:
        values.extend(pattern.findall(blob))
    return values


def _parse_pct(s):
    return float(s.strip().rstrip("%"))


def test_home_default_demo_tickers_are_the_diversified_set(mocked_inputs):
    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    ms = next(m for m in at.multiselect if m.key == "home_selected_etfs")
    assert set(ms.value) == {"VOO", "VXUS", "BND", "GLD", "TLT"}


def test_home_equal_weight_demo_matches_optimizer_equal_weight_result(mocked_inputs):
    # ── Home ──────────────────────────────────────────────────────────
    home_at = _apptest_from_file("app.py", default_timeout=180)
    home_at.session_state["language"] = "en"
    home_at.run()
    assert home_at.exception == []

    home_html = [m.value for m in home_at.markdown]
    hero_values = _extract_values(home_html, "hero-preview-metric-value")
    assert len(hero_values) == 3, hero_values
    home_ret, home_vol, home_sharpe = hero_values
    home_ret = _parse_pct(home_ret)
    home_vol = _parse_pct(home_vol)
    home_sharpe = float(home_sharpe.strip())

    # ── Portfolio Optimizer ──────────────────────────────────────────
    opt_at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    opt_at.session_state["language"] = "en"
    opt_at.run()
    assert opt_at.exception == []

    # Fresh United-States-only session already defaults the ETF selection
    # to the shared demo tickers (Issue #45 item 1) and the method
    # selectbox already defaults to "Equal Weight" -- matching Home exactly
    # with no extra widget interaction needed.
    ms = next(m for m in opt_at.multiselect if m.key == "selected_etfs_portfolio_United States")
    assert set(ms.value) == set(DIVERSIFIED_DEMO_TICKERS)

    run_btn = next(b for b in opt_at.button if (b.label or "") == "Build Optimized Portfolio")
    run_btn.click()
    opt_at.run()
    assert opt_at.exception == []
    assert opt_at.session_state["opt_result"]["method"] == "Equal Weight"

    opt_html = [m.value for m in opt_at.markdown]
    opt_ret = _extract_values(opt_html, "results-hero-metric-value")
    assert len(opt_ret) == 1, opt_ret
    opt_ret = _parse_pct(opt_ret[0])

    kpi_values = _extract_values(opt_html, "kpi-value")
    # scol1=Volatility, scol2=Sharpe, scol3=Diversification Ratio (see
    # pages/2_Portfolio_Optimizer.py's secondary KPI row, in that order).
    assert len(kpi_values) >= 2, kpi_values
    opt_vol = _parse_pct(kpi_values[0])
    opt_sharpe = float(kpi_values[1].strip())

    # Home rounds to 1 decimal place (%), Optimizer to 2 -- compare at the
    # coarser precision so this is an exact-engine-match check, not a
    # rounding-noise check.
    assert round(home_ret, 1) == pytest.approx(round(opt_ret, 1), abs=0.05)
    assert round(home_vol, 1) == pytest.approx(round(opt_vol, 1), abs=0.05)
    assert round(home_sharpe, 2) == pytest.approx(round(opt_sharpe, 2), abs=0.01)
