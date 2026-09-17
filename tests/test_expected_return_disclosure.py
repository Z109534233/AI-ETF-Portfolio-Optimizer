"""
Expected-return estimator disclosure tests (Issue #45 item 6):
run_optimization() uses returns_df.mean().values * 252 (historical
arithmetic mean daily return, annualized) -- the methodology text on both
Portfolio Optimizer and Home must say so accurately, name the estimator's
sensitivity/estimation-error, and must NOT imply Black-Litterman or
shrinkage estimation is already implemented.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.methodology import PORTFOLIO_OPTIMIZATION_METHODOLOGY
from src.portfolio_optimizer import run_optimization


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def test_methodology_estimator_field_matches_actual_implementation():
    m = PORTFOLIO_OPTIMIZATION_METHODOLOGY["expected_return"]
    assert "arithmetic mean" in m["estimator"].lower()
    assert "252" in m["annualization"]


def test_run_optimization_actually_uses_arithmetic_mean_times_252():
    """Ties the methodology's stated estimator directly to the real
    implementation so the two can never silently drift apart."""
    dates = pd.bdate_range("2022-01-01", periods=300)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates))),
        "B": 100 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(dates))),
    }, index=dates)
    result = run_optimization(df, method="Equal Weight", risk_free_rate=0.04)

    returns_df = df.pct_change(fill_method=None).dropna(how="all")
    expected_mean_returns = returns_df.mean().values
    expected_ret = float(np.dot([0.5, 0.5], expected_mean_returns) * 252)
    assert result["expected_return"] == pytest.approx(expected_ret, abs=1e-9)


def test_i18n_methodology_discloses_estimation_sensitivity_and_no_bl():
    from src.i18n import TRANSLATIONS
    for lang in ("zh-TW", "en"):
        text = TRANSLATIONS[lang]["opt_methodology_return_desc"]
        low = text.lower()
        assert "252" in text
        # Must name the estimator's fragility/sensitivity.
        assert ("fragile" in low or "sensitive" in low or "敏感" in text or "脆弱" in text)
        # Must explicitly say shrinkage/Black-Litterman are NOT implemented,
        # never imply they already are.
        assert "black-litterman" in low or "shrinkage" in low or "縮減估計" in text or "black-litterman" in text.lower()
        assert "not" in low or "未" in text or "尚未" in text


def test_home_demo_shows_the_same_estimator_caveat(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        dates = pd.bdate_range(start_date, end_date)
        rng = np.random.default_rng(9)
        return pd.DataFrame(
            {tk: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))) for tk in tickers},
            index=dates,
        )

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: {
        "rate": 0.04, "observed_date": "2026-01-01", "series_id": "DGS3MO",
        "source": "FRED", "status": "live", "reason": None,
    })

    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    captions = "\n".join(c.value for c in at.caption)
    assert "arithmetic mean" in captions.lower()
    assert "not a forecast" in captions.lower() or "demonstration only" in captions.lower()


def test_no_claim_that_black_litterman_or_shrinkage_is_currently_used():
    from src.i18n import TRANSLATIONS
    forbidden_affirmative = ["uses black-litterman", "uses shrinkage", "已採用縮減估計", "已使用 black-litterman"]
    for lang, mapping in TRANSLATIONS.items():
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            low = value.lower()
            for phrase in forbidden_affirmative:
                assert phrase not in low, (lang, key, value)
