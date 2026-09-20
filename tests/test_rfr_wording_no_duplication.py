"""
Regression tests for Issue #48 item 2: the ETF Analysis and Portfolio
Optimizer "Risk-Free Rate Assumption" methodology sentence used to
interpolate BOTH `{rf}` (the selected rate) AND `{provenance}` (which
ALSO prints that same rate, plus -- in the fallback case -- its own
parenthetical), producing rendered text like:

    目前無風險利率假設為 5.00%（5.00%（預設備用值，無法取得即時 DGS3MO 資料））
    The current risk-free rate assumption is 5.00% (5.00% (fallback default...))

The fix drops the redundant `{rf}` placeholder so `{provenance}` (which
already states the applicable rate) is the ONLY place the percentage
appears. These tests force the emergency-fallback path (the scenario that
originally produced the nested/duplicated parentheses) and assert the
rendered methodology text never contains a duplicated rate string, in
both zh-TW and en, on both affected pages.

No real network call: src.data_loader.download_etf_data and
src.risk_free_rate.get_cached_risk_free_rate are monkeypatched.
"""

import os
import re
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


def _full_history_price_frame(tickers, n=300, start="2023-01-02"):
    idx = pd.date_range(start, periods=n, freq="B")
    data = {}
    for i, tk in enumerate(tickers):
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        drift = 0.0003 + (i % 5) * 0.0001
        data[tk] = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    return pd.DataFrame(data, index=idx)


FALLBACK_RF = {
    "rate": 0.05, "observed_date": None, "series_id": "DGS3MO",
    "source": "FRED", "status": "fallback",
    "reason": "simulated: both live sources unavailable",
}


@pytest.fixture()
def mocked_fallback_inputs(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _full_history_price_frame(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: FALLBACK_RF)


# A duplicated/nested rate string looks like "5.00%（5.00%" or "5.00% (5.00%"
# -- the same percentage immediately followed by another opening bracket
# that itself starts with the identical percentage.
_DUPLICATED_RATE_PATTERN = re.compile(r"(\d+\.\d{2}%)\s*[（(]\s*\1")


def _assert_no_duplicated_rate(corpus, pct):
    assert not _DUPLICATED_RATE_PATTERN.search(corpus), (
        f"found a duplicated/nested rate string in rendered text: {corpus!r}"
    )
    assert f"{pct}（{pct}" not in corpus
    assert f"{pct} ({pct}" not in corpus
    # The fallback rate must still appear (just once, not duplicated).
    assert pct in corpus


def _methodology_expander_corpus(at, label):
    exp = next(e for e in at.expander if e.label == label)
    parts = [m.value for m in exp.markdown] + [c.value for c in exp.caption]
    parts += [s.value for s in exp.success] + [w.value for w in exp.warning]
    return "\n".join(parts)


@pytest.mark.parametrize("lang,expander_label", [("en", "Methodology & Assumptions"), ("zh-TW", "方法論與假設")])
def test_etf_analysis_methodology_never_duplicates_fallback_rate(mocked_fallback_inputs, lang, expander_label):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    corpus = _methodology_expander_corpus(at, expander_label)
    _assert_no_duplicated_rate(corpus, "5.00%")
    assert "fallback" in corpus.lower() or "備用值" in corpus


@pytest.mark.parametrize("lang,expander_label", [("en", "Methodology & Assumptions"), ("zh-TW", "方法論與假設")])
def test_portfolio_optimizer_methodology_never_duplicates_fallback_rate(mocked_fallback_inputs, lang, expander_label):
    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    run_btn = next(b for b in at.button if b.key == "opt_run_optimization_btn")
    run_btn.click()
    at.run()
    assert at.exception == []

    corpus = _methodology_expander_corpus(at, expander_label)
    _assert_no_duplicated_rate(corpus, "5.00%")
    assert "fallback" in corpus.lower() or "備用值" in corpus


@pytest.mark.parametrize("lang", ["en", "zh-TW"])
def test_format_rate_provenance_used_alone_never_self_duplicates(lang):
    """Direct unit-level guard on the building block itself: whatever
    format_rate_provenance() returns must never contain the pattern
    'X.XX%(X.XX%' / 'X.XX%（X.XX%' regardless of status, so wrapping it in
    a template that shows it exactly once (the fix applied to both
    affected pages) can never reintroduce the nested duplication."""
    from src.risk_free_rate import format_rate_provenance

    for rf in (
        {"rate": 0.05, "observed_date": None, "series_id": "DGS3MO", "source": "FRED", "status": "fallback", "reason": "x"},
        {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO", "source": "FRED", "status": "live", "reason": None},
        {"rate": 0.042, "observed_date": "2026-09-14", "series_id": "^IRX", "source": "Yahoo Finance", "status": "secondary_live", "reason": None},
    ):
        text = format_rate_provenance(rf, lang)
        assert not _DUPLICATED_RATE_PATTERN.search(text), text
        text_override = format_rate_provenance(rf, lang, selected_rate=0.09)
        assert not _DUPLICATED_RATE_PATTERN.search(text_override), text_override
