"""
M5 -- ETF Analysis Methodology (Issue #41 item A): deterministic tests.

Covers:
  - src/methodology.py's ETF_ANALYSIS_METHODOLOGY metadata and
    validate_etf_analysis_window() in isolation
  - i18n parity (zh-TW / en) for every new etf_methodology_* key
  - the rendered Methodology & Assumptions panel on pages/1_ETF_Analysis.py:
    it must show the CURRENT runtime risk-free rate, must NOT claim a live
    Treasury/FRED yield source, and must disclose the actual risk-free-rate
    usage scope (Sharpe + Sortino, not an unverified claim)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.methodology import ETF_ANALYSIS_METHODOLOGY, validate_etf_analysis_window
from src.i18n import TRANSLATIONS


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _full_history_price_frame(tickers, n=300, start="2023-01-02"):
    idx = pd.date_range(start, periods=n, freq="B")
    data = {}
    for i, tk in enumerate(tickers):
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        drift = 0.0003 + (i % 5) * 0.0001
        data[tk] = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def _mock_full_history_download(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _full_history_price_frame(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def _page_corpus(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [s.value for s in at.success] + [w.value for w in at.warning]
    return "\n".join(parts)


def _methodology_expander_corpus(at, label):
    """Scope the corpus to just the Methodology & Assumptions expander's own
    content -- the full page corpus can legitimately contain the substring
    "Treasury" elsewhere (e.g. TLT's real ETF name, "iShares 20+ Year
    Treasury Bond ETF", in an unrelated search-filter preview), which is not
    the risk-free-rate honesty claim this test is actually checking."""
    exp = next(e for e in at.expander if e.label == label)
    parts = [m.value for m in exp.markdown] + [c.value for c in exp.caption]
    parts += [s.value for s in exp.success] + [w.value for w in exp.warning]
    return "\n".join(parts)


# ── ETF_ANALYSIS_METHODOLOGY metadata matches actual implementation ─────
def test_methodology_metadata_matches_actual_implementation():
    m = ETF_ANALYSIS_METHODOLOGY
    assert "252" in m["expected_return_and_volatility"]["annualization"]
    assert "NOT" in m["risk_free_rate"]["nature"] and "Treasury" in m["risk_free_rate"]["nature"]
    assert "Sharpe" in m["risk_free_rate"]["usage"] and "Sortino" in m["risk_free_rate"]["usage"]
    assert "auto_adjust=True" in m["price_source"]["adjustment"]
    assert "forecast" in m["limitation"]


# ── validate_etf_analysis_window() ───────────────────────────────────────
def test_validation_passes_with_enough_observations():
    result = validate_etf_analysis_window(250)
    assert result["is_valid"] is True
    assert result["issues"] == []


def test_validation_fails_with_too_few_observations():
    result = validate_etf_analysis_window(5)
    assert result["is_valid"] is False
    assert any("5" in issue for issue in result["issues"])


# ── i18n parity for every new etf_methodology_* key ──────────────────────
def test_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "etf_methodology_title", "etf_methodology_subtitle",
        "etf_methodology_history_label", "etf_methodology_history_value",
        "etf_methodology_annualization_label", "etf_methodology_annualization_value",
        "etf_methodology_benchmark_label",
        "etf_methodology_rfr_label", "etf_methodology_rfr_value",
        "etf_methodology_rfr_usage_label", "etf_methodology_rfr_usage_value",
        "etf_methodology_data_source_label", "etf_methodology_data_source_value",
        "etf_methodology_limitation_label", "etf_methodology_limitation_value",
        "etf_methodology_validation_pass", "etf_methodology_validation_fail",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""


# ── Rendered panel: honest risk-free-rate disclosure ─────────────────────
def test_methodology_panel_shows_current_rfr_and_no_treasury_claim(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    expander_labels = [e.label for e in at.expander]
    assert "Methodology & Assumptions" in expander_labels

    corpus = _methodology_expander_corpus(at, "Methodology & Assumptions")
    # Default risk_free_rate slider value is 5.00% -- must be shown verbatim.
    assert "5.00%" in corpus
    # Must NOT claim an automatically-fetched Treasury/FRED yield -- the only
    # permitted mention of "Treasury" is the explicit NEGATION ("does not
    # automatically sync a live Treasury yield"), never an affirmative
    # sourcing claim.
    assert "does not automatically sync a live Treasury yield" in corpus
    assert "FRED" not in corpus
    # Must disclose the ACTUAL usage scope (Sharpe + Sortino), not vague text.
    assert "Sharpe Ratio" in corpus and "Sortino Ratio" in corpus


def test_methodology_panel_zh_tw_no_treasury_claim(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    assert at.exception == []

    expander_labels = [e.label for e in at.expander]
    assert "方法論與假設" in expander_labels

    corpus = _methodology_expander_corpus(at, "方法論與假設")
    assert "5.00%" in corpus
    assert "公債" in corpus and "不會自動同步" in corpus
    assert "夏普比率" in corpus and "Sortino" in corpus


def test_methodology_panel_validation_success_with_sufficient_history(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    successes = "\n".join(s.value for s in at.success)
    assert "Validation passed" in successes
