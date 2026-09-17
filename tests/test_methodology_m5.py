"""
M5 -- ETF Analysis Methodology (Issue #41 item A, updated for Issue #45
item 3's live risk-free-rate service): deterministic tests.

Covers:
  - src/methodology.py's ETF_ANALYSIS_METHODOLOGY metadata and
    validate_etf_analysis_window() in isolation
  - i18n parity (zh-TW / en) for every new etf_methodology_* key
  - the rendered Methodology & Assumptions panel on pages/1_ETF_Analysis.py:
    it must show the CURRENT runtime risk-free rate and its live/fallback
    provenance, and must disclose the actual risk-free-rate usage scope
    (Sharpe + Sortino, not an unverified claim)
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
from src.financial_metrics import annualized_return
from src import risk_free_rate as rf_mod


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
    assert "FRED" in m["risk_free_rate"]["nature"] and "fallback" in m["risk_free_rate"]["nature"]
    assert "Sharpe" in m["risk_free_rate"]["usage"] and "Sortino" in m["risk_free_rate"]["usage"]
    assert "auto_adjust=True" in m["price_source"]["adjustment"]
    assert "forecast" in m["limitation"]


# ── Annualization methodology text must match the actual CAGR implementation
def test_annualization_metadata_describes_cagr_not_mean_return_approximation():
    ann = ETF_ANALYSIS_METHODOLOGY["expected_return_and_volatility"]["annualization"]
    assert "CAGR" in ann
    assert "NOT a mean-daily-return" in ann
    assert "sqrt(252)" in ann


def test_annualized_return_matches_cagr_formula_not_mean_daily_times_252():
    """Ties the methodology's stated formula directly to
    src.financial_metrics.annualized_return()'s actual implementation, so the
    methodology text can never silently drift back to describing a
    mean-daily-return x252 approximation while the code stays CAGR-based
    (or vice versa)."""
    idx = pd.date_range("2023-01-02", periods=253, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)

    actual = annualized_return(prices)
    n_years = len(prices) / 252
    expected_cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1
    assert actual == pytest.approx(expected_cagr)

    mean_daily_times_252_approximation = prices.pct_change().dropna().mean() * 252
    assert actual != pytest.approx(mean_daily_times_252_approximation, rel=1e-2)


def test_i18n_annualization_value_describes_cagr_in_both_languages():
    en = TRANSLATIONS["en"]["etf_methodology_annualization_value"]
    zh = TRANSLATIONS["zh-TW"]["etf_methodology_annualization_value"]
    assert "CAGR" in en
    assert "not a mean-daily-return" in en
    assert "CAGR" in zh
    # the zh-TW copy must mention the old mean-return approximation only as
    # an explicit negation ("而非...的近似算法" -- "not the ... approximation"),
    # never as an affirmative statement of how return is actually annualized.
    assert "而非每日簡單報酬率平均值" in zh


# ── validate_etf_analysis_window() ───────────────────────────────────────
def test_validation_passes_with_enough_observations():
    result = validate_etf_analysis_window(250)
    assert result["is_valid"] is True
    assert result["issues"] == []


def test_validation_fails_with_too_few_observations():
    result = validate_etf_analysis_window(5)
    assert result["is_valid"] is False
    assert any("5" in issue for issue in result["issues"])


def test_validation_does_not_overclaim_statistical_stability():
    """The helper only counts usable observations -- it never re-derives
    return/volatility/ratios -- so its issue text must describe a minimum
    usability threshold, not a claim of statistical stability."""
    result = validate_etf_analysis_window(5)
    assert not any("stable" in issue.lower() for issue in result["issues"])


def test_i18n_validation_pass_text_does_not_overclaim_numerical_consistency():
    """The success copy must describe only what validate_etf_analysis_window()
    actually checks (a usable-history/window sufficiency check), never a
    claim that the displayed return/volatility/ratio figures were
    independently re-derived or reconciled."""
    en = TRANSLATIONS["en"]["etf_methodology_validation_pass"]
    zh = TRANSLATIONS["zh-TW"]["etf_methodology_validation_pass"]
    assert "numerically consistent" not in en
    assert "does not independently re-derive or reconcile" in en
    assert "數值上一致" not in zh
    assert "並非對上方數值的獨立重新計算或驗證" in zh


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


# ── Rendered panel: honest, dynamic risk-free-rate disclosure ────────────
@pytest.fixture
def _mock_live_rf(monkeypatch):
    """Deterministic live FRED observation -- no real network call."""
    fixed = {
        "rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
        "source": "FRED", "status": "live", "reason": None,
    }
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: fixed)


def test_methodology_panel_shows_live_fred_provenance(_mock_full_history_download, _mock_live_rf):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    expander_labels = [e.label for e in at.expander]
    assert "Methodology & Assumptions" in expander_labels

    corpus = _methodology_expander_corpus(at, "Methodology & Assumptions")
    # The slider defaults to the mocked live rate -- shown verbatim.
    assert "4.11%" in corpus
    # Must disclose the actual live source/series/as-of date dynamically.
    assert "FRED" in corpus and "DGS3MO" in corpus and "2026-09-14" in corpus
    # Must disclose the ACTUAL usage scope (Sharpe + Sortino), not vague text.
    assert "Sharpe Ratio" in corpus and "Sortino Ratio" in corpus


def test_methodology_panel_zh_tw_shows_live_fred_provenance(_mock_full_history_download, _mock_live_rf):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    assert at.exception == []

    expander_labels = [e.label for e in at.expander]
    assert "方法論與假設" in expander_labels

    corpus = _methodology_expander_corpus(at, "方法論與假設")
    assert "4.11%" in corpus
    assert "FRED" in corpus and "DGS3MO" in corpus
    assert "夏普比率" in corpus and "Sortino" in corpus


def test_methodology_panel_discloses_fallback_status_when_live_fetch_fails(
    _mock_full_history_download, monkeypatch,
):
    fallback = {
        "rate": 0.05, "observed_date": None, "series_id": "DGS3MO",
        "source": "FRED", "status": "fallback", "reason": "simulated network failure",
    }
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: fallback)
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = _methodology_expander_corpus(at, "Methodology & Assumptions")
    assert "5.00%" in corpus
    assert "fallback" in corpus.lower()


def test_methodology_panel_validation_success_with_sufficient_history(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    successes = "\n".join(s.value for s in at.success)
    assert "Validation passed" in successes
