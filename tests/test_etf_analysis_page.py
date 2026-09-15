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
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
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


# ============================================================================
# Issue #39: ETF Analysis UX cleanup (remove 3rd-level tabs, merge compare
# tables, complete zh-TW i18n) regression tests.
# ============================================================================

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
    """Every ticker download returns 300 business days of data -- enough for
    every workspace (Rankings, Compare Mode, Investment Verdict, Deep
    Analysis) to compute non-degenerate signals, regardless of which
    tickers/dates the page's sidebar defaults actually request."""
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _full_history_price_frame(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def _goto_workspace(at, workspace):
    ws = next(w for w in at.segmented_control if w.key == "etf_analysis_workspace")
    ws.set_value(workspace).run()
    return at


_MEDAL_TROPHY_EMOJI = ("\U0001F947", "\U0001F948", "\U0001F949", "\U0001F3C6")  # 🥇 🥈 🥉 🏆


def _page_corpus(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(parts)


def test_price_chart_type_is_a_selectbox_not_a_third_tab_row(_mock_full_history_download):
    """Priority 1: the old Historical/Normalized/Cumulative segmented_control
    nested under Performance -> Price was a THIRD tab level. It must now be a
    plain st.selectbox, not another segmented_control/tab."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    _goto_workspace(at, "Performance")
    assert at.exception == []

    assert not [w for w in at.segmented_control if w.key == "etf_price_view"], (
        "The Price chart-type control must not be a segmented_control (that "
        "is the 3rd nested tab row this issue removes)."
    )
    assert not [w for w in at.tabs if getattr(w, "key", None) == "etf_price_view"]

    price_selects = [w for w in at.selectbox if w.key == "etf_price_view"]
    assert len(price_selects) == 1, "Expected exactly one chart-type selectbox under Performance -> Price"
    assert set(price_selects[0].options) == {"Historical Prices", "Normalized Comparison", "Cumulative Return"}


def test_rankings_view_has_exactly_one_ranking_comparison_table(_mock_full_history_download):
    """Priority 2: the old Rankings view rendered BOTH an "ETF Ranking" table
    and a duplicate "ETF Compare Score" table with independently-thresholded
    risk labels. Only one canonical ranking table (Rank/ETF/Score/Trend/
    Risk/Expected Return/Portfolio View) may remain."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _goto_workspace(at, "Compare")
    assert at.exception == []

    corpus = _page_corpus(at)
    assert "ETF Compare Score" not in corpus, "The duplicate ETF Compare Score section must be removed entirely"

    # The canonical table's <th>Quant Score</th> column header is unique to
    # it -- the ETF Analytical Summary cards above also show a "Quant Score"
    # label, but as a <div>, not a <th>, and the pairwise Compare Mode table
    # below has its own Metric/Winner headers -- so counting <th> matches
    # pins "exactly one" ranking table.
    assert len(re.findall(r"<th[^>]*>Quant Score</th>", corpus)) == 1


def test_zh_tw_investment_verdict_has_no_raw_english_categorical_values(_mock_full_history_download):
    """Priority 3: Investment Verdict / Compare / Ranking must show Chinese
    categorical values, not English labels mixed into Chinese UI."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    _goto_workspace(at, "Compare")
    assert at.exception == []

    corpus = _page_corpus(at)
    for leaked_value in (
        "Medium High", "Medium-to-Long Term", "Short-to-Medium Term", "Long Term",
        "Balanced Investors", "Growth Investors", "Conservative Investors",
        "Low Risk", "Medium Risk", "High Risk", "Very High Risk",
    ):
        assert leaked_value not in corpus, f"Raw English value {leaked_value!r} leaked into the zh-TW page"


def test_no_medal_or_trophy_emoji_anywhere_on_etf_analysis(_mock_full_history_download):
    """Other detail 1 & 2: no 🥇🥈🥉🏆 anywhere on ETF Analysis, including
    the Rankings table and the pairwise Compare Mode table."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _goto_workspace(at, "Compare")
    assert at.exception == []

    corpus = _page_corpus(at)
    for emoji in _MEDAL_TROPHY_EMOJI:
        assert emoji not in corpus, f"Medal/trophy emoji {emoji!r} must not appear on ETF Analysis"


def test_compare_mode_summarizes_wins_without_repeating_trophy_per_row(_mock_full_history_download):
    """Other detail 2: the pairwise Compare Mode table must not show a
    trophy in every winner row; instead a single concise sentence below the
    table states how many of the 6 metrics each ETF leads."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    _goto_workspace(at, "Compare")
    assert at.exception == []

    captions = "\n".join(c.value for c in at.caption)
    assert (" of 6 metrics" in captions) or ("leads all 6 metrics" in captions), (
        "Expected a concise win-count summary sentence below the pairwise Compare Mode table"
    )


def test_overview_quantitative_insight_uses_fixed_information_accent(_mock_full_history_download):
    """Other detail 3: the ordinary Quantitative Insight panel on Overview
    must use the fixed informational accent (var(--primary)), never the
    Trend Signal's own color (which can be amber for a Neutral trend)."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    corpus = _page_corpus(at)
    assert '<div class="insight-panel" style="border-left-color:var(--primary);">' in corpus, (
        "Overview's Quantitative Insight panel must render with the fixed "
        "informational accent color, not an accent inherited from Trend Signal"
    )
