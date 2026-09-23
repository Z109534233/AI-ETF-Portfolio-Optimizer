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


# ============================================================================
# Issue #40: PR #40 follow-up review -- remaining i18n/style leaks flagged
# after the Issue #39 cleanup (raw English ETF DNA labels, raw "Momentum" in
# zh-TW, and decorative 🟢🟡🔴 trend emoji).
# ============================================================================

_TREND_EMOJI = ("\U0001F7E2", "\U0001F7E1", "\U0001F534")  # 🟢 🟡 🔴


def _goto_compare_view(at, view):
    nav = next(w for w in at.segmented_control if w.key == "etf_compare_view")
    nav.set_value(view).run()
    return at


def _momentum_dominant_price_frame(tickers, n=300, start="2023-01-02"):
    """The first ticker dominates every Quant Score input (annualized
    return, Sharpe, momentum, volatility) over every other ticker, so it is
    guaranteed to rank #1 AND to include Momentum among its top-3 "Why #1?"
    reasons regardless of which tickers the page's sidebar defaults
    request. Momentum(10) is unaffected by the +-eps oscillation added for
    realistic (non-zero) daily volatility, because 10 is even: the
    (-1)**t oscillation term cancels exactly 10 trading days apart."""
    idx = pd.date_range(start, periods=n, freq="B")
    t = np.arange(n)
    data = {}
    for i, tk in enumerate(tickers):
        if i == 0:
            data[tk] = 100 * np.exp(0.0012 * t) * (1 + 0.003 * (-1.0) ** t)
        else:
            data[tk] = 100 * np.exp(-0.0003 * t) * (1 + 0.02 * (-1.0) ** t)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def _mock_momentum_dominant_download(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _momentum_dominant_price_frame(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def test_zh_tw_etf_dna_dimension_labels_are_localized(_mock_full_history_download):
    """The Correlation workspace's ETF DNA cards used to hardcode
    ("Growth", "Risk", "Momentum", "Diversification", "Liquidity") as
    literal English labels regardless of page language. In zh-TW they must
    render as 成長/風險/動能/分散程度/流動性."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    _goto_workspace(at, "Compare")
    _goto_compare_view(at, "Correlation")
    assert at.exception == []

    corpus = _page_corpus(at)
    for localized in ("成長", "風險", "動能", "分散程度", "流動性"):
        assert localized in corpus, f"Expected localized ETF DNA dimension label {localized!r} in zh-TW"
    for leaked_value in ("Growth", "Risk", "Momentum", "Diversification", "Liquidity"):
        assert leaked_value not in corpus, f"Raw English ETF DNA dimension label {leaked_value!r} leaked into the zh-TW page"


def test_zh_tw_why_number_one_localizes_momentum_reason(_mock_momentum_dominant_download):
    """The "Why #1?" panel's Momentum reason was appended as the literal,
    language-independent string "Momentum". It must show the localized 動能
    label (via the same t_compare_metric() helper used in Compare Mode)
    when the page language is zh-TW."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    _goto_workspace(at, "Compare")
    assert at.exception == []

    corpus = _page_corpus(at)
    assert "為什麼第一" in corpus, "Expected the Why #1? panel to render in Compare -> Rankings"
    assert "動能" in corpus, "Expected the localized Momentum reason (動能) to appear in the Why #1? panel"
    assert "Momentum" not in corpus, "Raw English 'Momentum' must not leak into the zh-TW page"


def test_no_trend_status_light_emoji_anywhere_on_etf_analysis(_mock_full_history_download):
    """The trend color (var(--success)/var(--warning)/var(--danger)) must be
    kept, but the decorative 🟢🟡🔴 traffic-light emoji must not render on
    the ETF Analytical Summary cards or the Rankings table -- the page
    moved away from emoji status decoration. The 📊 page icon is unrelated
    and untouched."""
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = _page_corpus(at)
    for emoji in _TREND_EMOJI:
        assert emoji not in corpus, f"Trend status-light emoji {emoji!r} must not appear on Overview"

    _goto_workspace(at, "Compare")
    assert at.exception == []
    corpus = _page_corpus(at)
    for emoji in _TREND_EMOJI:
        assert emoji not in corpus, f"Trend status-light emoji {emoji!r} must not appear on the Compare/Rankings view"



def test_overview_quant_score_badge_explains_actual_formula(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = _page_corpus(at)
    assert "A 0–100 rule-based heuristic score, not a probability." in corpus
    assert "annualized return contributes up to ±22" in corpus
    assert "Sharpe Ratio up to ±18" in corpus
    assert "10-day momentum up to ±12" in corpus
    assert "adding up to 8 or subtracting up to 22" in corpus
    assert "maximum drawdown can subtract up to another 22" in corpus
    assert "&#9432;" in corpus


def test_overview_interpretation_button_is_not_mislabeled_as_ai(_mock_full_history_download):
    at = _apptest_from_file("pages/1_ETF_Analysis.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    labels = [b.label for b in at.button]
    assert "Generate Interpretation" in labels
    assert "AI Interpretation" not in labels
