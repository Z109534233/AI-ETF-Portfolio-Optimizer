"""
Risk Analytics page -- reviewer-readiness regression tests (Issue #43
sections A-D).

Covers:
  - A: the bilingual scope notice distinguishes the page-local ETF
    selection/weights from the "Current Portfolio" handoff preview, and
    shows the actual counts for both.
  - B: the stress-test historical-coverage disclosure is dynamic, using the
    ACTUAL selected start_date/end_date, and flags named historical
    scenarios whose real event window falls outside that selection.
  - C: the stress-test table is simplified to exactly Scenario | Market
    Shock | Estimated Portfolio Impact, with portfolio beta shown once
    above the table instead of repeated on every row.
  - D: the "Methodology & Assumptions" expander has a distinct info cue
    that "More Risk Metrics" does not.

No real network call: src.data_loader.download_etf_data is monkeypatched
with deterministic synthetic price data (same pattern as
tests/test_portfolio_optimizer.py's _fake_download).
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


@pytest.fixture()
def no_network(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def _run_risk_analytics(lang="en", current_portfolio=None):
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = lang
    if current_portfolio is not None:
        at.session_state["current_portfolio"] = current_portfolio
    at.run()
    assert at.exception == []
    return at


CURRENT_PORTFOLIO_5 = {
    "portfolio_id": "history-1", "strategy": "Maximum Sharpe Ratio", "market": "United States",
    "tickers": ["AAA", "BBB", "CCC", "DDD", "EEE"],
    "weights": {"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2},
    "investment_amount": 10000.0, "expected_return": 0.1, "volatility": 0.15,
    "sharpe_ratio": 0.6, "generated_at": "2026-01-01T00:00:00+00:00",
}


# ── Completion-gate smoke: zh-TW + en render without exception ───────────

def test_risk_analytics_renders_en_with_mocked_data(no_network):
    _run_risk_analytics("en")


def test_risk_analytics_renders_zh_tw_with_mocked_data(no_network):
    _run_risk_analytics("zh-TW")


# ── Issue #43 item A: scope notice distinguishes page-local vs current ────

def test_scope_notice_shows_page_local_count_with_no_current_portfolio(no_network):
    at = _run_risk_analytics("en")
    infos = "\n".join(i.value for i in at.info)
    assert "page's own selection" in infos or "own selection of" in infos
    assert "no \\\"Current Portfolio\\\"" in infos or "no \"Current Portfolio\"" in infos
    # Page-local sidebar default is 4 ETFs (n_default=4).
    assert "4 ETF" in infos


def test_scope_notice_shows_both_counts_when_current_portfolio_exists(no_network):
    at = _run_risk_analytics("en", current_portfolio=CURRENT_PORTFOLIO_5)
    infos = "\n".join(i.value for i in at.info)
    assert "4 ETF" in infos   # page-local selection (n_default=4)
    assert "5 ETF" in infos   # current_portfolio has 5 tickers


def test_scope_notice_never_silently_overwrites_either_weight_set(no_network):
    at = _run_risk_analytics("en", current_portfolio=CURRENT_PORTFOLIO_5)
    # current_portfolio weights must be untouched by rendering this page.
    assert at.session_state["current_portfolio"]["weights"] == CURRENT_PORTFOLIO_5["weights"]


# ── Issue #43 item B: dynamic stress historical-window disclosure ────────

def test_stress_window_disclosure_uses_actual_selected_dates(no_network):
    at = _run_risk_analytics("en")
    from src.utils import get_date_range_defaults
    start, end = get_date_range_defaults()
    captions = "\n".join(c.value for c in at.caption)
    assert str(start) in captions and str(end) in captions


def test_stress_historical_scenarios_flagged_outside_default_window(no_network):
    # Default window is 5 years back from today -- none of 2008/COVID/dot-com
    # bust fall inside it, so every historical scenario's caption must
    # disclose that its real event period isn't in the selected dataset.
    at = _run_risk_analytics("en")
    captions = "\n".join(c.value for c in at.caption)
    assert "actual period falls outside your currently selected data window" in captions
    assert "not a reconstruction of that event's actual historical price path" in captions


def test_stress_hypothetical_scenarios_not_flagged_outside_window(no_network):
    at = _run_risk_analytics("en")
    captions = [c.value for c in at.caption]
    hypothetical_captions = [c for c in captions if "Hypothetical Scenario" in c]
    assert hypothetical_captions
    for c in hypothetical_captions:
        assert "falls outside your currently selected data window" not in c


# ── Issue #43 item C: simplified stress table + beta shown once ──────────

def test_stress_table_has_exactly_three_columns_no_beta_no_dollar(no_network):
    at = _run_risk_analytics("en")
    stress_tables = [
        df.value for df in at.dataframe
        if hasattr(df.value, "columns") and "Market Shock" in list(df.value.columns)
    ]
    assert stress_tables, "stress test table not found"
    stress_df = stress_tables[0]
    assert list(stress_df.columns) == ["Market Shock", "Estimated Portfolio Impact"]
    assert "Portfolio Beta" not in stress_df.columns
    assert "Impact on $10,000" not in stress_df.columns


def test_portfolio_beta_rendered_once_above_table(no_network):
    at = _run_risk_analytics("en")
    corpus = "\n".join(m.value for m in at.markdown)
    assert corpus.count("This portfolio's β =") == 1


# ── Issue #43 item D: Methodology expander has a distinct info cue ───────

def test_methodology_expander_has_info_cue_more_metrics_does_not(no_network):
    at = _run_risk_analytics("en")
    markdown_values = [m.value for m in at.markdown]
    # The info_badge() cue is its own st.markdown() call rendered
    # immediately before the Methodology & Assumptions expander.
    badge_values = [v for v in markdown_values if "badge-blue" in v and "ⓘ" in v]
    assert badge_values, markdown_values
    assert "Explanation" in badge_values[0]

    expander_labels = [e.label for e in at.expander]
    assert "Methodology & Assumptions" in expander_labels
    assert "More risk metrics" in expander_labels or "More Risk Metrics" in expander_labels
