"""
Risk Analytics benchmark tests.

Issue #45 item 5 (still relevant to ETF Analysis, which still uses
region_benchmark_selector()): a benchmark drawn from the selectable ETF
universe must never be one of the currently-selected ETFs.

Issue #48 item 3 supersedes this for Risk Analytics specifically: the
benchmark there is now a FIXED external market index (^GSPC for United
States), never drawn from the selectable ETF universe at all -- so
selecting SPY/VOO as a portfolio holding can no longer "reset" the
benchmark to an economically arbitrary substitute ETF (e.g. SCHD, simply
because it happened to sort first in candidate_options). The displayed
alpha metric must still be explicitly labeled annualized.

No real network call: src.data_loader.download_etf_data is monkeypatched,
same pattern as tests/test_risk_analytics_page.py.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DEFAULT_ETFS
from src.ui import index_benchmark_for_region, region_benchmark_selector


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


def _fake_download_missing_index(tickers, start_date, end_date, price_field="Close"):
    """Same as _fake_download, but ^GSPC (or any ^-prefixed index ticker)
    never has usable price data -- simulates the index download failing
    while the selected ETFs still succeed."""
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for tk in tickers:
        if tk.startswith("^"):
            continue
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def no_network(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


@pytest.fixture()
def no_network_missing_index(monkeypatch):
    import src.data_loader as data_loader_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download_missing_index)


def _run_with_etf_selected(ticker, lang="en"):
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    ms = next(m for m in at.multiselect if m.key == "selected_etfs_United States")
    current = list(ms.value)
    if ticker not in current:
        current = current[:3] + [ticker]
    ms.set_value(current)
    at.run()
    assert at.exception == []
    return at


# ── Unit-level: region_benchmark_selector() exclusion contract (still used
# by ETF Analysis) ────────────────────────────────────────────────────────

def test_selector_excludes_current_selection_when_alternative_exists():
    def _render():
        import streamlit as st
        from src.ui import region_benchmark_selector
        benchmark, was_reset = region_benchmark_selector(
            "United States", ["SPY", "VOO", "QQQ"], "Benchmark", exclude=["SPY"],
        )
        st.session_state["_test_benchmark"] = benchmark
        st.session_state["_test_was_reset"] = was_reset

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_render)
    at.run()
    assert at.exception == []
    assert at.session_state["_test_benchmark"] != "SPY"


def test_selector_falls_back_to_full_options_when_exclude_removes_everything():
    def _render():
        import streamlit as st
        from src.ui import region_benchmark_selector
        benchmark, was_reset = region_benchmark_selector(
            "United States", ["SPY"], "Benchmark", exclude=["SPY"],
        )
        st.session_state["_test_benchmark"] = benchmark

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_render)
    at.run()
    assert at.exception == []
    # No external benchmark exists at all -- exclusion isn't applied rather
    # than leaving no options.
    assert at.session_state["_test_benchmark"] == "SPY"


# ── Unit-level: index_benchmark_for_region() is independent of the ETF
# universe entirely ────────────────────────────────────────────────────────

def test_index_benchmark_for_region_is_never_a_selectable_etf():
    us_benchmark = index_benchmark_for_region("United States")
    assert us_benchmark == "^GSPC"
    assert us_benchmark not in DEFAULT_ETFS


# ── Page-level: selecting SPY/VOO never changes the Risk Analytics
# benchmark away from the fixed external index ───────────────────────────

@pytest.mark.parametrize("ticker", ["SPY", "VOO"])
def test_benchmark_remains_gspc_when_broad_market_etf_selected(no_network, ticker):
    at = _run_with_etf_selected(ticker, "en")
    caption_corpus = "\n".join(c.value for c in at.caption)
    assert "^GSPC" in caption_corpus

    markdown_corpus = "\n".join(m.value for m in at.markdown)
    # The benchmark-relative section must name ^GSPC, never a selectable
    # ETF such as SCHD (SCHD may legitimately appear elsewhere on the page
    # as just another selected/selectable holding -- that's not the bug
    # this test targets, so only the benchmark-identifying text is scoped).
    assert "vs. S&P 500 Index (^GSPC)" in markdown_corpus or "Annualized Alpha (vs S&P 500 Index (^GSPC))" in markdown_corpus
    assert "vs. SCHD" not in markdown_corpus and "(vs SCHD)" not in markdown_corpus


def test_alpha_label_explicitly_says_annualized_when_shown(no_network):
    at = _run_with_etf_selected("SPY", "en")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Annualized Alpha" in corpus
    assert "Annualized Alpha (vs" in corpus
    assert "^GSPC" in corpus


def test_index_unavailable_never_silently_substitutes_an_etf(no_network_missing_index):
    """If ^GSPC can't be downloaded, benchmark metrics must be shown as
    unavailable with a clear reason -- never silently substituted with a
    selectable ETF such as SCHD."""
    at = _run_with_etf_selected("SPY", "en")
    corpus_md = "\n".join(m.value for m in at.markdown)
    corpus_info = "\n".join(i.value for i in at.info)
    assert "Annualized Alpha" not in corpus_md
    assert "could not be downloaded" in corpus_info or "^GSPC" in corpus_info
    assert "SCHD" not in corpus_md and "SCHD" not in corpus_info
