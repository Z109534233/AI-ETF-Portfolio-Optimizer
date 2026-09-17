"""
Risk Analytics benchmark self-inclusion tests (Issue #45 item 5):
selecting SPY as a portfolio ETF must never leave SPY as the benchmark, and
the displayed alpha metric must be explicitly labeled annualized.

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

from src.ui import region_benchmark_selector


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


def _run_with_spy_selected(lang="en"):
    at = _apptest_from_file("pages/4_Risk_Analytics.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    ms = next(m for m in at.multiselect if m.key == "selected_etfs_United States")
    current = list(ms.value)
    if "SPY" not in current:
        current = current[:3] + ["SPY"]
    ms.set_value(current)
    at.run()
    assert at.exception == []
    return at


# ── Unit-level: region_benchmark_selector() exclusion contract ──────────

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


# ── Page-level: selecting SPY resets the benchmark and labels alpha ──────

def test_benchmark_selectbox_never_shows_spy_when_spy_is_a_selected_etf(no_network):
    at = _run_with_spy_selected("en")
    bench_selects = [sb for sb in at.selectbox if sb.key and sb.key.startswith("selected_benchmark_United States")]
    assert bench_selects, "expected the region-keyed benchmark selectbox to be present"
    assert bench_selects[0].value != "SPY"


def test_alpha_label_explicitly_says_annualized_when_shown(no_network):
    at = _run_with_spy_selected("en")
    corpus = "\n".join(m.value for m in at.markdown)
    if "Annualized Alpha" in corpus:
        assert "Annualized Alpha (vs" in corpus
    # If alpha isn't shown at all, it must be because no external benchmark
    # remained -- never silently computed self-referentially and mislabeled.
    else:
        infos = "\n".join(i.value for i in at.info)
        assert "no valid external benchmark" in infos.lower() or "已是目前選擇" in infos


def test_no_test_hardcodes_a_fabricated_alpha_value(no_network):
    """Regression guard: the page must never show an alpha figure computed
    against SPY when SPY is itself a selected holding."""
    at = _run_with_spy_selected("en")
    bench_selects = [sb for sb in at.selectbox if sb.key and sb.key.startswith("selected_benchmark_United States")]
    assert bench_selects[0].value != "SPY"
