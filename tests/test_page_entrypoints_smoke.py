"""
Page entrypoint smoke tests (Issue #18 Stages 7/E).

Every page must render a sensible empty/default state on direct entry --
no current_portfolio, no prior session_state, no query params -- instead of
raising. This is exactly the "direct page entry has a sensible empty state
instead of exceptions" requirement from Stage 7, and the release gate's
"smoke-check every page entrypoint for uncaught exceptions" step.

Each page is loaded via streamlit.testing.v1.AppTest with a completely
fresh session (no current_portfolio, no cached results), which is the
worst case for every "empty state vs. exception" code path in the app.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

PAGES = [
    "app.py",
    "pages/1_ETF_Analysis.py",
    "pages/2_Portfolio_Optimizer.py",
    "pages/3_Investment_Simulator.py",
    "pages/4_Risk_Analytics.py",
    "pages/5_Machine_Learning.py",
    "pages/6_AI_Advisor.py",
    "pages/7_Portfolio_History.py",
    "pages/8_Market_Intelligence.py",
    "pages/9_Login.py",
]


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    # Same AppTest path-resolution / multi-page-nav workaround used by
    # tests/test_portfolio_optimizer.py and tests/test_portfolio_history.py.
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


@pytest.mark.parametrize("page_path", PAGES)
@pytest.mark.parametrize("lang", ["en", "zh-TW"])
def test_page_direct_entry_no_exception(page_path, lang):
    at = _apptest_from_file(page_path, default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    exc = at.exception[0] if at.exception else None
    assert exc is None, f"{page_path} ({lang}) raised on fresh/empty-state entry: {exc}"
