"""
Regression test for Issue #48 item 4: Home's "Platform Statistics" strip
used `len(NAV_ITEMS)` for its module-count stat, which includes Home
itself (a navigation entry, not an analysis/feature module) -- so it
showed 9 even though the page's own "Why Choose This Platform" section
says "八大核心模組" / "Eight core modules" (1 featured + 7 supporting
items = 8). This test guards the fix (len(NAV_ITEMS) - 1, labeled "功能
模組" / "Feature Modules") so it can never silently drift back to
len(NAV_ITEMS)/9, including if NAV_ITEMS ever gains or loses an entry.

No real network call: src.data_loader.download_etf_data (and
_with_status) and src.risk_free_rate.get_cached_risk_free_rate are
monkeypatched, same pattern as tests/test_home_optimizer_consistency.py.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.ui import NAV_ITEMS


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
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(dates)))
    return pd.DataFrame(data, index=dates)


def _fake_download_with_status(tickers, start_date, end_date, price_field="Close"):
    return _fake_download(tickers, start_date, end_date, price_field), False


FIXED_RF = {
    "rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
    "source": "FRED", "status": "live", "reason": None,
}


@pytest.fixture()
def mocked_inputs(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod
    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(data_loader_mod, "download_etf_data_with_status", _fake_download_with_status)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: FIXED_RF)


def _extract_stat_strip_values(at):
    pattern = re.compile(r'class="stat-strip-value">([^<]+)<')
    values = []
    for m in at.markdown:
        values.extend(pattern.findall(m.value))
    return values


def test_nav_items_includes_home_so_the_bug_scenario_is_real():
    """Sanity check on the fixture this test relies on: NAV_ITEMS includes
    Home, so len(NAV_ITEMS) alone really would over-count feature modules
    by one."""
    assert any(item["page"] == "app.py" for item in NAV_ITEMS)


@pytest.mark.parametrize("lang", ["en", "zh-TW"])
def test_home_feature_module_count_excludes_home_itself(mocked_inputs, lang):
    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    values = _extract_stat_strip_values(at)
    expected = str(len(NAV_ITEMS) - 1)
    assert expected in values, values
    # Never the raw, Home-inclusive count -- this is the exact bug this
    # test guards against.
    assert str(len(NAV_ITEMS)) not in values, values


@pytest.mark.parametrize("lang,label", [("en", "Feature Modules"), ("zh-TW", "功能模組")])
def test_home_feature_module_stat_label_excludes_home_wording(mocked_inputs, lang, label):
    at = _apptest_from_file("app.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []

    corpus = "\n".join(m.value for m in at.markdown)
    assert label in corpus
