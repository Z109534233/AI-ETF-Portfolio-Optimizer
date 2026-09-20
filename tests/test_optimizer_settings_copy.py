"""
Regression tests for Issue #48 item 4 (Portfolio Optimizer UI copy):

- The empty-state message (shown before the user has run an optimization)
  used the generic `msg_configure_and_run` string, which says "Configure
  settings in the sidebar" -- stale, since Portfolio Optimizer's settings
  live in the main-page "Edit Settings" expander, not the sidebar. A
  dedicated `opt_msg_configure_and_run` string is used instead; the
  generic string must remain unchanged for pages that still use sidebar
  controls (Investment Simulator, Machine Learning).
- The optimizer-method methodology strings (`opt_methodology_optimizer_desc_*`)
  used to say the weight bounds live in the sidebar; they must now say
  "settings" / "設定" instead, since those bounds also moved to the
  main-page "Edit Settings" expander.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.i18n import TRANSLATIONS


def test_optimizer_empty_state_uses_dedicated_edit_settings_copy():
    en = TRANSLATIONS["en"]["opt_msg_configure_and_run"]
    zh = TRANSLATIONS["zh-TW"]["opt_msg_configure_and_run"]
    assert "sidebar" not in en.lower()
    assert "Edit Settings" in en
    assert "側邊欄" not in zh
    assert "編輯設定" in zh


def test_generic_configure_and_run_copy_is_unchanged_for_sidebar_pages():
    """Investment Simulator and Machine Learning still use sidebar
    controls -- the generic string must still say so."""
    en = TRANSLATIONS["en"]["msg_configure_and_run"]
    zh = TRANSLATIONS["zh-TW"]["msg_configure_and_run"]
    assert "sidebar" in en.lower()
    assert "側邊欄" in zh


def test_optimizer_methodology_no_longer_mentions_sidebar_for_weight_bounds():
    for lang, forbidden in (("en", "sidebar"), ("zh-TW", "側邊欄")):
        equal_weight = TRANSLATIONS[lang]["opt_methodology_optimizer_desc_equal_weight"]
        risk_parity = TRANSLATIONS[lang]["opt_methodology_optimizer_desc_risk_parity"]
        assert forbidden not in equal_weight, equal_weight
        assert forbidden not in risk_parity, risk_parity


def test_optimizer_methodology_uses_neutral_settings_wording():
    assert "settings" in TRANSLATIONS["en"]["opt_methodology_optimizer_desc_equal_weight"].lower()
    assert "settings" in TRANSLATIONS["en"]["opt_methodology_optimizer_desc_risk_parity"].lower()
    assert "設定" in TRANSLATIONS["zh-TW"]["opt_methodology_optimizer_desc_equal_weight"]
    assert "設定" in TRANSLATIONS["zh-TW"]["opt_methodology_optimizer_desc_risk_parity"]


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def test_portfolio_optimizer_page_shows_edit_settings_empty_state(monkeypatch):
    import src.data_loader as data_loader_mod
    import src.risk_free_rate as rf_mod
    import numpy as np
    import pandas as pd

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        dates = pd.bdate_range(start_date, end_date)
        data = {}
        for tk in tickers:
            rng = np.random.default_rng(abs(hash(tk)) % (2**32))
            data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(dates)))
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)
    monkeypatch.setattr(rf_mod, "get_cached_risk_free_rate", lambda: {
        "rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
        "source": "FRED", "status": "live", "reason": None,
    })

    at = _apptest_from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    infos = "\n".join(i.value for i in at.info)
    assert "Edit Settings" in infos
    assert "sidebar" not in infos.lower()
