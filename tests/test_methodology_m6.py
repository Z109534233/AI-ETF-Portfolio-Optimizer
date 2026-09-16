"""
M6 -- Machine Learning Methodology (Issue #41 item K): deterministic tests.

Covers:
  - src/methodology.py's MACHINE_LEARNING_METHODOLOGY metadata and
    validate_ml_split() in isolation
  - i18n parity (zh-TW / en) for every new ml_methodology_* key
  - the rendered Methodology & Assumptions panel on pages/5_Machine_Learning.py:
    it must show a chronological/non-overlapping validation line without
    implying the MODEL performs well (Issue #41 item K's explicit
    requirement -- the success box validates the EVALUATION SETUP only)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.methodology import MACHINE_LEARNING_METHODOLOGY, validate_ml_split
from src.i18n import TRANSLATIONS


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _synthetic_prices_for(tickers, seed=11, n_days=800, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    data = {}
    for tk in tickers:
        daily_ret = drift + rng.normal(0, vol, n_days)
        data[tk] = 100 * np.cumprod(1 + daily_ret)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def _mock_ml_download(monkeypatch):
    import src.data_loader as data_loader_mod

    def _fake_download(tickers, start_date, end_date, price_field="Close"):
        return _synthetic_prices_for(tickers)

    monkeypatch.setattr(data_loader_mod, "download_etf_data", _fake_download)


def _methodology_expander_corpus(at, label):
    exp = next(e for e in at.expander if e.label == label)
    parts = [m.value for m in exp.markdown] + [c.value for c in exp.caption]
    parts += [s.value for s in exp.success] + [w.value for w in exp.warning]
    return "\n".join(parts)


# ── MACHINE_LEARNING_METHODOLOGY metadata matches actual implementation ──
def test_methodology_metadata_matches_actual_implementation():
    m = MACHINE_LEARNING_METHODOLOGY
    assert set(m["models"]["options"]) == {"Logistic Regression", "Random Forest"}
    assert "chronological" in m["split"]["method"]
    assert "none" in m["split"]["shuffling"].lower()
    assert "TRAINING set" in m["baseline"]["definition"]
    assert "no fundamental" in m["features"]["absent_inputs"].lower()
    assert "none" in m["absent_from_pipeline"]["cross_validation"].lower()
    assert "not modeled" in m["absent_from_pipeline"]["transaction_costs"]
    assert "not evidence of live-trading validity" in m["live_trading_validity"]


# ── validate_ml_split() ──────────────────────────────────────────────────
def test_validate_ml_split_passes_for_chronological_non_overlapping_windows():
    result = validate_ml_split("2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31")
    assert result["is_valid"] is True
    assert result["issues"] == []


def test_validate_ml_split_fails_for_overlapping_windows():
    result = validate_ml_split("2020-01-01", "2020-08-01", "2020-07-01", "2020-12-31")
    assert result["is_valid"] is False
    assert any("overlap" in issue for issue in result["issues"])


def test_validate_ml_split_fails_for_unparseable_dates():
    result = validate_ml_split(None, "2020-06-30", "2020-07-01", "2020-12-31")
    assert result["is_valid"] is False
    assert result["issues"]


# ── i18n parity for every new ml_methodology_* key ───────────────────────
def test_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "ml_methodology_title", "ml_methodology_subtitle",
        "ml_methodology_model_label", "ml_methodology_features_label",
        "ml_methodology_features_value", "ml_methodology_split_label",
        "ml_methodology_baseline_label", "ml_methodology_baseline_value",
        "ml_methodology_metrics_label", "ml_methodology_metrics_value",
        "ml_methodology_absent_label", "ml_methodology_absent_value",
        "ml_methodology_live_trading_label", "ml_methodology_live_trading_value",
        "ml_methodology_validation_pass", "ml_methodology_validation_fail",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""


# ── Rendered panel: chronological/non-overlap validation, no model-quality claim ──
def test_methodology_panel_validation_is_about_setup_not_model_quality(_mock_ml_download):
    at = _apptest_from_file("pages/5_Machine_Learning.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    run_btn = next((b for b in at.button if "Train Model" in (b.label or "")), None)
    if run_btn:
        run_btn.click()
        at.run()
    assert at.exception == []

    expander_labels = [e.label for e in at.expander]
    assert "Methodology & Assumptions" in expander_labels
    corpus = _methodology_expander_corpus(at, "Methodology & Assumptions")

    assert "chronological and non-overlapping" in corpus
    # Must never claim the MODEL performs well from this validation line.
    assert "the model performs well" not in corpus.lower()
    assert "good performance" not in corpus.lower()
    # Must disclose the absence of tuning/CV/transaction costs and the
    # single-window live-trading caveat.
    assert "cross-validation" in corpus.lower()
    assert "transaction costs" in corpus.lower()
    assert "not evidence of live-trading validity" in corpus


def test_methodology_panel_zh_tw_renders(_mock_ml_download):
    at = _apptest_from_file("pages/5_Machine_Learning.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    run_btn = next((b for b in at.button if b.label and "訓練模型" in b.label), None)
    if run_btn:
        run_btn.click()
        at.run()
    assert at.exception == []
    expander_labels = [e.label for e in at.expander]
    assert "方法論與假設" in expander_labels
    corpus = _methodology_expander_corpus(at, "方法論與假設")
    assert "依時間先後排列且互不重疊" in corpus
