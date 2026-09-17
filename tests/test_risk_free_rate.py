"""
Live risk-free rate service tests (Issue #45 item 3).

No real network call is made anywhere in this file -- every test mocks
src.risk_free_rate._http_get_fred_csv(), the one isolated network
boundary.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from src import risk_free_rate as rf_mod


FAKE_CSV_WITH_MISSING_ROWS = (
    "DATE,DGS3MO\n"
    "2026-09-10,4.15\n"
    "2026-09-11,.\n"
    "2026-09-12,.\n"
    "2026-09-14,4.11\n"
)


def test_get_risk_free_rate_success_parses_latest_non_missing_observation(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=5.0: FAKE_CSV_WITH_MISSING_ROWS)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "live"
    assert result["source"] == "FRED"
    assert result["series_id"] == "DGS3MO"
    assert result["observed_date"] == "2026-09-14"
    assert result["rate"] == pytest.approx(0.0411)
    assert result["reason"] is None


def test_get_risk_free_rate_skips_missing_dot_rows_at_the_end(monkeypatch):
    csv_text = "DATE,DGS3MO\n2026-09-10,4.20\n2026-09-11,.\n"
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=5.0: csv_text)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "live"
    assert result["observed_date"] == "2026-09-10"
    assert result["rate"] == pytest.approx(0.0420)


def test_get_risk_free_rate_falls_back_on_network_failure(monkeypatch):
    def _boom(timeout=5.0):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _boom)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"
    assert result["rate"] == rf_mod.DEFAULT_FALLBACK_RATE
    assert result["observed_date"] is None
    assert "ConnectionError" in result["reason"]


def test_get_risk_free_rate_falls_back_on_timeout(monkeypatch):
    import requests

    def _timeout(timeout=5.0):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _timeout)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"
    assert result["rate"] == rf_mod.DEFAULT_FALLBACK_RATE


def test_get_risk_free_rate_falls_back_when_every_row_is_missing(monkeypatch):
    csv_text = "DATE,DGS3MO\n2026-09-10,.\n2026-09-11,.\n"
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=5.0: csv_text)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"
    assert result["rate"] == rf_mod.DEFAULT_FALLBACK_RATE
    assert result["reason"]


def test_get_risk_free_rate_falls_back_on_empty_response(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=5.0: "")
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"


def test_no_test_hardcodes_todays_live_rate():
    """Regression guard for the issue's explicit instruction: application
    logic and tests must never hard-code the 4.11%/2026-09-14 sanity
    reference the issue mentions as application behavior -- it may only
    appear inside a test's own synthetic fixture data (as above), never as
    an assertion baked into src/risk_free_rate.py itself."""
    import inspect
    source = inspect.getsource(rf_mod)
    assert "4.11" not in source
    assert "2026-09-14" not in source


def test_format_rate_provenance_live():
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    text = rf_mod.format_rate_provenance(rf, "en")
    assert "4.11%" in text
    assert "FRED" in text
    assert "DGS3MO" in text
    assert "2026-09-14" in text


def test_format_rate_provenance_fallback_discloses_status():
    rf = {"rate": 0.05, "observed_date": None, "series_id": "DGS3MO",
          "source": "FRED", "status": "fallback", "reason": "network error"}
    text_en = rf_mod.format_rate_provenance(rf, "en")
    text_zh = rf_mod.format_rate_provenance(rf, "zh-TW")
    assert "5.00%" in text_en and "fallback" in text_en.lower()
    assert "5.00%" in text_zh and "無法取得" in text_zh


def test_cross_page_cached_helper_returns_one_consistent_rate_object(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=5.0: FAKE_CSV_WITH_MISSING_ROWS)
    rf_mod.get_cached_risk_free_rate.clear()
    a = rf_mod.get_cached_risk_free_rate()
    b = rf_mod.get_cached_risk_free_rate()
    assert a == b
    assert a["source"] == "FRED" and a["series_id"] == "DGS3MO"
