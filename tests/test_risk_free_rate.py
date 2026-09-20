"""
Live risk-free rate service tests (Issue #45 item 3; hardened 3-tier
fallback chain added in Issue #48 item 1: FRED DGS3MO -> Yahoo Finance
^IRX secondary proxy -> fixed emergency default).

No real network call is made anywhere in this file -- every test mocks
src.risk_free_rate._http_get_fred_csv() and/or
src.risk_free_rate._http_get_irx_history(), the two isolated network
boundaries. Every test that expects FRED to fail also mocks the ^IRX
boundary (to either succeed or fail deterministically) so the chain never
falls through to a real network call.
"""

import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest

from src import risk_free_rate as rf_mod


FAKE_CSV_WITH_MISSING_ROWS = (
    "DATE,DGS3MO\n"
    "2026-09-10,4.15\n"
    "2026-09-11,.\n"
    "2026-09-12,.\n"
    "2026-09-14,4.11\n"
)


def _fake_irx_history(dates, closes):
    return pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))


def _boom_irx(period=rf_mod.IRX_FETCH_PERIOD):
    raise ConnectionError("simulated ^IRX network failure")


# ── FRED success (no ^IRX call needed) ───────────────────────────────────

def test_get_risk_free_rate_success_parses_latest_non_missing_observation(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: FAKE_CSV_WITH_MISSING_ROWS)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "live"
    assert result["source"] == "FRED"
    assert result["series_id"] == "DGS3MO"
    assert result["observed_date"] == "2026-09-14"
    assert result["rate"] == pytest.approx(0.0411)
    assert result["reason"] is None


def test_get_risk_free_rate_skips_missing_dot_rows_at_the_end(monkeypatch):
    csv_text = "DATE,DGS3MO\n2026-09-10,4.20\n2026-09-11,.\n"
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: csv_text)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "live"
    assert result["observed_date"] == "2026-09-10"
    assert result["rate"] == pytest.approx(0.0420)


# ── FRED failure -> ^IRX success ──────────────────────────────────────────

def test_fred_timeout_falls_through_to_irx_success(monkeypatch, caplog):
    import requests

    def _timeout(timeout=15.0):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _timeout)
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(["2026-09-14"], [4.20]),
    )
    with caplog.at_level(logging.WARNING):
        result = rf_mod.get_risk_free_rate()
    assert result["status"] == "secondary_live"
    assert result["source"] == "Yahoo Finance"
    assert result["series_id"] == "^IRX"
    assert result["observed_date"] == "2026-09-14"
    assert result["rate"] == pytest.approx(0.0420)
    # NEVER labeled as FRED DGS3MO.
    assert result["source"] != "FRED"
    assert result["series_id"] != "DGS3MO"
    assert any("FRED fetch failed" in r.message for r in caplog.records)


def test_fred_http_and_parse_failure_falls_through_to_irx_success(monkeypatch):
    # HTTP/parse failure: FRED responds but with no usable observation.
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: "not,a,valid,csv")
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(["2026-09-13"], [4.05]),
    )
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "secondary_live"
    assert result["rate"] == pytest.approx(0.0405)
    assert result["observed_date"] == "2026-09-13"


def test_fred_connection_error_falls_through_to_irx_success(monkeypatch):
    def _boom(timeout=15.0):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _boom)
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(["2026-09-12"], [4.33]),
    )
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "secondary_live"
    assert result["rate"] == pytest.approx(0.0433)


# ── FRED + ^IRX both fail -> emergency 5% fallback ───────────────────────

def test_both_live_sources_fail_falls_back_to_emergency_default(monkeypatch, caplog):
    def _fred_boom(timeout=15.0):
        raise ConnectionError("simulated FRED network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _fred_boom)
    monkeypatch.setattr(rf_mod, "_http_get_irx_history", _boom_irx)
    with caplog.at_level(logging.WARNING):
        result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"
    assert result["rate"] == rf_mod.DEFAULT_FALLBACK_RATE
    assert result["observed_date"] is None
    assert "ConnectionError" in result["reason"]
    assert "IRX" in result["reason"] or "^IRX" in result["reason"]
    assert any("emergency fallback" in r.message for r in caplog.records)


def test_falls_back_when_every_fred_row_is_missing_and_irx_also_fails(monkeypatch):
    csv_text = "DATE,DGS3MO\n2026-09-10,.\n2026-09-11,.\n"
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: csv_text)
    monkeypatch.setattr(rf_mod, "_http_get_irx_history", _boom_irx)
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"
    assert result["rate"] == rf_mod.DEFAULT_FALLBACK_RATE
    assert result["reason"]


def test_falls_back_on_empty_fred_response_and_irx_also_empty(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: "")
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: pd.DataFrame(),
    )
    result = rf_mod.get_risk_free_rate()
    assert result["status"] == "fallback"


# ── ^IRX percent-to-decimal conversion + observation-date provenance ─────

def test_irx_percent_to_decimal_conversion_is_correct(monkeypatch):
    def _fred_boom(timeout=15.0):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _fred_boom)
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(
            ["2026-09-10", "2026-09-11", "2026-09-14"], [4.10, 4.18, 4.255],
        ),
    )
    result = rf_mod.get_risk_free_rate()
    # Latest valid Close (4.255%) is converted to its decimal equivalent,
    # not left as a raw percent or divided incorrectly.
    assert result["rate"] == pytest.approx(0.04255)
    assert result["observed_date"] == "2026-09-14"


def test_irx_uses_latest_valid_close_skipping_trailing_nan(monkeypatch):
    import numpy as np

    def _fred_boom(timeout=15.0):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _fred_boom)
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(
            ["2026-09-10", "2026-09-11", "2026-09-14"], [4.10, 4.18, np.nan],
        ),
    )
    result = rf_mod.get_risk_free_rate()
    assert result["observed_date"] == "2026-09-11"
    assert result["rate"] == pytest.approx(0.0418)


# ── Retry behavior on the FRED network boundary ──────────────────────────

def test_http_get_fred_csv_retries_transient_failure_then_succeeds(monkeypatch):
    import requests

    calls = {"n": 0}

    class _FakeResponse:
        status_code = 200
        text = FAKE_CSV_WITH_MISSING_ROWS

        def raise_for_status(self):
            return None

    def _flaky_get(url, timeout=None, headers=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("transient DNS failure")
        return _FakeResponse()

    monkeypatch.setattr(rf_mod.requests, "get", _flaky_get)
    monkeypatch.setattr(rf_mod.time, "sleep", lambda s: None)
    text = rf_mod._http_get_fred_csv()
    assert text == FAKE_CSV_WITH_MISSING_ROWS
    assert calls["n"] == 3


def test_http_get_fred_csv_gives_up_after_max_attempts(monkeypatch):
    import requests

    calls = {"n": 0}

    def _always_timeout(url, timeout=None, headers=None):
        calls["n"] += 1
        raise requests.exceptions.Timeout("simulated persistent timeout")

    monkeypatch.setattr(rf_mod.requests, "get", _always_timeout)
    monkeypatch.setattr(rf_mod.time, "sleep", lambda s: None)
    with pytest.raises(requests.exceptions.Timeout):
        rf_mod._http_get_fred_csv()
    assert calls["n"] == rf_mod.MAX_FETCH_ATTEMPTS


def test_http_get_fred_csv_uses_browser_like_headers(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        text = FAKE_CSV_WITH_MISSING_ROWS

        def raise_for_status(self):
            return None

    def _capture_get(url, timeout=None, headers=None):
        captured["timeout"] = timeout
        captured["headers"] = headers
        return _FakeResponse()

    monkeypatch.setattr(rf_mod.requests, "get", _capture_get)
    rf_mod._http_get_fred_csv()
    assert captured["timeout"] == rf_mod.REQUEST_TIMEOUT_SECONDS
    assert rf_mod.REQUEST_TIMEOUT_SECONDS == 15.0
    assert "Mozilla" in captured["headers"]["User-Agent"]
    assert "text/csv" in captured["headers"]["Accept"]


# ── Logging on failed live sources ────────────────────────────────────────

def test_logging_warning_emitted_when_fred_fails_even_if_irx_succeeds(monkeypatch, caplog):
    def _fred_boom(timeout=15.0):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", _fred_boom)
    monkeypatch.setattr(
        rf_mod, "_http_get_irx_history",
        lambda period=rf_mod.IRX_FETCH_PERIOD: _fake_irx_history(["2026-09-14"], [4.20]),
    )
    with caplog.at_level(logging.WARNING):
        rf_mod.get_risk_free_rate()
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("FRED fetch failed" in w and "ConnectionError" in w for w in warnings)
    # No stack trace/traceback text leaked into the log message itself.
    assert not any("Traceback" in w for w in warnings)


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


# ── format_rate_provenance() ──────────────────────────────────────────────

def test_format_rate_provenance_live():
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    text = rf_mod.format_rate_provenance(rf, "en")
    assert "4.11%" in text
    assert "FRED" in text
    assert "DGS3MO" in text
    assert "2026-09-14" in text
    assert text.count("4.11%") == 1


def test_format_rate_provenance_secondary_live_labels_yahoo_not_fred():
    rf = {"rate": 0.0420, "observed_date": "2026-09-14", "series_id": "^IRX",
          "source": "Yahoo Finance", "status": "secondary_live", "reason": None}
    text_en = rf_mod.format_rate_provenance(rf, "en")
    assert "4.20%" in text_en
    assert "Yahoo Finance" in text_en and "^IRX" in text_en
    assert "2026-09-14" in text_en
    assert "secondary" in text_en.lower()
    assert "FRED" not in text_en
    assert "DGS3MO" not in text_en

    text_zh = rf_mod.format_rate_provenance(rf, "zh-TW")
    assert "4.20%" in text_zh
    assert "Yahoo Finance" in text_zh and "^IRX" in text_zh
    assert "FRED" not in text_zh


def test_format_rate_provenance_fallback_discloses_status():
    rf = {"rate": 0.05, "observed_date": None, "series_id": "DGS3MO",
          "source": "FRED", "status": "fallback", "reason": "network error"}
    text_en = rf_mod.format_rate_provenance(rf, "en")
    text_zh = rf_mod.format_rate_provenance(rf, "zh-TW")
    assert "5.00%" in text_en and "fallback" in text_en.lower()
    assert "5.00%" in text_zh and "無法取得" in text_zh


def test_is_manual_override_false_when_selected_rate_matches_fetched_default():
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    # Mirrors how every page's slider initializes: round(rf["rate"] * 100, 2) / 100.
    untouched_slider_value = round(rf["rate"] * 100, 2) / 100
    assert rf_mod.is_manual_override(rf, untouched_slider_value) is False


def test_is_manual_override_true_when_slider_moved_away_from_fetched_rate():
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    assert rf_mod.is_manual_override(rf, 0.06) is True


def test_format_rate_provenance_matching_selected_rate_shows_fred_only_no_override_language():
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    text = rf_mod.format_rate_provenance(rf, "en", selected_rate=0.0411)
    assert "FRED" in text
    assert "override" not in text.lower()


def test_format_rate_provenance_discloses_manual_override_vs_fred():
    """Regression for Issue #46 review item 2: a slider moved away from
    the fetched DGS3MO rate must be captioned as a manual override, not as
    if it were still the live FRED value -- with the FRED reference
    rate/date still shown for context."""
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    text_en = rf_mod.format_rate_provenance(rf, "en", selected_rate=0.07)
    assert "7.00%" in text_en
    assert "overridden" in text_en.lower()
    assert "4.11%" in text_en and "FRED" in text_en and "DGS3MO" in text_en and "2026-09-14" in text_en

    text_zh = rf_mod.format_rate_provenance(rf, "zh-TW", selected_rate=0.07)
    assert "7.00%" in text_zh
    assert "手動" in text_zh
    assert "4.11%" in text_zh


def test_format_rate_provenance_discloses_manual_override_vs_irx_secondary():
    """Same manual-override contract, but with the secondary ^IRX proxy as
    the live reference instead of FRED -- the reference must still be
    labeled Yahoo Finance / ^IRX, never FRED (Issue #48 item 1)."""
    rf = {"rate": 0.0420, "observed_date": "2026-09-14", "series_id": "^IRX",
          "source": "Yahoo Finance", "status": "secondary_live", "reason": None}
    text_en = rf_mod.format_rate_provenance(rf, "en", selected_rate=0.06)
    assert "6.00%" in text_en
    assert "overridden" in text_en.lower()
    assert "4.20%" in text_en and "Yahoo Finance" in text_en and "^IRX" in text_en
    assert "FRED" not in text_en


def test_format_rate_provenance_without_selected_rate_is_unchanged():
    """Backward-compatible: callers that don't pass selected_rate (e.g.
    Home, which has no slider) keep the original FRED-only caption."""
    rf = {"rate": 0.0411, "observed_date": "2026-09-14", "series_id": "DGS3MO",
          "source": "FRED", "status": "live", "reason": None}
    assert rf_mod.format_rate_provenance(rf, "en") == rf_mod.format_rate_provenance(rf, "en", selected_rate=None)


def test_cross_page_cached_helper_returns_one_consistent_rate_object(monkeypatch):
    monkeypatch.setattr(rf_mod, "_http_get_fred_csv", lambda timeout=15.0: FAKE_CSV_WITH_MISSING_ROWS)
    rf_mod.get_cached_risk_free_rate.clear()
    a = rf_mod.get_cached_risk_free_rate()
    b = rf_mod.get_cached_risk_free_rate()
    assert a == b
    assert a["source"] == "FRED" and a["series_id"] == "DGS3MO"
