"""
ETF price-coverage snapshot parser tests (Issue #45 item 8) -- and
build_snapshot() from scripts/audit_etf_price_coverage.py, exercised with a
tiny fake universe and mocked provider responses. No network access
anywhere in this file.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from src.etf_coverage import load_coverage_snapshot, coverage_display_stat, coverage_unavailable_caption

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from scripts.audit_etf_price_coverage import build_snapshot, attempt_fetch  # noqa: E402


def _write_snapshot(tmp_path, data):
    path = tmp_path / "price_coverage_summary.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_coverage_snapshot_missing_file_returns_none(tmp_path):
    assert load_coverage_snapshot(str(tmp_path / "nope.json")) is None


def test_load_coverage_snapshot_invalid_json_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_coverage_snapshot(str(path)) is None


def test_load_coverage_snapshot_missing_required_field_returns_none(tmp_path):
    data = {"provider": "x", "audited_at": "y"}  # missing most required fields
    path = _write_snapshot(tmp_path, data)
    assert load_coverage_snapshot(path) is None


def test_load_coverage_snapshot_invalid_status_returns_none(tmp_path):
    data = {
        "provider": "x", "audited_at": "y", "universe_size": 3,
        "attempted_count": 3, "available_count": 2, "unavailable_count": 1,
        "status": "BOGUS",
    }
    path = _write_snapshot(tmp_path, data)
    assert load_coverage_snapshot(path) is None


def test_load_coverage_snapshot_valid_complete(tmp_path):
    data = {
        "provider": "Yahoo Finance (yfinance)", "audited_at": "2026-01-01T00:00:00Z",
        "universe_size": 3, "attempted_count": 3, "available_count": 2,
        "unavailable_count": 1, "status": "COMPLETE",
    }
    path = _write_snapshot(tmp_path, data)
    loaded = load_coverage_snapshot(path)
    assert loaded == data


def test_coverage_display_stat_none_when_no_snapshot():
    assert coverage_display_stat(None) is None


def test_coverage_display_stat_complete_shows_available_over_universe():
    snapshot = {
        "provider": "x", "audited_at": "y", "universe_size": 100,
        "attempted_count": 100, "available_count": 87, "unavailable_count": 13,
        "status": "COMPLETE",
    }
    value, label = coverage_display_stat(snapshot, "en")
    assert value == "87 / 100"
    assert "Verified" in label


def test_coverage_display_stat_partial_shows_available_over_attempted():
    snapshot = {
        "provider": "x", "audited_at": "y", "universe_size": 6074,
        "attempted_count": 200, "available_count": 180, "unavailable_count": 20,
        "status": "PARTIAL",
    }
    value, label = coverage_display_stat(snapshot, "en")
    assert value == "180 / 200"  # NOT the full universe size for a partial audit
    assert "Partial" in label

    value_zh, label_zh = coverage_display_stat(snapshot, "zh-TW")
    assert value_zh == "180 / 200"
    assert "部分驗證" in label_zh


def test_coverage_unavailable_caption_is_bilingual():
    assert coverage_unavailable_caption("en")
    assert coverage_unavailable_caption("zh-TW")
    assert coverage_unavailable_caption("en") != coverage_unavailable_caption("zh-TW")


def test_no_invented_exact_coverage_number_without_snapshot():
    """Never fabricate a coverage count when no snapshot file exists."""
    assert load_coverage_snapshot("/nonexistent/path/price_coverage_summary.json") is None
    assert coverage_display_stat(None, "zh-TW") is None


# ── scripts/audit_etf_price_coverage.py build_snapshot() ────────────────
def test_build_snapshot_aggregates_a_tiny_fake_universe_correctly():
    tickers_with_market = [
        ("AAA", "United States"), ("BBB", "United States"), ("CCC", "Taiwan"),
    ]
    results = {"AAA": "available", "BBB": "no_data", "CCC": "transient_error"}
    snapshot = build_snapshot(tickers_with_market, results, universe_size=3, complete=True,
                               audited_at="2026-01-01T00:00:00Z")
    assert snapshot["status"] == "COMPLETE"
    assert snapshot["universe_size"] == 3
    assert snapshot["attempted_count"] == 3
    assert snapshot["available_count"] == 1
    assert snapshot["unavailable_count"] == 2
    assert snapshot["per_market"]["United States"] == {"attempted": 2, "available": 1, "unavailable": 1}
    assert snapshot["per_market"]["Taiwan"] == {"attempted": 1, "available": 0, "unavailable": 1}


def test_build_snapshot_partial_marks_not_attempted_tickers_correctly():
    tickers_with_market = [("AAA", "United States"), ("BBB", "United States"), ("CCC", "United States")]
    results = {"AAA": "available"}  # BBB/CCC not attempted yet
    snapshot = build_snapshot(tickers_with_market, results, universe_size=3, complete=False)
    assert snapshot["status"] == "PARTIAL"
    assert snapshot["attempted_count"] == 1
    assert snapshot["available_count"] == 1
    assert snapshot["unavailable_count"] == 0
    assert snapshot["per_market"]["United States"]["attempted"] == 1


def test_build_snapshot_never_treats_transient_error_as_unsupported_proof():
    """transient_error must be counted as 'unavailable' in the aggregate
    stat but its per-ticker status must remain distinguishable in
    ticker_status -- a caller can tell it apart from a genuine no_data."""
    tickers_with_market = [("AAA", "United States")]
    results = {"AAA": "transient_error"}
    snapshot = build_snapshot(tickers_with_market, results, universe_size=1, complete=True)
    assert snapshot["ticker_status"]["AAA"] == "transient_error"
    assert snapshot["unavailable_count"] == 1


def test_attempt_fetch_returns_no_data_for_empty_frame(monkeypatch):
    import scripts.audit_etf_price_coverage as audit_mod
    import pandas as pd

    class _FakeYFinance:
        @staticmethod
        def download(**kwargs):
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYFinance)
    assert attempt_fetch("FAKE") == "no_data"


def test_attempt_fetch_returns_transient_error_on_exception(monkeypatch):
    import scripts.audit_etf_price_coverage as audit_mod

    class _FakeYFinance:
        @staticmethod
        def download(**kwargs):
            raise RuntimeError("simulated rate limit")

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYFinance)
    assert attempt_fetch("FAKE") == "transient_error"


def test_attempt_fetch_returns_available_for_usable_data(monkeypatch):
    import pandas as pd

    class _FakeYFinance:
        @staticmethod
        def download(**kwargs):
            return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYFinance)
    assert attempt_fetch("FAKE") == "available"
