"""
Tests for src/fx.py (Issue #22 -- multi-currency FX conversion for the
Portfolio Optimizer, allowing a mix of US/Taiwan/UK ETFs to be optimized in
a single chosen base currency).

No real network/yfinance call is ever made here: every test that would
otherwise hit the network monkeypatches src.fx.yf.download, mirroring the
existing pattern in tests/test_portfolio_optimizer.py
(`patch("src.data_loader.yf.download", ...)`). Identity-conversion and
no-op (already-base-currency) tests additionally assert the mock was NOT
called, to prove no wasted/needless network access happens for those cases.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest

from src import fx


def _fx_frame(values, dates):
    """Build a fake yfinance-style OHLC-ish DataFrame with just a Close
    column, matching what yf.download(..., auto_adjust=True) returns for a
    single ticker (flat columns, no MultiIndex -- see src/fx.py's
    _download_fx_leg, which mirrors src/data_loader.py's handling)."""
    return pd.DataFrame({"Close": values}, index=pd.to_datetime(dates))


def _make_fake_download(rate_by_ticker):
    """Return a fake yf.download(...) that looks up its response purely by
    the `tickers` kwarg, so different tests can register different FX pairs
    without needing to know call order. `rate_by_ticker` maps ticker ->
    either a DataFrame to return, or a callable(start, end) -> DataFrame,
    or None (simulating "no data"), or an Exception instance/class to raise.
    """
    calls = []

    def fake_download(tickers, start=None, end=None, **kwargs):
        calls.append(tickers)
        entry = rate_by_ticker.get(tickers)
        if entry is None:
            return pd.DataFrame()
        if isinstance(entry, Exception):
            raise entry
        if isinstance(entry, type) and issubclass(entry, Exception):
            raise entry("simulated failure")
        if callable(entry) and not isinstance(entry, pd.DataFrame):
            return entry(start, end)
        return entry

    fake_download.calls = calls
    return fake_download


DATES = pd.date_range("2024-01-01", periods=5, freq="B")


# ── Identity conversion (from_currency == to_currency) ──────────────────────

def test_get_fx_series_identity_returns_all_ones_without_network(monkeypatch):
    called = []
    monkeypatch.setattr(fx.yf, "download", lambda *a, **k: called.append(1) or pd.DataFrame())

    series = fx.get_fx_series("USD", "USD", "2024-01-01", "2024-01-10")

    assert series is not None
    assert (series == 1.0).all()
    assert not called  # identity case must never touch the network


def test_convert_prices_no_op_when_all_already_base_currency(monkeypatch):
    fake_download = _make_fake_download({})
    monkeypatch.setattr(fx.yf, "download", fake_download)

    prices = pd.DataFrame({"VOO": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=DATES)
    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD"}, "USD", "2024-01-01", "2024-01-10"
    )

    assert result["currency_adjusted"] is False
    assert list(result["converted_prices"].columns) == ["VOO"]
    assert result["unavailable_tickers"] == []
    assert not fake_download.calls  # no FX call attempted for a pure no-op


# ── Mocked successful direct-pair conversion ────────────────────────────────

def test_direct_pair_conversion_is_mathematically_correct(monkeypatch):
    # USD -> TWD via the direct "TWD=X" ticker.
    fx_rate_values = [31.0, 31.2, 31.1, 31.3, 31.4]
    fake_download = _make_fake_download({
        "TWD=X": _fx_frame(fx_rate_values, DATES),
    })
    monkeypatch.setattr(fx.yf, "download", fake_download)

    native_prices = [100.0, 101.0, 99.0, 102.0, 103.0]
    prices = pd.DataFrame({"VOO": native_prices}, index=DATES)

    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD"}, "TWD", "2024-01-01", "2024-01-10"
    )

    assert result["unavailable_tickers"] == []
    assert result["currency_adjusted"] is True
    assert result["base_currency"] == "TWD"
    converted = result["converted_prices"]["VOO"]
    for i, d in enumerate(DATES):
        assert converted[d] == pytest.approx(native_prices[i] * fx_rate_values[i])


def test_get_fx_series_direct_pair_matches_raw_rate(monkeypatch):
    fx_rate_values = [0.79, 0.80, 0.81, 0.80, 0.79]
    fake_download = _make_fake_download({
        "GBPUSD=X": _fx_frame(fx_rate_values, DATES),
    })
    monkeypatch.setattr(fx.yf, "download", fake_download)

    series = fx.get_fx_series("GBP", "USD", "2024-01-01", "2024-01-10")
    assert series is not None
    for i, d in enumerate(DATES):
        assert series[d] == pytest.approx(fx_rate_values[i])


# ── Mocked composed cross-rate (GBP -> TWD via GBP -> USD -> TWD) ───────────

def test_composed_cross_rate_gbp_to_twd_is_mathematically_correct(monkeypatch):
    gbpusd_values = [1.27, 1.28, 1.26, 1.29, 1.30]
    usdtwd_values = [31.0, 31.2, 31.1, 31.3, 31.4]
    fake_download = _make_fake_download({
        "GBPUSD=X": _fx_frame(gbpusd_values, DATES),
        "TWD=X": _fx_frame(usdtwd_values, DATES),
    })
    monkeypatch.setattr(fx.yf, "download", fake_download)

    series = fx.get_fx_series("GBP", "TWD", "2024-01-01", "2024-01-10")
    assert series is not None
    for i, d in enumerate(DATES):
        expected = gbpusd_values[i] * usdtwd_values[i]
        assert series[d] == pytest.approx(expected)

    # And end-to-end through convert_prices_to_base_currency too.
    native_prices = [50.0, 51.0, 49.0, 52.0, 53.0]
    prices = pd.DataFrame({"VUKE": native_prices}, index=DATES)
    result = fx.convert_prices_to_base_currency(
        prices, {"VUKE": "GBP"}, "TWD", "2024-01-01", "2024-01-10"
    )
    assert result["unavailable_tickers"] == []
    converted = result["converted_prices"]["VUKE"]
    for i, d in enumerate(DATES):
        expected = native_prices[i] * gbpusd_values[i] * usdtwd_values[i]
        assert converted[d] == pytest.approx(expected)


def test_required_fx_pairs_describes_direct_and_cross_pairs():
    pairs = fx.required_fx_pairs({"USD", "TWD", "GBP"}, "TWD")
    assert ("USD", "TWD") in pairs
    assert ("GBP", "TWD") in pairs
    assert ("TWD", "TWD") not in pairs  # base currency itself needs no pair


# ── FX unavailable: dropped from converted_prices, no exception, no fabrication ──

@pytest.mark.parametrize("bad_entry", [
    None,                     # yfinance returns empty/no data
    RuntimeError,             # yfinance raises
])
def test_unavailable_fx_ticker_is_dropped_not_fabricated(monkeypatch, bad_entry):
    fake_download = _make_fake_download({"TWD=X": bad_entry})
    monkeypatch.setattr(fx.yf, "download", fake_download)

    prices = pd.DataFrame({
        "VOO": [100.0, 101.0, 102.0, 103.0, 104.0],
    }, index=DATES)

    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD"}, "TWD", "2024-01-01", "2024-01-10"
    )

    assert "VOO" not in result["converted_prices"].columns
    assert result["unavailable_tickers"] == ["VOO"]
    assert result["currency_adjusted"] is False


def test_unavailable_fx_does_not_affect_other_tickers(monkeypatch):
    # VOO (USD) converts fine via TWD=X; a hypothetical GBP ticker's cross
    # fails because GBPUSD=X is unavailable -- VOO must still succeed.
    fx_rate_values = [31.0, 31.2, 31.1, 31.3, 31.4]
    fake_download = _make_fake_download({
        "TWD=X": _fx_frame(fx_rate_values, DATES),
        "GBPUSD=X": None,
    })
    monkeypatch.setattr(fx.yf, "download", fake_download)

    prices = pd.DataFrame({
        "VOO": [100.0, 101.0, 102.0, 103.0, 104.0],
        "VUKE": [50.0, 51.0, 49.0, 52.0, 53.0],
    }, index=DATES)

    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD", "VUKE": "GBP"}, "TWD", "2024-01-01", "2024-01-10"
    )

    assert "VOO" in result["converted_prices"].columns
    assert "VUKE" not in result["converted_prices"].columns
    assert result["unavailable_tickers"] == ["VUKE"]
    assert result["currency_adjusted"] is True  # VOO really was converted


# ── All-unavailable case still returns a well-formed dict ───────────────────

def test_all_unavailable_returns_well_formed_empty_result(monkeypatch):
    fake_download = _make_fake_download({"TWD=X": None, "GBPUSD=X": None})
    monkeypatch.setattr(fx.yf, "download", fake_download)

    prices = pd.DataFrame({
        "VOO": [100.0, 101.0, 102.0, 103.0, 104.0],
        "VUKE": [50.0, 51.0, 49.0, 52.0, 53.0],
    }, index=DATES)

    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD", "VUKE": "GBP"}, "TWD", "2024-01-01", "2024-01-10"
    )

    assert isinstance(result, dict)
    assert result["converted_prices"].empty
    assert set(result["unavailable_tickers"]) == {"VOO", "VUKE"}
    assert result["currency_adjusted"] is False
    assert result["base_currency"] == "TWD"
    assert isinstance(result["fx_source"], str) and result["fx_source"]
    assert isinstance(result["conversion_method"], str) and result["conversion_method"]


# ── Date alignment / forward-fill behavior ───────────────────────────────────

def test_fx_forward_fill_onto_denser_price_index(monkeypatch):
    # FX series only has 2 sparse dates; prices exist on 5 business days.
    # The FX rate must be forward-filled onto every price date >= the first
    # FX date, using the most recent known FX quote.
    sparse_fx_dates = [DATES[0], DATES[2]]
    fx_values = [30.0, 31.0]
    fake_download = _make_fake_download({
        "TWD=X": _fx_frame(fx_values, sparse_fx_dates),
    })
    monkeypatch.setattr(fx.yf, "download", fake_download)

    native_prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    prices = pd.DataFrame({"VOO": native_prices}, index=DATES)

    result = fx.convert_prices_to_base_currency(
        prices, {"VOO": "USD"}, "TWD", "2024-01-01", "2024-01-10"
    )

    converted = result["converted_prices"]["VOO"]
    # DATES[0] uses fx=30.0 (exact match), DATES[1] forward-fills 30.0
    # (last known rate before/at that date), DATES[2..4] forward-fill 31.0.
    assert converted[DATES[0]] == pytest.approx(native_prices[0] * 30.0)
    assert converted[DATES[1]] == pytest.approx(native_prices[1] * 30.0)
    assert converted[DATES[2]] == pytest.approx(native_prices[2] * 31.0)
    assert converted[DATES[3]] == pytest.approx(native_prices[3] * 31.0)
    assert converted[DATES[4]] == pytest.approx(native_prices[4] * 31.0)


def test_get_fx_series_identity_covers_requested_date_range():
    series = fx.get_fx_series("GBP", "GBP", "2024-01-01", "2024-01-10")
    assert series is not None
    assert len(series) > 0
    assert series.index.min() >= pd.Timestamp("2024-01-01")
    assert series.index.max() <= pd.Timestamp("2024-01-10")
