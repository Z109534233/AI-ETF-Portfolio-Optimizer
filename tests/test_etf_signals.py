"""
Tests for src/etf_signals.py (Issue #20 section 2 -- ETF Analysis signal
semantics + the OpenAI "AI Interpretation" call site).

Covers:
  - compute_quant_signals() is the ONE formula behind Trend Signal / Quant
    Score / Portfolio View / Signal Agreement -- deterministic and stable
    across repeated calls on the same input (the bug this replaces: three
    independently-tuned formulas could disagree on the same ticker).
  - Trend Signal and Portfolio View are derived by pure, independently
    testable helpers with fixed thresholds.
  - generate_etf_interpretation() mocked exactly like
    tests/test_openai_integration_points.py: no real network call, and it
    is never asked to compute/recompute a number -- only to explain
    numbers already present in the prompt.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd
import pytest

from src import openai_service as svc
from src.etf_signals import (
    compute_quant_signals, trend_signal_from_return, recent_trend_return,
    portfolio_view_from_score, generate_etf_interpretation, interpretation_fingerprint,
    TREND_LOOKBACK_DAYS,
)


class _FakeResponse:
    def __init__(self, text):
        self.output_text = text


class _FakeResponsesEndpoint:
    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text=None, exc=None):
        self.responses = _FakeResponsesEndpoint(text=text, exc=exc)


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(svc, "_secret", lambda name: None)


def _price_series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def _rising_prices(n=300, start=100.0, daily_return=0.001):
    vals = [start]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + daily_return))
    return _price_series(vals)


def _falling_prices(n=300, start=100.0, daily_return=-0.001):
    return _rising_prices(n=n, start=start, daily_return=daily_return)


def _noisy_rising_prices(n=300, start=100.0, base_return=0.003, wiggle=0.0006):
    # A deterministic (non-random) alternating wiggle on top of a steady
    # uptrend, so annualized_volatility()/sharpe_ratio() are well-defined
    # (non-zero) instead of the degenerate vol=0 / sharpe=0 you get from a
    # perfectly smooth compounding series -- needed for signal_agreement
    # tests where all four Quant-Score inputs must have an unambiguous sign.
    vals = [start]
    for i in range(n - 1):
        step = base_return + (wiggle if i % 2 == 0 else -wiggle)
        vals.append(vals[-1] * (1 + step))
    return _price_series(vals)


# ── trend_signal_from_return / portfolio_view_from_score (pure helpers) ────

def test_trend_signal_thresholds():
    assert trend_signal_from_return(0.06) == "Bullish"
    assert trend_signal_from_return(-0.06) == "Bearish"
    assert trend_signal_from_return(0.0) == "Neutral"
    assert trend_signal_from_return(0.05) == "Neutral"   # boundary is exclusive
    assert trend_signal_from_return(-0.05) == "Neutral"


def test_portfolio_view_thresholds():
    assert portfolio_view_from_score(65) == "Overweight"
    assert portfolio_view_from_score(35) == "Underweight"
    assert portfolio_view_from_score(50) == "Neutral"
    assert portfolio_view_from_score(64) == "Neutral"
    assert portfolio_view_from_score(36) == "Neutral"


# ── compute_quant_signals: one canonical formula, used consistently ────────

def test_compute_quant_signals_is_deterministic():
    p = _rising_prices()
    r1 = compute_quant_signals(p, 0.05)
    r2 = compute_quant_signals(p, 0.05)
    assert r1 == r2


def test_compute_quant_signals_trend_matches_portfolio_view_direction_for_strong_uptrend():
    p = _rising_prices(daily_return=0.003)
    signals = compute_quant_signals(p, 0.02)
    assert signals["trend"] == "Bullish"
    assert signals["score"] > 50
    assert signals["portfolio_view"] in ("Overweight", "Neutral")


def test_compute_quant_signals_downtrend_produces_bearish_and_low_score():
    p = _falling_prices(daily_return=-0.003)
    signals = compute_quant_signals(p, 0.02)
    assert signals["trend"] == "Bearish"
    assert signals["score"] < 50
    assert signals["portfolio_view"] in ("Underweight", "Neutral")


def test_compute_quant_signals_score_and_agreement_bounded():
    p = _rising_prices()
    signals = compute_quant_signals(p, 0.05)
    assert 0 <= signals["score"] <= 100
    assert 0 <= signals["signal_agreement"] <= 100


def test_compute_quant_signals_trend_uses_same_thresholds_as_top_level_helper():
    """The KPI-row Trend chip and every per-ticker Trend Signal must use the
    exact same rule -- this pins that compute_quant_signals() delegates to
    trend_signal_from_return() rather than an independently-tuned rule. Trend
    is driven by the RECENT-window return, not the full-window annualized
    return, so it must be compared against ret_recent, not ret_ann."""
    p = _rising_prices(daily_return=0.0005)
    signals = compute_quant_signals(p, 0.05)
    assert signals["trend"] == trend_signal_from_return(signals["ret_recent"])


# ── Trend Signal must reflect the RECENT window, not the full selected
# date range (Issue #20 release-gate review: an earlier version priced
# Trend off annualized_return(p), which spans the whole selected window and
# can label a ticker "Bullish" purely on a multi-year-old rally even while
# the last quarter is falling) ───────────────────────────────────────────

def test_recent_trend_return_uses_short_window_not_full_history():
    # Strong gains for a long stretch, then a sharp recent decline: the
    # full-history annualized_return is dominated by the long rally and
    # stays solidly positive, but the trailing TREND_LOOKBACK_DAYS window
    # is unambiguously negative.
    rally = _rising_prices(n=500, daily_return=0.004)
    decline_len = TREND_LOOKBACK_DAYS + 5
    decline_vals = [rally.iloc[-1]]
    for _ in range(decline_len):
        decline_vals.append(decline_vals[-1] * 0.99)
    combined = pd.concat([rally, _price_series(decline_vals[1:])])

    from src.financial_metrics import annualized_return
    assert annualized_return(combined) > 0.05  # full-window view: still "Bullish"
    assert recent_trend_return(combined) < -0.05  # recent-window view: "Bearish"


def test_trend_signal_can_disagree_with_full_window_return():
    rally = _rising_prices(n=500, daily_return=0.004)
    decline_len = TREND_LOOKBACK_DAYS + 5
    decline_vals = [rally.iloc[-1]]
    for _ in range(decline_len):
        decline_vals.append(decline_vals[-1] * 0.99)
    combined = pd.concat([rally, _price_series(decline_vals[1:])])

    signals = compute_quant_signals(combined, 0.02)
    assert signals["ret_ann"] > 0.05
    assert signals["trend"] == "Bearish"


def test_recent_trend_return_falls_back_gracefully_on_thin_history():
    p = _rising_prices(n=5, daily_return=0.01)
    # Must not raise and must not return NaN even though the series is far
    # shorter than TREND_LOOKBACK_DAYS.
    value = recent_trend_return(p)
    assert value == value  # not NaN
    assert value > 0


# ── Signal Agreement is a literal percentage of agreeing indicators, never
# rescaled/floored (Issue #20 release-gate review) ─────────────────────────

def test_signal_agreement_is_a_literal_percentage():
    p = _noisy_rising_prices()
    signals = compute_quant_signals(p, 0.02)
    # All four Quant-Score inputs (return, Sharpe, momentum, price-vs-MA)
    # agree with a strong sustained uptrend -> 100%, not a rescaled 95%.
    assert signals["signal_agreement"] == 100


def test_signal_agreement_multiple_of_25():
    # signal_agreement = round(agreement * 100) over 4 equally-weighted
    # indicators can only ever land on a multiple of 25.
    for base_return in (0.003, -0.003, 0.0001, -0.0001):
        p = _noisy_rising_prices(n=300, base_return=base_return)
        signals = compute_quant_signals(p, 0.02)
        assert signals["signal_agreement"] % 25 == 0


# ── generate_etf_interpretation (OpenAI Responses API call site) ───────────

def _sample_signals():
    return {
        "score": 58, "trend": "Bullish", "portfolio_view": "Neutral", "signal_agreement": 75,
        "ret_ann": 0.09, "ret_recent": 0.03, "vol": 0.22, "sharpe": 0.6, "mdd": -0.18, "mom": 0.01,
        "price": 105.0, "ma_short": 104.0, "ma_long": 100.0,
    }


def test_interpretation_rule_based_with_no_api_key():
    result = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", _sample_signals())
    assert result["source"] == "rule_based"
    assert result["text"]
    assert "trailing 60 trading days" in result["text"]
    assert "Quant Score" in result["text"]
    assert "Neutral band (36–64)" in result["text"]
    assert "Cumulative return over the period is strong" not in result["text"]
    assert result["error"]


def test_interpretation_uses_ai_when_call_succeeds(monkeypatch):
    fake_client = _FakeClient(text="Trend is bullish but the Quant Score is only middling because drawdown is elevated.")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", _sample_signals())
    assert result["source"] == "ai"
    assert "Quant Score" in result["text"]
    # The prompt sent to the model contains only already-computed values --
    # it is never asked to invent or recompute Trend/Score/View.
    call = fake_client.responses.calls[0]
    assert "58" in call["input"]
    assert "Bullish" in call["input"]


def test_interpretation_falls_back_to_rule_based_on_api_error(monkeypatch):
    fake_client = _FakeClient(exc=RuntimeError("simulated API failure"))
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", _sample_signals())
    assert result["source"] == "rule_based"
    assert result["text"]
    assert "different horizons and rules" in result["text"]


def test_interpretation_caches_by_fingerprint_avoiding_repeat_calls(monkeypatch):
    fake_client = _FakeClient(text="synthesis")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}
    signals = _sample_signals()

    r1 = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", signals, session_state=session_state)
    r2 = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", signals, session_state=session_state)
    assert r1 == r2
    assert len(fake_client.responses.calls) == 1  # unrelated rerun must not re-spend a call


def test_interpretation_recomputes_when_signals_change(monkeypatch):
    fake_client = _FakeClient(text="synthesis")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}

    generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", _sample_signals(), session_state=session_state)
    changed = _sample_signals()
    changed["score"] = 20
    changed["trend"] = "Bearish"
    generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", changed, session_state=session_state)

    assert len(fake_client.responses.calls) == 2


def test_interpretation_fingerprint_stable_for_identical_inputs():
    signals = _sample_signals()
    fp1 = interpretation_fingerprint("VOO", "en", "2024-01-01", "2024-06-01", signals)
    fp2 = interpretation_fingerprint("VOO", "en", "2024-01-01", "2024-06-01", dict(signals))
    assert fp1 == fp2



def test_rule_based_interpretation_zh_has_no_raw_english_trend_label():
    result = generate_etf_interpretation(
        "VOO", "zh-TW", "2024-01-01", "2024-06-01", _sample_signals()
    )
    assert result["source"] == "rule_based"
    assert "趨勢訊號「偏多」" in result["text"]
    assert "Bullish" not in result["text"]
    assert "中立區間（36–64）" in result["text"]
