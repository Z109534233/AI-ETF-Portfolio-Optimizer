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
    compute_quant_signals, trend_signal_from_return, portfolio_view_from_score,
    generate_etf_interpretation, interpretation_fingerprint,
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
    trend_signal_from_return() rather than an independently-tuned rule."""
    p = _rising_prices(daily_return=0.0005)
    signals = compute_quant_signals(p, 0.05)
    assert signals["trend"] == trend_signal_from_return(signals["ret_ann"])


# ── generate_etf_interpretation (OpenAI Responses API call site) ───────────

def _sample_signals():
    return {
        "score": 58, "trend": "Bullish", "portfolio_view": "Neutral", "signal_agreement": 75,
        "ret_ann": 0.09, "vol": 0.22, "sharpe": 0.6, "mdd": -0.18, "mom": 0.01,
        "price": 105.0, "ma_short": 104.0, "ma_long": 100.0,
    }


def test_interpretation_rule_based_with_no_api_key():
    result = generate_etf_interpretation("VOO", "en", "2024-01-01", "2024-06-01", _sample_signals())
    assert result["source"] == "rule_based"
    assert result["text"] is None
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
    assert result["text"] is None


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
