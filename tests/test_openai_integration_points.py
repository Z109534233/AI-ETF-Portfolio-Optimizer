"""
Mocked integration tests for the two OpenAI call sites migrated to the
central Responses API service (src.openai_service) in Issue #20:
  - src.ai_advisor.generate_advisor_narrative()  (Portfolio Analyst synthesis)
  - src.market_intelligence.generate_market_summary()  (Today's Market Summary)

No real network call is made anywhere in this file -- every test either
clears OPENAI_API_KEY or monkeypatches src.openai_service.get_client with a
fake client, matching the "no real OpenAI network call in pytest" gate.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from src import openai_service as svc
from src.ai_advisor import build_advisor_context, generate_advisor_narrative
from src.market_intelligence import generate_market_summary


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


def _sample_portfolio():
    return {
        "strategy": "Maximum Sharpe", "market": "United States",
        "tickers": ["QQQ", "BND"], "weights": {"QQQ": 0.7, "BND": 0.3},
        "investment_amount": 10000.0, "expected_return": 0.12,
        "volatility": 0.18, "sharpe_ratio": 0.61, "max_drawdown": -0.25,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }


# ── Portfolio Analyst (ai_advisor) ──────────────────────────────────────────

def test_advisor_narrative_is_rule_based_with_no_api_key():
    context = build_advisor_context(portfolio=_sample_portfolio(), portfolio_source="current")
    result = generate_advisor_narrative(context)
    assert result["source"] == "rule_based"
    assert "QQQ" in result["text"]


def test_advisor_narrative_uses_ai_when_call_succeeds(monkeypatch):
    fake_client = _FakeClient(text="A synthesis grounded only in the provided figures.")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    context = build_advisor_context(portfolio=_sample_portfolio(), portfolio_source="current")

    result = generate_advisor_narrative(context)
    assert result["source"] == "ai"
    assert result["text"] == "A synthesis grounded only in the provided figures."
    # The Responses API is used (not chat.completions), and the prompt sent
    # to the model contains the real computed weight, never a placeholder.
    call = fake_client.responses.calls[0]
    assert "QQQ: 70.0%" in call["input"]


def test_advisor_narrative_falls_back_to_rule_based_on_api_error(monkeypatch):
    fake_client = _FakeClient(exc=RuntimeError("simulated API failure"))
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    context = build_advisor_context(portfolio=_sample_portfolio(), portfolio_source="current")

    result = generate_advisor_narrative(context)
    assert result["source"] == "rule_based"
    assert "QQQ" in result["text"]


def test_advisor_narrative_caches_by_fingerprint_avoiding_repeat_calls(monkeypatch):
    fake_client = _FakeClient(text="synthesis")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    context = build_advisor_context(portfolio=_sample_portfolio(), portfolio_source="current")
    session_state = {}

    r1 = generate_advisor_narrative(context, session_state=session_state)
    r2 = generate_advisor_narrative(context, session_state=session_state)
    assert r1 == r2
    assert len(fake_client.responses.calls) == 1  # unrelated rerun must not re-spend a call


def test_advisor_narrative_recomputes_when_portfolio_changes(monkeypatch):
    fake_client = _FakeClient(text="synthesis")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}

    context1 = build_advisor_context(portfolio=_sample_portfolio(), portfolio_source="current")
    generate_advisor_narrative(context1, session_state=session_state)

    changed_portfolio = _sample_portfolio()
    changed_portfolio["weights"] = {"QQQ": 0.5, "BND": 0.5}
    context2 = build_advisor_context(portfolio=changed_portfolio, portfolio_source="current")
    generate_advisor_narrative(context2, session_state=session_state)

    assert len(fake_client.responses.calls) == 2


# ── Market Intelligence summary ─────────────────────────────────────────────

def _sample_news():
    return [
        {"title": "Fed signals possible rate cut", "impact": "Positive"},
        {"title": "Tech earnings beat expectations", "impact": "Positive"},
    ]


def _sample_sentiment():
    return {"bullish_pct": 60.0, "neutral_pct": 30.0, "bearish_pct": 10.0}


def test_market_summary_rule_based_with_no_api_key():
    result = generate_market_summary(_sample_news(), _sample_sentiment(), [])
    assert result["source"] == "rule_based"
    assert result["text"]


def test_market_summary_uses_ai_when_call_succeeds(monkeypatch):
    fake_client = _FakeClient(text="Markets were broadly positive today on Fed and tech news.")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = generate_market_summary(_sample_news(), _sample_sentiment(), [])
    assert result["source"] == "ai"
    assert "positive" in result["text"].lower()


def test_market_summary_falls_back_on_api_error(monkeypatch):
    fake_client = _FakeClient(exc=RuntimeError("boom"))
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = generate_market_summary(_sample_news(), _sample_sentiment(), [])
    assert result["source"] == "rule_based"
    assert result["text"]


def test_market_summary_no_news_never_calls_api(monkeypatch):
    fake_client = _FakeClient(text="should never be used")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = generate_market_summary([], _sample_sentiment(), [])
    assert result["source"] == "rule_based"
    assert len(fake_client.responses.calls) == 0


def test_market_summary_caches_by_fingerprint(monkeypatch):
    fake_client = _FakeClient(text="summary")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}

    r1 = generate_market_summary(_sample_news(), _sample_sentiment(), [], session_state=session_state)
    r2 = generate_market_summary(_sample_news(), _sample_sentiment(), [], session_state=session_state)
    assert r1 == r2
    assert len(fake_client.responses.calls) == 1
