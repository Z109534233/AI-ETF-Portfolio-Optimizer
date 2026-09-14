"""
src.openai_service tests (Issue #20 -- OpenAI Responses API architecture).

These are the "OpenAI security/cost/grounding acceptance" tests the review
requires: no real network call is ever made here -- every test either
clears the API key entirely or monkeypatches src.openai_service.get_client
with a fake client, so this suite can run in CI with no OPENAI_API_KEY and
no internet access.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from src import openai_service as svc


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch):
    """Every test starts with a clean slate: no OPENAI_API_KEY/OPENAI_MODEL
    in the environment, and st.secrets unavailable (raises/empty), so tests
    are deterministic regardless of the machine's actual environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(svc, "_secret", lambda name: None)


# ── No API key -> app still works, rule-based fallback ──────────────────────

def test_no_api_key_is_not_configured():
    assert svc.is_configured() is False
    assert svc.get_api_key() is None


def test_no_api_key_get_client_returns_none():
    assert svc.get_client() is None


def test_no_api_key_generate_text_falls_back_cleanly():
    result = svc.generate_text("system prompt", "user content")
    assert result["available"] is False
    assert result["source"] == "rule_based"
    assert result["text"] is None
    assert "not configured" in result["error"].lower() or "api key" in result["error"].lower()


def test_default_model_used_when_no_override():
    assert svc.get_model() == svc.DEFAULT_MODEL


def test_model_override_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna-mini")
    assert svc.get_model() == "gpt-5.6-luna-mini"


def test_api_key_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    assert svc.is_configured() is True
    assert svc.get_api_key() == "sk-test-not-a-real-key"


# ── Mocked successful Responses API call -> AI text is returned ────────────

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


def test_mocked_successful_call_returns_ai_text(monkeypatch):
    fake_client = _FakeClient(text="This portfolio shows a tension between a positive trend and a neutral quant score.")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    monkeypatch.setattr(svc, "get_model", lambda: "gpt-5.6-luna")

    result = svc.generate_text("system prompt", "structured user content")
    assert result["available"] is True
    assert result["source"] == "ai"
    assert "tension" in result["text"]
    assert result["model"] == "gpt-5.6-luna"
    # The Responses API (not legacy chat.completions) is used, and the
    # prompt is exactly what was passed in -- no numeric computation
    # happens inside this module.
    call = fake_client.responses.calls[0]
    assert call["instructions"] == "system prompt"
    assert call["input"] == "structured user content"


def test_mocked_empty_output_text_falls_back():
    fake_client = _FakeClient(text="")
    result = None
    import src.openai_service as mod
    old = mod.get_client
    mod.get_client = lambda: fake_client
    try:
        result = svc.generate_text("sys", "content")
    finally:
        mod.get_client = old
    assert result["available"] is False
    assert result["source"] == "rule_based"


# ── API exception -> app still works, accurate fallback badge ──────────────

def test_api_exception_falls_back_cleanly_without_raising(monkeypatch):
    fake_client = _FakeClient(exc=TimeoutError("simulated timeout"))
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = svc.generate_text("sys", "content")
    assert result["available"] is False
    assert result["source"] == "rule_based"
    assert "TimeoutError" in result["error"]
    assert "simulated timeout" in result["error"]


def test_exception_message_never_leaks_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    fake_client = _FakeClient(exc=RuntimeError("boom"))
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)

    result = svc.generate_text("sys", "content")
    assert "sk-super-secret-value" not in result["error"]
    assert "sk-super-secret-value" not in str(result)


# ── Fingerprinting: deterministic, sensitive to inputs ──────────────────────

def test_fingerprint_is_deterministic():
    a = svc.fingerprint("QQQ", "2024-01-01", "2024-06-01", 0.62)
    b = svc.fingerprint("QQQ", "2024-01-01", "2024-06-01", 0.62)
    assert a == b


def test_fingerprint_changes_when_any_input_changes():
    base = svc.fingerprint("QQQ", "2024-01-01", "2024-06-01", 0.62)
    changed_ticker = svc.fingerprint("VOO", "2024-01-01", "2024-06-01", 0.62)
    changed_value = svc.fingerprint("QQQ", "2024-01-01", "2024-06-01", 0.71)
    assert base != changed_ticker
    assert base != changed_value


def test_fingerprint_never_includes_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    fp = svc.fingerprint("QQQ", 0.62)
    assert "sk-should-not-appear" not in fp


# ── cached_generate: invalidate on change, no wasted calls otherwise ───────

def test_cached_generate_reuses_result_for_unchanged_fingerprint(monkeypatch):
    fake_client = _FakeClient(text="synthesis text")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}

    fp = svc.fingerprint("QQQ", 0.12, 0.18)
    r1 = svc.cached_generate(session_state, "advisor_cache", fp, "sys", "content")
    r2 = svc.cached_generate(session_state, "advisor_cache", fp, "sys", "content")

    assert r1 == r2
    # Only ONE real API call was made even though cached_generate() was
    # invoked twice -- an unrelated Streamlit rerun must never re-spend
    # an API call for the same underlying inputs.
    assert len(fake_client.responses.calls) == 1


def test_cached_generate_calls_again_when_fingerprint_changes(monkeypatch):
    fake_client = _FakeClient(text="synthesis text")
    monkeypatch.setattr(svc, "get_client", lambda: fake_client)
    session_state = {}

    fp1 = svc.fingerprint("QQQ", 0.12)
    fp2 = svc.fingerprint("QQQ", 0.20)  # e.g. volatility input changed
    svc.cached_generate(session_state, "advisor_cache", fp1, "sys", "content")
    svc.cached_generate(session_state, "advisor_cache", fp2, "sys", "content")

    assert len(fake_client.responses.calls) == 2


def test_cached_generate_falls_back_when_not_configured():
    session_state = {}
    fp = svc.fingerprint("QQQ", 0.12)
    result = svc.cached_generate(session_state, "advisor_cache", fp, "sys", "content")
    assert result["available"] is False
    assert result["source"] == "rule_based"
