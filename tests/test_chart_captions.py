"""
Chart/table "what this shows" captions + optional "Ask AI to interpret"
button -- deterministic tests (Issue #22 section L).

chart_caption() is pure text rendering (no API access at all) and is
covered indirectly by the smoke tests already asserting these pages render
without exception. This file focuses on ai_interpret_button()'s contract:
never calls the real OpenAI SDK, never auto-calls on render, only calls
(and caches) on explicit click.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from streamlit.testing.v1 import AppTest
import streamlit as st

from src import ui as ui_mod
from src import openai_service


def _apptest_from_file(rel_path, **kwargs):
    st.page_link = lambda *a, **k: None
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _render_interpret_key1():
    from src import ui as ui_mod
    import streamlit as st
    ui_mod.ai_interpret_button("test_key", st.session_state, "some context")


def _render_interpret_key2():
    from src import ui as ui_mod
    import streamlit as st
    ui_mod.ai_interpret_button("test_key2", st.session_state, "ctx")


def test_ai_interpret_button_renders_disabled_state_when_not_configured(monkeypatch):
    """Issue #24 follow-up: a silent no-op made the whole feature invisible
    when OpenAI isn't configured. It must now render a visibly disabled
    button (so the user can see the capability exists) plus a caption
    explaining why -- but still never spend an API call just from
    rendering."""
    monkeypatch.setattr(openai_service, "is_configured", lambda: False)
    calls = {"n": 0}
    monkeypatch.setattr(openai_service, "cached_generate", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    at = AppTest.from_function(_render_interpret_key1)
    at.run()
    assert at.exception == []
    assert len(at.button) == 1
    assert at.button[0].disabled is True
    assert calls["n"] == 0


def test_ai_interpret_button_does_not_call_api_until_clicked(monkeypatch):
    monkeypatch.setattr(openai_service, "is_configured", lambda: True)
    calls = {"n": 0}

    def _fake_cached_generate(session_state, cache_key, fp, system_prompt, context_text, **kwargs):
        calls["n"] += 1
        return {"available": True, "text": "MOCKED INTERPRETATION", "source": "ai"}

    monkeypatch.setattr(openai_service, "cached_generate", _fake_cached_generate)

    at = AppTest.from_function(_render_interpret_key1)
    at.run()
    assert at.exception == []
    assert calls["n"] == 0  # rendering the button alone must never call the API
    assert "MOCKED INTERPRETATION" not in "\n".join(i.value for i in at.info)

    at.button(key="test_key_btn").click()
    at.run()
    assert calls["n"] == 1
    assert "MOCKED INTERPRETATION" in "\n".join(i.value for i in at.info)

    # A second run with nothing changed must not call the API again --
    # cached_generate() itself is responsible for the fingerprint check, but
    # this confirms the button doesn't force a fresh call on every rerun.
    at.run()
    assert calls["n"] == 1


def test_ai_interpret_button_never_calls_real_openai_client(monkeypatch):
    """Guard against accidentally bypassing cached_generate() and calling
    generate_text()/the raw OpenAI SDK directly."""
    monkeypatch.setattr(openai_service, "is_configured", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("generate_text() must never be called directly by ai_interpret_button()")

    monkeypatch.setattr(openai_service, "generate_text", _boom)
    monkeypatch.setattr(openai_service, "cached_generate",
                         lambda session_state, cache_key, fp, sp, ctx, **k: {"available": True, "text": "ok", "source": "ai"})

    at = AppTest.from_function(_render_interpret_key2)
    at.run()
    at.button(key="test_key2_btn").click()
    at.run()
    assert at.exception == []
