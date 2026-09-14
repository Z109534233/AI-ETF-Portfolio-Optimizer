"""
Auth architecture -- deterministic tests (Issue #22 section C, extended by
Issue #26's mandatory require_login() guard).

No real OIDC provider is ever contacted: is_auth_configured() only checks
for the presence of a secrets section, and is_authenticated()/
get_current_user_id() only read st.user's attributes (mocked here). The
overriding requirement (Issue #22 section C) is that the app must NEVER
become inaccessible to a public demo visitor when auth isn't configured --
every test below confirms the safe fallback to the shared anonymous "demo"
identity in that state, and that require_login() only actively gates when
auth IS configured (Issue #26).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import streamlit as st

from src import auth
from src.database import DEFAULT_USER_ID


def test_not_configured_falls_back_to_public_demo(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: False)
    assert auth.is_authenticated() is False
    assert auth.get_current_user_id() == DEFAULT_USER_ID
    assert DEFAULT_USER_ID == "demo"


def test_configured_but_not_logged_in_uses_demo_identity(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)

    class _AnonUser:
        is_logged_in = False

    monkeypatch.setattr(st, "user", _AnonUser())
    assert auth.is_authenticated() is False
    assert auth.get_current_user_id() == DEFAULT_USER_ID


def test_logged_in_user_gets_stable_scoped_id(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)

    class _LoggedInUser:
        is_logged_in = True
        sub = "abc123"
        email = "alice@example.com"
        name = "Alice"

    monkeypatch.setattr(st, "user", _LoggedInUser())
    assert auth.is_authenticated() is True
    user_id = auth.get_current_user_id()
    assert user_id == "user:abc123"
    # Calling twice must return the exact same identifier (stability).
    assert auth.get_current_user_id() == user_id
    assert auth.get_current_user_display_name() == "Alice"


def test_logged_in_user_without_sub_falls_back_to_email(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)

    class _LoggedInUser:
        is_logged_in = True
        sub = None
        email = "bob@example.com"
        name = None

    monkeypatch.setattr(st, "user", _LoggedInUser())
    assert auth.get_current_user_id() == "user:bob@example.com"


def test_get_current_user_email_requires_authentication(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: False)
    assert auth.get_current_user_email() == ""


def test_get_current_user_email_returns_email_when_logged_in(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)

    class _LoggedInUser:
        is_logged_in = True
        sub = "abc123"
        email = "alice@example.com"
        name = "Alice"

    monkeypatch.setattr(st, "user", _LoggedInUser())
    assert auth.get_current_user_email() == "alice@example.com"


def test_require_login_is_noop_when_not_configured(monkeypatch):
    # Issue #22's non-negotiable guarantee, restated for Issue #26: a
    # deployment (or the test suite / CI) that never configured [auth]
    # secrets must never be blocked from the app by require_login().
    monkeypatch.setattr(auth, "is_auth_configured", lambda: False)
    calls = []
    monkeypatch.setattr(auth, "render_login_page", lambda: calls.append("rendered"))
    auth.require_login()
    assert calls == []


def test_require_login_is_noop_when_already_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)
    monkeypatch.setattr(auth, "is_authenticated", lambda: True)
    calls = []
    monkeypatch.setattr(auth, "render_login_page", lambda: calls.append("rendered"))
    auth.require_login()
    assert calls == []


def test_require_login_renders_login_screen_and_stops_when_gated(monkeypatch):
    # Configured but not signed in -- the one state where require_login()
    # must actively block (Issue #26's "direct child-page URLs cannot
    # bypass authentication" requirement).
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    calls = []
    monkeypatch.setattr(auth, "render_login_page", lambda: calls.append("rendered"))
    monkeypatch.setattr(st, "stop", lambda: calls.append("stopped"))
    auth.require_login()
    assert calls == ["rendered", "stopped"]


def test_is_auth_configured_handles_missing_secrets_gracefully():
    # st.secrets access without a configured secrets.toml must never raise
    # out of is_auth_configured() -- it must resolve to "not configured".
    assert auth.is_auth_configured() in (True, False)


def test_get_current_user_id_never_raises_on_broken_user_object(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)

    class _BrokenUser:
        @property
        def is_logged_in(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(st, "user", _BrokenUser())
    assert auth.get_current_user_id() == DEFAULT_USER_ID
    assert auth.is_authenticated() is False
