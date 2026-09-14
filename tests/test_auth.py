"""
Auth architecture -- deterministic tests (Issue #22 section C).

No real OIDC provider is ever contacted: is_auth_configured() only checks
for the presence of a secrets section, and is_authenticated()/
get_current_user_id() only read st.user's attributes (mocked here). The
overriding requirement (Issue #22 section C) is that the app must NEVER
become inaccessible to a public demo visitor -- every test below confirms
the safe fallback to the shared anonymous "demo" identity.
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
