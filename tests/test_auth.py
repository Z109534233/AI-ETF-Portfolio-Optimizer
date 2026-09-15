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


# ── require_login() / render_account_section() (Issue #26) ─────────────────
# require_login() is the central page guard called from app.py and every
# page in pages/. It must be a strict no-op (never redirect) unless a
# deployment BOTH has [auth] configured AND the visitor isn't signed in --
# otherwise the public demo would become unreachable, exactly the failure
# mode Issue #22 section C was written to prevent.

def test_require_login_noop_when_not_configured(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: False)
    redirects = []
    monkeypatch.setattr(st, "switch_page", lambda page: redirects.append(page))
    auth.require_login()
    assert redirects == []


def test_require_login_noop_when_already_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)
    monkeypatch.setattr(auth, "is_authenticated", lambda: True)
    redirects = []
    monkeypatch.setattr(st, "switch_page", lambda page: redirects.append(page))
    auth.require_login()
    assert redirects == []


def test_require_login_redirects_to_login_page_when_signed_out(monkeypatch):
    monkeypatch.setattr(auth, "is_auth_configured", lambda: True)
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    redirects = []
    monkeypatch.setattr(st, "switch_page", lambda page: redirects.append(page))
    auth.require_login()
    assert redirects == [auth.LOGIN_PAGE]


def test_render_account_section_silent_when_not_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: False)
    rendered = []
    monkeypatch.setattr(st, "markdown", lambda *a, **k: rendered.append(a))
    monkeypatch.setattr(st, "caption", lambda *a, **k: rendered.append(a))
    auth.render_account_section()
    assert rendered == []


def test_render_account_section_shows_signed_in_user_and_sign_out(monkeypatch):
    monkeypatch.setattr(auth, "is_authenticated", lambda: True)
    monkeypatch.setattr(auth, "get_current_user_display_name", lambda: "Alice")
    captions = []
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: captions.append(a))
    logged_out = []
    monkeypatch.setattr(st, "button", lambda *a, **k: True)
    monkeypatch.setattr(st, "logout", lambda: logged_out.append(True))
    auth.render_account_section()
    assert any("Alice" in str(c) for c in captions)
    assert logged_out == [True]
