"""
Authentication Architecture (Issue #22 section C)

Uses Streamlit's own native auth (st.login / st.logout / st.user), which is
an OpenID Connect (OIDC) client built into Streamlit itself -- no third-party
auth library or hard-coded credentials are added by this module. Streamlit's
native auth requires a [auth] section in .streamlit/secrets.toml pointing at
a real OIDC provider (Google, Microsoft Entra ID, Auth0, Okta, or any other
OIDC-compliant identity provider); see README/AUTH.md for the exact
deployment steps, since committing real provider credentials into this repo
would be a secret leak.

Design goal (Issue #22 section C): the app must NEVER become inaccessible to
a public demo visitor just because auth exists. Every function here is
therefore defensive -- if auth isn't configured (no [auth] secrets section
deployed) or the Streamlit runtime doesn't support it, the app falls back to
a single shared anonymous/demo identity (DEFAULT_USER_ID, "demo") rather than
raising or blocking access. Logged-in-only features (Current Holdings,
Watchlist, Daily Brief) simply operate on the shared "demo" namespace for
anonymous visitors, exactly as they did before auth existed -- this is what
"existing anonymous/demo DB data remains backward compatible" means in
practice (src/database.py's DEFAULT_USER_ID is the same constant used here).
"""

import streamlit as st

from src.database import DEFAULT_USER_ID


def is_auth_configured() -> bool:
    """True only if a real [auth] section is present in secrets (i.e. an
    operator has actually deployed OIDC provider credentials). Reading
    st.secrets for a section that was never configured raises in some
    Streamlit versions, so this is deliberately defensive -- "not
    configured" and "error reading secrets" are treated identically (both
    mean: fall back to public demo mode).
    """
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def is_authenticated() -> bool:
    """True only if auth is configured AND the current visitor has actually
    completed login. A visitor on a deployment with no auth configured, or
    who simply hasn't logged in, is NOT authenticated -- they use the shared
    public demo identity instead (see get_current_user_id())."""
    if not is_auth_configured():
        return False
    try:
        return bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        return False


def get_current_user_id() -> str:
    """The stable identifier user-scoped data (Current Holdings, Watchlist,
    Daily Brief) is stored/read under. Prefers the OIDC subject claim `sub`
    (guaranteed stable and unique per provider) over `email` (can change),
    falling back to email only if `sub` isn't exposed by the configured
    provider. Anonymous/public-demo visitors -- including every visitor on a
    deployment that never configured [auth] at all -- share DEFAULT_USER_ID,
    identical to this app's behavior before auth existed.
    """
    if not is_authenticated():
        return DEFAULT_USER_ID
    try:
        sub = getattr(st.user, "sub", None) or getattr(st.user, "email", None)
        return f"user:{sub}" if sub else DEFAULT_USER_ID
    except Exception:
        return DEFAULT_USER_ID


def get_current_user_display_name() -> str:
    """A human-readable label for the signed-in visitor (sidebar/greeting
    use only -- never used as a storage key, see get_current_user_id())."""
    if not is_authenticated():
        return ""
    try:
        return getattr(st.user, "name", None) or getattr(st.user, "email", None) or ""
    except Exception:
        return ""


def render_auth_status(sign_in_label: str, sign_out_label: str, signed_in_as_label: str) -> None:
    """Compact sign-in/sign-out control. Renders nothing but a disabled
    "not configured" caption when [auth] isn't deployed, so this is always
    safe to call from any page regardless of deployment state -- it never
    blocks the rest of the page from rendering.
    """
    if not is_auth_configured():
        st.caption(sign_in_label + " — " + "not configured for this deployment")
        return
    if is_authenticated():
        name = get_current_user_display_name()
        st.caption(f"{signed_in_as_label}: {name}")
        if st.button(sign_out_label, key="auth_sign_out_btn"):
            st.logout()
    else:
        if st.button(sign_in_label, key="auth_sign_in_btn", type="primary"):
            st.login()
