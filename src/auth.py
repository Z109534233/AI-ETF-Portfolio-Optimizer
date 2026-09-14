"""
Authentication Architecture (Issue #22 section C, extended by Issue #26)

Uses Streamlit's own native auth (st.login / st.logout / st.user), which is
an OpenID Connect (OIDC) client built into Streamlit itself -- no third-party
auth library or hard-coded credentials are added by this module (Authlib is
a runtime dependency Streamlit's native auth needs, not a separate auth
implementation of ours). Streamlit's native auth requires a [auth] section in
.streamlit/secrets.toml pointing at a real OIDC provider (Google is the
supported/documented provider for this app; see README/AUTH.md for the exact
deployment steps), since committing real provider credentials into this repo
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

Issue #26 adds the other half of that same rule as an opt-in: require_login()
is a central page guard that only starts requiring Google sign-in once an
operator actually deploys [auth] secrets -- a fresh checkout with no secrets
configured stays the same open public demo described above. See
pages/9_Login.py for the dedicated login page require_login() redirects to,
and render_account_section() for the compact sidebar widget every page picks
up via src.ui.render_sidebar_nav().
"""

import streamlit as st

from src.database import DEFAULT_USER_ID
from src.i18n import t

LOGIN_PAGE = "pages/9_Login.py"


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


def require_login() -> None:
    """Central page guard (Issue #26) -- call once, near the top of every
    page, right after st.set_page_config(). If this deployment has real
    Google OIDC secrets configured (is_auth_configured()) and the current
    visitor hasn't completed sign-in yet, this stops the current page from
    rendering and sends them to the dedicated login page (LOGIN_PAGE)
    instead.

    A deployment with no [auth] secrets configured at all -- including
    every checkout of this repo before a deployer opts in -- gets a no-op
    here and keeps behaving exactly like the pre-Issue#26 public demo: this
    guard can only ever make the app MORE restrictive than "always open",
    never less, and only once an operator has actually turned Google
    sign-in on.
    """
    if not is_auth_configured():
        return
    if is_authenticated():
        return
    st.switch_page(LOGIN_PAGE)


def render_account_section() -> None:
    """Compact signed-in account widget for the sidebar (Issue #26). Called
    once inside src.ui.render_sidebar_nav(), so every page picks it up
    automatically without each page having to render it itself.

    Deliberately silent (renders nothing) unless the visitor is actually
    signed in: require_login() already keeps anyone else off these pages
    once auth is configured, and a deployment with no [auth] secrets has no
    account concept to show at all -- exactly the same "never add visible
    clutter to the public demo" rule the rest of this module follows.
    """
    if not is_authenticated():
        return
    st.markdown(f"### {t('auth_account_section_title')}")
    st.caption(f"{t('auth_signed_in_as')}: {get_current_user_display_name()}")
    if st.button(t("auth_sign_out"), key="sidebar_auth_sign_out_btn", use_container_width=True):
        st.logout()
