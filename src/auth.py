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



def require_login() -> bool:
    """Show a dedicated Google sign-in screen when OIDC auth is configured.

    The gate is intentionally inactive until deployment secrets contain an
    [auth] section. This keeps local development and first-time deployment
    usable while the Google OAuth client is being configured. As soon as
    [auth] is present, every page that calls this helper requires a signed-in
    Google identity before rendering private portfolio tools.

    Returns True for an authenticated user and False when auth is not yet
    configured. For an unauthenticated visitor on an auth-enabled deployment,
    this function renders the login screen and stops the Streamlit script.
    """
    if not is_auth_configured():
        return False

    if is_authenticated():
        return True

    # Keep the login experience focused and separate from the analytics UI.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .block-container {
            max-width: 760px;
            padding-top: 8vh;
            padding-bottom: 6vh;
        }
        .login-shell {
            text-align: center;
            padding: 28px 20px 8px;
        }
        .login-mark {
            width: 64px;
            height: 64px;
            margin: 0 auto 18px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 25px;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: white;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            box-shadow: 0 14px 34px rgba(37, 99, 235, 0.24);
        }
        .login-title {
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin-bottom: 10px;
        }
        .login-subtitle {
            max-width: 580px;
            margin: 0 auto 22px;
            color: #8b93a7;
            font-size: 1rem;
            line-height: 1.65;
        }
        .login-note {
            margin-top: 18px;
            color: #8b93a7;
            font-size: 0.82rem;
            line-height: 1.55;
        }
        </style>
        <div class="login-shell">
            <div class="login-mark">AI</div>
            <div class="login-title">AI ETF Portfolio Optimizer</div>
            <div class="login-subtitle">
                Sign in to keep your portfolio data private and sync your saved
                analysis across sessions.<br>
                使用 Google 帳號登入，安全保存你的投資組合與分析紀錄。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.35, 1])
    with center:
        st.button(
            "Continue with Google",
            type="primary",
            use_container_width=True,
            on_click=st.login,
            key="google_login_btn",
        )
        st.markdown(
            '<div class="login-note">'
            'We use Google only to verify your identity. '
            'This app does not read your Gmail inbox, contacts, or Drive files.'
            '<br>Google 僅用於登入驗證，不會讀取 Gmail 信件、聯絡人或雲端硬碟。'
            '</div>',
            unsafe_allow_html=True,
        )

    st.stop()
    return False


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
