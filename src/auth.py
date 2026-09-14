"""
Authentication Architecture (Issue #22 section C, extended by Issue #26)

Uses Streamlit's own native auth (st.login / st.logout / st.user), which is
an OpenID Connect (OIDC) client built into Streamlit itself -- no third-party
auth library or hard-coded credentials are added by this module (Streamlit's
native auth uses Authlib under the hood, listed in requirements.txt, but this
app never talks to Authlib or any OAuth endpoint directly). Streamlit's
native auth requires a [auth] section in .streamlit/secrets.toml pointing at
a real OIDC provider (Google, by default for this deployment); see
AUTH.md for the exact deployment steps, since committing real provider
credentials into this repo would be a secret leak.

Two deployment states this module must always behave correctly in:

1. Auth NOT configured (no [auth] secrets deployed -- e.g. local dev, CI,
   or the test suite, none of which ship real OAuth credentials). The app
   must NEVER become inaccessible in this state: require_login() is a no-op
   and every page renders exactly as it did before Issue #26, on the shared
   anonymous/demo identity (DEFAULT_USER_ID, "demo"). This is the same
   "never break the public demo" guarantee Issue #22 established.

2. Auth configured (a real deployment, e.g. Streamlit Community Cloud with
   Google OAuth secrets set). require_login() now actively gates every page:
   an unauthenticated visitor sees the sign-in screen (rendered in place,
   matching the FinTech UI) instead of the page content, and direct URLs to
   any child page are gated the same way -- there is no bypass.
"""

import streamlit as st

from src.database import DEFAULT_USER_ID
from src.i18n import t


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
    (guaranteed stable and unique per provider, and the only claim Google
    itself guarantees never changes for a given account) over `email` (can
    change), falling back to email only if `sub` isn't exposed by the
    configured provider. Anonymous/public-demo visitors -- including every
    visitor on a deployment that never configured [auth] at all -- share
    DEFAULT_USER_ID, identical to this app's behavior before auth existed.
    """
    if not is_authenticated():
        return DEFAULT_USER_ID
    try:
        sub = getattr(st.user, "sub", None) or getattr(st.user, "email", None)
        return f"user:{sub}" if sub else DEFAULT_USER_ID
    except Exception:
        return DEFAULT_USER_ID


def get_current_user_email() -> str:
    """The signed-in visitor's email (sidebar account section display only --
    never used as a storage key, see get_current_user_id())."""
    if not is_authenticated():
        return ""
    try:
        return getattr(st.user, "email", None) or ""
    except Exception:
        return ""


def get_current_user_display_name() -> str:
    """A human-readable label for the signed-in visitor (sidebar/greeting
    use only -- never used as a storage key, see get_current_user_id())."""
    if not is_authenticated():
        return ""
    try:
        return getattr(st.user, "name", None) or getattr(st.user, "email", None) or ""
    except Exception:
        return ""


def _hide_default_sidebar_nav() -> None:
    """Hide Streamlit's own auto-generated multipage nav list (the one it
    builds from pages/*.py, shown above any custom sidebar content) while
    the sign-in screen is on screen. This app doesn't use st.navigation()
    (it relies on the classic pages/ auto-discovery), so this CSS rule is
    the only way to keep an unauthenticated visitor from using that built-in
    widget to browse into a page's content without signing in first --
    require_login() itself is still what actually blocks the content via
    st.stop(), this only avoids showing a dead-end nav list on top of it.
    """
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none;}</style>",
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    """The dedicated pre-login screen: a centered FinTech-styled card with
    the product brand, a bilingual "Continue with Google" CTA wired to
    st.login(), and a privacy note. Shown by require_login() in place of
    page content for any unauthenticated visitor on any page -- so a direct
    URL to a child page shows this exact screen too, not just app.py.

    Deliberately bilingual (Traditional Chinese + English shown together,
    not gated behind the session language toggle) since the sidebar
    language selector is hidden pre-login along with the rest of the normal
    sidebar (_hide_default_sidebar_nav() above + no render_sidebar_nav()
    call from any gated page while unauthenticated).
    """
    _hide_default_sidebar_nav()

    st.markdown(
        """
        <style>
        .auth-login-wrap { display:flex; justify-content:center; padding-top:48px; }
        .auth-login-card {
            max-width:440px; width:100%; text-align:center;
            background:var(--surface); border:1px solid var(--border);
            border-radius:var(--radius-lg); box-shadow:var(--shadow-sm);
            padding:40px 32px;
        }
        .auth-login-mark {
            width:56px; height:56px; margin:0 auto 20px; border-radius:16px;
            background:var(--primary); color:#fff; font-weight:800;
            font-size:22px; display:flex; align-items:center; justify-content:center;
        }
        .auth-login-title { color:var(--text); font-size:20px; font-weight:800; margin-bottom:6px; }
        .auth-login-subtitle { color:var(--text-secondary); font-size:13.5px; margin-bottom:24px; line-height:1.5; }
        .auth-login-privacy { color:var(--text-muted); font-size:11.5px; margin-top:16px; line-height:1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="auth-login-wrap"><div class="auth-login-card">'
        '<div class="auth-login-mark">AI</div>'
        '<div class="auth-login-title">AI ETF 投資組合最佳化平台</div>'
        '<div class="auth-login-title" style="font-size:15px;font-weight:700;margin-top:-4px;">'
        'AI ETF Portfolio Optimizer</div>'
        '<div class="auth-login-subtitle">登入以繼續使用你的個人化投資組合、觀察清單與每日摘要。<br>'
        'Sign in to continue to your personalized portfolio, watchlist, and daily brief.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    _, col_btn, _ = st.columns([1, 1.4, 1])
    with col_btn:
        if st.button(
            "Continue with Google ・ 使用 Google 繼續登入",
            type="primary", use_container_width=True, key="auth_login_cta_btn",
        ):
            st.login()

    st.markdown(
        '<div class="auth-login-wrap" style="padding-top:0;"><div class="auth-login-card" '
        'style="padding:16px 32px;box-shadow:none;">'
        '<div class="auth-login-privacy">'
        '🔒 我們僅會取得你的姓名與電子郵件（openid / profile / email），'
        '不會存取你的 Gmail、雲端硬碟、聯絡人或其他 Google 資料。<br>'
        'We only request your name and email (openid / profile / email) -- '
        'never your Gmail, Drive, Contacts, or any other Google data.'
        '</div></div></div>',
        unsafe_allow_html=True,
    )


def require_login() -> None:
    """Central access guard. Call this at the top of app.py and every page
    in pages/, immediately after st.set_page_config()+load_css() and before
    any sidebar or content rendering.

    No-op (app renders normally, shared demo identity) when auth isn't
    configured for this deployment -- see the module docstring. When auth
    IS configured, an unauthenticated visitor gets the sign-in screen
    instead of the page's real content (via st.stop(), so nothing below
    this call in the caller ever executes), regardless of which page URL
    they landed on directly -- there is no page that skips this check.
    """
    if not is_auth_configured():
        return
    if is_authenticated():
        return
    render_login_page()
    st.stop()


def render_account_section() -> None:
    """Compact, shared "signed in as" sidebar block: display name + email +
    a single Sign out button. Call once, inside `with st.sidebar:`, right
    after render_sidebar_nav() on every page -- this is the ONE place a
    sign-out button is rendered anywhere in the app, so there is never a
    duplicate.

    Renders nothing when auth isn't configured for this deployment (public
    demo mode has no account to show), and nothing if somehow reached while
    unauthenticated (require_login() already stops the script before the
    sidebar renders in that case) -- both are defensive no-ops, never a
    crash or a misleading "not signed in" widget cluttering every page.
    """
    if not is_auth_configured() or not is_authenticated():
        return
    name = get_current_user_display_name()
    email = get_current_user_email()
    st.markdown(f'<div class="sidebar-nav-label">{t("account_section_label")}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"**{name}**" if name else f"**{email}**")
        if email and email != name:
            st.caption(email)
        if st.button(t("account_sign_out"), key="auth_sign_out_btn", use_container_width=True):
            st.logout()
