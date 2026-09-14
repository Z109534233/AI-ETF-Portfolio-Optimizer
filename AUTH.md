# Authentication (Issue #22 section C, extended by Issue #26)

The app uses **Streamlit's own native authentication** (`st.login()` /
`st.logout()` / `st.user`), which is an OpenID Connect (OIDC) client built
into Streamlit itself (Streamlit >= 1.42, using Authlib internally — see
`requirements.txt`). No custom auth framework and no hard-coded credentials
are added by this app — see `src/auth.py`.

## Current status in this repository

- `src/auth.py` provides `is_auth_configured()`, `is_authenticated()`,
  `get_current_user_id()`, `get_current_user_email()`,
  `get_current_user_display_name()`, `require_login()`, and
  `render_account_section()`.
- `require_login()` is called at the top of `app.py` and every page in
  `pages/` (all 8), immediately after `st.set_page_config()` + `load_css()`
  and before any sidebar or page content renders. This means a direct URL
  to any child page is gated exactly the same way as the home page — there
  is no page that bypasses the check.
- `get_current_user_id()` returns a stable, provider-issued identifier
  (`user:<sub>`) when a visitor is signed in — `st.user.sub` is preferred,
  falling back to `st.user.email` only if a provider doesn't expose `sub`.
  My Portfolio's Current Holdings, Watchlist, and Daily Brief are wired to
  this, so they are user-scoped the moment auth is configured.
- `render_account_section()` renders the ONE shared "signed in as" block
  (display name + email + a single Sign out button) in the sidebar of every
  page, right after the nav — there is only ever one sign-out control
  anywhere in the app.

## Two deployment states

**1. Auth not configured** (no `[auth]` section in secrets — local dev, CI,
and the automated test suite all run in this state, since none of them ship
real OAuth credentials). `require_login()` is a **no-op** and the entire app
renders exactly as it did before Issue #26, on the shared anonymous/demo
identity (`"demo"`, `src.database.DEFAULT_USER_ID`). The app must never
become inaccessible just because auth code exists — this was Issue #22's
original design goal and still holds.

**2. Auth configured** (a real deployment with Google OAuth secrets set).
`require_login()` actively gates every page: an unauthenticated visitor sees
the dedicated sign-in screen (rendered in place, matching the FinTech UI,
with a bilingual "Continue with Google ・ 使用 Google 繼續登入" CTA and a
privacy note) instead of the page's real content. Streamlit's own
auto-generated page-nav list and this app's custom sidebar (language
switcher, nav links, quick settings) are both hidden until sign-in
completes.

## What a deployer needs to do to turn on real Google sign-in

Streamlit's native auth requires **no code changes** in this repo — only a
`[auth]` section in `.streamlit/secrets.toml` (or the equivalent secrets
mechanism on your hosting platform, e.g. Streamlit Community Cloud's
"Secrets" UI). A commented template is at `.streamlit/secrets.toml.example`.
**Never commit real secrets to this repository** —
`.streamlit/secrets.toml` is already git-ignored.

### 1. Google Cloud Console setup

1. Create (or reuse) a project at https://console.cloud.google.com/.
2. **APIs & Services → OAuth consent screen**: configure the app, and under
   **Scopes** add only `openid`, `.../auth/userinfo.email`, and
   `.../auth/userinfo.profile` — this app never requests Gmail, Drive,
   Contacts, or any other Google data/scope.
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   (type: Web application).
4. Set **Authorized redirect URIs** to your deployment's callback URL, e.g.
   `https://ai-etf-portfolio.streamlit.app/oauth2callback` for production
   (or `http://localhost:8501/oauth2callback` for local testing).
5. Copy the generated **Client ID** and **Client Secret**.

### 2. Streamlit Secrets

Add to `.streamlit/secrets.toml` (not committed) or your platform's Secrets
UI:

```toml
[auth]
redirect_uri = "https://ai-etf-portfolio.streamlit.app/oauth2callback"
cookie_secret = "<a long, random, locally-generated string>"
client_id = "<from Google Cloud Console>"
client_secret = "<from Google Cloud Console>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

`cookie_secret` should be generated locally (e.g. `python -c "import
secrets; print(secrets.token_hex(32))"`) and kept only in secrets storage —
it's used to sign Streamlit's own auth session cookie, not a Google value.

### 3. Redeploy

`is_auth_configured()` now returns `True`, `require_login()` becomes active
on every page, and the dedicated sign-in screen appears for anyone not yet
signed in.

### 4. No database migration required

`UserHolding`/`WatchlistItem` rows are already keyed by `user_id`, and
existing anonymous/demo rows (`user_id = "demo"`) keep working unchanged —
they simply stop being reachable by new visitors once sign-in is required,
same as before.

## Manual test checklist

- **Not configured** (no secrets): every page loads directly with no
  sign-in prompt, sidebar/nav all present — unchanged from pre-Issue #26
  behavior.
- **Configured, signed out**: visiting `app.py` OR any `pages/*.py` URL
  directly shows the sign-in screen, not the page content; the browser's
  URL bar can be used to try to "jump" straight to a child page and it is
  still gated the same way.
- **Configured, sign-in flow**: clicking "Continue with Google" redirects
  to Google, completes consent, and returns to the app fully signed in with
  no redirect loop and no extra rerun/flash of stale content.
- **Configured, signed in**: sidebar shows name + email + one Sign out
  button (not duplicated on any page); clicking Sign out returns to the
  sign-in screen; direct URLs to child pages during an active session load
  normally (no re-prompt).
- **Mobile viewport**: the sign-in card and CTA button reflow without
  horizontal overflow.
- **No DuplicateWidgetID / no infinite rerun** on any of the above, in both
  UI languages (the language toggle only affects the app's own text; the
  sign-in screen text is shown in both languages simultaneously since the
  language selector itself is hidden pre-login).

## Follow-up work (not done this round)

- Migrate off SQLite/file-based storage to Neon (tracked separately, out of
  scope for Issue #26 by explicit instruction).
- Scope Portfolio History (saved optimizer results) to the signed-in user
  the same way Current Holdings/Watchlist already are (was already a
  follow-up before this round).
