# Authentication (Issue #22 section C, extended by Issue #26)

The app uses **Streamlit's own native authentication** (`st.login()` /
`st.logout()` / `st.user`), which is an OpenID Connect (OIDC) client built
into Streamlit itself (Streamlit >= 1.42, with the `Authlib` package
installed — see `requirements.txt`). No third-party auth library and no
hard-coded credentials are added by this app — see `src/auth.py`.

## Current status in this repository

- `src/auth.py` provides `is_auth_configured()`, `is_authenticated()`,
  `get_current_user_id()`, `get_current_user_display_name()`,
  `require_login()`, and `render_account_section()`.
- `get_current_user_id()` returns a stable, provider-issued identifier
  (`user:<sub>`) when a visitor is signed in — preferring the OIDC `sub`
  claim (guaranteed stable per provider) over `email` (which can change) —
  and the shared constant `"demo"` (`src.database.DEFAULT_USER_ID`)
  otherwise, including on any deployment that has never configured auth at
  all.
- `require_login()` is a **central page guard**, called once near the top of
  `app.py` and every page in `pages/`. It only ever does something once a
  deployer has configured real `[auth]` secrets (`is_auth_configured()`):
  a signed-out visitor on a configured deployment is redirected to the
  dedicated login page, `pages/9_Login.py`. On a deployment with no `[auth]`
  secrets configured — including a fresh checkout of this repo — it is a
  strict no-op, so the app stays a fully open public demo exactly as before.
- `pages/9_Login.py` is a dedicated Google sign-in page: it shows a
  "Sign in with Google" button wired to `st.login()`, a friendly message if
  the visitor is already signed in (with a way back into the app), and a
  plain explanatory message (never a broken button) if `[auth]` isn't
  configured for the deployment.
- `render_account_section()` renders a compact "Account" widget (signed-in
  name + Sign out button) in the sidebar. It's called once, inside
  `src.ui.render_sidebar_nav()`, so every page picks it up automatically.
  It renders nothing at all unless the visitor is actually signed in.
- My Portfolio's **Current Holdings** and **Watchlist** tabs are wired to
  `get_current_user_id()`, so they are user-scoped the moment auth is
  configured, with zero further code changes needed.
- Saved-portfolio history (`Portfolio` table) gained a nullable `user_id`
  column (backward-compatible additive migration) but is **not yet filtered
  by user** in the UI — it remains the shared demo list this round. Wiring
  it up is a follow-up, not a blocker, since the schema is already ready.

## What a deployer needs to do to turn on real Google sign-in

Streamlit's native auth requires **no code changes** in this repo — only a
`[auth]` section in `.streamlit/secrets.toml` (or the equivalent secrets
mechanism on your hosting platform, e.g. Streamlit Community Cloud's "Secrets"
UI). **Never commit real secrets to this repository** — `.streamlit/secrets.toml`
is already git-ignored.

1. In [Google Cloud Console](https://console.cloud.google.com/), create an
   OAuth 2.0 Client ID (APIs & Services -> Credentials -> Create Credentials
   -> OAuth client ID -> Web application). Set its **Authorized redirect
   URI** to `https://<your-deployed-app-url>/oauth2callback`.
2. Add to `.streamlit/secrets.toml` (not committed):

   ```toml
   [auth]
   redirect_uri = "https://<your-deployed-app-url>/oauth2callback"
   cookie_secret = "<a long, random, locally-generated string>"
   client_id = "<from Google Cloud Console>"
   client_secret = "<from Google Cloud Console>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

   For local development, use `redirect_uri = "http://localhost:8501/oauth2callback"`
   and add that same URL to the OAuth client's Authorized redirect URIs in
   Google Cloud Console.
3. Redeploy. `src/auth.py`'s `is_auth_configured()` will now return `True`:
   - Every page's `require_login()` guard now redirects signed-out visitors
     to `pages/9_Login.py` to sign in with Google.
   - The sidebar's compact Account section (`render_account_section()`)
     shows the signed-in visitor's name and a Sign out button.
4. No database migration step is required — `UserHolding`/`WatchlistItem`
   rows are already keyed by `user_id`, and existing anonymous/demo rows
   (`user_id = "demo"`) keep working unchanged for anyone who doesn't sign in.

## Staying a public demo (the default)

If you don't add an `[auth]` section to secrets, the app behaves exactly as
it always has: every page is open to anyone, `require_login()` never
redirects, and Current Holdings/Watchlist/Daily Brief all operate on the
shared `"demo"` namespace. This is the default for a fresh checkout and for
CI (no secrets are ever committed), which is why the test suite never needs
real OAuth credentials to pass.

## Follow-up work (not done this round)

- Scope Portfolio History (saved optimizer results) to the signed-in user,
  the same way Current Holdings/Watchlist already are.
- Migrate the demo SQLite database (`database/portfolio.db`) to a hosted
  Postgres/Neon database for multi-user production use — out of scope for
  this round (Issue #26 explicitly does not touch data storage).
