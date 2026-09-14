# Authentication (Issue #22 section C)

The app uses **Streamlit's own native authentication** (`st.login()` /
`st.logout()` / `st.user`), which is an OpenID Connect (OIDC) client built
into Streamlit itself (Streamlit >= 1.42). No third-party auth library and
no hard-coded credentials are added by this app — see `src/auth.py`.

## Current status in this repository

- `src/auth.py` provides `is_auth_configured()`, `is_authenticated()`,
  `get_current_user_id()`, `get_current_user_display_name()`, and
  `render_auth_status()`.
- `get_current_user_id()` returns a stable, provider-issued identifier
  (`user:<sub>`) when a visitor is signed in, and the shared constant
  `"demo"` (`src.database.DEFAULT_USER_ID`) otherwise — including on any
  deployment that has never configured auth at all.
- My Portfolio's **Current Holdings** and **Watchlist** tabs are already
  wired to `get_current_user_id()`, so they are user-scoped the moment auth
  is configured, with zero code changes needed.
- **Public demo access is never blocked.** Every page renders normally for
  an anonymous/demo visitor; only the *storage namespace* for holdings/
  watchlist changes based on sign-in state.
- Saved-portfolio history (`Portfolio` table) gained a nullable `user_id`
  column (backward-compatible additive migration) but is **not yet filtered
  by user** in the UI — it remains the shared demo list this round. Wiring
  it up is a follow-up, not a blocker, since the schema is already ready.

## What a deployer needs to do to turn on real sign-in

Streamlit's native auth requires **no code changes** in this repo — only a
`[auth]` section in `.streamlit/secrets.toml` (or the equivalent secrets
mechanism on your hosting platform, e.g. Streamlit Community Cloud's "Secrets"
UI). **Never commit real secrets to this repository** — `.streamlit/secrets.toml`
is already git-ignored.

1. Register an OAuth/OIDC application with any OIDC-compliant identity
   provider (Google, Microsoft Entra ID, Auth0, Okta, GitHub via an OIDC
   proxy, etc.). Set its redirect/callback URL to
   `https://<your-deployed-app-url>/oauth2callback`.
2. Add to `.streamlit/secrets.toml` (not committed):

   ```toml
   [auth]
   redirect_uri = "https://<your-deployed-app-url>/oauth2callback"
   cookie_secret = "<a long, random, locally-generated string>"
   client_id = "<from your OIDC provider>"
   client_secret = "<from your OIDC provider>"
   server_metadata_url = "<your provider's .well-known/openid-configuration URL>"
   ```

3. Redeploy. `src/auth.py`'s `is_auth_configured()` will now return `True`,
   and `render_auth_status()` (currently rendered in My Portfolio's sidebar)
   will show a real "Sign in" button wired to `st.login()`.
4. No database migration step is required — `UserHolding`/`WatchlistItem`
   rows are already keyed by `user_id`, and existing anonymous/demo rows
   (`user_id = "demo"`) keep working unchanged for anyone who doesn't sign in.

## Follow-up work (not done this round)

- Scope Portfolio History (saved optimizer results) to the signed-in user,
  the same way Current Holdings/Watchlist already are.
- A global sign-in control in the main sidebar (currently only rendered on
  the My Portfolio page, since that's the only page with user-scoped data
  today).
