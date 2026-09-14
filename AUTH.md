# Authentication (Issue #22 section C)

The app uses **Streamlit's own native authentication** (`st.login()` /
`st.logout()` / `st.user`), which is an OpenID Connect (OIDC) client built
into Streamlit itself (Streamlit >= 1.42). No third-party auth library and
no hard-coded credentials are added by this app — see `src/auth.py`.

## Current status in this repository

- `src/auth.py` provides `is_auth_configured()`, `is_authenticated()`,
  `get_current_user_id()`, `get_current_user_display_name()`,
  `require_login()`, and `render_auth_status()`.
- `get_current_user_id()` returns a stable, provider-issued identifier
  (`user:<sub>`) when a visitor is signed in, and the shared constant
  `"demo"` (`src.database.DEFAULT_USER_ID`) otherwise — including on any
  deployment that has never configured auth at all.
- My Portfolio's **Current Holdings** and **Watchlist** tabs are already
  wired to `get_current_user_id()`, so they are user-scoped the moment auth
  is configured, with zero code changes needed.
- Before `[auth]` secrets are configured, the app remains available in demo/development mode.
  Once `[auth]` is configured, `require_login()` gates every app page behind the
  Google sign-in screen. Direct links to individual Streamlit pages are gated too.
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

1. In Google Cloud Console, create an OAuth client for a Web application.
   Set its authorized redirect URI to
   `https://<your-deployed-app-url>/oauth2callback`.
2. Add to `.streamlit/secrets.toml` (not committed):

   ```toml
   [auth]
   redirect_uri = "https://<your-deployed-app-url>/oauth2callback"
   cookie_secret = "<a long, random, locally-generated string>"
   client_id = "<from your OIDC provider>"
   client_secret = "<from your OIDC provider>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

3. Redeploy. Visitors will see the dedicated `Continue with Google` screen.
   After successful sign-in, Streamlit exposes the Google identity through
   `st.user`, and the shared sidebar shows the signed-in identity plus a logout button.
4. The app requests the standard OIDC identity scopes (`openid profile email`).
   It does not request Gmail mailbox, Contacts, or Google Drive access.

## Follow-up work

- Scope Portfolio History (saved optimizer results) to the signed-in user,
  the same way Current Holdings/Watchlist already are.
- Replace the current SQLite persistence layer with the deployed PostgreSQL/Neon
  database so user data survives Streamlit container restarts.
