"""
Page 9: Login (Issue #26).

Dedicated Google sign-in page. This is the page require_login() (see
src/auth.py) redirects any visitor to once a deployment has real [auth]
secrets configured and they haven't signed in yet. It is intentionally left
out of NAV_ITEMS (src/ui.py) -- it's a utility destination reached via
redirect or a direct link, not a primary section of the app.

Never a dead end, matching the rest of src/auth.py's defensive design:
- No [auth] secrets configured at all -> plain explanatory message instead
  of a login button that would just fail.
- Already signed in (e.g. visited directly) -> a welcome-back message and a
  way back into the app, instead of showing a redundant sign-in form.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

from src.utils import load_css, page_header
from src.ui import render_sidebar_nav, render_sidebar_footer
from src.i18n import t
from src.auth import is_auth_configured, is_authenticated, get_current_user_display_name

st.set_page_config(
    page_title="Sign in | AI ETF Portfolio Optimizer",
    page_icon="\U0001F510",
    layout="wide",
)

load_css()

with st.sidebar:
    render_sidebar_nav()
    render_sidebar_footer()

page_header(t("login_title"), t("login_subtitle"))

if not is_auth_configured():
    st.info(t("login_not_configured"))
elif is_authenticated():
    st.success(t("login_already_signed_in", name=get_current_user_display_name()))
    if st.button(t("login_continue_button"), type="primary"):
        st.switch_page("app.py")
else:
    st.write(t("login_prompt"))
    if st.button(t("login_google_button"), type="primary", key="login_page_google_btn"):
        st.login()
