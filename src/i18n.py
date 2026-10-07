"""
Internationalisation (i18n) Module
Centralised bilingual (Traditional Chinese / English) translation store for the
AI ETF Portfolio Optimizer. All user-facing strings across app.py, pages/, and
src/ (charts, ui, ai_advisor) are looked up through t() so no page builds its
own translation dictionary.

Usage:
    from src.i18n import t, get_language, set_language, language_selector

    st.write(t("app_title"))
    st.write(t("selected_etfs_count", count=4))
"""

import streamlit as st

from src.translations import TRANSLATIONS

LANGUAGE_KEY = "language"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("zh-TW", "en")

LANGUAGE_LABELS = {
    "zh-TW": "繁體中文",
    "en": "English",
}


# ============================================================================
# Translation Store
# ============================================================================
# ============================================================================
# Core API
# ============================================================================
def _normalize_language_code(value):
    """Map URL/browser language variants to this app's supported codes."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    code = str(value).strip().lower().replace("_", "-")
    if code == "en" or code.startswith("en-"):
        return "en"
    if code == "zh" or code.startswith("zh-"):
        return "zh-TW"
    return None


def _language_from_accept_language(header_value):
    """Use English for non-Chinese browser locales; zh-TW for Chinese."""
    header = str(header_value or "").strip()
    if not header:
        return None
    for token in header.split(","):
        code = token.split(";", 1)[0].strip()
        normalized = _normalize_language_code(code)
        if normalized:
            return normalized
    # The browser declared a locale but none of its preferred locales are
    # Chinese/English; English is the safest public-portfolio fallback.
    return "en"


def _url_language():
    try:
        return _normalize_language_code(st.query_params.get("lang"))
    except Exception:
        return None


def _browser_language():
    try:
        context = getattr(st, "context", None)
        headers = getattr(context, "headers", None) if context is not None else None
        if headers:
            value = headers.get("Accept-Language") or headers.get("accept-language")
            return _language_from_accept_language(value)
    except Exception:
        pass
    return None


def get_language() -> str:
    """Resolve language with URL > session > project default priority.

    Fresh public-portfolio sessions always open in English so admissions
    reviewers see a deterministic English landing page regardless of browser
    locale. An explicit URL language or the sidebar selector can still switch
    the session to Traditional Chinese.
    """
    url_lang = _url_language()
    if url_lang:
        st.session_state[LANGUAGE_KEY] = url_lang
        return url_lang

    if LANGUAGE_KEY not in st.session_state:
        st.session_state[LANGUAGE_KEY] = DEFAULT_LANGUAGE

    lang = st.session_state[LANGUAGE_KEY]
    return lang if lang in TRANSLATIONS else DEFAULT_LANGUAGE


def set_language(language: str) -> None:
    """Persist the active language into session_state. No-op for unknown codes."""
    if language in TRANSLATIONS:
        st.session_state[LANGUAGE_KEY] = language


def t(key: str, **kwargs) -> str:
    """Translate `key` into the current session language.

    Never raises: falls back to English, then to the raw key, so a missing
    translation degrades gracefully instead of crashing the page. Supports
    str.format() style interpolation via kwargs, e.g. t("selected_etfs_count", count=4).
    """
    lang = get_language()
    text = TRANSLATIONS.get(lang, {}).get(key)
    if text is None:
        text = TRANSLATIONS.get("en", {}).get(key)
    if text is None:
        text = key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


# ============================================================================
# Option-Value Display Mappings
#
# Several st.selectbox widgets return raw English strings that are also used
# internally for logic comparisons (src/portfolio_optimizer.py, src/machine_learning.py),
# saved to the database, or embedded in generated report text (src/ai_advisor.py).
# Those underlying values MUST stay in English so calculation/storage logic is
# untouched. These mappings translate them for DISPLAY ONLY, wherever the raw
# value would otherwise be shown to the user (KPI labels, chart titles, saved
# portfolio history, AI-generated analysis text). Use with st.selectbox's
# format_func=..., or call the tX() helper directly when rendering saved values.
# ============================================================================
OPTIMIZATION_METHOD_KEYS = {
    "Equal Weight": "opt_method_equal_weight",
    "Maximum Sharpe Ratio": "opt_method_max_sharpe",
    "Minimum Volatility": "opt_method_min_volatility",
    "Target Return": "opt_method_target_return",
    "Risk Parity": "opt_method_risk_parity",
}

MODEL_TYPE_KEYS = {
    "Random Forest": "ml_model_random_forest",
    "Logistic Regression": "ml_model_logistic_regression",
}

INVESTMENT_OBJECTIVE_KEYS = {
    "Long-term Growth": "ai_objective_long_term_growth",
    "Income & Dividends": "ai_objective_income_dividends",
    "Capital Preservation": "ai_objective_capital_preservation",
    "Balanced Growth & Income": "ai_objective_balanced",
    "Aggressive Growth": "ai_objective_aggressive_growth",
    "Retirement Planning": "ai_objective_retirement",
}

RISK_LEVEL_KEYS = {
    "Conservative": "ai_risk_conservative",
    "Moderate": "ai_risk_moderate",
    "Aggressive": "ai_risk_aggressive",
}

MARKET_SCENARIO_KEYS = {
    "Bull Market": "sim_scenario_bull",
    "Base Case": "sim_scenario_base",
    "Bear Market": "sim_scenario_bear",
    "Sideways Market": "sim_scenario_sideways",
}

# src/etf_database.py's ETFRecord.country / .sector are stored as plain
# English strings (the canonical data); these translate them for display
# only, in the same spirit as OPTIMIZATION_METHOD_KEYS etc. above.
COUNTRY_KEYS = {
    "United States": "mi_country_united_states",
    "Taiwan": "mi_country_taiwan",
    "United Kingdom": "mi_country_united_kingdom",
}

SECTOR_KEYS = {
    "Technology": "mi_sector_technology",
    "Broad Market": "mi_sector_broad_market",
    "Gold": "mi_sector_gold",
    "Bond": "mi_sector_bond",
    "Dividend": "mi_sector_dividend",
    "Financials": "mi_sector_financials",
    "Healthcare": "mi_sector_healthcare",
    "Real Estate": "mi_sector_real_estate",
    "Small Cap": "mi_sector_small_cap",
}

# src/etf_signals.py's compute_quant_signals() returns canonical English
# values ("Overweight"/"Neutral"/"Underweight", "Bullish"/"Neutral"/
# "Bearish") regardless of language -- these translate them for display
# only, same pattern as COUNTRY_KEYS etc. above.
PORTFOLIO_VIEW_KEYS = {
    "Overweight": "etf_portfolio_view_overweight",
    "Neutral": "etf_portfolio_view_neutral",
    "Underweight": "etf_portfolio_view_underweight",
}

TREND_SIGNAL_KEYS = {
    "Bullish": "etf_trend_bullish",
    "Neutral": "etf_trend_neutral",
    "Bearish": "etf_trend_bearish",
}

# src/etf_signals.py's risk_level_from_vol() returns the canonical
# ("Low"/"Medium"/"High") per-ETF classification; the Investment Verdict
# panel's own portfolio-wide bucketing additionally uses "Medium High" and
# "Very High" -- both share this one mapping so the same English value
# always renders the same Chinese term everywhere on ETF Analysis (Issue
# #39: before this, "中風險" and "Medium" for the same risk bucket could
# appear in different tables on the same page).
ETF_RISK_LEVEL_KEYS = {
    "Low": "etf_risk_low",
    "Medium": "etf_risk_medium",
    "Medium High": "etf_risk_medium_high",
    "High": "etf_risk_high",
    "Very High": "etf_risk_very_high",
}

# src/etf_signals.py's expected_return_label_from_ann_ret() canonical
# per-ETF Expected Return label (distinct scale from the Investment
# Verdict's Low/Medium/High/Very High -- see ETF_VERDICT_RETURN_KEYS).
ETF_RETURN_LABEL_KEYS = {
    "Poor": "etf_return_poor",
    "Fair": "etf_return_fair",
    "Good": "etf_return_good",
    "Very Good": "etf_return_very_good",
    "Excellent": "etf_return_excellent",
}

# Investment Verdict's portfolio-wide Expected Return tier (Low/Medium/
# High/Very High) -- a different scale from ETF_RETURN_LABEL_KEYS' per-ETF
# Poor..Excellent labels, so it gets its own mapping/helper even though
# both are called "Expected Return".
ETF_VERDICT_RETURN_KEYS = {
    "Low": "etf_verdict_return_low",
    "Medium": "etf_verdict_return_medium",
    "High": "etf_verdict_return_high",
    "Very High": "etf_verdict_return_very_high",
}

ETF_HORIZON_KEYS = {
    "Short-to-Medium Term": "etf_horizon_short_medium",
    "Medium-to-Long Term": "etf_horizon_medium_long",
    "Long Term": "etf_horizon_long",
}

ETF_SUITABLE_INVESTOR_KEYS = {
    "Growth Investors": "etf_suitable_growth",
    "Conservative Investors": "etf_suitable_conservative",
    "Balanced Investors": "etf_suitable_balanced",
}

# Compare Mode's pairwise metric names (Return/Risk/Sharpe/Volatility/
# Drawdown/Momentum) -- previously rendered as raw English regardless of
# language.
ETF_COMPARE_METRIC_KEYS = {
    "Return": "etf_compare_metric_return",
    "Risk": "etf_compare_metric_risk",
    "Sharpe": "etf_compare_metric_sharpe",
    "Volatility": "etf_compare_metric_volatility",
    "Drawdown": "etf_compare_metric_drawdown",
    "Momentum": "etf_compare_metric_momentum",
}

# src/goal_planner.py's canonical (lowercase) target_mode / risk_tolerance /
# status values -- translated for display only, same pattern as above.
GOAL_TARGET_MODE_KEYS = {
    "total_value": "gp_target_mode_total_value",
    "monthly_income": "gp_target_mode_monthly_income",
    "annual_income": "gp_target_mode_annual_income",
}

GOAL_RISK_KEYS = {
    "conservative": "gp_risk_conservative",
    "balanced": "gp_risk_balanced",
    "aggressive": "gp_risk_aggressive",
}

GOAL_STATUS_KEYS = {
    "on_track": "gp_status_on_track",
    "below_target": "gp_status_below_target",
    "above_target": "gp_status_above_target",
}

# Saved-portfolio DISPLAY-ONLY strategy values that are no longer one of the
# five methods the optimizer dropdown offers (OPTIMIZATION_METHOD_KEYS) --
# e.g. legacy DB rows saved before a method was retired. Deliberately kept
# SEPARATE from OPTIMIZATION_METHOD_KEYS (Issue #43 item K): a legacy value
# must never be re-added to the optimizer's own dropdown, only translated
# for display wherever a saved portfolio's strategy is shown.
SAVED_STRATEGY_DISPLAY_KEYS = {
    "Custom Allocation": "saved_strategy_custom_allocation",
}

# src/portfolio_optimizer.py's canonical, saved-metadata estimator method
# strings (pages/2_Portfolio_Optimizer.py's _experiment_metadata) -- display
# translations only; the stored value itself must stay exactly as saved
# (Issue #43 item J).
RETURN_ESTIMATOR_KEYS = {
    # "Historical CAGR" is INACCURATE for this app's actual estimator (see
    # run_optimization() -- it uses the arithmetic mean of daily returns
    # annualized by x252, not a CAGR) -- kept only so portfolios saved
    # before Issue #45 still display a translated (if inaccurate) label
    # instead of falling back to the raw English key string.
    "Historical CAGR (annualized_return)": "estimator_return_historical_cagr",
    "Arithmetic Mean Daily Return (annualized x252)": "estimator_return_arithmetic_mean",
}

COVARIANCE_ESTIMATOR_KEYS = {
    "Sample covariance (historical, annualized)": "estimator_covariance_sample",
}


def _translate_option(value: str, mapping: dict) -> str:
    """Translate a raw option value for display; returns the value unchanged
    if it has no mapping (e.g. 'Custom', or a value not in the table)."""
    key = mapping.get(value)
    return t(key) if key else value


def t_opt_method(value: str) -> str:
    return _translate_option(value, OPTIMIZATION_METHOD_KEYS)


def t_model_type(value: str) -> str:
    return _translate_option(value, MODEL_TYPE_KEYS)


def t_investment_objective(value: str) -> str:
    return _translate_option(value, INVESTMENT_OBJECTIVE_KEYS)


def t_risk_level(value: str) -> str:
    return _translate_option(value, RISK_LEVEL_KEYS)


def t_market_scenario(value: str) -> str:
    return _translate_option(value, MARKET_SCENARIO_KEYS)


def t_country(value: str) -> str:
    return _translate_option(value, COUNTRY_KEYS)


def t_sector(value: str) -> str:
    return _translate_option(value, SECTOR_KEYS)


def t_portfolio_view(value: str) -> str:
    return _translate_option(value, PORTFOLIO_VIEW_KEYS)


def t_trend_signal(value: str) -> str:
    return _translate_option(value, TREND_SIGNAL_KEYS)


def t_etf_risk_level(value: str) -> str:
    return _translate_option(value, ETF_RISK_LEVEL_KEYS)


def t_etf_return_label(value: str) -> str:
    return _translate_option(value, ETF_RETURN_LABEL_KEYS)


def t_etf_verdict_return(value: str) -> str:
    return _translate_option(value, ETF_VERDICT_RETURN_KEYS)


def t_investment_horizon(value: str) -> str:
    return _translate_option(value, ETF_HORIZON_KEYS)


def t_suitable_investor(value: str) -> str:
    return _translate_option(value, ETF_SUITABLE_INVESTOR_KEYS)


def t_compare_metric(value: str) -> str:
    return _translate_option(value, ETF_COMPARE_METRIC_KEYS)


def t_goal_target_mode(value: str) -> str:
    return _translate_option(value, GOAL_TARGET_MODE_KEYS)


def t_goal_risk(value: str) -> str:
    return _translate_option(value, GOAL_RISK_KEYS)


def t_goal_status(value: str) -> str:
    return _translate_option(value, GOAL_STATUS_KEYS)


def t_saved_strategy(value: str) -> str:
    """Display-only translation for a SAVED portfolio's strategy value
    (Issue #43 item K), which may be one of the five methods the optimizer
    dropdown currently offers, OR a legacy value no longer offered there
    (e.g. 'Custom Allocation' from an older DB row). The five current
    methods still go through t_opt_method()/OPTIMIZATION_METHOD_KEYS
    unchanged; legacy values are translated via the separate
    SAVED_STRATEGY_DISPLAY_KEYS mapping so nothing here ever needs to be
    (or should be) added back to the optimizer's own dropdown. An
    unrecognized value falls back to the raw saved string (or "--") rather
    than crashing.
    """
    if not value:
        return "—"
    if value in OPTIMIZATION_METHOD_KEYS:
        return t_opt_method(value)
    return _translate_option(value, SAVED_STRATEGY_DISPLAY_KEYS)


def t_return_estimator(value: str) -> str:
    return _translate_option(value, RETURN_ESTIMATOR_KEYS) if value else "—"


def t_covariance_estimator(value: str) -> str:
    return _translate_option(value, COVARIANCE_ESTIMATOR_KEYS) if value else "—"


def t_market_display(markets) -> str:
    """Display-only market/region summary for a saved portfolio (Issue #43
    item L): given a list of raw country strings (e.g. from a portfolio's
    saved metadata["markets"], or inferred from its holdings' known
    countries), returns one already-translated string -- the single
    translated market when there is exactly one, a localized "Multi-market
    (...)" summary when there is more than one, or a localized "Not
    recorded" (never a bare "--") when the list is empty/unknown. Never
    rewrites or infers anything about the underlying saved record itself --
    purely a display-layer summary of whatever country list the caller
    already determined.
    """
    unique = sorted({m for m in (markets or []) if m})
    if not unique:
        return t("saved_market_not_recorded")
    if len(unique) > 1:
        return t("saved_market_multi", markets="、".join(t_country(m) for m in unique) if get_language() == "zh-TW"
                  else ", ".join(t_country(m) for m in unique))
    return t_country(unique[0])


def _on_language_selector_change() -> None:
    selected_code = st.session_state.get("_language_selector_widget")
    if selected_code in SUPPORTED_LANGUAGES:
        set_language(selected_code)
        try:
            st.query_params["lang"] = selected_code
        except Exception:
            pass


def language_selector() -> None:
    """Render the language selector and keep explicit choices URL-shareable."""
    current = get_language()
    codes = list(LANGUAGE_LABELS.keys())
    try:
        idx = codes.index(current)
    except ValueError:
        idx = 0

    # Migrate the old widget state (which stored display labels) and honor an
    # explicit URL override even inside an already-open Streamlit session.
    prior = st.session_state.get("_language_selector_widget")
    if prior not in codes or (_url_language() and prior != current):
        st.session_state["_language_selector_widget"] = current

    st.markdown(
        f'<div class="sidebar-nav-label">{t("language_label")}</div>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        t("language_label"),
        codes,
        index=idx,
        format_func=lambda code: LANGUAGE_LABELS[code],
        label_visibility="collapsed",
        key="_language_selector_widget",
        on_change=_on_language_selector_change,
    )
