"""
Page 7: My Portfolio (Issue #22 -- absorbs the former standalone Portfolio
History page into one tabbed workspace: Goal Planner / Current Holdings /
Watchlist / Portfolio History / Daily Brief). File name and route are kept
as pages/7_Portfolio_History.py so every existing test/reference to this
page path (and its saved-portfolio backward-compatibility guarantees) stays
intact -- only the on-page navigation label ("My Portfolio") and content
changed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import (
    load_all_portfolios, delete_portfolio, init_database, APP_VERSION,
)
from src.etf_database import get_country, get_etf, to_yahoo_symbol
from src.data_loader import _download_single_ticker
from src.financial_metrics import portfolio_diagnosis, ACTIVE_POSITION_TOLERANCE
from src.charts import allocation_donut_chart, apply_dark_theme, CHART_COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, empty_state, error_state, status_card,
    chart_caption, ai_interpret_button, info_badge,
)
from src.i18n import (
    t, t_saved_strategy, t_country, t_goal_target_mode, t_goal_risk, t_goal_status,
    t_return_estimator, t_covariance_estimator, t_market_display,
    get_language,
)
from src.goal_planner import (
    build_goal_plan, VALID_TARGET_MODES, VALID_MARKET_PREFERENCES,
    VALID_RISK_TOLERANCES, VALID_BASE_CURRENCIES,
)
from src.simulator import simulate_investment
from src.daily_brief import build_brief_context, generate_daily_brief
from src.news import fetch_market_news
from src.theme import COLORS

st.set_page_config(
    page_title="My Portfolio | AI ETF Portfolio Optimizer",
    page_icon="\U0001F4C1",
    layout="wide"
)

load_css()
init_database()

page_header(t("my_portfolio_title"), t("my_portfolio_subtitle"))

with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('my_portfolio_title')}")
    render_sidebar_footer()

# Public portfolio demo deliberately has no login wall. Visitor-entered
# Current Holdings and Watchlist data live only in this Streamlit session,
# so one visitor can never see or overwrite another visitor's data.
st.caption(t("mp_guest_mode_note"))
st.caption(t("mp_session_privacy_note"))

_SESSION_HOLDINGS_KEY = "_guest_session_holdings"
_SESSION_WATCHLIST_KEY = "_guest_session_watchlist"
_SESSION_HOLDING_SEQ = "_guest_holding_seq"
_SESSION_WATCHLIST_SEQ = "_guest_watchlist_seq"


def _session_holdings() -> list:
    if _SESSION_HOLDINGS_KEY not in st.session_state:
        st.session_state[_SESSION_HOLDINGS_KEY] = []
    return st.session_state[_SESSION_HOLDINGS_KEY]


def _session_watchlist() -> list:
    if _SESSION_WATCHLIST_KEY not in st.session_state:
        st.session_state[_SESSION_WATCHLIST_KEY] = []
    return st.session_state[_SESSION_WATCHLIST_KEY]


def _next_session_id(counter_key: str) -> int:
    st.session_state[counter_key] = int(st.session_state.get(counter_key, 0)) + 1
    return st.session_state[counter_key]


def _add_session_holding(ticker: str, quantity: float, average_cost: float,
                         currency: str, purchase_date: str = None) -> None:
    rows = list(_session_holdings())
    rows.append({
        "id": _next_session_id(_SESSION_HOLDING_SEQ),
        "ticker": ticker,
        "quantity": float(quantity),
        "average_cost": float(average_cost),
        "currency": currency,
        "purchase_date": purchase_date,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state[_SESSION_HOLDINGS_KEY] = rows


def _delete_session_holding(holding_id: int) -> None:
    st.session_state[_SESSION_HOLDINGS_KEY] = [
        row for row in _session_holdings() if row["id"] != holding_id
    ]


def _add_session_watchlist(ticker: str) -> None:
    rows = list(_session_watchlist())
    rows.append({
        "id": _next_session_id(_SESSION_WATCHLIST_SEQ),
        "ticker": ticker,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state[_SESSION_WATCHLIST_KEY] = rows


def _delete_session_watchlist(item_id: int) -> None:
    st.session_state[_SESSION_WATCHLIST_KEY] = [
        row for row in _session_watchlist() if row["id"] != item_id
    ]


def _holdings_weights_from_live_values(live_values: dict) -> dict:
    total = sum(float(v) for v in live_values.values() if float(v) > 0)
    if total <= 0:
        return {}
    return {ticker: float(value) / total for ticker, value in live_values.items() if float(value) > 0}


def _set_holdings_as_current_portfolio(holdings_rows: list, live_values: dict) -> dict:
    weights = _holdings_weights_from_live_values(live_values)
    if not weights:
        return {}
    countries = [get_country(tk) for tk in weights if get_country(tk)]
    unique_countries = list(dict.fromkeys(countries))
    market = unique_countries[0] if len(unique_countries) == 1 else "Mixed"
    diag = portfolio_diagnosis(weights)
    st.session_state["current_portfolio"] = {
        "portfolio_id": "guest-current-holdings",
        "strategy": "Current Holdings",
        "market": market,
        "tickers": list(weights),
        "weights": weights,
        "investment_amount": sum(live_values.values()),
        "investment_goal": None,
        "risk_tolerance": None,
        "investment_horizon": None,
        "expected_return": None,
        "volatility": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
        "largest_position": diag.get("largest_weight"),
        "effective_holdings": diag.get("effective_holdings"),
        "historical_start_date": None,
        "historical_end_date": None,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "guest_session_holdings",
    }
    return weights


@st.cache_data(show_spinner=False)
def _simulate_goal_scenario(initial_capital: float, monthly_contribution: float,
                            annual_contribution: float, years: int,
                            annual_return: float, annual_volatility: float,
                            target_amount: float, n_simulations: int = 5000) -> dict:
    result = simulate_investment(
        initial_investment=initial_capital,
        monthly_contribution=monthly_contribution,
        years=years,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        annual_contribution=annual_contribution,
        n_simulations=n_simulations,
        seed=42,
    )
    finals = np.asarray(result["all_final_values"], dtype=float)
    summary = result["summary"]
    return {
        "target_share": float(np.mean(finals >= float(target_amount))),
        "p10": float(np.percentile(finals, 10)),
        "median": float(np.percentile(finals, 50)),
        "p90": float(np.percentile(finals, 90)),
        "real_median": float(summary["real_median_final"]),
        "n_simulations": int(n_simulations),
    }


IMPACT_VARIANT = {"Positive": "green", "Negative": "red", "Neutral": "neutral"}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_latest_price(ticker: str):
    """Live-only price lookup for Current Holdings / Watchlist: returns a
    float or None, NEVER a fabricated fallback value. Deliberately uses
    src.data_loader's internal per-ticker primitive rather than the public
    download_etf_data() -- that function substitutes fully simulated sample
    data when EVERY requested ticker fails, which is safe for analytics
    pages (always clearly banner-disclosed there) but would be a silent
    fabricated price here, violating Issue #22 section D ("no fake price
    fallback"). Cached for 5 minutes so switching tabs/rerunning the page
    doesn't refetch on every rerun.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=10)
    series = _download_single_ticker(to_yahoo_symbol(ticker), str(start), str(end))
    if series is None or series.empty:
        return None
    return float(series.iloc[-1])


def _is_known_or_live_ticker(ticker: str) -> bool:
    """Ticker validation (Issue #22 section D: "do not silently accept
    invalid tickers"): accept anything already in the curated ETF universe
    outright; for anything else, only accept it if a live price can
    actually be fetched for it, proving it is a real, tradeable symbol.
    """
    if get_etf(ticker):
        return True
    return _fetch_latest_price(ticker) is not None


# ── Portfolio History helpers (unchanged from the former standalone page) ───
def _safe_pct(value, decimals: int = 2) -> str:
    """Format a fraction as a percentage, or "—" for a missing/corrupt value
    (e.g. a NULL numeric column on a record saved before a schema change or
    edited outside the app) instead of crashing the whole page on one bad row."""
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return "—"


def _safe_num(value, fmt: str = ",.0f") -> str:
    try:
        return f"{float(value):{fmt}}"
    except (TypeError, ValueError):
        return "—"


def _active_holdings(holdings: dict) -> dict:
    """Holdings above ACTIVE_POSITION_TOLERANCE (src.financial_metrics --
    the same constant/threshold portfolio_diagnosis()'s active_holdings
    count already uses elsewhere in the app, so "active" means the same
    thing on every page), sorted by weight desc. Never mutates or discards
    the underlying raw weights -- callers that need the full saved dict
    (e.g. "Set as Current Portfolio", the CSV export, or
    portfolio_diagnosis()) must keep reading `holdings` itself, not this
    filtered view."""
    return dict(sorted(
        ((tk, w) for tk, w in holdings.items() if w > ACTIVE_POSITION_TOLERANCE),
        key=lambda kv: kv[1], reverse=True,
    ))


def _infer_market_from_holdings(holdings: dict):
    """The saved-portfolio schema does not persist which market/region a
    portfolio was built from (src/database.py's Portfolio table has no
    `market` column). Rather than a schema migration, infer it from the
    saved tickers' known country at reload time -- the same ETF database
    already used everywhere else for this -- via a simple majority vote.
    Returns None (handled gracefully downstream) if no ticker is recognized.
    """
    countries = [get_country(tk) for tk in holdings if get_country(tk)]
    if not countries:
        return None
    return max(set(countries), key=countries.count)


def _market_display_for_portfolio(p: dict) -> str:
    """Meaningful, already-translated Market display for Experiment Details
    (Issue #43 item L) -- never a bare "--" when there is meaningful
    information to show: prefers the saved metadata["markets"] list (added
    by pages/2_Portfolio_Optimizer.py's _experiment_metadata) when present;
    for older records saved before that metadata existed, infers the set of
    known countries from the portfolio's own saved holdings instead. Either
    way, t_market_display() renders one market, a localized "Multi-market
    (...)" summary, or a localized "Not recorded" -- purely a display-layer
    summary, never a rewrite of the saved record itself.
    """
    meta = p.get("metadata") or {}
    if meta.get("markets"):
        return t_market_display(meta["markets"])
    holdings = p.get("holdings") or {}
    countries = [get_country(tk) for tk in holdings if get_country(tk)]
    return t_market_display(countries)


def _set_as_current_portfolio(p: dict) -> None:
    """Reload a saved portfolio into the ONE canonical
    st.session_state["current_portfolio"] object (same shape Portfolio
    Optimizer builds it -- see pages/2_Portfolio_Optimizer.py) so Investment
    Simulator / Risk Analytics / AI Advisor can consume it directly without
    the user having to rebuild it in the Optimizer. largest_position/
    effective_holdings are recomputed from the saved weights (portfolio_diagnosis,
    the same function the Optimizer itself uses) rather than fabricated,
    since the DB does not store them. Fields the DB schema has no column for
    (market, investment_goal, risk_tolerance, investment_horizon, max_drawdown,
    historical_start_date/end_date) are set to None/inferred rather than guessed;
    every downstream reader already treats these as optional via .get().
    """
    holdings = p["holdings"] or {}
    diag = portfolio_diagnosis(holdings) if holdings else None
    st.session_state.current_portfolio = {
        "portfolio_id": f"history-{p['id']}",
        "strategy": p["optimization_method"],
        "market": _infer_market_from_holdings(holdings),
        "tickers": list(holdings.keys()),
        "weights": dict(holdings),
        "investment_amount": p["investment_amount"],
        "investment_goal": None,
        "risk_tolerance": None,
        "investment_horizon": None,
        "expected_return": p["expected_return"],
        "volatility": p["expected_volatility"],
        "sharpe_ratio": p["sharpe_ratio"],
        "max_drawdown": None,
        "largest_position": (
            {"ticker": diag["largest_ticker"], "weight": diag["largest_weight"]} if diag else None
        ),
        "effective_holdings": diag["effective_holdings"] if diag else None,
        "historical_start_date": None,
        "historical_end_date": None,
        "generated_at": p["created_at"],
    }
    st.session_state["_hist_just_set_name"] = p["name"]


def _render_portfolio_history_tab():
    # ── Load Portfolios ───────────────────────────────────────────────────────────
    try:
        portfolios = load_all_portfolios(raise_on_error=True)
    except Exception as e:
        error_state(t("hist_load_error_title"), t("hist_load_error_desc", error=str(e)))
        return

    if not portfolios:
        empty_state(
            t("hist_no_portfolios_title"),
            t("hist_no_portfolios_desc"),
            icon="layers",
        )
        return

    _just_set_name = st.session_state.pop("_hist_just_set_name", None)
    if _just_set_name:
        st.success(t("hist_set_as_current_success", name=_just_set_name))

    # ── Summary Table ─────────────────────────────────────────────────────────────
    section_header(t("hist_saved_portfolios_count", count=len(portfolios)))

    summary_rows = []
    _has_corrupt_record = False
    for p in portfolios:
        holdings = p["holdings"] or {}
        try:
            holdings_str = ", ".join([f"{tk} ({w:.0%})" for tk, w in
                                       sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:5]])
        except (TypeError, ValueError):
            holdings_str = "—"
            _has_corrupt_record = True
        _is_synthetic_demo = bool((p.get("metadata") or {}).get("synthetic_demo"))
        _display_name = (p["name"] or "—") + (t("hist_demo_portfolio_name_suffix") if _is_synthetic_demo else "")
        summary_rows.append({
            "ID": p["id"],
            t("hist_col_name"): _display_name,
            t("hist_col_created"): p["created_at"],
            t("hist_col_method"): t_saved_strategy(p["optimization_method"]) if p["optimization_method"] else "—",
            t("hist_col_investment"): f"${_safe_num(p['investment_amount'])}",
            t("hist_col_exp_return"): _safe_pct(p["expected_return"]),
            t("hist_col_exp_volatility"): _safe_pct(p["expected_volatility"]),
            t("hist_col_sharpe"): _safe_num(p["sharpe_ratio"], ",.2f"),
            t("hist_col_holdings"): holdings_str,
        })
        if any(v is None for v in (p["investment_amount"], p["expected_return"],
                                    p["expected_volatility"], p["sharpe_ratio"])):
            _has_corrupt_record = True

    summary_df = pd.DataFrame(summary_rows)
    with chart_card(t("hist_summary_card")):
        if _has_corrupt_record:
            st.caption(t("hist_corrupt_record_notice"))
        st.dataframe(summary_df.set_index("ID"), use_container_width=True)
        chart_caption(t("hist_summary_table_caption"))

        # ── Download History ──────────────────────────────────────────────────────
        csv = summary_df.to_csv(index=True).encode("utf-8")
        st.download_button(t("btn_download_history_csv"), csv, "portfolio_history.csv", "text/csv")

    # ── View Portfolio Details ────────────────────────────────────────────────────
    section_header(t("hist_view_details_title"))

    portfolio_names = {p["id"]: f"[{p['id']}] {p['name']} ({p['created_at']})" for p in portfolios}
    selected_id = st.selectbox(t("hist_select_portfolio"), options=list(portfolio_names.keys()),
                                format_func=lambda x: portfolio_names[x])

    selected_portfolio = next((p for p in portfolios if p["id"] == selected_id), None)

    if selected_portfolio:
        col_left, col_right = st.columns([1, 1])

        _current = st.session_state.get("current_portfolio")
        _is_current = bool(_current and _current.get("portfolio_id") == f"history-{selected_portfolio['id']}")
        # An unmissable badge for the committed demo DB's curated example rows
        # (scripts/reset_demo_portfolio_history.py) -- their expected_return/
        # volatility/Sharpe are hand-authored illustrative figures, never a
        # live optimizer result, so they must never be displayable as if they
        # were (Issue #20 release-gate review, automated PR reviewer finding).
        _is_synthetic_demo = bool((selected_portfolio.get("metadata") or {}).get("synthetic_demo"))

        with col_left:
            with chart_card(selected_portfolio["name"], selected_portfolio["created_at"]):
                if _is_synthetic_demo:
                    st.warning(t("hist_demo_portfolio_badge"))
                if _is_current:
                    st.caption(f"✓ {t('hist_current_portfolio_badge')}")
                detail_data = {
                    t("metric_optimization_method"): (
                        t_saved_strategy(selected_portfolio["optimization_method"])
                        if selected_portfolio["optimization_method"] else "—"
                    ),
                    t("field_investment_amount_usd"): f"${_safe_num(selected_portfolio['investment_amount'], ',.2f')}",
                    t("metric_expected_annual_return"): _safe_pct(selected_portfolio["expected_return"]),
                    t("metric_expected_volatility"): _safe_pct(selected_portfolio["expected_volatility"]),
                    t("metric_sharpe_ratio"): _safe_num(selected_portfolio["sharpe_ratio"], ",.2f"),
                }
                for k, v in detail_data.items():
                    st.metric(k, v)

                if selected_portfolio["holdings"]:
                    st.button(
                        t("hist_set_as_current_btn"), key=f"hist_set_current_{selected_portfolio['id']}",
                        type="primary", use_container_width=True, help=t("hist_set_as_current_help"),
                        on_click=_set_as_current_portfolio, args=(selected_portfolio,),
                    )

                if selected_portfolio["notes"]:
                    st.markdown(f"**{t('hist_notes_label')}**: {selected_portfolio['notes']}")

                # Holdings table -- Active Holdings Only (Issue #20 section 9B):
                # near-zero weights are hidden from this display table, but the
                # raw dict (used by "Set as Current Portfolio", CSV export, and
                # the donut chart's underlying diagnosis) is untouched.
                if selected_portfolio["holdings"]:
                    _all_holdings = selected_portfolio["holdings"]
                    _display_holdings = _active_holdings(_all_holdings)
                    holdings_df = pd.DataFrame([
                        {t("hist_col_ticker"): tk,
                         t("hist_col_region"): t_country(get_country(tk)) if get_country(tk) else t("hist_region_unknown"),
                         t("hist_col_weight"): f"{w:.2%}",
                         t("hist_col_amount"): f"${w * (selected_portfolio['investment_amount'] or 0):,.2f}"}
                        for tk, w in _display_holdings.items()
                    ])
                    _hidden_count = len(_all_holdings) - len(_display_holdings)
                    _holdings_label = t("hist_holdings_label")
                    if _hidden_count:
                        _holdings_label = f"{_holdings_label} ({t('hist_active_holdings_only', count=_hidden_count)})"
                    st.markdown(f"**{_holdings_label}**")
                    st.dataframe(holdings_df.set_index(t("hist_col_ticker")), use_container_width=True)
                    chart_caption(t("hist_holdings_table_caption"))
                    if _hidden_count:
                        with st.expander(t("hist_show_all_holdings")):
                            _raw_df = pd.DataFrame([
                                {t("hist_col_ticker"): tk, t("hist_col_weight"): f"{w:.4%}"}
                                for tk, w in sorted(_all_holdings.items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(_raw_df.set_index(t("hist_col_ticker")), use_container_width=True)
                            chart_caption(t("hist_all_holdings_table_caption"))

                # Experiment metadata (Issue #20 section 9C) -- shown for every
                # portfolio; legacy saves (before this feature existed) simply
                # have an empty dict rather than fabricated values. The small
                # ⓘ info_badge() cue (Issue #43 item O) gives this the same
                # info/explanation visual semantics as Risk Analytics'
                # Methodology & Assumptions, distinct from a plain expander.
                st.markdown(info_badge(t("hist_experiment_details_badge_label")), unsafe_allow_html=True)
                with st.expander(t("hist_experiment_details_title")):
                    _meta = selected_portfolio.get("metadata") or {}
                    if not _meta:
                        st.caption(t("hist_no_experiment_metadata"))
                        # Market is still meaningful even with zero other
                        # metadata (Issue #43 item L): inferred directly
                        # from the portfolio's own saved holdings, never a
                        # bare "--" when there is real information to show.
                        st.markdown(f"**{t('hist_meta_market')}**: {_market_display_for_portfolio(selected_portfolio)}")
                    else:
                        # Old-version provenance note (Issue #43 item M): a
                        # saved record's historical window/assumptions are
                        # whatever was true when IT was saved, not a stale
                        # value to "fix" against today's app defaults --
                        # this is a neutral disclosure, never a warning.
                        _record_version = _meta.get("app_version")
                        if _record_version and _record_version != APP_VERSION:
                            st.info(t("hist_legacy_version_notice", version=_record_version))
                        _meta_rows = {
                            t("hist_meta_schema_version"): _meta.get("schema_version", "—"),
                            t("hist_meta_market"): _market_display_for_portfolio(selected_portfolio),
                            t("hist_meta_historical_window"): (
                                f"{_meta.get('historical_start_date', '—')} → {_meta.get('historical_end_date', '—')}"
                            ),
                            t("hist_meta_risk_free_rate"): (
                                f"{_meta['risk_free_rate']:.2%}" if _meta.get("risk_free_rate") is not None else "—"
                            ),
                            t("hist_meta_weight_bounds"): (
                                f"{_meta['min_weight']:.0%} – {_meta['max_weight']:.0%}"
                                if _meta.get("min_weight") is not None and _meta.get("max_weight") is not None else "—"
                            ),
                            t("hist_meta_allow_short"): (
                                t("hist_meta_yes") if _meta.get("allow_short") else t("hist_meta_no")
                            ),
                            t("hist_meta_return_estimator"): t_return_estimator(_meta.get("expected_return_estimator")),
                            t("hist_meta_covariance_estimator"): t_covariance_estimator(_meta.get("covariance_estimator")),
                            t("hist_meta_asset_universe"): ", ".join(_meta.get("asset_universe", [])) or "—",
                            t("hist_meta_data_as_of"): _meta.get("data_as_of", "—"),
                            t("hist_meta_app_version"): _meta.get("app_version", "—"),
                        }
                        for k, v in _meta_rows.items():
                            st.markdown(f"**{k}**: {v}")

        with col_right:
            if selected_portfolio["holdings"]:
                _detail_method_label = (
                    t_saved_strategy(selected_portfolio["optimization_method"])
                    if selected_portfolio["optimization_method"] else None
                )
                with chart_card(t("hist_allocation_breakdown_card"), _detail_method_label):
                    fig = allocation_donut_chart(selected_portfolio["holdings"], "")
                    st.plotly_chart(fig, use_container_width=True, key=f"history_detail_donut_{selected_portfolio['id']}")
                    chart_caption(t("hist_detail_donut_caption"))
                    _detail_context_text = (
                        f"Portfolio: {selected_portfolio['name']}. "
                        f"Expected annual return: {_safe_pct(selected_portfolio['expected_return'])}, "
                        f"Expected volatility: {_safe_pct(selected_portfolio['expected_volatility'])}, "
                        f"Sharpe ratio: {_safe_num(selected_portfolio['sharpe_ratio'], ',.2f')}. "
                        "Holdings by weight: " + "; ".join(
                            f"{tk} {w:.2%}" for tk, w in _active_holdings(selected_portfolio["holdings"]).items()
                        )
                    )
                    ai_interpret_button(
                        f"hist_detail_ai_interpret_{selected_portfolio['id']}",
                        st.session_state, _detail_context_text,
                    )

    # ── Compare Two Portfolios ────────────────────────────────────────────────────
    section_header(t("hist_compare_title"))

    if len(portfolios) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            id_a = st.selectbox(t("hist_portfolio_a"), options=list(portfolio_names.keys()),
                                 format_func=lambda x: portfolio_names[x], key="compare_a")
        with col2:
            remaining = [k for k in portfolio_names.keys() if k != id_a]
            id_b = st.selectbox(t("hist_portfolio_b"), options=remaining,
                                 format_func=lambda x: portfolio_names[x], key="compare_b")

        port_a = next((p for p in portfolios if p["id"] == id_a), None)
        port_b = next((p for p in portfolios if p["id"] == id_b), None)

        if port_a and port_b:
            # Comparison table. Effective Holdings (inverse-HHI) is recomputed
            # from each portfolio's actual saved weights via the same
            # portfolio_diagnosis() the Optimizer itself uses -- a genuinely
            # available concentration metric, not a hard-coded/estimated one --
            # so "5 ETFs selected" vs. "5 ETFs actually diversifying risk" stays
            # visible when comparing, not just raw holdings count.
            diag_a = portfolio_diagnosis(port_a["holdings"]) if port_a["holdings"] else None
            diag_b = portfolio_diagnosis(port_b["holdings"]) if port_b["holdings"] else None
            metric_col = t("hist_col_name")
            compare_data = {
                metric_col: [t("metric_optimization_method"), t("field_investment_amount_usd"), t("metric_expected_return"),
                             t("metric_expected_volatility"), t("metric_sharpe_ratio"), t("metric_number_of_holdings"),
                             t("hist_col_effective_holdings")],
                port_a["name"]: [
                    t_saved_strategy(port_a["optimization_method"]) if port_a["optimization_method"] else "—",
                    f"${_safe_num(port_a['investment_amount'])}",
                    _safe_pct(port_a["expected_return"]),
                    _safe_pct(port_a["expected_volatility"]),
                    _safe_num(port_a["sharpe_ratio"], ",.2f"),
                    str(len(port_a["holdings"])),
                    f"{diag_a['effective_holdings']:.2f}" if diag_a else "—",
                ],
                port_b["name"]: [
                    t_saved_strategy(port_b["optimization_method"]) if port_b["optimization_method"] else "—",
                    f"${_safe_num(port_b['investment_amount'])}",
                    _safe_pct(port_b["expected_return"]),
                    _safe_pct(port_b["expected_volatility"]),
                    _safe_num(port_b["sharpe_ratio"], ",.2f"),
                    str(len(port_b["holdings"])),
                    f"{diag_b['effective_holdings']:.2f}" if diag_b else "—",
                ],
            }
            compare_df = pd.DataFrame(compare_data).set_index(metric_col)
            with chart_card(t("hist_comparison_table_card")):
                st.dataframe(compare_df, use_container_width=True)
                chart_caption(t("hist_comparison_table_caption"))
                _diag_a_str = f"{diag_a['effective_holdings']:.2f}" if diag_a else "—"
                _diag_b_str = f"{diag_b['effective_holdings']:.2f}" if diag_b else "—"
                _compare_context_text = (
                    f"{port_a['name']}: return {_safe_pct(port_a['expected_return'])}, "
                    f"volatility {_safe_pct(port_a['expected_volatility'])}, "
                    f"Sharpe {_safe_num(port_a['sharpe_ratio'], ',.2f')}, "
                    f"holdings {len(port_a['holdings'])}, effective holdings {_diag_a_str}. "
                    f"{port_b['name']}: return {_safe_pct(port_b['expected_return'])}, "
                    f"volatility {_safe_pct(port_b['expected_volatility'])}, "
                    f"Sharpe {_safe_num(port_b['sharpe_ratio'], ',.2f')}, "
                    f"holdings {len(port_b['holdings'])}, effective holdings {_diag_b_str}."
                )
                ai_interpret_button(
                    f"hist_compare_ai_interpret_{port_a['id']}_{port_b['id']}",
                    st.session_state, _compare_context_text,
                )

            # Side-by-side donut charts
            col_a, col_b = st.columns(2)
            with col_a:
                if port_a["holdings"]:
                    _a_method_label = t_saved_strategy(port_a["optimization_method"]) if port_a["optimization_method"] else None
                    with chart_card(port_a["name"], _a_method_label):
                        fig_a = allocation_donut_chart(port_a["holdings"], "")
                        st.plotly_chart(fig_a, use_container_width=True, key=f"history_compare_donut_a_{port_a['id']}")
                        chart_caption(t("hist_compare_donut_caption"))
            with col_b:
                if port_b["holdings"]:
                    _b_method_label = t_saved_strategy(port_b["optimization_method"]) if port_b["optimization_method"] else None
                    with chart_card(port_b["name"], _b_method_label):
                        fig_b = allocation_donut_chart(port_b["holdings"], "")
                        st.plotly_chart(fig_b, use_container_width=True, key=f"history_compare_donut_b_{port_b['id']}")
                        chart_caption(t("hist_compare_donut_caption"))

            # Allocation comparison bar chart
            with chart_card(t("hist_allocation_comparison_card")):
                all_tickers = list(set(list(port_a["holdings"].keys()) + list(port_b["holdings"].keys())))
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name=port_a["name"],
                    x=all_tickers,
                    y=[port_a["holdings"].get(tk, 0) * 100 for tk in all_tickers],
                    marker_color=CHART_COLORS[0]
                ))
                fig_bar.add_trace(go.Bar(
                    name=port_b["name"],
                    x=all_tickers,
                    y=[port_b["holdings"].get(tk, 0) * 100 for tk in all_tickers],
                    marker_color=CHART_COLORS[1]
                ))
                fig_bar.update_layout(title=t("chart_allocation_comparison_pct"),
                                       xaxis_title=t("chart_etf"), yaxis_title=t("chart_weight"), barmode="group")
                st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True,
                                 key=f"history_compare_bar_{port_a['id']}_{port_b['id']}")
                chart_caption(t("hist_allocation_bar_caption"))
    else:
        st.info(t("hist_compare_need_two"))

    # ── Delete Portfolio ──────────────────────────────────────────────────────────
    section_header(t("hist_delete_title"))

    del_id = st.selectbox(t("hist_select_delete"),
                           options=list(portfolio_names.keys()),
                           format_func=lambda x: portfolio_names[x],
                           key="delete_select")

    # Explicit second confirmation (Issue #43 item N): selecting a portfolio
    # above never deletes it by itself. The confirmation checkbox is keyed
    # by `del_id`, so it is a BRAND NEW (unchecked) widget whenever the
    # selection changes -- switching portfolios automatically invalidates
    # any prior confirmation without extra session-state bookkeeping. The
    # Delete button stays disabled until this box is checked.
    _del_name = portfolio_names.get(del_id, "")
    confirm_delete = st.checkbox(
        t("hist_delete_confirm_checkbox", name=_del_name), key=f"hist_delete_confirm_{del_id}",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(t("btn_delete_portfolio"), type="secondary", disabled=not confirm_delete):
            if delete_portfolio(del_id):
                st.success(t("hist_delete_success"))
                st.rerun()
            else:
                st.error(t("hist_delete_failed"))
    with col2:
        st.caption(t("hist_delete_warning"))


tab_goal, tab_holdings, tab_watchlist, tab_history, tab_brief = st.tabs([
    t("mp_tab_goal"), t("mp_tab_holdings"), t("mp_tab_watchlist"),
    t("mp_tab_history"), t("mp_tab_brief"),
])

# ══════════════════════════════════════════════════════════════════════════
# GOAL PLANNER (Issue #22 section B)
# ══════════════════════════════════════════════════════════════════════════
with tab_goal:
    section_header(t("gp_section_title"), t("gp_section_subtitle"))
    col_form, col_results = st.columns([1, 1.6])

    with col_form:
        with st.container(border=True):
            st.markdown(f"**{t('gp_inputs_title')}**")
            gp_current_age = st.number_input(t("gp_current_age"), min_value=15, max_value=100, value=35, step=1, key="gp_current_age")
            gp_target_age = st.number_input(t("gp_target_age"), min_value=16, max_value=100, value=65, step=1, key="gp_target_age")
            gp_current_capital = st.number_input(t("gp_current_capital"), min_value=0.0, value=10000.0, step=1000.0, key="gp_current_capital")
            gp_monthly_contribution = st.number_input(t("gp_monthly_contribution"), min_value=0.0, value=500.0, step=50.0, key="gp_monthly_contribution")
            gp_annual_contribution = st.number_input(t("gp_annual_contribution"), min_value=0.0, value=0.0, step=500.0, key="gp_annual_contribution")

            _gp_mode_labels = {m: t_goal_target_mode(m) for m in VALID_TARGET_MODES}
            gp_target_mode = st.selectbox(
                t("gp_target_mode_label"), list(VALID_TARGET_MODES),
                format_func=lambda x: _gp_mode_labels.get(x, x), key="gp_target_mode",
            )
            gp_target_amount = st.number_input(t("gp_target_amount_label"), min_value=0.0, value=10_000_000.0, step=100_000.0, key="gp_target_amount")

            _gp_market_labels = {m: (t("gp_market_mixed") if m == "Mixed" else t_country(m)) for m in VALID_MARKET_PREFERENCES}
            gp_market_preference = st.selectbox(
                t("gp_market_preference_label"), list(VALID_MARKET_PREFERENCES),
                format_func=lambda x: _gp_market_labels.get(x, x), key="gp_market_preference",
            )

            _gp_risk_labels = {r: t_goal_risk(r) for r in VALID_RISK_TOLERANCES}
            gp_risk_tolerance = st.selectbox(
                t("gp_risk_tolerance_label"), list(VALID_RISK_TOLERANCES),
                index=1, format_func=lambda x: _gp_risk_labels.get(x, x), key="gp_risk_tolerance",
            )

            gp_base_currency = st.selectbox(t("gp_base_currency_label"), list(VALID_BASE_CURRENCIES), key="gp_base_currency")

            gp_derived_horizon = int(gp_target_age) - int(gp_current_age)
            gp_override_horizon = st.checkbox(t("gp_horizon_override_checkbox"), value=False, key="gp_override_horizon")
            gp_horizon_years = None
            if gp_override_horizon:
                gp_horizon_years = st.number_input(
                    t("gp_horizon_override_years"), min_value=1.0, max_value=80.0,
                    value=float(max(gp_derived_horizon, 1)), step=1.0, key="gp_horizon_years",
                )
            else:
                st.caption(t("gp_horizon_derived_note", years=gp_derived_horizon))

    with col_results:
        gp_plan = build_goal_plan(
            current_age=int(gp_current_age), target_age=int(gp_target_age),
            current_capital=float(gp_current_capital), monthly_contribution=float(gp_monthly_contribution),
            target_mode=gp_target_mode, target_amount=float(gp_target_amount),
            market_preference=gp_market_preference, risk_tolerance=gp_risk_tolerance,
            base_currency=gp_base_currency, annual_contribution=float(gp_annual_contribution),
            horizon_years=gp_horizon_years,
        )

        if not gp_plan["valid"]:
            error_state(t("gp_invalid_title"), " ".join(gp_plan["errors"]))
        else:
            section_header(t("gp_results_title"), t("gp_results_subtitle"))
            if gp_plan["withdrawal_rate"] is not None:
                st.caption(t(
                    "gp_income_target_summary",
                    withdrawal=f"{gp_plan['withdrawal_rate']:.0%}",
                    target=f"{gp_plan['implied_target_total']:,.0f}",
                    currency=gp_base_currency,
                ))
            else:
                st.caption(t(
                    "gp_target_summary", years=f"{gp_plan['horizon_years']:.0f}",
                    target=f"{gp_plan['implied_target_total']:,.0f}", currency=gp_base_currency,
                ))

            _gp_status_color = {
                "on_track": COLORS["success"], "below_target": COLORS["danger"], "above_target": COLORS["primary"],
            }
            gp_scenario_cols = st.columns(3)
            for gp_col, gp_scenario_name in zip(gp_scenario_cols, ["conservative", "balanced", "aggressive"]):
                gp_sdata = gp_plan["scenarios"][gp_scenario_name]
                gp_mc = _simulate_goal_scenario(
                    float(gp_current_capital), float(gp_monthly_contribution),
                    float(gp_annual_contribution), int(round(gp_plan["horizon_years"])),
                    float(gp_sdata["expected_return"]), float(gp_sdata["expected_volatility"]),
                    float(gp_plan["implied_target_total"]),
                )
                with gp_col:
                    with st.container(border=True):
                        st.markdown(f"**{t_goal_risk(gp_scenario_name)}**")
                        st.metric(t("gp_expected_return_label"), f"{gp_sdata['expected_return']:.1%}")
                        st.metric(t("gp_expected_volatility_label"), f"{gp_sdata['expected_volatility']:.1%}")
                        st.metric(t("gp_target_attainment_label"), f"{gp_mc['target_share']:.1%}")
                        st.metric(t("gp_sim_median_label"), f"{gp_mc['median']:,.0f} {gp_base_currency}")
                        st.caption(t(
                            "gp_sim_range_caption",
                            p10=f"{gp_mc['p10']:,.0f}", p90=f"{gp_mc['p90']:,.0f}",
                            currency=gp_base_currency, n=f"{gp_mc['n_simulations']:,}",
                        ))
                        st.metric(t("gp_required_contribution_label"), f"{gp_sdata['required_monthly_contribution']:,.0f} {gp_base_currency}")
                        st.markdown(
                            f"<span style='color:{_gp_status_color[gp_sdata['status']]};font-weight:700;'>"
                            f"{t_goal_status(gp_sdata['status'])}</span>",
                            unsafe_allow_html=True,
                        )
                        _mix = gp_sdata.get("asset_mix", {})
                        if _mix:
                            st.caption(t(
                                "gp_asset_mix_label",
                                equity=f"{_mix.get('Equity', 0):.0%}",
                                bonds=f"{_mix.get('Fixed Income', 0):.0%}",
                            ))
                        if gp_sdata["example_etfs"]:
                            st.caption(f"{t('gp_example_etfs_label')}: {', '.join(gp_sdata['example_etfs'])}")
                            _market_display = t("gp_market_mixed") if gp_market_preference == "Mixed" else t_country(gp_market_preference)
                            st.caption(t(
                                f"gp_selection_logic_{gp_scenario_name}",
                                market=_market_display,
                            ))
                        else:
                            st.caption(t("gp_no_examples"))

            with st.expander(t("gp_assumptions_title"), expanded=False):
                st.markdown(f"- {t('gp_assumptions_hypothetical')}")
                st.markdown(f"- {t('gp_assumptions_return_source')}")
                st.markdown(f"- {t('gp_assumptions_monte_carlo')}")
                st.markdown(f"- {t('gp_assumptions_currency')}")
                st.markdown(f"- {t('gp_assumptions_inflation')}")
                st.markdown(f"- {t('gp_assumptions_rebalance')}")
                if gp_plan["withdrawal_rate"] is not None:
                    gp_rate_str = f"{gp_plan['withdrawal_rate']:.0%}"
                    st.markdown(f"- {t('gp_assumptions_withdrawal', rate=gp_rate_str)}")
                st.markdown(f"- {t('gp_assumptions_not_advice')}")

# ══════════════════════════════════════════════════════════════════════════
# CURRENT HOLDINGS (Issue #22 section D)
# ══════════════════════════════════════════════════════════════════════════
with tab_holdings:
    section_header(t("ch_section_title"), t("ch_section_subtitle"))

    with st.expander(t("ch_add_title"), expanded=True):
        ch_c1, ch_c2, ch_c3, ch_c4, ch_c5 = st.columns([1.2, 1, 1, 1, 1.2])
        with ch_c1:
            ch_ticker = st.text_input(t("ch_ticker_label"), key="ch_new_ticker").upper().strip()
        with ch_c2:
            ch_qty = st.number_input(t("ch_quantity_label"), min_value=0.0, value=0.0, step=1.0, key="ch_new_qty")
        with ch_c3:
            ch_cost = st.number_input(t("ch_avg_cost_label"), min_value=0.0, value=0.0, step=1.0, key="ch_new_cost")
        with ch_c4:
            ch_ccy = st.selectbox(t("ch_currency_label"), list(VALID_BASE_CURRENCIES), key="ch_new_ccy")
        with ch_c5:
            ch_date = st.date_input(t("ch_purchase_date_label"), value=None, key="ch_new_date")

        if st.button(t("ch_add_button"), type="primary", key="ch_add_btn"):
            if not ch_ticker:
                pass
            elif ch_qty <= 0:
                st.error(t("ch_quantity_required"))
            elif not _is_known_or_live_ticker(ch_ticker):
                st.error(t("ch_invalid_ticker_error", ticker=ch_ticker))
            else:
                _add_session_holding(ch_ticker, ch_qty, ch_cost, ch_ccy, str(ch_date) if ch_date else None)
                st.session_state["_ch_just_added"] = ch_ticker
                st.rerun()

    # st.rerun() above starts a fresh script run immediately -- a
    # st.success() called before it would never actually reach the browser
    # (same reason Portfolio History's "_hist_just_set_name" flag exists).
    # Stash the ticker in session_state and show the message on the NEXT run.
    _ch_just_added = st.session_state.pop("_ch_just_added", None)
    if _ch_just_added:
        st.success(t("ch_added_success", ticker=_ch_just_added))

    holdings_rows = _session_holdings()
    if not holdings_rows:
        empty_state(t("ch_empty_title"), t("ch_empty_desc"), icon="layers")
    else:
        section_header(t("ch_table_title"))
        ch_table_rows = []
        _ch_live_values = {}
        _ch_missing_price = False
        for _h in holdings_rows:
            _price = _fetch_latest_price(_h["ticker"])
            if _price is not None:
                _mv = _price * _h["quantity"]
                _pl = (_price - _h["average_cost"]) * _h["quantity"]
                _ch_live_values[_h["ticker"]] = _ch_live_values.get(_h["ticker"], 0.0) + _mv
                _price_str, _mv_str, _pl_str = f"{_price:,.2f}", f"{_mv:,.2f}", f"{_pl:,.2f}"
            else:
                _ch_missing_price = True
                _price_str = _mv_str = _pl_str = t("ch_price_unavailable")
            ch_table_rows.append({
                t("ch_col_ticker"): _h["ticker"],
                t("ch_col_quantity"): _h["quantity"],
                t("ch_col_avg_cost"): _h["average_cost"],
                t("ch_col_currency"): _h["currency"],
                t("ch_col_purchase_date"): _h["purchase_date"] or "—",
                t("ch_col_market_price"): _price_str,
                t("ch_col_market_value"): _mv_str,
                t("ch_col_unrealized_pl"): _pl_str,
            })
        st.dataframe(pd.DataFrame(ch_table_rows), use_container_width=True, hide_index=True)
        chart_caption(t("ch_holdings_table_caption"))
        st.caption(t("ch_price_note"))
        _ch_context_text = "; ".join(
            f"{row[t('ch_col_ticker')]}: qty {row[t('ch_col_quantity')]}, "
            f"avg cost {row[t('ch_col_avg_cost')]}, price {row[t('ch_col_market_price')]}, "
            f"market value {row[t('ch_col_market_value')]}, unrealized P/L {row[t('ch_col_unrealized_pl')]}"
            for row in ch_table_rows
        )
        ai_interpret_button("ch_holdings_ai_interpret_guest_session", st.session_state, _ch_context_text)

        _handoff_weights = _holdings_weights_from_live_values(_ch_live_values)
        if _handoff_weights and not _ch_missing_price:
            st.caption(t("ch_handoff_note"))
            _risk_col, _opt_col = st.columns(2)
            with _risk_col:
                if st.button(t("ch_analyze_risk_button"), key="ch_analyze_risk", use_container_width=True, type="primary"):
                    _set_holdings_as_current_portfolio(holdings_rows, _ch_live_values)
                    _countries = [get_country(tk) for tk in _handoff_weights if get_country(tk)]
                    _unique_countries = list(dict.fromkeys(_countries))
                    _risk_region = _unique_countries[0] if len(_unique_countries) == 1 else t("field_all_regions")
                    st.session_state["_selected_region_shadow"] = _risk_region
                    st.session_state["selected_region"] = _risk_region
                    _shadow_map = dict(st.session_state.get("_selected_etfs_shadow", {}))
                    _shadow_map[_risk_region] = list(_handoff_weights)
                    st.session_state["_selected_etfs_shadow"] = _shadow_map
                    for _tk, _w in _handoff_weights.items():
                        st.session_state[f"w_{_tk}"] = float(_w) * 100.0
                    st.switch_page("pages/4_Risk_Analytics.py")
            with _opt_col:
                if st.button(t("ch_optimize_button"), key="ch_optimize", use_container_width=True):
                    _countries = [get_country(tk) for tk in _handoff_weights if get_country(tk)]
                    _unique_countries = list(dict.fromkeys(_countries))
                    if _unique_countries:
                        st.session_state["_selected_regions_shadow"] = _unique_countries
                        st.session_state["selected_regions"] = _unique_countries
                        st.session_state["_selected_etfs_multi_master"] = list(_handoff_weights)
                        _widget_key = "selected_etfs_portfolio_" + "+".join(sorted(_unique_countries))
                        st.session_state[_widget_key] = list(_handoff_weights)
                    st.switch_page("pages/2_Portfolio_Optimizer.py")
        elif _ch_missing_price:
            st.caption(t("ch_handoff_requires_prices"))

        ch_del_map = {_h["id"]: f"{_h['ticker']} ({_h['quantity']:g} @ {_h['average_cost']:g})" for _h in holdings_rows}
        ch_del_id = st.selectbox(t("ch_remove_button"), options=list(ch_del_map.keys()), format_func=lambda x: ch_del_map[x], key="ch_remove_select")
        if st.button(t("ch_remove_button"), key="ch_remove_btn"):
            _delete_session_holding(ch_del_id)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# WATCHLIST (Issue #22 section D)
# ══════════════════════════════════════════════════════════════════════════
with tab_watchlist:
    section_header(t("wl_section_title"), t("wl_section_subtitle"))

    wl_c1, wl_c2 = st.columns([3, 1])
    with wl_c1:
        wl_ticker = st.text_input(t("wl_add_label"), key="wl_new_ticker").upper().strip()
    with wl_c2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        wl_add_clicked = st.button(t("wl_add_button"), type="primary", key="wl_add_btn", use_container_width=True)

    watchlist_rows = _session_watchlist()
    _wl_existing = {w["ticker"] for w in watchlist_rows}

    if wl_add_clicked and wl_ticker:
        if wl_ticker in _wl_existing:
            st.info(t("wl_already_present", ticker=wl_ticker))
        elif not _is_known_or_live_ticker(wl_ticker):
            st.error(t("wl_invalid_ticker_error", ticker=wl_ticker))
        else:
            _add_session_watchlist(wl_ticker)
            st.session_state["_wl_just_added"] = wl_ticker
            st.rerun()

    # Same rerun-timing reasoning as Current Holdings' "_ch_just_added" flag.
    _wl_just_added = st.session_state.pop("_wl_just_added", None)
    if _wl_just_added:
        st.success(t("wl_added_success", ticker=_wl_just_added))

    if not watchlist_rows:
        empty_state(t("wl_empty_title"), t("wl_empty_desc"), icon="layers")
    else:
        section_header(t("wl_table_title"))
        wl_table_rows = []
        for _w in watchlist_rows:
            _wp = _fetch_latest_price(_w["ticker"])
            wl_table_rows.append({
                t("wl_col_ticker"): _w["ticker"],
                t("wl_col_price"): f"{_wp:,.2f}" if _wp is not None else t("ch_price_unavailable"),
                t("wl_col_added"): _w["created_at"] or "—",
            })
        st.dataframe(pd.DataFrame(wl_table_rows), use_container_width=True, hide_index=True)
        chart_caption(t("wl_table_caption"))

        wl_del_map = {_w["id"]: _w["ticker"] for _w in watchlist_rows}
        wl_del_id = st.selectbox(t("wl_remove_button"), options=list(wl_del_map.keys()), format_func=lambda x: wl_del_map[x], key="wl_remove_select")
        if st.button(t("wl_remove_button"), key="wl_remove_btn"):
            _delete_session_watchlist(wl_del_id)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# PORTFOLIO HISTORY (existing functionality, unchanged, now a tab)
# ══════════════════════════════════════════════════════════════════════════
with tab_history:
    section_header(t("hist_title"), t("hist_subtitle"))
    _render_portfolio_history_tab()

# ══════════════════════════════════════════════════════════════════════════
# DAILY BRIEF (Issue #22 section E)
# ══════════════════════════════════════════════════════════════════════════
with tab_brief:
    section_header(t("db_section_title"), t("db_section_subtitle"))
    _brief_holdings = _session_holdings()
    _brief_watchlist = _session_watchlist()

    if not _brief_holdings and not _brief_watchlist:
        empty_state(t("db_empty_no_data"), t("db_section_subtitle"), icon="layers")
    else:
        try:
            _brief_news = fetch_market_news(limit=10)
        except Exception:
            _brief_news = []

        _brief_context = build_brief_context(_brief_holdings, _brief_watchlist, _brief_news)
        _brief_result = generate_daily_brief(_brief_context, st.session_state, language=get_language())

        with chart_card(
            t("db_section_title"),
            tag=t("ai_tag_generated") if _brief_result["source"] == "ai" else t("ai_tag_rule_based"),
        ):
            st.markdown(_brief_result["text"])

        section_header(t("db_risk_news_title"))
        _notable = [m for m in _brief_context["news_matches"] if m.get("impact") != "Neutral"]
        if _notable:
            _notable_cols = st.columns(min(len(_notable), 4))
            for _col, _m in zip(_notable_cols, _notable):
                with _col:
                    st.markdown(
                        status_card(
                            _m["ticker"], _m.get("sector", ""), _m.get("impact_label", _m["impact"]),
                            IMPACT_VARIANT.get(_m["impact"], "neutral"),
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(t("db_no_major_change"))

        section_header(t("db_watchpoints_title"))
        if _brief_watchlist:
            st.markdown(" · ".join(w["ticker"] for w in _brief_watchlist))
        else:
            st.caption(t("db_empty_no_data"))

disclaimer_box()
render_footer()
