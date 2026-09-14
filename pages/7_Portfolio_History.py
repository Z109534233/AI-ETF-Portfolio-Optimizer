"""
Page 7: Portfolio History
View, compare, and manage saved portfolios from SQLite database.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database import load_all_portfolios, delete_portfolio, init_database

# Weights below this are hidden from the holdings summary (Issue #20
# section 9B) -- e.g. an optimizer run that assigns 0.02% to a ticker
# clutters the UI without being a meaningful allocation. Raw weights are
# never discarded: total_disclosed_weight-style totals and "Set as Current
# Portfolio" both keep operating on the FULL saved dict, only the display
# table/summary text filters below this threshold, and always labels that
# it did.
_ACTIVE_HOLDING_THRESHOLD = 0.001
from src.etf_database import get_country
from src.financial_metrics import portfolio_diagnosis
from src.charts import allocation_donut_chart, apply_dark_theme, CHART_COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, empty_state, error_state
)
from src.i18n import t, t_opt_method, t_country

st.set_page_config(
    page_title="Portfolio History | AI ETF Portfolio Optimizer",
    page_icon="📚",
    layout="wide"
)

load_css()
init_database()

page_header(t("hist_title"), t("hist_subtitle"))

with st.sidebar:
    render_sidebar_nav()
    render_sidebar_footer()


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
    """Holdings at/above _ACTIVE_HOLDING_THRESHOLD, sorted by weight desc.
    Never mutates or discards the underlying raw weights -- callers that
    need the full saved dict (e.g. "Set as Current Portfolio", the CSV
    export, or portfolio_diagnosis()) must keep reading `holdings` itself,
    not this filtered view."""
    return dict(sorted(
        ((tk, w) for tk, w in holdings.items() if w >= _ACTIVE_HOLDING_THRESHOLD),
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


# ── Load Portfolios ───────────────────────────────────────────────────────────
try:
    portfolios = load_all_portfolios(raise_on_error=True)
except Exception as e:
    error_state(t("hist_load_error_title"), t("hist_load_error_desc", error=str(e)))
    disclaimer_box()
    render_footer()
    st.stop()

if not portfolios:
    empty_state(
        t("hist_no_portfolios_title"),
        t("hist_no_portfolios_desc"),
        icon="layers",
    )
    st.stop()

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
    summary_rows.append({
        "ID": p["id"],
        t("hist_col_name"): p["name"] or "—",
        t("hist_col_created"): p["created_at"],
        t("hist_col_method"): t_opt_method(p["optimization_method"]) if p["optimization_method"] else "—",
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

    with col_left:
        with chart_card(selected_portfolio["name"], selected_portfolio["created_at"]):
            if _is_current:
                st.caption(f"✓ {t('hist_current_portfolio_badge')}")
            detail_data = {
                t("metric_optimization_method"): (
                    t_opt_method(selected_portfolio["optimization_method"])
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
                if _hidden_count:
                    with st.expander(t("hist_show_all_holdings")):
                        _raw_df = pd.DataFrame([
                            {t("hist_col_ticker"): tk, t("hist_col_weight"): f"{w:.4%}"}
                            for tk, w in sorted(_all_holdings.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(_raw_df.set_index(t("hist_col_ticker")), use_container_width=True)

            # Experiment metadata (Issue #20 section 9C) -- shown for every
            # portfolio; legacy saves (before this feature existed) simply
            # have an empty dict rather than fabricated values.
            with st.expander(t("hist_experiment_details_title")):
                _meta = selected_portfolio.get("metadata") or {}
                if not _meta:
                    st.caption(t("hist_no_experiment_metadata"))
                else:
                    _meta_rows = {
                        t("hist_meta_schema_version"): _meta.get("schema_version", "—"),
                        t("hist_meta_market"): t_country(_meta["market"]) if _meta.get("market") else "—",
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
                        t("hist_meta_return_estimator"): _meta.get("expected_return_estimator", "—"),
                        t("hist_meta_covariance_estimator"): _meta.get("covariance_estimator", "—"),
                        t("hist_meta_asset_universe"): ", ".join(_meta.get("asset_universe", [])) or "—",
                        t("hist_meta_data_as_of"): _meta.get("data_as_of", "—"),
                        t("hist_meta_app_version"): _meta.get("app_version", "—"),
                    }
                    for k, v in _meta_rows.items():
                        st.markdown(f"**{k}**: {v}")

    with col_right:
        if selected_portfolio["holdings"]:
            with chart_card(t("hist_allocation_breakdown_card")):
                fig = allocation_donut_chart(selected_portfolio["holdings"], "")
                st.plotly_chart(fig, use_container_width=True, key=f"history_detail_donut_{selected_portfolio['id']}")

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
                t_opt_method(port_a["optimization_method"]) if port_a["optimization_method"] else "—",
                f"${_safe_num(port_a['investment_amount'])}",
                _safe_pct(port_a["expected_return"]),
                _safe_pct(port_a["expected_volatility"]),
                _safe_num(port_a["sharpe_ratio"], ",.2f"),
                str(len(port_a["holdings"])),
                f"{diag_a['effective_holdings']:.2f}" if diag_a else "—",
            ],
            port_b["name"]: [
                t_opt_method(port_b["optimization_method"]) if port_b["optimization_method"] else "—",
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

        # Side-by-side donut charts
        col_a, col_b = st.columns(2)
        with col_a:
            if port_a["holdings"]:
                with chart_card(port_a["name"]):
                    fig_a = allocation_donut_chart(port_a["holdings"], "")
                    st.plotly_chart(fig_a, use_container_width=True, key=f"history_compare_donut_a_{port_a['id']}")
        with col_b:
            if port_b["holdings"]:
                with chart_card(port_b["name"]):
                    fig_b = allocation_donut_chart(port_b["holdings"], "")
                    st.plotly_chart(fig_b, use_container_width=True, key=f"history_compare_donut_b_{port_b['id']}")

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
else:
    st.info(t("hist_compare_need_two"))

# ── Delete Portfolio ──────────────────────────────────────────────────────────
section_header(t("hist_delete_title"))

del_id = st.selectbox(t("hist_select_delete"),
                       options=list(portfolio_names.keys()),
                       format_func=lambda x: portfolio_names[x],
                       key="delete_select")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button(t("btn_delete_portfolio"), type="secondary"):
        if delete_portfolio(del_id):
            st.success(t("hist_delete_success"))
            st.rerun()
        else:
            st.error(t("hist_delete_failed"))
with col2:
    st.caption(t("hist_delete_warning"))

disclaimer_box()
render_footer()
