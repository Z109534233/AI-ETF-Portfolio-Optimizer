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
from src.etf_database import get_country
from src.charts import allocation_donut_chart, apply_dark_theme, CHART_COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, empty_state
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

# ── Load Portfolios ───────────────────────────────────────────────────────────
portfolios = load_all_portfolios()

if not portfolios:
    empty_state(
        t("hist_no_portfolios_title"),
        t("hist_no_portfolios_desc"),
        icon="layers",
    )
    st.stop()

# ── Summary Table ─────────────────────────────────────────────────────────────
section_header(t("hist_saved_portfolios_count", count=len(portfolios)))

summary_rows = []
for p in portfolios:
    holdings_str = ", ".join([f"{tk} ({w:.0%})" for tk, w in
                               sorted(p["holdings"].items(), key=lambda x: x[1], reverse=True)[:5]])
    summary_rows.append({
        "ID": p["id"],
        t("hist_col_name"): p["name"],
        t("hist_col_created"): p["created_at"],
        t("hist_col_method"): t_opt_method(p["optimization_method"]),
        t("hist_col_investment"): f"${p['investment_amount']:,.0f}",
        t("hist_col_exp_return"): f"{p['expected_return']:.2%}",
        t("hist_col_exp_volatility"): f"{p['expected_volatility']:.2%}",
        t("hist_col_sharpe"): f"{p['sharpe_ratio']:.2f}",
        t("hist_col_holdings"): holdings_str,
    })

summary_df = pd.DataFrame(summary_rows)
with chart_card(t("hist_summary_card")):
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

    with col_left:
        with chart_card(selected_portfolio["name"], selected_portfolio["created_at"]):
            detail_data = {
                t("metric_optimization_method"): t_opt_method(selected_portfolio["optimization_method"]),
                t("field_investment_amount_usd"): f"${selected_portfolio['investment_amount']:,.2f}",
                t("metric_expected_annual_return"): f"{selected_portfolio['expected_return']:.2%}",
                t("metric_expected_volatility"): f"{selected_portfolio['expected_volatility']:.2%}",
                t("metric_sharpe_ratio"): f"{selected_portfolio['sharpe_ratio']:.2f}",
            }
            for k, v in detail_data.items():
                st.metric(k, v)

            if selected_portfolio["notes"]:
                st.markdown(f"**{t('hist_notes_label')}**: {selected_portfolio['notes']}")

            # Holdings table
            if selected_portfolio["holdings"]:
                holdings_df = pd.DataFrame([
                    {t("hist_col_ticker"): tk,
                     t("hist_col_region"): t_country(get_country(tk)) if get_country(tk) else t("hist_region_unknown"),
                     t("hist_col_weight"): f"{w:.2%}",
                     t("hist_col_amount"): f"${w * selected_portfolio['investment_amount']:,.2f}"}
                    for tk, w in sorted(selected_portfolio["holdings"].items(), key=lambda x: x[1], reverse=True)
                ])
                st.markdown(f"**{t('hist_holdings_label')}**")
                st.dataframe(holdings_df.set_index(t("hist_col_ticker")), use_container_width=True)

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
        # Comparison table
        metric_col = t("hist_col_name")
        compare_data = {
            metric_col: [t("metric_optimization_method"), t("field_investment_amount_usd"), t("metric_expected_return"),
                         t("metric_expected_volatility"), t("metric_sharpe_ratio"), t("metric_number_of_holdings")],
            port_a["name"]: [
                t_opt_method(port_a["optimization_method"]),
                f"${port_a['investment_amount']:,.0f}",
                f"{port_a['expected_return']:.2%}",
                f"{port_a['expected_volatility']:.2%}",
                f"{port_a['sharpe_ratio']:.2f}",
                str(len(port_a["holdings"])),
            ],
            port_b["name"]: [
                t_opt_method(port_b["optimization_method"]),
                f"${port_b['investment_amount']:,.0f}",
                f"{port_b['expected_return']:.2%}",
                f"{port_b['expected_volatility']:.2%}",
                f"{port_b['sharpe_ratio']:.2f}",
                str(len(port_b["holdings"])),
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
