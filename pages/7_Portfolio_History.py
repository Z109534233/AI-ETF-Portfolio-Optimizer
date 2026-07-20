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
from src.charts import allocation_donut_chart, apply_dark_theme, CHART_COLORS
from src.utils import load_css, page_header, disclaimer_box, metric_card_html

st.set_page_config(
    page_title="Portfolio History | AI ETF Portfolio Optimizer",
    page_icon="📚",
    layout="wide"
)

load_css()
init_database()

page_header(
    "Portfolio History",
    "View, compare, and manage your saved portfolios",
    "📚"
)

# ── Load Portfolios ───────────────────────────────────────────────────────────
portfolios = load_all_portfolios()

if not portfolios:
    st.info(
        "No saved portfolios found. "
        "Go to the **Portfolio Optimizer** page to create and save a portfolio."
    )
    st.stop()

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown(f"### Saved Portfolios ({len(portfolios)} total)")

summary_rows = []
for p in portfolios:
    holdings_str = ", ".join([f"{t} ({w:.0%})" for t, w in
                               sorted(p["holdings"].items(), key=lambda x: x[1], reverse=True)[:5]])
    summary_rows.append({
        "ID": p["id"],
        "Name": p["name"],
        "Created": p["created_at"],
        "Method": p["optimization_method"],
        "Investment": f"${p['investment_amount']:,.0f}",
        "Exp. Return": f"{p['expected_return']:.2%}",
        "Exp. Volatility": f"{p['expected_volatility']:.2%}",
        "Sharpe": f"{p['sharpe_ratio']:.2f}",
        "Holdings": holdings_str,
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df.set_index("ID"), use_container_width=True)

# ── Download History ──────────────────────────────────────────────────────────
csv = summary_df.to_csv(index=True).encode("utf-8")
st.download_button("Download History (CSV)", csv, "portfolio_history.csv", "text/csv")

# ── View Portfolio Details ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### View Portfolio Details")

portfolio_names = {p["id"]: f"[{p['id']}] {p['name']} ({p['created_at']})" for p in portfolios}
selected_id = st.selectbox("Select Portfolio", options=list(portfolio_names.keys()),
                            format_func=lambda x: portfolio_names[x])

selected_portfolio = next((p for p in portfolios if p["id"] == selected_id), None)

if selected_portfolio:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"#### {selected_portfolio['name']}")
        detail_data = {
            "Created": selected_portfolio["created_at"],
            "Optimization Method": selected_portfolio["optimization_method"],
            "Investment Amount": f"${selected_portfolio['investment_amount']:,.2f}",
            "Expected Annual Return": f"{selected_portfolio['expected_return']:.2%}",
            "Expected Volatility": f"{selected_portfolio['expected_volatility']:.2%}",
            "Sharpe Ratio": f"{selected_portfolio['sharpe_ratio']:.2f}",
        }
        for k, v in detail_data.items():
            st.metric(k, v)

        if selected_portfolio["notes"]:
            st.markdown(f"**Notes**: {selected_portfolio['notes']}")

        # Holdings table
        if selected_portfolio["holdings"]:
            holdings_df = pd.DataFrame([
                {"Ticker": t, "Weight": f"{w:.2%}", "Amount": f"${w * selected_portfolio['investment_amount']:,.2f}"}
                for t, w in sorted(selected_portfolio["holdings"].items(), key=lambda x: x[1], reverse=True)
            ])
            st.markdown("**Holdings**")
            st.dataframe(holdings_df.set_index("Ticker"), use_container_width=True)

    with col_right:
        if selected_portfolio["holdings"]:
            fig = allocation_donut_chart(selected_portfolio["holdings"], selected_portfolio["name"])
            st.plotly_chart(fig, use_container_width=True)

# ── Compare Two Portfolios ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Compare Two Portfolios")

if len(portfolios) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        id_a = st.selectbox("Portfolio A", options=list(portfolio_names.keys()),
                             format_func=lambda x: portfolio_names[x], key="compare_a")
    with col2:
        remaining = [k for k in portfolio_names.keys() if k != id_a]
        id_b = st.selectbox("Portfolio B", options=remaining,
                             format_func=lambda x: portfolio_names[x], key="compare_b")

    port_a = next((p for p in portfolios if p["id"] == id_a), None)
    port_b = next((p for p in portfolios if p["id"] == id_b), None)

    if port_a and port_b:
        # Comparison table
        compare_data = {
            "Metric": ["Optimization Method", "Investment Amount", "Expected Return",
                        "Expected Volatility", "Sharpe Ratio", "Number of Holdings"],
            port_a["name"]: [
                port_a["optimization_method"],
                f"${port_a['investment_amount']:,.0f}",
                f"{port_a['expected_return']:.2%}",
                f"{port_a['expected_volatility']:.2%}",
                f"{port_a['sharpe_ratio']:.2f}",
                str(len(port_a["holdings"])),
            ],
            port_b["name"]: [
                port_b["optimization_method"],
                f"${port_b['investment_amount']:,.0f}",
                f"{port_b['expected_return']:.2%}",
                f"{port_b['expected_volatility']:.2%}",
                f"{port_b['sharpe_ratio']:.2f}",
                str(len(port_b["holdings"])),
            ],
        }
        compare_df = pd.DataFrame(compare_data).set_index("Metric")
        st.dataframe(compare_df, use_container_width=True)

        # Side-by-side donut charts
        col_a, col_b = st.columns(2)
        with col_a:
            if port_a["holdings"]:
                fig_a = allocation_donut_chart(port_a["holdings"], port_a["name"])
                st.plotly_chart(fig_a, use_container_width=True)
        with col_b:
            if port_b["holdings"]:
                fig_b = allocation_donut_chart(port_b["holdings"], port_b["name"])
                st.plotly_chart(fig_b, use_container_width=True)

        # Allocation comparison bar chart
        all_tickers = list(set(list(port_a["holdings"].keys()) + list(port_b["holdings"].keys())))
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name=port_a["name"],
            x=all_tickers,
            y=[port_a["holdings"].get(t, 0) * 100 for t in all_tickers],
            marker_color=CHART_COLORS[0]
        ))
        fig_bar.add_trace(go.Bar(
            name=port_b["name"],
            x=all_tickers,
            y=[port_b["holdings"].get(t, 0) * 100 for t in all_tickers],
            marker_color=CHART_COLORS[1]
        ))
        fig_bar.update_layout(title="Allocation Comparison (%)",
                               xaxis_title="ETF", yaxis_title="Weight (%)", barmode="group")
        st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True)
else:
    st.info("Save at least 2 portfolios to enable comparison.")

# ── Delete Portfolio ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Delete Portfolio")

del_id = st.selectbox("Select Portfolio to Delete",
                       options=list(portfolio_names.keys()),
                       format_func=lambda x: portfolio_names[x],
                       key="delete_select")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Delete Portfolio", type="secondary"):
        if delete_portfolio(del_id):
            st.success(f"Portfolio deleted successfully.")
            st.rerun()
        else:
            st.error("Failed to delete portfolio.")
with col2:
    st.caption("⚠️ This action cannot be undone.")

disclaimer_box()
