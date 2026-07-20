"""
Page 2: Portfolio Optimizer
Mean-variance optimization, efficient frontier, and portfolio backtesting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.portfolio_optimizer import (
    run_optimization, monte_carlo_simulation, backtest_portfolio,
    compute_efficient_frontier
)
from src.financial_metrics import (
    covariance_matrix, annualized_return, annualized_volatility,
    sharpe_ratio, maximum_drawdown, drawdown_series
)
from src.database import save_portfolio, init_database
from src.report_generator import generate_portfolio_report
from src.charts import (
    efficient_frontier_chart, allocation_donut_chart,
    portfolio_growth_chart, drawdown_chart, apply_dark_theme
)
from src.utils import (
    load_css, page_header, disclaimer_box, dataframe_to_csv,
    weights_to_dataframe, get_date_range_defaults, metric_card_html
)
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state
)
from src.theme import COLORS

st.set_page_config(
    page_title="Portfolio Optimizer | AI ETF Portfolio Optimizer",
    page_icon="⚡",
    layout="wide"
)

load_css()
init_database()

page_header(
    "Portfolio Optimizer",
    "Mean-variance optimization, efficient frontier, and portfolio backtesting"
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown("### Portfolio Settings")

    selected_etfs = st.multiselect(
        "Select ETFs",
        options=DEFAULT_ETFS,
        default=["VOO", "QQQ", "BND", "GLD", "VNQ"],
        help="Select 2 or more ETFs to optimize"
    )

    custom_ticker = st.text_input("Add Custom Ticker", placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    investment_amount = st.number_input("Investment Amount ($)", min_value=100.0,
                                         max_value=10_000_000.0, value=10000.0, step=500.0)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100

    st.markdown("---")
    st.markdown("### Optimization Constraints")
    optimization_method = st.selectbox(
        "Optimization Method",
        ["Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility", "Target Return", "Risk Parity"]
    )

    min_weight = st.slider("Min Weight per ETF (%)", 0.0, 20.0, 0.0, 1.0) / 100
    max_weight = st.slider("Max Weight per ETF (%)", 10.0, 100.0, 100.0, 5.0) / 100
    allow_short = st.checkbox("Allow Short Selling", value=False)

    target_return_pct = None
    if optimization_method == "Target Return":
        target_return_pct = st.slider("Target Annual Return (%)", 1.0, 30.0, 10.0, 0.5) / 100

    n_simulations = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, 500)

    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_etfs) < 2:
    st.warning("Please select at least 2 ETFs to run portfolio optimization.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ── State Management ──────────────────────────────────────────────────────────
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None

if run_btn or st.session_state.opt_result is None:
    with st.spinner("Downloading data and running optimization..."):
        raw_prices = download_etf_data(selected_etfs, str(start_date), str(end_date))
        if raw_prices.empty:
            error_state("No Price Data Available", "Check your ticker symbols and date range, then try again.")
            st.stop()

        prices_df = clean_price_data(raw_prices)
        prices_df = prices_df[[t for t in selected_etfs if t in prices_df.columns]]

        if prices_df.empty or len(prices_df) < 20:
            error_state("Insufficient Price Data", "Widen the date range or choose different ETFs for optimization.")
            st.stop()

        result = run_optimization(
            prices_df=prices_df,
            method=optimization_method,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            target_return=target_return_pct
        )

        if result.get("error"):
            st.warning(f"Optimization note: {result['error']}")

        st.session_state.opt_result = result
        st.session_state.prices_df = prices_df

result = st.session_state.opt_result
prices_df = st.session_state.prices_df

if result is None or prices_df is None:
    st.info("Configure settings in the sidebar and click **Run Optimization**.")
    st.stop()

weights = result["weights"]
exp_ret = result["expected_return"]
exp_vol = result["expected_volatility"]
sharpe = result["sharpe_ratio"]
div_ratio = result.get("diversification_ratio", 1.0)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header("Optimization Results")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(metric_card_html("Expected Annual Return", f"{exp_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Expected Volatility", f"{exp_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Sharpe Ratio", f"{sharpe:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Diversification Ratio", f"{div_ratio:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html("Method", optimization_method, color=COLORS["warning"]), unsafe_allow_html=True)

# ── Allocation Table & Donut ──────────────────────────────────────────────────
section_header("Portfolio Allocation")
col_left, col_right = st.columns([1, 1])

with col_left:
    with chart_card("Allocation Table", f"{len(weights)} holdings"):
        alloc_df = weights_to_dataframe(weights, investment_amount)
        st.dataframe(alloc_df[["Ticker", "Weight", "Allocation ($)"]].style.hide(axis="index"),
                     use_container_width=True)

with col_right:
    with chart_card("Allocation Breakdown", optimization_method):
        fig_donut = allocation_donut_chart(weights, "")
        st.plotly_chart(fig_donut, use_container_width=True, key="opt_allocation_donut")

# ── Efficient Frontier ────────────────────────────────────────────────────────
section_header("Efficient Frontier & Monte Carlo Simulation", f"{n_simulations:,} simulated portfolios")

with st.spinner("Running Monte Carlo simulation..."):
    returns_df = prices_df.pct_change().dropna()
    mean_returns = returns_df.mean().values
    cov = covariance_matrix(prices_df).values
    cov += np.eye(len(prices_df.columns)) * 1e-8

    mc_df = monte_carlo_simulation(mean_returns, cov, n_simulations, risk_free_rate)

with chart_card("Efficient Frontier"):
    fig_ef = efficient_frontier_chart(mc_df, weights, None, mean_returns, cov)
    st.plotly_chart(fig_ef, use_container_width=True, key="opt_efficient_frontier")

# ── Backtest ──────────────────────────────────────────────────────────────────
section_header("Historical Backtest", f"{optimization_method} vs. Equal Weight")

backtest_df = backtest_portfolio(prices_df, weights, investment_amount)
equal_weights = {t: 1.0 / len(weights) for t in weights.keys()}
equal_backtest_df = backtest_portfolio(prices_df, equal_weights, investment_amount)

if not backtest_df.empty:
    import plotly.graph_objects as go
    with chart_card("Portfolio Backtest vs Equal Weight"):
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["Portfolio Value"],
            name=f"{optimization_method}", line=dict(color=COLORS["primary"], width=2.5)
        ))
        if not equal_backtest_df.empty:
            fig_bt.add_trace(go.Scatter(
                x=equal_backtest_df.index, y=equal_backtest_df["Portfolio Value"],
                name="Equal Weight", line=dict(color=COLORS["text_muted"], width=1.5, dash="dash")
            ))
        fig_bt.update_layout(title="Portfolio Backtest vs Equal Weight",
                              xaxis_title="Date", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(apply_dark_theme(fig_bt), use_container_width=True, key="opt_backtest_growth")

    # Drawdown comparison
    with chart_card("Drawdown Comparison"):
        fig_dd = go.Figure()
        dd = drawdown_series(backtest_df["Portfolio Value"]) * 100
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                     name=optimization_method, line=dict(color=COLORS["danger"], width=1.5)))
        if not equal_backtest_df.empty:
            dd_eq = drawdown_series(equal_backtest_df["Portfolio Value"]) * 100
            fig_dd.add_trace(go.Scatter(x=dd_eq.index, y=dd_eq, fill="tozeroy",
                                         name="Equal Weight", line=dict(color=COLORS["text_muted"], width=1.5),
                                         fillcolor="rgba(148,163,184,0.1)"))
        fig_dd.update_layout(title="Drawdown Comparison (%)", xaxis_title="Date", yaxis_title="Drawdown (%)")
        st.plotly_chart(apply_dark_theme(fig_dd), use_container_width=True, key="opt_backtest_drawdown")

    # Backtest metrics
    bt_metrics = {
        "Total Return": f"{backtest_df['Cumulative Return'].iloc[-1]:.2%}",
        "Annualized Return": f"{annualized_return(backtest_df['Portfolio Value']):.2%}",
        "Annualized Volatility": f"{annualized_volatility(backtest_df['Portfolio Value']):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(backtest_df['Portfolio Value'], risk_free_rate):.2f}",
        "Maximum Drawdown": f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}",
        "Final Value": f"${backtest_df['Portfolio Value'].iloc[-1]:,.2f}",
    }
    st.markdown("**Backtest Performance Summary**")
    cols = st.columns(len(bt_metrics))
    for i, (k, v) in enumerate(bt_metrics.items()):
        with cols[i]:
            st.metric(k, v)

# ── Save & Download ───────────────────────────────────────────────────────────
section_header("Save & Export")

col1, col2, col3 = st.columns(3)

with col1:
    portfolio_name = st.text_input("Portfolio Name", value=f"Portfolio_{optimization_method.replace(' ', '_')}")
    notes = st.text_area("Notes (optional)", height=80)
    if st.button("Save Portfolio", type="primary"):
        metrics_to_save = {
            "Expected Annual Return": f"{exp_ret:.2%}",
            "Expected Annual Volatility": f"{exp_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
        }
        success = save_portfolio(
            name=portfolio_name,
            weights=weights,
            investment_amount=investment_amount,
            optimization_method=optimization_method,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            notes=notes
        )
        if success:
            st.success(f"Portfolio '{portfolio_name}' saved successfully!")
        else:
            st.error("Failed to save portfolio. Please try again.")

with col2:
    alloc_csv = dataframe_to_csv(alloc_df)
    st.download_button(
        "Download Allocation CSV",
        alloc_csv, "portfolio_allocation.csv", "text/csv",
        use_container_width=True
    )

with col3:
    try:
        bt_metrics_full = {
            "Expected Annual Return": f"{exp_ret:.2%}",
            "Expected Annual Volatility": f"{exp_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Diversification Ratio": f"{div_ratio:.2f}",
        }
        if not backtest_df.empty:
            bt_metrics_full["Maximum Drawdown"] = f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}"
            bt_metrics_full["Annualized Return (Backtest)"] = f"{annualized_return(backtest_df['Portfolio Value']):.2%}"

        pdf_bytes = generate_portfolio_report(
            portfolio_name=portfolio_name,
            weights=weights,
            metrics=bt_metrics_full,
            investment_amount=investment_amount,
            optimization_method=optimization_method,
            notes=notes
        )
        st.download_button(
            "Download PDF Report",
            pdf_bytes, f"{portfolio_name}.pdf", "application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"PDF generation unavailable: {e}")

disclaimer_box()
render_footer()
