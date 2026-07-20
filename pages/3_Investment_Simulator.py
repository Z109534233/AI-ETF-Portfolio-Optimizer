"""
Page 3: Investment Simulator
Long-term investment projection with Monte Carlo simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.simulator import simulate_investment, compound_growth_projection, scenario_comparison, MARKET_SCENARIOS
from src.database import save_simulation, init_database
from src.charts import monte_carlo_paths_chart, future_value_distribution_chart, apply_dark_theme
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, dataframe_to_csv
from src.ui import render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer
from src.theme import COLORS

st.set_page_config(
    page_title="Investment Simulator | AI ETF Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

load_css()
init_database()

page_header(
    "Investment Simulator",
    "Long-term investment projection with Monte Carlo simulation"
)

st.info(
    "**Projection Disclaimer**: All projections are hypothetical and based on assumed return and "
    "volatility parameters. They do not represent guaranteed outcomes. Market conditions vary "
    "significantly over time. This tool is for educational purposes only."
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown("### Simulation Parameters")

    initial_investment = st.number_input("Initial Investment ($)", 100.0, 1_000_000.0, 10000.0, 500.0)
    monthly_contribution = st.number_input("Monthly Contribution ($)", 0.0, 50000.0, 500.0, 100.0)
    years = st.slider("Investment Period (Years)", 1, 40, 20)

    st.markdown("---")
    st.markdown("### Market Assumptions")
    scenario = st.selectbox("Market Scenario", list(MARKET_SCENARIOS.keys()) + ["Custom"])

    if scenario == "Custom":
        annual_return = st.slider("Expected Annual Return (%)", -5.0, 30.0, 10.0, 0.5) / 100
        annual_volatility = st.slider("Expected Annual Volatility (%)", 1.0, 50.0, 15.0, 0.5) / 100
    else:
        annual_return = MARKET_SCENARIOS[scenario]["return"]
        annual_volatility = MARKET_SCENARIOS[scenario]["volatility"]
        st.info(f"Return: {annual_return:.1%} | Volatility: {annual_volatility:.1%}")

    inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.5, 0.25) / 100
    annual_fee = st.slider("Annual Management Fee (%)", 0.0, 3.0, 0.1, 0.05) / 100
    n_simulations = st.slider("Number of Simulations", 200, 5000, 1000, 100)

    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Run Simulation ────────────────────────────────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None

if run_btn or st.session_state.sim_result is None:
    with st.spinner("Running Monte Carlo simulation..."):
        sim_result = simulate_investment(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            inflation_rate=inflation_rate,
            annual_fee=annual_fee,
            n_simulations=n_simulations
        )
        st.session_state.sim_result = sim_result
        st.session_state.sim_params = {
            "initial_investment": initial_investment,
            "monthly_contribution": monthly_contribution,
            "years": years,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "inflation_rate": inflation_rate,
            "annual_fee": annual_fee,
        }

sim_result = st.session_state.sim_result
if sim_result is None:
    st.info("Configure parameters in the sidebar and click **Run Simulation**.")
    st.stop()

summary = sim_result["summary"]
annual_table = sim_result["annual_table"]
paths_df = sim_result["paths"]
final_values = sim_result["all_final_values"]
total_contributed = summary["total_contributed"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header("Simulation Results", f"{n_simulations:,} simulated paths over {years} years")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html("Median Final Value", f"${summary['median_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Total Contributed", f"${total_contributed:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Median Investment Gain", f"${summary['median_gain']:,.0f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Inflation-Adj. Value", f"${summary['real_median_final']:,.0f}", color=COLORS["warning"]), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(metric_card_html("Optimistic (90th pct)", f"${summary['optimistic_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Pessimistic (10th pct)", f"${summary['pessimistic_final']:,.0f}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Probability of Profit", f"{summary['probability_profit']:.1%}", color=COLORS["primary"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header("Projection Charts")
with chart_card("Simulation Detail"):
    tab1, tab2, tab3, tab4 = st.tabs(["Monte Carlo Paths", "Compound Growth", "Value Distribution", "Annual Table"])

    with tab1:
        fig_mc = monte_carlo_paths_chart(paths_df, f"Monte Carlo Simulation — {years} Years")
        st.plotly_chart(fig_mc, use_container_width=True)

    with tab2:
        # Compound growth projection (deterministic)
        growth_df = compound_growth_projection(
            initial_investment, monthly_contribution, years,
            annual_return, annual_fee, inflation_rate
        )
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Balance"],
            name="Portfolio Balance", line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Contributed"],
            name="Total Contributed", line=dict(color=COLORS["text_muted"], width=2, dash="dash")
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Real Balance"],
            name="Real Value (Inflation-Adj.)", line=dict(color=COLORS["warning"], width=1.5, dash="dot")
        ))
        fig_growth.update_layout(
            title="Compound Growth Projection (Deterministic)",
            xaxis_title="Years", yaxis_title="Value ($)"
        )
        st.plotly_chart(apply_dark_theme(fig_growth), use_container_width=True)

        # Contributions vs gains
        fig_bar = go.Figure()
        yearly = growth_df[growth_df["Year"] == growth_df["Year"].astype(int)]
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Contributed"],
                                  name="Contributed", marker_color=COLORS["primary"]))
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Gain"].clip(lower=0),
                                  name="Investment Gain", marker_color=COLORS["success"]))
        fig_bar.update_layout(title="Contributions vs Investment Gains",
                               xaxis_title="Year", yaxis_title="Value ($)", barmode="stack")
        st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True)

    with tab3:
        fig_dist = future_value_distribution_chart(final_values, total_contributed)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Percentile table
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pct_data = {
            "Percentile": [f"{p}th" for p in percentiles],
            "Final Value": [f"${np.percentile(final_values, p):,.0f}" for p in percentiles],
            "Gain": [f"${np.percentile(final_values, p) - total_contributed:,.0f}" for p in percentiles],
            "Return Multiple": [f"{np.percentile(final_values, p) / initial_investment:.1f}x" for p in percentiles],
        }
        st.markdown("**Outcome Percentile Table**")
        st.dataframe(pd.DataFrame(pct_data).set_index("Percentile"), use_container_width=True)

    with tab4:
        st.markdown("**Annual Balance Summary (Median Path)**")
        display_table = annual_table.copy()
        for col in ["Portfolio Value", "Total Contributed", "Investment Gain", "Real Value (Inflation-Adj.)"]:
            display_table[col] = display_table[col].apply(lambda x: f"${x:,.0f}")
        display_table["Return %"] = display_table["Return %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_table.set_index("Year"), use_container_width=True)

        csv = dataframe_to_csv(annual_table)
        st.download_button("Download Annual Table (CSV)", csv, "simulation_annual.csv", "text/csv")

# ── Scenario Comparison ───────────────────────────────────────────────────────
section_header("Market Scenario Comparison")
with st.spinner("Computing scenario comparison..."):
    scenario_df = scenario_comparison(initial_investment, monthly_contribution, years, annual_fee)
with chart_card("Scenario Comparison Table"):
    st.dataframe(scenario_df.set_index("Scenario"), use_container_width=True)

# ── Save Simulation ───────────────────────────────────────────────────────────
section_header("Save Simulation")
if st.button("Save Simulation to History"):
    success = save_simulation(
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
        expected_return=annual_return,
        final_value=summary["median_final"]
    )
    if success:
        st.success("Simulation saved to history.")
    else:
        st.warning("Could not save simulation.")

disclaimer_box()
render_footer()
