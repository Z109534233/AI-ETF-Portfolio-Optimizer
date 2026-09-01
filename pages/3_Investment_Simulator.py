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
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header, chart_card,
    render_footer, render_current_portfolio_handoff,
)
from src.theme import COLORS
from src.i18n import t, t_market_scenario, MARKET_SCENARIO_KEYS

st.set_page_config(
    page_title="Investment Simulator | AI ETF Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

load_css()
init_database()

page_header(t("sim_title"), t("sim_subtitle"))

# ── Current Portfolio handoff (Round 2B-4) ───────────────────────────────────
# Proof-of-handoff preview only -- the rest of this page (below) remains
# fully self-contained and never requires a current_portfolio to exist;
# see render_current_portfolio_handoff() in src/ui.py.
render_current_portfolio_handoff(
    t("handoff_empty_state_title"), t("handoff_empty_state_body_sim"),
)

st.info(t("sim_projection_disclaimer"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('sim_sidebar_params')}")

    initial_investment = st.number_input(t("sim_initial_investment"), 100.0, 1_000_000.0, 10000.0, 500.0)
    monthly_contribution = st.number_input(t("sim_monthly_contribution"), 0.0, 50000.0, 500.0, 100.0)
    years = st.slider(t("sim_investment_years"), 1, 40, 20)

    st.markdown("---")
    st.markdown(f"### {t('sim_market_assumptions')}")
    # Pre-resolve labels once (within a valid script context) rather than
    # passing a format_func that reads st.session_state on every invocation.
    _scenario_labels = {k: t_market_scenario(k) for k in MARKET_SCENARIOS}
    _scenario_labels["Custom"] = t("sim_scenario_custom")
    scenario = st.selectbox(
        t("sim_market_scenario"),
        list(MARKET_SCENARIOS.keys()) + ["Custom"],
        format_func=lambda x: _scenario_labels.get(x, x),
    )

    if scenario == "Custom":
        annual_return = st.slider(t("sim_expected_annual_return_pct"), -5.0, 30.0, 10.0, 0.5) / 100
        annual_volatility = st.slider(t("sim_expected_annual_volatility_pct"), 1.0, 50.0, 15.0, 0.5) / 100
    else:
        annual_return = MARKET_SCENARIOS[scenario]["return"]
        annual_volatility = MARKET_SCENARIOS[scenario]["volatility"]
        st.info(t("sim_scenario_return_vol", ret=f"{annual_return:.1%}", vol=f"{annual_volatility:.1%}"))

    inflation_rate = st.slider(t("sim_inflation_rate_pct"), 0.0, 10.0, 2.5, 0.25) / 100
    annual_fee = st.slider(t("sim_annual_fee_pct"), 0.0, 3.0, 0.1, 0.05) / 100
    n_simulations = st.slider(t("sim_number_of_simulations"), 200, 5000, 1000, 100)

    run_btn = st.button(t("btn_run_simulation"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Run Simulation ────────────────────────────────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None

if run_btn or st.session_state.sim_result is None:
    with st.spinner(t("sim_running_monte_carlo")):
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
    st.info(t("msg_configure_and_run", action=t("btn_run_simulation")))
    st.stop()

summary = sim_result["summary"]
annual_table = sim_result["annual_table"]
paths_df = sim_result["paths"]
final_values = sim_result["all_final_values"]
total_contributed = summary["total_contributed"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("sim_results_title"), t("sim_results_sub", count=f"{n_simulations:,}", years=years))
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html(t("metric_median_final_value"), f"${summary['median_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_total_contributed"), f"${total_contributed:,.0f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_median_investment_gain"), f"${summary['median_gain']:,.0f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_inflation_adjusted_value"), f"${summary['real_median_final']:,.0f}", color=COLORS["warning"]), unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(metric_card_html(t("metric_optimistic_90"), f"${summary['optimistic_final']:,.0f}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_pessimistic_10"), f"${summary['pessimistic_final']:,.0f}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_probability_of_profit"), f"{summary['probability_profit']:.1%}", color=COLORS["primary"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("sim_projection_charts_title"))
with chart_card(t("sim_simulation_detail_card")):
    tab1, tab2, tab3, tab4 = st.tabs([
        t("sim_tab_monte_carlo"), t("sim_tab_compound_growth"), t("sim_tab_value_distribution"), t("sim_tab_annual_table")
    ])

    with tab1:
        fig_mc = monte_carlo_paths_chart(paths_df, t("chart_monte_carlo_simulation") + f" — {years}")
        st.plotly_chart(fig_mc, use_container_width=True, key="sim_monte_carlo_paths")

    with tab2:
        # Compound growth projection (deterministic)
        growth_df = compound_growth_projection(
            initial_investment, monthly_contribution, years,
            annual_return, annual_fee, inflation_rate
        )
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Balance"],
            name=t("chart_portfolio_balance"), line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Contributed"],
            name=t("chart_total_contributed"), line=dict(color=COLORS["text_muted"], width=2, dash="dash")
        ))
        fig_growth.add_trace(go.Scatter(
            x=growth_df["Year"], y=growth_df["Real Balance"],
            name=t("chart_real_value_inflation_adj"), line=dict(color=COLORS["warning"], width=1.5, dash="dot")
        ))
        fig_growth.update_layout(
            title=t("chart_portfolio_growth_deterministic"),
            xaxis_title=t("chart_years"), yaxis_title=t("chart_value_usd")
        )
        st.plotly_chart(apply_dark_theme(fig_growth), use_container_width=True, key="sim_compound_growth")

        # Contributions vs gains
        fig_bar = go.Figure()
        yearly = growth_df[growth_df["Year"] == growth_df["Year"].astype(int)]
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Contributed"],
                                  name=t("chart_contributed"), marker_color=COLORS["primary"]))
        fig_bar.add_trace(go.Bar(x=yearly["Year"], y=yearly["Gain"].clip(lower=0),
                                  name=t("chart_investment_gain"), marker_color=COLORS["success"]))
        fig_bar.update_layout(title=t("chart_contributions_vs_gains"),
                               xaxis_title=t("chart_year"), yaxis_title=t("chart_value_usd"), barmode="stack")
        st.plotly_chart(apply_dark_theme(fig_bar), use_container_width=True, key="sim_contributions_vs_gains")

    with tab3:
        fig_dist = future_value_distribution_chart(final_values, total_contributed)
        st.plotly_chart(fig_dist, use_container_width=True, key="sim_value_distribution")

        # Percentile table
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_col = t("sim_percentile_col")
        pct_data = {
            percentile_col: [f"{p}th" for p in percentiles],
            t("sim_final_value_col"): [f"${np.percentile(final_values, p):,.0f}" for p in percentiles],
            t("sim_gain_col"): [f"${np.percentile(final_values, p) - total_contributed:,.0f}" for p in percentiles],
            t("sim_return_multiple_col"): [f"{np.percentile(final_values, p) / initial_investment:.1f}x" for p in percentiles],
        }
        st.markdown(f"**{t('sim_outcome_percentile_table')}**")
        st.dataframe(pd.DataFrame(pct_data).set_index(percentile_col), use_container_width=True)

    with tab4:
        st.markdown(f"**{t('sim_annual_balance_summary')}**")
        display_table = annual_table.copy()
        for col in ["Portfolio Value", "Total Contributed", "Investment Gain", "Real Value (Inflation-Adj.)"]:
            display_table[col] = display_table[col].apply(lambda x: f"${x:,.0f}")
        display_table["Return %"] = display_table["Return %"].apply(lambda x: f"{x:.1f}%")
        display_table = display_table.rename(columns={
            "Portfolio Value": t("metric_final_value"),
            "Total Contributed": t("chart_total_contributed"),
            "Investment Gain": t("chart_investment_gain"),
            "Real Value (Inflation-Adj.)": t("chart_real_value_inflation_adj"),
            "Return %": t("chart_return"),
            "Year": t("chart_year"),
        })
        st.dataframe(display_table.set_index(t("chart_year")), use_container_width=True)

        csv = dataframe_to_csv(annual_table)
        st.download_button(t("btn_download_annual_table"), csv, "simulation_annual.csv", "text/csv")

# ── Scenario Comparison ───────────────────────────────────────────────────────
section_header(t("sim_scenario_comparison_title"))
with st.spinner(t("msg_running_optimization")):
    scenario_df = scenario_comparison(initial_investment, monthly_contribution, years, annual_fee)
with chart_card(t("sim_scenario_comparison_card")):
    display_scenario_df = scenario_df.copy()
    display_scenario_df["Scenario"] = display_scenario_df["Scenario"].apply(t_market_scenario)
    st.dataframe(display_scenario_df.rename(columns={"Scenario": t("sim_market_scenario")}).set_index(t("sim_market_scenario")),
                 use_container_width=True)

# ── Save Simulation ───────────────────────────────────────────────────────────
section_header(t("sim_save_simulation_title"))
if st.button(t("btn_save_simulation_history")):
    success = save_simulation(
        initial_investment=initial_investment,
        monthly_contribution=monthly_contribution,
        years=years,
        expected_return=annual_return,
        final_value=summary["median_final"]
    )
    if success:
        st.success(t("sim_saved_success"))
    else:
        st.warning(t("sim_save_failed"))

disclaimer_box()
render_footer()
