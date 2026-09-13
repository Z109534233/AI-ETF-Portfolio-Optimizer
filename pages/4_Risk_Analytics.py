"""
Page 4: Risk Analytics
Comprehensive portfolio risk analysis with stress testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data
from src.data_cleaner import clean_price_data
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    maximum_drawdown, calmar_ratio, beta, alpha,
    downside_deviation, tracking_error, information_ratio,
    correlation_matrix, covariance_matrix, drawdown_series, diversification_ratio
)
from src.risk_analytics import (
    historical_var_cvar, concentration_from_weights, holdings_overlap_matrix,
    STRESS_SCENARIOS, var_exception_backtest,
)
from src.charts import (
    correlation_heatmap, return_distribution_chart, drawdown_chart,
    rolling_metrics_chart, apply_dark_theme, CHART_COLORS
)
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header,
    chart_card, render_footer, error_state, style_signed_columns,
    region_selector, region_etf_options, region_etf_multiselect, region_benchmark_selector,
    render_current_portfolio_handoff,
)
from src.theme import COLORS
from src.i18n import t, t_country

st.set_page_config(
    page_title="Risk Analytics | AI ETF Portfolio Optimizer",
    page_icon="🛡️",
    layout="wide"
)

load_css()

page_header(t("risk_title"), t("risk_subtitle"))

# ── Current Portfolio handoff (Round 2B-4) ───────────────────────────────────
# Proof-of-handoff preview only -- the rest of this page (below) remains
# fully self-contained and never requires a current_portfolio to exist;
# see render_current_portfolio_handoff() in src/ui.py.
render_current_portfolio_handoff(
    t("handoff_empty_state_title"), t("handoff_empty_state_body_risk"),
)
current_portfolio = st.session_state.get("current_portfolio")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('risk_sidebar_settings')}")

    # ── Region Selector (Global ETF Support) ─────────────────────────────
    # "United States" preserves the exact original ETF list (DEFAULT_ETFS)
    # so existing behavior is unchanged unless the user explicitly picks a
    # different region.
    #
    # region_selector() / region_etf_multiselect() (src/ui.py) are the SAME
    # shared helpers used by ETF Analysis and AI Advisor: they read and
    # write one canonical st.session_state["selected_region"] /
    # ["selected_etfs_<region>"], so picking a market or ETF here is
    # immediately reflected on those other pages too, not just persisted
    # within this page.
    selected_region, ALL_REGIONS_LABEL = region_selector()
    etf_options = region_etf_options(selected_region, ALL_REGIONS_LABEL)
    selected_etfs = region_etf_multiselect(
        selected_region, etf_options, t("field_select_etfs"),
        help_text=t("risk_select_etfs_help"), n_default=4,
    )

    custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    # Portfolio weights
    st.markdown(f"#### {t('risk_weights_label')}")
    weights_input = {}
    if selected_etfs:
        equal_w = 1.0 / len(selected_etfs)
        for ticker in selected_etfs:
            w = st.slider(t("risk_weight_pct", ticker=ticker), 0.0, 100.0, equal_w * 100, 1.0, key=f"w_{ticker}")
            weights_input[ticker] = w / 100.0
        total_w = sum(weights_input.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(t("risk_weights_normalised_warning", total=f"{total_w:.1%}"))
            if total_w > 0:
                weights_input = {k: v / total_w for k, v in weights_input.items()}

    # Market-aware benchmark selector -- see the identical fix + rationale
    # in pages/1_ETF_Analysis.py (Global ETF Universe + Benchmark
    # Architecture round: fixes the confirmed "Taiwan region + QQQ
    # benchmark" bug, which affected this page too via the exact same
    # hardcoded-DEFAULT_ETFS/index=2 pattern).
    benchmark = region_benchmark_selector(selected_region, etf_options, t("field_benchmark"))
    risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, 5.0, 0.25) / 100

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("field_start_date"), value=default_start)
    end_date = st.date_input(t("field_end_date"), value=default_end)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if not selected_etfs:
    st.warning(t("msg_select_one_etf"))
    st.stop()

# ── Data Loading ──────────────────────────────────────────────────────────────
with st.spinner(t("msg_downloading_market_data")):
    all_tickers = list(set(selected_etfs + [benchmark]))
    # Map each display ticker to its actual Yahoo Finance-fetchable symbol
    # (e.g. "0050" -> "0050.TW"). Tickers not in the ETF database (including
    # the US-only benchmark) pass through unchanged.
    yahoo_tickers = [to_yahoo_symbol(tk) for tk in all_tickers]
    raw_prices = download_etf_data(yahoo_tickers, str(start_date), str(end_date))

if raw_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

prices = clean_price_data(raw_prices)
prices = rename_yahoo_columns(prices)
etf_prices = prices[[tk for tk in selected_etfs if tk in prices.columns]]
bench_prices = prices[benchmark].dropna() if benchmark in prices.columns else None

if etf_prices.empty:
    error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
    st.stop()

# Build portfolio price series
weights_arr = np.array([weights_input.get(tk, 0) for tk in etf_prices.columns])
if weights_arr.sum() > 0:
    weights_arr = weights_arr / weights_arr.sum()
returns_df = etf_prices.pct_change().dropna()
port_returns = (returns_df * weights_arr).sum(axis=1)
port_prices = (1 + port_returns).cumprod() * 100

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("risk_portfolio_metrics_title"))

ann_ret = annualized_return(port_prices)
ann_vol = annualized_volatility(port_prices)
sr = sharpe_ratio(port_prices, risk_free_rate)
so_r = sortino_ratio(port_prices, risk_free_rate)
mdd = maximum_drawdown(port_prices)
cal = calmar_ratio(port_prices)
# Historical VaR/CVaR (M3): same value_at_risk()/conditional_var() estimator
# as before, wrapped so method/confidence/holding period/window are always
# disclosed, and so too little data produces an explicit "unavailable"
# state instead of a number computed from a handful of days.
var_cvar_result = historical_var_cvar(port_prices, confidence=0.95)
var95 = var_cvar_result["var"] if var_cvar_result["available"] else None
cvar95 = var_cvar_result["cvar"] if var_cvar_result["available"] else None
dd_dev = downside_deviation(port_prices, risk_free_rate)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html(t("metric_annualized_return"), f"{ann_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_annualized_volatility"), f"{ann_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sr:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_sortino_ratio"), f"{so_r:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_maximum_drawdown"), f"{mdd:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
    st.markdown(metric_card_html(t("metric_calmar_ratio"), f"{cal:.2f}", color=COLORS["warning"]), unsafe_allow_html=True)
with col4:
    if var_cvar_result["available"]:
        st.markdown(metric_card_html(t("metric_var_95"), f"{var95:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
        st.markdown(metric_card_html(t("metric_cvar_95"), f"{cvar95:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
    else:
        st.markdown(metric_card_html(t("metric_var_95"), t("risk_var_unavailable"), color=COLORS["text_muted"]), unsafe_allow_html=True)
        st.markdown(metric_card_html(t("metric_cvar_95"), t("risk_var_unavailable"), color=COLORS["text_muted"]), unsafe_allow_html=True)
if not var_cvar_result["available"]:
    st.caption(t("risk_var_unavailable_reason", reason=var_cvar_result["reason"]))
else:
    # VaR interpretation tooltip/caption (Issue #20 section 5A): a VaR
    # figure is a loss-THRESHOLD estimate at the stated confidence level,
    # never a maximum-loss guarantee -- losses can and do exceed it.
    st.caption(f"ℹ️ {t('risk_var_interpretation_caption')}")

# ── Methodology & Assumptions (M3) ──────────────────────────────────────
# Compact disclosure of the ACTUAL risk methodology -- see
# src/methodology.py's RISK_METHODOLOGY, the single source of truth this
# panel and tests/test_methodology_m3.py both read from.
with st.expander(t("risk_methodology_title"), expanded=False):
    st.caption(t("risk_methodology_subtitle"))
    if var_cvar_result["available"]:
        _var_window = t(
            "risk_methodology_var_window_value",
            start=var_cvar_result["window_start"], end=var_cvar_result["window_end"],
            n=var_cvar_result["n_observations"],
        )
    else:
        _var_window = t("risk_var_unavailable")
    _var_confidence_pct = f"{var_cvar_result['confidence']:.0%}"
    _var_desc = t("risk_methodology_var_desc", confidence=_var_confidence_pct)
    st.markdown(
        f"- **{t('risk_methodology_var_label')}** — {_var_desc} {_var_window}\n"
        f"- **{t('risk_methodology_mdd_label')}** — {t('risk_methodology_mdd_desc')}\n"
        f"- **{t('risk_methodology_concentration_label')}** — {t('risk_methodology_concentration_desc')}\n"
        f"- **{t('risk_methodology_correlation_label')}** — {t('risk_methodology_correlation_desc')}\n"
        f"- **{t('risk_methodology_stress_label')}** — {t('risk_methodology_stress_desc')}"
    )

# ── VaR Backtesting (Issue #20 section 5B) ───────────────────────────────
# An out-of-sample exception backtest: at every historical forecast date,
# the VaR threshold is estimated using ONLY the trailing window of returns
# strictly before that date (see src.risk_analytics.var_exception_backtest
# for the no-look-ahead guarantee), then checked against what actually
# happened. The Kupiec Proportion-of-Failures test's p-value only measures
# compatibility with the null hypothesis that the model is correctly
# calibrated at this confidence level -- it is not a "the model passed"
# certificate, and this page never states one.
section_header(t("risk_var_backtest_title"), t("risk_var_backtest_subtitle"))
_var_backtest = var_exception_backtest(port_prices, confidence=0.95, window=250)
if not _var_backtest["available"]:
    st.info(t("risk_var_backtest_unavailable", reason=_var_backtest["reason"]))
else:
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.markdown(metric_card_html(
            t("risk_var_backtest_n_forecasts"), f"{_var_backtest['n_forecasts']:,}", color=COLORS["primary"],
        ), unsafe_allow_html=True)
    with bcol2:
        _expected_str = f"{_var_backtest['expected_exceptions']:.1f}"
        _n_exceptions_value = f"{_var_backtest['n_exceptions']} ({t('risk_var_backtest_expected', n=_expected_str)})"
        st.markdown(metric_card_html(
            t("risk_var_backtest_n_exceptions"), _n_exceptions_value, color=COLORS["danger"],
        ), unsafe_allow_html=True)
    with bcol3:
        st.markdown(metric_card_html(
            t("risk_var_backtest_exception_rate"), f"{_var_backtest['exception_rate']:.2%}", color=COLORS["warning"],
        ), unsafe_allow_html=True)
    with bcol4:
        st.markdown(metric_card_html(
            t("risk_var_backtest_kupiec_pvalue"), f"{_var_backtest['kupiec_p_value']:.4f}", color=COLORS["cyan"],
        ), unsafe_allow_html=True)

    st.caption(t(
        "risk_var_backtest_window_disclosure",
        window=_var_backtest["window"], start=_var_backtest["backtest_start"], end=_var_backtest["backtest_end"],
    ))
    with st.expander(t("risk_var_backtest_interpretation_title"), expanded=False):
        st.markdown(t(
            "risk_var_backtest_null_hypothesis",
            confidence=f"{_var_backtest['confidence']:.0%}",
        ))
        st.markdown(t(
            "risk_var_backtest_pvalue_meaning",
            p_value=f"{_var_backtest['kupiec_p_value']:.4f}",
        ))
        st.markdown(t("risk_var_backtest_no_pass_claim"))

# Benchmark metrics
if bench_prices is not None and len(bench_prices) > 10:
    b = beta(port_prices, bench_prices)
    a = alpha(port_prices, bench_prices, risk_free_rate)
    te = tracking_error(port_prices, bench_prices)
    ir = information_ratio(port_prices, bench_prices)

    section_header(t("risk_benchmark_metrics_title"), t("risk_benchmark_metrics_sub", benchmark=benchmark))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card_html(t("metric_beta"), f"{b:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card_html(t("metric_alpha"), f"{a:.2%}", color=COLORS["success"] if a >= 0 else COLORS["danger"]), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card_html(t("metric_tracking_error"), f"{te:.2%}", color=COLORS["warning"]), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card_html(t("metric_information_ratio"), f"{ir:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("risk_charts_title"))
with chart_card(t("risk_detail_card")):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("risk_tab_drawdown"), t("risk_tab_rolling_metrics"), t("risk_tab_return_distribution"),
        t("risk_tab_correlation"), t("risk_tab_risk_contribution")
    ])

    with tab1:
        fig_dd = drawdown_chart(etf_prices)
        st.plotly_chart(fig_dd, use_container_width=True, key="risk_drawdown_all")

        # Portfolio drawdown
        dd_series = drawdown_series(port_prices) * 100
        fig_port_dd = go.Figure()
        fig_port_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill="tozeroy", name=t("chart_portfolio_drawdown"),
            line=dict(color=COLORS["danger"], width=2),
            fillcolor="rgba(248,113,113,0.15)"
        ))
        fig_port_dd.update_layout(title=t("chart_portfolio_drawdown_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
        st.plotly_chart(apply_dark_theme(fig_port_dd), use_container_width=True, key="risk_drawdown_portfolio")

    with tab2:
        col_sel = st.selectbox(t("risk_select_rolling_etf"), etf_prices.columns.tolist(), key="risk_rolling")
        window = st.slider(t("risk_rolling_window_days"), 21, 252, 63, key="risk_window")
        p = etf_prices[col_sel].dropna()
        if len(p) > window:
            fig_roll = rolling_metrics_chart(p, window)
            st.plotly_chart(fig_roll, use_container_width=True, key="risk_rolling_metrics")

        # Rolling beta
        if bench_prices is not None and len(bench_prices) > window:
            ret_etf = p.pct_change().dropna()
            ret_bench = bench_prices.pct_change().dropna()
            common = ret_etf.index.intersection(ret_bench.index)
            if len(common) > window:
                rolling_beta = pd.Series(index=common, dtype=float)
                for i in range(window, len(common)):
                    r_e = ret_etf.loc[common[i - window:i]]
                    r_b = ret_bench.loc[common[i - window:i]]
                    cov_mat = np.cov(r_e, r_b)
                    rolling_beta.iloc[i] = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] != 0 else 1.0
                rolling_beta = rolling_beta.dropna()
                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta,
                                               name=f"{t('chart_beta')} ({window}d)",
                                               line=dict(color=COLORS["primary"], width=2)))
                fig_beta.add_hline(y=1.0, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.6)
                fig_beta.update_layout(title=t("chart_rolling_beta_window", benchmark=benchmark, window=window),
                                        xaxis_title=t("chart_date"), yaxis_title=t("chart_beta"))
                st.plotly_chart(apply_dark_theme(fig_beta), use_container_width=True, key="risk_rolling_beta")

    with tab3:
        fig_dist = return_distribution_chart(etf_prices)
        st.plotly_chart(fig_dist, use_container_width=True, key="risk_return_distribution_all")

        # Portfolio return distribution
        fig_port_dist = go.Figure()
        fig_port_dist.add_trace(go.Histogram(
            x=port_returns * 100, nbinsx=60,
            marker_color=COLORS["primary"], opacity=0.8, name=t("chart_portfolio_returns")
        ))
        if var95 is not None:
            fig_port_dist.add_vline(x=float(var95 * 100), line_dash="dash", line_color=COLORS["danger"],
                                     annotation_text=f"{t('metric_var_95')}: {var95:.2%}")
        if cvar95 is not None:
            fig_port_dist.add_vline(x=float(cvar95 * 100), line_dash="dash", line_color=COLORS["warning"],
                                     annotation_text=f"{t('metric_cvar_95')}: {cvar95:.2%}")
        fig_port_dist.update_layout(title=t("chart_portfolio_daily_return_dist"),
                                     xaxis_title=t("chart_daily_return_pct"), yaxis_title=t("chart_frequency"))
        st.plotly_chart(apply_dark_theme(fig_port_dist), use_container_width=True, key="risk_return_distribution_portfolio")

    with tab4:
        if len(etf_prices.columns) >= 2:
            corr = correlation_matrix(etf_prices)
            fig_corr = correlation_heatmap(corr)
            st.plotly_chart(fig_corr, use_container_width=True, key="risk_correlation_heatmap")
            st.caption(t("risk_correlation_vs_overlap_note"))

            # Holdings Overlap (M3): a DIFFERENT measure than the return
            # correlation above -- see RISK_METHODOLOGY["correlation_vs_overlap"].
            # Opt-in (unchecked by default) since it fetches each ETF's
            # underlying holdings on demand rather than automatically on
            # every page load/rerun.
            show_overlap = st.checkbox(t("risk_show_holdings_overlap"), value=False, key="risk_show_overlap_cb")
            if show_overlap:
                with st.spinner(t("risk_loading_holdings_overlap")):
                    overlap = holdings_overlap_matrix(list(etf_prices.columns))
                overlap_rows = []
                for (pair_a, pair_b), res in overlap.items():
                    if res["available"]:
                        overlap_rows.append({
                            t("risk_col_pair"): f"{pair_a} / {pair_b}",
                            t("risk_col_overlap_score"): f"{res['overlap_score']:.1%}",
                            t("risk_col_shared_holdings"): res["shared_holdings_count"],
                        })
                    else:
                        overlap_rows.append({
                            t("risk_col_pair"): f"{pair_a} / {pair_b}",
                            t("risk_col_overlap_score"): t("risk_overlap_unavailable"),
                            t("risk_col_shared_holdings"): "—",
                        })
                if overlap_rows:
                    st.dataframe(
                        pd.DataFrame(overlap_rows).set_index(t("risk_col_pair")),
                        use_container_width=True,
                    )
        else:
            st.info(t("risk_select_2_correlation"))

    with tab5:
        # Risk contribution
        if len(etf_prices.columns) >= 2:
            cov = covariance_matrix(etf_prices).values.copy()
            cov += np.eye(len(etf_prices.columns)) * 1e-8
            port_vol_val = np.sqrt(weights_arr @ cov @ weights_arr)
            if port_vol_val > 0:
                marginal = cov @ weights_arr / port_vol_val
                risk_contrib = weights_arr * marginal
                risk_contrib_pct = risk_contrib / risk_contrib.sum()

                fig_rc = go.Figure(go.Bar(
                    x=etf_prices.columns.tolist(),
                    y=risk_contrib_pct * 100,
                    marker_color=CHART_COLORS[:len(etf_prices.columns)],
                    hovertemplate=f"<b>%{{x}}</b><br>{t('chart_risk_contribution_pct')}: %{{y:.2f}}%<extra></extra>"
                ))
                fig_rc.update_layout(title=t("chart_risk_contribution_by_etf"),
                                      xaxis_title=t("chart_etf"), yaxis_title=t("chart_risk_contribution_pct"))
                st.plotly_chart(apply_dark_theme(fig_rc), use_container_width=True, key="risk_contribution_bar")

# ── Portfolio Concentration (M3) ─────────────────────────────────────────
# Deliberately SEPARATE from this page's own ad-hoc weight sliders above
# (which let you explore risk for ANY ETF/weight combination): concentration
# and effective-holdings metrics must be verifiable against the canonical
# current_portfolio built in Portfolio Optimizer, never a different,
# page-local weight source -- see RISK_METHODOLOGY["concentration"].
if current_portfolio and current_portfolio.get("weights"):
    section_header(t("risk_canonical_concentration_title"), t("risk_canonical_concentration_sub"))
    _canon_diag = concentration_from_weights(current_portfolio["weights"])
    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    with ccol1:
        st.markdown(metric_card_html(
            t("opt_col_largest_position"),
            f"{_canon_diag['largest_ticker']} {_canon_diag['largest_weight']:.2%}",
            color=COLORS["primary"],
        ), unsafe_allow_html=True)
    with ccol2:
        st.markdown(metric_card_html(
            t("opt_diag_top2_concentration"), f"{_canon_diag['top2_concentration']:.2%}", color=COLORS["purple"],
        ), unsafe_allow_html=True)
    with ccol3:
        st.markdown(metric_card_html(
            t("opt_diag_effective_holdings"),
            f"{_canon_diag['effective_holdings']:.2f} / {_canon_diag['selected_holdings']}",
            color=COLORS["cyan"],
        ), unsafe_allow_html=True)
    with ccol4:
        st.markdown(metric_card_html(
            t("opt_diag_active_etfs"),
            f"{_canon_diag['active_holdings']} / {_canon_diag['selected_holdings']}",
            color=COLORS["warning"],
        ), unsafe_allow_html=True)
    st.caption(t("risk_canonical_concentration_note"))

# ── Stress Tests ──────────────────────────────────────────────────────────────
section_header(t("risk_stress_test_title"), t("risk_stress_test_caption"))
st.caption(t("risk_stress_methodology_note"))

scenario_col = t("risk_col_scenario")
provenance_col = t("risk_col_provenance")
shock_col = t("risk_col_market_shock")
beta_col = t("risk_col_portfolio_beta")
impact_col = t("risk_col_estimated_impact")
dollar_impact_col = t("risk_col_impact_10k")

_PROVENANCE_LABEL_KEY = {
    "hypothetical": "risk_provenance_hypothetical",
    "historical": "risk_provenance_historical",
}

# Shock magnitudes and scenario identity are UNCHANGED from before this task
# (see src/risk_analytics.py's STRESS_SCENARIOS docstring) -- this task only
# adds the Provenance column and methodology disclosure above.
if bench_prices is not None:
    b_val = beta(port_prices, bench_prices)
else:
    b_val = 1.0

stress_rows = []
for _scenario in STRESS_SCENARIOS.values():
    market_shock = _scenario["shock"]
    port_impact = market_shock * b_val
    dollar_impact = port_impact * 10000  # Assume $10,000 portfolio
    stress_rows.append({
        scenario_col: t(_scenario["i18n_key"]),
        provenance_col: t(_PROVENANCE_LABEL_KEY[_scenario["provenance"]]),
        shock_col: f"{market_shock:.0%}",
        beta_col: f"{b_val:.2f}",
        impact_col: f"{port_impact:.2%}",
        dollar_impact_col: f"${dollar_impact:,.0f}",
    })

stress_df = pd.DataFrame(stress_rows).set_index(scenario_col)
with chart_card(t("risk_stress_test_impact_card")):
    st.dataframe(
        style_signed_columns(stress_df, [impact_col, dollar_impact_col]),
        use_container_width=True,
    )
    for _scenario in STRESS_SCENARIOS.values():
        st.caption(f"**{t(_scenario['i18n_key'])}** ({t(_PROVENANCE_LABEL_KEY[_scenario['provenance']])}) — {t(_scenario['note_i18n_key'])}")

disclaimer_box()
render_footer()
