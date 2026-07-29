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
from src.i18n import t, t_opt_method, OPTIMIZATION_METHOD_KEYS

st.set_page_config(
    page_title="Portfolio Optimizer | AI ETF Portfolio Optimizer",
    page_icon="⚡",
    layout="wide"
)

load_css()
init_database()

page_header(t("opt_title"), t("opt_subtitle"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('opt_sidebar_settings')}")

    selected_etfs = st.multiselect(
        t("field_select_etfs"),
        options=DEFAULT_ETFS,
        default=["VOO", "QQQ", "BND", "GLD", "VNQ"],
        help=t("opt_select_etfs_help")
    )

    custom_ticker = st.text_input(t("field_add_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker and custom_ticker not in selected_etfs:
        selected_etfs.append(custom_ticker)

    investment_amount = st.number_input(t("field_investment_amount_usd"), min_value=100.0,
                                         max_value=10_000_000.0, value=10000.0, step=500.0)

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("field_start_date"), value=default_start)
    end_date = st.date_input(t("field_end_date"), value=default_end)

    risk_free_rate = st.slider(t("field_risk_free_rate_pct"), 0.0, 10.0, 5.0, 0.25) / 100

    st.markdown("---")
    st.markdown(f"### {t('opt_constraints_label')}")
    # Pre-resolve labels once (within a valid script context) rather than
    # passing a format_func that reads st.session_state on every invocation.
    _opt_method_labels = {k: t_opt_method(k) for k in OPTIMIZATION_METHOD_KEYS}
    optimization_method = st.selectbox(
        t("opt_method_label"),
        list(OPTIMIZATION_METHOD_KEYS.keys()),
        format_func=lambda x: _opt_method_labels.get(x, x),
    )

    min_weight = st.slider(t("opt_min_weight"), 0.0, 20.0, 0.0, 1.0) / 100
    max_weight = st.slider(t("opt_max_weight"), 10.0, 100.0, 100.0, 5.0) / 100
    allow_short = st.checkbox(t("opt_allow_short"), value=False)

    target_return_pct = None
    if optimization_method == "Target Return":
        target_return_pct = st.slider(t("opt_target_return_pct"), 1.0, 30.0, 10.0, 0.5) / 100

    n_simulations = st.slider(t("opt_mc_simulations"), 1000, 10000, 5000, 500)

    run_btn = st.button(t("btn_run_optimization"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Validation ────────────────────────────────────────────────────────────────
if len(selected_etfs) < 2:
    st.warning(t("msg_select_two_etfs"))
    st.stop()

if start_date >= end_date:
    st.error(t("msg_start_before_end"))
    st.stop()

# ── State Management ──────────────────────────────────────────────────────────
# `run_inputs` fingerprints everything the optimization result depends on.
# Without this, changing the ETF selection, date range, or optimization
# method WITHOUT re-clicking "Run Optimization" would silently keep
# showing stale results computed from a previous, different selection
# (st.button() only returns True on the exact rerun it was clicked in;
# every other widget change also triggers a rerun but leaves run_btn
# False, and the old code only recomputed when run_btn was True or no
# result existed yet). We now recompute whenever the actual inputs change.
run_inputs = (
    tuple(sorted(selected_etfs)), str(start_date), str(end_date),
    optimization_method, round(min_weight, 6), round(max_weight, 6),
    allow_short, target_return_pct,
)

if "opt_result" not in st.session_state:
    st.session_state.opt_result = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "opt_run_inputs" not in st.session_state:
    st.session_state.opt_run_inputs = None

inputs_changed = run_inputs != st.session_state.opt_run_inputs

if run_btn or inputs_changed or st.session_state.opt_result is None:
    with st.spinner(t("msg_running_optimization")):
        raw_prices = download_etf_data(selected_etfs, str(start_date), str(end_date))
        if raw_prices.empty:
            error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
            st.stop()

        prices_df = clean_price_data(raw_prices)
        prices_df = prices_df[[tk for tk in selected_etfs if tk in prices_df.columns]]

        # ── Validate prices_df ──────────────────────────────────────────
        # clean_price_data() now drops any ticker with zero valid data, so
        # a ticker that failed to download (common on Streamlit Cloud when
        # Yahoo Finance rate-limits cloud IPs) is silently absent from
        # prices_df.columns rather than lingering as an all-NaN column.
        # Report exactly which requested tickers are missing.
        missing_tickers = [tk for tk in selected_etfs if tk not in prices_df.columns]
        if missing_tickers:
            st.warning(
                f"No usable price data for: {', '.join(missing_tickers)}. "
                "These tickers were excluded from the optimization."
            )

        if prices_df.empty or len(prices_df.columns) < 2:
            error_state(
                t("msg_no_price_data_title"),
                "At least 2 ETFs with valid price data are required to run "
                "portfolio optimization. Try different tickers or a wider "
                "date range."
            )
            st.stop()

        if len(prices_df) < 20:
            error_state(
                t("msg_no_price_data_title"),
                f"Only {len(prices_df)} trading day(s) of overlapping data "
                "were found — at least 20 are required for a meaningful "
                "optimization. Widen the date range."
            )
            st.stop()

        # ── Validate returns_df ─────────────────────────────────────────
        # how="all" (not the pandas default how="any"): a single ticker
        # missing one date must not wipe out that date for every other
        # ticker too. See covariance_matrix() for the full rationale.
        returns_df_check = prices_df.pct_change(fill_method=None).dropna(how="all")
        if returns_df_check.empty or len(returns_df_check) < 10:
            error_state(
                t("msg_no_price_data_title"),
                "Not enough overlapping daily returns could be computed "
                "from the downloaded price data. Widen the date range or "
                "choose different ETFs."
            )
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
            st.warning(t("opt_note_prefix", error=result["error"]))

        st.session_state.opt_result = result
        st.session_state.prices_df = prices_df
        st.session_state.opt_run_inputs = run_inputs

result = st.session_state.opt_result
prices_df = st.session_state.prices_df

if result is None or prices_df is None or prices_df.empty:
    st.info(t("msg_configure_and_run", action=t("btn_run_optimization")))
    st.stop()

weights = result["weights"]
exp_ret = result["expected_return"]
exp_vol = result["expected_volatility"]
sharpe = result["sharpe_ratio"]
div_ratio = result.get("diversification_ratio", 1.0)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("opt_results_title"))
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(metric_card_html(t("metric_expected_annual_return"), f"{exp_ret:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_expected_volatility"), f"{exp_vol:.2%}", color=COLORS["danger"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_sharpe_ratio"), f"{sharpe:.2f}", color=COLORS["primary"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_diversification_ratio"), f"{div_ratio:.2f}", color=COLORS["purple"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html(t("metric_method"), t_opt_method(optimization_method), color=COLORS["warning"]), unsafe_allow_html=True)

# ── Allocation Table & Donut ──────────────────────────────────────────────────
section_header(t("opt_allocation_title"))
col_left, col_right = st.columns([1, 1])

with col_left:
    with chart_card(t("opt_allocation_table_card"), t("opt_allocation_table_holdings", count=len(weights))):
        alloc_df = weights_to_dataframe(weights, investment_amount)
        st.dataframe(alloc_df[["Ticker", "Weight", "Allocation ($)"]].style.hide(axis="index"),
                     use_container_width=True)

with col_right:
    with chart_card(t("opt_allocation_breakdown_card"), t_opt_method(optimization_method)):
        fig_donut = allocation_donut_chart(weights, "")
        st.plotly_chart(fig_donut, use_container_width=True, key="opt_allocation_donut")

# ── Efficient Frontier ────────────────────────────────────────────────────────
section_header(t("opt_efficient_frontier_title"), t("opt_efficient_frontier_sub", count=f"{n_simulations:,}"))

n_tickers = len(prices_df.columns)

with st.spinner(t("msg_running_optimization")):
    # how="all" (not the pandas default how="any"): a single ticker missing
    # one date must not wipe out that date for every other ticker too.
    returns_df = prices_df.pct_change(fill_method=None).dropna(how="all")
    mean_returns = returns_df.mean().values
    cov_df = covariance_matrix(prices_df)
    cov = cov_df.values

    # Defensive validation: covariance_matrix() guarantees cov.shape ==
    # (n_tickers, n_tickers) and run_optimization() already validated this
    # same data upstream, so these checks should never fire in practice --
    # but if prices_df ever reaches here in an unexpected state (e.g. a
    # future code change, or a Yahoo Finance edge case not seen before),
    # we show a clear error instead of crashing on `cov += np.eye(...)`
    # with an opaque numpy shape-mismatch exception.
    if cov.shape != (n_tickers, n_tickers):
        error_state(
            t("msg_no_price_data_title"),
            f"Internal error: covariance matrix shape {cov.shape} does not "
            f"match {n_tickers} selected ETFs. Please re-run the optimization."
        )
        st.stop()

    if not np.isfinite(cov).all():
        # A NaN on the diagonal means that ticker has no valid data at all;
        # a NaN only off the diagonal means two otherwise-fine tickers just
        # don't share any trading dates. Distinguish these so the message
        # names the actual problem ticker(s) instead of every ticker.
        diag_nan = ~np.isfinite(np.diag(cov))
        if diag_nan.any():
            bad_tickers = [prices_df.columns[i] for i in range(n_tickers) if diag_nan[i]]
            error_state(
                t("msg_no_price_data_title"),
                f"No valid price data for: {', '.join(bad_tickers)}. "
                "Remove these tickers or widen the date range, then re-run "
                "the optimization."
            )
        else:
            error_state(
                t("msg_no_price_data_title"),
                "Some selected ETFs have no overlapping trading dates with "
                "each other. Widen the date range or choose ETFs with more "
                "shared trading history."
            )
        st.stop()

    cov += np.eye(n_tickers) * 1e-8

    mc_df = monte_carlo_simulation(mean_returns, cov, n_simulations, risk_free_rate)

with chart_card(t("opt_efficient_frontier_card")):
    fig_ef = efficient_frontier_chart(mc_df, weights, None, mean_returns, cov)
    st.plotly_chart(fig_ef, use_container_width=True, key="opt_efficient_frontier")

# ── Backtest ──────────────────────────────────────────────────────────────────
section_header(t("opt_backtest_title"), t("opt_backtest_sub", method=t_opt_method(optimization_method)))

backtest_df = backtest_portfolio(prices_df, weights, investment_amount)
equal_weights = {tk: 1.0 / len(weights) for tk in weights.keys()}
equal_backtest_df = backtest_portfolio(prices_df, equal_weights, investment_amount)

if not backtest_df.empty:
    import plotly.graph_objects as go
    with chart_card(t("opt_backtest_card")):
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=backtest_df.index, y=backtest_df["Portfolio Value"],
            name=t_opt_method(optimization_method), line=dict(color=COLORS["primary"], width=2.5)
        ))
        if not equal_backtest_df.empty:
            fig_bt.add_trace(go.Scatter(
                x=equal_backtest_df.index, y=equal_backtest_df["Portfolio Value"],
                name=t("chart_equal_weight"), line=dict(color=COLORS["text_muted"], width=1.5, dash="dash")
            ))
        fig_bt.update_layout(title=t("chart_portfolio_backtest_vs_equal"),
                              xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"))
        st.plotly_chart(apply_dark_theme(fig_bt), use_container_width=True, key="opt_backtest_growth")

    # Drawdown comparison
    with chart_card(t("opt_drawdown_comparison_card")):
        fig_dd = go.Figure()
        dd = drawdown_series(backtest_df["Portfolio Value"]) * 100
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy",
                                     name=t_opt_method(optimization_method), line=dict(color=COLORS["danger"], width=1.5)))
        if not equal_backtest_df.empty:
            dd_eq = drawdown_series(equal_backtest_df["Portfolio Value"]) * 100
            fig_dd.add_trace(go.Scatter(x=dd_eq.index, y=dd_eq, fill="tozeroy",
                                         name=t("chart_equal_weight"), line=dict(color=COLORS["text_muted"], width=1.5),
                                         fillcolor="rgba(148,163,184,0.1)"))
        fig_dd.update_layout(title=t("chart_drawdown_comparison_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
        st.plotly_chart(apply_dark_theme(fig_dd), use_container_width=True, key="opt_backtest_drawdown")

    # Backtest metrics
    bt_metrics = {
        t("metric_total_return"): f"{backtest_df['Cumulative Return'].iloc[-1]:.2%}",
        t("metric_annualized_return"): f"{annualized_return(backtest_df['Portfolio Value']):.2%}",
        t("metric_annualized_volatility"): f"{annualized_volatility(backtest_df['Portfolio Value']):.2%}",
        t("metric_sharpe_ratio"): f"{sharpe_ratio(backtest_df['Portfolio Value'], risk_free_rate):.2f}",
        t("metric_maximum_drawdown"): f"{maximum_drawdown(backtest_df['Portfolio Value']):.2%}",
        t("metric_final_value"): f"${backtest_df['Portfolio Value'].iloc[-1]:,.2f}",
    }
    st.markdown(f"**{t('opt_backtest_summary')}**")
    cols = st.columns(len(bt_metrics))
    for i, (k, v) in enumerate(bt_metrics.items()):
        with cols[i]:
            st.metric(k, v)

# ── Save & Download ───────────────────────────────────────────────────────────
section_header(t("opt_save_export_title"))

col1, col2, col3 = st.columns(3)

with col1:
    portfolio_name = st.text_input(t("field_portfolio_name"), value=f"Portfolio_{optimization_method.replace(' ', '_')}")
    notes = st.text_area(t("field_notes_optional"), height=80)
    if st.button(t("btn_save_portfolio"), type="primary"):
        # NOTE: optimization_method is stored in English (the raw selectbox
        # value) so it stays consistent regardless of which language was
        # active when saved; it is translated only at display time via
        # t_opt_method() wherever it is shown (e.g. Portfolio History page).
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
            st.success(t("opt_portfolio_saved_success", name=portfolio_name))
        else:
            st.error(t("opt_portfolio_save_failed"))

with col2:
    alloc_csv = dataframe_to_csv(alloc_df)
    st.download_button(
        t("btn_download_allocation_csv"),
        alloc_csv, "portfolio_allocation.csv", "text/csv",
        use_container_width=True
    )

with col3:
    try:
        # NOTE: these dict keys are fixed English labels that
        # src/report_generator.py matches against internally to build the
        # PDF's metrics table — they must NOT be translated, or the PDF
        # report would silently lose all metric rows.
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
            t("btn_download_pdf_report"),
            pdf_bytes, f"{portfolio_name}.pdf", "application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning(t("opt_pdf_unavailable", error=e))

disclaimer_box()
render_footer()
