"""
Page 5: Machine Learning
Educational ML demonstration for ETF direction prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import download_etf_data, DEFAULT_ETFS
from src.data_cleaner import clean_price_data
from src.machine_learning import run_ml_pipeline, DISCLAIMER
from src.charts import (
    feature_importance_chart, confusion_matrix_chart, apply_dark_theme, CHART_COLORS
)
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults

st.set_page_config(
    page_title="Machine Learning | AI ETF Portfolio Optimizer",
    page_icon="🤖",
    layout="wide"
)

load_css()

page_header(
    "Machine Learning",
    "Educational ML demonstration for ETF return direction prediction",
    "🤖"
)

st.warning(f"**Educational Disclaimer**: {DISCLAIMER}")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Settings")

    selected_etf = st.selectbox(
        "Select ETF",
        options=DEFAULT_ETFS,
        index=0,
        help="Select one ETF for ML analysis"
    )

    custom_ticker = st.text_input("Custom Ticker", placeholder="e.g. ARKK").upper().strip()
    if custom_ticker:
        selected_etf = custom_ticker

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression"],
        help="Random Forest generally performs better for non-linear patterns"
    )

    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100

    st.markdown("---")
    st.markdown("### Features Used")
    st.markdown("""
    - Lagged returns (1, 2, 3, 5, 10 days)
    - SMA ratios (5, 10, 20, 50 days)
    - EMA ratios (12, 26 days)
    - RSI (14 days)
    - MACD (12/26/9)
    - Momentum (5, 10, 20 days)
    - Rolling volatility (10, 21 days)
    - Bollinger Band position
    """)

    run_btn = st.button("Train Model", type="primary", use_container_width=True)

# ── Run ML Pipeline ───────────────────────────────────────────────────────────
if "ml_result" not in st.session_state:
    st.session_state.ml_result = None
if "ml_ticker" not in st.session_state:
    st.session_state.ml_ticker = None

if run_btn or st.session_state.ml_result is None:
    with st.spinner(f"Downloading data and training {model_type}..."):
        raw_prices = download_etf_data([selected_etf], str(start_date), str(end_date))
        if raw_prices.empty:
            st.error("No price data available. Check ticker and date range.")
            st.stop()

        prices_df = clean_price_data(raw_prices)
        if selected_etf not in prices_df.columns:
            st.error(f"No data for {selected_etf}.")
            st.stop()

        prices = prices_df[selected_etf].dropna()

        result = run_ml_pipeline(prices, model_type=model_type, test_size=test_size)
        st.session_state.ml_result = result
        st.session_state.ml_ticker = selected_etf
        st.session_state.ml_prices = prices

result = st.session_state.ml_result
if result is None:
    st.info("Configure settings in the sidebar and click **Train Model**.")
    st.stop()

if result.get("error"):
    st.error(f"ML Error: {result['error']}")
    st.stop()

ticker_used = st.session_state.ml_ticker
metrics = result["metrics"]
feature_importance = result["feature_importance"]
y_pred = result["y_pred"]
y_prob = result["y_prob"]
y_test = result["y_test"]
test_index = result["test_index"]
model_name = result["model_name"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown(f"### {model_name} Results — {ticker_used}")
st.caption(f"Training samples: {result['train_size']} | Test samples: {result['test_size']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html("Accuracy", f"{metrics['Accuracy']:.2%}", color="#3B82F6"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Precision", f"{metrics['Precision']:.2%}", color="#10B981"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Recall", f"{metrics['Recall']:.2%}", color="#8B5CF6"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("F1 Score", f"{metrics['F1 Score']:.2%}", color="#F59E0B"), unsafe_allow_html=True)

if metrics.get("ROC AUC") != "N/A":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(metric_card_html("ROC AUC", f"{metrics['ROC AUC']:.4f}", color="#06B6D4"), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Confusion Matrix", "Predictions", "Model Limitations"])

with tab1:
    fig_fi = feature_importance_chart(feature_importance)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("**Top 10 Features**")
    fi_df = feature_importance.head(10).reset_index()
    fi_df.columns = ["Feature", "Importance Score"]
    fi_df["Importance Score"] = fi_df["Importance Score"].apply(lambda x: f"{x:.4f}")
    st.dataframe(fi_df.set_index("Feature"), use_container_width=True)

with tab2:
    cm = metrics["Confusion Matrix"]
    fig_cm = confusion_matrix_chart(cm)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**Confusion Matrix Interpretation**")
    st.markdown("""
    | | Predicted Down | Predicted Up |
    |---|---|---|
    | **Actual Down** | True Negative | False Positive |
    | **Actual Up** | False Negative | True Positive |
    """)

with tab3:
    # Actual vs predicted direction
    if len(test_index) > 0:
        prices_series = st.session_state.ml_prices
        test_prices = prices_series.loc[prices_series.index.isin(test_index)]

        if not test_prices.empty:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=test_prices.index, y=test_prices.values,
                name=ticker_used, line=dict(color="#9CA3AF", width=1.5)
            ))

            # Mark correct and incorrect predictions
            correct_up = [test_index[i] for i in range(len(y_test))
                          if y_pred[i] == 1 and y_test.iloc[i] == 1]
            correct_down = [test_index[i] for i in range(len(y_test))
                            if y_pred[i] == 0 and y_test.iloc[i] == 0]
            wrong = [test_index[i] for i in range(len(y_test))
                     if y_pred[i] != y_test.iloc[i]]

            def get_prices_at(idx_list):
                valid = [i for i in idx_list if i in test_prices.index]
                return test_prices.loc[valid] if valid else pd.Series(dtype=float)

            cp_up = get_prices_at(correct_up)
            cp_down = get_prices_at(correct_down)
            cp_wrong = get_prices_at(wrong)

            if not cp_up.empty:
                fig_pred.add_trace(go.Scatter(
                    x=cp_up.index, y=cp_up.values, mode="markers",
                    name="Correct Up", marker=dict(color="#10B981", size=6, symbol="triangle-up")
                ))
            if not cp_down.empty:
                fig_pred.add_trace(go.Scatter(
                    x=cp_down.index, y=cp_down.values, mode="markers",
                    name="Correct Down", marker=dict(color="#3B82F6", size=6, symbol="triangle-down")
                ))
            if not cp_wrong.empty:
                fig_pred.add_trace(go.Scatter(
                    x=cp_wrong.index, y=cp_wrong.values, mode="markers",
                    name="Incorrect", marker=dict(color="#EF4444", size=6, symbol="x")
                ))

            fig_pred.update_layout(title="Actual vs Predicted Direction (Test Set)",
                                    xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(apply_dark_theme(fig_pred), use_container_width=True)

        # Prediction probability over time
        prob_series = pd.Series(y_prob, index=test_index[:len(y_prob)])
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=prob_series.index, y=prob_series.values,
            name="P(Up)", line=dict(color="#3B82F6", width=1.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
        ))
        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="#9CA3AF", opacity=0.6)
        fig_prob.update_layout(title="Prediction Probability P(Up Direction)",
                                xaxis_title="Date", yaxis_title="Probability",
                                yaxis=dict(range=[0, 1]))
        st.plotly_chart(apply_dark_theme(fig_prob), use_container_width=True)

with tab4:
    st.markdown("### Model Limitations & Caveats")
    st.markdown("""
    **This machine learning model is for educational demonstration only.**

    **Known Limitations:**

    1. **Market Efficiency**: Financial markets are highly competitive. If a simple ML model could reliably predict price direction, arbitrage would quickly eliminate the opportunity.

    2. **Non-Stationarity**: Financial time series are non-stationary — patterns that worked historically may not persist in the future.

    3. **Overfitting Risk**: Even with time-series splitting, models may capture noise rather than signal. The test accuracy may not generalise to live trading.

    4. **Feature Engineering**: The features used (RSI, MACD, etc.) are based on price history only. Fundamental data, macroeconomic factors, and news sentiment are not included.

    5. **Transaction Costs**: Real trading involves bid-ask spreads, commissions, and market impact that are not modelled here.

    6. **Regime Changes**: A model trained on bull-market data may perform poorly in bear markets and vice versa.

    7. **Look-Ahead Bias**: Despite careful time-series splitting, subtle look-ahead bias may still exist in feature engineering.

    **Accuracy Benchmark**: A random classifier achieves ~50% accuracy. Meaningful outperformance requires sustained accuracy above 55% after transaction costs.

    **Conclusion**: Use this tool to understand ML concepts in finance, not to make real investment decisions.
    """)

disclaimer_box()
