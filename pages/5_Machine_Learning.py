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
from src.ui import render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer, error_state
from src.theme import COLORS
from src.i18n import t, t_model_type, MODEL_TYPE_KEYS

st.set_page_config(
    page_title="Machine Learning | AI ETF Portfolio Optimizer",
    page_icon="🤖",
    layout="wide"
)

load_css()

page_header(t("ml_title"), t("ml_subtitle"))

st.warning(t("ml_disclaimer_banner"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('ml_sidebar_settings')}")

    selected_etf = st.selectbox(
        t("field_select_etfs"),
        options=DEFAULT_ETFS,
        index=0,
        help=t("ml_select_etf_help")
    )

    custom_ticker = st.text_input(t("ml_custom_ticker"), placeholder="e.g. ARKK").upper().strip()
    if custom_ticker:
        selected_etf = custom_ticker

    default_start, default_end = get_date_range_defaults()
    start_date = st.date_input(t("field_start_date"), value=default_start)
    end_date = st.date_input(t("field_end_date"), value=default_end)

    # Pre-resolve labels once (within a valid script context) rather than
    # passing a format_func that reads st.session_state on every invocation.
    _model_type_labels = {k: t_model_type(k) for k in MODEL_TYPE_KEYS}
    model_type = st.selectbox(
        t("ml_model_label"),
        list(MODEL_TYPE_KEYS.keys()),
        format_func=lambda x: _model_type_labels.get(x, x),
        help=t("ml_model_help")
    )

    test_size = st.slider(t("ml_test_set_size_pct"), 10, 40, 20, 5) / 100

    st.markdown("---")
    st.markdown(f"### {t('ml_features_used_title')}")
    st.markdown(t("ml_features_list"))

    run_btn = st.button(t("btn_train_model"), type="primary", use_container_width=True)

    render_sidebar_footer()

# ── Run ML Pipeline ───────────────────────────────────────────────────────────
if "ml_result" not in st.session_state:
    st.session_state.ml_result = None
if "ml_ticker" not in st.session_state:
    st.session_state.ml_ticker = None

if run_btn or st.session_state.ml_result is None:
    with st.spinner(f"{t('msg_downloading_market_data')} ({t_model_type(model_type)})"):
        raw_prices = download_etf_data([selected_etf], str(start_date), str(end_date))
        if raw_prices.empty:
            error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
            st.stop()

        prices_df = clean_price_data(raw_prices)
        if selected_etf not in prices_df.columns:
            error_state(t("ml_ticker_not_found_title"), t("ml_ticker_not_found_desc", ticker=selected_etf))
            st.stop()

        prices = prices_df[selected_etf].dropna()

        result = run_ml_pipeline(prices, model_type=model_type, test_size=test_size)
        st.session_state.ml_result = result
        st.session_state.ml_ticker = selected_etf
        st.session_state.ml_prices = prices

result = st.session_state.ml_result
if result is None:
    st.info(t("msg_configure_and_run", action=t("btn_train_model")))
    st.stop()

if result.get("error"):
    error_state(t("ml_training_failed_title"), str(result["error"]))
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
section_header(t("ml_results_title", model=t_model_type(model_name), ticker=ticker_used),
               t("ml_results_sub", train=result["train_size"], test=result["test_size"]))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html(t("metric_accuracy"), f"{metrics['Accuracy']:.2%}", color=COLORS["primary"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_precision"), f"{metrics['Precision']:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_recall"), f"{metrics['Recall']:.2%}", color=COLORS["purple"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_f1_score"), f"{metrics['F1 Score']:.2%}", color=COLORS["warning"]), unsafe_allow_html=True)

if metrics.get("ROC AUC") != "N/A":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(metric_card_html(t("metric_roc_auc"), f"{metrics['ROC AUC']:.4f}", color=COLORS["cyan"]), unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("ml_diagnostics_title"))
with chart_card(t("ml_model_detail_card")):
    tab1, tab2, tab3, tab4 = st.tabs([
        t("ml_tab_feature_importance"), t("ml_tab_confusion_matrix"), t("ml_tab_predictions"), t("ml_tab_model_limitations")
    ])

    with tab1:
        fig_fi = feature_importance_chart(feature_importance)
        st.plotly_chart(fig_fi, use_container_width=True, key="ml_feature_importance")

        st.markdown(f"**{t('ml_top_10_features')}**")
        fi_df = feature_importance.head(10).reset_index()
        fi_df.columns = [t("chart_feature"), t("chart_importance_score")]
        fi_df[t("chart_importance_score")] = fi_df[t("chart_importance_score")].apply(lambda x: f"{x:.4f}")
        st.dataframe(fi_df.set_index(t("chart_feature")), use_container_width=True)

    with tab2:
        cm = metrics["Confusion Matrix"]
        fig_cm = confusion_matrix_chart(cm)
        st.plotly_chart(fig_cm, use_container_width=True, key="ml_confusion_matrix")

        st.markdown(f"**{t('ml_confusion_matrix_interpretation')}**")
        st.markdown(t("ml_confusion_matrix_table"))

    with tab3:
        # Actual vs predicted direction
        if len(test_index) > 0:
            prices_series = st.session_state.ml_prices
            test_prices = prices_series.loc[prices_series.index.isin(test_index)]

            if not test_prices.empty:
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=test_prices.index, y=test_prices.values,
                    name=ticker_used, line=dict(color=COLORS["text_muted"], width=1.5)
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
                        name=t("chart_correct_up"), marker=dict(color=COLORS["success"], size=6, symbol="triangle-up")
                    ))
                if not cp_down.empty:
                    fig_pred.add_trace(go.Scatter(
                        x=cp_down.index, y=cp_down.values, mode="markers",
                        name=t("chart_correct_down"), marker=dict(color=COLORS["primary"], size=6, symbol="triangle-down")
                    ))
                if not cp_wrong.empty:
                    fig_pred.add_trace(go.Scatter(
                        x=cp_wrong.index, y=cp_wrong.values, mode="markers",
                        name=t("chart_incorrect"), marker=dict(color=COLORS["danger"], size=6, symbol="x")
                    ))

                fig_pred.update_layout(title=t("chart_actual_vs_predicted"),
                                        xaxis_title=t("chart_date"), yaxis_title=t("chart_price"))
                st.plotly_chart(apply_dark_theme(fig_pred), use_container_width=True, key="ml_predictions")

            # Prediction probability over time
            prob_series = pd.Series(y_prob, index=test_index[:len(y_prob)])
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=prob_series.index, y=prob_series.values,
                name="P(Up)", line=dict(color=COLORS["primary"], width=1.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
            ))
            fig_prob.add_hline(y=0.5, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.6)
            fig_prob.update_layout(title=t("chart_prediction_probability_up"),
                                    xaxis_title=t("chart_date"), yaxis_title=t("chart_probability"),
                                    yaxis=dict(range=[0, 1]))
            st.plotly_chart(apply_dark_theme(fig_prob), use_container_width=True, key="ml_prediction_probability")

    with tab4:
        st.markdown(f"### {t('ml_model_limitations_title')}")
        st.markdown(t("ml_model_limitations_body"))

disclaimer_box()
render_footer()
