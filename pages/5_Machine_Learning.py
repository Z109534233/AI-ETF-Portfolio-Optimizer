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
from src.etf_database import get_countries, get_tickers_by_country, to_yahoo_symbol, rename_yahoo_columns
from src.machine_learning import run_ml_pipeline, DISCLAIMER
from src.charts import (
    feature_importance_chart, confusion_matrix_chart, apply_dark_theme, CHART_COLORS
)
from src.utils import load_css, page_header, disclaimer_box, metric_card_html, get_date_range_defaults
from src.ui import (
    render_sidebar_nav, render_sidebar_footer, section_header, chart_card, render_footer, error_state,
    region_selector, region_etf_options, chart_caption, ai_interpret_button, results_hero,
)
from src.theme import COLORS
from src.i18n import t, t_model_type, t_country, MODEL_TYPE_KEYS
from src.auth import require_login

st.set_page_config(
    page_title="Machine Learning | AI ETF Portfolio Optimizer",
    page_icon="🤖",
    layout="wide"
)

require_login()

load_css()

page_header(t("ml_title"), t("ml_subtitle"))

st.warning(t("ml_disclaimer_banner"))

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.markdown(f"### {t('ml_sidebar_settings')}")

    # ── Region Selector (Global ETF Support) ─────────────────────────────
    # "United States" preserves the exact original ETF list (DEFAULT_ETFS)
    # so existing behavior is unchanged unless the user explicitly picks a
    # different region. The ML pipeline itself (src/machine_learning.py)
    # is fully ticker-agnostic -- it only ever sees a price Series -- so no
    # model code needs to change to support new markets.
    #
    # region_selector() (src/ui.py) is the SAME shared helper used by ETF
    # Analysis, Risk Analytics, and AI Advisor: it reads and writes one
    # canonical st.session_state["selected_region"], so picking a market
    # here is immediately reflected on those other pages too. This page's
    # own ETF choice below stays page-local (a single st.selectbox, not a
    # multiselect, so it isn't the same shape as the other pages' shared
    # multi-ETF selection) but is correctly re-scoped to whichever region
    # is globally selected.
    selected_region, ALL_REGIONS_LABEL = region_selector()
    etf_options = region_etf_options(selected_region, ALL_REGIONS_LABEL)

    selected_etf = st.selectbox(
        t("field_select_etfs"),
        options=etf_options,
        index=0,
        help=t("ml_select_etf_help"),
        key=f"ml_etf_select_{selected_region}",
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
# Priority-0 stale-result fix (Issue #20 section 6A): a trained result must
# be bound to EVERY input that changes what training actually produces --
# ticker, date range, model, test size, and the lookahead horizon (fixed at
# 1 here since this page has no lookahead control yet, but included so it
# can never silently drift if one is added later). LOOKAHEAD_PERIODS below
# is the single source of truth for both the fingerprint and the pipeline
# call -- they can never disagree.
LOOKAHEAD_PERIODS = 1
current_fingerprint = (
    selected_etf, str(start_date), str(end_date), model_type,
    round(test_size, 4), LOOKAHEAD_PERIODS,
)

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None
if "ml_ticker" not in st.session_state:
    st.session_state.ml_ticker = None
if "ml_fingerprint" not in st.session_state:
    st.session_state.ml_fingerprint = None

if run_btn or st.session_state.ml_result is None:
    with st.spinner(f"{t('msg_downloading_market_data')} ({t_model_type(model_type)})"):
        yahoo_ticker = to_yahoo_symbol(selected_etf)
        raw_prices = download_etf_data([yahoo_ticker], str(start_date), str(end_date))
        if raw_prices.empty:
            error_state(t("msg_no_price_data_title"), t("msg_no_price_data_desc"))
            st.stop()

        prices_df = clean_price_data(raw_prices)
        prices_df = rename_yahoo_columns(prices_df)
        if selected_etf not in prices_df.columns:
            error_state(t("ml_ticker_not_found_title"), t("ml_ticker_not_found_desc", ticker=selected_etf))
            st.stop()

        prices = prices_df[selected_etf].dropna()

        result = run_ml_pipeline(
            prices, model_type=model_type, test_size=test_size, lookahead=LOOKAHEAD_PERIODS,
        )
        st.session_state.ml_result = result
        st.session_state.ml_ticker = selected_etf
        st.session_state.ml_prices = prices
        st.session_state.ml_fingerprint = current_fingerprint

result = st.session_state.ml_result
if result is None:
    st.info(t("msg_configure_and_run", action=t("btn_train_model")))
    st.stop()

if result.get("error"):
    error_state(t("ml_training_failed_title"), str(result["error"]))
    st.stop()

# Stale-result guard: if any sidebar input changed since the last successful
# training run WITHOUT clicking "Train Model" again, the OLD result must
# never keep being shown as if it were current -- see PRODUCT SPEC section
# 6A ("Inputs have changed. Please retrain the model.").
if st.session_state.ml_fingerprint is not None and st.session_state.ml_fingerprint != current_fingerprint:
    st.warning(t("ml_inputs_changed_retrain"))
    st.stop()

ticker_used = st.session_state.ml_ticker
metrics = result["metrics"]
feature_importance = result["feature_importance"]
y_pred = result["y_pred"]
y_prob = result["y_prob"]
y_test = result["y_test"]
test_index = result["test_index"]
model_name = result["model_name"]

# ── Results Hero (Issue #24 item 3) ─────────────────────────────────────────
# The FIRST thing visible once a model has actually been trained -- a
# strong, elevated, colored header that is unmistakably different from the
# training-setup controls in the sidebar, so it's immediately clear "this
# is the answer" rather than just another control. The existing dynamic
# model/ticker/train-test disclosure below stays as a section_header
# (weaker, sub-section-within-results styling) so none of that detail is
# lost -- it's just no longer the strongest visual element on the page.
results_hero(t("ml_results_hero_title"), t("ml_results_hero_subtitle"))

# ── KPI Cards ─────────────────────────────────────────────────────────────────
section_header(t("ml_results_title", model=t_model_type(model_name), ticker=ticker_used),
               t("ml_results_sub", train=result["train_size"], test=result["test_size"]))

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(metric_card_html(t("metric_accuracy"), f"{metrics['Accuracy']:.2%}", color=COLORS["primary"]), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html(t("metric_precision"), f"{metrics['Precision']:.2%}", color=COLORS["success"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html(t("metric_recall"), f"{metrics['Recall']:.2%}", color=COLORS["purple"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html(t("metric_f1_score"), f"{metrics['F1 Score']:.2%}", color=COLORS["warning"]), unsafe_allow_html=True)
with col5:
    st.markdown(metric_card_html(
        t("metric_baseline_accuracy"), f"{result['baseline_accuracy']:.2%}",
        color=COLORS["text_muted"],
    ), unsafe_allow_html=True)

if metrics.get("ROC AUC") != "N/A":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(metric_card_html(t("metric_roc_auc"), f"{metrics['ROC AUC']:.4f}", color=COLORS["cyan"]), unsafe_allow_html=True)

# ── Baseline comparison (M5 -- Stage 5): every model's accuracy is judged
# against a simple "always predict the training set's majority class"
# baseline computed on the SAME held-out test set, not left as an abstract
# "~50% for a coin flip" claim buried in prose -- so a user can see
# concretely whether THIS model, on THIS ETF/period, beat a trivial rule.
_baseline_diff_pp = (metrics["Accuracy"] - result["baseline_accuracy"]) * 100
if _baseline_diff_pp > 0:
    st.caption(f"📊 {t('ml_beats_baseline', diff=f'{_baseline_diff_pp:.1f}')}")
else:
    st.caption(f"⚠️ {t('ml_does_not_beat_baseline', diff=f'{abs(_baseline_diff_pp):.1f}')}")
st.caption(t("ml_baseline_help"))

# ── Honest performance interpretation (Issue #20 section 6B): a model can
# beat the majority-class baseline on accuracy while its ROC-AUC sits at
# essentially chance level (~0.5), meaning it has little to no genuine
# ranking/discriminatory power on this test window -- accuracy alone can
# hide that. Surfaced only when ROC-AUC is actually available (binary
# classifiers with both classes present in y_test).
_roc_auc_val = metrics.get("ROC AUC")
if isinstance(_roc_auc_val, (int, float)) and abs(_roc_auc_val - 0.5) <= 0.05:
    st.warning(t(
        "ml_weak_discriminatory_power",
        diff=f"{abs(_baseline_diff_pp):.1f}", roc_auc=f"{_roc_auc_val:.3f}",
    ))

# ── Target / data-window disclosure (M5): the target definition and the
# actual out-of-sample test date range were previously never shown -- only
# observation COUNTS -- leaving "is this out-of-sample?" and "as of when?"
# unanswered. i18n keys ml_prediction/ml_next_day_direction existed but
# were dead (never referenced here); this replaces that gap outright.
with st.expander(t("ml_target_definition_label"), expanded=False):
    st.markdown(f"- **{t('ml_target_definition_label')}** — "
                f"{t('ml_target_definition_value', lookahead=result['lookahead_periods'])}")
    st.markdown(f"- **{t('ml_data_window_label')}** — "
                f"{t('ml_data_window_value', train_start=result['train_start'], train_end=result['train_end'], train=result['train_size'], test_start=result['test_start'], test_end=result['test_end'], test=result['test_size'])}")

# ── Charts ────────────────────────────────────────────────────────────────────
section_header(t("ml_diagnostics_title"))
with chart_card(t("ml_model_detail_card")):
    tab1, tab2, tab3, tab4 = st.tabs([
        t("ml_tab_feature_importance"), t("ml_tab_confusion_matrix"), t("ml_tab_predictions"), t("ml_tab_model_limitations")
    ])

    with tab1:
        # Accurate labeling/disclosure (Issue #20 section 6C): Random
        # Forest's feature_importances_ is impurity-based (mean decrease in
        # Gini impurity across the fitted trees) and Logistic Regression's
        # here is absolute coefficient magnitude on standardized features --
        # neither implies the feature CAUSES price direction, only that the
        # fitted model relied on it.
        _fi_title_key = "ml_fi_title_rf" if model_name == "Random Forest" else "ml_fi_title_lr"
        st.markdown(f"**{t(_fi_title_key)}**")
        st.caption(t("ml_fi_causality_disclaimer"))
        fig_fi = feature_importance_chart(feature_importance)
        st.plotly_chart(fig_fi, use_container_width=True, key="ml_feature_importance")
        chart_caption(t("ml_caption_feature_importance"))
        _fi_context_text = (
            f"Model: {t_model_type(model_name)}. Ticker: {ticker_used}. "
            "Top features by importance score: " +
            ", ".join(f"{feat} ({score:.4f})" for feat, score in feature_importance.head(5).items())
        )
        ai_interpret_button("ml_ai_feature_importance", st.session_state, _fi_context_text)

        st.markdown(f"**{t('ml_top_10_features')}**")
        fi_df = feature_importance.head(10).reset_index()
        fi_df.columns = [t("chart_feature"), t("chart_importance_score")]
        fi_df[t("chart_importance_score")] = fi_df[t("chart_importance_score")].apply(lambda x: f"{x:.4f}")
        st.dataframe(fi_df.set_index(t("chart_feature")), use_container_width=True)
        chart_caption(t("ml_caption_feature_table"))

    with tab2:
        cm = metrics["Confusion Matrix"]
        fig_cm = confusion_matrix_chart(cm)
        st.plotly_chart(fig_cm, use_container_width=True, key="ml_confusion_matrix")
        chart_caption(t("ml_caption_confusion_matrix"))
        _tn, _fp, _fn, _tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        _cm_context_text = (
            f"Confusion matrix on the held-out test set: "
            f"True Negative (correctly predicted down)={_tn}, "
            f"False Positive (predicted up, actually down)={_fp}, "
            f"False Negative (predicted down, actually up)={_fn}, "
            f"True Positive (correctly predicted up)={_tp}. "
            f"Accuracy={metrics['Accuracy']:.2%}, Precision={metrics['Precision']:.2%}, Recall={metrics['Recall']:.2%}."
        )
        ai_interpret_button("ml_ai_confusion_matrix", st.session_state, _cm_context_text)

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
                chart_caption(t("ml_caption_predictions"))

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
            chart_caption(t("ml_caption_prediction_probability"))
            if not prob_series.empty:
                _prob_context_text = (
                    f"Model: {t_model_type(model_name)}. Ticker: {ticker_used}. "
                    f"Most recent predicted probability of an upward move: {prob_series.iloc[-1]:.2%} "
                    f"(test period ending {prob_series.index[-1].date()}). "
                    f"Test accuracy: {metrics['Accuracy']:.2%}, baseline accuracy: {result['baseline_accuracy']:.2%}."
                )
                ai_interpret_button("ml_ai_prediction_probability", st.session_state, _prob_context_text)

    with tab4:
        st.markdown(f"### {t('ml_model_limitations_title')}")
        st.markdown(t("ml_model_limitations_body"))

disclaimer_box()
render_footer()
