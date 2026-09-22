"""
Bootstrap Weight Stability tests (Issue #50).

Two groups of deterministic tests, no live network access required (the
AppTest-driven group below monkeypatches src.data_loader.download_etf_data
with a synthetic fixture, same pattern already used elsewhere in
tests/test_portfolio_optimizer.py):

  1. Pure-function tests of
     src.portfolio_optimizer.bootstrap_max_sharpe_weight_stability() --
     200-attempt default, joint ROW resampling (not independent per-asset
     resampling), deterministic seeding, forwarded
     risk_free_rate/min_weight/max_weight/allow_short, failed-solve
     exclusion/counting, summary-statistic correctness, and the "no fake
     summary from too few successes" guard.
  2. i18n coverage for every new opt_bootstrap_* / opt_methodology_backtest_sep
     key (parity + non-empty in both languages), the Michaud (1989) citation
     wording (present, but never phrased as an implementation claim), the
     exact zh-TW/en methodology punctuation fix, and Streamlit AppTest
     checks that the on-demand panel only appears for Maximum Sharpe Ratio.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

import src.portfolio_optimizer as portfolio_optimizer_mod
from src.portfolio_optimizer import (
    bootstrap_max_sharpe_weight_stability,
    DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP,
    MIN_BOOTSTRAP_SUCCESSES_FOR_SUMMARY,
    bootstrap_boundary_metrics, BOOTSTRAP_INCLUSION_EPSILON,
)
from src.i18n import TRANSLATIONS, t, set_language


def _make_correlated_returns(n_days=300, seed=11):
    """3-asset synthetic daily return matrix: B is an EXACT linear function
    of A (B = 2*A, zero noise) on every single row, C is independent. This
    exact row-wise relationship is what the joint-row-resampling test below
    exploits -- it only survives a resample that keeps each day's A and B
    values together."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0006, 0.01, n_days)
    b = a * 2.0
    c = rng.normal(0.0002, 0.008, n_days)
    return pd.DataFrame({"A": a, "B": b, "C": c})


# ── Defaults ──────────────────────────────────────────────────────────────
def test_default_n_bootstrap_is_200_and_reported_as_attempted():
    assert DEFAULT_N_BOOTSTRAP == 200
    returns_df = _make_correlated_returns()
    result = bootstrap_max_sharpe_weight_stability(returns_df, seed=1)
    assert result["attempted"] == 200
    assert result["n_bootstrap"] == 200


def test_default_seed_is_42():
    assert DEFAULT_BOOTSTRAP_SEED == 42
    returns_df = _make_correlated_returns()
    result = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=10)
    assert result["seed"] == 42


# ── Joint ROW resampling (not independent per-asset resampling) ──────────
def test_joint_row_resampling_preserves_cross_asset_relationship(monkeypatch):
    returns_df = _make_correlated_returns(n_days=250, seed=5)
    seen_calls = []

    def _spy_optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight, allow_short):
        seen_calls.append((np.array(mean_returns, dtype=float), np.array(cov_matrix, dtype=float)))
        n = len(mean_returns)
        return np.array([1.0 / n] * n), True

    monkeypatch.setattr(portfolio_optimizer_mod, "optimize_max_sharpe", _spy_optimize_max_sharpe)
    bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=30, seed=9)

    assert len(seen_calls) == 30
    # Columns are ordered A, B, C (see _make_correlated_returns) -- B == 2*A
    # on every original row, so ANY joint-row resample must exactly preserve
    # mean(B) == 2*mean(A) and Cov(A,B) == 2*Var(A). Resampling each column
    # independently would draw different row indices per asset and break
    # this exact relationship almost certainly, across 30 independent draws.
    for mean_returns, cov in seen_calls:
        assert abs(mean_returns[1] - 2 * mean_returns[0]) < 1e-9
        assert abs(cov[0, 1] - 2 * cov[0, 0]) < 1e-6
        assert abs(cov[1, 1] - 4 * cov[0, 0]) < 1e-6


# ── Deterministic seeding ──────────────────────────────────────────────────
def test_same_seed_gives_identical_output():
    returns_df = _make_correlated_returns(n_days=260, seed=21)
    r1 = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=40, seed=42)
    r2 = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=40, seed=42)
    assert r1["weights_by_ticker"] == r2["weights_by_ticker"]
    assert r1["successful"] == r2["successful"]
    assert r1["summary"] == r2["summary"]


def test_different_seed_changes_resample_path():
    returns_df = _make_correlated_returns(n_days=260, seed=21)
    r1 = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=40, seed=42)
    r2 = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=40, seed=7)
    assert r1["weights_by_ticker"] != r2["weights_by_ticker"]


# ── Forwards the SAME current constraints / risk-free rate ───────────────
def test_forwards_current_risk_free_rate_and_constraints(monkeypatch):
    returns_df = _make_correlated_returns(n_days=200, seed=3)
    captured = []

    def _spy(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight, allow_short):
        captured.append((risk_free_rate, min_weight, max_weight, allow_short))
        n = len(mean_returns)
        return np.array([1.0 / n] * n), True

    monkeypatch.setattr(portfolio_optimizer_mod, "optimize_max_sharpe", _spy)
    bootstrap_max_sharpe_weight_stability(
        returns_df, risk_free_rate=0.037, min_weight=0.05, max_weight=0.65,
        allow_short=True, n_bootstrap=15, seed=2,
    )
    assert len(captured) == 15
    assert all(c == (0.037, 0.05, 0.65, True) for c in captured)


# ── Failed solves excluded from the distribution, counted accurately ─────
def test_failed_solves_are_excluded_and_counted_accurately(monkeypatch):
    returns_df = _make_correlated_returns(n_days=180, seed=8)
    call_counter = {"n": 0}

    def _spy(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight, allow_short):
        call_counter["n"] += 1
        n = len(mean_returns)
        converged = call_counter["n"] % 3 != 0  # every 3rd solve "fails"
        return np.array([1.0 / n] * n), converged

    monkeypatch.setattr(portfolio_optimizer_mod, "optimize_max_sharpe", _spy)
    result = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=30, seed=4)

    expected_successful = sum(1 for i in range(1, 31) if i % 3 != 0)
    assert result["attempted"] == 30
    assert result["successful"] == expected_successful
    for ticker in result["tickers"]:
        assert len(result["weights_by_ticker"][ticker]) == expected_successful


# ── Summary statistics match numpy directly ───────────────────────────────
def test_summary_statistics_match_numpy_percentiles(monkeypatch):
    returns_df = _make_correlated_returns(n_days=150, seed=13)
    fixed_weights = [np.array([w, 1.0 - w, 0.0]) for w in np.linspace(0.0, 1.0, 25)]
    call_counter = {"n": 0}

    def _spy(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight, allow_short):
        w = fixed_weights[call_counter["n"] % len(fixed_weights)]
        call_counter["n"] += 1
        return w, True

    monkeypatch.setattr(portfolio_optimizer_mod, "optimize_max_sharpe", _spy)
    result = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=25, seed=6)

    assert result["successful"] == 25
    ticker_a_weights = np.array(result["weights_by_ticker"]["A"])
    expected_p5, expected_p25, expected_p50, expected_p75, expected_p95 = np.percentile(
        ticker_a_weights, [5, 25, 50, 75, 95]
    )
    summary_a = result["summary"]["A"]
    assert abs(summary_a["median"] - expected_p50) < 1e-9
    assert abs(summary_a["p25"] - expected_p25) < 1e-9
    assert abs(summary_a["p75"] - expected_p75) < 1e-9
    assert abs(summary_a["iqr"] - (expected_p75 - expected_p25)) < 1e-9
    assert abs(summary_a["p5"] - expected_p5) < 1e-9
    assert abs(summary_a["p95"] - expected_p95) < 1e-9
    assert abs(summary_a["min"] - ticker_a_weights.min()) < 1e-9
    assert abs(summary_a["max"] - ticker_a_weights.max()) < 1e-9


# ── No fabricated summary when too few resamples converge ────────────────
def test_no_summary_when_successful_count_below_threshold(monkeypatch):
    returns_df = _make_correlated_returns(n_days=150, seed=14)
    call_counter = {"n": 0}

    def _spy(mean_returns, cov_matrix, risk_free_rate, min_weight, max_weight, allow_short):
        call_counter["n"] += 1
        n = len(mean_returns)
        converged = call_counter["n"] <= 5  # only 5 of 50 ever converge
        return np.array([1.0 / n] * n), converged

    monkeypatch.setattr(portfolio_optimizer_mod, "optimize_max_sharpe", _spy)
    result = bootstrap_max_sharpe_weight_stability(returns_df, n_bootstrap=50, seed=17)

    assert result["successful"] == 5
    assert result["successful"] < MIN_BOOTSTRAP_SUCCESSES_FOR_SUMMARY
    assert result["summary"] is None
    assert result["attempted"] == 50


# ── i18n coverage ──────────────────────────────────────────────────────────
NEW_BOOTSTRAP_I18N_KEYS = [
    "opt_bootstrap_title", "opt_bootstrap_explanation", "opt_bootstrap_button",
    "opt_bootstrap_running", "opt_bootstrap_seed_note", "opt_bootstrap_success_ratio",
    "opt_bootstrap_chart_title", "opt_bootstrap_point_estimate_label",
    "opt_bootstrap_table_col_etf", "opt_bootstrap_table_col_point_estimate",
    "opt_bootstrap_table_col_p5_p95", "opt_bootstrap_table_col_median",
    "opt_bootstrap_table_col_p25", "opt_bootstrap_table_col_p75", "opt_bootstrap_table_col_iqr",
    "opt_bootstrap_table_col_p5", "opt_bootstrap_table_col_p95", "opt_bootstrap_table_col_min",
    "opt_bootstrap_table_col_max", "opt_bootstrap_insufficient_note", "opt_bootstrap_interpretation_title",
    "opt_bootstrap_interpretation_note", "opt_bootstrap_michaud_citation", "opt_bootstrap_max_sharpe_only_note",
    "opt_methodology_backtest_sep",
]


def test_bootstrap_i18n_keys_exist_in_both_languages_and_are_non_empty():
    for key in NEW_BOOTSTRAP_I18N_KEYS:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""


def test_michaud_citation_present_without_claiming_resampled_efficiency_implementation():
    zh = TRANSLATIONS["zh-TW"]["opt_bootstrap_michaud_citation"]
    en = TRANSLATIONS["en"]["opt_bootstrap_michaud_citation"]

    assert "Michaud" in zh and "Michaud" in en
    assert "1989" in zh and "1989" in en
    assert "Resampled Efficiency" in zh and "Resampled Efficiency" in en
    assert "並未實作" in zh
    assert "does not implement" in en.lower()

    # Never phrased as an implementation claim.
    for text in (zh.lower(), en.lower()):
        assert "implements michaud" not in text
        assert "implements resampled efficiency" not in text
        assert "implements the resampled efficiency" not in text


def test_interpretation_note_does_not_overclaim_mu_as_sole_cause():
    for lang in ("zh-TW", "en"):
        text = TRANSLATIONS[lang]["opt_bootstrap_interpretation_note"]
        assert text.strip() != ""
    en = TRANSLATIONS["en"]["opt_bootstrap_interpretation_note"].lower()
    assert "does not prove" in en or "not prove" in en
    assert "not a forecast" in en


# ── Exact zh-TW/en methodology punctuation rendering (Issue #50 item 1) ──
def test_backtest_methodology_line_uses_localized_punctuation_zh_tw():
    set_language("zh-TW")
    label = t("opt_methodology_backtest_label")
    value = t("opt_methodology_backtest_value")
    sep = t("opt_methodology_backtest_sep")
    desc = t("opt_methodology_backtest_desc")
    line = f"- **{label}** — {value}{sep}{desc}\n"

    assert sep == "。"
    assert "固定配置歷史回測。系統將" in line
    # The exact bug this regresses: an ASCII period directly after the
    # Chinese label, instead of the full-width Chinese period.
    assert "固定配置歷史回測. " not in line
    assert "固定配置歷史回測." not in line


def test_backtest_methodology_line_uses_localized_punctuation_en():
    set_language("en")
    label = t("opt_methodology_backtest_label")
    value = t("opt_methodology_backtest_value")
    sep = t("opt_methodology_backtest_sep")
    desc = t("opt_methodology_backtest_desc")
    line = f"- **{label}** — {value}{sep}{desc}\n"

    assert sep == ". "
    assert "Fixed-Allocation Historical Backtest. The CURRENT" in line


# ── Streamlit AppTest: on-demand panel wiring ─────────────────────────────
def _fake_download(tickers, start_date, end_date, price_field="Close"):
    dates = pd.bdate_range(start_date, end_date)
    data = {}
    for tk in tickers:
        rng = np.random.default_rng(abs(hash(tk)) % (2**32))
        data[tk] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates)))
    return pd.DataFrame(data, index=dates)


_BOOTSTRAP_TEST_TICKERS = ["VOO", "VTI", "QQQ"]


def _setup_bootstrap_page(lang="en", method="Maximum Sharpe Ratio"):
    """Run pages/2_Portfolio_Optimizer.py via AppTest with a small synthetic
    (network-free) price fixture, select a few US ETFs, switch to `method`,
    and click "Build Optimized Portfolio". Mirrors
    tests/test_portfolio_optimizer.py's _setup_ef_page() pattern."""
    import streamlit as st
    import src.data_loader as data_loader_mod
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None
    original = data_loader_mod.download_etf_data
    data_loader_mod.download_etf_data = _fake_download
    try:
        at = AppTest.from_file(os.path.join(REPO_ROOT, "pages/2_Portfolio_Optimizer.py"), default_timeout=180)
        at.session_state["language"] = lang
        at.run()

        ms = next((w for w in at.multiselect if w.key and w.key.startswith("selected_etfs_")), None)
        if ms:
            available = [tk for tk in _BOOTSTRAP_TEST_TICKERS if tk in ms.options]
            if available:
                ms.set_value(available)
                at.run()

        for w in at.selectbox:
            if w.key == "optimization_method":
                w.set_value(method)
                at.run()
                break

        run_btn = next((b for b in at.button if b.key == "opt_run_optimization_btn"), None)
        if run_btn:
            run_btn.click()
            at.run()
        return at
    finally:
        data_loader_mod.download_etf_data = original


def _click_bootstrap_button(at):
    btn = next((b for b in at.button if b.key == "opt_bootstrap_run_btn"), None)
    assert btn is not None, "Analyze Weight Stability button not found"
    btn.click()
    at.run()
    return at


def test_bootstrap_panel_hidden_for_non_max_sharpe_methods():
    at = _setup_bootstrap_page(lang="en", method="Minimum Volatility")
    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)

    expander_labels = [e.label for e in at.expander]
    assert t("opt_bootstrap_title") not in expander_labels
    assert next((b for b in at.button if b.key == "opt_bootstrap_run_btn"), None) is None


def test_bootstrap_panel_shown_and_runs_for_max_sharpe_en():
    at = _setup_bootstrap_page(lang="en", method="Maximum Sharpe Ratio")
    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)

    expander_labels = [e.label for e in at.expander]
    assert "Bootstrap Weight Stability" in expander_labels

    at = _click_bootstrap_button(at)
    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)

    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    assert "Successful convergences:" in corpus
    assert "Michaud" in corpus
    assert "1989" in corpus
    # No raw i18n key should ever leak into the rendered page.
    for key in NEW_BOOTSTRAP_I18N_KEYS:
        assert key not in corpus


def test_bootstrap_panel_shown_and_runs_for_max_sharpe_zh_tw():
    at = _setup_bootstrap_page(lang="zh-TW", method="Maximum Sharpe Ratio")
    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)

    expander_labels = [e.label for e in at.expander]
    assert "Bootstrap 權重穩定度" in expander_labels

    at = _click_bootstrap_button(at)
    exc = at.exception[0] if at.exception else None
    assert exc is None, str(exc)

    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    assert "成功收斂次數" in corpus
    assert "Michaud" in corpus
    for key in NEW_BOOTSTRAP_I18N_KEYS:
        assert key not in corpus


# ── Issue #52 UX polish: boundary metrics + point-estimate overlay ──────────
def test_boundary_metrics_compute_inclusion_and_top_holding_rates():
    weights_by_ticker = {
        "A": [0.0, 0.0, 0.40, 0.60],
        "B": [1.0, 0.80, 0.30, 0.20],
    }
    summary = {
        "A": {"median": 0.005, "p95": 0.55, "max": 0.60},
        "B": {"median": 0.55, "p95": 0.95, "max": 1.00},
    }
    metrics = bootstrap_boundary_metrics(weights_by_ticker, summary)
    assert abs(metrics["A"]["positive_inclusion_rate"] - 0.50) < 1e-12
    assert abs(metrics["A"]["top_holding_rate"] - 0.50) < 1e-12
    assert metrics["A"]["is_boundary_pattern"] is True
    assert metrics["B"]["is_boundary_pattern"] is False


def test_boundary_metrics_ignore_numerical_dust_and_count_tied_maxima():
    eps = BOOTSTRAP_INCLUSION_EPSILON
    weights_by_ticker = {
        "A": [eps / 10, 0.50],
        "B": [1.0 - eps / 10, 0.50],
    }
    summary = {
        "A": {"median": 0.0, "p95": 0.50, "max": 0.50},
        "B": {"median": 0.75, "p95": 1.0, "max": 1.0},
    }
    metrics = bootstrap_boundary_metrics(weights_by_ticker, summary)
    assert abs(metrics["A"]["positive_inclusion_rate"] - 0.50) < 1e-12
    assert abs(metrics["A"]["top_holding_rate"] - 0.50) < 1e-12


def test_point_estimate_overlay_is_single_scatter_trace_and_localized():
    from src.charts import bootstrap_weight_stability_box_chart

    draws = {"GLD": [0.40, 0.55, 0.70], "VOO": [0.60, 0.45, 0.30]}
    points = {"GLD": 0.5858, "VOO": 0.4142}

    set_language("en")
    fig = bootstrap_weight_stability_box_chart(draws, point_estimate_weights=points)
    scatter = [trace for trace in fig.data if trace.type == "scatter"]
    assert len(scatter) == 1
    assert list(scatter[0].x) == ["GLD", "VOO"]
    assert np.allclose(list(scatter[0].y), [58.58, 41.42])
    assert scatter[0].name == "Current point estimate"

    set_language("zh-TW")
    fig_zh = bootstrap_weight_stability_box_chart(draws, point_estimate_weights=points)
    scatter_zh = [trace for trace in fig_zh.data if trace.type == "scatter"]
    assert len(scatter_zh) == 1
    assert scatter_zh[0].name == "目前點估計"


def test_box_chart_remains_backward_safe_without_point_estimate():
    from src.charts import bootstrap_weight_stability_box_chart

    fig = bootstrap_weight_stability_box_chart({"A": [0.2, 0.3], "B": [0.8, 0.7]})
    assert not [trace for trace in fig.data if trace.type == "scatter"]


def test_bootstrap_page_summary_table_source_is_compact_five_columns():
    source_path = os.path.join(REPO_ROOT, "pages/2_Portfolio_Optimizer.py")
    source = open(source_path, encoding="utf-8").read()
    start = source.index("_bootstrap_summary_rows = [")
    end = source.index("st.markdown(f\"**{t('opt_bootstrap_interpretation_title')}**\")", start)
    block = source[start:end]
    assert '"etf": _tkr' in block
    assert '"point_estimate":' in block
    assert '"median":' in block
    assert '"iqr":' in block
    assert '"p5_p95":' in block
    assert block.count("st.column_config.TextColumn(") == 5
    assert 't("opt_bootstrap_table_col_p5_p95")' in block
    assert 't("opt_bootstrap_table_col_point_estimate")' in block
    assert "hide_index=True" in block


def test_bootstrap_seed_note_scopes_reproducibility_to_same_data_window():
    zh = TRANSLATIONS["zh-TW"]["opt_bootstrap_seed_note"]
    en = TRANSLATIONS["en"]["opt_bootstrap_seed_note"]
    assert "{data_as_of}" in zh and "{data_as_of}" in en
    assert "資料窗口" in zh
    assert "data window" in en.lower()


def test_boundary_copy_is_bilingual_and_reviewer_safe():
    zh = TRANSLATIONS["zh-TW"]["opt_bootstrap_boundary_note"]
    en = TRANSLATIONS["en"]["opt_bootstrap_boundary_note"]
    assert "納入／排除" in zh
    assert "inclusion/exclusion" in en.lower()
    assert "{positive_rate}" in zh and "{top_rate}" in zh
    assert "{positive_rate}" in en and "{top_rate}" in en


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))



def test_boundary_pattern_requires_at_least_five_percent_inclusion():
    """Regression: 1 inclusion in 200 draws (0.5%) must not be described as
    a meaningful boundary/sensitivity pattern even if that one draw is large."""
    weights_by_ticker = {
        "BND": [0.40] + [0.0] * 199,
        "VOO": [0.60] + [1.0] * 199,
    }
    summary = {
        "BND": {"median": 0.0, "p95": 0.0, "max": 0.40},
        "VOO": {"median": 1.0, "p95": 1.0, "max": 1.0},
    }
    metrics = bootstrap_boundary_metrics(weights_by_ticker, summary)
    assert metrics["BND"]["positive_inclusion_rate"] == pytest.approx(0.005)
    assert metrics["BND"]["is_boundary_pattern"] is False
