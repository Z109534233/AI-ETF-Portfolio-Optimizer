"""
AI Advisor -- deterministic synthesis tests (Issue #18 Stage 6).

Before this stage the AI Advisor page was fully disconnected from the rest
of the app: it built its own from-scratch portfolio in the sidebar and
never consumed Portfolio Optimizer's canonical current_portfolio, Investment
Simulator, Risk Analytics, Machine Learning, or Market Intelligence outputs.

These tests cover src.ai_advisor.build_advisor_context() /
generate_rule_based_narrative() / _build_prompt() against canned,
deterministic inputs -- no network access, no OpenAI calls -- verifying:
  - every section is grounded in real, already-computed values (never
    invented), and states an explicit reason when unavailable
  - simulator/ML data is only surfaced when it actually corresponds to the
    current portfolio (never a stale/contradictory prior run)
  - missing components (no portfolio, no simulator run, no ML run, no news)
    are handled cleanly without raising
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.ai_advisor import (
    build_advisor_context, generate_rule_based_narrative, _build_prompt, advisor_fingerprint,
)
from src.machine_learning import run_ml_pipeline


def _sample_portfolio():
    return {
        "portfolio_id": "abc123",
        "strategy": "Maximum Sharpe",
        "market": "United States",
        "tickers": ["QQQ", "BND"],
        "weights": {"QQQ": 0.7, "BND": 0.3},
        "investment_amount": 10000.0,
        "expected_return": 0.12,
        "volatility": 0.18,
        "sharpe_ratio": 0.61,
        "max_drawdown": -0.25,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }


def _sample_prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0004, 0.01, n)
    prices = 100 * (1 + pd.Series(daily_returns)).cumprod()
    prices.index = pd.date_range("2025-01-01", periods=n, freq="B")
    return prices


# ── build_advisor_context: portfolio/risk grounding ─────────────────────────

def test_no_portfolio_all_sections_explicitly_unavailable():
    context = build_advisor_context(portfolio=None, portfolio_source="current")
    assert context["portfolio"]["available"] is False
    assert context["risk"]["available"] is False
    assert context["simulator"]["future_projection"]["available"] is False
    assert context["simulator"]["historical_simulation"]["available"] is False
    assert context["ml"]["available"] is False
    assert context["news"]["available"] is False
    # every unavailable section carries a stated reason, never a silent gap
    for section in ("portfolio", "risk", "ml", "news"):
        assert context[section]["reason"]


def test_portfolio_available_reuses_canonical_numbers_verbatim():
    portfolio = _sample_portfolio()
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    p = context["portfolio"]
    assert p["available"] is True
    assert p["expected_return"] == portfolio["expected_return"]
    assert p["volatility"] == portfolio["volatility"]
    assert p["sharpe_ratio"] == portfolio["sharpe_ratio"]
    assert p["weights"] == portfolio["weights"]
    assert p["strategy"] == "Maximum Sharpe"


def test_risk_concentration_always_available_when_portfolio_available():
    portfolio = _sample_portfolio()
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    risk = context["risk"]
    assert risk["available"] is True
    assert risk["concentration"]["largest_ticker"] == "QQQ"
    assert risk["concentration"]["largest_weight"] == pytest.approx(0.7)
    # no price series passed in -> VaR/CVaR explicitly unavailable, not fabricated
    assert risk["var_cvar"]["available"] is False
    assert risk["var_cvar"]["reason"]


def test_var_cvar_computed_from_real_price_series_when_provided():
    portfolio = _sample_portfolio()
    prices = _sample_prices(n=60)
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", portfolio_prices=prices,
    )
    vc = context["risk"]["var_cvar"]
    assert vc["available"] is True
    assert vc["n_observations"] == 59  # 60 prices -> 59 daily returns
    assert vc["var"] is not None and vc["cvar"] is not None
    # CVaR (expected shortfall) must be at least as severe as VaR
    assert vc["cvar"] <= vc["var"]


# ── Simulator integration: only surfaced for a matching current-portfolio run ─

def test_simulator_future_projection_surfaced_when_strategy_matches():
    portfolio = _sample_portfolio()
    sim_result = {"summary": {"median_final": 15000.0, "probability_profit": 0.8}}
    sim_params = {
        "portfolio_strategy": "Maximum Sharpe", "years": 10,
        "assumption_source": "Portfolio Optimizer expected return/volatility",
        "n_simulations": 1000,
    }
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        sim_result=sim_result, sim_params=sim_params,
    )
    fp = context["simulator"]["future_projection"]
    assert fp["available"] is True
    assert fp["summary"]["median_final"] == 15000.0


def test_simulator_not_surfaced_when_strategy_mismatched_stale_run():
    portfolio = _sample_portfolio()
    sim_result = {"summary": {"median_final": 999999.0}}
    sim_params = {"portfolio_strategy": "Equal Weight"}  # different strategy -> stale
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        sim_result=sim_result, sim_params=sim_params,
    )
    fp = context["simulator"]["future_projection"]
    assert fp["available"] is False
    assert fp["reason"]


def test_simulator_not_surfaced_for_custom_portfolio():
    portfolio = _sample_portfolio()
    sim_result = {"summary": {"median_final": 15000.0}}
    sim_params = {"portfolio_strategy": "Maximum Sharpe"}
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="custom",
        sim_result=sim_result, sim_params=sim_params,
    )
    assert context["simulator"]["future_projection"]["available"] is False


def test_historical_simulation_requires_strategy_and_market_match():
    portfolio = _sample_portfolio()
    hist_result = {"summary": {"final_value": 12000.0, "gain": 2000.0, "annualized_mwr": 0.08}}
    hist_params = {"strategy": "Maximum Sharpe", "market": "United States"}
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        hist_result=hist_result, hist_params=hist_params,
    )
    assert context["simulator"]["historical_simulation"]["available"] is True

    hist_params_wrong_market = {"strategy": "Maximum Sharpe", "market": "Taiwan"}
    context2 = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        hist_result=hist_result, hist_params=hist_params_wrong_market,
    )
    assert context2["simulator"]["historical_simulation"]["available"] is False


# ── ML integration: only surfaced when valid and for an actual holding ──────

def test_ml_surfaced_when_valid_and_ticker_is_current_holding():
    portfolio = _sample_portfolio()
    ml_result = {
        "error": None, "model_name": "Random Forest",
        "metrics": {"Accuracy": 0.58},
        "baseline_accuracy": 0.52,
        "test_start": "2025-06-01", "test_end": "2025-12-01",
        "lookahead_periods": 1,
    }
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        ml_result=ml_result, ml_ticker="QQQ",
    )
    ml = context["ml"]
    assert ml["available"] is True
    assert ml["beats_baseline"] is True
    assert ml["accuracy"] == 0.58


def test_ml_hidden_when_ticker_not_a_current_holding():
    portfolio = _sample_portfolio()
    ml_result = {"error": None, "metrics": {"Accuracy": 0.9}, "baseline_accuracy": 0.5}
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        ml_result=ml_result, ml_ticker="ARKK",  # not in QQQ/BND
    )
    assert context["ml"]["available"] is False
    assert "ARKK" in context["ml"]["reason"]


def test_ml_hidden_when_run_errored_reason_is_the_real_error():
    portfolio = _sample_portfolio()
    ml_result = {"error": "Insufficient data for ML analysis. Need at least 50 observations."}
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        ml_result=ml_result, ml_ticker="QQQ",
    )
    assert context["ml"]["available"] is False
    assert context["ml"]["reason"] == ml_result["error"]


def test_ml_context_wired_to_real_run_ml_pipeline_output():
    """Regression test for a real bug caught during review: _ml_context()
    read metrics["accuracy"] (lowercase) but src.machine_learning's
    run_ml_pipeline()/_compute_metrics() key it "Accuracy" (capitalized,
    see pages/5_Machine_Learning.py's own display code) -- so accuracy was
    silently always None and beats_baseline was always False, even for a
    model that genuinely beat the baseline. Calling the REAL pipeline here
    (not a hand-built canned dict) means a future key-name drift in either
    module fails this test instead of silently degrading to a wrong,
    misleadingly-confident "does not beat baseline" claim.
    """
    portfolio = _sample_portfolio()
    prices = _sample_prices(n=200, seed=7)
    real_ml_result = run_ml_pipeline(prices, model_type="Random Forest", test_size=0.2)
    assert real_ml_result.get("error") is None, real_ml_result.get("error")

    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current",
        ml_result=real_ml_result, ml_ticker="QQQ",
    )
    ml = context["ml"]
    assert ml["available"] is True
    assert isinstance(ml["accuracy"], float)
    assert 0.0 <= ml["accuracy"] <= 1.0
    assert isinstance(ml["baseline_accuracy"], float)


def test_ml_never_surfaced_for_custom_portfolio():
    portfolio = _sample_portfolio()
    ml_result = {"error": None, "metrics": {"Accuracy": 0.9}, "baseline_accuracy": 0.5}
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="custom",
        ml_result=ml_result, ml_ticker="QQQ",
    )
    assert context["ml"]["available"] is False


# ── Market Intelligence integration: pure function of holdings + real news ──

def test_news_relevant_holdings_and_impact_text():
    portfolio = _sample_portfolio()
    news_items = [
        {"title": "Tech stocks rally on strong chip earnings", "impact": "Positive"},
        {"title": "Bond yields climb as Fed signals rate hike", "impact": "Negative"},
    ]
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", news_items=news_items,
    )
    news = context["news"]
    assert news["available"] is True
    assert news["headline_count"] == 2
    tickers_seen = {e["ticker"] for e in news["affected_holdings"]}
    assert tickers_seen == {"QQQ", "BND"}
    assert news["relevant_holdings_count"] == 2  # both QQQ and BND matched non-neutral news
    assert news["portfolio_impact_text"]


def test_news_unavailable_when_no_items():
    portfolio = _sample_portfolio()
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current", news_items=[])
    assert context["news"]["available"] is False
    assert context["news"]["reason"]


# ── Narrative grounding: no invented numbers, missing sections handled ──────

def test_rule_based_narrative_contains_actual_computed_values():
    portfolio = _sample_portfolio()
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    narrative = generate_rule_based_narrative(context, "Long-term Growth", "Moderate", 10)
    assert "QQQ" in narrative
    assert "70.0%" in narrative or "70.00%" in narrative
    assert "12.00%" in narrative  # expected_return 0.12


def test_rule_based_narrative_states_unavailable_reasons_not_fabricated_numbers():
    portfolio = _sample_portfolio()
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    narrative = generate_rule_based_narrative(context)
    # simulator/ml/news were never provided -> must be disclosed as unavailable,
    # never silently omitted or replaced with a plausible-looking number
    assert context["simulator"]["future_projection"]["available"] is False
    assert context["ml"]["available"] is False
    assert context["news"]["available"] is False
    assert narrative.count("Not available") + narrative.count("not available") >= 1 or "無法使用" in narrative


def test_rule_based_narrative_handles_no_portfolio_without_raising():
    context = build_advisor_context(portfolio=None)
    narrative = generate_rule_based_narrative(context)
    assert isinstance(narrative, str) and len(narrative) > 0


def test_prompt_uses_actual_computed_values_not_placeholders():
    portfolio = _sample_portfolio()
    prices = _sample_prices(n=60)
    ml_result = {
        "error": None, "model_name": "Random Forest", "metrics": {"Accuracy": 0.58},
        "baseline_accuracy": 0.52, "test_start": "2025-06-01", "test_end": "2025-12-01",
        "lookahead_periods": 1,
    }
    context = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", portfolio_prices=prices,
        ml_result=ml_result, ml_ticker="QQQ",
    )
    prompt = _build_prompt(context, "Long-term Growth", "Moderate", 10)
    assert "QQQ: 70.0%" in prompt
    assert "Maximum Sharpe" in prompt
    assert "58.00%" in prompt  # ML accuracy
    assert "experimental" in prompt.lower()
    # sections never provided must say so explicitly in the prompt, not be omitted
    assert "not available" in prompt.lower()


def test_prompt_handles_fully_missing_context_without_raising():
    context = build_advisor_context(portfolio=None)
    prompt = _build_prompt(context, "Long-term Growth", "Moderate", 10)
    assert isinstance(prompt, str)
    assert "not available" in prompt.lower()


# ── advisor_fingerprint: cache invalidation must track everything the
# prompt actually quotes (Issue #20 release-gate review) ───────────────────
# advisor_fingerprint() feeds src.openai_service.cached_generate(): if it
# omits a field _build_prompt() reads, running Investment Simulator/
# Machine Learning after an initial "Generate Analysis" click and clicking
# "Regenerate Analysis" again can silently keep serving a stale cached
# narrative that still says that section is "not available" while the
# freshly-built context (and the page's own Deterministic Data expander)
# shows real numbers -- a visible, self-contradicting page.

def test_fingerprint_changes_when_simulator_future_projection_appears():
    portfolio = _sample_portfolio()
    context_without = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    fp_without = advisor_fingerprint(context_without, "Long-term Growth", "Moderate", 10)

    sim_result = {"summary": {"median_final": 15000.0, "probability_profit": 0.8}}
    sim_params = {"portfolio_strategy": "Maximum Sharpe", "years": 10, "n_simulations": 1000}
    context_with = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", sim_result=sim_result, sim_params=sim_params,
    )
    fp_with = advisor_fingerprint(context_with, "Long-term Growth", "Moderate", 10)

    assert context_without["simulator"]["future_projection"]["available"] is False
    assert context_with["simulator"]["future_projection"]["available"] is True
    assert fp_without != fp_with


def test_fingerprint_changes_when_historical_simulation_appears():
    portfolio = _sample_portfolio()
    context_without = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    fp_without = advisor_fingerprint(context_without, "Long-term Growth", "Moderate", 10)

    hist_result = {"summary": {"final_value": 12000.0, "gain": 2000.0, "annualized_mwr": 0.08}}
    hist_params = {"strategy": "Maximum Sharpe", "market": "United States"}
    context_with = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", hist_result=hist_result, hist_params=hist_params,
    )
    fp_with = advisor_fingerprint(context_with, "Long-term Growth", "Moderate", 10)

    assert fp_without != fp_with


def test_fingerprint_changes_when_ml_result_appears():
    portfolio = _sample_portfolio()
    prices = _sample_prices()
    context_without = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    fp_without = advisor_fingerprint(context_without, "Long-term Growth", "Moderate", 10)

    ml_result = {
        "metrics": {"accuracy": 0.58, "baseline_accuracy": 0.55, "auc": 0.6},
        "model_type": "Random Forest", "test_size": 0.2, "lookahead_periods": 1,
    }
    context_with = build_advisor_context(
        portfolio=portfolio, portfolio_source="current", portfolio_prices=prices,
        ml_result=ml_result, ml_ticker="QQQ",
    )
    fp_with = advisor_fingerprint(context_with, "Long-term Growth", "Moderate", 10)

    assert fp_without != fp_with


def test_fingerprint_stable_when_nothing_relevant_changes():
    portfolio = _sample_portfolio()
    context_a = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    context_b = build_advisor_context(portfolio=dict(portfolio), portfolio_source="current")
    fp_a = advisor_fingerprint(context_a, "Long-term Growth", "Moderate", 10)
    fp_b = advisor_fingerprint(context_b, "Long-term Growth", "Moderate", 10)
    assert fp_a == fp_b


def test_fingerprint_changes_when_investment_amount_changes():
    # Regression test: investment_amount is quoted verbatim in _build_prompt()
    # ("Investment amount: ...") but was previously omitted from the manually
    # curated fingerprint field list, so changing only the dollar amount left
    # the cached AI narrative (which had quoted the old amount) on screen.
    portfolio = _sample_portfolio()
    other = dict(portfolio, investment_amount=50000.0)
    fp_a = advisor_fingerprint(
        build_advisor_context(portfolio=portfolio, portfolio_source="current"),
        "Long-term Growth", "Moderate", 10,
    )
    fp_b = advisor_fingerprint(
        build_advisor_context(portfolio=other, portfolio_source="current"),
        "Long-term Growth", "Moderate", 10,
    )
    assert fp_a != fp_b


def test_fingerprint_changes_when_max_drawdown_or_market_changes():
    portfolio = _sample_portfolio()
    fp_base = advisor_fingerprint(
        build_advisor_context(portfolio=portfolio, portfolio_source="current"),
        "Long-term Growth", "Moderate", 10,
    )
    fp_drawdown = advisor_fingerprint(
        build_advisor_context(portfolio=dict(portfolio, max_drawdown=-0.55), portfolio_source="current"),
        "Long-term Growth", "Moderate", 10,
    )
    fp_market = advisor_fingerprint(
        build_advisor_context(portfolio=dict(portfolio, market="Taiwan"), portfolio_source="current"),
        "Long-term Growth", "Moderate", 10,
    )
    assert fp_base != fp_drawdown
    assert fp_base != fp_market


# ── Page-level stale-result guard (Issue #20 release-gate review, same
# Priority-0 fix pattern as Machine Learning's section 6A) ─────────────────

def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _stale_warning_shown(at) -> bool:
    warnings = "\n".join(w.value for w in at.warning)
    return "changed" in warnings.lower() or "變更" in warnings


def _generate_once(lang="en"):
    at = _apptest_from_file("pages/6_AI_Advisor.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    gen_btn = next((b for b in at.button if (b.label or "") == "Generate AI Analysis"), None)
    assert gen_btn is not None, "expected the Generate AI Analysis button (fresh session -> use_custom mode)"
    gen_btn.click()
    at.run()
    assert at.exception == []
    return at


def test_ai_advisor_no_stale_warning_immediately_after_generate():
    at = _generate_once()
    assert not _stale_warning_shown(at)


def test_ai_advisor_stale_warning_on_horizon_change_without_regenerate():
    at = _generate_once()
    sliders = [s for s in at.slider if (s.label or "") == "Investment Horizon (Years)"]
    assert sliders, "expected the investment-horizon slider on the sidebar"
    original = sliders[0].value
    sliders[0].set_value(original + 5 if original + 5 <= 40 else original - 5)
    at.run()
    assert at.exception == []
    assert _stale_warning_shown(at), "changing the investment horizon without regenerating must show the stale-result warning"


def test_ai_advisor_stale_warning_clears_after_regenerate():
    at = _generate_once()
    sliders = [s for s in at.slider if (s.label or "") == "Investment Horizon (Years)"]
    original = sliders[0].value
    sliders[0].set_value(original + 5 if original + 5 <= 40 else original - 5)
    at.run()
    assert _stale_warning_shown(at)

    # The stale-result guard st.stop()s the page right after the warning
    # (same as Machine Learning's), so the page-body "Regenerate Analysis"
    # button (rendered only past that point, for a non-stale result) isn't
    # available here -- the sidebar's "Generate AI Analysis" button always
    # is, exactly like ML's sidebar "Train Model" button.
    gen_btn = next((b for b in at.button if (b.label or "") == "Generate AI Analysis"), None)
    assert gen_btn is not None
    gen_btn.click()
    at.run()
    assert at.exception == []
    assert not _stale_warning_shown(at), "after regenerating, the stale-result warning must clear"
