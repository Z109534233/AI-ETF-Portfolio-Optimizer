"""
Investment Simulator -- stale-result regression tests (Issue #20 release-gate
review: the ML page's Priority-0 stale-result fix, applied elsewhere too).

The Monte Carlo ("Future Projection") result previously had no
input-fingerprint guard -- changing a sidebar widget (e.g. switching the
market scenario) without clicking "Run Simulation" again left the OLD
result on screen with no warning that it no longer matched the displayed
inputs. This mirrors the exact bug class fixed for Machine Learning in
Issue #20 section 6A.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _run_monte_carlo_once(lang="en"):
    at = _apptest_from_file("pages/3_Investment_Simulator.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
    assert run_btn is not None, "expected the Run Simulation button in Future Projection mode"
    run_btn.click()
    at.run()
    assert at.exception == []
    return at


def _stale_warning_shown(at) -> bool:
    warnings = "\n".join(w.value for w in at.warning)
    return "changed" in warnings.lower() or "變更" in warnings


def test_monte_carlo_no_stale_warning_immediately_after_run():
    at = _run_monte_carlo_once()
    assert not _stale_warning_shown(at)


def test_monte_carlo_stale_warning_on_scenario_change_without_rerun():
    at = _run_monte_carlo_once()
    selects = [s for s in at.selectbox if s.key == "sim_market_scenario_choice"]
    assert selects, "expected the Market Scenario selectbox (default assumption source with no current_portfolio)"
    other_options = [o for o in selects[0].options if o != selects[0].value]
    assert other_options
    selects[0].set_value(other_options[0])
    at.run()
    assert at.exception == []
    assert _stale_warning_shown(at), "changing the market scenario without re-running must show the stale-result warning"


def test_monte_carlo_stale_warning_clears_after_rerun():
    at = _run_monte_carlo_once()
    selects = [s for s in at.selectbox if s.key == "sim_market_scenario_choice"]
    other_options = [o for o in selects[0].options if o != selects[0].value]
    selects[0].set_value(other_options[0])
    at.run()
    assert _stale_warning_shown(at)

    run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
    run_btn.click()
    at.run()
    assert at.exception == []
    assert not _stale_warning_shown(at), "after re-running, the stale-result warning must clear"


def _max_sharpe_current_portfolio():
    return {
        "strategy": "Maximum Sharpe Ratio",
        "market": "United States",
        "tickers": ["VOO", "GLD", "BND"],
        "weights": {"VOO": 0.4142, "GLD": 0.5858, "BND": 0.0},
        "investment_amount": 10000.0,
        "expected_return": 0.1729,
        "volatility": 0.145,
        "equal_weight_reference_return": 0.081,
        "equal_weight_reference_volatility": 0.128,
    }


def test_max_sharpe_handoff_defaults_to_non_optimized_equal_weight_reference():
    at = _apptest_from_file("pages/3_Investment_Simulator.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.session_state["current_portfolio"] = _max_sharpe_current_portfolio()
    at.run()
    assert at.exception == []

    source = next(
        (s for s in at.selectbox if s.key == "projection_assumption_source"), None
    )
    assert source is not None
    assert source.value == "Equal-Weight Historical Reference"

    captions = "\n".join(c.value for c in at.caption)
    assert "equal-weight (1/N)" in captions
    warnings = "\n".join(w.value for w in at.warning)
    assert "optimizer's curse" not in warnings.lower()


def test_explicit_max_sharpe_in_sample_projection_source_shows_optimizer_curse_warning():
    at = _apptest_from_file("pages/3_Investment_Simulator.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.session_state["current_portfolio"] = _max_sharpe_current_portfolio()
    at.run()
    assert at.exception == []

    source = next(
        (s for s in at.selectbox if s.key == "projection_assumption_source"), None
    )
    source.set_value("Portfolio Historical Statistics")
    at.run()
    assert at.exception == []

    warnings = "\n".join(w.value for w in at.warning)
    assert "optimizer's curse" in warnings.lower()
    assert "selection bias" in warnings.lower()
