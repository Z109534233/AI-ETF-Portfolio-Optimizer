"""
M3 -- Risk Methodology: deterministic tests.

Pure pytest `assert` tests covering:
  - src/methodology.py's RISK_METHODOLOGY metadata
  - src/risk_analytics.py's historical_var_cvar() (method/confidence/
    holding-period/window disclosure, wrapping financial_metrics.
    value_at_risk()/conditional_var() verbatim, and an explicit
    insufficient-data "unavailable" state)
  - maximum_drawdown() verified against a hand-computed price fixture
  - concentration_from_weights() traces to the SAME portfolio_diagnosis()
    Portfolio Optimizer uses -- no separate/divergent computation
  - holdings_overlap_matrix() (mocked source, no network -- same
    established mocking pattern as tests/test_portfolio_optimizer.py's
    HLD-* tests) is a DIFFERENT number from return correlation, with an
    explicit unavailable state when holdings data can't be retrieved
  - STRESS_SCENARIOS: shock magnitudes are unchanged from before this task,
    and every scenario has a valid, correctly-assigned hypothetical/
    historical provenance
  - i18n parity (zh-TW / en) for every new risk_* / risk_methodology_* /
    risk_scenario_note_* key
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.methodology import RISK_METHODOLOGY
from src.financial_metrics import (
    value_at_risk, conditional_var, daily_returns, maximum_drawdown, portfolio_diagnosis,
)
from src.risk_analytics import (
    historical_var_cvar, concentration_from_weights, holdings_overlap_matrix,
    STRESS_SCENARIOS, MIN_VAR_OBSERVATIONS,
)
from src.i18n import TRANSLATIONS
import src.holdings as _hld_mod


def _hld_reset():
    _hld_mod._cached_fetch_raw.clear()
    _hld_mod._LAST_GOOD_SNAPSHOT.clear()


def _make_prices(n_days: int = 40, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_days))
    return pd.Series(prices, index=dates)


# ── Methodology metadata ─────────────────────────────────────────────────
def test_methodology_metadata_matches_actual_implementation():
    m = RISK_METHODOLOGY
    assert "historical" in m["var_cvar"]["method"].lower()
    assert m["var_cvar"]["holding_period"].startswith("1 trading day")
    assert "canonical" in m["concentration"]["source"].lower()
    assert "return_correlation" in m["correlation_vs_overlap"]
    assert "holdings_overlap" in m["correlation_vs_overlap"]
    assert "linear approximation" in m["stress_scenarios"]["calculation"]
    assert "hypothetical" in m["stress_scenarios"]["provenance_labeling"]
    assert "historical" in m["stress_scenarios"]["provenance_labeling"]


# ── historical_var_cvar() ─────────────────────────────────────────────────
def test_historical_var_cvar_matches_underlying_estimator_exactly():
    prices = _make_prices(n_days=40)
    result = historical_var_cvar(prices, confidence=0.95)
    returns = daily_returns(prices)

    assert result["available"] is True
    assert result["var"] == value_at_risk(prices, 0.95)
    assert result["cvar"] == conditional_var(prices, 0.95)
    assert result["n_observations"] == len(returns)
    assert result["confidence"] == 0.95
    assert result["holding_period_days"] == 1
    assert result["window_start"] == str(returns.index.min().date())
    assert result["window_end"] == str(returns.index.max().date())


def test_historical_var_cvar_explicit_unavailable_for_too_little_data():
    prices = _make_prices(n_days=5)  # 4 daily returns, well under MIN_VAR_OBSERVATIONS
    result = historical_var_cvar(prices, confidence=0.95)
    assert result["available"] is False
    assert "var" not in result
    assert str(MIN_VAR_OBSERVATIONS) in result["reason"]


# ── Maximum Drawdown verification against a hand-computed fixture ───────
def test_maximum_drawdown_matches_hand_computed_fixture():
    dates = pd.bdate_range("2023-01-01", periods=10)
    prices = pd.Series([100, 110, 120, 90, 95, 100, 80, 85, 90, 100], index=dates)
    expected_mdd = (80 - 120) / 120  # peak 120 -> trough 80
    assert abs(maximum_drawdown(prices) - expected_mdd) < 1e-9


# ── concentration_from_weights() traces to the canonical function ───────
def test_concentration_from_weights_matches_portfolio_diagnosis():
    weights = {"A": 0.6, "B": 0.25, "C": 0.15}
    assert concentration_from_weights(weights) == portfolio_diagnosis(weights)


# ── holdings_overlap_matrix() -- mocked source, no network ──────────────
_M3_FIXTURE_A = {"topHoldings": {"holdings": [
    {"symbol": "H1", "holdingName": "Holding One", "holdingPercent": {"raw": 0.5}},
    {"symbol": "H2", "holdingName": "Holding Two", "holdingPercent": {"raw": 0.3}},
    {"symbol": "H3", "holdingName": "Holding Three", "holdingPercent": {"raw": 0.2}},
]}}
_M3_FIXTURE_B = {"topHoldings": {"holdings": [
    {"symbol": "H1", "holdingName": "Holding One", "holdingPercent": {"raw": 0.4}},
    {"symbol": "H2", "holdingName": "Holding Two", "holdingPercent": {"raw": 0.1}},
    {"symbol": "H4", "holdingName": "Holding Four", "holdingPercent": {"raw": 0.5}},
]}}


def _m3_mock_fetch(fixture_by_symbol):
    def _fetch(yahoo_symbol):
        if yahoo_symbol in fixture_by_symbol:
            return fixture_by_symbol[yahoo_symbol], True
        return None, True
    return _fetch


def test_holdings_overlap_matrix_computes_expected_overlap_score():
    _hld_reset()
    with patch.object(
        _hld_mod, "_fetch_yahoo_topholdings_raw",
        _m3_mock_fetch({"ZZZTESTA": _M3_FIXTURE_A, "ZZZTESTB": _M3_FIXTURE_B}),
    ):
        overlap = holdings_overlap_matrix(["ZZZTESTA", "ZZZTESTB"])

    result = overlap[("ZZZTESTA", "ZZZTESTB")]
    assert result["available"] is True
    # Shared holdings: H1 (min(0.5, 0.4)=0.4), H2 (min(0.3, 0.1)=0.1); H3/H4 not shared.
    assert abs(result["overlap_score"] - 0.5) < 1e-9
    assert result["shared_holdings_count"] == 2


def test_holdings_overlap_matrix_explicit_unavailable_when_source_has_nothing():
    _hld_reset()
    with patch.object(
        _hld_mod, "_fetch_yahoo_topholdings_raw",
        _m3_mock_fetch({"ZZZTESTA": _M3_FIXTURE_A}),  # ZZZTESTC has no fixture
    ):
        overlap = holdings_overlap_matrix(["ZZZTESTA", "ZZZTESTC"])

    result = overlap[("ZZZTESTA", "ZZZTESTC")]
    assert result["available"] is False
    assert "reason" in result
    assert "overlap_score" not in result


# ── STRESS_SCENARIOS: unchanged magnitudes, valid provenance ────────────
def test_stress_scenario_shock_magnitudes_unchanged_regression_guard():
    expected_shocks = {
        "equity_decline": -0.30, "rate_shock": -0.15, "high_vol": -0.20,
        "defensive": 0.05, "gfc_2008": -0.50, "covid_2020": -0.34,
        "tech_bubble": -0.45,
    }
    assert set(STRESS_SCENARIOS.keys()) == set(expected_shocks.keys())
    for key, shock in expected_shocks.items():
        assert abs(STRESS_SCENARIOS[key]["shock"] - shock) < 1e-9


def test_stress_scenario_provenance_is_valid_and_correctly_assigned():
    historical_keys = {"gfc_2008", "covid_2020", "tech_bubble"}
    for key, scenario in STRESS_SCENARIOS.items():
        assert scenario["provenance"] in ("hypothetical", "historical")
        assert scenario["provenance"] == ("historical" if key in historical_keys else "hypothetical")
        assert "i18n_key" in scenario and "note_i18n_key" in scenario


# ── i18n parity for every new risk_* methodology/disclosure key ─────────
def test_risk_methodology_i18n_keys_exist_in_both_languages():
    new_keys = [
        "risk_var_unavailable", "risk_var_unavailable_reason",
        "risk_methodology_title", "risk_methodology_subtitle",
        "risk_methodology_var_window_value", "risk_methodology_var_label",
        "risk_methodology_var_desc", "risk_methodology_mdd_label",
        "risk_methodology_mdd_desc", "risk_methodology_concentration_label",
        "risk_methodology_concentration_desc", "risk_methodology_correlation_label",
        "risk_methodology_correlation_desc", "risk_methodology_stress_label",
        "risk_methodology_stress_desc", "risk_correlation_vs_overlap_note",
        "risk_show_holdings_overlap", "risk_loading_holdings_overlap",
        "risk_col_pair", "risk_col_overlap_score", "risk_col_shared_holdings",
        "risk_overlap_unavailable", "risk_canonical_concentration_title",
        "risk_canonical_concentration_sub", "risk_canonical_concentration_note",
        "risk_col_provenance", "risk_provenance_hypothetical", "risk_provenance_historical",
        "risk_stress_methodology_note", "risk_scenario_note_hypothetical",
        "risk_scenario_note_2008", "risk_scenario_note_covid", "risk_scenario_note_tech_bubble",
        "risk_methodology_alpha_label", "risk_methodology_alpha_desc",
        "risk_benchmark_index_scope_undefined", "risk_benchmark_index_unavailable",
        "metric_alpha_annualized",
    ]
    for key in new_keys:
        assert key in TRANSLATIONS["zh-TW"], f"missing zh-TW translation for {key}"
        assert key in TRANSLATIONS["en"], f"missing en translation for {key}"
        assert TRANSLATIONS["zh-TW"][key] != key
        assert TRANSLATIONS["en"][key] != key
        assert TRANSLATIONS["zh-TW"][key].strip() != ""
        assert TRANSLATIONS["en"][key].strip() != ""

    # Every STRESS_SCENARIOS i18n_key / note_i18n_key referenced by the page
    # must also actually resolve (no raw key ever shown to the user).
    for scenario in STRESS_SCENARIOS.values():
        for key in (scenario["i18n_key"], scenario["note_i18n_key"]):
            assert key in TRANSLATIONS["zh-TW"] and key in TRANSLATIONS["en"], f"missing translation for {key}"
