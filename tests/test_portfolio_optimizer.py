"""
Portfolio Optimizer engine tests (Round 2A).

Plain-assertion script (no pytest dependency -- none is declared in
requirements.txt) covering src/portfolio_optimizer.py against a
deterministic synthetic price fixture, so these never depend on internet
access / live Yahoo Finance data. Run directly:

    python tests/test_portfolio_optimizer.py

Tests I and J additionally exercise pages/2_Portfolio_Optimizer.py via
streamlit.testing.v1.AppTest to verify global market-state persistence and
i18n behavior at the page level, not just the pure engine functions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.portfolio_optimizer import (
    run_optimization, validate_weight_constraints, backtest_portfolio,
)
from src.financial_metrics import portfolio_return, portfolio_volatility


def make_synthetic_prices(seed: int = 42, n_days: int = 300) -> pd.DataFrame:
    """4 tickers with distinct drift/volatility and a shared market factor
    (so correlations aren't trivially zero), deterministic via a fixed seed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    drifts = [0.0006, 0.0003, 0.0004, 0.0002]
    vols = [0.014, 0.008, 0.011, 0.006]
    market = rng.normal(0, 0.010, n_days)
    prices = {}
    for tkr, mu, sigma in zip(tickers, drifts, vols):
        idio = rng.normal(0, sigma, n_days)
        daily_ret = mu + 0.3 * market + idio
        price = 100 * np.cumprod(1 + daily_ret)
        prices[tkr] = price
    return pd.DataFrame(prices, index=dates)


PRICES = make_synthetic_prices()
TICKERS = list(PRICES.columns)
RESULTS = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and status == "FAIL" else ""))


# ── Test A: Equal Weight ─────────────────────────────────────────────────
def test_a_equal_weight():
    result = run_optimization(PRICES, method="Equal Weight")
    w = result["weights"]
    n = len(w)
    check("A.error_none", result.get("error") is None, str(result.get("error")))
    check("A.all_approx_equal", all(abs(v - 1.0 / n) < 1e-6 for v in w.values()), str(w))
    check("A.sum_approx_one", abs(sum(w.values()) - 1.0) < 1e-6, str(sum(w.values())))


# ── Test B: Maximum Sharpe ───────────────────────────────────────────────
def test_b_max_sharpe():
    result = run_optimization(PRICES, method="Maximum Sharpe Ratio", risk_free_rate=0.03)
    w = result["weights"]
    weights_arr = np.array(list(w.values()))
    check("B.error_none", result.get("error") is None, str(result.get("error")))
    check("B.sum_approx_one", abs(weights_arr.sum() - 1.0) < 1e-6, str(weights_arr.sum()))
    check("B.bounds_respected", bool(np.all(weights_arr >= -1e-9)) and bool(np.all(weights_arr <= 1.0 + 1e-9)),
          str(weights_arr))
    # Recompute Sharpe independently from the returned weights and compare
    # against the reported sharpe_ratio -- catches any "different sections
    # use different weight arrays" divergence.
    returns_df = PRICES.pct_change(fill_method=None).dropna(how="all")
    mean_returns = returns_df.mean().values
    cov = returns_df.cov().values * 252
    recomputed_ret = portfolio_return(weights_arr, mean_returns)
    recomputed_vol = portfolio_volatility(weights_arr, cov)
    recomputed_sharpe = (recomputed_ret - 0.03) / recomputed_vol if recomputed_vol > 0 else 0.0
    check("B.sharpe_matches_weights", abs(recomputed_sharpe - result["sharpe_ratio"]) < 1e-4,
          f"reported={result['sharpe_ratio']} recomputed={recomputed_sharpe}")


# ── Test C: Minimum Volatility ───────────────────────────────────────────
def test_c_min_volatility():
    result_minvol = run_optimization(PRICES, method="Minimum Volatility")
    result_equal = run_optimization(PRICES, method="Equal Weight")
    check("C.error_none", result_minvol.get("error") is None, str(result_minvol.get("error")))
    check(
        "C.minvol_le_equal_weight_vol",
        result_minvol["expected_volatility"] <= result_equal["expected_volatility"] + 1e-9,
        f"minvol={result_minvol['expected_volatility']} equal={result_equal['expected_volatility']}",
    )


# ── Test D: Strategy outputs correspond to the selected method ──────────
def test_d_strategy_outputs():
    r_equal = run_optimization(PRICES, method="Equal Weight")
    r_sharpe = run_optimization(PRICES, method="Maximum Sharpe Ratio")
    r_minvol = run_optimization(PRICES, method="Minimum Volatility")
    check("D.method_label_equal", r_equal["method"] == "Equal Weight")
    check("D.method_label_sharpe", r_sharpe["method"] == "Maximum Sharpe Ratio")
    check("D.method_label_minvol", r_minvol["method"] == "Minimum Volatility")
    w_equal = np.array(list(r_equal["weights"].values()))
    w_sharpe = np.array(list(r_sharpe["weights"].values()))
    w_minvol = np.array(list(r_minvol["weights"].values()))
    check("D.sharpe_differs_from_equal", not np.allclose(w_equal, w_sharpe, atol=1e-3),
          f"equal={w_equal} sharpe={w_sharpe}")
    check("D.minvol_differs_from_equal", not np.allclose(w_equal, w_minvol, atol=1e-3),
          f"equal={w_equal} minvol={w_minvol}")


# ── Test E: infeasible minimum-weight constraint ─────────────────────────
def test_e_infeasible_min():
    n = len(TICKERS)  # 4 ETFs
    feasible, code = validate_weight_constraints(n, min_weight=0.30, max_weight=1.0, allow_short=False)
    check("E.detected_infeasible", not feasible and code == "infeasible_min_weight", str((feasible, code)))
    result = run_optimization(PRICES, method="Maximum Sharpe Ratio", min_weight=0.30, max_weight=1.0)
    check("E.run_optimization_blocks", result.get("error_code") == "infeasible_min_weight", str(result.get("error_code")))


# ── Test F: infeasible maximum-weight constraint ─────────────────────────
def test_f_infeasible_max():
    n = len(TICKERS)  # 4 ETFs
    feasible, code = validate_weight_constraints(n, min_weight=0.0, max_weight=0.20, allow_short=False)
    check("F.detected_infeasible", not feasible and code == "infeasible_max_weight", str((feasible, code)))
    result = run_optimization(PRICES, method="Minimum Volatility", min_weight=0.0, max_weight=0.20)
    check("F.run_optimization_blocks", result.get("error_code") == "infeasible_max_weight", str(result.get("error_code")))


# ── Test G: result consistency (weights/allocations/metrics agree) ──────
def test_g_result_consistency():
    investment_amount = 10000.0
    result = run_optimization(PRICES, method="Maximum Sharpe Ratio")
    w = result["weights"]
    check("G.sum_weights_approx_one", abs(sum(w.values()) - 1.0) < 1e-6, str(sum(w.values())))
    allocations = {tk: wt * investment_amount for tk, wt in w.items()}
    check("G.sum_allocations_approx_amount", abs(sum(allocations.values()) - investment_amount) < 1e-3,
          str(sum(allocations.values())))
    # Metrics must be derivable from these SAME weights (not a different array)
    weights_arr = np.array(list(w.values()))
    returns_df = PRICES.pct_change(fill_method=None).dropna(how="all")
    mean_returns = returns_df.mean().values
    cov = returns_df.cov().values * 252
    recomputed_ret = portfolio_return(weights_arr, mean_returns)
    check("G.expected_return_matches", abs(recomputed_ret - result["expected_return"]) < 1e-6,
          f"reported={result['expected_return']} recomputed={recomputed_ret}")


# ── Test H: backtest uses the actual selected-strategy weights ──────────
def test_h_backtest_uses_selected_weights():
    result = run_optimization(PRICES, method="Minimum Volatility")
    w = result["weights"]
    bt = backtest_portfolio(PRICES, w, initial_investment=10000.0)
    check("H.backtest_not_empty", not bt.empty)
    # Manually recompute day-1 portfolio return from the SAME weights and
    # compare to the backtest's own day-1 daily return.
    w_arr = np.array([w[tk] for tk in PRICES.columns])
    w_arr = w_arr / w_arr.sum()
    day1_asset_returns = PRICES.pct_change().dropna().iloc[0].values
    expected_day1_return = float(np.dot(w_arr, day1_asset_returns))
    actual_day1_return = float(bt["Daily Return"].iloc[0])
    check("H.day1_return_matches_weights", abs(expected_day1_return - actual_day1_return) < 1e-9,
          f"expected={expected_day1_return} actual={actual_day1_return}")
    # And that it's NOT silently equal-weight when Min Volatility produced
    # a non-uniform allocation.
    n = len(w)
    is_uniform = all(abs(v - 1.0 / n) < 1e-6 for v in w.values())
    check("H.minvol_weights_non_uniform_for_this_fixture", not is_uniform, str(w))


# ── Test I: global market state survives running optimization ──────────
def test_i_global_market_state():
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None  # pre-existing AppTest sub-page limitation, unrelated to this change

    at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()

    region_w = None
    for w in at.selectbox:
        if w.key == "selected_region":
            region_w = w
            break
    region_w.set_value("Taiwan")
    at.run()

    ms = None
    for w in at.multiselect:
        if w.key and w.key.startswith("selected_etfs_"):
            ms = w
            break
    if ms and len(ms.options) >= 2:
        ms.set_value(ms.options[:2])
        at.run()

    run_btn = None
    for b in at.button:
        if "Optimized Portfolio" in (b.label or ""):
            run_btn = b
            break
    if run_btn:
        run_btn.click()
        at.run()

    region_after = None
    for w in at.selectbox:
        if w.key == "selected_region":
            region_after = w
            break
    exc = at.exception[0] if at.exception else None
    check("I.no_exception", exc is None, str(exc))
    check("I.market_still_taiwan", region_after is not None and region_after.value == "Taiwan",
          str(region_after.value if region_after else None))


# ── Test J: i18n -- both languages render without raw keys ──────────────
def test_j_i18n():
    import re
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None

    key_pattern = re.compile(
        r'\b(?:opt_|field_|label_|title_|btn_|portfolio_|optimizer_)[a-zA-Z0-9_]*\b'
        r'|\b[A-Z][A-Z0-9]*_[A-Z0-9_]*\b'
    )

    for lang in ("zh-TW", "en"):
        at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
        at.session_state["language"] = lang
        at.run()
        exc = at.exception[0] if at.exception else None
        check(f"J.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue
        leaked = []
        for m in at.markdown:
            leaked += key_pattern.findall(m.value)
        for kind in ("selectbox", "radio", "checkbox", "button", "expander"):
            for w in getattr(at, kind, []):
                label = getattr(w, "label", None)
                if label:
                    leaked += key_pattern.findall(str(label))
        check(f"J.{lang}.no_raw_keys", len(leaked) == 0, str(leaked))


def main():
    test_a_equal_weight()
    test_b_max_sharpe()
    test_c_min_volatility()
    test_d_strategy_outputs()
    test_e_infeasible_min()
    test_f_infeasible_max()
    test_g_result_consistency()
    test_h_backtest_uses_selected_weights()
    test_i_global_market_state()
    test_j_i18n()

    n_fail = sum(1 for _, status, _ in RESULTS if status == "FAIL")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("FAILURES:")
        for name, status, detail in RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
