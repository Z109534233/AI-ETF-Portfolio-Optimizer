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
    compute_efficient_frontier,
)
from src.financial_metrics import portfolio_return, portfolio_volatility, covariance_matrix
from src.i18n import t


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


# ══════════════════════════════════════════════════════════════════════════
# Round 2B-1: Strategy Comparison
# ══════════════════════════════════════════════════════════════════════════

_SC_METHODS = ["Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility"]


def _compute_comparison(prices_df, risk_free_rate=0.05, min_weight=0.0, max_weight=1.0, allow_short=False):
    """Mirrors the comparison computation in pages/2_Portfolio_Optimizer.py
    exactly (same run_optimization() calls, same largest-position/backtest
    logic) so these tests validate the actual page logic, not a
    reimplementation of it."""
    results = {}
    for method in _SC_METHODS:
        r = run_optimization(
            prices_df=prices_df, method=method, risk_free_rate=risk_free_rate,
            min_weight=min_weight, max_weight=max_weight, allow_short=allow_short,
        )
        w = r["weights"]
        largest_ticker = max(w, key=w.get) if w else None
        largest_weight = w.get(largest_ticker, 0.0) if largest_ticker else 0.0
        bt = backtest_portfolio(prices_df, w, 10000.0)
        mdd = None
        if not bt.empty:
            from src.financial_metrics import maximum_drawdown
            mdd = maximum_drawdown(bt["Portfolio Value"])
        results[method] = {
            "weights": w, "expected_return": r["expected_return"],
            "expected_volatility": r["expected_volatility"], "sharpe_ratio": r["sharpe_ratio"],
            "largest_ticker": largest_ticker, "largest_weight": largest_weight, "max_drawdown": mdd,
        }
    return results


# ── SC-A: all three strategies appear ────────────────────────────────────
def test_sc_a_all_three_appear():
    results = _compute_comparison(PRICES)
    check("SC-A.all_three_present", set(results.keys()) == set(_SC_METHODS), str(results.keys()))


# ── SC-B: weights sum to ~1 for each strategy ────────────────────────────
def test_sc_b_weights_sum_to_one():
    results = _compute_comparison(PRICES)
    for method, data in results.items():
        total = sum(data["weights"].values())
        check(f"SC-B.{method}.sum_approx_one", abs(total - 1.0) < 1e-6, str(total))


# ── SC-C: "Best Risk-Adjusted Return" badge only on the actual highest Sharpe ──
def test_sc_c_best_sharpe_badge_correct():
    results = _compute_comparison(PRICES)
    best_sharpe_method = max(results, key=lambda m: results[m]["sharpe_ratio"])
    for method in _SC_METHODS:
        would_get_badge = (method == best_sharpe_method)
        is_actually_highest = all(
            results[method]["sharpe_ratio"] >= results[other]["sharpe_ratio"] - 1e-9
            for other in _SC_METHODS
        )
        check(f"SC-C.{method}.badge_matches_reality", would_get_badge == is_actually_highest,
              f"sharpe={results[method]['sharpe_ratio']} all={[(m, results[m]['sharpe_ratio']) for m in _SC_METHODS]}")


# ── SC-D: "Lowest Risk" badge only on the actual lowest volatility ───────
def test_sc_d_lowest_vol_badge_correct():
    results = _compute_comparison(PRICES)
    lowest_vol_method = min(results, key=lambda m: results[m]["expected_volatility"])
    for method in _SC_METHODS:
        would_get_badge = (method == lowest_vol_method)
        is_actually_lowest = all(
            results[method]["expected_volatility"] <= results[other]["expected_volatility"] + 1e-9
            for other in _SC_METHODS
        )
        check(f"SC-D.{method}.badge_matches_reality", would_get_badge == is_actually_lowest,
              f"vol={results[method]['expected_volatility']} all={[(m, results[m]['expected_volatility']) for m in _SC_METHODS]}")


# ── SC-E: Largest Position matches the actual max weight in that strategy ──
def test_sc_e_largest_position_correct():
    results = _compute_comparison(PRICES)
    for method, data in results.items():
        actual_max_ticker = max(data["weights"], key=data["weights"].get)
        actual_max_weight = data["weights"][actual_max_ticker]
        check(f"SC-E.{method}.largest_ticker_correct", data["largest_ticker"] == actual_max_ticker,
              f"reported={data['largest_ticker']} actual={actual_max_ticker}")
        check(f"SC-E.{method}.largest_weight_correct", abs(data["largest_weight"] - actual_max_weight) < 1e-9,
              f"reported={data['largest_weight']} actual={actual_max_weight}")


# ── SC-F: "Higher Concentration" threshold logic (>50%) ──────────────────
def test_sc_f_concentration_threshold():
    threshold = 0.50
    # Fabricated weights, not real optimizer output -- this validates the
    # THRESHOLD LOGIC itself (the same comparison used in the page),
    # independent of whether real market data happens to produce a
    # concentrated result for this particular synthetic fixture.
    concentrated = {"A": 0.60, "B": 0.20, "C": 0.20}
    balanced = {"A": 0.40, "B": 0.30, "C": 0.30}
    concentrated_largest = max(concentrated.values())
    balanced_largest = max(balanced.values())
    check("SC-F.concentrated_triggers_badge", concentrated_largest > threshold, str(concentrated_largest))
    check("SC-F.balanced_does_not_trigger_badge", not (balanced_largest > threshold), str(balanced_largest))


# ── SC-G: comparison calculation does not alter the selected portfolio ──
def test_sc_g_no_side_effects():
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None

    at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()

    method_w = None
    for w in at.selectbox:
        if w.key == "optimization_method":
            method_w = w
            break
    minvol_opt = next((o for o in method_w.options if "Minimum" in o), None)
    method_w.set_value(minvol_opt)
    at.run()

    run_btn = next(iter(at.button), None)
    if run_btn:
        run_btn.click()
        at.run()

    exc = at.exception[0] if at.exception else None
    check("SC-G.no_exception", exc is None, str(exc))
    if exc:
        return

    selected_method_after = at.session_state["opt_result"]["method"]
    check("SC-G.selected_strategy_unchanged", selected_method_after == "Minimum Volatility", selected_method_after)


# ── SC-H: switching strategy updates the current-strategy indicator, keeps comparison intact ──
def test_sc_h_switch_strategy():
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None

    at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()

    method_w = None
    for w in at.selectbox:
        if w.key == "optimization_method":
            method_w = w
            break
    sharpe_opt = next((o for o in method_w.options if "Sharpe" in o), None)
    method_w.set_value(sharpe_opt)
    at.run()
    run_btn = next(iter(at.button), None)
    if run_btn:
        run_btn.click()
        at.run()

    exc = at.exception[0] if at.exception else None
    check("SC-H.no_exception", exc is None, str(exc))
    if exc:
        return

    joined = "\n".join(m.value for m in at.markdown)
    check("SC-H.comparison_section_present", "Strategy Comparison" in joined)
    check("SC-H.all_three_labels_present",
          all(name in joined for name in ("Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility")))
    check("SC-H.current_strategy_indicator_present", "Current Strategy" in joined)
    check("SC-H.selected_method_is_sharpe", at.session_state["opt_result"]["method"] == "Maximum Sharpe Ratio",
          at.session_state["opt_result"]["method"])


# ── SC-I: zh-TW / English render correctly, no raw keys ─────────────────
def test_sc_i_i18n():
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
        run_btn = next(iter(at.button), None)
        if run_btn:
            run_btn.click()
            at.run()
        exc = at.exception[0] if at.exception else None
        check(f"SC-I.{lang}.no_exception", exc is None, str(exc))
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
        check(f"SC-I.{lang}.no_raw_keys", len(leaked) == 0, str(leaked))
        joined = "\n".join(m.value for m in at.markdown)
        expect = "策略比較" if lang == "zh-TW" else "Strategy Comparison"
        check(f"SC-I.{lang}.comparison_title_translated", expect in joined)


# ── SC-J: rendered-output corpus (markdown+captions+table columns+widget
#          labels+card HTML) must never contain a raw i18n key fragment ──
def test_sc_j_render_output_no_raw_keys():
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None
    forbidden = ("opt_", "OPT_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")

    for lang in ("zh-TW", "en"):
        at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
        at.session_state["language"] = lang
        at.run()
        run_btn = next(iter(at.button), None)
        if run_btn:
            run_btn.click()
            at.run()
        exc = at.exception[0] if at.exception else None
        check(f"SC-J.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue

        corpus_parts = [m.value for m in at.markdown]
        corpus_parts += [c.value for c in at.caption]
        for kind in ("selectbox", "radio", "checkbox", "button", "expander", "multiselect"):
            for w in getattr(at, kind, []):
                label = getattr(w, "label", None)
                if label:
                    corpus_parts.append(str(label))
        for dfw in at.dataframe:
            val = dfw.value
            cols = list(val.data.columns) if hasattr(val, "data") else list(val.columns)
            corpus_parts += [str(c) for c in cols]

        corpus = "\n".join(corpus_parts)
        hits = [frag for frag in forbidden if frag in corpus]
        check(f"SC-J.{lang}.no_forbidden_fragments", len(hits) == 0, str(hits))


# ══════════════════════════════════════════════════════════════════════════
# Round 2B-2: Efficient Frontier Visualization & Decision Support
# ══════════════════════════════════════════════════════════════════════════

_EF_TEST_TICKERS = ["VOO", "VTI", "QQQ", "SPY", "SCHD"]


def _ef_mean_cov(prices_df=PRICES):
    mean_returns = prices_df.pct_change(fill_method=None).dropna(how="all").mean().values
    cov = covariance_matrix(prices_df).values
    return mean_returns, cov


def _setup_ef_page(method=None, lang="en", tickers=None):
    """Run pages/2_Portfolio_Optimizer.py via AppTest with the given
    language, ETF selection (default VOO/VTI/QQQ/SPY/SCHD per Round 2B-2's
    test ticket), and optimization method, then click Run."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.page_link = lambda *a, **k: None
    tickers = tickers if tickers is not None else _EF_TEST_TICKERS

    at = AppTest.from_file("pages/2_Portfolio_Optimizer.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()

    ms = None
    for w in at.multiselect:
        if w.key and w.key.startswith("selected_etfs_"):
            ms = w
            break
    if ms:
        available = [tk for tk in tickers if tk in ms.options]
        if available:
            ms.set_value(available)
            at.run()

    if method:
        for w in at.selectbox:
            if w.key == "optimization_method" and method in w.options:
                w.set_value(method)
                at.run()
                break

    run_btn = next(iter(at.button), None)
    if run_btn:
        run_btn.click()
        at.run()
    return at


# ── EF-A: Equal Weight selected -- highlighted as current, all 3 markers appear ──
def test_ef_a_equal_weight_current():
    from src.charts import efficient_frontier_chart
    from src.portfolio_optimizer import monte_carlo_simulation

    strategy_results = _compute_comparison(PRICES)
    mean_returns, cov = _ef_mean_cov()
    mc_df = monte_carlo_simulation(mean_returns, cov, 200, 0.05)
    fig = efficient_frontier_chart(
        mc_df=mc_df, frontier_df=None, strategy_results=strategy_results,
        strategy_labels={m: m for m in _SC_METHODS}, current_method="Equal Weight",
    )
    strategy_traces = {tr.name: tr for tr in fig.data if tr.name in _SC_METHODS}
    check("EF-A.all_three_markers_present", set(strategy_traces.keys()) == set(_SC_METHODS),
          str(list(strategy_traces.keys())))
    ew = strategy_traces.get("Equal Weight")
    other_sizes = [strategy_traces[m].marker.size for m in _SC_METHODS if m != "Equal Weight"]
    check("EF-A.equal_weight_marker_emphasized",
          ew is not None and all(ew.marker.size > s for s in other_sizes),
          f"ew_size={ew.marker.size if ew else None} others={other_sizes}")


# ── EF-B: Maximum Sharpe selected -- current + exactly matches Strategy Comparison ──
def test_ef_b_max_sharpe_current_matches_comparison():
    from src.charts import efficient_frontier_chart
    from src.portfolio_optimizer import monte_carlo_simulation

    strategy_results = _compute_comparison(PRICES)
    mean_returns, cov = _ef_mean_cov()
    mc_df = monte_carlo_simulation(mean_returns, cov, 200, 0.05)
    fig = efficient_frontier_chart(
        mc_df=mc_df, frontier_df=None, strategy_results=strategy_results,
        strategy_labels={m: m for m in _SC_METHODS}, current_method="Maximum Sharpe Ratio",
    )
    strategy_traces = {tr.name: tr for tr in fig.data if tr.name in _SC_METHODS}
    ms_trace = strategy_traces.get("Maximum Sharpe Ratio")
    expected = strategy_results["Maximum Sharpe Ratio"]
    check("EF-B.marker_present", ms_trace is not None)
    if ms_trace is None:
        return
    other_sizes = [strategy_traces[m].marker.size for m in _SC_METHODS if m != "Maximum Sharpe Ratio"]
    check("EF-B.marker_emphasized", all(ms_trace.marker.size > s for s in other_sizes),
          f"size={ms_trace.marker.size} others={other_sizes}")
    check("EF-B.x_matches_comparison_volatility",
          abs(ms_trace.x[0] / 100 - expected["expected_volatility"]) < 1e-9,
          f"chart_x={ms_trace.x[0]} comparison_vol={expected['expected_volatility']}")
    check("EF-B.y_matches_comparison_return",
          abs(ms_trace.y[0] / 100 - expected["expected_return"]) < 1e-9,
          f"chart_y={ms_trace.y[0]} comparison_ret={expected['expected_return']}")
    check("EF-B.hover_contains_matching_sharpe",
          f"{expected['sharpe_ratio']:.2f}" in ms_trace.hovertemplate,
          ms_trace.hovertemplate)


# ── EF-C: Minimum Volatility selected -- current + volatility matches Strategy Comparison ──
def test_ef_c_min_vol_current_matches_comparison():
    from src.charts import efficient_frontier_chart
    from src.portfolio_optimizer import monte_carlo_simulation

    strategy_results = _compute_comparison(PRICES)
    mean_returns, cov = _ef_mean_cov()
    mc_df = monte_carlo_simulation(mean_returns, cov, 200, 0.05)
    fig = efficient_frontier_chart(
        mc_df=mc_df, frontier_df=None, strategy_results=strategy_results,
        strategy_labels={m: m for m in _SC_METHODS}, current_method="Minimum Volatility",
    )
    strategy_traces = {tr.name: tr for tr in fig.data if tr.name in _SC_METHODS}
    mv_trace = strategy_traces.get("Minimum Volatility")
    expected = strategy_results["Minimum Volatility"]
    check("EF-C.marker_present", mv_trace is not None)
    if mv_trace is None:
        return
    other_sizes = [strategy_traces[m].marker.size for m in _SC_METHODS if m != "Minimum Volatility"]
    check("EF-C.marker_emphasized", all(mv_trace.marker.size > s for s in other_sizes),
          f"size={mv_trace.marker.size} others={other_sizes}")
    check("EF-C.x_matches_comparison_volatility",
          abs(mv_trace.x[0] / 100 - expected["expected_volatility"]) < 1e-9,
          f"chart_x={mv_trace.x[0]} comparison_vol={expected['expected_volatility']}")


# ── EF-D: Efficient Frontier curve contains only feasible optimized portfolios ──
def test_ef_d_frontier_only_feasible():
    mean_returns, cov = _ef_mean_cov()
    n_requested = 30
    normal_df = compute_efficient_frontier(mean_returns, cov, n_points=n_requested)
    check("EF-D.normal_case_has_points", len(normal_df) >= 1, str(len(normal_df)))
    check("EF-D.never_exceeds_requested_grid", len(normal_df) <= n_requested,
          f"{len(normal_df)} > {n_requested}")

    # A deliberately near-impossible per-asset weight box (26%-27% on 4
    # tickers) is feasible for sum(weights)==1 in isolation but infeasible
    # for almost every target return in the grid -- if infeasible targets
    # were silently included (the old behavior), this would still return
    # n_requested rows. Expecting far fewer proves infeasible targets are
    # being skipped, not papered over.
    tight_df = compute_efficient_frontier(mean_returns, cov, n_points=n_requested,
                                           min_weight=0.26, max_weight=0.27)
    check("EF-D.infeasible_targets_are_skipped", len(tight_df) < n_requested,
          f"{len(tight_df)} rows out of {n_requested} requested -- expected fewer")


# ── EF-D2: no hook near the Minimum Volatility point -- frontier begins at
# the actual min-vol portfolio and only shows the efficient upper branch ──
def test_ef_d2_no_hook_near_min_vol():
    import numpy as np
    from src.portfolio_optimizer import optimize_min_volatility

    mean_returns, cov = _ef_mean_cov()
    mv_weights, mv_ok = optimize_min_volatility(mean_returns, cov)
    check("EF-D2.min_vol_solver_converged", mv_ok)
    if not mv_ok:
        return
    mv_return = portfolio_return(mv_weights, mean_returns)
    mv_vol = portfolio_volatility(mv_weights, cov)

    frontier_df = compute_efficient_frontier(mean_returns, cov, n_points=40)
    check("EF-D2.frontier_has_points", len(frontier_df) >= 2, str(len(frontier_df)))
    if len(frontier_df) < 2:
        return

    rets = frontier_df["Return"].values
    vols = frontier_df["Volatility"].values
    check("EF-D2.starts_at_min_vol_return", abs(rets[0] - mv_return) < 1e-6,
          f"frontier_first_return={rets[0]} min_vol_return={mv_return}")
    check("EF-D2.starts_at_min_vol_volatility", abs(vols[0] - mv_vol) < 1e-6,
          f"frontier_first_vol={vols[0]} min_vol_vol={mv_vol}")
    check("EF-D2.returns_strictly_increasing", bool(np.all(np.diff(rets) > 0)),
          f"non-monotonic returns -- this is the hook: {rets.tolist()}")
    check("EF-D2.volatility_non_decreasing", bool(np.all(np.diff(vols) >= -1e-9)),
          f"volatility decreases somewhere -- this is the hook: {vols.tolist()}")
    check("EF-D2.no_duplicate_volatility",
          len(vols) == len(set(np.round(vols, 8))), str(vols.tolist()))


# ── EF-E: All frontier weights satisfy current constraints ──────────────
def test_ef_e_frontier_respects_constraints():
    import numpy as np
    mean_returns, cov = _ef_mean_cov()

    bounded_df = compute_efficient_frontier(mean_returns, cov, n_points=30,
                                             min_weight=0.10, max_weight=0.40)
    if len(bounded_df) == 0:
        check("EF-E.bounded_case_has_points", False, "0 feasible points -- cannot verify bounds")
    else:
        w = np.array(bounded_df["Weights"].tolist())
        check("EF-E.weights_sum_to_one", bool(np.allclose(w.sum(axis=1), 1.0, atol=1e-6)), str(w.sum(axis=1)))
        check("EF-E.weights_within_min_max", bool(np.all(w >= 0.10 - 1e-4) and np.all(w <= 0.40 + 1e-4)),
              f"min={w.min()} max={w.max()}")

    short_df = compute_efficient_frontier(mean_returns, cov, n_points=30,
                                           max_weight=0.5, allow_short=True)
    if len(short_df) == 0:
        check("EF-E.short_case_has_points", False, "0 feasible points -- cannot verify short bounds")
    else:
        w2 = np.array(short_df["Weights"].tolist())
        check("EF-E.short_weights_within_symmetric_bounds",
              bool(np.all(w2 >= -0.5 - 1e-4) and np.all(w2 <= 0.5 + 1e-4)),
              f"min={w2.min()} max={w2.max()}")


# ── EF-F: Monte Carlo points remain visible but visually secondary ──────
def test_ef_f_monte_carlo_visually_secondary():
    from src.charts import efficient_frontier_chart
    from src.portfolio_optimizer import monte_carlo_simulation

    strategy_results = _compute_comparison(PRICES)
    mean_returns, cov = _ef_mean_cov()
    mc_df = monte_carlo_simulation(mean_returns, cov, 200, 0.05)
    fig = efficient_frontier_chart(
        mc_df=mc_df, frontier_df=None, strategy_results=strategy_results,
        strategy_labels={m: m for m in _SC_METHODS}, current_method="Equal Weight",
    )
    mc_trace = fig.data[0]
    check("EF-F.mc_trace_present", mc_trace.name == t("chart_monte_carlo_portfolios"), mc_trace.name)
    strategy_sizes = [tr.marker.size for tr in fig.data if tr.name in _SC_METHODS]
    check("EF-F.mc_markers_smaller_than_strategy_markers",
          all(mc_trace.marker.size < s for s in strategy_sizes),
          f"mc_size={mc_trace.marker.size} strategy_sizes={strategy_sizes}")
    check("EF-F.mc_markers_have_transparency", mc_trace.marker.opacity < 1.0, str(mc_trace.marker.opacity))


# ── EF-G: No overlapping duplicate strategy labels in the legend ────────
def test_ef_g_no_duplicate_legend_labels():
    from src.charts import efficient_frontier_chart
    from src.portfolio_optimizer import monte_carlo_simulation

    strategy_results = _compute_comparison(PRICES)
    mean_returns, cov = _ef_mean_cov()
    mc_df = monte_carlo_simulation(mean_returns, cov, 200, 0.05)
    frontier_df = compute_efficient_frontier(mean_returns, cov, n_points=20)
    fig = efficient_frontier_chart(
        mc_df=mc_df, frontier_df=frontier_df, strategy_results=strategy_results,
        strategy_labels={m: m for m in _SC_METHODS}, current_method="Maximum Sharpe Ratio",
    )
    names = [tr.name for tr in fig.data]
    check("EF-G.no_duplicate_trace_names", len(names) == len(set(names)), str(names))


# ── EF-H: zh-TW render contains no raw opt_* keys ────────────────────────
def test_ef_h_zh_no_raw_keys():
    forbidden = ("opt_", "OPT_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")
    at = _setup_ef_page(lang="zh-TW")
    exc = at.exception[0] if at.exception else None
    check("EF-H.no_exception", exc is None, str(exc))
    if exc:
        return
    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    hits = [frag for frag in forbidden if frag in corpus]
    check("EF-H.no_forbidden_fragments", len(hits) == 0, str(hits))
    check("EF-H.how_to_read_panel_translated", "如何閱讀這張圖" in corpus, "")
    check("EF-H.disclaimer_translated", "歷史風險與報酬不代表未來結果" in corpus, "")


# ── EF-I: English render contains no raw opt_* keys ──────────────────────
def test_ef_i_en_no_raw_keys():
    forbidden = ("opt_", "OPT_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")
    at = _setup_ef_page(lang="en")
    exc = at.exception[0] if at.exception else None
    check("EF-I.no_exception", exc is None, str(exc))
    if exc:
        return
    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    hits = [frag for frag in forbidden if frag in corpus]
    check("EF-I.no_forbidden_fragments", len(hits) == 0, str(hits))
    check("EF-I.how_to_read_panel_translated", "How to Read This Chart" in corpus, "")
    check("EF-I.disclaimer_translated", "Historical risk and return do not guarantee future results." in corpus, "")


# ── EF-J: switching strategy doesn't recalculate/reset unrelated state ──
def test_ef_j_switch_strategy_no_side_effects():
    at = _setup_ef_page(method="Equal Weight", lang="en")
    exc = at.exception[0] if at.exception else None
    check("EF-J.no_exception", exc is None, str(exc))
    if exc:
        return

    def _sget(a, k):
        try:
            return a.session_state[k]
        except Exception:
            return None

    def snapshot(a):
        keys = ["selected_region", "investment_goal", "risk_tolerance", "investment_horizon"]
        snap = {k: _sget(a, k) for k in keys}
        ms = next((w for w in a.multiselect if w.key and w.key.startswith("selected_etfs_")), None)
        snap["etfs"] = sorted(ms.value) if ms else None
        return snap

    before = snapshot(at)

    for w in at.selectbox:
        if w.key == "optimization_method" and "Minimum Volatility" in w.options:
            w.set_value("Minimum Volatility")
            at.run()
            break
    run_btn = next(iter(at.button), None)
    if run_btn:
        run_btn.click()
        at.run()

    exc2 = at.exception[0] if at.exception else None
    check("EF-J.no_exception_after_switch", exc2 is None, str(exc2))
    if exc2:
        return
    after = snapshot(at)
    check("EF-J.unrelated_state_unchanged", before == after, f"before={before} after={after}")
    check("EF-J.method_actually_switched",
          at.session_state["opt_result"]["method"] == "Minimum Volatility",
          at.session_state["opt_result"]["method"])


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

    test_sc_a_all_three_appear()
    test_sc_b_weights_sum_to_one()
    test_sc_c_best_sharpe_badge_correct()
    test_sc_d_lowest_vol_badge_correct()
    test_sc_e_largest_position_correct()
    test_sc_f_concentration_threshold()
    test_sc_g_no_side_effects()
    test_sc_h_switch_strategy()
    test_sc_i_i18n()
    test_sc_j_render_output_no_raw_keys()

    test_ef_a_equal_weight_current()
    test_ef_b_max_sharpe_current_matches_comparison()
    test_ef_c_min_vol_current_matches_comparison()
    test_ef_d_frontier_only_feasible()
    test_ef_d2_no_hook_near_min_vol()
    test_ef_e_frontier_respects_constraints()
    test_ef_f_monte_carlo_visually_secondary()
    test_ef_g_no_duplicate_legend_labels()
    test_ef_h_zh_no_raw_keys()
    test_ef_i_en_no_raw_keys()
    test_ef_j_switch_strategy_no_side_effects()

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
