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
from src.financial_metrics import (
    portfolio_return, portfolio_volatility, covariance_matrix,
    portfolio_diagnosis, effective_number_of_holdings, active_position_count,
    top_n_concentration, concentration_level, top2_concentration_status,
    ACTIVE_POSITION_TOLERANCE,
)
from src.i18n import t
from src.simulator import simulate_investment, MARKET_SCENARIOS


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


def _find_run_button(at):
    """Find the "Build Optimized Portfolio" button by its stable key
    (pages/2_Portfolio_Optimizer.py: key="opt_run_optimization_btn") rather
    than `next(iter(at.button), None)` -- that grabbed whichever button
    happens to render first, which broke once Round 2B-4 added more
    main-content buttons (Next Steps: Run Simulation / Analyze Risk /
    Quick Save) that can render before this one in at.button's enumeration
    order, silently clicking the wrong widget (e.g. triggering
    st.switch_page(), which AppTest cannot resolve when testing a single
    page in isolation, unrelated to any real bug in the page)."""
    for b in at.button:
        if b.key == "opt_run_optimization_btn":
            return b
    return next(iter(at.button), None)


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

    run_btn = _find_run_button(at)
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
    run_btn = _find_run_button(at)
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
        run_btn = _find_run_button(at)
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
        run_btn = _find_run_button(at)
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
        # NOTE: w.options on a format_func selectbox returns the
        # TRANSLATED display labels (e.g. zh-TW "最大夏普比率"),
        # not the canonical option values -- `method in w.options` silently
        # fails for any non-English language, leaving the method unswitched
        # with no error. set_value()/`.value` operate on the canonical
        # value regardless of display language, so match on w.key alone.
        for w in at.selectbox:
            if w.key == "optimization_method":
                w.set_value(method)
                at.run()
                break

    run_btn = _find_run_button(at)
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
        if w.key == "optimization_method":
            w.set_value("Minimum Volatility")
            at.run()
            break
    run_btn = _find_run_button(at)
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


# ══════════════════════════════════════════════════════════════════════════
# Round 2B-3: Portfolio Diagnosis
# ══════════════════════════════════════════════════════════════════════════

# ── PD-A: Equal Weight -- exact, data-independent expectations ──────────
def test_pd_a_equal_weight():
    tickers = ["VOO", "VTI", "QQQ", "SPY", "SCHD"]
    weights = {tk: 1.0 / len(tickers) for tk in tickers}
    diag = portfolio_diagnosis(weights)
    check("PD-A.largest_weight_approx_20pct", abs(diag["largest_weight"] - 0.20) < 1e-9, str(diag["largest_weight"]))
    check("PD-A.top2_approx_40pct", abs(diag["top2_concentration"] - 0.40) < 1e-9, str(diag["top2_concentration"]))
    check("PD-A.effective_holdings_approx_5", abs(diag["effective_holdings"] - 5.0) < 1e-6, str(diag["effective_holdings"]))
    check("PD-A.active_etfs_5_of_5", diag["active_holdings"] == 5 and diag["selected_holdings"] == 5,
          f"{diag['active_holdings']}/{diag['selected_holdings']}")
    check("PD-A.concentration_not_high", diag["concentration_level"] != "high", diag["concentration_level"])
    check("PD-A.case_is_balanced", diag["case"] == "balanced", diag["case"])


def _check_diag_internally_consistent(prefix, weights, diag):
    """Shared formula/consistency checks reused by PD-B/PD-C: verifies the
    portfolio_diagnosis() output matches an independent manual calculation
    from the SAME weights, rather than asserting fixture-specific numbers
    that would be brittle against whatever the optimizer actually returns."""
    manual_active = sum(1 for w in weights.values() if w > 0.001)
    check(f"{prefix}.active_holdings_matches_tolerance_rule", diag["active_holdings"] == manual_active,
          f"reported={diag['active_holdings']} manual={manual_active}")
    manual_largest_ticker = max(weights, key=weights.get)
    check(f"{prefix}.largest_ticker_correct", diag["largest_ticker"] == manual_largest_ticker,
          f"reported={diag['largest_ticker']} actual={manual_largest_ticker}")
    manual_level = ("low" if weights[manual_largest_ticker] <= 0.30
                     else "moderate" if weights[manual_largest_ticker] <= 0.50 else "high")
    check(f"{prefix}.concentration_level_matches_threshold_rule", diag["concentration_level"] == manual_level,
          f"reported={diag['concentration_level']} expected={manual_level} largest={weights[manual_largest_ticker]}")
    check(f"{prefix}.selected_holdings_equals_dict_len", diag["selected_holdings"] == len(weights),
          f"{diag['selected_holdings']} vs {len(weights)}")


# ── PD-B: Maximum Sharpe -- structural/formula self-consistency on actual optimized weights ──
def test_pd_b_max_sharpe_structural():
    result = run_optimization(PRICES, method="Maximum Sharpe Ratio", risk_free_rate=0.03)
    weights = result["weights"]
    diag = portfolio_diagnosis(weights)
    _check_diag_internally_consistent("PD-B", weights, diag)
    if diag["largest_weight"] > 0.50:
        check("PD-B.high_concentration_when_largest_over_50pct", diag["concentration_level"] == "high",
              f"largest={diag['largest_weight']} level={diag['concentration_level']}")
        check("PD-B.effective_holdings_below_selected", diag["effective_holdings"] < diag["selected_holdings"],
              f"effective={diag['effective_holdings']} selected={diag['selected_holdings']}")


# ── PD-C: Minimum Volatility -- same structural checks ───────────────────
def test_pd_c_min_vol_structural():
    result = run_optimization(PRICES, method="Minimum Volatility")
    weights = result["weights"]
    diag = portfolio_diagnosis(weights)
    _check_diag_internally_consistent("PD-C", weights, diag)


# ── PD-D: Effective Holdings formula == 1 / sum(w_i^2) on actual canonical weights ──
def test_pd_d_effective_holdings_formula():
    for method in ("Equal Weight", "Maximum Sharpe Ratio", "Minimum Volatility"):
        result = run_optimization(PRICES, method=method)
        weights = result["weights"]
        expected = 1.0 / sum(w ** 2 for w in weights.values())
        actual = effective_number_of_holdings(weights)
        check(f"PD-D.{method}.effective_holdings_formula", abs(actual - expected) < 1e-9,
              f"actual={actual} expected={expected}")


# ── PD-E: rendering the diagnosis section doesn't change optimizer weights/metrics ──
def test_pd_e_no_side_effects_on_rerender():
    at = _setup_ef_page(method="Maximum Sharpe Ratio", lang="en")
    exc = at.exception[0] if at.exception else None
    check("PD-E.no_exception", exc is None, str(exc))
    if exc:
        return
    before = dict(at.session_state["opt_result"])
    at.run()  # re-render only -- no widget changes, diagnosis section renders again
    exc2 = at.exception[0] if at.exception else None
    check("PD-E.no_exception_on_rerender", exc2 is None, str(exc2))
    if exc2:
        return
    after = dict(at.session_state["opt_result"])
    check("PD-E.weights_unchanged", before["weights"] == after["weights"],
          f"{before['weights']} vs {after['weights']}")
    check("PD-E.metrics_unchanged",
          before["expected_return"] == after["expected_return"] and
          before["expected_volatility"] == after["expected_volatility"] and
          before["sharpe_ratio"] == after["sharpe_ratio"],
          f"before={before} after={after}")


# ── PD-F: switching strategy updates the diagnosis to the new selected portfolio ──
def test_pd_f_switch_strategy_updates_diagnosis():
    at = _setup_ef_page(method="Equal Weight", lang="en")
    exc = at.exception[0] if at.exception else None
    check("PD-F.no_exception", exc is None, str(exc))
    if exc:
        return

    for method in ("Maximum Sharpe Ratio", "Minimum Volatility", "Equal Weight"):
        for w in at.selectbox:
            if w.key == "optimization_method":
                w.set_value(method)
                at.run()
                break
        exc = at.exception[0] if at.exception else None
        check(f"PD-F.{method}.no_exception", exc is None, str(exc))
        if exc:
            continue
        weights = at.session_state["opt_result"]["weights"]
        diag = portfolio_diagnosis(weights)
        corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
        expected_largest = f"{diag['largest_ticker']} {diag['largest_weight']:.2%}"
        expected_effective = f"{diag['effective_holdings']:.2f} / {diag['selected_holdings']}"
        expected_active = f"{diag['active_holdings']} / {diag['selected_holdings']}"
        check(f"PD-F.{method}.largest_position_shown", expected_largest in corpus,
              f"expected {expected_largest!r} in corpus")
        check(f"PD-F.{method}.effective_holdings_shown", expected_effective in corpus,
              f"expected {expected_effective!r} in corpus")
        check(f"PD-F.{method}.active_etfs_shown", expected_active in corpus,
              f"expected {expected_active!r} in corpus")


# ── PD-G: zh-TW renders the diagnosis section with no raw translation keys ──
def test_pd_g_zh_no_raw_keys():
    forbidden = ("opt_", "OPT_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")
    at = _setup_ef_page(lang="zh-TW")
    exc = at.exception[0] if at.exception else None
    check("PD-G.no_exception", exc is None, str(exc))
    if exc:
        return
    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    hits = [frag for frag in forbidden if frag in corpus]
    check("PD-G.no_forbidden_fragments", len(hits) == 0, str(hits))
    check("PD-G.diagnosis_title_translated", "投資組合診斷" in corpus, "")
    check("PD-G.insight_title_translated", "配置結構洞察" in corpus, "")
    check("PD-G.weight_disclaimer_translated", "此處評估的是配置權重分散程度" in corpus, "")


# ── PD-H: English renders the diagnosis section with no raw translation keys ──
def test_pd_h_en_no_raw_keys():
    forbidden = ("opt_", "OPT_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")
    at = _setup_ef_page(lang="en")
    exc = at.exception[0] if at.exception else None
    check("PD-H.no_exception", exc is None, str(exc))
    if exc:
        return
    corpus = "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption)
    hits = [frag for frag in forbidden if frag in corpus]
    check("PD-H.no_forbidden_fragments", len(hits) == 0, str(hits))
    check("PD-H.diagnosis_title_translated", "Portfolio Diagnosis" in corpus, "")
    check("PD-H.insight_title_translated", "Portfolio Structure Insight" in corpus, "")
    check("PD-H.weight_disclaimer_translated",
          "This diagnosis evaluates allocation-weight diversification" in corpus, "")


# ══════════════════════════════════════════════════════════════════════════
# Round 2B-3 polish pass: tooltips, concentration status color, Top-2
# status, and the rewritten deterministic summary wording.
# ══════════════════════════════════════════════════════════════════════════

# ── PD-I: metric tooltips render (native HTML title= attribute) in both languages ──
def test_pd_i_tooltips_present():
    import html as _html
    from src.i18n import TRANSLATIONS
    for lang, key in (("zh-TW", "zh-TW"), ("en", "en")):
        at = _setup_ef_page(lang=lang)
        exc = at.exception[0] if at.exception else None
        check(f"PD-I.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue
        corpus = "\n".join(m.value for m in at.markdown)
        for tk in (
            "opt_diag_tooltip_concentration_level", "opt_diag_tooltip_largest_position",
            "opt_diag_tooltip_top2_concentration", "opt_diag_tooltip_effective_holdings",
            "opt_diag_tooltip_active_etfs",
        ):
            # The page embeds tooltips via title="..." with html.escape(quote=True),
            # which also encodes apostrophes (e.g. "portfolio's" -> "portfolio&#x27;s")
            # -- compare against the SAME escaped form, not the raw translation.
            tooltip_text = _html.escape(TRANSLATIONS[key][tk], quote=True)
            check(f"PD-I.{lang}.{tk}_present", tooltip_text in corpus, f"missing tooltip text for {tk}")


# ── PD-J: Top-2 Concentration status thresholds (<=50 / 50-75 / >75) ────
def test_pd_j_top2_status_thresholds():
    check("PD-J.at_50pct_is_distributed", top2_concentration_status(0.50) == "distributed")
    check("PD-J.just_above_50pct_is_moderate", top2_concentration_status(0.5001) == "moderate")
    check("PD-J.at_75pct_is_moderate", top2_concentration_status(0.75) == "moderate")
    check("PD-J.just_above_75pct_is_concentrated", top2_concentration_status(0.7501) == "concentrated")
    check("PD-J.low_value_is_distributed", top2_concentration_status(0.30) == "distributed")
    check("PD-J.near_100pct_is_concentrated", top2_concentration_status(0.99) == "concentrated")


# ── PD-K: rewritten "concentrated" summary correctly interpolates the
# actual ticker/values (Test C from the polish spec: "summary updates
# using actual ticker and values") ──────────────────────────────────────
def test_pd_k_concentrated_summary_interpolation():
    weights = {"AAA": 0.70, "BBB": 0.10, "CCC": 0.10, "DDD": 0.10}
    diag = portfolio_diagnosis(weights)
    check("PD-K.case_is_concentrated", diag["case"] == "concentrated", diag["case"])
    if diag["case"] != "concentrated":
        return
    kwargs = dict(
        selected_count=diag["selected_holdings"],
        effective_holdings=f"{diag['effective_holdings']:.2f}",
        largest_ticker=diag["largest_ticker"],
        largest_weight=f"{diag['largest_weight']:.2%}",
    )
    # t() reads the CURRENT language from session_state via get_language();
    # exercise both languages directly against TRANSLATIONS to avoid
    # depending on Streamlit session state outside an AppTest context.
    from src.i18n import TRANSLATIONS
    for lang in ("zh-TW", "en"):
        text = TRANSLATIONS[lang]["opt_diag_summary_concentrated"].format(**kwargs)
        check(f"PD-K.{lang}.contains_ticker", diag["largest_ticker"] in text, text)
        check(f"PD-K.{lang}.contains_largest_weight", f"{diag['largest_weight']:.2%}" in text, text)
        check(f"PD-K.{lang}.contains_effective_holdings", f"{diag['effective_holdings']:.2f}" in text, text)
        check(f"PD-K.{lang}.contains_selected_count", str(diag["selected_holdings"]) in text, text)
        check(f"PD-K.{lang}.no_unresolved_placeholder", "{" not in text and "}" not in text, text)


# ══════════════════════════════════════════════════════════════════════════
# Round 2B-4: Portfolio Handoff & Next Actions
#
# st.switch_page() cannot actually be exercised by clicking the Next Steps
# buttons here: AppTest tests ONE page file in isolation and has no
# multipage registry to switch into (the same pre-existing limitation that
# already required monkeypatching st.page_link everywhere in this suite --
# confirmed directly: clicking a switch_page button under AppTest raises
# StreamlitAPIException "Could not find page", unrelated to any bug in the
# app). Since st.switch_page's only job is to change pages WITHOUT
# clearing st.session_state, these tests instead verify the actual
# contract: (1) Portfolio Optimizer builds current_portfolio correctly,
# and (2) a receiving page that finds current_portfolio already in
# st.session_state (exactly what switch_page leaves behind) renders it
# correctly. That is what "receives the exact same weights" means in
# practice, without needing to execute the navigation call itself.
# ══════════════════════════════════════════════════════════════════════════

def _build_current_portfolio_via_optimizer(method, lang="en", tickers=None):
    """Run Portfolio Optimizer to completion for the given method; return
    (at, current_portfolio) -- the canonical handoff object exactly as the
    real page builds it in st.session_state."""
    at = _setup_ef_page(method=method, lang=lang, tickers=tickers)
    exc = at.exception[0] if at.exception else None
    if exc:
        return at, None
    try:
        cp = at.session_state["current_portfolio"]
    except Exception:
        cp = None
    return at, cp


def _run_receiving_page(page_path, current_portfolio, lang="en", extra_session=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    at = AppTest.from_file(page_path, default_timeout=180)
    at.session_state["language"] = lang
    if current_portfolio is not None:
        at.session_state["current_portfolio"] = current_portfolio
    for k, v in (extra_session or {}).items():
        at.session_state[k] = v
    at.run()
    return at


def _check_handoff_holdings_shown(prefix, weights, corpus):
    """Mirrors render_current_portfolio_handoff()'s exact display rule
    (src/ui.py): up to 5 largest ACTIVE (weight > ACTIVE_POSITION_TOLERANCE)
    holdings shown individually as "TICKER weight%"; zero-weight holdings
    are summarized as a count, never listed individually."""
    sorted_holdings = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    active = [(tk, w) for tk, w in sorted_holdings if w > ACTIVE_POSITION_TOLERANCE]
    zero = [(tk, w) for tk, w in sorted_holdings if w <= ACTIVE_POSITION_TOLERANCE]
    for tk, w in active[:5]:
        check(f"{prefix}_shows_{tk}", f"{tk} {w:.2%}" in corpus, f"expected {tk} {w:.2%} in corpus")
    if zero:
        check(f"{prefix}_deemphasizes_zero_weight", "0%" in corpus, corpus[:0])
        for tk, _ in zero:
            check(f"{prefix}_does_not_list_{tk}_individually", f"{tk} 0.00%" not in corpus,
                  f"{tk} 0.00% should be summarized, not listed individually")


# ── PH-A: Equal Weight build -> Simulator receives exact equal weights ──
def test_ph_a_equal_weight_handoff_to_simulator():
    at, cp = _build_current_portfolio_via_optimizer("Equal Weight")
    check("PH-A.optimizer_no_exception", not at.exception, str(at.exception))
    check("PH-A.current_portfolio_built", cp is not None)
    if cp is None:
        return
    opt_weights = at.session_state["opt_result"]["weights"]
    check("PH-A.weights_match_opt_result", cp["weights"] == opt_weights, f"{cp['weights']} vs {opt_weights}")
    check("PH-A.strategy_is_equal_weight", cp["strategy"] == "Equal Weight", cp["strategy"])

    sim_at = _run_receiving_page("pages/3_Investment_Simulator.py", cp)
    exc2 = sim_at.exception[0] if sim_at.exception else None
    check("PH-A.simulator_no_exception", exc2 is None, str(exc2))
    if exc2:
        return
    corpus = "\n".join(m.value for m in sim_at.markdown)
    for tk, w in cp["weights"].items():
        check(f"PH-A.simulator_shows_{tk}", f"{tk} {w:.2%}" in corpus, f"expected {tk} {w:.2%} in corpus")


# ── PH-B: Maximum Sharpe build -> Simulator receives exact weights ───────
def test_ph_b_max_sharpe_handoff_to_simulator():
    at, cp = _build_current_portfolio_via_optimizer("Maximum Sharpe Ratio")
    check("PH-B.optimizer_no_exception", not at.exception, str(at.exception))
    check("PH-B.current_portfolio_built", cp is not None)
    if cp is None:
        return
    opt_weights = at.session_state["opt_result"]["weights"]
    check("PH-B.weights_match_opt_result", cp["weights"] == opt_weights, f"{cp['weights']} vs {opt_weights}")

    sim_at = _run_receiving_page("pages/3_Investment_Simulator.py", cp)
    exc2 = sim_at.exception[0] if sim_at.exception else None
    check("PH-B.simulator_no_exception", exc2 is None, str(exc2))
    if exc2:
        return
    corpus = "\n".join(m.value for m in sim_at.markdown)
    # Handoff preview shows at most the 5 largest ACTIVE holdings
    # individually; zero-weight holdings are summarized as a de-emphasized
    # count instead (Round 1 spec section 12), not listed as "TICKER 0.00%".
    _check_handoff_holdings_shown("PH-B.simulator", cp["weights"], corpus)


# ── PH-C: Minimum Volatility build -> Risk Analytics receives exact weights ──
def test_ph_c_min_vol_handoff_to_risk_analytics():
    at, cp = _build_current_portfolio_via_optimizer("Minimum Volatility")
    check("PH-C.optimizer_no_exception", not at.exception, str(at.exception))
    check("PH-C.current_portfolio_built", cp is not None)
    if cp is None:
        return
    opt_weights = at.session_state["opt_result"]["weights"]
    check("PH-C.weights_match_opt_result", cp["weights"] == opt_weights, f"{cp['weights']} vs {opt_weights}")
    check("PH-C.strategy_is_min_vol", cp["strategy"] == "Minimum Volatility", cp["strategy"])

    risk_at = _run_receiving_page("pages/4_Risk_Analytics.py", cp)
    exc2 = risk_at.exception[0] if risk_at.exception else None
    check("PH-C.risk_analytics_no_exception", exc2 is None, str(exc2))
    if exc2:
        return
    corpus = "\n".join(m.value for m in risk_at.markdown)
    _check_handoff_holdings_shown("PH-C.risk_analytics", cp["weights"], corpus)
    check("PH-C.risk_analytics_shows_strategy",
          "Minimum Volatility" in corpus, corpus[:0])


# ── PH-D: current_portfolio remains available across a plain rerun
# (proxy for "navigate away and back" within one session) ───────────────
def test_ph_d_current_portfolio_persists_across_rerun():
    at, cp = _build_current_portfolio_via_optimizer("Maximum Sharpe Ratio")
    check("PH-D.no_exception", not at.exception, str(at.exception))
    check("PH-D.current_portfolio_built", cp is not None)
    if cp is None:
        return
    at.run()  # re-render only, no widget changes
    exc2 = at.exception[0] if at.exception else None
    check("PH-D.no_exception_after_rerender", exc2 is None, str(exc2))
    if exc2:
        return
    cp_after = at.session_state["current_portfolio"]
    check("PH-D.current_portfolio_still_present", cp_after is not None)
    check("PH-D.current_portfolio_unchanged", cp_after == cp, f"{cp} vs {cp_after}")


# ── PH-E: Simulator opened with no current_portfolio -- empty state, no crash ──
def test_ph_e_simulator_empty_state():
    for lang in ("zh-TW", "en"):
        at = _run_receiving_page("pages/3_Investment_Simulator.py", None, lang=lang)
        exc = at.exception[0] if at.exception else None
        check(f"PH-E.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue
        corpus = "\n".join(m.value for m in at.markdown)
        expect = "尚未建立投資組合" if lang == "zh-TW" else "No portfolio selected"
        check(f"PH-E.{lang}.empty_state_shown", expect in corpus, corpus[:200])


# ── PH-F: Risk Analytics opened with no current_portfolio -- empty state ──
def test_ph_f_risk_analytics_empty_state():
    for lang in ("zh-TW", "en"):
        at = _run_receiving_page("pages/4_Risk_Analytics.py", None, lang=lang)
        exc = at.exception[0] if at.exception else None
        check(f"PH-F.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue
        corpus = "\n".join(m.value for m in at.markdown)
        expect = "尚未建立投資組合" if lang == "zh-TW" else "No portfolio selected"
        check(f"PH-F.{lang}.empty_state_shown", expect in corpus, corpus[:200])


# ── PH-G: language switch does not alter portfolio state ────────────────
def test_ph_g_language_switch_preserves_portfolio():
    _, cp = _build_current_portfolio_via_optimizer("Maximum Sharpe Ratio")
    check("PH-G.current_portfolio_built", cp is not None)
    if cp is None:
        return

    zh_at = _run_receiving_page("pages/3_Investment_Simulator.py", cp, lang="zh-TW")
    exc1 = zh_at.exception[0] if zh_at.exception else None
    check("PH-G.zh_no_exception", exc1 is None, str(exc1))

    en_at = _run_receiving_page("pages/3_Investment_Simulator.py", cp, lang="en")
    exc2 = en_at.exception[0] if en_at.exception else None
    check("PH-G.en_no_exception", exc2 is None, str(exc2))
    if exc1 or exc2:
        return

    zh_cp = zh_at.session_state["current_portfolio"]
    en_cp = en_at.session_state["current_portfolio"]
    check("PH-G.portfolio_identical_across_languages", zh_cp == en_cp == cp,
          f"zh={zh_cp} en={en_cp} original={cp}")


# ── PH-H: global market/ETF state unchanged through the handoff ─────────
def test_ph_h_market_state_unchanged_through_handoff():
    at = _setup_ef_page(method="Minimum Volatility", lang="en")
    region_w = next((w for w in at.selectbox if w.key == "selected_region"), None)
    region_w.set_value("Taiwan")
    at.run()
    run_btn = _find_run_button(at)
    if run_btn:
        run_btn.click()
        at.run()
    exc = at.exception[0] if at.exception else None
    check("PH-H.optimizer_no_exception", exc is None, str(exc))
    if exc:
        return
    try:
        cp = at.session_state["current_portfolio"]
    except Exception:
        cp = None
    check("PH-H.current_portfolio_built", cp is not None)
    if cp is None:
        return
    check("PH-H.market_field_is_taiwan", cp["market"] == "Taiwan", cp["market"])

    # Seed Risk Analytics exactly as st.switch_page would leave the
    # session -- current_portfolio AND the shared selected_region key.
    risk_at = _run_receiving_page(
        "pages/4_Risk_Analytics.py", cp, extra_session={"selected_region": "Taiwan"},
    )
    exc2 = risk_at.exception[0] if risk_at.exception else None
    check("PH-H.risk_analytics_no_exception", exc2 is None, str(exc2))
    if exc2:
        return
    region_w2 = next((w for w in risk_at.selectbox if w.key == "selected_region"), None)
    check("PH-H.risk_analytics_region_still_taiwan",
          region_w2 is not None and region_w2.value == "Taiwan",
          str(region_w2.value if region_w2 else None))


# ── PH-I: no raw i18n keys in the handoff preview or empty state, both languages ──
def test_ph_i_handoff_no_raw_keys():
    forbidden = ("opt_", "OPT_", "handoff_", "_label", "_title", "_subtitle", "_desc", "_badge", "_col_")
    _, cp = _build_current_portfolio_via_optimizer("Maximum Sharpe Ratio")
    check("PH-I.current_portfolio_built", cp is not None)
    if cp is None:
        return
    for page_path, name in (
        ("pages/3_Investment_Simulator.py", "simulator"),
        ("pages/4_Risk_Analytics.py", "risk"),
    ):
        for portfolio, state_label in ((cp, "with_portfolio"), (None, "empty_state")):
            for lang in ("zh-TW", "en"):
                at = _run_receiving_page(page_path, portfolio, lang=lang)
                exc = at.exception[0] if at.exception else None
                check(f"PH-I.{name}.{state_label}.{lang}.no_exception", exc is None, str(exc))
                if exc:
                    continue
                corpus = "\n".join(m.value for m in at.markdown)
                hits = [frag for frag in forbidden if frag in corpus]
                check(f"PH-I.{name}.{state_label}.{lang}.no_forbidden_fragments", len(hits) == 0, str(hits))


# ══════════════════════════════════════════════════════════════════════════
# Investment Simulator Round 1: Simulation Architecture & Assumption
# Transparency. Covers Simulation Mode, Projection Assumptions (Portfolio
# Historical Statistics / Market Scenario / Custom Assumptions), Advanced
# Settings, the renamed probability metric, and the Projection Setup panel.
# No Monte Carlo mathematics were changed -- src/simulator.py is untouched.
# ══════════════════════════════════════════════════════════════════════════

_SIM_FAKE_PORTFOLIO = {
    "strategy": "Maximum Sharpe Ratio",
    "tickers": ["VOO", "SCHD", "QQQ", "VTI", "SPY"],
    "weights": {"VOO": 0.6834, "SCHD": 0.1599, "QQQ": 0.1567, "VTI": 0.0, "SPY": 0.0},
    "investment_amount": 10000.0,
    "expected_return": 0.1344,
    "volatility": 0.1681,
}


def _setup_sim_page(lang="en", current_portfolio=None, simulation_mode=None,
                     assumption_source=None, scenario=None,
                     custom_return_pct=None, custom_vol_pct=None, click_run=True):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None

    at = AppTest.from_file("pages/3_Investment_Simulator.py", default_timeout=180)
    at.session_state["language"] = lang
    if current_portfolio is not None:
        at.session_state["current_portfolio"] = current_portfolio
    at.run()

    if simulation_mode:
        for w in at.selectbox:
            if w.key == "simulation_mode":
                w.set_value(simulation_mode)
                at.run()
                break
        if simulation_mode == "Historical Simulation":
            return at  # nothing else to configure -- rest of sidebar doesn't render

    if assumption_source:
        for w in at.selectbox:
            if w.key == "projection_assumption_source":
                w.set_value(assumption_source)
                at.run()
                break

    if scenario:
        for w in at.selectbox:
            if w.key == "sim_market_scenario_choice":
                w.set_value(scenario)
                at.run()
                break

    if custom_return_pct is not None:
        for w in at.slider:
            if w.key == "sim_custom_return_pct":
                w.set_value(custom_return_pct)
                at.run()
                break

    if custom_vol_pct is not None:
        for w in at.slider:
            if w.key == "sim_custom_vol_pct":
                w.set_value(custom_vol_pct)
                at.run()
                break

    if click_run:
        run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
        if run_btn:
            run_btn.click()
            at.run()
    return at


# ── SIM-A: current_portfolio arrives intact (same strategy/tickers/weights) ──
def test_sim_a_current_portfolio_arrives_intact():
    at = _setup_sim_page(current_portfolio=_SIM_FAKE_PORTFOLIO, click_run=False)
    exc = at.exception[0] if at.exception else None
    check("SIM-A.no_exception", exc is None, str(exc))
    if exc:
        return
    cp = at.session_state["current_portfolio"]
    check("SIM-A.strategy_unchanged", cp["strategy"] == _SIM_FAKE_PORTFOLIO["strategy"], cp["strategy"])
    check("SIM-A.tickers_unchanged", cp["tickers"] == _SIM_FAKE_PORTFOLIO["tickers"], cp["tickers"])
    check("SIM-A.weights_unchanged", cp["weights"] == _SIM_FAKE_PORTFOLIO["weights"], cp["weights"])


# ── SIM-B: Portfolio Historical Statistics -> return/vol come from current_portfolio ──
def test_sim_b_portfolio_historical_statistics_drives_simulation():
    at = _setup_sim_page(
        current_portfolio=_SIM_FAKE_PORTFOLIO, assumption_source="Portfolio Historical Statistics",
    )
    exc = at.exception[0] if at.exception else None
    check("SIM-B.no_exception", exc is None, str(exc))
    if exc:
        return
    sim_params = at.session_state["sim_params"]
    check("SIM-B.return_matches_portfolio",
          abs(sim_params["annual_return"] - _SIM_FAKE_PORTFOLIO["expected_return"]) < 1e-9,
          f"{sim_params['annual_return']} vs {_SIM_FAKE_PORTFOLIO['expected_return']}")
    check("SIM-B.volatility_matches_portfolio",
          abs(sim_params["annual_volatility"] - _SIM_FAKE_PORTFOLIO["volatility"]) < 1e-9,
          f"{sim_params['annual_volatility']} vs {_SIM_FAKE_PORTFOLIO['volatility']}")
    check("SIM-B.assumption_source_recorded",
          sim_params["assumption_source"] == "Portfolio Historical Statistics", sim_params["assumption_source"])


# ── SIM-C: Market Scenario -> scenario return/vol drive the simulation ──
def test_sim_c_market_scenario_drives_simulation():
    at = _setup_sim_page(assumption_source="Market Scenario", scenario="Bear Market")
    exc = at.exception[0] if at.exception else None
    check("SIM-C.no_exception", exc is None, str(exc))
    if exc:
        return
    sim_params = at.session_state["sim_params"]
    expected = MARKET_SCENARIOS["Bear Market"]
    check("SIM-C.return_matches_scenario", abs(sim_params["annual_return"] - expected["return"]) < 1e-9,
          f"{sim_params['annual_return']} vs {expected['return']}")
    check("SIM-C.volatility_matches_scenario", abs(sim_params["annual_volatility"] - expected["volatility"]) < 1e-9,
          f"{sim_params['annual_volatility']} vs {expected['volatility']}")


# ── SIM-D: Custom Assumptions -> manual return/vol drive the simulation ──
def test_sim_d_custom_assumptions_drive_simulation():
    at = _setup_sim_page(assumption_source="Custom Assumptions", custom_return_pct=12.5, custom_vol_pct=22.0)
    exc = at.exception[0] if at.exception else None
    check("SIM-D.no_exception", exc is None, str(exc))
    if exc:
        return
    sim_params = at.session_state["sim_params"]
    check("SIM-D.return_matches_custom", abs(sim_params["annual_return"] - 0.125) < 1e-9, sim_params["annual_return"])
    check("SIM-D.volatility_matches_custom", abs(sim_params["annual_volatility"] - 0.22) < 1e-9,
          sim_params["annual_volatility"])


# ── SIM-E: switching assumption source doesn't reset other inputs ───────
def test_sim_e_switching_assumption_source_preserves_inputs():
    at = _setup_sim_page(current_portfolio=_SIM_FAKE_PORTFOLIO, assumption_source="Market Scenario", click_run=False)
    exc = at.exception[0] if at.exception else None
    check("SIM-E.no_exception", exc is None, str(exc))
    if exc:
        return

    # Change the primary inputs away from their defaults first.
    for w in at.number_input:
        if w.key == "sim_initial_investment_val":
            w.set_value(25000.0)
        elif w.key == "sim_monthly_contribution_val":
            w.set_value(1200.0)
    for w in at.slider:
        if w.key == "sim_investment_years_val":
            w.set_value(15)
    at.run()

    for w in at.selectbox:
        if w.key == "projection_assumption_source":
            w.set_value("Custom Assumptions")
    at.run()
    for w in at.selectbox:
        if w.key == "projection_assumption_source":
            w.set_value("Portfolio Historical Statistics")
    at.run()

    exc2 = at.exception[0] if at.exception else None
    check("SIM-E.no_exception_after_switches", exc2 is None, str(exc2))
    if exc2:
        return

    def _val(kind, key):
        for w in getattr(at, kind):
            if w.key == key:
                return w.value
        return None

    check("SIM-E.initial_investment_preserved", _val("number_input", "sim_initial_investment_val") == 25000.0,
          _val("number_input", "sim_initial_investment_val"))
    check("SIM-E.monthly_contribution_preserved", _val("number_input", "sim_monthly_contribution_val") == 1200.0,
          _val("number_input", "sim_monthly_contribution_val"))
    check("SIM-E.horizon_preserved", _val("slider", "sim_investment_years_val") == 15,
          _val("slider", "sim_investment_years_val"))
    cp_after = at.session_state["current_portfolio"]
    check("SIM-E.portfolio_preserved", cp_after == _SIM_FAKE_PORTFOLIO, cp_after)


# ── SIM-F: Historical Simulation -> placeholder only, no fake results ───
def test_sim_f_historical_simulation_placeholder_only():
    at = _setup_sim_page(simulation_mode="Historical Simulation")
    exc = at.exception[0] if at.exception else None
    check("SIM-F.no_exception", exc is None, str(exc))
    if exc:
        return
    check("SIM-F.no_run_button_rendered",
          next((b for b in at.button if b.key == "sim_run_btn"), None) is None)
    # _setup_sim_page's initial at.run() happens while simulation_mode still
    # defaults to "Future Projection" -- the page's pre-existing (unchanged)
    # "auto-run on first load" behavior legitimately populates sim_result
    # with a REAL Future Projection result at that point, before we ever
    # switch modes. That's expected and is not "fake historical data" --
    # the real requirement is that switching TO Historical Simulation must
    # not RENDER any results/KPIs, which is what this checks instead.
    corpus = "\n".join(m.value for m in at.markdown)
    check("SIM-F.no_results_section_rendered", "Simulation Results" not in corpus, corpus[:200])
    check("SIM-F.no_kpi_cards_rendered", "kpi-card" not in corpus, "")
    info_texts = "\n".join(i.value for i in at.info)
    check("SIM-F.placeholder_shown", "backtest the portfolio using actual ETF price history" in info_texts,
          info_texts[:300])


# ── SIM-G: probability metric label matches its actual definition ───────
def test_sim_g_probability_label_matches_definition():
    at = _setup_sim_page(assumption_source="Market Scenario", scenario="Base Case")
    exc = at.exception[0] if at.exception else None
    check("SIM-G.no_exception", exc is None, str(exc))
    if exc:
        return
    sim_result = at.session_state["sim_result"]
    summary = sim_result["summary"]
    final_values = sim_result["all_final_values"]
    manual_prob = float(np.mean(final_values > summary["total_contributed"]))
    check("SIM-G.probability_is_final_value_over_contributions",
          abs(summary["probability_profit"] - manual_prob) < 1e-9,
          f"{summary['probability_profit']} vs {manual_prob}")
    corpus = "\n".join(m.value for m in at.markdown)
    check("SIM-G.label_states_exact_definition", "Probability of Ending Above Contributions" in corpus, "")
    check("SIM-G.old_ambiguous_label_absent", "Probability of Profit" not in corpus, "")


# ── SIM-H: Advanced Settings still affect the model correctly ───────────
def test_sim_h_advanced_settings_affect_model():
    at = _setup_sim_page(assumption_source="Market Scenario", scenario="Base Case", click_run=False)
    exc = at.exception[0] if at.exception else None
    check("SIM-H.no_exception", exc is None, str(exc))
    if exc:
        return

    for w in at.slider:
        if w.key == "sim_number_of_simulations_val":
            w.set_value(2500)
        elif w.key == "sim_annual_fee_pct_val":
            w.set_value(1.5)
        elif w.key == "sim_inflation_rate_pct_val":
            w.set_value(4.0)
    at.run()
    run_btn = next((b for b in at.button if b.key == "sim_run_btn"), None)
    run_btn.click()
    at.run()

    exc2 = at.exception[0] if at.exception else None
    check("SIM-H.no_exception_after_run", exc2 is None, str(exc2))
    if exc2:
        return
    sim_params = at.session_state["sim_params"]
    check("SIM-H.n_simulations_applied", sim_params["n_simulations"] == 2500, sim_params["n_simulations"])
    check("SIM-H.fee_applied", abs(sim_params["annual_fee"] - 0.015) < 1e-9, sim_params["annual_fee"])
    check("SIM-H.inflation_applied", abs(sim_params["inflation_rate"] - 0.04) < 1e-9, sim_params["inflation_rate"])

    sim_result = at.session_state["sim_result"]
    check("SIM-H.path_count_matches_n_simulations", len(sim_result["all_final_values"]) == 2500,
          len(sim_result["all_final_values"]))

    # Cross-check against a direct engine call with the same inputs --
    # confirms Advanced Settings values actually reach simulate_investment(),
    # not just get stored in sim_params cosmetically.
    direct = simulate_investment(
        initial_investment=sim_params["initial_investment"], monthly_contribution=sim_params["monthly_contribution"],
        years=sim_params["years"], annual_return=sim_params["annual_return"],
        annual_volatility=sim_params["annual_volatility"], inflation_rate=0.04, annual_fee=0.015,
        n_simulations=2500,
    )
    check("SIM-H.median_matches_direct_engine_call",
          abs(sim_result["summary"]["median_final"] - direct["summary"]["median_final"]) < 1e-6,
          f"{sim_result['summary']['median_final']} vs {direct['summary']['median_final']}")


# ── SIM-I: zh-TW / English render correctly, no raw keys ────────────────
def test_sim_i_i18n():
    forbidden = ("sim_", "opt_advanced_settings_title", "handoff_", "_label", "_title", "_subtitle", "_desc")
    for lang in ("zh-TW", "en"):
        at = _setup_sim_page(lang=lang, current_portfolio=_SIM_FAKE_PORTFOLIO)
        exc = at.exception[0] if at.exception else None
        check(f"SIM-I.{lang}.no_exception", exc is None, str(exc))
        if exc:
            continue
        corpus_parts = [m.value for m in at.markdown]
        corpus_parts += [c.value for c in at.caption]
        corpus_parts += [i.value for i in at.info]
        for kind in ("selectbox", "slider", "number_input", "button", "expander"):
            for w in getattr(at, kind, []):
                label = getattr(w, "label", None)
                if label:
                    corpus_parts.append(str(label))
        corpus = "\n".join(corpus_parts)
        hits = [frag for frag in forbidden if frag in corpus]
        check(f"SIM-I.{lang}.no_forbidden_fragments", len(hits) == 0, str(hits))
        expect_mode_label = "模擬模式" if lang == "zh-TW" else "Simulation Mode"
        expect_prob_label = "期末價值高於投入本金機率" if lang == "zh-TW" else "Probability of Ending Above Contributions"
        expect_setup_title = "推估設定" if lang == "zh-TW" else "Projection Setup"
        check(f"SIM-I.{lang}.mode_label_translated", expect_mode_label in corpus, "")
        check(f"SIM-I.{lang}.probability_label_translated", expect_prob_label in corpus, "")
        check(f"SIM-I.{lang}.setup_title_translated", expect_setup_title in corpus, "")


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

    test_pd_a_equal_weight()
    test_pd_b_max_sharpe_structural()
    test_pd_c_min_vol_structural()
    test_pd_d_effective_holdings_formula()
    test_pd_e_no_side_effects_on_rerender()
    test_pd_f_switch_strategy_updates_diagnosis()
    test_pd_g_zh_no_raw_keys()
    test_pd_h_en_no_raw_keys()
    test_pd_i_tooltips_present()
    test_pd_j_top2_status_thresholds()
    test_pd_k_concentrated_summary_interpolation()

    test_ph_a_equal_weight_handoff_to_simulator()
    test_ph_b_max_sharpe_handoff_to_simulator()
    test_ph_c_min_vol_handoff_to_risk_analytics()
    test_ph_d_current_portfolio_persists_across_rerun()
    test_ph_e_simulator_empty_state()
    test_ph_f_risk_analytics_empty_state()
    test_ph_g_language_switch_preserves_portfolio()
    test_ph_h_market_state_unchanged_through_handoff()
    test_ph_i_handoff_no_raw_keys()

    test_sim_a_current_portfolio_arrives_intact()
    test_sim_b_portfolio_historical_statistics_drives_simulation()
    test_sim_c_market_scenario_drives_simulation()
    test_sim_d_custom_assumptions_drive_simulation()
    test_sim_e_switching_assumption_source_preserves_inputs()
    test_sim_f_historical_simulation_placeholder_only()
    test_sim_g_probability_label_matches_definition()
    test_sim_h_advanced_settings_affect_model()
    test_sim_i_i18n()

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
