"""
Deterministic ETF identity colors across allocation charts (Issue #43 item
P): the same ticker must always render the same color regardless of
process, rerun, or input ordering -- never Python's randomized built-in
hash().
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.theme import color_for_ticker, colors_for_tickers, CHART_COLORS
from src.charts import allocation_donut_chart


def test_same_ticker_same_color_regardless_of_call_order():
    order_1 = colors_for_tickers(["QQQ", "BND", "VOO"])
    order_2 = colors_for_tickers(["VOO", "QQQ", "BND"])
    by_ticker_1 = dict(zip(["QQQ", "BND", "VOO"], order_1))
    by_ticker_2 = dict(zip(["VOO", "QQQ", "BND"], order_2))
    assert by_ticker_1["QQQ"] == by_ticker_2["QQQ"]
    assert by_ticker_1["BND"] == by_ticker_2["BND"]
    assert by_ticker_1["VOO"] == by_ticker_2["VOO"]


def test_color_for_ticker_is_stable_across_repeated_calls():
    assert color_for_ticker("QQQ") == color_for_ticker("QQQ")
    assert color_for_ticker("QQQ") == color_for_ticker("QQQ")


def test_color_for_ticker_is_one_of_the_chart_palette_colors():
    for ticker in ("QQQ", "BND", "VOO", "CUSTOMXYZ", "0050", "VUSA"):
        assert color_for_ticker(ticker) in CHART_COLORS


def test_custom_unlisted_ticker_gets_a_stable_color_too():
    assert color_for_ticker("MYWEIRDTICKER123") == color_for_ticker("MYWEIRDTICKER123")


def test_stable_across_different_pythonhashseed_processes():
    """Python's built-in hash() is randomized per-process (PYTHONHASHSEED)
    for strings -- using it would make colors change across reruns/
    processes. Prove color_for_ticker() does NOT depend on it by running it
    in two subprocesses with deliberately different PYTHONHASHSEED values
    and confirming identical output, rather than just inspecting source."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); "
        "from src.theme import color_for_ticker; "
        "print(color_for_ticker('QQQ'))"
    ) % REPO_ROOT

    def _run_with_seed(seed):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True, check=True)
        return result.stdout.strip()

    assert _run_with_seed("1") == _run_with_seed("42")


def test_two_portfolio_donuts_share_identical_color_for_same_ticker():
    """The actual regression this item fixes: allocation_donut_chart() used
    to assign colors by dict/list ORDER (CHART_COLORS[:len(labels)]), so
    the same ticker could get a different color in two side-by-side
    donuts. Build two portfolios where QQQ is in a different position in
    each and confirm QQQ's slice color matches in both figures."""
    portfolio_a = {"QQQ": 0.6, "BND": 0.4}
    portfolio_b = {"VOO": 0.5, "SCHD": 0.3, "QQQ": 0.2}

    fig_a = allocation_donut_chart(portfolio_a, "")
    fig_b = allocation_donut_chart(portfolio_b, "")

    labels_a = list(fig_a.data[0].labels)
    labels_b = list(fig_b.data[0].labels)
    colors_a = list(fig_a.data[0].marker.colors)
    colors_b = list(fig_b.data[0].marker.colors)

    qqq_color_a = colors_a[labels_a.index("QQQ")]
    qqq_color_b = colors_b[labels_b.index("QQQ")]
    assert qqq_color_a == qqq_color_b


def test_donut_reordered_inputs_keep_same_per_ticker_colors():
    weights_1 = {"QQQ": 0.5, "BND": 0.3, "VOO": 0.2}
    weights_2 = {"VOO": 0.2, "QQQ": 0.5, "BND": 0.3}  # same tickers, different order

    fig_1 = allocation_donut_chart(weights_1, "")
    fig_2 = allocation_donut_chart(weights_2, "")

    by_ticker_1 = dict(zip(fig_1.data[0].labels, fig_1.data[0].marker.colors))
    by_ticker_2 = dict(zip(fig_2.data[0].labels, fig_2.data[0].marker.colors))

    for ticker in weights_1:
        assert by_ticker_1[ticker] == by_ticker_2[ticker]
