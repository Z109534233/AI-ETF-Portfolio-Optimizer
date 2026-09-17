"""
Shared cross-asset demonstration portfolio (Issue #45 item 1).

Professor review finding: the previous demo/default ordering could surface
VOO/VTI/SPY/QQQ-style portfolios with extreme mutual overlap -- VOO and SPY
track the same index, VTI is highly correlated with both, so a
diversification ratio near 1 and a corner Maximum-Sharpe solution look like
the app doesn't understand the economics of diversification.

ONE explicit constant is used instead, for the Home demo and the Portfolio
Optimizer's initial US-market demo selection ONLY -- never forced onto
every page/user selection, which remains completely free to pick any ETF:

  VOO  -- US large-cap equities
  VXUS -- international (ex-US) equities
  BND  -- US aggregate/investment-grade bonds
  GLD  -- gold
  TLT  -- long-term US Treasuries

Five distinct asset classes are used intentionally to demonstrate
diversification mechanics across equities/bonds/commodities/duration. This
is a cross-asset DEMONSTRATION portfolio (跨資產示範組合), not a
recommendation -- every UI surface that uses it must label it as such.
"""

DIVERSIFIED_DEMO_TICKERS = ["VOO", "VXUS", "BND", "GLD", "TLT"]


def demo_portfolio_available(universe) -> bool:
    """True only when every ticker in DIVERSIFIED_DEMO_TICKERS is actually
    present in `universe` (e.g. a page's real available-options list)."""
    available = set(universe)
    return all(tk in available for tk in DIVERSIFIED_DEMO_TICKERS)


def resolve_demo_tickers(universe, fallback_n: int = 5) -> list:
    """The demo/default ticker selection for a given available `universe`:
    the shared cross-asset demo set when every one of its tickers is
    actually available in `universe`, else the first `fallback_n` tickers
    of `universe` (preserves prior behavior for a universe that genuinely
    doesn't carry all five, e.g. a non-US-only market filter). Never
    raises, and never returns a ticker that isn't actually in `universe`.
    """
    universe = list(universe)
    if demo_portfolio_available(universe):
        return list(DIVERSIFIED_DEMO_TICKERS)
    return universe[:fallback_n]
