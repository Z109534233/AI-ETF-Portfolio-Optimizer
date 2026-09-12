# AI ETF Portfolio Optimizer — Automated Review Rules

These rules govern the OpenAI-powered PR reviewer (`automation/reviewer.py`).
They exist to catch methodological and financial-correctness risks in an ETF
portfolio optimization tool, where a subtly wrong calculation or a silently
dropped asset can be worse than a visible crash.

## Review priorities

The reviewer must evaluate every pull request against the following
priorities, in roughly descending order of importance:

1. **Correctness of financial calculations** — returns, volatility, Sharpe
   ratio, drawdown, and any other quantitative metric must be computed
   correctly and consistently with standard portfolio theory.
2. **No hidden look-ahead bias** — no calculation may use future data
   (e.g. a later price, a later rebalance date, or a metric computed over
   the full sample) to make a decision at an earlier point in time.
3. **No fabricated market data** — prices, returns, or fundamentals must
   come from a real data source or an explicit, clearly labeled synthetic
   path (e.g. tests). Silent substitution of placeholder/mock data for
   real data in production code paths is not acceptable.
4. **Correct expected-return methodology** — the method used to estimate
   expected returns (historical mean, CAPM, shrinkage, etc.) must be
   applied correctly and must match what the UI/docs claim is being done.
5. **Correct covariance methodology** — the covariance/correlation matrix
   must be computed correctly (sample size, annualization, alignment of
   dates across assets) and must be positive semi-definite where the
   optimizer requires it.
6. **Optimization constraints actually applied** — constraints described
   to the user (weight bounds, sector caps, long-only, budget constraint,
   etc.) must actually be enforced by the solver, not merely documented
   or applied cosmetically after the fact.
7. **Backtest assumptions explicitly documented** — rebalance frequency,
   transaction costs (or the explicit absence of them), slippage,
   survivorship bias, and the historical window used must be stated in
   code comments, docstrings, or user-facing text — not left implicit.
8. **No silent ETF dropping** — if an ETF is excluded (insufficient
   history, missing data, failed fetch), this must be visible to the user
   (warning, log, or UI message), not silently removed from the universe.
9. **Correct handling of short historical data** — ETFs with limited
   price history must be handled explicitly (excluded with a clear
   reason, or handled with a documented minimum-history rule) rather than
   causing silent NaNs, crashes, or biased statistics.
10. **No hard-coded fake analytical results** — metrics, weights, or
    recommendations must never be hard-coded or stubbed as if they were
    computed, whether in application code or in tests that are supposed
    to validate real logic.
11. **Streamlit state preservation** — changes must not break
    `st.session_state` usage in a way that resets user input, loses
    selections, or causes unexpected reruns/state loss.
12. **zh-TW / English i18n parity** — any new or changed user-facing
    string must exist in both the zh-TW and English translation tables,
    with no missing keys in either language.
13. **No raw translation keys** — the UI must never render a raw i18n key
    (e.g. `portfolio.optimizer.title`) instead of the translated string.
14. **No unrelated feature changes** — a PR must not bundle changes
    unrelated to its stated task (scope creep, opportunistic refactors,
    unrelated dependency bumps).
15. **Tests must match the requested task** — new or modified tests must
    actually exercise the behavior the task describes, not be trivial,
    tautological, or disconnected from the change.
16. **Do not approve code only because it compiles** — passing syntax
    checks, type checks, or even a green test suite is not sufficient for
    a PASS verdict if the underlying logic is financially or
    methodologically wrong.

## Verdicts

The reviewer must return exactly one of the following verdicts:

- **PASS** — implementation satisfies the task and no material defect is
  found.
- **FIX** — implementation is generally recoverable but contains concrete
  issues Claude should fix.
- **STOP** — serious methodological risk, destructive changes, scope
  violation, security issue, or the reviewer cannot safely determine
  correctness.

When in doubt between PASS and FIX, prefer FIX. When in doubt between FIX
and STOP, prefer STOP — this tool makes real allocation decisions, and a
false PASS is more costly than a false alarm.
