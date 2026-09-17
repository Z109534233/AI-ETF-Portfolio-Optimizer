"""
Methodology & Disclosure Module (M1 -> M4)
Single source of truth for the platform's stated calculation methodology,
assumptions, and post-hoc validation helpers -- surfaced through compact
"Methodology & Assumptions" UI panels and exercised by deterministic tests.

Every entry here documents ONLY what the corresponding src/ module actually
does. If the implementation changes, this module must be updated in the
same change -- it must never describe a method the app doesn't implement,
and it must never silently fall out of sync with the real calculation.

No Streamlit import here on purpose: this module is pure Python/NumPy so it
can be imported and unit-tested without a Streamlit runtime, by
src/portfolio_optimizer.py (a calculation module) and by every
pages/*.py file that renders a methodology panel.
"""

from datetime import date

import numpy as np

# ============================================================================
# M1 -- Portfolio Optimization Methodology
# ============================================================================
# Describes src/portfolio_optimizer.py + src/financial_metrics.py's
# covariance_matrix() as actually implemented -- see run_optimization(),
# optimize_max_sharpe(), and covariance_matrix() for the code this
# metadata must always match.

PORTFOLIO_OPTIMIZATION_METHODOLOGY = {
    "expected_return": {
        "price_field": "Close (split/dividend-adjusted via yfinance auto_adjust=True)",
        "return_type": "simple daily percentage returns (DataFrame.pct_change)",
        "sampling_frequency": "daily",
        "estimator": "arithmetic mean of daily returns",
        "annualization": "mean daily return * 252 trading days",
        "missing_data_handling": (
            "dates missing for every ticker are dropped; dates missing for "
            "only some tickers are kept (each pair's statistics are computed "
            "from their own overlapping dates). No ETF's price is ever "
            "forward- or back-filled before its own first valid trading date."
        ),
        "common_history_window": (
            "every selected ETF is sliced to the common valid-data window "
            "(the latest first-valid-date to the earliest last-valid-date "
            "across the selection) before optimization, so a later-inception "
            "ETF shortens the analysis window instead of being backfilled."
        ),
    },
    "covariance": {
        "estimator": "sample covariance (pandas DataFrame.cov())",
        "sampling_frequency": "daily",
        "annualization": "daily covariance matrix * 252 trading days",
        "alignment": (
            "pandas .cov() uses pairwise-complete observations per ticker "
            "pair; rows where every ticker is missing are dropped first, but "
            "no ticker is ever excluded from the matrix outright."
        ),
        "nan_handling": (
            "a ticker with zero valid overlapping data produces NaN on its "
            "own diagonal; run_optimization() detects this explicitly and "
            "reports it as an error rather than treating it as zero risk."
        ),
        "psd_stabilization": (
            "a small ridge term (identity * 1e-8) is added to the diagonal "
            "only to keep the matrix numerically solvable during "
            "optimization; this is not shrinkage of the estimator itself "
            "and does not materially change reported risk."
        ),
        "shrinkage": "none -- no Ledoit-Wolf or other shrinkage estimator is applied",
    },
    "max_sharpe": {
        "objective": (
            "maximize (portfolio_return - risk_free_rate) / portfolio_volatility, "
            "implemented as minimizing the negative Sharpe ratio"
        ),
        "solver": "scipy.optimize.minimize, method='SLSQP'",
        "risk_free_rate": (
            "user-configurable annual rate, used directly as a flat annual "
            "rate in the Sharpe ratio numerator -- not compounded or "
            "converted to a daily rate. The slider's default value is the "
            "latest live FRED DGS3MO (3-Month Treasury) observation when "
            "available (src.risk_free_rate); if a live observation cannot "
            "be fetched, it falls back to a fixed 5% default with that "
            "status disclosed, never presented as if it were current"
        ),
        "constraints": (
            "sum(weights) == 1 (fully invested, no cash); per-asset weight "
            "bounded to [min_weight, max_weight], or [-max_weight, max_weight] "
            "if short selling is allowed"
        ),
    },
    "insufficient_history": {
        "policy": (
            "no ETF is ever silently dropped for insufficient history. If "
            "fewer than 2 ETFs have overlapping valid price data, or fewer "
            "than 20 overlapping trading days / 10 overlapping daily returns "
            "exist, the page stops with an explicit error instead of running "
            "optimization on a smaller or fabricated dataset."
        ),
    },
    "backtest_label": {
        "type": "Fixed-Allocation Historical Backtest",
        "explanation": (
            "the single set of weights produced by the CURRENT optimization "
            "run is applied unchanged across the entire historical price "
            "window to compute a hypothetical value path. It is NOT a "
            "walk-forward backtest: weights are never re-optimized at any "
            "point using only data available as of that date, so it does not "
            "reflect how the strategy would actually have been selected and "
            "rebalanced in real time."
        ),
    },
}


# ============================================================================
# M2 -- Simulation & Backtest Methodology
# ============================================================================
# Describes src/simulator.py as actually implemented -- see
# simulate_investment(), historical_backtest(), prepare_historical_prices(),
# and xirr() for the code this metadata must always match.

SIMULATION_METHODOLOGY = {
    "monte_carlo": {
        "purpose": (
            "a projection of a RANGE of plausible future outcomes under "
            "stated assumptions -- never a prediction or guarantee of what "
            "markets will actually do."
        ),
        "distribution": "independent monthly returns drawn from a Normal distribution",
        "timestep": "monthly",
        "n_simulations": "user-configurable (default 1,000; 200-5,000)",
        "contribution_timing": (
            "the monthly contribution is added to the portfolio AFTER that "
            "month's simulated return is applied, at every month of the horizon"
        ),
        "fees": (
            "an annual fee is converted to an equivalent monthly rate and "
            "subtracted directly from the mean monthly return before each "
            "random draw"
        ),
        "inflation_treatment": (
            "nominal paths are simulated first; a separate 'real' "
            "(inflation-adjusted) path is derived by dividing by a "
            "compounded monthly inflation factor -- inflation never alters "
            "the nominal simulation itself"
        ),
        "reported_percentiles": (
            "10th (pessimistic) / 50th (median) / 90th (optimistic) as "
            "headline KPIs, plus a 5/25/50/75/95 percentile table"
        ),
        "assumption_modes": {
            "Portfolio Historical Statistics": (
                "the CURRENT portfolio's own historical expected return/ "
                "volatility from Portfolio Optimizer, used as a forward "
                "assumption -- labeled as a historical estimate, never a "
                "prediction"
            ),
            "Market Scenario": (
                "one of 4 hard-coded hypothetical return/volatility pairs "
                "(Bull/Base/Bear/Sideways) -- labeled as hypothetical, not "
                "a market forecast"
            ),
            "Custom Assumptions": "user-entered return/volatility, used as-is",
        },
    },
    "historical_simulation": {
        "data": (
            "real downloaded ETF prices only -- no synthetic or fabricated "
            "returns are ever used in Historical Simulation mode"
        ),
        "start_date": (
            "constrained to the common valid-data window across every "
            "active ticker (the same get_common_date_range() logic as "
            "Portfolio Optimizer); a user-chosen start earlier than that "
            "window is clamped forward to the common start, never backfilled"
        ),
        "pre_inception_handling": (
            "prepare_historical_prices() only forward-fills genuine "
            "mid-series gaps within the requested window; any date before a "
            "ticker's own first valid price is never included, let alone filled"
        ),
    },
    "rebalancing": {
        "convention": "monthly",
        "mechanics": (
            "shares are held constant (pure mark-to-market) between "
            "rebalance points -- no daily rebalancing. On the first trading "
            "day of every calendar month after the starting month, the "
            "position is marked to market, the monthly contribution (which "
            "may be $0) is added, and the total is reallocated back to the "
            "target weights in the same step."
        ),
    },
    "contributions_and_returns": {
        "total_invested": (
            "initial_investment + sum of every monthly contribution actually "
            "made over the simulated/backtested period"
        ),
        "gain": "final portfolio value - total_invested",
        "fractional_shares": "assumed (a simplification -- not a real brokerage constraint)",
        "xirr": (
            "money-weighted annualized return solved from the ACTUAL dated "
            "cash flows (initial investment + every monthly contribution, "
            "each on its real date, plus the final value as a single "
            "liquidating inflow) via Brent's method; returns None (shown as "
            "unavailable, never a fabricated number) if it cannot be solved "
            "robustly"
        ),
    },
    "backtest_label": {
        "type": "Fixed-Allocation Historical Backtest",
        "look_ahead_bias": (
            "the weights applied throughout the ENTIRE historical window "
            "are the CURRENT Portfolio Optimizer result -- i.e. weights "
            "chosen using information that includes data from later in (or "
            "after) the very window being tested. This is look-ahead bias "
            "by construction, and is never described as walk-forward or as "
            "evidence the strategy would have been selected this way in "
            "real time."
        ),
    },
}


# ============================================================================
# M3 -- Risk Methodology
# ============================================================================
# Describes src/risk_analytics.py + the pre-existing src/financial_metrics.py
# risk functions it wraps (value_at_risk, conditional_var, maximum_drawdown,
# portfolio_diagnosis, correlation_matrix) as actually implemented.

RISK_METHODOLOGY = {
    "var_cvar": {
        "method": "historical -- empirical percentile of realized daily returns (no parametric/Monte Carlo VaR)",
        "confidence": "user-configurable via the confidence level passed to historical_var_cvar() (95% default)",
        "holding_period": "1 trading day (VaR/CVaR are computed on daily returns, not scaled to any longer horizon)",
        "window": "the full selected date range's overlapping daily returns for the current price series",
        "insufficient_data": (
            "if fewer than MIN_VAR_OBSERVATIONS (30) daily returns are "
            "available, the page shows an explicit 'unavailable' state with "
            "the observation count and requirement, never a number computed "
            "from too little data"
        ),
    },
    "max_drawdown": {
        "formula": "min over time of (price - rolling_max(price)) / rolling_max(price)",
        "verification": "computed directly from the SAME price series shown in the growth/backtest chart -- never a separately-sourced or hard-coded figure",
    },
    "concentration": {
        "source": (
            "largest-position, top-N concentration, effective-holdings (1/"
            "Herfindahl), and active-position-count are ALWAYS computed from "
            "the canonical current_portfolio weights dict (portfolio_diagnosis()) "
            "-- the same function and the same weights Portfolio Optimizer uses "
            "-- never from a separately-entered or independently-normalized "
            "weight source"
        ),
    },
    "correlation_vs_overlap": {
        "return_correlation": "Pearson correlation of each ETF pair's daily returns (financial_metrics.correlation_matrix) -- a statement about how the ETFs' PRICES have moved together",
        "holdings_overlap": (
            "sum over shared itemized holding tickers of min(weight_in_A, "
            "weight_in_B) -- a statement about how much of the ETFs' "
            "UNDERLYING PORTFOLIOS are the same securities. High return "
            "correlation does not imply holdings overlap (different segments "
            "can move together macro-economically) and low return "
            "correlation does not imply low overlap (a few differing "
            "positions can dominate short-term volatility) -- the two are "
            "never conflated or substituted for one another."
        ),
        "unavailable_handling": "a pair is marked unavailable (with a stated reason) rather than scored, whenever either ETF's underlying holdings cannot be retrieved",
    },
    "alpha_beta": {
        "beta_formula": (
            "beta = Cov(portfolio daily returns, benchmark daily returns) / "
            "Var(benchmark daily returns), estimated over the full selected "
            "date range's overlapping daily returns"
        ),
        "alpha_formula": (
            "alpha = portfolio_annualized_return - (risk_free_rate + beta * "
            "(benchmark_annualized_return - risk_free_rate)) -- a CAPM-style "
            "excess-return residual computed directly from each side's own "
            "annualized return, NOT an intercept estimated from an actual "
            "OLS regression of portfolio returns on benchmark returns"
        ),
        "annualization": (
            "both the portfolio and benchmark returns feeding this formula "
            "are annualized, and risk_free_rate is the page's own annual "
            "rate assumption -- the resulting alpha is therefore always an "
            "ANNUALIZED figure, and the UI labels it as such"
        ),
        "benchmark_self_inclusion": (
            "the benchmark selector excludes the page's own currently-"
            "selected ETFs wherever at least one external option remains, "
            "so the comparison is not self-referential; if every available "
            "benchmark option in the current market is itself a selected "
            "ETF, alpha/beta are shown as unavailable rather than computed "
            "against a benchmark that is part of the portfolio being measured"
        ),
    },
    "stress_scenarios": {
        "calculation": "portfolio_impact = market_shock * portfolio_beta -- a simple linear approximation, not a full historical portfolio reconstruction or factor-model replay",
        "provenance_labeling": (
            "each scenario is explicitly labeled 'hypothetical' (a stated, "
            "illustrative assumption) or 'historical' (a commonly-cited, "
            "approximate broad-market peak-to-trough decline for a named "
            "past event) -- see src/risk_analytics.py's STRESS_SCENARIOS; "
            "shock magnitudes are pre-existing values, unchanged by this task"
        ),
        "limitation": (
            "even 'historical' scenarios use the SAME linear beta-scaling as "
            "hypothetical ones -- this is never a claim that the current "
            "portfolio would have actually behaved this way during that "
            "event, only an illustrative, beta-scaled approximation"
        ),
    },
}


# ============================================================================
# M4 -- Market Intelligence & Sentiment Validation
# ============================================================================
# Describes src/market_intelligence.py + src/news.py as actually implemented.

MARKET_INTELLIGENCE_METHODOLOGY = {
    "pipeline": {
        "stages": [
            "source (Yahoo Finance news feed, via a small set of US-listed "
            "broad-market ticker proxies)",
            "headline sentiment (keyword-based Positive/Negative/Neutral "
            "classifier, src/news.py's classify_headline_sentiment())",
            "event classification (rule-based keyword matching into a fixed "
            "set of event categories, classify_event()/classify_event_types())",
            "relevance gating (does this headline's event type apply to a "
            "given market/ETF at all, via EVENT_MARKET_IMPACT / "
            "EVENT_TYPE_TO_ETF_TYPE_IMPACT -- headlines that don't gate in "
            "never receive an impact or confidence score for that market/ETF)",
            "deterministic impact/sentiment/confidence scoring (weighted "
            "formulas over the gated headlines)",
            "summary text (template-generated by default; OpenAI-generated "
            "only when an API key is configured AND the call succeeds)",
        ],
        "deterministic_vs_ai": (
            "every stage up to and including impact/sentiment/confidence "
            "scoring is deterministic rule-based code -- given the same "
            "headlines, it always produces the same numbers. Only the "
            "summary text at the very end can be AI-generated, and only "
            "when an OpenAI key is configured and the call actually "
            "succeeds; every section is tagged 'Rule-Based' or "
            "'AI-Generated' accordingly, and the 'AI-Generated' tag is "
            "never shown for a section that fell back to its rule-based "
            "template."
        ),
    },
    "bullish_bearish_neutral": {
        "policy": (
            "mood is only ever Bullish or Bearish when the weighted "
            "directional signal exceeds a stated threshold (avg > 0.15 or "
            "< -0.15); anything inside that band -- including a genuine "
            "lack of relevant headlines -- is Neutral. A direction is never "
            "forced when the signal doesn't clearly warrant one."
        ),
    },
    "measures": {
        "impact": "how significant an event/headline is estimated to be for a market or ETF (0-100 score / 1-5 stars) -- NOT a confidence level and NOT a relevance flag",
        "confidence": "how one-sided the underlying weighted sentiment signal is (30-95%, never a fabricated 0% or 100%) -- NOT how important the event is",
        "relevance": (
            "a binary gating step applied BEFORE impact/confidence are "
            "computed: does this headline's classified event type apply to "
            "this market/ETF at all. A headline that doesn't gate in for a "
            "given market/ETF never produces an impact or confidence score "
            "for it in the first place -- relevance is never blended "
            "numerically with impact or confidence."
        ),
    },
    "affected_markets": {
        "basis": (
            "evidence-based event-type -> market/ETF-type weight tables "
            "(EVENT_MARKET_IMPACT, EVENT_AFFECTED_MARKETS, "
            "EVENT_TYPE_TO_ETF_TYPE_IMPACT) reflecting known, general "
            "economic relationships (e.g. a tariff/semiconductor headline "
            "is mapped to Taiwan for its chip-export exposure even when "
            "the headline never says 'Taiwan') -- NOT keyword name-matching "
            "against a market's name, and not a statistically-derived model."
        ),
        "limitation": (
            "headlines are sourced from a small set of US-listed tickers' "
            "news feeds (S&P 500/Nasdaq/Dow proxies -- see "
            "news.NEWS_SOURCE_TICKERS), so Taiwan- and UK-specific stories "
            "that never surface in US financial headlines can be "
            "under-represented or missed entirely. Taiwan and UK coverage "
            "is structurally thinner than US coverage as a direct result."
        ),
    },
    "validation": {
        "status": "NOT externally validated against any independently labeled ground-truth dataset",
        "no_fabricated_metrics": (
            "no precision, recall, F1, or accuracy figure is computed or "
            "displayed anywhere in this pipeline, because none has been "
            "measured -- every score is a transparent, auditable rule/"
            "formula applied to today's headlines, not a statistically "
            "validated prediction"
        ),
    },
    "source_and_as_of": {
        "policy": (
            "a headline's publish time is shown only when the source "
            "actually provided one (src/news.py's fetch_market_news()); "
            "otherwise the page shows an explicit placeholder, never a "
            "fabricated time. The publisher shown is the feed's own "
            "attribution, defaulting to the aggregator name ('Yahoo "
            "Finance') only when the source didn't disclose a specific "
            "publisher -- itself a true statement of where the headline "
            "came from, not an invented outlet name."
        ),
    },
}


# ============================================================================
# M5 -- ETF Analysis Methodology
# ============================================================================
# Describes pages/1_ETF_Analysis.py + src/etf_signals.py's
# compute_quant_signals() + src/financial_metrics.py as actually implemented
# for the single-ticker analytics on that page (Sharpe/Sortino, price
# source/adjustment, and the risk-free-rate scope) -- see
# compute_quant_signals(), compute_all_metrics(), and src/data_loader.py's
# download_etf_data() for the code this metadata must always match.

ETF_ANALYSIS_METHODOLOGY = {
    "expected_return_and_volatility": {
        "return_type": "simple daily percentage returns (DataFrame.pct_change)",
        "annualization": (
            "annualized return uses CAGR: (last usable price / first usable "
            "price) ** (252 / number of usable trading days) - 1 -- NOT a "
            "mean-daily-return x 252 approximation; annualized volatility is "
            "the daily-return standard deviation x sqrt(252)"
        ),
    },
    "risk_free_rate": {
        "nature": (
            "a user-adjustable assumption from this page's own sidebar "
            "slider, defaulted to the latest live FRED DGS3MO (3-Month "
            "Treasury) observation when available (src.risk_free_rate); "
            "if a live observation cannot be fetched, the slider falls "
            "back to a fixed 5% default with that fallback status "
            "disclosed in a caption, never presented as if it were a "
            "current live rate. The user can always override the value manually."
        ),
        "usage": (
            "used directly in this page's Sharpe Ratio (the KPI row and "
            "compute_quant_signals()'s Quant Score inputs) and in the "
            "Sortino Ratio shown in the Compare and Deep Analysis metrics "
            "tables (compute_all_metrics() passes the same risk_free_rate "
            "to both sharpe_ratio() and sortino_ratio())"
        ),
    },
    "price_source": {
        "provider": "Yahoo Finance via the yfinance library (src/data_loader.py)",
        "adjustment": (
            "auto_adjust=True -- the downloaded Close price is already "
            "adjusted for dividends and stock splits by yfinance itself, "
            "this app does not re-derive its own adjustment"
        ),
        "gap_handling": (
            "missing trading days are forward-filled then back-filled "
            "(src/data_cleaner.py's clean_price_data()) after all requested "
            "tickers are downloaded together; a ticker with zero valid data "
            "anywhere is dropped entirely rather than filled"
        ),
    },
    "limitation": (
        "every metric on this page describes OBSERVED HISTORY over the "
        "selected date range only -- it is not a forecast of future return, "
        "volatility, or risk-adjusted performance"
    ),
}


# ============================================================================
# M6 -- Machine Learning Methodology
# ============================================================================
# Describes pages/5_Machine_Learning.py + src/machine_learning.py as
# actually implemented -- see run_ml_pipeline(), prepare_ml_dataset(),
# time_series_split(), and _baseline_majority_class_accuracy() for the code
# this metadata must always match.

MACHINE_LEARNING_METHODOLOGY = {
    "models": {
        "options": ["Logistic Regression", "Random Forest"],
        "logistic_regression": "scikit-learn LogisticRegression on standardized features (StandardScaler fit on the training split only)",
        "random_forest": "scikit-learn RandomForestClassifier (100 trees, max_depth=5, min_samples_leaf=10)",
    },
    "target": {
        "definition": "binary: 1 if the simple return over the next `lookahead` trading day(s) is positive, else 0",
        "lookahead": "user/page-configured number of trading days (this page fixes it at 1 -- next-day direction)",
    },
    "features": {
        "source": "technical indicators derived only from this ticker's own price (and, if given, volume) history -- src/technical_indicators.py's create_ml_features()",
        "types": "lagged returns, SMA/EMA ratios, RSI, MACD, momentum, rolling volatility, Bollinger Band position",
        "absent_inputs": (
            "no fundamental data (earnings, valuation ratios), no "
            "macroeconomic series, and no news/sentiment data are used as "
            "features -- price/volume-derived technical indicators only"
        ),
    },
    "split": {
        "method": "chronological time-series split (time_series_split()) -- the first (1 - test_size) fraction of rows trains the model, the remaining rows are held out as the test set",
        "shuffling": "none -- rows are never shuffled or randomly assigned across train/test, which would leak future information into training",
    },
    "baseline": {
        "definition": "always predict whichever class (up or down) was more frequent in the TRAINING set only, then score that constant prediction on the same held-out test set (_baseline_majority_class_accuracy())",
        "purpose": "the minimum bar a model must clear to be providing any real directional information beyond the test period's own class balance",
    },
    "metrics": {
        "reported": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC (when both classes are present in the test set)", "Confusion Matrix"],
        "computed_on": "the held-out test set only -- the model never sees this data during training",
    },
    "walk_forward_cv": {
        "method": (
            "expanding-window scikit-learn TimeSeriesSplit, run ONLY on the "
            "PRE-HOLDOUT training region (the same X_train/y_train the final "
            "model is fit on) -- the final chronological holdout above is "
            "never used for CV and is never seen by any fold"
        ),
        "folds": "target 5 folds, reduced automatically when there is not enough pre-holdout data for that many chronological folds; if fewer than 2 valid folds are possible, the page shows an explicit unavailable state instead of fabricated statistics",
        "preprocessing": "each fold's LogisticRegression StandardScaler is re-fit on that fold's own training rows only; RandomForest has no scaler to fit",
        "shuffling": "none -- folds are chronological and expanding, never shuffled",
        "purpose": "fold-to-fold dispersion in Accuracy/ROC AUC measures TEMPORAL INSTABILITY of the model across different historical windows -- it does not prove, and is never described as proving, future predictability",
        "not_used_for": "hyperparameter tuning -- fixed hyperparameters are still used for both model types; this CV is diagnostic-only",
    },
    "absent_from_pipeline": {
        "hyperparameter_tuning": "none -- fixed hyperparameters are used for both model types, no grid/random search is performed",
        "transaction_costs": "not modeled -- no bid-ask spread, commission, or market-impact cost is subtracted from any reported metric",
    },
    "live_trading_validity": (
        "this pipeline evaluates one model on one historical held-out "
        "window for one ticker -- it is not evidence of live-trading "
        "validity, and results on a different ticker, date range, or "
        "future period may differ substantially"
    ),
}


def validate_ml_split(train_start: str, train_end: str, test_start: str, test_end: str) -> dict:
    """Factual check that the train/test windows are chronological and
    non-overlapping (M6 acceptance criteria): train_start <= train_end,
    test_start <= test_end, and train_end < test_start. This validates the
    EVALUATION SETUP only -- it says nothing about whether the model itself
    performs well, and must never be rendered in a way that implies that.

    Returns {"is_valid": bool, "issues": [str, ...]} -- never raises; an
    unparseable date string simply fails the check with an explicit issue.
    """
    issues = []
    try:
        t_start = date.fromisoformat(str(train_start))
        t_end = date.fromisoformat(str(train_end))
        te_start = date.fromisoformat(str(test_start))
        te_end = date.fromisoformat(str(test_end))
    except (TypeError, ValueError):
        return {"is_valid": False, "issues": ["train/test dates could not be parsed"]}

    if t_start > t_end:
        issues.append("train_start is after train_end")
    if te_start > te_end:
        issues.append("test_start is after test_end")
    if t_end >= te_start:
        issues.append("train and test periods overlap or are not chronological")

    return {"is_valid": len(issues) == 0, "issues": issues}


def validate_etf_analysis_window(observations: int, min_observations: int = 20) -> dict:
    """Factual check that the focus ticker's usable historical window (M5)
    meets this page's minimum observation-count threshold before showing
    annualized metrics. This is a MINIMUM USABILITY/SUFFICIENCY check only --
    it does NOT re-derive or reconcile the displayed return/volatility/ratio
    figures against the price data, and it is not a claim of statistical
    stability. Returns {"is_valid": bool, "issues": [str, ...]}."""
    issues = []
    if observations < min_observations:
        issues.append(
            f"only {observations} usable observation(s) are available; at "
            f"least {min_observations} are recommended as a minimum "
            f"usability threshold before this page's annualized metrics are shown"
        )
    return {"is_valid": len(issues) == 0, "issues": issues}


def validate_optimization_result(weights: dict, mean_returns, cov_matrix,
                                  reported_return: float, reported_volatility: float,
                                  reported_sharpe: float, risk_free_rate: float,
                                  min_weight: float = 0.0, max_weight: float = 1.0,
                                  allow_short: bool = False, tol: float = 1e-4) -> dict:
    """Post-optimization validation (M1 acceptance criteria #4).

    Independently re-derives return/volatility/Sharpe from `weights` +
    `mean_returns` + `cov_matrix` and checks that: (1) weights sum to 1,
    (2) every weight respects the configured bounds, and (3) the reported
    metrics are numerically consistent with the weights actually returned.

    Returns {"is_valid": bool, "issues": [str, ...]} -- an empty `issues`
    list means every check passed. Never raises; a malformed/empty
    `weights` dict simply fails the "sum to 1" check rather than crashing,
    so callers can always safely inspect the result.
    """
    issues = []
    w = np.array(list(weights.values()), dtype=float) if weights else np.array([])

    weight_sum = float(w.sum()) if w.size else 0.0
    if abs(weight_sum - 1.0) > 1e-4:
        issues.append(f"weights sum to {weight_sum:.6f}, expected 1.0")

    lower_bound = -max_weight if allow_short else min_weight
    upper_bound = max_weight
    if w.size and (np.min(w) < lower_bound - 1e-6 or np.max(w) > upper_bound + 1e-6):
        issues.append(
            f"weight(s) fall outside the configured bounds "
            f"[{lower_bound}, {upper_bound}]: min={np.min(w):.6f}, max={np.max(w):.6f}"
        )

    mean_returns_arr = np.asarray(mean_returns, dtype=float)
    cov_arr = np.asarray(cov_matrix, dtype=float)
    if w.size and w.size == mean_returns_arr.size and cov_arr.shape == (w.size, w.size):
        recomputed_return = float(np.dot(w, mean_returns_arr) * 252)
        recomputed_volatility = float(np.sqrt(max(float(w @ cov_arr @ w), 0.0)))
        recomputed_sharpe = (
            (recomputed_return - risk_free_rate) / recomputed_volatility
            if recomputed_volatility > 0 else 0.0
        )

        if abs(recomputed_return - reported_return) > tol:
            issues.append(
                f"reported expected_return {reported_return:.6f} is inconsistent "
                f"with the weights (recomputed {recomputed_return:.6f})"
            )
        if abs(recomputed_volatility - reported_volatility) > tol:
            issues.append(
                f"reported expected_volatility {reported_volatility:.6f} is "
                f"inconsistent with the weights (recomputed {recomputed_volatility:.6f})"
            )
        if abs(recomputed_sharpe - reported_sharpe) > max(tol, 1e-3):
            issues.append(
                f"reported sharpe_ratio {reported_sharpe:.6f} is inconsistent "
                f"with the weights (recomputed {recomputed_sharpe:.6f})"
            )
    else:
        issues.append("weights/mean_returns/cov_matrix shapes do not align; cannot verify metrics")

    return {"is_valid": len(issues) == 0, "issues": issues}
