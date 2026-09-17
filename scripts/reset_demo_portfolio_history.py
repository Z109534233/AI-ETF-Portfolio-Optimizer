"""
Demo Portfolio History Reset (Issue #20 section 9A -- demo DB hygiene)
========================================================================
The committed database/portfolio.db previously accumulated ~140 rows from
manual testing during development (many exact duplicates -- the same
strategy saved repeatedly within seconds of each other). That is not a
credible demo state for a portfolio review: it looks like debug noise, not
curated examples.

This is a CONTROLLED, MANUALLY-RUN developer script -- it is never
imported or executed by the Streamlit app itself, and it does NOT touch a
user's real local database at runtime. Re-run it to rebuild the committed
demo database from scratch with a small, curated set of example
portfolios using real, currently-supported tickers:

    python scripts/reset_demo_portfolio_history.py

This DELETES and recreates database/portfolio.db. Never run this against
a database containing real user data you want to keep.

Every entry's expected_return/expected_volatility/sharpe_ratio below is a
hand-authored illustrative figure -- NOT computed by running the app's real
optimizer against downloaded historical prices (there is no live
market-data fetch in this script). Each entry's metadata therefore carries
"synthetic_demo": True, which pages/7_Portfolio_History.py reads to render
an explicit "Curated Demo Example" badge wherever these rows are displayed,
so they can never be mistaken for a live optimization result (Issue #20
release-gate review, automated PR reviewer finding).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import src.database as dbmod
from src.etf_database import get_etf

CURATED_PORTFOLIOS = [
    {
        "name": "US_Balanced",
        "weights": {"VOO": 0.40, "VXUS": 0.20, "BND": 0.30, "GLD": 0.10},
        "investment_amount": 10000.0,
        # NOT "Equal Weight" -- these weights are a deliberately unequal
        # 40/20/30/10 tilt, not a 1/N split. Labeling non-equal weights
        # "Equal Weight" is a false optimization-method claim (Issue #20
        # release-gate review); "Custom Allocation" is an honest label the
        # app's t_opt_method() already falls back to displaying verbatim
        # for any strategy string outside its five real optimizer methods.
        "optimization_method": "Custom Allocation",
        "expected_return": 0.075, "expected_volatility": 0.11, "sharpe_ratio": 0.50,
        "notes": "Curated demo example -- diversified US-listed balanced allocation across equities, ex-US equities, bonds and gold.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": "United States",
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.03, "min_weight": 0.0, "max_weight": 1.0, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Custom Allocation", "asset_universe": ["VOO", "VXUS", "BND", "GLD"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
    {
        "name": "US_Maximum_Sharpe_Ratio",
        "weights": {"QQQ": 0.55, "SCHD": 0.25, "BND": 0.20},
        "investment_amount": 10000.0,
        "optimization_method": "Maximum Sharpe Ratio",
        "expected_return": 0.135, "expected_volatility": 0.19, "sharpe_ratio": 0.66,
        "notes": "Curated demo example -- growth-tilted, optimized for risk-adjusted return.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": "United States",
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.03, "min_weight": 0.0, "max_weight": 0.7, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Maximum Sharpe Ratio", "asset_universe": ["QQQ", "SCHD", "BND"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
    {
        "name": "US_Minimum_Volatility",
        "weights": {"BND": 0.50, "SCHD": 0.30, "VOO": 0.20},
        "investment_amount": 10000.0,
        "optimization_method": "Minimum Volatility",
        "expected_return": 0.058, "expected_volatility": 0.065, "sharpe_ratio": 0.43,
        "notes": "Curated demo example -- optimized to minimize portfolio volatility.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": "United States",
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.03, "min_weight": 0.0, "max_weight": 0.7, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Minimum Volatility", "asset_universe": ["BND", "SCHD", "VOO"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
    {
        "name": "Taiwan_Growth",
        "weights": {"0050": 0.60, "006208": 0.40},
        "investment_amount": 300000.0,
        "optimization_method": "Maximum Sharpe Ratio",
        "expected_return": 0.095, "expected_volatility": 0.17, "sharpe_ratio": 0.51,
        "notes": "Curated demo example -- Taiwan-listed broad-market growth allocation.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": "Taiwan",
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.015, "min_weight": 0.0, "max_weight": 1.0, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Maximum Sharpe Ratio", "asset_universe": ["0050", "006208"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
    {
        "name": "UK_UCITS_Diversified",
        "weights": {"VWRL": 0.50, "VGOV": 0.30, "VUKE": 0.20},
        "investment_amount": 10000.0,
        # See US_Balanced above -- 50/30/20 is not an equal (1/3 each) split.
        "optimization_method": "Custom Allocation",
        "expected_return": 0.068, "expected_volatility": 0.10, "sharpe_ratio": 0.48,
        "notes": "Curated demo example -- UK-listed UCITS ETFs spanning global equities, gilts and UK large-cap.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": "United Kingdom",
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.04, "min_weight": 0.0, "max_weight": 1.0, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Custom Allocation", "asset_universe": ["VWRL", "VGOV", "VUKE"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
    {
        "name": "Global_Diversified",
        "weights": {"VOO": 0.40, "VWRL": 0.30, "0050": 0.30},
        "investment_amount": 10000.0,
        "optimization_method": "Risk Parity",
        "expected_return": 0.082, "expected_volatility": 0.13, "sharpe_ratio": 0.49,
        "notes": "Curated demo example -- cross-market allocation spanning US, UK-listed global, and Taiwan-listed ETFs.",
        "metadata": {
            "schema_version": 1, "synthetic_demo": True, "market": None,
            "historical_start_date": "2019-01-01", "historical_end_date": "2024-01-01",
            "risk_free_rate": 0.03, "min_weight": 0.0, "max_weight": 1.0, "allow_short": False,
            "expected_return_estimator": "Arithmetic Mean Daily Return (annualized x252)",
            "covariance_estimator": "Sample covariance (historical, annualized)",
            "strategy": "Risk Parity", "asset_universe": ["VOO", "VWRL", "0050"],
            "data_as_of": "2024-01-01", "app_version": dbmod.APP_VERSION,
        },
    },
]


def _validate_tickers() -> None:
    """Fail loudly rather than seed a demo database with a ticker the
    app's own ETF universe no longer recognizes."""
    for entry in CURATED_PORTFOLIOS:
        for ticker in entry["weights"]:
            if get_etf(ticker) is None:
                raise SystemExit(
                    f"Refusing to seed: {ticker!r} (from {entry['name']!r}) is not in the "
                    "current ETF universe. Update CURATED_PORTFOLIOS or refresh the universe first."
                )


def main() -> None:
    _validate_tickers()

    if os.path.exists(dbmod.DB_PATH):
        os.remove(dbmod.DB_PATH)
    dbmod.init_database()

    for entry in CURATED_PORTFOLIOS:
        ok = dbmod.save_portfolio(
            name=entry["name"], weights=entry["weights"],
            investment_amount=entry["investment_amount"],
            optimization_method=entry["optimization_method"],
            expected_return=entry["expected_return"],
            expected_volatility=entry["expected_volatility"],
            sharpe_ratio=entry["sharpe_ratio"],
            notes=entry["notes"], metadata=entry["metadata"],
        )
        if not ok:
            raise SystemExit(f"Failed to seed portfolio {entry['name']!r}")
        print(f"Seeded: {entry['name']}")

    print(f"\nDone -- {len(CURATED_PORTFOLIOS)} curated portfolios written to {dbmod.DB_PATH}")


if __name__ == "__main__":
    main()
