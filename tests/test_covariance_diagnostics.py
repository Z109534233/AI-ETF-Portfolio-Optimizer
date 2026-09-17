"""
Covariance-matrix numerical-robustness diagnostics tests (Issue #45 item 1):
src.financial_metrics.covariance_diagnostics()/covariance_diagnostics_level()
and their wiring into src.portfolio_optimizer.run_optimization().
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from src.financial_metrics import covariance_diagnostics, covariance_diagnostics_level
from src.portfolio_optimizer import run_optimization


def _prices(n_days=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    base = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_days))
    return dates, base


def test_diagnostics_ok_for_genuinely_diversified_assets():
    dates, _ = _prices()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "A": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates))),
        "B": 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, len(dates))),
        "C": 100 * np.cumprod(1 + rng.normal(-0.0001, 0.008, len(dates))),
    }, index=dates)
    diag = covariance_diagnostics(df)
    assert diag["n_assets"] == 3
    assert diag["rank"] == 3
    assert diag["is_rank_deficient"] is False
    assert diag["condition_number"] is not None
    level = covariance_diagnostics_level(diag)
    assert level in ("ok", "moderate")  # independently-drawn random series should not be "severe"


def test_diagnostics_severe_for_duplicated_ticker_returns():
    """Two columns with IDENTICAL return series are perfectly collinear --
    the raw covariance matrix must be reported rank-deficient/severe, never
    silently regularized away before diagnostics are computed."""
    dates, base = _prices()
    df = pd.DataFrame({"A": base, "B": base, "C": base * 1.5}, index=dates)
    diag = covariance_diagnostics(df)
    assert diag["is_rank_deficient"] is True
    assert diag["rank"] < diag["n_assets"]
    assert covariance_diagnostics_level(diag) == "severe"
    # avg pairwise correlation of perfectly-collinear series is ~1.0
    assert diag["avg_pairwise_correlation"] > 0.99


def test_diagnostics_level_uses_measured_wording_not_singular_claim():
    """covariance_diagnostics_level() returns a bucket key, never the
    literal word 'singular' -- callers must phrase this as
    ill-conditioned/near-collinear, not an unconditional singularity claim."""
    dates, base = _prices()
    df = pd.DataFrame({"A": base, "B": base}, index=dates)
    diag = covariance_diagnostics(df)
    level = covariance_diagnostics_level(diag)
    assert level in ("severe", "moderate", "ok")
    assert level != "singular"


def test_run_optimization_attaches_covariance_diagnostics():
    dates, base = _prices()
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "A": base,
        "B": 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(dates))),
    }, index=dates)
    result = run_optimization(df, method="Equal Weight", risk_free_rate=0.04)
    assert "covariance_diagnostics" in result
    assert "covariance_diagnostics_level" in result
    diag = result["covariance_diagnostics"]
    assert diag["n_assets"] == 2
    assert result["covariance_diagnostics_level"] in ("ok", "moderate", "severe")


def test_run_optimization_diagnostics_computed_before_regularization():
    """Duplicated-ticker input (perfectly collinear) must surface as
    "severe" in the optimizer's own diagnostics -- proving the diagnostic
    is computed from the RAW covariance, not the ridge-regularized one
    (which would otherwise mask the collinearity)."""
    dates, base = _prices()
    df = pd.DataFrame({"A": base, "B": base}, index=dates)
    result = run_optimization(df, method="Equal Weight", risk_free_rate=0.04)
    assert result["covariance_diagnostics"]["is_rank_deficient"] is True
    assert result["covariance_diagnostics_level"] == "severe"
