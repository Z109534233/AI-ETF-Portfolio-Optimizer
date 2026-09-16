"""
Portfolio History -- deterministic tests (Issue #18 Stage 4).

Covers the reload-to-canonical-state gap that was the most explicit finding
in the issue: saved portfolios previously had no way back into
st.session_state["current_portfolio"], so Investment Simulator / Risk
Analytics / AI Advisor could never consume a saved portfolio directly.

Every test here runs against an isolated, throwaway SQLite file (via the
`isolated_db` fixture below) -- NEVER the real committed database/portfolio.db,
which already has ~140 rows of pre-existing data from prior development that
must not be touched by a test run.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

import src.database as dbmod
from src.financial_metrics import portfolio_diagnosis


def _apptest_from_file(rel_path, **kwargs):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    # Pre-existing AppTest limitation (also worked around the same way in
    # tests/test_portfolio_optimizer.py): AppTest runs a single page in
    # isolation and cannot resolve st.page_link()'s multi-page-app
    # navigation, which every page's sidebar renders unconditionally.
    st.page_link = lambda *a, **k: None
    path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point src.database at a throwaway SQLite file for this test only.
    Every caller of get_session()/get_engine() (save_portfolio, load_all_portfolios,
    delete_portfolio, and the page module that imports them by reference)
    reads DB_DIR/DB_PATH from src.database's module globals at call time, so
    patching them here transparently redirects all of it without touching
    the real committed database/portfolio.db.
    """
    monkeypatch.setattr(dbmod, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test_portfolio.db"))
    dbmod.init_database()
    return dbmod


PORTFOLIO_A_HOLDINGS = {"VOO": 0.6, "VTI": 0.4}
PORTFOLIO_B_HOLDINGS = {"QQQ": 0.5, "SCHD": 0.3, "SPY": 0.2}


def _seed_portfolio(db, name, holdings, investment_amount=10000.0,
                     method="Maximum Sharpe Ratio", ret=0.12, vol=0.15, sharpe=0.8, notes=""):
    ok = db.save_portfolio(
        name=name, weights=holdings, investment_amount=investment_amount,
        optimization_method=method, expected_return=ret, expected_volatility=vol,
        sharpe_ratio=sharpe, notes=notes,
    )
    assert ok, "seed save_portfolio() call itself failed -- fixture is broken"


# ── save / load round trip ───────────────────────────────────────────────
def test_save_and_load_round_trip(isolated_db):
    _seed_portfolio(isolated_db, "RoundTrip", PORTFOLIO_A_HOLDINGS,
                    investment_amount=25000.0, method="Equal Weight", ret=0.10, vol=0.12, sharpe=0.83)
    loaded = isolated_db.load_all_portfolios()
    assert len(loaded) == 1
    p = loaded[0]
    assert p["name"] == "RoundTrip"
    assert p["optimization_method"] == "Equal Weight"
    assert p["investment_amount"] == 25000.0
    assert abs(p["expected_return"] - 0.10) < 1e-9
    assert abs(p["expected_volatility"] - 0.12) < 1e-9
    assert abs(p["sharpe_ratio"] - 0.83) < 1e-9
    assert p["holdings"] == PORTFOLIO_A_HOLDINGS


def test_load_all_portfolios_raise_on_error_distinguishes_failure_from_empty(isolated_db, monkeypatch):
    # A genuinely empty (but reachable) DB must not raise.
    assert isolated_db.load_all_portfolios(raise_on_error=True) == []

    # A real failure (session cannot even be opened) must raise when asked,
    # not silently look identical to "no portfolios saved yet".
    monkeypatch.setattr(isolated_db, "get_session", lambda: None)
    with pytest.raises(RuntimeError):
        isolated_db.load_all_portfolios(raise_on_error=True)
    # Default behavior (existing callers, e.g. Market Intelligence) is unchanged.
    assert isolated_db.load_all_portfolios() == []


# ── reload into canonical current_portfolio state ────────────────────────
def test_set_as_current_portfolio_populates_canonical_state(isolated_db):
    _seed_portfolio(isolated_db, "Reloadable", PORTFOLIO_A_HOLDINGS,
                    investment_amount=10000.0, method="Maximum Sharpe Ratio",
                    ret=0.19, vol=0.10, sharpe=1.4)
    saved = isolated_db.load_all_portfolios()[0]

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    btn = next((b for b in at.button if b.key == f"hist_set_current_{saved['id']}"), None)
    assert btn is not None, "Set as Current Portfolio button not found"
    btn.click()
    at.run()
    assert at.exception == []

    cp = at.session_state["current_portfolio"]
    assert cp["portfolio_id"] == f"history-{saved['id']}"
    assert cp["strategy"] == "Maximum Sharpe Ratio"
    assert cp["weights"] == PORTFOLIO_A_HOLDINGS
    assert cp["tickers"] == list(PORTFOLIO_A_HOLDINGS.keys())
    assert cp["investment_amount"] == 10000.0
    assert abs(cp["expected_return"] - 0.19) < 1e-9
    assert abs(cp["volatility"] - 0.10) < 1e-9
    assert abs(cp["sharpe_ratio"] - 1.4) < 1e-9
    # VOO/VTI are both US-listed -- market should be inferred, not left unset.
    assert cp["market"] == "United States"
    expected_diag = portfolio_diagnosis(PORTFOLIO_A_HOLDINGS)
    assert cp["largest_position"]["ticker"] == expected_diag["largest_ticker"]
    assert abs(cp["effective_holdings"] - expected_diag["effective_holdings"]) < 1e-9

    success_texts = [s.value for s in at.success]
    assert any("Reloadable" in s for s in success_texts), success_texts


def test_reloaded_portfolio_handoff_to_ai_advisor_no_exception(isolated_db):
    """Issue #18 Stage 7: a portfolio reloaded via "Set as Current Portfolio"
    has historical_start_date/historical_end_date = None (the saved-portfolio
    DB schema has no date columns -- see _set_as_current_portfolio() in
    pages/7_Portfolio_History.py). AI Advisor must fall back to a sensible
    default date range instead of passing the literal string "None" through
    to the price downloader.
    """
    _seed_portfolio(isolated_db, "ReloadToAdvisor", PORTFOLIO_A_HOLDINGS,
                     investment_amount=5000.0, method="Minimum Volatility",
                     ret=0.09, vol=0.12, sharpe=0.6)
    saved = isolated_db.load_all_portfolios()[0]

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    btn = next((b for b in at.button if b.key == f"hist_set_current_{saved['id']}"), None)
    assert btn is not None
    btn.click()
    at.run()
    cp = at.session_state["current_portfolio"]
    assert cp["historical_start_date"] is None
    assert cp["historical_end_date"] is None

    import streamlit as st
    st.page_link = lambda *a, **k: None
    ai_at = _apptest_from_file("pages/6_AI_Advisor.py", default_timeout=180)
    ai_at.session_state["language"] = "en"
    ai_at.session_state["current_portfolio"] = cp
    ai_at.run()
    exc = ai_at.exception[0] if ai_at.exception else None
    assert exc is None, str(exc)
    ai_result = ai_at.session_state["ai_result"] if "ai_result" in ai_at.session_state else None
    assert ai_result is not None
    assert ai_result["context"]["portfolio"]["weights"] == PORTFOLIO_A_HOLDINGS


def test_set_as_current_portfolio_shows_active_badge_on_reselect(isolated_db):
    _seed_portfolio(isolated_db, "BadgeCheck", PORTFOLIO_A_HOLDINGS)
    saved = isolated_db.load_all_portfolios()[0]

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    btn = next(b for b in at.button if b.key == f"hist_set_current_{saved['id']}")
    btn.click()
    at.run()

    corpus = "\n".join(c.value for c in at.caption)
    assert "Currently Active" in corpus, corpus


# ── compare two portfolios ────────────────────────────────────────────────
def test_compare_two_portfolios_shows_effective_holdings(isolated_db):
    _seed_portfolio(isolated_db, "PortA", PORTFOLIO_A_HOLDINGS, ret=0.10, vol=0.12, sharpe=0.83)
    _seed_portfolio(isolated_db, "PortB", PORTFOLIO_B_HOLDINGS, ret=0.15, vol=0.18, sharpe=0.83)
    saved = isolated_db.load_all_portfolios()
    assert len(saved) == 2

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    dataframes = list(at.dataframe)
    assert dataframes, "no dataframe elements rendered"
    found_effective_holdings_row = any(
        "Effective Holdings" in "\n".join(str(x) for x in df.value.index)
        for df in dataframes if hasattr(df.value, "index")
    )
    assert found_effective_holdings_row, [df.value for df in dataframes]

    expected_a = portfolio_diagnosis(PORTFOLIO_A_HOLDINGS)["effective_holdings"]
    expected_b = portfolio_diagnosis(PORTFOLIO_B_HOLDINGS)["effective_holdings"]
    compare_tables = [df.value for df in dataframes if "Effective Holdings" in "\n".join(str(x) for x in getattr(df.value, "index", []))]
    assert compare_tables, "comparison table not found among dataframes"
    compare_df = compare_tables[0]
    row = compare_df.loc["Effective Holdings"]
    values = [row[col] for col in compare_df.columns]
    assert f"{expected_a:.2f}" in values
    assert f"{expected_b:.2f}" in values


# ── empty-state vs error-state ───────────────────────────────────────────
def test_empty_state_when_no_portfolios_saved(isolated_db):
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "No Saved Portfolios Yet" in corpus
    assert "Could Not Load Saved Portfolios" not in corpus


def test_error_state_when_database_fails_distinct_from_empty_state(isolated_db, monkeypatch):
    _seed_portfolio(isolated_db, "Unreachable", PORTFOLIO_A_HOLDINGS)

    def _boom():
        raise RuntimeError("disk I/O error (simulated)")

    monkeypatch.setattr(dbmod, "get_session", _boom)
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Could Not Load Saved Portfolios" in corpus
    assert "disk I/O error (simulated)" in corpus
    # Must not be confused with the true empty-state copy.
    assert "No Saved Portfolios Yet" not in corpus


# ── corrupt / partial record handling ─────────────────────────────────────
def test_corrupt_record_renders_null_safe_instead_of_crashing(isolated_db):
    import sqlite3

    _seed_portfolio(isolated_db, "Healthy", PORTFOLIO_A_HOLDINGS, ret=0.10, vol=0.12, sharpe=0.83)

    # Insert a record with true SQL NULL numeric fields via a raw connection
    # -- going through the ORM (dbmod.Portfolio(...)) does NOT produce this,
    # since SQLAlchemy substitutes each Column's declared `default=` whenever
    # the attribute is None at flush time. A raw INSERT is the only way to
    # reproduce a genuinely corrupt/legacy row (e.g. hand-edited, or from
    # before a column had a NOT NULL/default constraint) for this test.
    conn = sqlite3.connect(dbmod.DB_PATH)
    conn.execute(
        "INSERT INTO portfolios (name, investment_amount, optimization_method, "
        "expected_return, expected_volatility, sharpe_ratio, notes) VALUES "
        "('Corrupt', NULL, NULL, NULL, NULL, NULL, '')"
    )
    conn.commit()
    conn.close()

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(c.value for c in at.caption)
    assert "missing or have unreadable numeric data" in corpus


# ── experiment metadata (Issue #20 section 9C) ───────────────────────────
SAMPLE_METADATA = {
    "schema_version": 1,
    "historical_start_date": "2023-01-01",
    "historical_end_date": "2024-01-01",
    "market": "United States",
    "risk_free_rate": 0.05,
    "min_weight": 0.0,
    "max_weight": 1.0,
    "allow_short": False,
    "expected_return_estimator": "Historical CAGR (annualized_return)",
    "covariance_estimator": "Sample covariance (historical, annualized)",
    "strategy": "Maximum Sharpe Ratio",
    "asset_universe": ["VOO", "VTI"],
    "generated_at": "2024-01-01T00:00:00+00:00",
    "data_as_of": "2024-01-01",
    "app_version": "test",
}


def test_metadata_roundtrip(isolated_db):
    ok = isolated_db.save_portfolio(
        name="WithMetadata", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.12,
        expected_volatility=0.15, sharpe_ratio=0.8, metadata=SAMPLE_METADATA,
    )
    assert ok
    loaded = isolated_db.load_all_portfolios()[0]
    assert loaded["metadata"] == SAMPLE_METADATA


def test_metadata_defaults_to_empty_dict_when_not_provided(isolated_db):
    _seed_portfolio(isolated_db, "NoMetadata", PORTFOLIO_A_HOLDINGS)
    loaded = isolated_db.load_all_portfolios()[0]
    assert loaded["metadata"] == {}


def test_metadata_column_migration_on_legacy_database(tmp_path, monkeypatch):
    """A database file created before metadata_json existed (e.g. an old
    local copy of database/portfolio.db) must gain the column in place, and
    every pre-existing row must keep loading with metadata == {} rather
    than crash init_database() or load_all_portfolios()."""
    import sqlite3

    legacy_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(legacy_path)
    conn.execute(
        "CREATE TABLE portfolios (id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR(200) NOT NULL, "
        "created_at DATETIME, investment_amount FLOAT, optimization_method VARCHAR(100), "
        "expected_return FLOAT, expected_volatility FLOAT, sharpe_ratio FLOAT, notes TEXT)"
    )
    conn.execute(
        "CREATE TABLE portfolio_holdings (id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id INTEGER NOT NULL, "
        "ticker VARCHAR(20), weight FLOAT, amount FLOAT)"
    )
    conn.execute(
        "INSERT INTO portfolios (name, investment_amount, optimization_method, expected_return, "
        "expected_volatility, sharpe_ratio, notes) VALUES ('LegacyRow', 10000.0, 'Equal Weight', 0.1, 0.15, 0.6, '')"
    )
    conn.execute("INSERT INTO portfolio_holdings (portfolio_id, ticker, weight, amount) VALUES (1, 'VOO', 1.0, 10000.0)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(dbmod, "DB_DIR", str(tmp_path))
    monkeypatch.setattr(dbmod, "DB_PATH", legacy_path)
    engine = dbmod.init_database()
    assert engine is not None

    from sqlalchemy import inspect as sa_inspect
    columns = {col["name"] for col in sa_inspect(engine).get_columns("portfolios")}
    assert "metadata_json" in columns

    loaded = dbmod.load_all_portfolios()
    assert len(loaded) == 1
    assert loaded[0]["name"] == "LegacyRow"
    assert loaded[0]["metadata"] == {}
    assert loaded[0]["holdings"] == {"VOO": 1.0}


# ── duplicate-save detection (Issue #20 section 9D) ──────────────────────
def test_find_duplicate_portfolio_detects_identical_save(isolated_db):
    _seed_portfolio(isolated_db, "Original", PORTFOLIO_A_HOLDINGS,
                     investment_amount=10000.0, method="Equal Weight")
    dup = isolated_db.find_duplicate_portfolio(PORTFOLIO_A_HOLDINGS, "Equal Weight", 10000.0)
    assert dup is not None
    assert dup["name"] == "Original"


def test_find_duplicate_portfolio_ignores_different_weights(isolated_db):
    _seed_portfolio(isolated_db, "Original", PORTFOLIO_A_HOLDINGS,
                     investment_amount=10000.0, method="Equal Weight")
    dup = isolated_db.find_duplicate_portfolio(PORTFOLIO_B_HOLDINGS, "Equal Weight", 10000.0)
    assert dup is None


def test_find_duplicate_portfolio_ignores_different_method_or_amount(isolated_db):
    _seed_portfolio(isolated_db, "Original", PORTFOLIO_A_HOLDINGS,
                     investment_amount=10000.0, method="Equal Weight")
    assert isolated_db.find_duplicate_portfolio(PORTFOLIO_A_HOLDINGS, "Minimum Volatility", 10000.0) is None
    assert isolated_db.find_duplicate_portfolio(PORTFOLIO_A_HOLDINGS, "Equal Weight", 5000.0) is None


# ── Active Holdings Only + Experiment Details (Issue #20 section 9B/9C) ──
def test_active_holdings_only_hides_near_zero_weights(isolated_db):
    holdings = {"VOO": 0.995, "BND": 0.0005}
    _seed_portfolio(isolated_db, "NearZero", holdings)
    saved = isolated_db.load_all_portfolios()[0]

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()

    dataframes = list(at.dataframe)
    holdings_tables = [df.value for df in dataframes if "VOO" in getattr(df.value, "index", [])]
    assert holdings_tables, "holdings detail table not found"
    assert "BND" not in holdings_tables[0].index

    corpus = "\n".join(m.value for m in at.markdown)
    assert "Active Holdings Only" in corpus


def test_experiment_details_shows_metadata_when_present(isolated_db):
    isolated_db.save_portfolio(
        name="MetaPortfolio", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.12,
        expected_volatility=0.15, sharpe_ratio=0.8, metadata=SAMPLE_METADATA,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Historical CAGR" in corpus
    assert "Sample covariance" in corpus


def test_experiment_details_empty_state_for_legacy_portfolio(isolated_db):
    _seed_portfolio(isolated_db, "Legacy", PORTFOLIO_A_HOLDINGS)
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(c.value for c in at.caption)
    assert "no metadata was recorded" in corpus


# ── Demo/synthetic badge (Issue #20 release-gate review, automated PR
# reviewer finding): scripts/reset_demo_portfolio_history.py's curated rows
# carry hand-authored, not live-optimizer-computed, performance figures --
# metadata["synthetic_demo"] must render an unmissable badge so they can
# never be mistaken for a real optimization result. ───────────────────────

def test_synthetic_demo_portfolio_shows_demo_badge(isolated_db):
    isolated_db.save_portfolio(
        name="US_Balanced", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Custom Allocation", expected_return=0.075,
        expected_volatility=0.11, sharpe_ratio=0.50,
        metadata={"schema_version": 1, "synthetic_demo": True},
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    warnings = "\n".join(w.value for w in at.warning)
    assert "Curated Demo Example" in warnings
    summary_corpus = str(at.dataframe[0].value)
    assert "(Demo)" in summary_corpus


def test_real_saved_portfolio_shows_no_demo_badge(isolated_db):
    _seed_portfolio(isolated_db, "RealUserPortfolio", PORTFOLIO_A_HOLDINGS)
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    warnings = "\n".join(w.value for w in at.warning)
    assert "Curated Demo Example" not in warnings
    summary_corpus = str(at.dataframe[0].value)
    assert "(Demo)" not in summary_corpus


# ── Issue #43 item K: legacy/custom saved-strategy display ───────────────
def test_legacy_custom_allocation_strategy_localized_in_zh_tw(isolated_db):
    isolated_db.save_portfolio(
        name="LegacyCustom", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Custom Allocation", expected_return=0.08,
        expected_volatility=0.12, sharpe_ratio=0.6,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    assert at.exception == []
    summary_corpus = str(at.dataframe[0].value)
    assert "自訂配置" in summary_corpus
    assert "Custom Allocation" not in summary_corpus


def test_custom_allocation_never_added_to_optimizer_dropdown():
    from src.i18n import OPTIMIZATION_METHOD_KEYS
    assert "Custom Allocation" not in OPTIMIZATION_METHOD_KEYS
    assert len(OPTIMIZATION_METHOD_KEYS) == 5


def test_unknown_legacy_strategy_falls_back_safely_without_crashing():
    from src.i18n import t_saved_strategy
    assert t_saved_strategy("Some Future Method") == "Some Future Method"
    assert t_saved_strategy("") == "—"
    assert t_saved_strategy(None) == "—"


# ── Issue #43 item J: saved methodology metadata localized at display ────
def test_experiment_details_localizes_estimator_labels_in_zh_tw(isolated_db):
    isolated_db.save_portfolio(
        name="MetaPortfolioZh", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.12,
        expected_volatility=0.15, sharpe_ratio=0.8, metadata=SAMPLE_METADATA,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "歷史複合年化成長率" in corpus
    assert "歷史樣本共變異數" in corpus
    # No raw snake_case internal function-argument name leaked into zh-TW.
    assert "annualized_return" not in corpus


# ── Issue #43 item L: meaningful Market display (never a bare dash) ──────
def test_multi_market_display_from_metadata_markets_list(isolated_db):
    meta = dict(SAMPLE_METADATA)
    meta["markets"] = ["United States", "Taiwan"]
    isolated_db.save_portfolio(
        name="MultiMarket", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.1,
        expected_volatility=0.15, sharpe_ratio=0.7, metadata=meta,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Multi-market" in corpus
    assert "United States" in corpus and "Taiwan" in corpus


def test_market_inferred_from_holdings_when_no_markets_metadata(isolated_db):
    # VOO/VTI are both US-listed and SAMPLE_METADATA has no "markets" list
    # (only the older singular "market" string) -- the display must infer
    # from holdings rather than trust the old field.
    isolated_db.save_portfolio(
        name="InferredMarket", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.1,
        expected_volatility=0.15, sharpe_ratio=0.7, metadata=SAMPLE_METADATA,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "**Market**: United States" in corpus


def test_market_shows_localized_not_recorded_instead_of_bare_dash(isolated_db):
    isolated_db.save_portfolio(
        name="UnknownMarket", weights={"ZZZFAKETICKER": 1.0}, investment_amount=10000.0,
        optimization_method="Equal Weight", expected_return=0.05,
        expected_volatility=0.1, sharpe_ratio=0.5,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "zh-TW"
    at.run()
    assert at.exception == []
    corpus = "\n".join(m.value for m in at.markdown)
    assert "未記錄" in corpus
    assert "市場**: —" not in corpus


# ── Issue #43 item M: old-version provenance notice ───────────────────────
def test_legacy_app_version_notice_shown_when_version_differs(isolated_db):
    meta = dict(SAMPLE_METADATA)
    meta["app_version"] = "2020.01-ancient"
    isolated_db.save_portfolio(
        name="OldVersionPortfolio", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.1,
        expected_volatility=0.15, sharpe_ratio=0.7, metadata=meta,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    assert any(
        "created with app version" in i.value and "2020.01-ancient" in i.value
        for i in at.info
    )


def test_no_legacy_notice_when_version_matches_current(isolated_db):
    meta = dict(SAMPLE_METADATA)
    meta["app_version"] = dbmod.APP_VERSION
    isolated_db.save_portfolio(
        name="CurrentVersionPortfolio", weights=PORTFOLIO_A_HOLDINGS, investment_amount=10000.0,
        optimization_method="Maximum Sharpe Ratio", expected_return=0.1,
        expected_volatility=0.15, sharpe_ratio=0.7, metadata=meta,
    )
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    assert not any("created with app version" in i.value for i in at.info)


# ── Issue #43 item N: delete requires explicit second confirmation ───────
def test_delete_button_disabled_until_confirmed_then_deletes(isolated_db):
    _seed_portfolio(isolated_db, "ToDelete", PORTFOLIO_A_HOLDINGS)
    saved = isolated_db.load_all_portfolios()[0]

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []

    delete_btn = next(b for b in at.button if b.label == "Delete Portfolio")
    assert delete_btn.disabled is True

    checkbox = next(c for c in at.checkbox if c.key == f"hist_delete_confirm_{saved['id']}")
    checkbox.set_value(True)
    at.run()

    delete_btn = next(b for b in at.button if b.label == "Delete Portfolio")
    assert delete_btn.disabled is False
    delete_btn.click()
    at.run()
    assert at.exception == []
    assert isolated_db.load_all_portfolios() == []


def test_changing_selected_portfolio_resets_delete_confirmation(isolated_db):
    _seed_portfolio(isolated_db, "PortA", PORTFOLIO_A_HOLDINGS)
    _seed_portfolio(isolated_db, "PortB", PORTFOLIO_B_HOLDINGS)
    saved = isolated_db.load_all_portfolios()
    id_a = next(p["id"] for p in saved if p["name"] == "PortA")
    id_b = next(p["id"] for p in saved if p["name"] == "PortB")

    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()

    # Explicitly select PortA first -- load_all_portfolios() orders newest
    # first, so the delete dropdown's own default selection is not assumed.
    delete_select = next(sb for sb in at.selectbox if sb.key == "delete_select")
    delete_select.set_value(id_a)
    at.run()

    checkbox_a = next(c for c in at.checkbox if c.key == f"hist_delete_confirm_{id_a}")
    checkbox_a.set_value(True)
    at.run()
    delete_btn = next(b for b in at.button if b.label == "Delete Portfolio")
    assert delete_btn.disabled is False

    delete_select = next(sb for sb in at.selectbox if sb.key == "delete_select")
    delete_select.set_value(id_b)
    at.run()

    # A different portfolio is now selected -- its own confirmation
    # checkbox is a fresh, unchecked widget, so the button must be disabled
    # again even though PortA's checkbox was previously checked.
    delete_btn = next(b for b in at.button if b.label == "Delete Portfolio")
    assert delete_btn.disabled is True
    assert len(isolated_db.load_all_portfolios()) == 2


# ── Issue #43 item Q: no literal Markdown heading-marker leak ────────────
def test_no_literal_markdown_heading_leak_in_captions_and_alerts(isolated_db):
    _seed_portfolio(isolated_db, "HeadingLeakCheck", PORTFOLIO_A_HOLDINGS)
    at = _apptest_from_file("pages/7_Portfolio_History.py", default_timeout=180)
    at.session_state["language"] = "en"
    at.run()
    assert at.exception == []
    # st.markdown() elements legitimately include real "### " headings
    # (e.g. the sidebar title) that render correctly -- this regression
    # check instead targets non-markdown display surfaces (captions,
    # success/error/warning/info banners) where a literal, unrendered
    # "### " prefix would indicate the old leaked-markdown bug.
    for collection in (at.caption, at.success, at.error, at.warning, at.info):
        for el in collection:
            assert "### " not in el.value, el.value
