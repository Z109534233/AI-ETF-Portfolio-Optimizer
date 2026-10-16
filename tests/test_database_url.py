"""Database backend selection tests -- no external connection required."""

import src.database as db


def test_database_url_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db._configured_database_url().startswith("sqlite:///")
    assert db.database_backend() == "SQLite"
    assert db.database_is_persistent_configured() is False


def test_legacy_postgres_scheme_is_normalized(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://example.invalid/appdb")
    assert db._configured_database_url() == "postgresql://example.invalid/appdb"
    assert db.database_backend() == "PostgreSQL"
    assert db.database_is_persistent_configured() is True
