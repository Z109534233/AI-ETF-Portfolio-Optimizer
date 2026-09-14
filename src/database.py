"""
Database Module
SQLite database management using SQLAlchemy for portfolio storage.
"""

import os
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, ForeignKey, Text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Database path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "portfolio.db")

# Recorded in a saved portfolio's experiment metadata (Issue #20 section 9C)
# so a reloaded portfolio's methodology can be traced to the app revision
# that produced it. Bump when the optimization/estimator methodology changes.
APP_VERSION = "2026.09-issue20"

Base = declarative_base()


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    investment_amount = Column(Float, default=10000.0)
    optimization_method = Column(String(100), default="Equal Weight")
    expected_return = Column(Float, default=0.0)
    expected_volatility = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    notes = Column(Text, default="")
    # Experiment-tracking metadata (Issue #20 section 9C) -- a versioned
    # JSON blob rather than one column per field, so new metadata fields can
    # be added later without another schema migration. Nullable so every
    # portfolio saved before this column existed still loads (see
    # _ensure_metadata_column() below); load_all_portfolios() always
    # returns a "metadata" dict, defaulting to {} when this is NULL/absent.
    metadata_json = Column(Text, nullable=True)

    holdings = relationship("PortfolioHolding", back_populates="portfolio",
                            cascade="all, delete-orphan")


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(20), nullable=False)
    weight = Column(Float, default=0.0)
    amount = Column(Float, default=0.0)

    portfolio = relationship("Portfolio", back_populates="holdings")


class SimulationHistory(Base):
    __tablename__ = "simulation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    initial_investment = Column(Float, default=0.0)
    monthly_contribution = Column(Float, default=0.0)
    years = Column(Integer, default=10)
    expected_return = Column(Float, default=0.0)
    final_value = Column(Float, default=0.0)


def get_engine():
    """Create and return SQLAlchemy engine."""
    os.makedirs(DB_DIR, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    return engine


def _ensure_metadata_column(engine) -> None:
    """Backward-compatible migration: add portfolios.metadata_json to a
    database file created before this column existed (e.g. the committed
    demo database/portfolio.db, or any user's existing local file) without
    touching any existing row. Base.metadata.create_all() only creates
    missing TABLES, never adds a column to an existing table, so this is a
    lightweight in-place ALTER TABLE run once per process at startup."""
    try:
        existing_columns = {col["name"] for col in inspect(engine).get_columns("portfolios")}
    except Exception:
        return  # table doesn't exist yet -- create_all() will create it fresh, already correct
    if "metadata_json" not in existing_columns:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE portfolios ADD COLUMN metadata_json TEXT")


def init_database():
    """Initialize database and create all tables if they don't exist."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        _ensure_metadata_column(engine)
        return engine
    except Exception as e:
        print(f"Database initialization error: {e}")
        return None


def get_session():
    """Return a new database session."""
    engine = init_database()
    if engine is None:
        return None
    Session = sessionmaker(bind=engine)
    return Session()


def save_portfolio(name: str, weights: dict, investment_amount: float,
                   optimization_method: str, expected_return: float,
                   expected_volatility: float, sharpe_ratio: float,
                   notes: str = "", metadata: dict = None) -> bool:
    """Save a portfolio to the database. Returns True on success.

    `metadata` (Issue #20 section 9C -- experiment tracking) is an optional
    dict of reproducibility fields (historical window, market, risk-free
    rate, weight bounds, allow_short, estimator choices, strategy, asset
    universe, generated timestamp, app version, etc.) stored as a single
    versioned JSON blob. Old callers that don't pass it still work --
    metadata_json stays NULL, and Portfolio History shows "no metadata
    recorded" rather than fabricating any of these fields.
    """
    session = get_session()
    if session is None:
        return False
    try:
        portfolio = Portfolio(
            name=name,
            investment_amount=investment_amount,
            optimization_method=optimization_method,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            notes=notes,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        session.add(portfolio)
        session.flush()

        for ticker, weight in weights.items():
            holding = PortfolioHolding(
                portfolio_id=portfolio.id,
                ticker=ticker,
                weight=float(weight),
                amount=float(weight) * investment_amount
            )
            session.add(holding)

        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error saving portfolio: {e}")
        return False
    finally:
        session.close()


def load_all_portfolios(raise_on_error: bool = False) -> list:
    """Load all saved portfolios from the database.

    `raise_on_error=False` (default, existing behavior preserved for
    current callers) swallows any failure and returns [] -- indistinguishable
    from a genuinely empty table. Portfolio History needs to tell those two
    states apart (a real DB/connection error vs. "no portfolios saved yet"),
    so it passes `raise_on_error=True` and handles the exception itself.
    """
    session = get_session()
    if session is None:
        if raise_on_error:
            raise RuntimeError("Could not open a database session.")
        return []
    try:
        portfolios = session.query(Portfolio).order_by(Portfolio.created_at.desc()).all()
        result = []
        for p in portfolios:
            holdings = {h.ticker: h.weight for h in p.holdings}
            try:
                metadata = json.loads(p.metadata_json) if getattr(p, "metadata_json", None) else {}
            except (TypeError, ValueError):
                metadata = {}  # corrupt/legacy value -- never crash the page over it
            result.append({
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "N/A",
                "investment_amount": p.investment_amount,
                "optimization_method": p.optimization_method,
                "expected_return": p.expected_return,
                "expected_volatility": p.expected_volatility,
                "sharpe_ratio": p.sharpe_ratio,
                "notes": p.notes,
                "holdings": holdings,
                "metadata": metadata,
            })
        return result
    except Exception as e:
        print(f"Error loading portfolios: {e}")
        if raise_on_error:
            raise
        return []
    finally:
        session.close()


def find_duplicate_portfolio(weights: dict, optimization_method: str, investment_amount: float,
                              tolerance: float = 1e-6):
    """Return the most recently saved portfolio with the same method,
    investment amount, and holdings weights (within `tolerance`), or None.

    Used by the Save & Actions UI (Issue #20 section 9D) to warn before an
    accidental repeated identical save, rather than silently accumulating
    debug duplicates in Portfolio History the way the committed demo
    database previously did. Does not delete or block anything itself --
    callers decide whether to still save after the warning.
    """
    try:
        existing = load_all_portfolios(raise_on_error=True)
    except Exception:
        return None
    rounded_weights = {tk: round(w, 6) for tk, w in weights.items()}
    for p in existing:
        if p["optimization_method"] != optimization_method:
            continue
        if abs((p["investment_amount"] or 0) - investment_amount) > tolerance:
            continue
        other_weights = {tk: round(w, 6) for tk, w in (p["holdings"] or {}).items()}
        if other_weights == rounded_weights:
            return p
    return None


def delete_portfolio(portfolio_id: int) -> bool:
    """Delete a portfolio by ID. Returns True on success."""
    session = get_session()
    if session is None:
        return False
    try:
        portfolio = session.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if portfolio:
            session.delete(portfolio)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error deleting portfolio: {e}")
        return False
    finally:
        session.close()


def save_simulation(initial_investment: float, monthly_contribution: float,
                    years: int, expected_return: float, final_value: float) -> bool:
    """Save a simulation result to history."""
    session = get_session()
    if session is None:
        return False
    try:
        sim = SimulationHistory(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            expected_return=expected_return,
            final_value=final_value
        )
        session.add(sim)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()
