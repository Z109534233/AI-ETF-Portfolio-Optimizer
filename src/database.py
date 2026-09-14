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
    # User-scoping (Issue #22 section C): NULL for every portfolio saved
    # before auth existed (and for anonymous/public-demo saves after) --
    # load_all_portfolios() treats NULL the same as the shared "demo" owner,
    # so this column never hides or breaks pre-existing saved history.
    user_id = Column(String(100), nullable=True)

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


# Default owner for every row created before user-scoping existed (Issue #22
# section C -- auth architecture). Anonymous/public-demo activity is also
# recorded under this same value, so a fresh deployment with no auth
# configured behaves exactly as before: one shared demo namespace. A signed-in
# user's stable identifier (see src/auth.py) is stored here for their own
# rows so per-user data can be isolated without a destructive migration.
DEFAULT_USER_ID = "demo"


class UserHolding(Base):
    """A single lot in a user's real, self-reported "Current Holdings" list
    (Issue #22 section D) -- deliberately a SEPARATE table from
    PortfolioHolding above, which stores an OPTIMIZER's target weights, not
    what a user actually owns. No price/value columns are stored here: current
    market value and unrealized P/L are always computed on read from live
    data (never a fabricated/fallback price), so this table can never go
    stale relative to the market.
    """
    __tablename__ = "user_holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, default=DEFAULT_USER_ID)
    ticker = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False, default=0.0)
    average_cost = Column(Float, nullable=False, default=0.0)
    currency = Column(String(10), nullable=False, default="USD")
    purchase_date = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchlistItem(Base):
    """A user's watchlist (Issue #22 section D) -- tracked tickers with no
    quantity/cost basis, purely for monitoring."""
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, default=DEFAULT_USER_ID)
    ticker = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


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
    lightweight in-place ALTER TABLE run every time init_database() runs
    (get_engine() creates a fresh, unmemoized Engine each call -- there is
    no "already migrated this process" flag).

    Because Streamlit can serve concurrent sessions, two callers can both
    see the column missing before either has run ALTER TABLE; the loser
    then hits SQLite's "duplicate column name" error. That is swallowed
    here (the column exists either way, which is the only postcondition
    this function promises) rather than propagated up to fail the whole
    init_database() call -- a caller must never get get_session() == None
    just because it lost a harmless migration race.
    """
    try:
        existing_columns = {col["name"] for col in inspect(engine).get_columns("portfolios")}
    except Exception:
        return  # table doesn't exist yet -- create_all() will create it fresh, already correct
    if "metadata_json" not in existing_columns:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql("ALTER TABLE portfolios ADD COLUMN metadata_json TEXT")
        except Exception:
            pass  # another concurrent caller already added it -- column exists either way


def _ensure_user_id_column(engine, table: str) -> None:
    """Same guarded-migration pattern as _ensure_metadata_column() above,
    generalized to any table needing a backward-compatible `user_id` column
    (Issue #22 section C: auth architecture). Every row that existed before
    this column did defaults to NULL here, and every reader in this module
    treats NULL/missing the same as DEFAULT_USER_ID -- so pre-existing
    anonymous/demo data is never hidden or reassigned by this migration.
    """
    try:
        existing_columns = {col["name"] for col in inspect(engine).get_columns(table)}
    except Exception:
        return  # table doesn't exist yet -- create_all() will create it fresh, already correct
    if "user_id" not in existing_columns:
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR(100)")
        except Exception:
            pass  # another concurrent caller already added it -- column exists either way


def init_database():
    """Initialize database and create all tables if they don't exist."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        _ensure_metadata_column(engine)
        _ensure_user_id_column(engine, "portfolios")
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


# ── Current Holdings (Issue #22 section D) ──────────────────────────────────
def add_user_holding(user_id: str, ticker: str, quantity: float, average_cost: float,
                      currency: str, purchase_date: str = None) -> bool:
    """Add one lot to a user's Current Holdings. Ticker validity (must be in
    the supported ETF universe) is enforced by the CALLER before this is
    invoked -- this function only persists, it does not re-validate, so it
    stays reusable for any future bulk-import path."""
    session = get_session()
    if session is None:
        return False
    try:
        session.add(UserHolding(
            user_id=user_id, ticker=ticker, quantity=float(quantity),
            average_cost=float(average_cost), currency=currency, purchase_date=purchase_date,
        ))
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding holding: {e}")
        return False
    finally:
        session.close()


def load_user_holdings(user_id: str) -> list:
    """All holdings lots for one user, oldest first. Returns [] (never
    raises) on any DB error, matching load_all_portfolios()'s default
    fail-safe behavior."""
    session = get_session()
    if session is None:
        return []
    try:
        rows = (session.query(UserHolding)
                .filter(UserHolding.user_id == user_id)
                .order_by(UserHolding.created_at.asc()).all())
        return [
            {
                "id": r.id, "ticker": r.ticker, "quantity": r.quantity,
                "average_cost": r.average_cost, "currency": r.currency,
                "purchase_date": r.purchase_date,
                "created_at": r.created_at.strftime("%Y-%m-%d") if r.created_at else None,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Error loading holdings: {e}")
        return []
    finally:
        session.close()


def delete_user_holding(user_id: str, holding_id: int) -> bool:
    """Delete one holding lot. Scoped to `user_id` so one user can never
    delete another user's row even if they guess/replay an id."""
    session = get_session()
    if session is None:
        return False
    try:
        row = (session.query(UserHolding)
               .filter(UserHolding.id == holding_id, UserHolding.user_id == user_id).first())
        if row:
            session.delete(row)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error deleting holding: {e}")
        return False
    finally:
        session.close()


# ── Watchlist (Issue #22 section D) ─────────────────────────────────────────
def add_watchlist_item(user_id: str, ticker: str) -> bool:
    """Add a ticker to a user's watchlist. Silently no-ops (returns True
    without inserting a duplicate row) if the ticker is already on this
    user's watchlist, so repeated clicks can never create duplicate rows."""
    session = get_session()
    if session is None:
        return False
    try:
        existing = (session.query(WatchlistItem)
                    .filter(WatchlistItem.user_id == user_id, WatchlistItem.ticker == ticker).first())
        if existing:
            return True
        session.add(WatchlistItem(user_id=user_id, ticker=ticker))
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding watchlist item: {e}")
        return False
    finally:
        session.close()


def load_watchlist(user_id: str) -> list:
    """All tickers on one user's watchlist, oldest first. Returns [] on
    any DB error."""
    session = get_session()
    if session is None:
        return []
    try:
        rows = (session.query(WatchlistItem)
                .filter(WatchlistItem.user_id == user_id)
                .order_by(WatchlistItem.created_at.asc()).all())
        return [{"id": r.id, "ticker": r.ticker,
                 "created_at": r.created_at.strftime("%Y-%m-%d") if r.created_at else None} for r in rows]
    except Exception as e:
        print(f"Error loading watchlist: {e}")
        return []
    finally:
        session.close()


def remove_watchlist_item(user_id: str, item_id: int) -> bool:
    """Remove one watchlist entry. Scoped to `user_id` (same reasoning as
    delete_user_holding() above)."""
    session = get_session()
    if session is None:
        return False
    try:
        row = (session.query(WatchlistItem)
               .filter(WatchlistItem.id == item_id, WatchlistItem.user_id == user_id).first())
        if row:
            session.delete(row)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error removing watchlist item: {e}")
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
