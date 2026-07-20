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


def init_database():
    """Initialize database and create all tables if they don't exist."""
    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
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
                   notes: str = "") -> bool:
    """Save a portfolio to the database. Returns True on success."""
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
            notes=notes
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


def load_all_portfolios() -> list:
    """Load all saved portfolios from the database."""
    session = get_session()
    if session is None:
        return []
    try:
        portfolios = session.query(Portfolio).order_by(Portfolio.created_at.desc()).all()
        result = []
        for p in portfolios:
            holdings = {h.ticker: h.weight for h in p.holdings}
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
            })
        return result
    except Exception as e:
        print(f"Error loading portfolios: {e}")
        return []
    finally:
        session.close()


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
