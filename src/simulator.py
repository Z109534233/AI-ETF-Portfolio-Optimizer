"""
Investment Simulator Module
Monte Carlo simulation for long-term investment projections.
"""

import numpy as np
import pandas as pd


MARKET_SCENARIOS = {
    "Bull Market": {"return": 0.15, "volatility": 0.12},
    "Base Case": {"return": 0.10, "volatility": 0.15},
    "Bear Market": {"return": 0.02, "volatility": 0.25},
    "Sideways Market": {"return": 0.04, "volatility": 0.10},
}


def simulate_investment(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_return: float,
    annual_volatility: float,
    inflation_rate: float = 0.025,
    annual_fee: float = 0.001,
    n_simulations: int = 1000,
    seed: int = 42
) -> dict:
    """
    Run Monte Carlo simulation for long-term investment growth.

    Returns a dict with:
    - paths: DataFrame of simulation paths (shape: months x n_simulations)
    - summary: dict with key statistics
    - annual_table: DataFrame with yearly balance summary
    """
    np.random.seed(seed)
    months = years * 12
    monthly_return = (1 + annual_return) ** (1 / 12) - 1
    monthly_vol = annual_volatility / np.sqrt(12)
    monthly_fee = (1 + annual_fee) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1

    # Simulate paths
    paths = np.zeros((months + 1, n_simulations))
    paths[0, :] = initial_investment

    for t in range(1, months + 1):
        random_returns = np.random.normal(monthly_return - monthly_fee, monthly_vol, n_simulations)
        paths[t, :] = paths[t - 1, :] * (1 + random_returns) + monthly_contribution

    # Inflation-adjusted paths
    inflation_factors = np.array([(1 + monthly_inflation) ** t for t in range(months + 1)])
    real_paths = paths / inflation_factors[:, np.newaxis]

    final_values = paths[-1, :]
    real_final_values = real_paths[-1, :]

    # Total contributed capital
    total_contributed = initial_investment + monthly_contribution * months

    summary = {
        "median_final": float(np.median(final_values)),
        "mean_final": float(np.mean(final_values)),
        "optimistic_final": float(np.percentile(final_values, 90)),
        "pessimistic_final": float(np.percentile(final_values, 10)),
        "real_median_final": float(np.median(real_final_values)),
        "total_contributed": total_contributed,
        "median_gain": float(np.median(final_values)) - total_contributed,
        "probability_profit": float(np.mean(final_values > total_contributed)),
        "probability_double": float(np.mean(final_values > 2 * initial_investment)),
    }

    # Annual table (median path)
    median_path_idx = np.argsort(final_values)[n_simulations // 2]
    median_path = paths[:, median_path_idx]

    annual_rows = []
    for yr in range(years + 1):
        month_idx = yr * 12
        contributed = initial_investment + monthly_contribution * month_idx
        value = median_path[month_idx]
        gain = value - contributed
        real_value = value / ((1 + monthly_inflation) ** month_idx)
        annual_rows.append({
            "Year": yr,
            "Portfolio Value": round(value, 2),
            "Total Contributed": round(contributed, 2),
            "Investment Gain": round(gain, 2),
            "Real Value (Inflation-Adj.)": round(real_value, 2),
            "Return %": round((value / contributed - 1) * 100, 2) if contributed > 0 else 0.0,
        })

    annual_table = pd.DataFrame(annual_rows)

    # Create paths DataFrame (sample for performance)
    sample_n = min(n_simulations, 200)
    sample_indices = np.random.choice(n_simulations, sample_n, replace=False)
    date_index = pd.date_range(start="today", periods=months + 1, freq="ME")
    paths_df = pd.DataFrame(
        paths[:, sample_indices],
        index=date_index,
        columns=[f"Sim_{i}" for i in range(sample_n)]
    )

    return {
        "paths": paths_df,
        "summary": summary,
        "annual_table": annual_table,
        "all_final_values": final_values,
    }


def compound_growth_projection(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_return: float,
    annual_fee: float = 0.001,
    inflation_rate: float = 0.025
) -> pd.DataFrame:
    """
    Simple deterministic compound growth projection (no randomness).
    """
    months = years * 12
    monthly_return = (1 + annual_return - annual_fee) ** (1 / 12) - 1
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1

    rows = []
    balance = initial_investment
    for t in range(months + 1):
        contributed = initial_investment + monthly_contribution * t
        real_balance = balance / ((1 + monthly_inflation) ** t)
        rows.append({
            "Month": t,
            "Year": t / 12,
            "Balance": round(balance, 2),
            "Contributed": round(contributed, 2),
            "Gain": round(balance - contributed, 2),
            "Real Balance": round(real_balance, 2),
        })
        if t < months:
            balance = balance * (1 + monthly_return) + monthly_contribution

    return pd.DataFrame(rows)


def scenario_comparison(
    initial_investment: float,
    monthly_contribution: float,
    years: int,
    annual_fee: float = 0.001
) -> pd.DataFrame:
    """Compare investment outcomes across different market scenarios."""
    rows = []
    for scenario_name, params in MARKET_SCENARIOS.items():
        result = simulate_investment(
            initial_investment=initial_investment,
            monthly_contribution=monthly_contribution,
            years=years,
            annual_return=params["return"],
            annual_volatility=params["volatility"],
            annual_fee=annual_fee,
            n_simulations=500,
            seed=42
        )
        summary = result["summary"]
        rows.append({
            "Scenario": scenario_name,
            "Expected Return": f"{params['return']:.1%}",
            "Volatility": f"{params['volatility']:.1%}",
            "Median Final Value": f"${summary['median_final']:,.0f}",
            "Optimistic (90th)": f"${summary['optimistic_final']:,.0f}",
            "Pessimistic (10th)": f"${summary['pessimistic_final']:,.0f}",
            "Prob. of Profit": f"{summary['probability_profit']:.1%}",
        })
    return pd.DataFrame(rows)
