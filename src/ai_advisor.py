"""
AI Advisor Module
Generates educational portfolio explanations using OpenAI API or rule-based fallback.
"""

import streamlit as st


DISCLAIMER = (
    "This content is for educational purposes only and does not constitute financial advice. "
    "Always consult a qualified financial adviser before making investment decisions."
)


def get_openai_client():
    """Return an OpenAI client if API key is available, else None."""
    try:
        import openai
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def generate_ai_analysis(
    portfolio_weights: dict,
    metrics: dict,
    investment_objective: str = "Long-term Growth",
    risk_level: str = "Moderate",
    investment_horizon: int = 10
) -> str:
    """
    Generate AI portfolio analysis using OpenAI GPT.
    Falls back to rule-based analysis if API key is unavailable.
    """
    client = get_openai_client()

    if client is None:
        return generate_rule_based_analysis(
            portfolio_weights, metrics, investment_objective, risk_level, investment_horizon
        )

    # Build prompt
    weights_str = "\n".join([f"  - {ticker}: {w:.1%}" for ticker, w in portfolio_weights.items()])
    metrics_str = "\n".join([f"  - {k}: {v}" for k, v in metrics.items() if k != "Confusion Matrix"])

    prompt = f"""You are an educational financial analyst assistant. Analyse the following ETF portfolio and provide a clear, structured educational explanation.

Portfolio Allocation:
{weights_str}

Portfolio Metrics:
{metrics_str}

Investment Objective: {investment_objective}
Risk Level: {risk_level}
Investment Horizon: {investment_horizon} years

Please provide a structured analysis including:
1. Portfolio Summary (2-3 sentences)
2. Allocation Explanation (what each major holding represents)
3. Main Strengths (2-3 points)
4. Main Risks (2-3 points)
5. Diversification Observations
6. Concentration Warnings (if any)
7. Long-term Considerations
8. Educational Suggestions

Keep the tone professional and educational. Do not provide personalised financial advice.
End with a clear disclaimer that this is for educational purposes only."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional educational financial analyst. Provide clear, structured portfolio analysis for educational purposes only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"AI analysis unavailable: {e}. Using rule-based analysis.")
        return generate_rule_based_analysis(
            portfolio_weights, metrics, investment_objective, risk_level, investment_horizon
        )


def generate_rule_based_analysis(
    portfolio_weights: dict,
    metrics: dict,
    investment_objective: str = "Long-term Growth",
    risk_level: str = "Moderate",
    investment_horizon: int = 10
) -> str:
    """
    Generate a rule-based portfolio analysis without AI.
    """
    if not portfolio_weights:
        return "No portfolio data available for analysis."

    tickers = list(portfolio_weights.keys())
    weights = list(portfolio_weights.values())
    top_holding = tickers[weights.index(max(weights))]
    top_weight = max(weights)
    n_holdings = len(tickers)

    # Classify portfolio type
    equity_etfs = {"VOO", "QQQ", "SPY", "VTI", "VT", "VXUS", "SCHD", "IWM", "XLK", "XLF", "XLV", "VNQ"}
    bond_etfs = {"BND", "TLT"}
    commodity_etfs = {"GLD"}

    equity_weight = sum(w for t, w in portfolio_weights.items() if t in equity_etfs)
    bond_weight = sum(w for t, w in portfolio_weights.items() if t in bond_etfs)
    commodity_weight = sum(w for t, w in portfolio_weights.items() if t in commodity_etfs)

    # Extract metrics
    exp_return = metrics.get("Expected Annual Return", metrics.get("Annualized Return", "N/A"))
    exp_vol = metrics.get("Expected Annual Volatility", metrics.get("Annualized Volatility", "N/A"))
    sharpe = metrics.get("Sharpe Ratio", "N/A")
    max_dd = metrics.get("Maximum Drawdown", "N/A")

    lines = []
    lines.append("## Portfolio Analysis Report")
    lines.append(f"*Generated for educational purposes | {investment_objective} | {risk_level} Risk | {investment_horizon}-Year Horizon*\n")

    lines.append("### 1. Portfolio Summary")
    lines.append(
        f"This portfolio consists of {n_holdings} ETF(s) with a primary focus on "
        f"{'equity' if equity_weight > 0.5 else 'diversified'} assets. "
        f"The largest holding is **{top_holding}** at **{top_weight:.1%}** of the portfolio. "
        f"The portfolio is designed for a {investment_horizon}-year investment horizon with a {risk_level.lower()} risk profile."
    )

    lines.append("\n### 2. Allocation Explanation")
    for ticker, weight in sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True):
        category = _get_etf_category(ticker)
        lines.append(f"- **{ticker}** ({weight:.1%}): {category}")

    lines.append("\n### 3. Main Strengths")
    if equity_weight > 0.6:
        lines.append("- Strong equity exposure provides long-term growth potential aligned with economic expansion.")
    if n_holdings >= 5:
        lines.append(f"- Diversification across {n_holdings} ETFs reduces single-security concentration risk.")
    if bond_weight > 0.1:
        lines.append("- Fixed income allocation provides portfolio stability and income generation.")
    if commodity_weight > 0.05:
        lines.append("- Commodity exposure (e.g., gold) offers inflation hedging and portfolio diversification.")
    if not lines[-1].startswith("-"):
        lines.append("- Broad market exposure through index ETFs provides cost-efficient diversification.")

    lines.append("\n### 4. Main Risks")
    if top_weight > 0.5:
        lines.append(f"- **Concentration Risk**: {top_holding} represents {top_weight:.1%} of the portfolio, creating significant single-ETF dependency.")
    if equity_weight > 0.9:
        lines.append("- **Equity Market Risk**: High equity allocation means the portfolio is sensitive to broad market downturns.")
    if bond_weight < 0.1 and investment_horizon < 5:
        lines.append("- **Duration Risk**: Limited fixed income exposure may increase short-term volatility for shorter horizons.")
    lines.append("- **Market Risk**: All investments carry inherent market risk and past performance does not guarantee future results.")

    lines.append("\n### 5. Diversification Observations")
    if n_holdings >= 5 and equity_weight < 0.9:
        lines.append("The portfolio demonstrates reasonable diversification across asset classes.")
    elif n_holdings < 3:
        lines.append("The portfolio has limited diversification. Consider adding more ETFs across different asset classes.")
    else:
        lines.append("The portfolio shows moderate diversification. Adding international or fixed income exposure could improve risk-adjusted returns.")

    lines.append("\n### 6. Concentration Warnings")
    concentrated = [(t, w) for t, w in portfolio_weights.items() if w > 0.4]
    if concentrated:
        for t, w in concentrated:
            lines.append(f"- **{t}** represents {w:.1%} of the portfolio. A single ETF above 40% creates meaningful concentration risk.")
    else:
        lines.append("- No single ETF exceeds 40% allocation. Concentration risk appears well-managed.")

    lines.append("\n### 7. Long-term Considerations")
    lines.append(f"- For a {investment_horizon}-year horizon, maintaining discipline through market cycles is essential.")
    lines.append("- Regular rebalancing (annually or semi-annually) helps maintain target allocations.")
    lines.append("- Consider the impact of expense ratios on long-term compounding returns.")
    if investment_horizon >= 10:
        lines.append("- Long-term investors historically benefit from staying invested through short-term volatility.")

    lines.append("\n### 8. Educational Suggestions")
    lines.append("- Review the correlation matrix to understand how holdings interact during market stress.")
    lines.append("- Use the Investment Simulator to model different contribution and return scenarios.")
    lines.append("- Consider your personal tax situation and account type when evaluating ETF selections.")
    lines.append("- Regularly review your investment objectives and risk tolerance as circumstances change.")

    lines.append(f"\n---\n*{DISCLAIMER}*")

    return "\n".join(lines)


def _get_etf_category(ticker: str) -> str:
    """Return a brief description of a well-known ETF."""
    descriptions = {
        "VOO": "Vanguard S&P 500 ETF — tracks the 500 largest US companies",
        "QQQ": "Invesco QQQ — tracks the Nasdaq-100 technology-heavy index",
        "SPY": "SPDR S&P 500 ETF — one of the oldest and most liquid US equity ETFs",
        "VTI": "Vanguard Total Stock Market ETF — broad US equity exposure",
        "VT": "Vanguard Total World Stock ETF — global equity diversification",
        "VXUS": "Vanguard Total International Stock ETF — non-US equity exposure",
        "SCHD": "Schwab US Dividend Equity ETF — dividend-focused US equities",
        "BND": "Vanguard Total Bond Market ETF — broad US fixed income exposure",
        "TLT": "iShares 20+ Year Treasury Bond ETF — long-duration US government bonds",
        "GLD": "SPDR Gold Shares — gold commodity exposure for inflation hedging",
        "IWM": "iShares Russell 2000 ETF — US small-cap equity exposure",
        "XLK": "Technology Select Sector SPDR — US technology sector",
        "XLF": "Financial Select Sector SPDR — US financial sector",
        "XLV": "Health Care Select Sector SPDR — US healthcare sector",
        "VNQ": "Vanguard Real Estate ETF — US real estate investment trusts",
    }
    return descriptions.get(ticker, f"{ticker} — ETF holding")
