"""
AI Advisor Module
Generates educational portfolio explanations using OpenAI API or rule-based fallback.
"""

import streamlit as st

from src.etf_database import get_etf
from src.i18n import t, get_language


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

    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文), including all section "
        "headings and body text."
        if get_language() == "zh-TW"
        else "Respond entirely in English."
    )

    prompt = f"""You are an educational financial analyst assistant. Analyse the following ETF portfolio and provide a clear, structured educational explanation.

{language_instruction}

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
        return t("ai_no_portfolio_data")

    tickers = list(portfolio_weights.keys())
    weights = list(portfolio_weights.values())
    top_holding = tickers[weights.index(max(weights))]
    top_weight = max(weights)
    n_holdings = len(tickers)

    # Classify portfolio type using src/etf_database.py's category field --
    # covers every registered ETF (US/Taiwan/UK and any future market)
    # instead of a hardcoded per-ticker set. Tickers not in the database
    # (e.g. a free-text custom ticker) are excluded from every bucket, same
    # as the previous hardcoded-set behavior.
    def _category(ticker: str):
        record = get_etf(ticker)
        return record.category if record else None

    equity_weight = sum(w for tk, w in portfolio_weights.items() if _category(tk) == "Equity")
    bond_weight = sum(w for tk, w in portfolio_weights.items() if _category(tk) == "Fixed Income")
    commodity_weight = sum(w for tk, w in portfolio_weights.items() if _category(tk) == "Commodity")

    # Extract metrics
    exp_return = metrics.get("Expected Annual Return", metrics.get("Annualized Return", "N/A"))
    exp_vol = metrics.get("Expected Annual Volatility", metrics.get("Annualized Volatility", "N/A"))
    sharpe = metrics.get("Sharpe Ratio", "N/A")
    max_dd = metrics.get("Maximum Drawdown", "N/A")

    focus = t("ai_report_focus_equity") if equity_weight > 0.5 else t("ai_report_focus_diversified")

    lines = []
    lines.append(t("ai_report_title"))
    lines.append(t("ai_report_meta", objective=investment_objective, risk=risk_level, horizon=investment_horizon) + "\n")

    lines.append(t("ai_report_section1"))
    lines.append(t(
        "ai_report_summary_text",
        n_holdings=n_holdings, focus=focus, top_holding=top_holding,
        top_weight=f"{top_weight:.1%}", horizon=investment_horizon, risk_lower=risk_level.lower()
    ))

    lines.append("\n" + t("ai_report_section2"))
    for ticker, weight in sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True):
        category = _get_etf_category(ticker)
        lines.append(f"- **{ticker}** ({weight:.1%}): {category}")

    lines.append("\n" + t("ai_report_section3"))
    if equity_weight > 0.6:
        lines.append("- " + t("ai_report_strength_equity"))
    if n_holdings >= 5:
        lines.append("- " + t("ai_report_strength_diversification", n=n_holdings))
    if bond_weight > 0.1:
        lines.append("- " + t("ai_report_strength_bonds"))
    if commodity_weight > 0.05:
        lines.append("- " + t("ai_report_strength_commodity"))
    if not lines[-1].startswith("-"):
        lines.append("- " + t("ai_report_strength_default"))

    lines.append("\n" + t("ai_report_section4"))
    if top_weight > 0.5:
        lines.append("- " + t("ai_report_risk_concentration", ticker=top_holding, weight=f"{top_weight:.1%}"))
    if equity_weight > 0.9:
        lines.append("- " + t("ai_report_risk_equity_market"))
    if bond_weight < 0.1 and investment_horizon < 5:
        lines.append("- " + t("ai_report_risk_duration"))
    lines.append("- " + t("ai_report_risk_market"))

    lines.append("\n" + t("ai_report_section5"))
    if n_holdings >= 5 and equity_weight < 0.9:
        lines.append(t("ai_report_diversification_good"))
    elif n_holdings < 3:
        lines.append(t("ai_report_diversification_limited"))
    else:
        lines.append(t("ai_report_diversification_moderate"))

    lines.append("\n" + t("ai_report_section6"))
    concentrated = [(tk, w) for tk, w in portfolio_weights.items() if w > 0.4]
    if concentrated:
        for tk, w in concentrated:
            lines.append("- " + t("ai_report_concentration_line", ticker=tk, weight=f"{w:.1%}"))
    else:
        lines.append("- " + t("ai_report_concentration_none"))

    lines.append("\n" + t("ai_report_section7"))
    lines.append("- " + t("ai_report_longterm_discipline", horizon=investment_horizon))
    lines.append("- " + t("ai_report_longterm_rebalancing"))
    lines.append("- " + t("ai_report_longterm_expense_ratio"))
    if investment_horizon >= 10:
        lines.append("- " + t("ai_report_longterm_stay_invested"))

    lines.append("\n" + t("ai_report_section8"))
    lines.append("- " + t("ai_report_edu_correlation"))
    lines.append("- " + t("ai_report_edu_simulator"))
    lines.append("- " + t("ai_report_edu_tax"))
    lines.append("- " + t("ai_report_edu_review_objectives"))

    lines.append(f"\n---\n*{t('disclaimer_full')}*")

    return "\n".join(lines)


def _get_etf_category(ticker: str) -> str:
    """Return a brief description of a well-known ETF, in the active session language."""
    key = f"etf_desc_{ticker}"
    from src.i18n import TRANSLATIONS
    lang = get_language()
    text = TRANSLATIONS.get(lang, {}).get(key) or TRANSLATIONS.get("en", {}).get(key)
    if text:
        return text
    return t("etf_desc_fallback", ticker=ticker)
