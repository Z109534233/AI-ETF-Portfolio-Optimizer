"""
AI Advisor Module (M6 -- issue #18 Stage 6)

The AI Advisor is the SYNTHESIS layer of the app: it does not compute its
own portfolio/risk/simulation/ML numbers using a different method. Instead
build_advisor_context() assembles one structured, source-labeled context
dict entirely from the app's OWN already-computed outputs --

  - portfolio   -> the canonical st.session_state["current_portfolio"]
                   built by Portfolio Optimizer
  - risk        -> src.financial_metrics.portfolio_diagnosis() (the SAME
                   function Portfolio Optimizer/Risk Analytics use) plus
                   src.risk_analytics.historical_var_cvar() when a price
                   series is available
  - simulator   -> a prior Investment Simulator run in THIS session, only
                   surfaced when its stored strategy/market matches the
                   current portfolio (never a stale or contradictory run)
  - ml          -> a prior Machine Learning run in THIS session, only
                   surfaced when it succeeded (no error) and its ticker is
                   an actual current holding
  - news        -> src.market_intelligence.get_affected_etfs() /
                   analyze_portfolio_impact() applied to the current
                   holdings

Every section that isn't genuinely available is marked
{"available": False, "reason": "..."} instead of being silently omitted or
filled with an invented number. generate_advisor_narrative() (AI or
rule-based) is grounded ONLY in this context -- it never introduces a
metric that isn't present in it.
"""

from datetime import datetime, timezone

import streamlit as st

from src.financial_metrics import portfolio_diagnosis
from src.risk_analytics import historical_var_cvar
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


# ============================================================================
# Context assembly -- pure functions, no network/session access, so they are
# fully deterministic and unit-testable with canned inputs.
# ============================================================================

def _risk_context(weights: dict, portfolio_prices=None) -> dict:
    concentration = portfolio_diagnosis(weights)
    if portfolio_prices is not None and not portfolio_prices.empty:
        var_cvar = historical_var_cvar(portfolio_prices)
    else:
        var_cvar = {
            "available": False,
            "reason": "no price history loaded for this portfolio in this session",
        }
    return {"available": True, "concentration": concentration, "var_cvar": var_cvar}


def _simulator_context(portfolio: dict, portfolio_source: str,
                        sim_result=None, sim_params=None,
                        hist_result=None, hist_params=None) -> dict:
    future_unavailable = {
        "available": False,
        "reason": "run Investment Simulator (Future Projection) with the current portfolio to include this",
    }
    historical_unavailable = {
        "available": False,
        "reason": "run Investment Simulator (Historical Simulation) with the current portfolio to include this",
    }

    if portfolio_source != "current":
        return {"future_projection": future_unavailable, "historical_simulation": historical_unavailable}

    future = future_unavailable
    if sim_result and sim_params and sim_params.get("portfolio_strategy") == portfolio.get("strategy"):
        future = {
            "available": True,
            "assumption_source": sim_params.get("assumption_source"),
            "years": sim_params.get("years"),
            "annual_return": sim_params.get("annual_return"),
            "annual_volatility": sim_params.get("annual_volatility"),
            "n_simulations": sim_params.get("n_simulations"),
            "summary": sim_result.get("summary", {}),
        }

    historical = historical_unavailable
    if (hist_result and hist_params
            and hist_params.get("strategy") == portfolio.get("strategy")
            and hist_params.get("market") == portfolio.get("market")):
        historical = {
            "available": True,
            "summary": hist_result.get("summary", {}),
            "redistributed_from_missing": hist_params.get("redistributed_from_missing", False),
        }

    return {"future_projection": future, "historical_simulation": historical}


def _ml_context(portfolio: dict, portfolio_source: str, tickers: list,
                 ml_result=None, ml_ticker=None) -> dict:
    if portfolio_source != "current" or not ml_result:
        return {"available": False, "reason": "no Machine Learning run available in this session"}
    if ml_result.get("error"):
        return {"available": False, "reason": ml_result["error"]}
    if ml_ticker not in tickers:
        return {
            "available": False,
            "reason": f"the last Machine Learning run ({ml_ticker}) is not a current holding",
        }
    metrics = ml_result.get("metrics", {}) or {}
    accuracy = metrics.get("accuracy")
    baseline = ml_result.get("baseline_accuracy")
    return {
        "available": True,
        "ticker": ml_ticker,
        "model_name": ml_result.get("model_name"),
        "accuracy": accuracy,
        "baseline_accuracy": baseline,
        "beats_baseline": bool(accuracy is not None and baseline is not None and accuracy > baseline),
        "test_start": ml_result.get("test_start"),
        "test_end": ml_result.get("test_end"),
        "lookahead_periods": ml_result.get("lookahead_periods"),
    }


def _news_context(portfolio: dict, tickers: list, news_items=None) -> dict:
    if not news_items:
        return {"available": False, "reason": "no market news available in this session"}
    from src.market_intelligence import get_affected_etfs, analyze_portfolio_impact

    affected = get_affected_etfs(news_items, tickers=tickers) if tickers else []
    impact_text = (
        analyze_portfolio_impact(portfolio["weights"], affected)
        if portfolio and portfolio.get("weights") else None
    )
    relevant = [e for e in affected if e.get("impact") != "Neutral"]
    return {
        "available": True,
        "headline_count": len(news_items),
        "affected_holdings": affected,
        "relevant_holdings_count": len(relevant),
        "portfolio_impact_text": impact_text,
    }


def build_advisor_context(
    portfolio: dict = None,
    portfolio_source: str = "current",
    portfolio_prices=None,
    sim_result: dict = None, sim_params: dict = None,
    hist_result: dict = None, hist_params: dict = None,
    ml_result: dict = None, ml_ticker: str = None,
    news_items: list = None,
) -> dict:
    """Assemble the full cross-module context dict described in the module
    docstring. `portfolio_source` is "current" (the canonical
    st.session_state["current_portfolio"]) or "custom" (a hypothetical
    portfolio built directly on this page) -- simulator/ML integration only
    ever applies to "current" since those cached runs are keyed to the
    canonical portfolio's strategy/market, not to an arbitrary custom one.
    """
    context = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "portfolio_source": portfolio_source,
    }

    if not portfolio or not portfolio.get("weights"):
        context["portfolio"] = {"available": False, "reason": "no portfolio built yet"}
        context["risk"] = {"available": False, "reason": "no portfolio available"}
        context["simulator"] = {
            "future_projection": {"available": False, "reason": "no portfolio available"},
            "historical_simulation": {"available": False, "reason": "no portfolio available"},
        }
        context["ml"] = {"available": False, "reason": "no portfolio available"}
        context["news"] = {"available": False, "reason": "no portfolio available"}
        return context

    weights = portfolio["weights"]
    tickers = portfolio.get("tickers") or list(weights.keys())

    context["portfolio"] = {
        "available": True,
        "strategy": portfolio.get("strategy"),
        "market": portfolio.get("market"),
        "tickers": tickers,
        "weights": weights,
        "investment_amount": portfolio.get("investment_amount"),
        "expected_return": portfolio.get("expected_return"),
        "volatility": portfolio.get("volatility"),
        "sharpe_ratio": portfolio.get("sharpe_ratio"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "generated_at": portfolio.get("generated_at"),
    }
    context["risk"] = _risk_context(weights, portfolio_prices)
    context["simulator"] = _simulator_context(
        portfolio, portfolio_source, sim_result, sim_params, hist_result, hist_params
    )
    context["ml"] = _ml_context(portfolio, portfolio_source, tickers, ml_result, ml_ticker)
    context["news"] = _news_context(portfolio, tickers, news_items)
    return context


# ============================================================================
# Narrative generation -- AI (OpenAI, when configured) or rule-based,
# always grounded in the context dict above.
# ============================================================================

def _fmt_pct(x, digits=2):
    return f"{x:.{digits}%}" if isinstance(x, (int, float)) else "N/A"


def _fmt_num(x, digits=2):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "N/A"


def _fmt_money(x):
    return f"${x:,.0f}" if isinstance(x, (int, float)) else "N/A"


def _build_prompt(context: dict, investment_objective: str, risk_level: str,
                   investment_horizon: int) -> str:
    """Serialize the context dict into a plain-text brief for the LLM.
    Every section explicitly states "not available" with its reason when
    absent, and the instructions forbid introducing any number that is not
    printed here -- this is what the leakage/grounding tests check.
    """
    lines = []
    p = context["portfolio"]
    if p["available"]:
        weights_str = "\n".join(f"  - {tk}: {w:.1%}" for tk, w in sorted(
            p["weights"].items(), key=lambda kv: kv[1], reverse=True))
        lines.append(
            f"CURRENT PORTFOLIO (source: {context['portfolio_source']}, strategy: {p['strategy']}, "
            f"market: {p['market']}):\n{weights_str}\n"
            f"Expected annual return: {_fmt_pct(p['expected_return'])}\n"
            f"Expected annual volatility: {_fmt_pct(p['volatility'])}\n"
            f"Sharpe ratio: {_fmt_num(p['sharpe_ratio'])}\n"
            f"Maximum drawdown (backtest): {_fmt_pct(p['max_drawdown'])}\n"
            f"Investment amount: {_fmt_money(p['investment_amount'])}"
        )
    else:
        lines.append(f"CURRENT PORTFOLIO: not available ({p['reason']}).")

    r = context["risk"]
    if r.get("available"):
        c = r["concentration"]
        lines.append(
            f"\nRISK / CONCENTRATION: largest holding {c['largest_ticker']} at {_fmt_pct(c['largest_weight'])}; "
            f"effective holdings {_fmt_num(c['effective_holdings'], 1)}; HHI {_fmt_num(c['hhi'], 3)}; "
            f"concentration level: {c['concentration_level']}."
        )
        vc = r["var_cvar"]
        if vc.get("available"):
            lines.append(
                f"Historical VaR ({vc['confidence']:.0%} confidence, {vc['holding_period_days']}-day holding period, "
                f"{vc['n_observations']} observations from {vc['window_start']} to {vc['window_end']}): "
                f"{_fmt_pct(vc['var'])}; CVaR: {_fmt_pct(vc['cvar'])}."
            )
        else:
            lines.append(f"Historical VaR/CVaR: not available ({vc.get('reason')}).")
    else:
        lines.append(f"\nRISK / CONCENTRATION: not available ({r.get('reason')}).")

    sim = context["simulator"]
    fp = sim["future_projection"]
    if fp.get("available"):
        s = fp["summary"]
        lines.append(
            f"\nINVESTMENT SIMULATOR -- Future Projection (Monte Carlo, {fp['n_simulations']} paths, "
            f"assumptions: {fp['assumption_source']}): median value after {fp['years']} years "
            f"{_fmt_money(s.get('median_final'))}; probability of profit {_fmt_pct(s.get('probability_profit'))}. "
            f"This is a projection, not a guarantee."
        )
    else:
        lines.append(f"\nINVESTMENT SIMULATOR -- Future Projection: not available ({fp.get('reason')}).")

    hs = sim["historical_simulation"]
    if hs.get("available"):
        s = hs["summary"]
        lines.append(
            f"INVESTMENT SIMULATOR -- Historical Simulation: ending value {_fmt_money(s.get('final_value'))}, "
            f"total gain {_fmt_money(s.get('gain'))}, annualized money-weighted return "
            f"{_fmt_pct(s.get('annualized_mwr'))}."
        )
    else:
        lines.append(f"INVESTMENT SIMULATOR -- Historical Simulation: not available ({hs.get('reason')}).")

    ml = context["ml"]
    if ml.get("available"):
        beats = "beats" if ml["beats_baseline"] else "does not beat"
        lines.append(
            f"\nMACHINE LEARNING ({ml['ticker']}, {ml['model_name']}): out-of-sample accuracy "
            f"{_fmt_pct(ml['accuracy'])} vs. majority-class baseline {_fmt_pct(ml['baseline_accuracy'])} "
            f"({beats} baseline), test window {ml['test_start']} to {ml['test_end']}, "
            f"{ml['lookahead_periods']}-period-ahead target. This is an experimental, probabilistic "
            f"signal, not a trading recommendation."
        )
    else:
        lines.append(f"\nMACHINE LEARNING: not available ({ml.get('reason')}).")

    news = context["news"]
    if news.get("available"):
        lines.append(
            f"\nMARKET INTELLIGENCE: {news['headline_count']} recent headlines reviewed; "
            f"{news['relevant_holdings_count']} current holding(s) have news classified as relevant today. "
            f"{news.get('portfolio_impact_text') or ''}"
        )
    else:
        lines.append(f"\nMARKET INTELLIGENCE: not available ({news.get('reason')}).")

    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文), including all section "
        "headings and body text."
        if get_language() == "zh-TW"
        else "Respond entirely in English."
    )

    return f"""You are an educational financial analyst assistant. Using ONLY the structured
data below (never invent a number that is not printed here -- if a section says "not
available", state plainly that it is not available instead of guessing), provide a
clear, structured educational explanation.

{language_instruction}

Investor profile: {investment_objective} objective, {risk_level} risk tolerance, {investment_horizon}-year horizon.

{chr(10).join(lines)}

Please provide a structured analysis including:
1. Portfolio Summary
2. Risk & Concentration
3. Simulator Outlook (future projection and/or historical simulation, if available)
4. Machine Learning Signal (only if available, framed as experimental/probabilistic)
5. Market Intelligence / News Relevance (only if available)
6. Main Risks and Tradeoffs
7. Educational Suggestions

Keep the tone professional and educational. Do not provide personalised financial advice.
For every section marked "not available" above, say so explicitly rather than fabricating content.
End with a clear disclaimer that this is for educational purposes only."""


def generate_advisor_narrative(context: dict, investment_objective: str = "Long-term Growth",
                                risk_level: str = "Moderate", investment_horizon: int = 10) -> str:
    """Generate the synthesis narrative -- AI (OpenAI) when configured, else
    a rule-based narrative. Both are grounded exclusively in `context`.
    """
    client = get_openai_client()
    if client is None:
        return generate_rule_based_narrative(context, investment_objective, risk_level, investment_horizon)

    prompt = _build_prompt(context, investment_objective, risk_level, investment_horizon)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a professional educational financial analyst. Only discuss numbers "
                    "explicitly present in the structured context provided by the user; when a "
                    "section is marked unavailable, say so plainly instead of guessing. Provide "
                    "clear, structured portfolio analysis for educational purposes only, not "
                    "personalized financial advice."
                )},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1400,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"AI analysis unavailable: {e}. Using rule-based analysis.")
        return generate_rule_based_narrative(context, investment_objective, risk_level, investment_horizon)


def generate_rule_based_narrative(context: dict, investment_objective: str = "Long-term Growth",
                                   risk_level: str = "Moderate", investment_horizon: int = 10) -> str:
    """Rule-based synthesis narrative -- same grounding rules as the AI
    path (see _build_prompt): every line either quotes a value already
    present in `context`, or states a reason why a section is unavailable.
    """
    p = context["portfolio"]
    if not p["available"]:
        return t("ai_no_portfolio_data")

    lines = [t("ai_report_title"), t(
        "ai_report_meta", objective=investment_objective, risk=risk_level, horizon=investment_horizon
    ) + "\n"]

    weights = p["weights"]
    tickers = list(weights.keys())
    top_holding = max(weights, key=weights.get)
    top_weight = weights[top_holding]
    n_holdings = len(tickers)

    from src.etf_database import get_etf

    def _category(ticker):
        record = get_etf(ticker)
        return record.category if record else None

    equity_weight = sum(w for tk, w in weights.items() if _category(tk) == "Equity")
    focus = t("ai_report_focus_equity") if equity_weight > 0.5 else t("ai_report_focus_diversified")

    lines.append(t("ai_report_section1"))
    lines.append(t(
        "ai_report_summary_text",
        n_holdings=n_holdings, focus=focus, top_holding=top_holding,
        top_weight=_fmt_pct(top_weight), horizon=investment_horizon, risk_lower=risk_level.lower(),
    ))
    lines.append(t(
        "ai_synthesis_portfolio_metrics",
        ret=_fmt_pct(p["expected_return"]), vol=_fmt_pct(p["volatility"]),
        sharpe=_fmt_num(p["sharpe_ratio"]), strategy=p["strategy"],
    ))

    lines.append("\n" + t("ai_section_risk"))
    r = context["risk"]
    if r.get("available"):
        c = r["concentration"]
        lines.append(t(
            "ai_risk_concentration_line", ticker=c["largest_ticker"], weight=_fmt_pct(c["largest_weight"]),
            effective_holdings=_fmt_num(c["effective_holdings"], 1),
        ))
        vc = r["var_cvar"]
        if vc.get("available"):
            lines.append(t(
                "ai_risk_var_line", confidence=f"{vc['confidence']:.0%}",
                holding_period=vc["holding_period_days"], var=_fmt_pct(vc["var"]), cvar=_fmt_pct(vc["cvar"]),
                n_obs=vc["n_observations"], window_start=vc["window_start"], window_end=vc["window_end"],
            ))
        else:
            lines.append(t("ai_risk_var_unavailable", reason=vc.get("reason", "")))
    else:
        lines.append(t("ai_section_unavailable", reason=r.get("reason", "")))

    lines.append("\n" + t("ai_section_simulator"))
    sim = context["simulator"]
    fp = sim["future_projection"]
    if fp.get("available"):
        s = fp["summary"]
        lines.append(t(
            "ai_sim_future_line", years=fp["years"], median=_fmt_money(s.get("median_final")),
            prob=_fmt_pct(s.get("probability_profit")), source=fp.get("assumption_source", ""),
        ))
    else:
        lines.append(t("ai_sim_future_unavailable", reason=fp.get("reason", "")))
    hs = sim["historical_simulation"]
    if hs.get("available"):
        s = hs["summary"]
        lines.append(t(
            "ai_sim_historical_line", final=_fmt_money(s.get("final_value")),
            gain=_fmt_money(s.get("gain")), mwr=_fmt_pct(s.get("annualized_mwr")),
        ))
    else:
        lines.append(t("ai_sim_historical_unavailable", reason=hs.get("reason", "")))

    lines.append("\n" + t("ai_section_ml"))
    ml = context["ml"]
    if ml.get("available"):
        beats = t("ai_ml_beats") if ml["beats_baseline"] else t("ai_ml_below")
        lines.append(t(
            "ai_ml_line", ticker=ml["ticker"], model=ml["model_name"], accuracy=_fmt_pct(ml["accuracy"]),
            baseline=_fmt_pct(ml["baseline_accuracy"]), beats=beats,
            test_start=ml["test_start"], test_end=ml["test_end"],
        ))
    else:
        lines.append(t("ai_section_unavailable", reason=ml.get("reason", "")))

    lines.append("\n" + t("ai_section_news"))
    news = context["news"]
    if news.get("available"):
        lines.append(t(
            "ai_news_line", count=news["headline_count"], relevant=news["relevant_holdings_count"],
        ))
        if news.get("portfolio_impact_text"):
            lines.append(f"- {news['portfolio_impact_text']}")
    else:
        lines.append(t("ai_section_unavailable", reason=news.get("reason", "")))

    lines.append("\n" + t("ai_report_section4"))
    if top_weight > 0.5:
        lines.append("- " + t("ai_report_risk_concentration", ticker=top_holding, weight=_fmt_pct(top_weight)))
    if equity_weight > 0.9:
        lines.append("- " + t("ai_report_risk_equity_market"))
    lines.append("- " + t("ai_report_risk_market"))

    lines.append("\n" + t("ai_report_section8"))
    lines.append("- " + t("ai_report_edu_correlation"))
    lines.append("- " + t("ai_report_edu_simulator"))
    lines.append("- " + t("ai_report_edu_tax"))
    lines.append("- " + t("ai_report_edu_review_objectives"))

    lines.append(f"\n---\n*{t('disclaimer_full')}*")

    return "\n".join(lines)
