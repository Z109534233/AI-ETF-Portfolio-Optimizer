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
from zoneinfo import ZoneInfo

import streamlit as st

from src.financial_metrics import portfolio_diagnosis
from src.risk_analytics import historical_var_cvar
from src.i18n import (
    t, get_language, t_opt_method, t_investment_objective, t_risk_level,
)
from src.openai_service import cached_generate, fingerprint, is_configured


DISCLAIMER = (
    "This content is for educational purposes only and does not constitute financial advice. "
    "Always consult a qualified financial adviser before making investment decisions."
)

# "Holding" means an economically meaningful current position in Advisor
# logic. Tiny optimizer residuals and explicit 0% allocations must not count
# as holdings in counts, ML eligibility, or news relevance.
ADVISOR_ACTIVE_WEIGHT_THRESHOLD = 0.005


def _active_weights(weights: dict) -> dict:
    active = {k: float(v) for k, v in (weights or {}).items() if float(v) >= ADVISOR_ACTIVE_WEIGHT_THRESHOLD}
    total = sum(active.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in active.items()}


def _format_as_of_taipei(utc_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return utc_iso


def assumption_source_label(source: str) -> str:
    """Translate simulator assumption-source storage values for display."""
    return {
        "Portfolio Historical Statistics": t("sim_assumption_source_portfolio"),
        "Conservative Assumptions": t("sim_assumption_source_conservative"),
        "Market Scenario": t("sim_market_scenario"),
        "Custom Assumptions": t("sim_assumption_source_custom"),
    }.get(source, source or "—")

_ADVISOR_SYSTEM_INSTRUCTIONS = (
    "You are a professional educational financial analyst. Only discuss numbers "
    "explicitly present in the structured context provided by the user; when a "
    "section is marked unavailable, say so plainly instead of guessing. Provide "
    "clear, structured portfolio analysis for educational purposes only, not "
    "personalized financial advice. Never invent a number, and never state a "
    "number that differs from what was given to you."
)


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
            "reason": t("ai_reason_no_price_history"),
        }
    return {"available": True, "concentration": concentration, "var_cvar": var_cvar}


def _simulator_context(portfolio: dict, portfolio_source: str,
                        sim_result=None, sim_params=None,
                        hist_result=None, hist_params=None) -> dict:
    future_unavailable = {
        "available": False,
        "reason": t("ai_reason_run_future_sim"),
    }
    historical_unavailable = {
        "available": False,
        "reason": t("ai_reason_run_historical_sim"),
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
        return {"available": False, "reason": t("ai_reason_no_ml_run")}
    if ml_result.get("error"):
        return {"available": False, "reason": ml_result["error"]}
    if ml_ticker not in tickers:
        return {
            "available": False,
            "reason": t("ai_reason_ml_not_holding", ticker=ml_ticker),
        }
    metrics = ml_result.get("metrics", {}) or {}
    # src.machine_learning._compute_metrics() keys this "Accuracy"
    # (capitalized) -- see pages/5_Machine_Learning.py's own display code.
    accuracy = metrics.get("Accuracy")
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
        return {"available": False, "reason": t("ai_reason_no_market_news")}
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
    _as_of = datetime.now(timezone.utc).isoformat()
    context = {
        "as_of": _as_of,
        "as_of_display": _format_as_of_taipei(_as_of),
        "portfolio_source": portfolio_source,
    }

    if not portfolio or not portfolio.get("weights"):
        context["portfolio"] = {"available": False, "reason": t("ai_reason_no_portfolio")}
        context["risk"] = {"available": False, "reason": t("ai_reason_no_portfolio")}
        context["simulator"] = {
            "future_projection": {"available": False, "reason": t("ai_reason_no_portfolio")},
            "historical_simulation": {"available": False, "reason": t("ai_reason_no_portfolio")},
        }
        context["ml"] = {"available": False, "reason": t("ai_reason_no_portfolio")}
        context["news"] = {"available": False, "reason": t("ai_reason_no_portfolio")}
        return context

    weights = _active_weights(portfolio["weights"])
    tickers = list(weights.keys())

    if not weights:
        context["portfolio"] = {"available": False, "reason": t("ai_reason_no_active_holdings")}
        context["risk"] = {"available": False, "reason": t("ai_reason_no_active_holdings")}
        context["simulator"] = {
            "future_projection": {"available": False, "reason": t("ai_reason_no_active_holdings")},
            "historical_simulation": {"available": False, "reason": t("ai_reason_no_active_holdings")},
        }
        context["ml"] = {"available": False, "reason": t("ai_reason_no_active_holdings")}
        context["news"] = {"available": False, "reason": t("ai_reason_no_active_holdings")}
        return context

    # Downstream "holding" logic intentionally uses only >=0.5% positions.
    # This removes 0% optimizer residuals from counts, ML eligibility and
    # news relevance while preserving the portfolio's relative allocation.
    active_portfolio = dict(portfolio)
    active_portfolio["weights"] = weights
    active_portfolio["tickers"] = tickers

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
        "risk_tolerance": portfolio.get("risk_tolerance"),
        "generated_at": portfolio.get("generated_at"),
    }
    context["risk"] = _risk_context(weights, portfolio_prices)
    context["simulator"] = _simulator_context(
        active_portfolio, portfolio_source, sim_result, sim_params, hist_result, hist_params
    )
    context["ml"] = _ml_context(active_portfolio, portfolio_source, tickers, ml_result, ml_ticker)
    context["news"] = _news_context(active_portfolio, tickers, news_items)
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


def _is_conservative_profile(risk_level: str) -> bool:
    """Accept either the canonical profile key or its localized zh-TW label."""
    normalized = str(risk_level or "").strip().lower()
    return normalized in {"conservative", "保守型", "保守"}


def conservative_profile_mismatch(context: dict, risk_level: str):
    """Return deterministic mismatch details for a Conservative profile.

    The reviewer-facing guardrails are: annualized volatility > 12%,
    largest holding > 50%, or effective holdings < 2.
    """
    p = (context or {}).get("portfolio", {})
    if not p.get("available") or not _is_conservative_profile(risk_level):
        return None

    weights = p.get("weights") or {}
    concentration = (context or {}).get("risk", {}).get("concentration", {}) or {}

    top_weight = concentration.get("largest_weight")
    if not isinstance(top_weight, (int, float)):
        top_weight = max(weights.values(), default=0.0)

    effective = concentration.get("effective_holdings")
    if not isinstance(effective, (int, float)):
        effective = float(len(weights))

    vol = p.get("volatility")
    if not (
        (isinstance(vol, (int, float)) and vol > 0.12)
        or top_weight > 0.50
        or effective < 2
    ):
        return None

    return {
        "volatility": vol,
        "largest_weight": top_weight,
        "effective_holdings": effective,
    }


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
            f"CURRENT PORTFOLIO (source: {context['portfolio_source']}, strategy: {t_opt_method(p['strategy'])}, "
            f"market: {p['market']}):\n{weights_str}\n"
            f"In-sample historical annualized return: {_fmt_pct(p['expected_return'])} "
            f"(optimizer/backtest statistic; not a forward-looking expected return)\n"
            f"In-sample historical annualized volatility: {_fmt_pct(p['volatility'])}\n"
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
            f"assumptions: {assumption_source_label(fp['assumption_source'])}; "
            f"annual-return assumption {_fmt_pct(fp.get('annual_return'))}; "
            f"annual-volatility assumption {_fmt_pct(fp.get('annual_volatility'))}): "
            f"median value after {fp['years']} years {_fmt_money(s.get('median_final'))}; "
            f"share of simulated paths ending above cumulative contributions "
            f"{_fmt_pct(s.get('probability_profit'))}. This is a simulated-path share under the stated assumptions, "
            f"not a real-world probability of profit or a guarantee."
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

    # Deterministic consistency check is included in the AI prompt too, so
    # the OpenAI-assisted path cannot omit a mismatch that the rule-based
    # path would flag.
    mismatch = conservative_profile_mismatch(context, risk_level)
    if mismatch:
        lines.append(
            "\nRISK-PROFILE CONSISTENCY CHECK: MISMATCH. The user selected Conservative, "
            f"while annualized volatility is {_fmt_pct(mismatch['volatility'])}, the largest holding is "
            f"{_fmt_pct(mismatch['largest_weight'])}, and effective holdings are "
            f"{_fmt_num(mismatch['effective_holdings'], 1)}. "
            "State explicitly that this allocation is inconsistent with the selected Conservative "
            "profile under the app's rule (volatility >12%, largest holding >50%, or effective holdings <2)."
        )

    language_instruction = (
        "Respond entirely in Traditional Chinese (zh-TW/繁體中文), including all section "
        "headings and body text. Translate app labels into Traditional Chinese; do not leave "
        "English strategy names, assumption-source labels, or unavailable-reason boilerplate in the final report."
        if get_language() == "zh-TW"
        else "Respond entirely in English."
    )
    objective_display = t_investment_objective(investment_objective)
    risk_display = t_risk_level(risk_level)

    return f"""You are an educational financial analyst assistant. Using ONLY the structured
data below (never invent a number that is not printed here -- if a section says "not
available", state plainly that it is not available instead of guessing), provide a
clear, structured educational explanation.

{language_instruction}

Investor profile: {objective_display} objective, {risk_display} risk tolerance, {investment_horizon}-year horizon.

{chr(10).join(lines)}

Please provide a structured analysis including:
1. Portfolio Summary
2. Risk & Concentration
3. Simulator Outlook (future projection and/or historical simulation, if available)
4. Machine Learning Signal (only if available, framed as experimental/probabilistic)
5. Market Intelligence / News Relevance (only if available)
6. Main Risks and Tradeoffs
7. Educational Review Points

Keep the tone professional and educational. Do not provide personalised financial advice.
For every section marked "not available" above, say so explicitly rather than fabricating content.
In Section 1, label portfolio return/volatility as in-sample historical statistics, not future expectations.
If Section 3 has a future projection, explicitly distinguish its assumption source and annual-return assumption
from the in-sample historical statistic shown in Section 1.
End with a clear disclaimer that this is for educational purposes only."""


def advisor_fingerprint(context: dict, investment_objective: str, risk_level: str,
                         investment_horizon: int) -> str:
    """Deterministic fingerprint of every input that can change the
    narrative's content, for src.openai_service.cached_generate() -- so a
    Streamlit rerun triggered by an unrelated widget never re-spends an
    OpenAI call, but any change to the portfolio, its computed metrics, the
    investor-profile inputs, or any of the other module outputs actually
    quoted in _build_prompt() invalidates the cached result.

    Fingerprints the fully-rendered prompt text itself rather than a
    hand-maintained list of context fields: _build_prompt() is the single
    source of truth for what the LLM sees, so hashing its exact output can
    never drift out of sync with it the way a manually-curated field list
    previously did (a field quoted in the prompt but missing from the list
    could leave a stale cached narrative on screen, e.g. still saying
    "Investment Simulator: not available" next to a freshly-rendered
    "Deterministic Data" section that already shows real numbers).
    """
    prompt = _build_prompt(context, investment_objective, risk_level, investment_horizon)
    return fingerprint(prompt)


def generate_advisor_narrative(context: dict, investment_objective: str = "Long-term Growth",
                                risk_level: str = "Moderate", investment_horizon: int = 10,
                                session_state=None) -> dict:
    """Generate the synthesis narrative via the central OpenAI Responses API
    service (src.openai_service) when configured, else a rule-based
    narrative -- both grounded exclusively in `context` (see _build_prompt:
    the LLM is only ever asked to explain numbers already computed here, it
    never recomputes or invents one).

    Returns {"text": str, "source": "ai" | "rule_based"} -- callers must use
    "source" (not "was a client configured") to decide the AI-Generated vs.
    Rule-Based badge, since a configured client whose call still fails must
    fall back to "rule_based" too.

    When `session_state` (any dict-like, e.g. st.session_state) is given,
    the OpenAI call is cached by advisor_fingerprint() so an unrelated
    Streamlit rerun reuses the prior result instead of re-spending a call.
    """
    prompt = _build_prompt(context, investment_objective, risk_level, investment_horizon)

    if session_state is not None:
        fp = advisor_fingerprint(context, investment_objective, risk_level, investment_horizon)
        result = cached_generate(
            session_state, "_ai_advisor_openai_cache", fp,
            _ADVISOR_SYSTEM_INSTRUCTIONS, prompt,
        )
    else:
        from src.openai_service import generate_text
        result = generate_text(_ADVISOR_SYSTEM_INSTRUCTIONS, prompt)

    if result["available"]:
        return {"text": result["text"], "source": "ai"}

    # Only warn on a genuine API failure -- not configured at all is the
    # expected default state and should silently use the rule-based path.
    # The error detail itself is an SDK exception message (type(e).__name__:
    # e, see src/openai_service.py) and is deliberately left untranslated
    # inside the localized sentence -- it is a diagnostic string, not UI
    # copy, and is never a secret (fingerprint()/generate_text() never
    # surface the API key).
    if is_configured() and result.get("error"):
        st.warning(t("ai_generation_failed_fallback", error=result["error"]))
    return {"text": generate_rule_based_narrative(
        context, investment_objective, risk_level, investment_horizon), "source": "rule_based"}


def generate_rule_based_narrative(context: dict, investment_objective: str = "Long-term Growth",
                                   risk_level: str = "Moderate", investment_horizon: int = 10) -> str:
    """Rule-based synthesis narrative -- same grounding rules as the AI
    path (see _build_prompt): every line either quotes a value already
    present in `context`, or states a reason why a section is unavailable.
    """
    p = context["portfolio"]
    if not p["available"]:
        return t("ai_no_portfolio_data")

    _objective_display = t_investment_objective(investment_objective)
    _risk_display = t_risk_level(risk_level)
    lines = [t("ai_report_title"), t(
        "ai_report_meta", objective=_objective_display, risk=_risk_display, horizon=investment_horizon
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
    concentration = context.get("risk", {}).get("concentration", {})
    _effective_holdings = concentration.get("effective_holdings", n_holdings)
    if top_weight > 0.50 or _effective_holdings < 2:
        focus = t("ai_report_focus_concentrated")
    else:
        focus = t("ai_report_focus_equity") if equity_weight > 0.5 else t("ai_report_focus_diversified")

    lines.append(t("ai_report_section1"))
    lines.append(t(
        "ai_report_summary_text",
        n_holdings=n_holdings, focus=focus, top_holding=top_holding,
        top_weight=_fmt_pct(top_weight), horizon=investment_horizon, risk=_risk_display,
    ))
    lines.append(t(
        "ai_synthesis_portfolio_metrics",
        ret=_fmt_pct(p["expected_return"]), vol=_fmt_pct(p["volatility"]),
        sharpe=_fmt_num(p["sharpe_ratio"]), strategy=t_opt_method(p["strategy"]),
    ))
    _summary_fp = context.get("simulator", {}).get("future_projection", {})
    if _summary_fp.get("available") and isinstance(_summary_fp.get("annual_return"), (int, float)):
        lines.append(t(
            "ai_report_projection_bridge",
            source=assumption_source_label(_summary_fp.get("assumption_source", "")),
            ret=_fmt_pct(_summary_fp.get("annual_return")),
        ))

    lines.append("\n" + t("ai_report_section2_risk"))
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

    lines.append("\n" + t("ai_report_section3_simulator"))
    sim = context["simulator"]
    fp = sim["future_projection"]
    if fp.get("available"):
        s = fp["summary"]
        lines.append(t(
            "ai_sim_future_line", years=fp["years"], median=_fmt_money(s.get("median_final")),
            prob=_fmt_pct(s.get("probability_profit")),
            source=assumption_source_label(fp.get("assumption_source", "")),
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

    lines.append("\n" + t("ai_report_section4_ml"))
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

    lines.append("\n" + t("ai_report_section5_news"))
    news = context["news"]
    if news.get("available"):
        lines.append(t(
            "ai_news_line", count=news["headline_count"], relevant=news["relevant_holdings_count"],
        ))
        if news.get("portfolio_impact_text"):
            lines.append(f"- {news['portfolio_impact_text']}")
    else:
        lines.append(t("ai_section_unavailable", reason=news.get("reason", "")))

    lines.append("\n" + t("ai_report_section6_risks"))
    if top_weight > 0.5:
        lines.append("- " + t("ai_report_risk_concentration", ticker=top_holding, weight=_fmt_pct(top_weight)))
    _profile_mismatch = conservative_profile_mismatch(context, risk_level)
    if _profile_mismatch:
        lines.append("- " + t(
            "ai_report_risk_profile_mismatch",
            volatility=_fmt_pct(_profile_mismatch["volatility"]),
            largest_weight=_fmt_pct(_profile_mismatch["largest_weight"]),
            effective_holdings=_fmt_num(_profile_mismatch["effective_holdings"], 1),
        ))
    if equity_weight > 0.9:
        lines.append("- " + t("ai_report_risk_equity_market"))
    lines.append("- " + t("ai_report_risk_market"))

    lines.append("\n" + t("ai_report_section7_education"))
    lines.append("- " + t("ai_report_edu_correlation"))
    lines.append("- " + t("ai_report_edu_simulator"))
    lines.append("- " + t("ai_report_edu_tax"))
    lines.append("- " + t("ai_report_edu_review_objectives"))

    lines.append(f"\n---\n*{t('disclaimer_full')}*")

    return "\n".join(lines)
