"""
Daily Portfolio Brief (Issue #22 section E) -- "Today's Portfolio Brief".

This is explicitly NOT a buy/sell engine. It assembles a small, deterministic,
already-computed context (which tickers the user holds/watches, which of
those tickers today's news might affect, and -- if the caller already has
one -- a portfolio-level risk/concentration diagnosis) and either asks the
central OpenAI Responses API service (src.openai_service) to summarize it in
plain language, or falls back to a rule-based templated summary when OpenAI
isn't configured or the call fails. Same pattern as src/ai_advisor.py:
  1. AI only ever explains numbers already computed elsewhere -- it never
     invents a price, probability, score, or ticker, and never gives
     buy/sell advice.
  2. The app keeps working with no OPENAI_API_KEY configured, or when the
     API call fails -- callers always get usable text back.
  3. This module does no live pricing/network/Streamlit work itself: the
     caller (the Streamlit page) already has today's holdings, watchlist,
     news, and (optionally) a risk diagnosis, and passes them in as plain
     data.

UI wording this module's output supports (headings live in the page layer,
not here): "Today's Portfolio Brief", "Today's Watchpoints",
"Risk/News Attention", "No major change detected".
"""

import hashlib
import json
from datetime import date

from src.market_intelligence import get_affected_etfs
from src import openai_service

_LANGUAGE_NAME = {
    "zh-TW": "Traditional Chinese (繁體中文)",
    "en": "English",
}


def _system_instructions(language: str) -> str:
    lang_name = _LANGUAGE_NAME.get(language, _LANGUAGE_NAME["en"])
    return (
        "You are summarizing an already-computed, structured portfolio "
        "monitoring context for a retail investor. ONLY reference the facts "
        "given below. Do NOT invent prices, probabilities, scores, tickers, or "
        "buy/sell advice. If nothing notable happened, say so plainly. "
        f"Respond entirely in {lang_name}."
    )


_NOTABLE_IMPACTS = ("Positive", "Negative")


def _unique_sorted(tickers) -> list:
    return sorted({t for t in tickers if t})


def build_brief_context(holdings: list, watchlist: list, news_items: list,
                         portfolio_risk: dict = None, as_of_date: str = None) -> dict:
    """Assemble a deterministic, structured context for the daily brief.

    Pure function: no Streamlit, no OpenAI, no network. Every field is
    traceable to one of the inputs -- nothing here fabricates a price,
    probability, or score.

    - holdings_tickers / watchlist_tickers: sorted unique ticker lists
      extracted from `holdings` / `watchlist` (each expected to be a list of
      dicts with a "ticker" key, matching src.database.load_user_holdings /
      load_watchlist).
    - news_matches: src.market_intelligence.get_affected_etfs() restricted to
      ONLY the union of the user's own holdings+watchlist tickers (never the
      module's generic default watchlist). Empty when there are no relevant
      tickers at all -- get_affected_etfs() is not even called in that case.
    - notable_count / has_notable_activity: how many of those matches are
      Positive or Negative (Neutral doesn't count as "notable") -- this is
      what decides whether "No major change detected" applies.
    - portfolio_risk / risk_available: whatever the caller already computed
      (e.g. via src.financial_metrics.portfolio_diagnosis()), passed through
      verbatim; this module never recomputes it.
    - as_of_date: ISO date string, overridable so tests are deterministic.
    - news_headline_count: total number of headlines considered, regardless
      of whether they matched a held/watched ticker.
    """
    holdings = holdings or []
    watchlist = watchlist or []
    news_items = news_items or []

    holdings_tickers = _unique_sorted(h.get("ticker") for h in holdings)
    watchlist_tickers = _unique_sorted(w.get("ticker") for w in watchlist)
    all_relevant_tickers = _unique_sorted(holdings_tickers + watchlist_tickers)

    news_matches = (
        get_affected_etfs(news_items, tickers=all_relevant_tickers)
        if all_relevant_tickers else []
    )
    notable_count = sum(1 for m in news_matches if m.get("impact") in _NOTABLE_IMPACTS)

    return {
        "as_of_date": as_of_date or date.today().isoformat(),
        "holdings_tickers": holdings_tickers,
        "watchlist_tickers": watchlist_tickers,
        "all_relevant_tickers": all_relevant_tickers,
        "news_matches": news_matches,
        "notable_count": notable_count,
        "has_notable_activity": notable_count > 0,
        "portfolio_risk": portfolio_risk,
        "risk_available": portfolio_risk is not None,
        "news_headline_count": len(news_items),
    }


def fingerprint_brief_context(context: dict, language: str = "en") -> str:
    """Deterministic fingerprint of the parts of `context` (plus the
    requested `language`) that can change the brief's content: as_of_date,
    holdings/watchlist tickers, the news match results, notable_count, and
    language. The same context+language on a Streamlit rerun produces the
    same fingerprint (so src.openai_service.cached_generate() reuses the
    cached result instead of re-spending an API call); a new holding, a
    newly-classified headline, a new day, OR switching the selected
    language all change it -- so switching zh-TW <-> en can never return a
    stale cached brief written in the other language (Issue #43 item S).
    """
    payload = {
        "as_of_date": context.get("as_of_date"),
        "holdings_tickers": sorted(context.get("holdings_tickers") or []),
        "watchlist_tickers": sorted(context.get("watchlist_tickers") or []),
        "news_matches": sorted(
            (
                (m.get("ticker"), m.get("impact"))
                for m in (context.get("news_matches") or [])
            ),
            key=lambda pair: (pair[0] or "", pair[1] or ""),
        ),
        "notable_count": context.get("notable_count"),
        "language": language,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rule_based_brief(context: dict, language: str = "en") -> str:
    """Deterministic, template-based (no-LLM) textual summary built ONLY
    from `context`'s fields, in the requested `language` (Issue #43 item R
    -- previously English-only regardless of the app's selected language).
    Also used as the fallback when OpenAI is unavailable or its call fails.

    `news_matches[*]["impact_label"]` is already localized by
    src.market_intelligence.impact_label() at the time build_brief_context()
    was called (via get_affected_etfs()), so it is used here directly
    rather than the raw "Positive"/"Negative" impact value -- this is what
    keeps raw English impact words out of the zh-TW template.
    """
    holdings_tickers = context.get("holdings_tickers") or []
    watchlist_tickers = context.get("watchlist_tickers") or []
    news_matches = context.get("news_matches") or []
    notable = [m for m in news_matches if m.get("impact") in _NOTABLE_IMPACTS]
    is_zh = language == "zh-TW"

    if not holdings_tickers and not watchlist_tickers:
        if is_zh:
            return (
                "尚無持股或觀察清單項目。請先新增一項持股或觀察清單股票，"
                "即可開始查看每日投資組合摘要。"
            )
        return (
            "No holdings or watchlist items yet. Add a holding or a "
            "watchlist ticker to start seeing a daily portfolio brief."
        )

    if is_zh:
        lines = [
            f"今日投資組合摘要（{context.get('as_of_date', '今天')}）："
            f"追蹤 {len(holdings_tickers)} 項持股與 "
            f"{len(watchlist_tickers)} 項觀察清單股票。"
        ]
        if not context.get("has_notable_activity"):
            lines.append(
                "今日未偵測到重大變化：您的持股與觀察清單目前沒有顯著的"
                "正面或負面新聞影響。"
            )
        else:
            mentions = "、".join(f"{m['ticker']}（{m['impact_label']}）" for m in notable)
            lines.append(
                f"風險／新聞關注：今日新聞可能影響 {len(notable)} 項標的："
                f"{mentions}。"
            )
        if context.get("risk_available"):
            risk = context.get("portfolio_risk") or {}
            largest_ticker = risk.get("largest_ticker")
            largest_weight = risk.get("largest_weight")
            if largest_ticker is not None and largest_weight is not None:
                lines.append(
                    f"集中度檢查：最大持股為 {largest_ticker}，"
                    f"占投資組合 {largest_weight:.1%}。"
                )
        return " ".join(lines)

    lines = [
        f"Today's Portfolio Brief ({context.get('as_of_date', 'today')}): "
        f"tracking {len(holdings_tickers)} holding(s) and "
        f"{len(watchlist_tickers)} watchlist ticker(s)."
    ]

    if not context.get("has_notable_activity"):
        lines.append(
            "No major change detected today: no notable Positive/Negative "
            "news impact found for your holdings or watchlist."
        )
    else:
        mentions = ", ".join(f"{m['ticker']} ({m['impact_label']})" for m in notable)
        lines.append(
            f"Risk/News Attention: {len(notable)} ticker(s) may be affected "
            f"by today's news: {mentions}."
        )

    if context.get("risk_available"):
        risk = context.get("portfolio_risk") or {}
        largest_ticker = risk.get("largest_ticker")
        largest_weight = risk.get("largest_weight")
        if largest_ticker is not None and largest_weight is not None:
            lines.append(
                f"Concentration check: largest position is {largest_ticker} "
                f"at {largest_weight:.1%} of the portfolio."
            )

    return " ".join(lines)


def _build_user_content(context: dict, language: str) -> str:
    payload = {
        "as_of_date": context.get("as_of_date"),
        "holdings_tickers": context.get("holdings_tickers"),
        "watchlist_tickers": context.get("watchlist_tickers"),
        "news_matches": [
            {"ticker": m.get("ticker"), "impact": m.get("impact_label") or m.get("impact")}
            for m in (context.get("news_matches") or [])
        ],
        "notable_count": context.get("notable_count"),
        "has_notable_activity": context.get("has_notable_activity"),
        "portfolio_risk": context.get("portfolio_risk") if context.get("risk_available") else None,
        "language": language,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def generate_daily_brief(context: dict, session_state, language: str = "en",
                          cache_key: str = "daily_brief_cache") -> dict:
    """Orchestration entrypoint the page calls: AI summary (via the central
    OpenAI Responses API service, cached by a fingerprint that includes
    `language`) when configured, else the language-correct rule-based
    fallback -- both grounded exclusively in `context`.

    Always returns a dict with at least "text" (str, never None) and
    "source" ("ai" | "rule_based"); never raises. `language` should be the
    caller's current src.i18n.get_language() ("zh-TW" or "en") -- since the
    cache fingerprint depends on it, switching the app's language always
    produces a fresh, language-correct result instead of a stale cached one
    written in the previously selected language (Issue #43 item S).
    """
    if openai_service.is_configured():
        fp = fingerprint_brief_context(context, language)
        user_content = _build_user_content(context, language)
        result = openai_service.cached_generate(
            session_state, cache_key, fp, _system_instructions(language), user_content,
        )
        if result.get("available") and result.get("text"):
            return {"text": result["text"], "source": "ai"}

    return {"text": rule_based_brief(context, language), "source": "rule_based"}
