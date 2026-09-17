"""
Daily Portfolio Brief -- deterministic context/fingerprint/rule-based-text
tests, plus mocked OpenAI orchestration tests (Issue #22 section E).

No real network access, no real OpenAI calls anywhere in this file: every
test that reaches generate_daily_brief() either lets src.openai_service
report "not configured" (the default with no OPENAI_API_KEY in the test
environment) or monkeypatches src.openai_service.is_configured /
cached_generate directly. The lower-level single-shot text-generation call
and the real OpenAI client class are never touched.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

from src import openai_service as svc
from src.daily_brief import (
    build_brief_context,
    fingerprint_brief_context,
    rule_based_brief,
    generate_daily_brief,
)


# ── fixtures ─────────────────────────────────────────────────────────────

def _holdings():
    return [
        {"ticker": "QQQ", "quantity": 10, "average_cost": 300.0, "currency": "USD",
         "purchase_date": "2024-01-01"},
        {"ticker": "BND", "quantity": 20, "average_cost": 70.0, "currency": "USD",
         "purchase_date": "2024-02-01"},
    ]


def _watchlist():
    return [{"ticker": "GLD"}, {"ticker": "QQQ"}]  # QQQ deliberately overlaps holdings


def _news_tech_positive():
    # "Technology" sector keyword ("tech") -> matches QQQ, classified Positive.
    return [{"title": "Tech stocks rally on strong earnings", "impact": "Positive"}]


def _news_neutral_only():
    # No sector keyword for Technology/Bond/Gold anywhere in this title.
    return [{"title": "Local weather forecast calls for rain this weekend", "impact": "Neutral"}]


def _news_unrelated_ticker_sector():
    # Real-estate-flavoured headline: would match VNQ (Real Estate) generically,
    # but VNQ is not in this user's holdings/watchlist, so it must never appear.
    return [{"title": "Housing market and REIT prices surge", "impact": "Positive"}]


# ── build_brief_context: ticker extraction / dedup / restriction ───────────

def test_extracts_and_dedupes_tickers_from_holdings_and_watchlist():
    context = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    assert context["holdings_tickers"] == ["BND", "QQQ"]
    assert context["watchlist_tickers"] == ["GLD", "QQQ"]
    # QQQ appears in both lists but the combined relevant-ticker set is deduped
    assert context["all_relevant_tickers"] == ["BND", "GLD", "QQQ"]


def test_news_matches_restricted_to_users_own_tickers_only():
    news = _news_unrelated_ticker_sector()
    context = build_brief_context(_holdings(), _watchlist(), news, as_of_date="2026-09-14")
    matched_tickers = {m["ticker"] for m in context["news_matches"]}
    # Real-estate news would match VNQ generically via get_affected_etfs'
    # default watchlist, but VNQ is not held/watched by this user.
    assert "VNQ" not in matched_tickers
    assert matched_tickers == set(context["all_relevant_tickers"])


def test_no_relevant_tickers_skips_market_intelligence_entirely():
    context = build_brief_context([], [], _news_tech_positive(), as_of_date="2026-09-14")
    assert context["all_relevant_tickers"] == []
    assert context["news_matches"] == []
    assert context["notable_count"] == 0
    assert context["has_notable_activity"] is False


# ── notable_count / has_notable_activity ────────────────────────────────────

def test_positive_news_on_held_ticker_is_notable():
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    qqq_match = next(m for m in context["news_matches"] if m["ticker"] == "QQQ")
    assert qqq_match["impact"] == "Positive"
    assert context["notable_count"] >= 1
    assert context["has_notable_activity"] is True


def test_all_neutral_or_no_matches_means_no_major_change():
    context = build_brief_context(_holdings(), _watchlist(), _news_neutral_only(),
                                   as_of_date="2026-09-14")
    assert all(m["impact"] == "Neutral" for m in context["news_matches"])
    assert context["notable_count"] == 0
    assert context["has_notable_activity"] is False


def test_empty_news_means_no_major_change():
    context = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    assert context["notable_count"] == 0
    assert context["has_notable_activity"] is False
    assert context["news_headline_count"] == 0


# ── fingerprint_brief_context: stability + sensitivity ──────────────────────

def test_fingerprint_stable_for_identical_context():
    c1 = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(), as_of_date="2026-09-14")
    c2 = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(), as_of_date="2026-09-14")
    assert fingerprint_brief_context(c1) == fingerprint_brief_context(c2)


def test_fingerprint_changes_with_new_holding():
    c1 = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    new_holdings = _holdings() + [{"ticker": "GLD", "quantity": 5, "average_cost": 180.0,
                                    "currency": "USD", "purchase_date": "2024-03-01"}]
    c2 = build_brief_context(new_holdings, _watchlist(), [], as_of_date="2026-09-14")
    assert fingerprint_brief_context(c1) != fingerprint_brief_context(c2)


def test_fingerprint_changes_with_new_watchlist_item():
    c1 = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    c2 = build_brief_context(_holdings(), _watchlist() + [{"ticker": "TLT"}], [],
                              as_of_date="2026-09-14")
    assert fingerprint_brief_context(c1) != fingerprint_brief_context(c2)


def test_fingerprint_changes_with_news_reclassification():
    c1 = build_brief_context(_holdings(), _watchlist(), _news_neutral_only(), as_of_date="2026-09-14")
    c2 = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(), as_of_date="2026-09-14")
    assert fingerprint_brief_context(c1) != fingerprint_brief_context(c2)


def test_fingerprint_changes_with_new_day():
    c1 = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    c2 = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-15")
    assert fingerprint_brief_context(c1) != fingerprint_brief_context(c2)


# ── rule_based_brief ─────────────────────────────────────────────────────

def test_rule_based_brief_never_crashes_on_fully_empty_input():
    context = build_brief_context([], [], [], as_of_date="2026-09-14")
    text = rule_based_brief(context)
    assert isinstance(text, str) and text
    assert "no holdings or watchlist" in text.lower()


def test_rule_based_brief_says_no_major_change_when_not_notable():
    context = build_brief_context(_holdings(), _watchlist(), _news_neutral_only(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context)
    assert "no major change" in text.lower()


def test_rule_based_brief_mentions_tickers_when_notable():
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context)
    assert "QQQ" in text
    assert "no major change" not in text.lower()


# ── generate_daily_brief orchestration (OpenAI mocked, never real) ─────────

def test_generate_daily_brief_uses_rule_based_when_not_configured(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: False)
    calls = []
    monkeypatch.setattr(svc, "cached_generate", lambda *a, **k: calls.append((a, k)))

    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    result = generate_daily_brief(context, session_state={})

    assert result["source"] == "rule_based"
    assert "QQQ" in result["text"]
    assert calls == []  # cached_generate must never be called when not configured


def test_generate_daily_brief_uses_ai_when_configured_and_available(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: True)

    def _fake_cached_generate(session_state, cache_key, fp, system, user_content, **kwargs):
        return {"available": True, "text": "AI-written brief.", "source": "ai"}

    monkeypatch.setattr(svc, "cached_generate", _fake_cached_generate)

    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    result = generate_daily_brief(context, session_state={})

    assert result == {"text": "AI-written brief.", "source": "ai"}


def test_generate_daily_brief_falls_back_when_ai_unavailable(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: True)

    def _fake_cached_generate(session_state, cache_key, fp, system, user_content, **kwargs):
        return {"available": False, "text": None, "source": "rule_based", "error": "boom"}

    monkeypatch.setattr(svc, "cached_generate", _fake_cached_generate)

    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    result = generate_daily_brief(context, session_state={})

    assert result["source"] == "rule_based"
    assert "QQQ" in result["text"]


def test_generate_daily_brief_never_crashes_on_fully_empty_input(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: False)
    context = build_brief_context([], [], [], as_of_date="2026-09-14")
    result = generate_daily_brief(context, session_state={})
    assert result["source"] == "rule_based"
    assert isinstance(result["text"], str) and result["text"]


# ── Issue #43 item R: rule_based_brief() must be language-aware ────────────

_FORBIDDEN_ENGLISH_TEMPLATE_PHRASES = (
    "Today's Portfolio Brief",
    "No major change detected",
    "Risk/News Attention",
    "Positive/Negative",
)


def test_rule_based_brief_zh_tw_no_holdings_state():
    context = build_brief_context([], [], [], as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "尚無持股" in text
    for phrase in _FORBIDDEN_ENGLISH_TEMPLATE_PHRASES:
        assert phrase not in text


def test_rule_based_brief_zh_tw_opening_line_has_date_and_counts():
    context = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "2026-09-14" in text
    assert "2" in text  # 2 holdings (QQQ, BND)
    assert "2" in text  # 2 watchlist tickers (GLD, QQQ)


def test_rule_based_brief_zh_tw_no_major_change():
    context = build_brief_context(_holdings(), _watchlist(), _news_neutral_only(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "今日未偵測到重大變化" in text
    for phrase in _FORBIDDEN_ENGLISH_TEMPLATE_PHRASES:
        assert phrase not in text


def test_rule_based_brief_zh_tw_risk_news_attention_mentions_ticker():
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "風險／新聞關注" in text
    assert "QQQ" in text
    for phrase in _FORBIDDEN_ENGLISH_TEMPLATE_PHRASES:
        assert phrase not in text


def test_rule_based_brief_zh_tw_concentration_check():
    context = build_brief_context(_holdings(), _watchlist(), [],
                                   portfolio_risk={"largest_ticker": "QQQ", "largest_weight": 0.7},
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "集中度檢查" in text
    assert "QQQ" in text
    assert "70.0%" in text


def test_rule_based_brief_zh_tw_localized_impact_label_not_raw_english():
    """impact_label is already localized (src.market_intelligence.impact_label())
    at build_brief_context() time -- the zh-TW template must use it, never
    the raw 'Positive'/'Negative' impact value."""
    import streamlit as st
    st.session_state["language"] = "zh-TW"
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context, language="zh-TW")
    assert "Positive" not in text
    assert "Negative" not in text


def test_rule_based_brief_english_still_english_by_default():
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    text = rule_based_brief(context)  # language defaults to "en"
    assert "Today's Portfolio Brief" in text
    assert "QQQ" in text


# ── Issue #43 item S: AI path language + cache correctness ────────────────

def test_fingerprint_is_sensitive_to_language():
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    fp_en = fingerprint_brief_context(context, "en")
    fp_zh = fingerprint_brief_context(context, "zh-TW")
    assert fp_en != fp_zh


def test_generate_daily_brief_does_not_return_stale_english_after_language_switch(monkeypatch):
    """Simulates switching the app from en -> zh-TW: the AI cache must miss
    (different fingerprint) so a fresh, zh-TW-instructed call is made,
    never a cached English result reused as-is."""
    monkeypatch.setattr(svc, "is_configured", lambda: True)
    seen_instructions = []

    def _fake_cached_generate(session_state, cache_key, fp, system, user_content, **kwargs):
        seen_instructions.append(system)
        cached = session_state.get(cache_key)
        if cached and cached.get("fingerprint") == fp:
            return cached["result"]
        text = "Traditional Chinese response." if "Traditional Chinese" in system else "English response."
        result = {"available": True, "text": text, "source": "ai"}
        session_state[cache_key] = {"fingerprint": fp, "result": result}
        return result

    monkeypatch.setattr(svc, "cached_generate", _fake_cached_generate)

    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    session_state = {}

    result_en = generate_daily_brief(context, session_state, language="en")
    assert result_en["text"] == "English response."

    result_zh = generate_daily_brief(context, session_state, language="zh-TW")
    assert result_zh["text"] == "Traditional Chinese response."
    assert len(seen_instructions) == 2  # both requests actually reached the (mocked) API


def test_mocked_ai_request_contains_correct_language_instruction(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: True)
    captured = {}

    def _fake_cached_generate(session_state, cache_key, fp, system, user_content, **kwargs):
        captured["system"] = system
        return {"available": True, "text": "ok", "source": "ai"}

    monkeypatch.setattr(svc, "cached_generate", _fake_cached_generate)
    context = build_brief_context(_holdings(), _watchlist(), [], as_of_date="2026-09-14")

    generate_daily_brief(context, {}, language="zh-TW")
    assert "Traditional Chinese" in captured["system"]

    generate_daily_brief(context, {}, language="en")
    assert "English" in captured["system"]


def test_generate_daily_brief_fallback_is_language_correct(monkeypatch):
    monkeypatch.setattr(svc, "is_configured", lambda: True)
    monkeypatch.setattr(
        svc, "cached_generate",
        lambda *a, **k: {"available": False, "text": None, "source": "rule_based", "error": "boom"},
    )
    context = build_brief_context(_holdings(), _watchlist(), _news_tech_positive(),
                                   as_of_date="2026-09-14")
    result = generate_daily_brief(context, {}, language="zh-TW")
    assert result["source"] == "rule_based"
    assert "風險／新聞關注" in result["text"]
    assert "Risk/News Attention" not in result["text"]


# ── self-check: no unmocked OpenAI/network call anywhere in this file ──────

def test_no_raw_openai_or_generate_text_usage_in_this_file():
    forbidden = [
        "openai" + "." + "OpenAI(",
        "svc." + "generate_text(",
        "openai_service." + "generate_text(",
    ]
    with open(__file__, "r", encoding="utf-8") as f:
        source = f.read()
    for needle in forbidden:
        assert source.count(needle) == 0, needle
