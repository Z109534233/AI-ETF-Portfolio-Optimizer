"""
Market Intelligence -- reviewer-readiness regression tests (Issue #43
sections E-I).

Covers:
  - E: raw English source headlines must stay source-faithful but visually
    distinct (quoted) inside zh-TW rule-based prose, and the OpenAI prompt
    must instruct the model to keep quoted headline titles clearly marked.
  - F: Market Impact Score shows ONE numeric scale + a qualitative label,
    never a second star-rating string for the same concept.
  - G: the Fear & Greed KPI card is not rendered while unavailable (no
    verified source connected).
  - H: star salience classes scale with star count, never green/red.
  - I: rule-based vs mocked-AI tag correctness for generate_market_summary()
    (complements the existing mocked tests in
    tests/test_openai_integration_points.py).

No real network call anywhere in this file: fetch_market_news()/
fetch_market_indices() are monkeypatched, and the OpenAI paths are either
left unconfigured (no OPENAI_API_KEY) or monkeypatched directly.
"""

import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pytest

import streamlit as st

import src.market_intelligence as mi_mod
from src.market_intelligence import (
    generate_market_summary, _generate_rule_based_summary, _market_summary_prompt,
    stars_to_impact_label, fetch_fear_greed_index, calculate_market_impact,
    get_economic_calendar, generate_today_ai_summary,
)
from src.ui import star_rating_html, _star_salience_class


def _apptest_from_file(rel_path, **kwargs):
    from streamlit.testing.v1 import AppTest
    st.page_link = lambda *a, **k: None
    path = os.path.join(REPO_ROOT, rel_path)
    return AppTest.from_file(path, **kwargs)


def _sample_news():
    return [
        {"title": "Fed signals possible rate cut", "impact": "Positive",
         "publisher": "Reuters", "link": "https://example.com/1", "published": datetime(2026, 9, 14, 9, 0)},
        {"title": "Tech earnings beat expectations", "impact": "Positive",
         "publisher": "Bloomberg", "link": "https://example.com/2", "published": datetime(2026, 9, 14, 10, 0)},
        {"title": "Oil prices tumble on oversupply concerns", "impact": "Negative",
         "publisher": "AP", "link": "https://example.com/3", "published": datetime(2026, 9, 14, 11, 0)},
    ]


def _sample_sentiment():
    return {"bullish_pct": 60.0, "neutral_pct": 20.0, "bearish_pct": 20.0}


def _run_market_intelligence(lang="en"):
    at = _apptest_from_file("pages/8_Market_Intelligence.py", default_timeout=180)
    at.session_state["language"] = lang
    at.run()
    assert at.exception == []
    return at


@pytest.fixture()
def mocked_market_data(monkeypatch):
    """No real yfinance network calls anywhere in this file."""
    import src.news as news_mod
    monkeypatch.setattr(news_mod, "fetch_market_news", lambda limit=10: _sample_news())
    monkeypatch.setattr(mi_mod, "fetch_market_indices", lambda: {
        "sp500": {"label": "S&P 500", "available": True, "price": 5000.0, "change": 10.0, "change_pct": 0.2},
        "nasdaq": {"label": "NASDAQ", "available": True, "price": 17000.0, "change": -5.0, "change_pct": -0.1},
        "dow": {"label": "Dow Jones", "available": True, "price": 40000.0, "change": 20.0, "change_pct": 0.05},
        "russell": {"label": "Russell 2000", "available": False},
        "vix": {"label": "VIX", "available": True, "price": 15.0, "change": 0.5, "change_pct": 3.4},
    })


# ── Issue #43 item E: raw source headlines stay source-faithful but quoted ──

def test_rule_based_summary_quotes_additional_headlines_in_zh_tw(monkeypatch):
    # Monkeypatch mi_mod's own get_language() reference rather than
    # mutating the real global st.session_state -- this module has no
    # active ScriptRunContext in a plain unit test, and writing to the
    # bare-mode session-state singleton would leak into (and corrupt) the
    # isolated per-AppTest session state of unrelated page tests running
    # later in the same pytest process.
    monkeypatch.setattr(mi_mod, "get_language", lambda: "zh-TW")
    news = _sample_news()
    text = _generate_rule_based_summary(news, _sample_sentiment(), [])
    # The exact source headline text is preserved verbatim...
    assert "Tech earnings beat expectations" in text
    # ...and the SECOND/THIRD headlines are wrapped in Chinese corner
    # quotes rather than sitting unmarked inside the Chinese sentence.
    assert "「Tech earnings beat expectations」" in text
    assert "「Oil prices tumble on oversupply concerns」" in text


def test_rule_based_summary_quotes_additional_headlines_in_en(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "en")
    news = _sample_news()
    text = _generate_rule_based_summary(news, _sample_sentiment(), [])
    assert "Tech earnings beat expectations" in text
    assert '"Tech earnings beat expectations"' in text


def test_market_summary_prompt_instructs_quoting_headlines_in_zh_tw(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "zh-TW")
    prompt = _market_summary_prompt(_sample_news(), _sample_sentiment(), [])
    assert "「」" in prompt or "quoted" in prompt.lower()
    assert "never translate or rewrite a headline" in prompt.lower()


def test_market_summary_prompt_english_has_no_quoting_instruction_noise(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "en")
    prompt = _market_summary_prompt(_sample_news(), _sample_sentiment(), [])
    assert "Respond entirely in English." in prompt


# ── Issue #43 item F: Market Impact Score has one scale, not two ──────────

def test_stars_to_impact_label_is_qualitative_text_not_stars():
    for stars in range(0, 6):
        label = stars_to_impact_label(stars)
        assert "★" not in label and "☆" not in label


def test_market_impact_score_kpi_has_no_duplicate_star_string(mocked_market_data):
    at = _run_market_intelligence("en")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Market Impact Score" in corpus
    # Isolate the Market Impact Score KPI card itself (there are other,
    # legitimate star ratings elsewhere on the page for per-event/per-ETF
    # cards) and confirm its OWN secondary text is the qualitative label,
    # never a second star-rating string duplicating the numeric score/100.
    idx = corpus.index("Market Impact Score")
    card_html = corpus[idx:idx + 900]
    assert "/100" in card_html
    assert "★" not in card_html and "☆" not in card_html
    assert "High Impact" in card_html or "kpi-sub" in card_html


# ── Issue #43 item G: Fear & Greed hidden while unavailable ───────────────

def test_fear_greed_index_reports_unavailable_with_no_source():
    result = fetch_fear_greed_index()
    assert result["available"] is False


def test_fear_greed_card_not_rendered_when_unavailable(mocked_market_data):
    at = _run_market_intelligence("en")
    corpus = "\n".join(m.value for m in at.markdown)
    # No broken/placeholder KPI card next to the indices...
    assert "Data source not yet connected" not in corpus
    # ...but a short, non-KPI note that it isn't connected yet.
    captions = "\n".join(c.value for c in at.caption)
    assert "Fear & Greed" in captions or "恐懼與貪婪" in captions


def test_fear_greed_card_renders_when_available(mocked_market_data, monkeypatch):
    monkeypatch.setattr(mi_mod, "fetch_fear_greed_index", lambda: {
        "available": True, "label": "Fear & Greed Index", "value": 55, "category": "Neutral",
    })
    at = _run_market_intelligence("en")
    corpus = "\n".join(m.value for m in at.markdown)
    assert "Fear & Greed Index" in corpus
    assert "55" in corpus


# ── Issue #43 item H: star salience scales with impact, not direction ─────

def test_star_salience_class_ramps_with_star_count():
    assert _star_salience_class(1) == "star-rating-muted"
    assert _star_salience_class(2) == "star-rating-muted"
    assert _star_salience_class(3) == "star-rating-medium"
    assert _star_salience_class(4) == "star-rating-strong"
    assert _star_salience_class(5) == "star-rating-strong"


def test_star_rating_html_uses_salience_class_not_direction_color():
    low = star_rating_html(1)
    mid = star_rating_html(3)
    high = star_rating_html(5)
    assert "star-rating-muted" in low
    assert "star-rating-medium" in mid
    assert "star-rating-strong" in high
    # Never a green/red "badge-green"/"badge-red" direction class on a star
    # rating -- stars measure impact/relevance, not positive/negative.
    for html in (low, mid, high):
        assert "badge-green" not in html and "badge-red" not in html


def test_star_rating_html_empty_stars_always_muted():
    html = star_rating_html(0)
    assert "star-empty" in html
    assert "★" not in html.replace("star-filled", "")  # no filled stars rendered


# ── Issue #43 item I: rule-based vs AI tag correctness (page level) ───────

def test_market_summary_tag_is_rule_based_with_no_api_key(mocked_market_data):
    at = _run_market_intelligence("en")
    # chart_card()'s tag is rendered as markdown HTML -- assert the
    # Rule-Based tag text appears (no OPENAI_API_KEY in the test
    # environment, so every AI-capable card falls back to rule-based).
    markdown_corpus = "\n".join(m.value for m in at.markdown)
    assert '<span class="badge badge-neutral">Rule-Based</span>' in markdown_corpus
    assert '<span class="badge badge-neutral">AI-Generated</span>' not in markdown_corpus


# ── Completion-gate smoke: zh-TW + en render without exception ───────────

def test_market_intelligence_page_renders_zh_tw_with_mocked_data(mocked_market_data):
    _run_market_intelligence("zh-TW")


def test_market_intelligence_page_renders_en_with_mocked_data(mocked_market_data):
    _run_market_intelligence("en")


def test_generate_market_summary_ai_tag_only_on_mocked_success(monkeypatch):
    from src import openai_service as svc

    class _FakeResponse:
        def __init__(self, text):
            self.output_text = text

    class _FakeResponses:
        def create(self, **kwargs):
            return _FakeResponse("AI-written market summary.")

    class _FakeClient:
        def __init__(self):
            self.responses = _FakeResponses()

    # 1) Unavailable -> rule-based tag/content.
    result_unconfigured = generate_market_summary(_sample_news(), _sample_sentiment(), [])
    assert result_unconfigured["source"] == "rule_based"

    # 2) Configured + mocked successful generation -> AI-generated tag/content.
    monkeypatch.setattr(svc, "get_client", lambda: _FakeClient())
    result_configured = generate_market_summary(_sample_news(), _sample_sentiment(), [])
    assert result_configured["source"] == "ai"
    assert result_configured["text"] == "AI-written market summary."



def _all_up_indices():
    return {
        "sp500": {"label": "S&P 500", "available": True, "price": 6900.0, "change_pct": 0.97},
        "nasdaq": {"label": "NASDAQ", "available": True, "price": 24000.0, "change_pct": 1.57},
        "dow": {"label": "Dow Jones", "available": True, "price": 47000.0, "change_pct": 0.40},
        "russell": {"label": "Russell 2000", "available": True, "price": 2500.0, "change_pct": 0.55},
        "vix": {"label": "VIX", "available": True, "price": 14.61, "change_pct": -4.20},
    }


def test_rule_based_summary_uses_actual_indices_before_bearish_news(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "zh-TW")
    bearish = {"bullish_pct": 10.0, "neutral_pct": 10.0, "bearish_pct": 80.0}
    text = _generate_rule_based_summary(
        _sample_news(), bearish, [], market_indices=_all_up_indices(),
    )
    assert "實際主要股價指數今日整體走揚" in text
    assert "VIX 為 14.61" in text
    assert "偏低且下降" in text
    assert "市場承壓" not in text


def test_top_market_overview_uses_index_direction_not_headline_mood(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "zh-TW")
    # Make every headline negative while the actual displayed indices rise.
    news = [
        {"title": "Stocks face tariff concerns", "impact": "Negative"},
        {"title": "Fed warns of inflation risk", "impact": "Negative"},
    ]
    result = generate_today_ai_summary(news, _all_up_indices())
    overview = result["sections"][0]["text"]
    assert "市場整體走揚" in overview
    assert "新聞分類" in overview
    assert "市場承壓" not in overview


def test_single_uncorroborated_fed_headline_cannot_receive_high_impact_rating():
    one_story = [{
        "title": "Howard Marks comments on Federal Reserve policy",
        "impact": "Neutral", "publisher": "Example", "link": "", "published": None,
    }]
    result = calculate_market_impact(one_story)
    assert result["score"] <= 69
    assert result["stars"] <= 3


def test_unverified_static_economic_calendar_is_not_presented_as_live():
    assert get_economic_calendar() == []


def test_market_summary_prompt_makes_index_snapshot_authoritative(monkeypatch):
    monkeypatch.setattr(mi_mod, "get_language", lambda: "en")
    prompt = _market_summary_prompt(
        _sample_news(), _sample_sentiment(), [], market_indices=_all_up_indices(),
    )
    assert "MARKET DIRECTION MUST FOLLOW THE ACTUAL INDEX SNAPSHOT" in prompt
    assert "Do not invent upcoming Fed decisions" in prompt
    assert "S&P 500 +0.97%" in prompt
