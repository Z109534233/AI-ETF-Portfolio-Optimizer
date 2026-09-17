"""
"AI unavailable" broken-feature framing removal tests (Issue #45 item 4).

Covers:
  - no translation string still uses the literal "API key not configured"
    broken-feature phrasing anywhere in the app
  - AI Portfolio Analyst's top mode indicator is a neutral
    Rule-Based/AI-Assisted statement, not an "AI unavailable" warning
  - AI Portfolio Analyst still produces a rule-based narrative with no
    OpenAI configured, and tags a mocked successful OpenAI call correctly
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.i18n import TRANSLATIONS
from src import openai_service
from src.ai_advisor import build_advisor_context, generate_advisor_narrative


FORBIDDEN_BROKEN_PHRASES = [
    "api key not configured",
    "openai api key not configured",
]


def test_no_translation_uses_broken_feature_api_key_phrasing():
    hits = []
    for lang, mapping in TRANSLATIONS.items():
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            low = value.lower()
            for phrase in FORBIDDEN_BROKEN_PHRASES:
                if phrase in low:
                    hits.append((lang, key, value))
    assert hits == [], hits


def test_ai_mode_info_is_neutral_rule_based_not_broken_feature():
    for lang in ("zh-TW", "en"):
        info_text = TRANSLATIONS[lang]["ai_mode_info"]
        success_text = TRANSLATIONS[lang]["ai_mode_success"]
        assert "not configured" not in info_text.lower()
        assert "尚未設定" not in info_text
        # Must read as a neutral CURRENT MODE indicator, not an error/warning.
        assert ("Rule-Based" in info_text) or ("規則式" in info_text)
        assert ("AI-Assisted" in success_text) or ("AI 輔助" in success_text)


def test_ai_advisor_still_produces_rule_based_narrative_without_openai(monkeypatch):
    monkeypatch.setattr(openai_service, "is_configured", lambda: False)
    portfolio = {
        "strategy": "Equal Weight", "market": "United States",
        "tickers": ["VOO", "BND"], "weights": {"VOO": 0.6, "BND": 0.4},
        "investment_amount": 10000.0, "expected_return": 0.08, "volatility": 0.12,
        "sharpe_ratio": 0.5, "max_drawdown": -0.15, "generated_at": "2026-01-01T00:00:00Z",
    }
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    narrative = generate_advisor_narrative(context, "Growth", "Moderate", 10)
    assert narrative["source"] == "rule_based"
    assert narrative["text"]  # a genuine non-empty narrative, not a stub


def test_ai_advisor_tags_mocked_successful_openai_call(monkeypatch):
    monkeypatch.setattr(openai_service, "is_configured", lambda: True)
    monkeypatch.setattr(
        openai_service, "generate_text",
        lambda *a, **k: {"available": True, "text": "MOCKED AI NARRATIVE", "source": "ai", "model": "test"},
    )
    portfolio = {
        "strategy": "Equal Weight", "market": "United States",
        "tickers": ["VOO", "BND"], "weights": {"VOO": 0.6, "BND": 0.4},
        "investment_amount": 10000.0, "expected_return": 0.08, "volatility": 0.12,
        "sharpe_ratio": 0.5, "max_drawdown": -0.15, "generated_at": "2026-01-01T00:00:00Z",
    }
    context = build_advisor_context(portfolio=portfolio, portfolio_source="current")
    narrative = generate_advisor_narrative(context, "Growth", "Moderate", 10)
    assert narrative["source"] == "ai"
    assert narrative["text"] == "MOCKED AI NARRATIVE"


def test_etf_ai_interpretation_fallback_caption_is_neutral_not_broken():
    for lang in ("zh-TW", "en"):
        text = TRANSLATIONS[lang]["etf_ai_interpretation_unavailable"]
        assert "not configured" not in text.lower()
        assert "currently unavailable" not in text.lower()
        assert "目前無法使用" not in text
