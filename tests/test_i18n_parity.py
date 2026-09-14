"""
i18n key parity / placeholder parity / raw-key-leak tests (Issue #20
section 12 release gate: "i18n key parity + placeholder parity + raw key
scan").

Three independent guarantees:
  1. Every key defined for zh-TW is also defined for en, and vice versa --
     a key missing from one language falls back to the other (src.i18n.t()),
     which silently shows the wrong language rather than raising, so this
     must be caught statically instead of by manually clicking every page
     in both languages.
  2. Every {placeholder} used in a zh-TW string appears in the matching en
     string (and vice versa) -- a placeholder present in only one language
     would raise a KeyError from str.format() the first time that language
     renders the string with real data.
  3. Every t("...") call anywhere in the app's source references a key that
     actually exists in TRANSLATIONS -- a typo'd key doesn't crash (t()
     falls back to the raw key string), so a mismatch would otherwise only
     surface as an unnoticed raw key leaking into the rendered UI.
"""

import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.i18n import TRANSLATIONS

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
T_CALL_RE = re.compile(r"\bt\(\s*[\"'](\w+)[\"']")


def _placeholders(value) -> set:
    return set(PLACEHOLDER_RE.findall(value)) if isinstance(value, str) else set()


def test_zh_and_en_define_the_same_key_set():
    zh_keys = set(TRANSLATIONS["zh-TW"].keys())
    en_keys = set(TRANSLATIONS["en"].keys())
    assert zh_keys - en_keys == set(), f"keys only in zh-TW: {sorted(zh_keys - en_keys)}"
    assert en_keys - zh_keys == set(), f"keys only in en: {sorted(en_keys - zh_keys)}"


def test_placeholders_match_between_languages():
    zh, en = TRANSLATIONS["zh-TW"], TRANSLATIONS["en"]
    mismatches = []
    for key in zh.keys() & en.keys():
        pz, pe = _placeholders(zh[key]), _placeholders(en[key])
        if pz != pe:
            mismatches.append((key, pz, pe))
    assert mismatches == [], mismatches


# Issue #20 section 12 release-gate item 8: "terminology scan for stale
# forbidden labels (AI Score, misleading Confidence, old action-advice
# labels) except where intentionally discussed in tests/docs". Each of
# these was a real mislabeled/renamed string found and fixed during the
# review (see git history for pages/1_ETF_Analysis.py, pages/6_AI_Advisor.py,
# pages/8_Market_Intelligence.py, app.py, src/i18n.py) -- this pins the fix
# so none of them can silently come back.
FORBIDDEN_TERMS = [
    "AI Score", "AI ETF Summary", "AI Market Intelligence Center", "AI Market Sentiment",
    "Today's AI Summary", "今日 AI 摘要", "AI 市場情報中心", "AI Advisor", "AI 投資分析",
    "AI 市場情緒", "AI Investment Analysis", "AI Investment Insights", "AI 顧問",
]


def test_no_forbidden_stale_terminology_in_translations():
    hits = []
    for lang, mapping in TRANSLATIONS.items():
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            for term in FORBIDDEN_TERMS:
                if term in value:
                    hits.append((lang, key, term, value))
    assert hits == [], hits


def test_every_t_call_in_source_references_a_real_key():
    en_keys = set(TRANSLATIONS["en"].keys())
    missing = {}
    source_files = (
        glob.glob(os.path.join(REPO_ROOT, "pages", "*.py"))
        + glob.glob(os.path.join(REPO_ROOT, "src", "*.py"))
        + [os.path.join(REPO_ROOT, "app.py")]
    )
    for path in source_files:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for m in T_CALL_RE.finditer(content):
            key = m.group(1)
            if key not in en_keys:
                missing.setdefault(key, set()).add(os.path.relpath(path, REPO_ROOT))
    assert missing == {}, {k: sorted(v) for k, v in missing.items()}
