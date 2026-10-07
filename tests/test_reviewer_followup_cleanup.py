"""Regression checks for the admissions-review cleanup pass."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_author_and_ai_assistance_disclosure():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**Author:** Tzu-Hsin Tseng" in text
    assert "## Development attribution" in text
    assert "Generative AI tools were used as development assistants" in text
    assert "no longer applies full-sample" not in text


def test_license_uses_real_author_and_current_year():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Tzu-Hsin Tseng" in text
    assert "Copyright (c) 2025 AI ETF Portfolio Optimizer" not in text


def test_only_test_ci_remains_in_workflows_directory():
    workflows = sorted(p.name for p in (ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows == ["tests.yml"]


def test_walk_forward_copy_is_localized_not_inline_bilingual_logic():
    page = (ROOT / "pages" / "2_Portfolio_Optimizer.py").read_text(encoding="utf-8")
    assert '_wf_title = t("opt_walk_forward_title")' in page
    assert 't("opt_walk_forward_method_note"' in page
    assert '_wf_title = "Walk-Forward Out-of-Sample Backtest"' not in page
    assert '_wf_title = "Walk-Forward 樣本外回測"' not in page


def test_backtest_docstring_describes_rolling_not_expanding_window():
    text = (ROOT / "src" / "backtesting.py").read_text(encoding="utf-8")
    assert "fixed-length rolling-window walk-forward backtest" in text
    assert "expanding calendar walk-forward backtest" not in text


def test_home_hero_uses_compact_academic_title():
    text = (ROOT / "src" / "ui.py").read_text(encoding="utf-8")
    assert 'title = "Quantitative ETF Portfolio Analytics"' in text
    assert 'title = "ETF Portfolio Analytics & Quantitative Decision Platform"' not in text
