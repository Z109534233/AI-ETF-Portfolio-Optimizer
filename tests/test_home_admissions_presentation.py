"""Admissions-facing Home regression checks.

The public landing page should read like a quantitative-finance portfolio,
not a retail-investing marketing page, and should use the applicant's real
name consistently with the CV.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_home_uses_academic_section_framing_and_real_name():
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "Quantitative Analytics Modules" in app
    assert "Analytical Workflow" in app
    assert "Validation & Methodology" in app
    assert "Developed by TZU-HSIN TSENG" in app

    # These retail-marketing sections should not be rendered on Home.
    assert "persona_row(" not in app
    assert "faq_accordion(" not in app
    assert "Why Choose This Platform" not in app


def test_home_does_not_surface_missing_price_coverage_as_warning():
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "coverage_unavailable_caption" not in app


def test_market_intelligence_home_copy_has_no_unverified_provider_warning():
    app = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "does not yet use a verified live provider" not in app
    assert "尚未接入已驗證的即時來源" not in app
