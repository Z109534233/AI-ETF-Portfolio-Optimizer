"""
Utilities Module
Helper functions for the AI ETF Portfolio Optimizer.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date

from src.theme import COLORS, icon_svg
from src.ui import kpi_card as _kpi_card


def load_css(css_path: str = None) -> None:
    """Load custom CSS into Streamlit. Silently skips if file not found."""
    if css_path is None:
        css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    try:
        if os.path.exists(css_path):
            with open(css_path, "r") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency string."""
    return f"${value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a number as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with comma separators."""
    return f"{value:,.{decimals}f}"


def metric_card_html(label: str, value: str, delta: str = None,
                      color: str = "#3B82F6") -> str:
    """Generate HTML for a styled KPI metric card (icon + trend aware).

    Kept backward compatible with existing call sites across all pages:
    `delta` starting with '+'/'-' renders as a colored trend indicator,
    otherwise it is shown as plain supporting text.
    """
    trend = delta if delta and delta.strip()[:1] in ("+", "-") else None
    sub = delta if delta and trend is None else None
    return _kpi_card(label, value, sub=sub, color=color, trend=trend)


def page_header(title: str, subtitle: str = None, icon: str = None) -> None:
    """Render a styled page header with a clear title/subtitle hierarchy."""
    subtitle_html = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def info_box(text: str, color: str = "#3B82F6") -> None:
    """Render a styled info box."""
    st.markdown(f"""
    <div style="
        background: {COLORS['primary_soft']};
        border: 1px solid {color};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: {COLORS['text']};
        font-size: 13px;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)


def disclaimer_box(text: str = None) -> None:
    """Render an educational disclaimer box."""
    if text is None:
        text = (
            "This platform is for <strong>educational purposes only</strong> and does not constitute "
            "financial advice. All analysis, projections, and AI-generated content are for "
            "demonstration purposes. Past performance does not guarantee future results. "
            "Always consult a qualified financial adviser before making investment decisions."
        )
    st.markdown(f"""
    <div style="
        background: {COLORS['warning_soft']};
        border: 1px solid rgba(251,191,36,0.35);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #FDE68A;
        font-size: 12px;
        line-height: 1.5;
        display: flex;
        gap: 8px;
        align-items: flex-start;
    ">
        <span style="flex-shrink:0;margin-top:1px;">{icon_svg('shield', 14, '#FDE68A')}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=True).encode("utf-8")


def weights_to_dataframe(weights: dict, investment_amount: float = 10000.0) -> pd.DataFrame:
    """Convert weights dict to a formatted DataFrame."""
    rows = []
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "Ticker": ticker,
            "Weight": f"{weight:.2%}",
            "Allocation ($)": f"${weight * investment_amount:,.2f}",
            "Weight (decimal)": round(weight, 4),
        })
    return pd.DataFrame(rows)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    return float(result)


def get_date_range_defaults() -> tuple:
    """Return default start and end dates (5 years back to today)."""
    end = date.today()
    start = date(end.year - 5, end.month, end.day)
    return start, end


def validate_weights(weights: dict) -> bool:
    """Check that weights sum to approximately 1.0."""
    total = sum(weights.values())
    return abs(total - 1.0) < 0.01


def color_metric(value: float, positive_is_good: bool = True) -> str:
    """Return green or red color string based on value sign."""
    if positive_is_good:
        return COLORS["success"] if value >= 0 else COLORS["danger"]
    else:
        return COLORS["danger"] if value >= 0 else COLORS["success"]


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    base = os.path.dirname(os.path.dirname(__file__))
    dirs = ["data", "database", "reports", "images", "assets"]
    for d in dirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)
