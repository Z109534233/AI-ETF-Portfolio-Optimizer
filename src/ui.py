"""
UI Component Library
Reusable Streamlit UI helpers implementing the AI ETF Portfolio Optimizer
design system (see assets/style.css + src/theme.py). Used by app.py and all
pages to keep layout, typography and card styling consistent site-wide.
"""

import contextlib
import streamlit as st

from src.theme import COLORS, icon_svg

# ── Navigation ────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    {"page": "app.py", "label": "Dashboard"},
    {"page": "pages/1_ETF_Analysis.py", "label": "ETF Analysis"},
    {"page": "pages/2_Portfolio_Optimizer.py", "label": "Portfolio Optimizer"},
    {"page": "pages/3_Investment_Simulator.py", "label": "Investment Simulator"},
    {"page": "pages/4_Risk_Analytics.py", "label": "Risk Analytics"},
    {"page": "pages/5_Machine_Learning.py", "label": "Machine Learning"},
    {"page": "pages/6_AI_Advisor.py", "label": "AI Advisor"},
    {"page": "pages/7_Portfolio_History.py", "label": "Portfolio History"},
]


def render_sidebar_nav() -> None:
    """Render the branded product header and primary navigation list.

    The currently active page is highlighted automatically by Streamlit
    (st.page_link sets aria-current="page"), styled via assets/style.css.
    """
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-mark">AI</div>
        <div class="sidebar-brand-text">
            <div class="sidebar-brand-name">AI ETF Optimizer</div>
            <div class="sidebar-brand-sub">Portfolio Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
    for item in NAV_ITEMS:
        st.page_link(item["page"], label=item["label"])


def render_sidebar_footer() -> None:
    """Render the pinned-to-bottom sidebar footer. Call last inside `with st.sidebar:`."""
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-badge">Educational Use Only</div>
        <div class="sidebar-footer-text">Not financial advice</div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Section (Home page) ───────────────────────────────────────────────────
def hero_section() -> None:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badges">
            <span class="badge badge-blue">Live Market Data</span>
            <span class="badge badge-green">Portfolio Analytics</span>
            <span class="badge badge-neutral">Educational Use</span>
        </div>
        <h1 class="hero-title">AI ETF Portfolio Optimizer</h1>
        <p class="hero-subtitle">AI-Powered ETF Portfolio Analytics and Optimization Platform</p>
        <p class="hero-desc">
            Analyse ETF performance, measure portfolio risk, compare allocation strategies,
            simulate long-term investment outcomes, and generate explainable portfolio
            insights — in one professional analytics workspace.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, _ = st.columns([1.3, 1, 3])
    with col1:
        if st.button("Launch Portfolio Optimizer", type="primary", use_container_width=True):
            st.switch_page("pages/2_Portfolio_Optimizer.py")
    with col2:
        if st.button("Analyze ETFs", type="secondary", use_container_width=True):
            st.switch_page("pages/1_ETF_Analysis.py")


# ── Section / Page Headers ──────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = None) -> None:
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{title}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, variant: str = "neutral") -> str:
    """Return an inline badge <span> for composing into other HTML blocks."""
    return f'<span class="badge badge-{variant}">{text}</span>'


# ── KPI Cards ───────────────────────────────────────────────────────────────────
_LABEL_ICON_MAP = [
    (("return", "growth", "gain"), "trending-up"),
    (("drawdown", "loss"), "trending-down"),
    (("volatility", "risk", "std"), "activity"),
    (("sharpe", "sortino", "calmar", "ratio", "score"), "target"),
    (("value", "$", "amount", "invest"), "dollar"),
    (("diversif", "holdings", "assets"), "layers"),
    (("allocation", "weight"), "pie-chart"),
    (("var", "cvar"), "shield"),
]


def _infer_icon(label: str) -> str:
    low = label.lower()
    for keys, icon in _LABEL_ICON_MAP:
        if any(k in low for k in keys):
            return icon
    return "bar-chart"


def kpi_card(label: str, value: str, sub: str = None, color: str = "#3B82F6",
             icon: str = None, trend: str = None) -> str:
    """Build a KPI metric card. `trend` should be a short string starting with
    '+' or '-' to render a colored up/down indicator; otherwise shown as plain text.
    """
    icon_name = icon or _infer_icon(label)
    icon_html = icon_svg(icon_name, 16, color)

    trend_html = ""
    if trend:
        is_down = trend.strip().startswith("-")
        is_up = trend.strip().startswith("+")
        if is_up or is_down:
            t_color = COLORS["danger"] if is_down else COLORS["success"]
            arrow = "&#9660;" if is_down else "&#9650;"
            trend_html = f'<span class="kpi-trend" style="color:{t_color};">{arrow} {trend.lstrip("+-")}</span>'
        else:
            trend_html = f'<span class="kpi-trend kpi-trend-neutral">{trend}</span>'

    sub_html = f'<span class="kpi-sub">{sub}</span>' if sub else ""

    return f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-label">{label}</span>
            <span class="kpi-icon" style="background:{color}22;">{icon_html}</span>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-bottom">{sub_html}{trend_html}</div>
    </div>
    """


# ── Chart Card Container ───────────────────────────────────────────────────────
@contextlib.contextmanager
def chart_card(title: str, subtitle: str = None, tag: str = None):
    """Context manager producing a bordered card with a title/subtitle header
    and an optional top-right tag. Use like:

        with chart_card("ETF Performance", "Normalized comparison"):
            st.plotly_chart(fig, use_container_width=True)
    """
    container = st.container(border=True)
    with container:
        sub_html = f'<div class="chart-card-subtitle">{subtitle}</div>' if subtitle else ""
        tag_html = f'<span class="badge badge-neutral">{tag}</span>' if tag else ""
        st.markdown(f"""
        <div class="chart-card-header">
            <div>
                <div class="chart-card-title">{title}</div>
                {sub_html}
            </div>
            {tag_html}
        </div>
        """, unsafe_allow_html=True)
        yield container


# ── Feature Overview Card ───────────────────────────────────────────────────────
def feature_card(title: str, desc: str, icon: str = "activity") -> str:
    return f"""
    <div class="feature-card">
        <div class="feature-card-icon">{icon_svg(icon, 20, COLORS["primary"])}</div>
        <div class="feature-card-title">{title}</div>
        <div class="feature-card-desc">{desc}</div>
    </div>
    """


# ── Empty / Error States ─────────────────────────────────────────────────────────
def empty_state(title: str, description: str, icon: str = "layers") -> None:
    st.markdown(f"""
    <div class="state-card state-empty">
        <div class="state-icon">{icon_svg(icon, 26, COLORS["text_muted"])}</div>
        <div class="state-title">{title}</div>
        <div class="state-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def error_state(title: str, description: str) -> None:
    st.markdown(f"""
    <div class="state-card state-error">
        <div class="state-icon">{icon_svg("shield", 26, COLORS["danger"])}</div>
        <div class="state-title">{title}</div>
        <div class="state-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Tables ──────────────────────────────────────────────────────────────────────
def style_signed_columns(df, columns):
    """Return a pandas Styler that colors signed numeric/currency/percent
    string columns green (>=0) or red (<0), matching the design system.
    """
    def _color(val):
        try:
            cleaned = str(val).replace("$", "").replace(",", "").replace("%", "").replace("x", "")
            num = float(cleaned)
        except (ValueError, TypeError):
            return ""
        color = COLORS["success"] if num >= 0 else COLORS["danger"]
        return f"color:{color}; font-weight:600;"

    return df.style.applymap(_color, subset=columns)


# ── Footer ──────────────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="app-footer">
        <div class="app-footer-brand">AI ETF Portfolio Optimizer</div>
        <div class="app-footer-sub">Built with Python, Streamlit and financial analytics</div>
        <div class="app-footer-disclaimer">Educational use only — not financial advice.</div>
    </div>
    """, unsafe_allow_html=True)
