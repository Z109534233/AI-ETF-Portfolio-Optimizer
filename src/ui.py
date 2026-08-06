"""
UI Component Library
Reusable Streamlit UI helpers implementing the AI ETF Portfolio Optimizer
design system (see assets/style.css + src/theme.py). Used by app.py and all
pages to keep layout, typography and card styling consistent site-wide.
"""

import contextlib
import streamlit as st

from src.theme import COLORS, icon_svg
from src.i18n import t, language_selector, get_language

# ── Navigation ────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    {"page": "app.py", "label_key": "nav_home"},
    {"page": "pages/1_ETF_Analysis.py", "label_key": "nav_etf_analysis"},
    {"page": "pages/2_Portfolio_Optimizer.py", "label_key": "nav_portfolio_optimizer"},
    {"page": "pages/3_Investment_Simulator.py", "label_key": "nav_investment_simulator"},
    {"page": "pages/4_Risk_Analytics.py", "label_key": "nav_risk_analytics"},
    {"page": "pages/5_Machine_Learning.py", "label_key": "nav_machine_learning"},
    {"page": "pages/6_AI_Advisor.py", "label_key": "nav_ai_advisor"},
    {"page": "pages/8_Market_Intelligence.py", "label_key": "nav_market_intelligence", "icon": "📰"},
    {"page": "pages/7_Portfolio_History.py", "label_key": "nav_portfolio_history"},
]


def render_sidebar_nav() -> None:
    """Render the language switcher, branded product header, and primary
    navigation list. The currently active page is highlighted automatically
    by Streamlit (st.page_link sets aria-current="page"), styled via
    assets/style.css.
    """
    language_selector()

    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-mark">AI</div>
        <div class="sidebar-brand-text">
            <div class="sidebar-brand-name">{t("sidebar_brand_name")}</div>
            <div class="sidebar-brand-sub">{t("sidebar_brand_sub")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-nav-label">{t("nav_section_label")}</div>', unsafe_allow_html=True)
    for item in NAV_ITEMS:
        st.page_link(item["page"], label=t(item["label_key"]), icon=item.get("icon"))


def render_sidebar_footer() -> None:
    """Render the pinned-to-bottom sidebar footer. Call last inside `with st.sidebar:`."""
    st.markdown(f"""
    <div class="sidebar-footer">
        <div class="sidebar-footer-badge">{t("sidebar_footer_badge")}</div>
        <div class="sidebar-footer-text">{t("sidebar_footer_text")}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Section (Home page) ───────────────────────────────────────────────────
def hero_section() -> None:
    """
    Apple/Stripe/Bloomberg-style hero, two columns: left is a plain-
    language title + natural product description + two CTAs; right is a
    static, non-functional "Dashboard Preview" mockup (placeholder numbers,
    not wired to any real computation) so the hero is never visually
    empty on one side and communicates what the platform does at a glance.
    Shorter than the previous single-column version (no more large
    multi-line tagline, tighter padding) so Platform Statistics can still
    land in the first viewport. Bilingual strings are written out directly
    here (via get_language()) rather than added as new src/i18n.py keys,
    to keep this change to ui.py + style.css only.
    """
    lang = get_language()
    if lang == "zh-TW":
        title = "AI ETF Portfolio Optimizer"
        subtitle = "利用 AI 協助投資人分析 ETF、建立最佳投資組合，並掌握最新市場動態"
        btn_primary = "開始分析"
        btn_secondary = "探索功能"
        preview_title = "投資組合預覽"
        preview_metrics = ["預期報酬", "波動", "Sharpe"]
    else:
        title = "AI ETF Portfolio Optimizer"
        subtitle = "We use AI to help investors analyze ETFs, build optimal portfolios, and stay on top of the latest market trends"
        btn_primary = "Start Analysis"
        btn_secondary = "Explore Features"
        preview_title = "Portfolio Preview"
        preview_metrics = ["Expected Return", "Volatility", "Sharpe"]

    holdings = [("VOO", "40%"), ("QQQ", "35%"), ("0050", "25%")]
    metric_values = ["12.8%", "14.5%", "1.31"]

    with st.container(border=True):
        st.markdown('<div class="hero-marker"></div>', unsafe_allow_html=True)
        col_left, col_right = st.columns([1.15, 1], gap="large")

        with col_left:
            st.markdown(
                f'<h1 class="hero-title-new">{title}</h1>'
                f'<p class="hero-subtitle-new">{subtitle}</p>',
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 1])
            with btn_col1:
                st.markdown('<div class="hero-cta-row">', unsafe_allow_html=True)
                if st.button(btn_primary, type="primary", use_container_width=True, key="hero_start_analysis"):
                    st.switch_page("pages/1_ETF_Analysis.py")
                st.markdown('</div>', unsafe_allow_html=True)
            with btn_col2:
                st.markdown(
                    f'<div class="hero-cta-secondary-link">{btn_secondary}</div>',
                    unsafe_allow_html=True,
                )

        with col_right:
            holdings_html = "".join(
                f'<div class="hero-preview-row"><span class="hero-preview-row-label">{ticker}</span>'
                f'<span class="hero-preview-row-value">{weight}</span></div>'
                for ticker, weight in holdings
            )
            metrics_html = "".join(
                f'<div class="hero-preview-metric"><div class="hero-preview-metric-label">{label}</div>'
                f'<div class="hero-preview-metric-value">{value}</div></div>'
                for label, value in zip(preview_metrics, metric_values)
            )
            st.markdown(
                '<div class="hero-preview-card">'
                f'<div class="hero-preview-caption">{preview_title}</div>'
                f'{holdings_html}'
                '<div class="hero-preview-metrics">' + metrics_html + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ── Section / Page Headers ──────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = None) -> None:
    # Built as a single-line string deliberately: when subtitle is None,
    # sub_html is "" and, if placed on its own line inside a multi-line
    # HTML block, that line becomes blank. Streamlit's markdown renderer
    # treats a blank line as the end of a raw-HTML block, so everything
    # after it (the closing </div>, etc.) gets re-parsed as plain markdown
    # text instead of HTML and is displayed literally on the page. Keeping
    # the whole block on one line makes that impossible regardless of
    # which optional pieces are present.
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    html = f'<div class="section-header"><div class="section-title">{title}</div>{sub_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def badge(text: str, variant: str = "neutral") -> str:
    """Return an inline badge <span> for composing into other HTML blocks."""
    return f'<span class="badge badge-{variant}">{text}</span>'


# ── KPI Cards ───────────────────────────────────────────────────────────────────
_LABEL_ICON_MAP = [
    (("return", "growth", "gain", "報酬", "成長", "收益"), "trending-up"),
    (("drawdown", "loss", "回撤", "虧損"), "trending-down"),
    (("volatility", "risk", "std", "波動", "風險"), "activity"),
    (("sharpe", "sortino", "calmar", "ratio", "score", "比率", "分數"), "target"),
    (("value", "$", "amount", "invest", "價值", "金額", "投資"), "dollar"),
    (("diversif", "holdings", "assets", "分散", "持股"), "layers"),
    (("allocation", "weight", "配置", "權重"), "pie-chart"),
    (("var", "cvar", "風險值"), "shield"),
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
        # Built as a single-line string deliberately: when subtitle/tag are
        # None, sub_html/tag_html are "" and, if placed on their own line
        # inside a multi-line HTML block, that line becomes blank.
        # Streamlit's markdown renderer treats a blank line as the end of
        # a raw-HTML block, so everything after it (the closing </div>,
        # the tag <span>, etc.) gets re-parsed as plain markdown text
        # instead of HTML and is displayed literally on the page -- this
        # was the exact cause of "</div>" and the badge <span> showing up
        # as raw text instead of rendering. Keeping the whole header on
        # one line makes that impossible regardless of which optional
        # pieces are present.
        header_html = (
            '<div class="chart-card-header">'
            f'<div><div class="chart-card-title">{title}</div>{sub_html}</div>'
            f'{tag_html}'
            '</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)
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


# ── Process Flow (How It Works) ──────────────────────────────────────────────────
def process_flow(steps: list) -> None:
    """Render a horizontal numbered step timeline. `steps` is a list of label
    strings. Built as a single-line HTML string (no embedded newlines) to
    avoid the blank-line-terminates-raw-HTML-block markdown rendering bug.
    """
    parts = []
    for i, label in enumerate(steps, start=1):
        parts.append(
            f'<div class="process-step">'
            f'<div class="process-step-number">{i}</div>'
            f'<div class="process-step-title">{label}</div>'
            f'</div>'
        )
        if i < len(steps):
            parts.append('<div class="process-arrow">&#8594;</div>')
    html = '<div class="process-flow">' + "".join(parts) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Question Grid (Common Investment Questions) ──────────────────────────────────
def question_grid(questions: list, conclusion: str = None) -> None:
    """Render a grid of question chips with an optional concluding statement
    below. Built as single-line HTML strings to avoid the blank-line
    raw-HTML-termination bug.
    """
    help_icon = icon_svg("help-circle", 16, COLORS["primary"])
    items = "".join(f'<div class="question-item">{help_icon}<span>{q}</span></div>' for q in questions)
    st.markdown(f'<div class="question-grid">{items}</div>', unsafe_allow_html=True)
    if conclusion:
        st.markdown(f'<div class="question-conclusion">{conclusion}</div>', unsafe_allow_html=True)


# ── News Card (Market Intelligence) ──────────────────────────────────────────────
def news_card(title: str, time_str: str, source: str, impact_label: str,
               impact_variant: str = "neutral", url: str = None) -> str:
    """Render a single breaking-news card: title (optionally linked), publish
    time, source, and a market-impact badge with an outbound link icon.
    Built as one HTML string (no embedded blank lines) to avoid the
    blank-line raw-HTML-termination bug documented on chart_card()/section_header().
    """
    title_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
    clock_icon = icon_svg("clock", 13, COLORS["text_muted"])
    link_html = (
        f'<a class="news-card-link" href="{url}" target="_blank" rel="noopener noreferrer">'
        f'{icon_svg("external-link", 14, COLORS["primary"])}</a>'
    ) if url else ""
    return (
        '<div class="news-card">'
        f'<div class="news-card-title">{title_html}</div>'
        f'<div class="news-card-meta">{clock_icon}<span>{time_str}</span><span>&middot;</span><span>{source}</span></div>'
        '<div class="news-card-footer">'
        f'<span class="badge badge-{impact_variant}">{impact_label}</span>'
        f'{link_html}'
        '</div>'
        '</div>'
    )


# ── Status Card (Affected ETFs / Market Intelligence) ───────────────────────────
def status_card(ticker: str, sector: str, status_label: str, status_variant: str = "neutral",
                 stars_html: str = None) -> str:
    """Render a compact status card: a ticker, its sector/category, an
    impact badge (e.g. Positive/Negative/Neutral), and an optional star
    rating. Built as a single-line HTML string for the same blank-line-
    safety reason as news_card().
    """
    stars_block = f'<div class="status-card-stars">{stars_html}</div>' if stars_html else ""
    return (
        '<div class="status-card">'
        '<div class="status-card-top">'
        f'<div class="status-card-ticker">{ticker}</div>'
        f'<span class="badge badge-{status_variant}">{status_label}</span>'
        '</div>'
        f'<div class="status-card-sector">{sector}</div>'
        f'{stars_block}'
        '</div>'
    )


# ── Star Rating ───────────────────────────────────────────────────────────────
def star_rating_html(stars: int, max_stars: int = 5) -> str:
    """Render a colored star rating (filled amber stars + muted empty stars)."""
    stars = max(0, min(stars, max_stars))
    filled = f'<span class="star-filled">{"★" * stars}</span>' if stars else ""
    empty = f'<span class="star-empty">{"☆" * (max_stars - stars)}</span>' if stars < max_stars else ""
    return f'<span class="star-rating">{filled}{empty}</span>'


# ── Market Impact Card (Affected Markets, Market Intelligence) ──────────────────
def market_impact_card(market: str, impact_level_caption: str, stars_html: str, impact_label: str,
                        affected_by_caption: str = None, affected_by: list = None) -> str:
    """
    Render an "Affected Markets" card: market name, an "Impact Level"
    caption, a star rating, a plain-text impact label (e.g. "High Impact"
    -- never just bare stars), and an optional "Affected by:" list of the
    headlines driving that rating. Built as a single-line HTML string for
    the same blank-line-safety reason as chart_card()/news_card().
    """
    affected_html = ""
    if affected_by:
        items = "".join(f'<div class="affected-by-item">{title}</div>' for title in affected_by)
        affected_html = (
            f'<div class="affected-by-caption">{affected_by_caption}</div>'
            f'<div class="affected-by-list">{items}</div>'
        )
    return (
        '<div class="status-card market-impact-card">'
        f'<div class="status-card-ticker">{market}</div>'
        f'<div class="status-card-sector">{impact_level_caption}</div>'
        f'<div class="status-card-stars">{stars_html}</div>'
        f'<div class="market-impact-label">{impact_label}</div>'
        f'{affected_html}'
        '</div>'
    )


# ── AI Market Sentiment Card (Market Intelligence) ───────────────────────────────
def ai_sentiment_card(mood_emoji: str, mood_label: str, mood_variant: str,
                       confidence_label: str, confidence: int,
                       drivers_label: str, drivers: list,
                       updated_label: str, updated_at: str) -> str:
    """
    Render the "AI Market Sentiment" card: a mood badge (emoji + Bullish/
    Neutral/Bearish), a confidence percentage, a "Top Drivers" list
    explaining what drove the assessment, and a last-updated timestamp.
    Built as a single-line HTML string for the same blank-line-safety
    reason as chart_card()/news_card().
    """
    drivers_html = ""
    if drivers:
        items = "".join(f'<div class="affected-by-item">{title}</div>' for title in drivers)
        drivers_html = (
            f'<div class="affected-by-caption">{drivers_label}</div>'
            f'<div class="affected-by-list">{items}</div>'
        )
    return (
        '<div class="status-card ai-sentiment-card">'
        f'<span class="badge badge-{mood_variant} ai-sentiment-mood">{mood_emoji} {mood_label}</span>'
        '<div class="ai-sentiment-confidence-row">'
        f'<span class="status-card-sector">{confidence_label}</span>'
        f'<span class="ai-sentiment-confidence-value">{confidence}%</span>'
        '</div>'
        f'{drivers_html}'
        f'<div class="ai-sentiment-updated">{updated_label}: {updated_at}</div>'
        '</div>'
    )


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
    st.markdown(f"""
    <div class="app-footer">
        <div class="app-footer-brand">{t("footer_brand")}</div>
        <div class="app-footer-tagline">{t("footer_tagline")}</div>
        <div class="app-footer-disclaimer">{t("footer_disclaimer")}</div>
        <div class="app-footer-sub">{t("footer_built_with")}</div>
    </div>
    """, unsafe_allow_html=True)
