"""
UI Component Library
Reusable Streamlit UI helpers implementing the AI ETF Portfolio Optimizer
design system (see assets/style.css + src/theme.py). Used by app.py and all
pages to keep layout, typography and card styling consistent site-wide.
"""

import contextlib
import streamlit as st

from src.theme import COLORS, icon_svg
from src.i18n import t, t_country, t_opt_method, language_selector, get_language
from src.etf_database import get_countries, get_tickers_by_country
from src.data_loader import DEFAULT_ETFS
from src.financial_metrics import ACTIVE_POSITION_TOLERANCE

# ── Global Market / Region Selector ─────────────────────────────────────────
# Shared by every page that lets the user scope ETFs to a market (currently
# ETF Analysis, Risk Analytics, Machine Learning, AI Advisor). A single
# canonical st.session_state["selected_region"] is the source of truth --
# picking a market on one page is immediately reflected on every other page
# that calls region_selector(), and survives reruns from any other widget
# (tabs, chart type, checkboxes, language switch) since none of them touch
# this key.
def region_selector(default_index: int = 1):
    """Render the shared region/market selectbox. Returns
    (selected_region, ALL_REGIONS_LABEL).

    "_selected_region_shadow" is a plain (non-widget) session_state entry
    that mirrors the widget's value after every render. It's needed on top
    of key="selected_region" alone because Streamlit can drop a widget's
    own keyed state if something earlier in the same script run -- the
    language selector inside render_sidebar_nav(), called first thing on
    every page -- triggers st.rerun() before this widget is reached on that
    particular pass. A plain session_state entry isn't tied to widget
    instantiation, so it survives that and reseeds the widget on the next
    run instead of silently falling back to index=1.
    """
    ALL_REGIONS_LABEL = t("field_all_regions")
    region_options = [ALL_REGIONS_LABEL] + get_countries()
    region_labels = {c: t_country(c) for c in get_countries()}

    if "_selected_region_shadow" not in st.session_state:
        st.session_state["_selected_region_shadow"] = region_options[default_index]
    _shadow = st.session_state["_selected_region_shadow"]
    _index = region_options.index(_shadow) if _shadow in region_options else default_index

    selected_region = st.selectbox(
        t("field_select_region"), region_options, index=_index,
        format_func=lambda x: ALL_REGIONS_LABEL if x == ALL_REGIONS_LABEL else region_labels.get(x, x),
        key="selected_region",
    )
    st.session_state["_selected_region_shadow"] = selected_region
    return selected_region, ALL_REGIONS_LABEL


def region_etf_options(selected_region: str, all_regions_label: str) -> list:
    """ETF ticker universe for `selected_region` -- the same "All Regions"
    / "United States" / single-country branching every page used
    identically before this was centralized here."""
    if selected_region == all_regions_label:
        return DEFAULT_ETFS + [tk for c in get_countries() for tk in get_tickers_by_country(c) if tk not in DEFAULT_ETFS]
    elif selected_region == "United States":
        return DEFAULT_ETFS
    return get_tickers_by_country(selected_region)


def region_etf_multiselect(selected_region: str, etf_options: list, label: str,
                            help_text: str = None, n_default: int = 3):
    """Shared ETF multiselect, scoped to the current global region, backed
    by st.session_state["selected_etfs_<region>"] -- picking ETFs on one
    page carries over to any other page calling this helper for the same
    region. Invalid tickers left over from a since-changed region are
    dropped automatically since they're filtered out of `etf_options`.
    `n_default` only applies the first time a given region is ever visited
    in this session; after that the shared shadow state takes over.
    """
    if "_selected_etfs_shadow" not in st.session_state:
        st.session_state["_selected_etfs_shadow"] = {}
    _shadow_map = st.session_state["_selected_etfs_shadow"]
    _default = [tk for tk in _shadow_map.get(selected_region, []) if tk in etf_options]
    if not _default:
        _default = etf_options[:n_default]

    selected_etfs = st.multiselect(
        label, options=etf_options, default=_default, help=help_text,
        key=f"selected_etfs_{selected_region}",
    )
    _shadow_map[selected_region] = selected_etfs
    st.session_state["_selected_etfs_shadow"] = _shadow_map
    return selected_etfs


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


def render_current_portfolio_handoff(empty_title: str, empty_description: str,
                                      max_holdings: int = 5) -> bool:
    """Compact 'Current Portfolio' preview for pages that receive a
    portfolio built in Portfolio Optimizer, via the ONE canonical
    st.session_state["current_portfolio"] object (Round 2B-4).

    Shows strategy / top holdings / investment amount when a current
    portfolio exists; otherwise renders the shared empty_state() with the
    caller-supplied text. This is a proof-of-handoff preview only -- it
    never calls st.stop(), so the calling page's own existing controls and
    logic keep working standalone regardless of whether a portfolio was
    ever built in Portfolio Optimizer.

    Returns True if a portfolio was found and previewed, False if the
    empty state was shown.
    """
    portfolio = st.session_state.get("current_portfolio")
    if not portfolio:
        empty_state(empty_title, empty_description, icon="layers")
        return False

    # Active holdings shown prominently (largest first); zero-weight
    # selected ETFs (e.g. Maximum Sharpe pinning some tickers to 0%) are
    # summarized as a single de-emphasized count rather than listed
    # individually, so a long tail of "0.00%" entries never dominates this
    # compact preview. They stay part of the canonical portfolio -- never
    # dropped, just not enumerated here.
    weights = portfolio.get("weights") or {}
    sorted_holdings = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    active_holdings = [(tk, w) for tk, w in sorted_holdings if w > ACTIVE_POSITION_TOLERANCE]
    zero_holdings = [(tk, w) for tk, w in sorted_holdings if w <= ACTIVE_POSITION_TOLERANCE]

    shown = active_holdings[:max_holdings]
    holdings_text = " &middot; ".join(f"{tk} {w:.2%}" for tk, w in shown)
    remaining_active = len(active_holdings) - len(shown)
    if remaining_active > 0:
        holdings_text += f" &middot; +{remaining_active}"
    if zero_holdings:
        holdings_text += (
            f' <span style="color:{COLORS["text_muted"]};font-weight:400;">'
            f'({t("handoff_zero_weight_suffix", count=len(zero_holdings))})</span>'
        )

    strategy_label = t_opt_method(portfolio.get("strategy", ""))
    amount = portfolio.get("investment_amount")
    amount_text = f"${amount:,.0f}" if amount is not None else "—"

    st.markdown(f"""
    <div style="background:{COLORS['surface']};border-left:3px solid {COLORS['primary']};
                border-radius:var(--radius-lg);padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:11px;font-weight:700;color:{COLORS['primary']};
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">
            {t('handoff_current_portfolio_title')}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:4px 28px;font-size:13px;color:{COLORS['text_secondary']};">
            <div><b style="color:{COLORS['text']};">{t('handoff_strategy_label')}:</b> {strategy_label}</div>
            <div><b style="color:{COLORS['text']};">{t('handoff_holdings_label')}:</b> {holdings_text}</div>
            <div><b style="color:{COLORS['text']};">{t('handoff_investment_amount_label')}:</b> {amount_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return True


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
