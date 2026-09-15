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
from src.etf_database import get_countries, get_tickers_by_country, get_etf
from src.data_loader import DEFAULT_ETFS
from src.financial_metrics import ACTIVE_POSITION_TOLERANCE
import src.openai_service as _openai_service

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


_COUNTRY_SHORT_CODE = {"United States": "US", "Taiwan": "TW", "United Kingdom": "UK"}


def region_multiselect(default: list = None) -> list:
    """Portfolio Optimizer's true multi-select country control (Issue #24
    item 1). Unlike region_selector() above (single-select, shared GLOBAL
    state used by ETF Analysis / Risk Analytics / Machine Learning / AI
    Advisor), this is a dedicated, independent st.multiselect scoped ONLY
    to the Portfolio Optimizer -- picking Taiwan + United States here never
    affects any other page's market filter, and vice versa. There is no
    "All Regions" option: the user picks exactly the 1-3 countries they
    want, and the ETF universe below is the union of ONLY those countries.

    Backed by a plain "_selected_regions_shadow" session_state mirror for
    the same reason region_selector() needs one: render_sidebar_nav()'s
    language selector can trigger st.rerun() before this widget is reached
    on a given script pass, which would otherwise silently drop the
    widget's own keyed state back to `default`.
    """
    countries = get_countries()
    if default is None:
        default = [countries[0]] if countries else []
    labels = {c: t_country(c) for c in countries}

    if "_selected_regions_shadow" not in st.session_state:
        st.session_state["_selected_regions_shadow"] = default
    shadow = [c for c in st.session_state["_selected_regions_shadow"] if c in countries]

    selected = st.multiselect(
        t("field_select_countries"), countries, default=shadow,
        format_func=lambda c: labels.get(c, c), help=t("field_select_countries_help"),
        key="selected_regions",
    )
    st.session_state["_selected_regions_shadow"] = selected
    return selected


def region_etf_options_multi(selected_regions: list) -> list:
    """ETF ticker universe for the Portfolio Optimizer's multi-country
    picker -- the union of tickers from ONLY the countries in
    `selected_regions`, in a stable, deterministic order (countries in the
    order the user picked them; tickers within a country in
    get_tickers_by_country()'s own order). No "All Regions" branch: an
    empty `selected_regions` deliberately returns an empty universe so the
    page's own validation can show a clear "select at least one country"
    message rather than silently defaulting to everything."""
    seen = []
    for c in selected_regions:
        for tk in get_tickers_by_country(c):
            if tk not in seen:
                seen.append(tk)
    return seen


def multi_region_etf_multiselect(selected_regions: list, etf_options: list, label: str,
                                  help_text: str = None, n_default: int = 5) -> list:
    """ETF multiselect for the Portfolio Optimizer's multi-country picker
    (Issue #24 item 1). Deliberately NOT region_etf_multiselect() above:
    that helper keys its stored selection PER region string, so switching
    which region(s) are active swaps to a completely different stored
    list. Here we instead keep ONE persistent "master" selection
    (session_state["_selected_etfs_multi_master"]) that survives country
    changes -- removing a country prunes only the tickers that belonged
    EXCLUSIVELY to it (they're no longer in `etf_options`), while every
    other already-selected ticker from a still-active country stays
    selected, exactly as required.

    The widget's own `key` still changes with the active country
    combination (f"selected_etfs_portfolio_{'+'.join(sorted(...))}"),
    matching the safety pattern documented on
    _render_etf_universe_filters(): changing a multiselect's `options`
    between reruns while it keeps the SAME widget key can corrupt/reset
    its stored selection, so a new combination always gets a fresh widget
    key + the pruned "master" list as that key's one-time initial default;
    revisiting a previously-seen combination naturally restores Streamlit's
    own stored value for that key, which was always built by this same
    safe pruning logic to begin with.
    """
    master_key = "_selected_etfs_multi_master"
    if master_key not in st.session_state:
        st.session_state[master_key] = etf_options[:n_default]
    carry_over = [tk for tk in st.session_state[master_key] if tk in etf_options]
    if not carry_over and etf_options:
        carry_over = etf_options[:n_default]

    label_map = _build_etf_label_map(etf_options, include_market=len(selected_regions) > 1)

    if len(etf_options) > 10:
        matches = _render_etf_universe_filters(etf_options)
        if matches:
            _preview_n = 12
            preview = " · ".join(label_map.get(tk, tk) for tk in matches[:_preview_n])
            if len(matches) > _preview_n:
                preview += f" … (+{len(matches) - _preview_n})"
            st.caption(t("etf_filter_match_count", n=len(matches)))
            st.caption(preview)
        else:
            st.caption(t("etf_filter_no_matches"))

    widget_key = "selected_etfs_portfolio_" + "+".join(sorted(selected_regions))
    selected = st.multiselect(
        label, options=etf_options, default=carry_over, help=help_text,
        format_func=lambda tk: label_map.get(tk, tk), key=widget_key,
    )
    st.session_state[master_key] = selected
    return selected


def region_etf_options(selected_region: str, all_regions_label: str) -> list:
    """ETF ticker universe for `selected_region` -- the same "All Regions"
    / "United States" / single-country branching every page used
    identically before this was centralized here."""
    if selected_region == all_regions_label:
        return DEFAULT_ETFS + [tk for c in get_countries() for tk in get_tickers_by_country(c) if tk not in DEFAULT_ETFS]
    elif selected_region == "United States":
        return DEFAULT_ETFS
    return get_tickers_by_country(selected_region)


# ── Market-Aware Benchmark Selector ───────────────────────────────────────
# Fixes the confirmed "Taiwan + QQQ" benchmark bug: the benchmark selector
# used to be `st.selectbox(options=DEFAULT_ETFS, index=2)` on each page --
# a plain, un-keyed widget hardcoded to the US-only DEFAULT_ETFS list
# (DEFAULT_ETFS[2] == "QQQ"), completely independent of the selected
# region. It never re-evaluated when the region changed, so a Taiwan
# session kept showing/using QQQ as the benchmark even though QQQ isn't
# even in the Taiwan ETF universe. Root cause: no session-state key tied
# the benchmark to the region at all, unlike region_etf_multiselect()
# above (which HAS always been region-keyed).
_MARKET_DEFAULT_BENCHMARK = {
    "Taiwan": "0050",           # Taiwan broad-market index tracker
    "United States": "SPY",     # S&P 500 -- the most common US broad-market benchmark
    "United Kingdom": "VUKE",   # Vanguard FTSE 100 UCITS ETF -- broad UK-market benchmark
}
_GLOBAL_DEFAULT_BENCHMARK = "VT"  # used only for the "All Regions" scope


def market_default_benchmark(selected_region: str, all_regions_label: str) -> str:
    """The sensible starting benchmark for a given region. NOT a claim that
    every ETF in that region has exposure to that region (PRODUCT SPEC
    A3) -- this is only ever used as an initial/fallback suggestion; the
    user can always override it via region_benchmark_selector() below."""
    if selected_region == all_regions_label:
        return _GLOBAL_DEFAULT_BENCHMARK
    return _MARKET_DEFAULT_BENCHMARK.get(selected_region, _GLOBAL_DEFAULT_BENCHMARK)


def region_benchmark_selector(selected_region: str, etf_options: list, label: str,
                               help_text: str = None) -> str:
    """Shared, market-aware benchmark ETF selector (ETF Analysis + Risk
    Analytics both call this instead of each keeping its own hardcoded
    selectbox). Backed by st.session_state["_benchmark_shadow"][region], a
    PER-REGION shadow map so each region remembers its own last valid
    choice independently -- switching United States -> Taiwan -> United
    States restores your last United States benchmark, it does not carry
    QQQ (or any other now-invalid ticker) across into Taiwan, and it does
    not reset a manually-chosen Taiwan benchmark just because you looked at
    the US tab in between (PRODUCT SPEC A2/A4/B16).

    The widget itself is also keyed per-region
    (key=f"selected_benchmark_{region}"), so Streamlit treats it as a
    genuinely different widget instance per region -- the same mechanism
    that already made region_etf_multiselect() region-safe.
    """
    if "_benchmark_shadow" not in st.session_state:
        st.session_state["_benchmark_shadow"] = {}
    shadow = st.session_state["_benchmark_shadow"]

    current = shadow.get(selected_region)
    if current not in etf_options:
        default = market_default_benchmark(selected_region, t("field_all_regions"))
        current = default if default in etf_options else (etf_options[0] if etf_options else None)

    label_map = _build_etf_label_map(etf_options)
    index = etf_options.index(current) if current in etf_options else 0
    benchmark = st.selectbox(
        label, options=etf_options, index=index, help=help_text,
        format_func=lambda tk: label_map.get(tk, tk),
        key=f"selected_benchmark_{selected_region}",
    )
    shadow[selected_region] = benchmark
    st.session_state["_benchmark_shadow"] = shadow
    return benchmark


def _build_etf_label_map(tickers: list, include_market: bool = False) -> dict:
    """Precompute {ticker: "0050 — 元大台灣50"} for every ticker up front,
    in one pass -- this is what a multiselect's `format_func` should read
    from (a plain, side-effect-free dict lookup), rather than calling
    get_language()/t()/get_etf() itself on every option on every render.
    Confirmed by direct testing: a format_func that calls session-state-
    reading helpers (get_language(), t()) once PER OPTION corrupts this
    multiselect's stored selection on a later, unrelated rerun (observed
    even with the widget's own `options` held perfectly constant) --
    precomputing the labels once, here, and using a trivial dict.get as
    format_func avoids it entirely (and is strictly cheaper besides).

    Chinese name shown when the UI language is zh-TW and one is on record,
    else the English/generic fund name, with a compact [Leveraged]/
    [Inverse] tag for special-structure products (Taiwan ETF universe
    expansion section 9: these stay selectable, never blocked -- just
    visually identified so they're never mistaken for an ordinary
    long-term equity holding). Tickers not in the database (e.g. a
    free-text custom ticker) map to themselves.
    """
    lang = get_language()
    leveraged_tag = t("etf_badge_leveraged")
    inverse_tag = t("etf_badge_inverse")
    labels = {}
    for ticker in tickers:
        record = get_etf(ticker)
        if not record:
            labels[ticker] = ticker
            continue
        name = record.display_name_zh if (lang == "zh-TW" and record.display_name_zh) else record.name
        tag = ""
        if record.return_type == "Leveraged":
            tag = f" [{leveraged_tag}]"
        elif record.return_type == "Inverse":
            tag = f" [{inverse_tag}]"
        # Market suffix (Issue #24 item 1: "Make ETF labels clearly
        # identify ticker + name and, where useful, market") -- only added
        # when the caller is showing a cross-market universe (multiple
        # countries at once); a single-market list never needs it since
        # every option is obviously from the same market already.
        market_tag = ""
        if include_market:
            code = _COUNTRY_SHORT_CODE.get(record.country)
            if code:
                market_tag = f" ({code})"
        labels[ticker] = f"{ticker} — {name}{tag}{market_tag}" if name else f"{ticker}{tag}{market_tag}"
    return labels


# Internal category values (ETFRecord.category) -> the compact "ETF Type"
# filter buckets from the Taiwan ETF universe UX spec. "Multi-Asset" has no
# populated Taiwan records yet but stays a selectable, empty-result-safe
# filter option rather than being hidden; Real Estate / Money Market appear
# once the US bulk universe is loaded (see build_us_snapshot()'s keyword
# classification in scripts/refresh_etf_universe.py).
_ETF_TYPE_FILTER_MAP = {
    "Equity": {"Equity"},
    "Bond": {"Fixed Income"},
    "Multi-Asset": {"Multi-Asset"},
    "Commodity": {"Commodity"},
    "Real Estate": {"Real Estate"},
    "Money Market": {"Money Market"},
}


def _render_etf_search_box() -> str:
    return st.text_input(
        t("etf_filter_search_label"), key="tw_etf_filter_search",
        placeholder=t("etf_filter_search_placeholder"),
    )


def _render_etf_dropdown_filters(etf_options: list, records_by_ticker: dict, extra_filters: bool = False) -> dict:
    """Type / Management Style / Return Type / Issuer selectboxes (+
    Underlying Market / Trading Currency when `extra_filters=True` -- ETF
    Analysis workspace redesign only; Portfolio Optimizer / Risk Analytics /
    AI Advisor never pass this, so their filter row is exactly the 4
    columns it always was). Returns the chosen values; does not filter or
    render the search box itself -- see _render_etf_universe_filters()."""
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    _type_options = ["All", "Equity", "Bond", "Multi-Asset", "Commodity", "Real Estate", "Money Market", "Other"]
    _type_labels = {
        "All": t("etf_filter_all"), "Equity": t("etf_filter_type_equity"),
        "Bond": t("etf_filter_type_bond"), "Multi-Asset": t("etf_filter_type_multi_asset"),
        "Commodity": t("etf_filter_type_commodity"), "Real Estate": t("etf_filter_type_real_estate"),
        "Money Market": t("etf_filter_type_money_market"), "Other": t("etf_filter_type_other"),
    }
    with fcol1:
        etf_type = st.selectbox(
            t("etf_filter_type_label"), _type_options, format_func=lambda x: _type_labels.get(x, x),
            key="tw_etf_filter_type",
        )

    _style_options = ["All", "Passive", "Active"]
    _style_labels = {
        "All": t("etf_filter_all"), "Passive": t("etf_filter_style_passive"), "Active": t("etf_filter_style_active"),
    }
    with fcol2:
        mgmt_style = st.selectbox(
            t("etf_filter_style_label"), _style_options, format_func=lambda x: _style_labels.get(x, x),
            key="tw_etf_filter_style",
        )

    _return_options = ["All", "Standard", "Leveraged", "Inverse"]
    _return_labels = {
        "All": t("etf_filter_all"), "Standard": t("etf_filter_return_standard"),
        "Leveraged": t("etf_filter_return_leveraged"), "Inverse": t("etf_filter_return_inverse"),
    }
    with fcol3:
        return_type = st.selectbox(
            t("etf_filter_return_label"), _return_options, format_func=lambda x: _return_labels.get(x, x),
            key="tw_etf_filter_return",
        )

    _issuer_options = ["All"] + sorted({r.issuer for r in records_by_ticker.values() if r and r.issuer})
    _all_label = t("etf_filter_all")  # precomputed ONCE -- see _build_etf_label_map()'s
    # docstring: a format_func that calls t()/get_language() itself (fresh,
    # inside the lambda) rather than closing over an already-computed value
    # corrupts AppTest's widget-state reconciliation between reruns (it
    # calls the stored format_func again outside any active script/session
    # context, where get_language() falls back to the default language --
    # observed here as a ValueError when the resulting mismatched label
    # isn't found in the options list captured from the first render).
    with fcol4:
        issuer = st.selectbox(
            t("etf_filter_issuer_label"), _issuer_options,
            format_func=lambda x: _all_label if x == "All" else x,
            key="tw_etf_filter_issuer",
        )

    values = {"type": etf_type, "style": mgmt_style, "return": return_type, "issuer": issuer,
              "market": "All", "currency": "All"}
    if extra_filters:
        fcol5, fcol6 = st.columns(2)
        _market_options = ["All"] + sorted({r.underlying_market for r in records_by_ticker.values() if r and r.underlying_market})
        with fcol5:
            values["market"] = st.selectbox(
                t("etf_filter_market_label"), _market_options,
                format_func=lambda x: _all_label if x == "All" else x,
                key="tw_etf_filter_market",
            )
        _currency_options = ["All"] + sorted({r.currency for r in records_by_ticker.values() if r and r.currency})
        with fcol6:
            values["currency"] = st.selectbox(
                t("etf_filter_currency_label"), _currency_options,
                format_func=lambda x: _all_label if x == "All" else x,
                key="tw_etf_filter_currency",
            )
    return values


def _apply_etf_filters(etf_options: list, records_by_ticker: dict, filters: dict, search_query: str) -> list:
    _known_categories = {"Equity", "Fixed Income", "Multi-Asset", "Commodity", "Real Estate", "Money Market"}
    out = []
    for tk in etf_options:
        record = records_by_ticker.get(tk)
        category = record.category if record else None
        etf_type = filters["type"]
        if etf_type != "All":
            if etf_type == "Other":
                if category in _known_categories:
                    continue
            elif category not in _ETF_TYPE_FILTER_MAP.get(etf_type, set()):
                continue
        if filters["style"] != "All" and (record is None or record.management_style != filters["style"]):
            continue
        if filters["return"] != "All" and (record is None or record.return_type != filters["return"]):
            continue
        if filters["issuer"] != "All" and (record is None or record.issuer != filters["issuer"]):
            continue
        if filters.get("market", "All") != "All" and (record is None or record.underlying_market != filters["market"]):
            continue
        if filters.get("currency", "All") != "All" and (record is None or record.currency != filters["currency"]):
            continue
        if search_query:
            q = search_query.strip().lower()
            haystack = " ".join(filter(None, [
                tk, record.name if record else None,
                record.display_name_zh if record else None,
                record.issuer if record else None,
                record.isin if record else None,
            ])).lower()
            if q not in haystack:
                continue
        out.append(tk)
    return out


def _render_etf_universe_filters(etf_options: list, extra_filters: bool = False, collapse_dropdowns: bool = False) -> list:
    """Compact filter/search row (ETF universe UX section B9: Search / Asset
    Type / Management Style / Return Type / Issuer), shown for ANY region
    once its universe is large enough to need one -- originally built for
    Taiwan only (hence the "tw_etf_filter_*" widget keys, kept as-is for
    backward compatibility rather than churned for a cosmetic rename), now
    shared by every market since the US/Taiwan bulk-imported universes made
    "just scroll a raw multiselect" unusable everywhere, not just Taiwan.
    Returns the list of tickers matching the current criteria for a
    READ-ONLY discovery preview the caller renders below -- deliberately
    NEVER used to change the multiselect's own `options=`.

    Tested and confirmed unsafe: changing a multiselect's `options` between
    reruns while it keeps the same widget `key` can corrupt/reset its
    already-stored selection (observed under Streamlit's rerun model, not
    just an exception-avoidance concern). So the picker below always keeps
    the FULL, stable `etf_options` list as its `options` -- these filters
    only help the user find/preview what to pick; Streamlit's own
    multiselect dropdown already supports type-to-search over the
    (name-labeled) options for quick manual narrowing.

    `extra_filters`/`collapse_dropdowns` default to False, which renders
    EXACTLY the original 4-column dropdown row then the search box below it
    -- Portfolio Optimizer / Risk Analytics / AI Advisor never pass either,
    so their sidebar is pixel-identical to before. ETF Analysis (workspace
    redesign) passes both True: the search box stays always visible, and
    the (now 6-column) dropdown row collapses into an expander (PRODUCT
    SPEC section 11: "Keep always visible: ETF Search. Move into collapsed:
    Advanced ETF Filters").
    """
    records_by_ticker = {tk: get_etf(tk) for tk in etf_options}

    if collapse_dropdowns:
        search_query = _render_etf_search_box()
        with st.expander(t("etf_advanced_filters_label"), expanded=False):
            filters = _render_etf_dropdown_filters(etf_options, records_by_ticker, extra_filters)
    else:
        filters = _render_etf_dropdown_filters(etf_options, records_by_ticker, extra_filters)
        search_query = _render_etf_search_box()

    return _apply_etf_filters(etf_options, records_by_ticker, filters, search_query)


def region_etf_multiselect(selected_region: str, etf_options: list, label: str,
                            help_text: str = None, n_default: int = 3,
                            extra_filters: bool = False, collapse_filters: bool = False):
    """Shared ETF multiselect, scoped to the current global region, backed
    by st.session_state["selected_etfs_<region>"] -- picking ETFs on one
    page carries over to any other page calling this helper for the same
    region. Invalid tickers left over from a since-changed region are
    dropped automatically since they're filtered out of `etf_options`.
    `n_default` only applies the first time a given region is ever visited
    in this session; after that the shared shadow state takes over.

    For any region whose universe is large (now the normal case for
    Taiwan/United States/United Kingdom -- see src/etf_database.py, which
    loads a real bulk-imported master universe per market), a compact Type
    / Management Style / Return Type / Issuer / search filter row is shown
    above the picker (ETF universe UX section B9) as a discovery aid. The
    picker's own `options` always stay the FULL `etf_options` list,
    regardless of the filters -- see _render_etf_universe_filters()'s
    docstring for why narrowing a multiselect's `options` between reruns of
    the same widget `key` is unsafe. Options are labeled "TICKER — name"
    via format_func, so Streamlit's native in-dropdown type-to-search
    already lets the user narrow by typing too.

    `extra_filters`/`collapse_filters` default to False (unchanged
    behavior for every existing caller -- Portfolio Optimizer, Risk
    Analytics, AI Advisor). ETF Analysis (workspace redesign) passes both
    True for its collapsed "Advanced ETF Filters" sidebar panel.
    """
    if "_selected_etfs_shadow" not in st.session_state:
        st.session_state["_selected_etfs_shadow"] = {}
    _shadow_map = st.session_state["_selected_etfs_shadow"]
    _current_selection = [tk for tk in _shadow_map.get(selected_region, []) if tk in etf_options]
    _default = _current_selection if _current_selection else etf_options[:n_default]

    label_map = _build_etf_label_map(etf_options)

    if len(etf_options) > 10:
        matches = _render_etf_universe_filters(etf_options, extra_filters=extra_filters, collapse_dropdowns=collapse_filters)
        if matches:
            _preview_n = 12
            preview = " · ".join(label_map.get(tk, tk) for tk in matches[:_preview_n])
            if len(matches) > _preview_n:
                preview += f" … (+{len(matches) - _preview_n})"
            st.caption(t("etf_filter_match_count", n=len(matches)))
            st.caption(preview)
        else:
            st.caption(t("etf_filter_no_matches"))

    selected_etfs = st.multiselect(
        label, options=etf_options, default=_default, help=help_text,
        format_func=lambda tk: label_map.get(tk, tk), key=f"selected_etfs_{selected_region}",
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
    {"page": "pages/8_Market_Intelligence.py", "label_key": "nav_market_intelligence"},
    {"page": "pages/6_AI_Advisor.py", "label_key": "nav_ai_advisor"},
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
    # Positioning (Issue #20 section 1A): the visible product positioning
    # leads with ETF portfolio analytics / quantitative decision support,
    # not "AI" -- AI is one input (rule-based interpretation, optional
    # OpenAI-assisted narrative) alongside performance analysis, portfolio
    # optimization, risk analytics, simulation, and machine learning. The
    # brand/browser tab title (see st.set_page_config elsewhere) may still
    # say "AI ETF Portfolio Optimizer".
    if lang == "zh-TW":
        title = "ETF 投資組合分析與量化決策平台"
        subtitle = "整合績效分析、投資組合最佳化、風險分析、模擬、機器學習與 AI 輔助解讀。"
        btn_primary = "開始分析"
        btn_secondary = "探索功能"
        preview_title = "投資組合預覽（示範）"
        preview_metrics = ["預期報酬", "波動", "Sharpe"]
    else:
        title = "ETF Portfolio Analytics & Quantitative Decision Platform"
        subtitle = "Performance analysis, portfolio optimization, risk analytics, simulation, machine learning, and AI-assisted interpretation, in one platform."
        btn_primary = "Start Analysis"
        btn_secondary = "Explore Features"
        preview_title = "Portfolio Preview (Demo)"
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


# ── Setup vs Results Visual Hierarchy (Issue #24 items 2/3) ──────────────────
def setup_summary_bar(parts: list, title: str = None) -> None:
    """Compact, single-line 'Current setup' summary -- replaces a wide,
    equally-weighted multi-field settings strip that gave setup inputs the
    same visual prominence as the actual results. Deliberately plain,
    muted, low-emphasis text (not a card, no color) so the eye moves past
    it straight to results_hero() below, which carries the strong visual
    weight instead. `parts` is a list of short pre-formatted strings (e.g.
    "2 Markets", "5 ETFs", "USD", "Max Sharpe"), joined with a bullet.
    `title` is the already-translated eyebrow label (e.g. t("opt_setup_
    summary_title") / t("sim_projection_setup_title")) -- defaults to the
    Portfolio Optimizer's own "Portfolio Setup" wording for backward
    compatibility with its original single-page call site; every other
    page should pass its own page-appropriate title explicitly."""
    if title is None:
        title = t("opt_setup_summary_title")
    st.markdown(
        '<div class="setup-summary-bar">'
        f'<span class="setup-summary-eyebrow">{title}</span>'
        "&nbsp;&nbsp;" + "&nbsp;&nbsp;•&nbsp;&nbsp;".join(parts) +
        "</div>",
        unsafe_allow_html=True,
    )


def results_hero(title: str, subtitle: str = None) -> None:
    """Strong, unmistakable visual break between Setup and Results (Issue
    #24 item 3) -- a colored, elevated header block, deliberately much
    stronger than section_header() (used for sub-sections WITHIN results),
    so a first-time user can find "the answer" within a few seconds. Meant
    to be the very first thing rendered after a portfolio/analysis is
    successfully built."""
    sub_html = f'<div class="results-hero-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="results-hero"><div class="results-hero-eyebrow">{t("results_hero_eyebrow")}</div>'
        f'<div class="results-hero-title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def results_hero_metric(label: str, value: str, color: str = None) -> None:
    """The ONE visually dominant result number (Issue #24 visual-acceptance
    round item B1) -- rendered directly below results_hero(), large and
    isolated, so it reads as THE answer rather than one of several
    equal-weight KPI cards. Callers render their own smaller secondary
    metrics (e.g. a plain st.columns(3) row of metric_card_html()) below
    this, which are visually subordinate by simply being smaller, not by
    any hidden state this function manages."""
    color = color or "var(--primary)"
    st.markdown(
        '<div class="results-hero-metric">'
        f'<div class="results-hero-metric-label">{label}</div>'
        f'<div class="results-hero-metric-value" style="color:{color};">{value}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Hero Metric Panel + Insight Panel (Issue #29 visual-acceptance round) ───
# Generalizes the results_hero_metric() + secondary-row pattern Portfolio
# Optimizer already established (Issue #24 item B1) into one reusable
# helper, PLUS an optional right-side badge/metadata block aligned with the
# primary metric -- ETF Analysis needs the rating badge (trend + Quant
# Score/Signal Agreement) next to Annualized Return, which the plain
# results_hero_metric() had no slot for. Only meant to be adopted on other
# pages if the same one-protagonist-result shape genuinely fits there;
# results_hero_metric() itself is left as-is for existing callers.
def hero_metric_panel(primary_label: str, primary_value: str, primary_color: str = None,
                       badge: dict = None, secondary: list = None) -> None:
    """One primary metric (large, isolated) with an optional badge on the
    right and an optional row of smaller secondary metrics underneath.

    `badge`, if given, is a dict: {"emoji": str, "label": str, "color": str,
    "meta": [(meta_label, meta_value), ...]} -- `label`/`meta` values must
    already be translated by the caller (this function does no i18n lookup
    itself, matching every other src/ui.py renderer).

    `secondary`, if given, is a list of (label, value, color) tuples
    rendered via kpi_card() in one st.columns(len(secondary)) row -- the
    same small/subordinate styling used everywhere else on the page, never
    the large hero typography.
    """
    color = primary_color or "var(--primary)"
    if badge:
        left_col, right_col = st.columns([2, 1])
    else:
        left_col, right_col = st.container(), None

    with left_col:
        st.markdown(
            '<div class="hero-metric-panel-primary">'
            f'<div class="results-hero-metric-label">{primary_label}</div>'
            f'<div class="results-hero-metric-value" style="color:{color};">{primary_value}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if right_col is not None:
        badge_color = badge.get("color") or "var(--text-secondary)"
        emoji = badge.get("emoji", "")
        meta_html = "".join(
            f'<div class="hero-metric-panel-badge-meta-row"><span>{ml}</span><span>{mv}</span></div>'
            for ml, mv in badge.get("meta", [])
        )
        with right_col:
            st.markdown(
                '<div class="hero-metric-panel-badge">'
                f'<span class="badge" style="background:{badge_color}22;color:{badge_color};'
                f'border-color:{badge_color}55;">{emoji} {badge["label"]}</span>'
                f'{meta_html}'
                '</div>',
                unsafe_allow_html=True,
            )

    if secondary:
        cols = st.columns(len(secondary))
        for col, (s_label, s_value, s_color) in zip(cols, secondary):
            with col:
                st.markdown(kpi_card(s_label, s_value, color=s_color or COLORS["primary"]), unsafe_allow_html=True)


def insight_panel(title: str, bullets: list, accent_color: str = None, footer: str = None) -> None:
    """Flat, single-container replacement for nested Result -> Summary ->
    ... -> Insights card stacks: a colored left accent border, a short
    heading, 2-4 already-translated bullet-point strings, and one optional
    compact context/footer line. Meant to be the ONE insight block a
    workspace renders for its protagonist result, not a per-field card."""
    color = accent_color or "var(--primary)"
    bullets_html = "".join(f'<div class="insight-panel-bullet">&bull; {b}</div>' for b in bullets)
    footer_html = f'<div class="insight-panel-footer">{footer}</div>' if footer else ""
    st.markdown(
        f'<div class="insight-panel" style="border-left-color:{color};">'
        f'<div class="insight-panel-title">{title}</div>'
        f'{bullets_html}'
        f'{footer_html}'
        '</div>',
        unsafe_allow_html=True,
    )


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


# ── Rule-Based Market Sentiment Card (Market Intelligence) ───────────────────────
# Function name kept as ai_sentiment_card() for backward compatibility with
# existing call sites; the rendered title/labels are caller-supplied
# (see pages/8_Market_Intelligence.py, which passes "Rule-Based Market
# Sentiment" / "Sentiment Skew" -- Issue #20 section 8/10: this engine is
# 100% rule-based, no ML model or OpenAI call).
def ai_sentiment_card(mood_emoji: str, mood_label: str, mood_variant: str,
                       confidence_label: str, confidence: int,
                       drivers_label: str, drivers: list,
                       updated_label: str, updated_at: str) -> str:
    """
    Render the Rule-Based Market Sentiment card: a mood badge (emoji +
    Bullish/Neutral/Bearish), a sentiment-skew percentage, a "Top Drivers"
    list explaining what drove the assessment, and a last-updated
    timestamp. Built as a single-line HTML string for the same
    blank-line-safety reason as chart_card()/news_card().
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

    return df.style.map(_color, subset=columns)


# ── Chart / Table Explanations (Issue #22 section L) ─────────────────────────
def chart_caption(text: str) -> None:
    """Deterministic 'What this shows' caption for a chart/table, always
    correct with zero API calls -- the caller writes `text` itself from the
    chart's own known semantics (never inferred or fabricated here). Meant
    to be short: one or two sentences, not a paragraph."""
    st.caption(f"{t('chart_caption_prefix')} {text}")


def ai_interpret_button(cache_key: str, session_state, context_text: str) -> None:
    """Optional, user-triggered 'Ask AI to interpret' enrichment for a chart
    or table (Issue #22 section L). No API call happens just from rendering
    this button; a call only happens on click, and only once per distinct
    (cache_key, context_text, language) combination -- cached_generate()
    reuses the prior result on any rerun that doesn't change those (e.g.
    switching tabs), so repeated clicks or reruns with unchanged inputs
    never re-spend a call. The system prompt restricts the model to ONLY
    the metrics passed in `context_text` -- it must never invent a number,
    price, or piece of advice.

    When OpenAI isn't configured, still renders a disabled button (never
    clickable, never spends a call) plus a caption explaining why -- a
    silent no-op here would make the whole feature invisible to a user who
    never sees the button and has no way to know the capability exists or
    why it's unavailable.
    """
    if not _openai_service.is_configured():
        st.button(
            t("chart_ask_ai_interpret"), key=f"{cache_key}_btn_disabled", disabled=True,
            help=t("chart_ai_interpret_unavailable"),
        )
        st.caption(t("chart_ai_interpret_unavailable"))
        return
    if st.button(t("chart_ask_ai_interpret"), key=f"{cache_key}_btn"):
        system_prompt = (
            "You are explaining an already-computed chart or table to a "
            "retail investor. Only reference the metrics given below -- "
            "never invent a number, price, or piece of financial advice. "
            "Keep the explanation to 2-4 concise sentences."
        )
        fp = _openai_service.fingerprint(context_text, get_language())
        result = _openai_service.cached_generate(session_state, cache_key, fp, system_prompt, context_text, max_output_tokens=300)
        session_state[f"{cache_key}_last_result"] = result

    last_result = session_state.get(f"{cache_key}_last_result")
    if last_result and last_result.get("text"):
        st.info(last_result["text"])


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
