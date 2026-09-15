"""
ETF Holdings & Exposure -- Central Holdings Data Service (Round 1: single-ETF).

Architecture:

    yfinance Ticker.funds_data (public API)   quoteSummary "topHoldings"
    -- top_holdings / asset_classes           (private fallback adapter)
                    \\                          /
                     v                        v
              _normalize_public_holdings()  _normalize_yahoo_holdings()
                              \\              /
                               v            v
                        get_etf_holdings()  (tries public first, falls
                                             back to the private adapter,
                                             then last-known-good, then an
                                             honest unavailable status)
                                v
                pages/1_ETF_Analysis.py "Holdings & Exposure"  (UI)

No holdings weight in this module is ever hand-typed/fabricated -- every
HoldingRecord traces back to a live Yahoo Finance response (via one of the
two adapters below), normalized as-is. When neither source has anything for
a ticker (rate-limited, network failure, or the instrument type genuinely
isn't covered), the caller gets an explicit status instead of invented
numbers -- see HoldingsSnapshot.status.

Two source adapters, tried in order:

1. `_fetch_public_funds_holdings_raw()` -- yfinance's own supported
   `Ticker(...).funds_data` wrapper (`top_holdings` DataFrame +
   `asset_classes` dict). This is the PRIMARY adapter: it's a maintained,
   public yfinance API rather than a private request path, so it's the one
   most likely to keep working as Yahoo's backend changes.
2. `_fetch_yahoo_topholdings_raw()` -- the original direct quoteSummary
   "topHoldings" call via `yf.Ticker(...)._data.get_raw_json`. Kept only as
   a FALLBACK for when the public wrapper raises or returns nothing (e.g. an
   older yfinance release, or a response shape the public wrapper doesn't
   parse) -- never the first thing tried.

If one adapter fails but the other succeeds, the real data from whichever
one worked is shown; only when BOTH fail does this fall through to the
last-known-good cache, and only when that's also empty does it show the
honest "unavailable" state (see HoldingsSnapshot.status).

Future database compatibility (see PRODUCT SPEC section 15 -- not built this
round): every HoldingRecord field maps 1:1 onto a future `etf_holdings_snapshots`
table: etf_ticker, holding_ticker, holding_name, weight, asset_type, sector,
country, as_of_date (-> data_date), source, retrieved_at.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import yfinance as yf

from src.etf_database import get_etf, to_yahoo_symbol

# Valid asset_type values a HoldingRecord can carry (PRODUCT SPEC section
# 14: never assume every holding is a stock). Not every value is currently
# reachable through the Yahoo adapter below -- ETF-of-ETFs / Futures /
# Derivative require source data this adapter doesn't get from Yahoo's
# topHoldings module -- but the schema and UI are ready for a future source
# adapter that does disclose them.
VALID_ASSET_TYPES = ("Equity", "Bond", "Cash", "Futures", "ETF", "Derivative",
                      "Preferred", "Convertible", "Other")

# What an itemized (non-aggregate) holding row should default to when
# Yahoo's topHoldings module doesn't say (it never does -- it only lists a
# holding's symbol/name/weight, no per-row type). Using the ETF's OWN
# canonical category (from the real TWSE/TPEx/Nasdaq Trader-sourced master
# -- see src/etf_database.py) is real information, not a guess: a Fixed
# Income ETF's "top holdings" are bonds, not stocks, even though Yahoo's
# response shape is identical either way.
_DEFAULT_HOLDING_ASSET_TYPE_BY_CATEGORY = {
    "Fixed Income": "Bond",
    "Money Market": "Cash",
}

# Holdings composition changes far more slowly than daily prices (typical
# index-fund rebalances are monthly/quarterly) -- a 24h cache TTL avoids
# re-hitting the source on every Streamlit rerun/widget interaction while
# still refreshing daily, mirroring download_etf_data()'s ttl=3600 rationale
# in src/data_loader.py but tuned to how often this specific data changes.
_HOLDINGS_CACHE_TTL_SECONDS = 24 * 3600

# Status codes (PRODUCT SPEC section 13):
#   "updated"       -- A: fetched successfully this call
#   "cached"        -- B: source failed/unsupported now, but a prior
#                          successful snapshot exists and is shown instead
#   "unavailable"   -- C: source temporarily unreachable, no prior snapshot
#   "not_supported" -- D: source was reachable but has no holdings data for
#                          this instrument (e.g. not a fund Yahoo tracks
#                          holdings for)
STATUS_UPDATED = "updated"
STATUS_CACHED = "cached"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NOT_SUPPORTED = "not_supported"

# Aggregate, source-disclosed (never fabricated) fund-level composition
# buckets that are NOT part of Yahoo's itemized top-holdings list, so adding
# them never double-counts an itemized row. ("stockPosition" is deliberately
# excluded -- it would double-count the itemized equity holdings above it.)
_AGGREGATE_BUCKETS = [
    ("cashPosition", "CASH", "Cash & Cash Equivalents", "Cash"),
    ("bondPosition", "BOND_AGG", "Aggregate Bond Allocation", "Bond"),
    ("otherPosition", "OTHER_AGG", "Other Assets", "Other"),
    ("preferredPosition", "PREFERRED_AGG", "Preferred Securities", "Preferred"),
    ("convertiblePosition", "CONVERTIBLE_AGG", "Convertible Securities", "Convertible"),
]


@dataclass
class HoldingRecord:
    """One holding/position within one ETF, at one point in time.

    Conceptual schema (PRODUCT SPEC section 2 / 15): maps 1:1 onto a future
    `etf_holdings_snapshots` table row.
    """
    etf_ticker: str                    # the ETF this holding belongs to, e.g. "0050"
    holding_ticker: str                # the holding's own ticker, e.g. "2330"; a
                                        # sentinel like "CASH" for an aggregate bucket
                                        # that has no itemized security identifier
    holding_name: str                  # e.g. "台積電" / "Taiwan Semiconductor Mfg"
    asset_type: str                    # "Equity" / "Cash" / "Bond" / "Other" /
                                        # "Preferred" / "Convertible"
    weight: float                      # fraction of ETF NAV, e.g. 0.5686 for 56.86%
    quantity: Optional[float] = None   # shares/units held, if the source discloses it
    market_value: Optional[float] = None  # position value in the ETF's own currency, if disclosed
    currency: Optional[str] = None        # currency of `market_value`, if disclosed
    sector: Optional[str] = None       # per-holding sector, if the source discloses it
    country: Optional[str] = None      # per-holding country, if the source discloses it
    is_aggregate: bool = False         # True for a fund-level bucket (e.g. "CASH")
                                        # rather than one itemized security
    data_date: Optional[str] = None    # "as of" date for this holding's weight (ISO)
    source: str = "Yahoo Finance"
    source_url: Optional[str] = None


@dataclass
class HoldingsSnapshot:
    """The full holdings picture for one ETF at the time it was retrieved.

    `issuer`/`exchange`/`isin`/`listing_market`/`fund_group_id` are read
    straight from the ETF's CANONICAL master record (src/etf_database.py --
    ETF Holdings & Exposure round: "identify the ETF through the canonical
    master record", not just its bare ticker), so the UI can show who
    issues it, where it's listed, and (for a multi-currency LSE fund) which
    other tickers are the SAME underlying fund -- without this module
    duplicating that data itself.
    """
    etf_ticker: str
    holdings: List[HoldingRecord] = field(default_factory=list)
    data_date: Optional[str] = None    # None only when status == "unavailable"/"not_supported"
    source: str = "Yahoo Finance"
    source_url: Optional[str] = None
    status: str = STATUS_UNAVAILABLE
    retrieved_at: Optional[str] = None  # when THIS call ran, regardless of data_date
    issuer: Optional[str] = None
    exchange: Optional[str] = None
    isin: Optional[str] = None
    listing_market: Optional[str] = None
    fund_group_id: Optional[str] = None


# Process-local "last known good" cache, keyed by ETF ticker. Deliberately
# separate from the @st.cache_data-decorated fetch below: st.cache_data's
# TTL controls how often the SOURCE is hit, but a failed fetch inside that
# TTL window must still be able to fall back to the last successful
# snapshot (PRODUCT SPEC section 4/13) rather than showing "unavailable"
# for up to a full day whenever Yahoo has one bad request.
_LAST_GOOD_SNAPSHOT: Dict[str, HoldingsSnapshot] = {}


def _source_url(yahoo_symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{yahoo_symbol}/holdings"


def _fetch_public_funds_holdings_raw(yahoo_symbol: str):
    """PRIMARY source adapter -- yfinance's own supported `funds_data`
    wrapper. Returns (top_holdings_df_or_None, asset_classes_dict_or_None,
    reached_source).

    reached_source distinguishes "the request itself failed" (network error,
    rate limit, timeout, or this yfinance release doesn't expose
    `funds_data` at all -> Status C) from "Yahoo/yfinance answered but this
    instrument isn't a fund with holdings data" (Status D -- e.g. a single
    stock ticker) -- never raises.
    """
    try:
        funds_data = yf.Ticker(yahoo_symbol).funds_data
        if funds_data is None:
            return None, None, False
        top_holdings = getattr(funds_data, "top_holdings", None)
        asset_classes = getattr(funds_data, "asset_classes", None)
    except Exception:
        return None, None, False
    reached = (top_holdings is not None) or bool(asset_classes)
    return top_holdings, asset_classes, reached


@st.cache_data(ttl=_HOLDINGS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_fetch_public(yahoo_symbol: str):
    """Cached wrapper around the primary source adapter -- only hits the
    network once per yahoo_symbol per TTL window (PRODUCT SPEC section 14:
    never re-scrape on every Streamlit rerun)."""
    return _fetch_public_funds_holdings_raw(yahoo_symbol)


def _fetch_yahoo_topholdings_raw(yahoo_symbol: str):
    """FALLBACK source adapter, only used when the public `funds_data`
    adapter above returns nothing. Returns (result_dict_or_None,
    reached_source).

    reached_source distinguishes "the request itself failed" (network error,
    rate limit, timeout -> Status C) from "Yahoo answered but has nothing
    for this module/symbol" (Status D) -- never raises.
    """
    try:
        ticker_obj = yf.Ticker(yahoo_symbol)
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{yahoo_symbol}"
        raw = ticker_obj._data.get_raw_json(url, params={"modules": "topHoldings,fundProfile"})
    except Exception:
        return None, False

    result_list = (raw or {}).get("quoteSummary", {}).get("result")
    if not result_list:
        return None, True
    return result_list[0], True


@st.cache_data(ttl=_HOLDINGS_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_fetch_raw(yahoo_symbol: str):
    """Cached wrapper around the fallback source adapter -- only hits the
    network once per yahoo_symbol per TTL window (PRODUCT SPEC section 14:
    never re-scrape on every Streamlit rerun)."""
    return _fetch_yahoo_topholdings_raw(yahoo_symbol)


_HOLDING_SUFFIXES_TO_STRIP = (".TW", ".TWO")


def _strip_holding_suffix(symbol: str) -> str:
    """Constituent-stock symbols come back from Yahoo with the same market
    suffix convention as ETF yahoo_symbols (e.g. "2330.TW"). Display tickers
    everywhere else in this app are suffix-free (see rename_yahoo_columns()
    in src/etf_database.py for the equivalent ETF-level transform), so this
    strips the two Taiwan suffixes this app already knows about. Unknown/
    unsuffixed symbols (e.g. US constituents like "AAPL") pass through
    unchanged -- this never guesses at suffixes it hasn't confirmed."""
    for suffix in _HOLDING_SUFFIXES_TO_STRIP:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _normalize_yahoo_holdings(etf_ticker: str, yahoo_symbol: str, result: dict,
                               data_date: str, default_asset_type: str = "Equity") -> List[HoldingRecord]:
    """Turn one Yahoo quoteSummary result into normalized HoldingRecords.
    Never invents a row that isn't backed by a field actually present in
    `result` -- a missing bucket is simply omitted, not zero-filled.

    `default_asset_type` labels the itemized rows (Yahoo's topHoldings
    doesn't carry a per-row type at all -- see _DEFAULT_HOLDING_ASSET_TYPE_BY_CATEGORY's
    docstring for why this comes from the ETF's own canonical category
    rather than a blanket "Equity")."""
    src_url = _source_url(yahoo_symbol)
    top = (result or {}).get("topHoldings") or {}
    records: List[HoldingRecord] = []

    for row in (top.get("holdings") or []):
        symbol = row.get("symbol")
        pct = row.get("holdingPercent")
        weight = pct.get("raw") if isinstance(pct, dict) else pct
        if not symbol or weight is None:
            continue
        records.append(HoldingRecord(
            etf_ticker=etf_ticker, holding_ticker=_strip_holding_suffix(symbol),
            holding_name=row.get("holdingName") or symbol,
            asset_type=default_asset_type, weight=float(weight),
            is_aggregate=False, data_date=data_date,
            source="Yahoo Finance", source_url=src_url,
        ))

    for key, sentinel, label, asset_type in _AGGREGATE_BUCKETS:
        raw_val = top.get(key)
        pct = raw_val.get("raw") if isinstance(raw_val, dict) else raw_val
        if pct:
            records.append(HoldingRecord(
                etf_ticker=etf_ticker, holding_ticker=sentinel, holding_name=label,
                asset_type=asset_type, weight=float(pct),
                is_aggregate=True, data_date=data_date,
                source="Yahoo Finance", source_url=src_url,
            ))

    records.sort(key=lambda h: h.weight, reverse=True)
    return records


def _normalize_public_holdings(etf_ticker: str, yahoo_symbol: str, top_holdings_df, asset_classes: Optional[dict],
                                data_date: str, default_asset_type: str = "Equity") -> List[HoldingRecord]:
    """Turn yfinance's `funds_data.top_holdings` DataFrame + `asset_classes`
    dict into normalized HoldingRecords. Same never-invent-a-row contract as
    _normalize_yahoo_holdings() above -- a missing/empty frame or bucket is
    simply omitted, not zero-filled.

    `top_holdings_df` is indexed by holding symbol with "Name" and "Holding
    Percent" columns (fraction 0-1, confirmed against a live VOO response);
    `asset_classes` is a flat {"cashPosition": 0.006, "bondPosition": 0.0,
    ...} dict using the SAME bucket keys as `_AGGREGATE_BUCKETS` below, so it
    reuses that table directly rather than duplicating the bucket labels.
    """
    src_url = _source_url(yahoo_symbol)
    records: List[HoldingRecord] = []

    if top_holdings_df is not None and not top_holdings_df.empty:
        for symbol, row in top_holdings_df.iterrows():
            pct = row.get("Holding Percent")
            if symbol is None or pct is None or (isinstance(pct, float) and pct != pct):  # pct != pct -> NaN
                continue
            name = row.get("Name") or symbol
            records.append(HoldingRecord(
                etf_ticker=etf_ticker, holding_ticker=_strip_holding_suffix(str(symbol)),
                holding_name=str(name), asset_type=default_asset_type, weight=float(pct),
                is_aggregate=False, data_date=data_date,
                source="Yahoo Finance", source_url=src_url,
            ))

    for key, sentinel, label, asset_type in _AGGREGATE_BUCKETS:
        pct = (asset_classes or {}).get(key)
        if pct:
            records.append(HoldingRecord(
                etf_ticker=etf_ticker, holding_ticker=sentinel, holding_name=label,
                asset_type=asset_type, weight=float(pct),
                is_aggregate=True, data_date=data_date,
                source="Yahoo Finance", source_url=src_url,
            ))

    records.sort(key=lambda h: h.weight, reverse=True)
    return records


def get_etf_holdings(ticker: str) -> HoldingsSnapshot:
    """Public entry point: the single reusable holdings service every page
    should call (PRODUCT SPEC section 2/3: "identify the ETF through the
    canonical master record" -- ticker/exchange/issuer/ISIN/provider symbol/
    listing market all come from ETF_DATABASE via get_etf(), never a second
    independent lookup).

    `ticker` is the platform's own display ticker (e.g. "0050" or "00981A"),
    never a raw Yahoo symbol -- management_style/return_type/asset_class are
    never consulted to decide WHETHER to fetch (section 18: an Active ETF or
    a leveraged/inverse ETF is fetched exactly the same way as a plain
    passive equity ETF), but the ETF's `category` DOES inform what an
    itemized holding row defaults to (Bond vs Equity -- section 14).
    """
    record = get_etf(ticker)
    yahoo_symbol = record.yahoo_symbol if record else to_yahoo_symbol(ticker)
    default_asset_type = _DEFAULT_HOLDING_ASSET_TYPE_BY_CATEGORY.get(
        record.category, "Equity") if record else "Equity"
    retrieved_at = datetime.now().strftime("%Y-%m-%d")
    src_url = _source_url(yahoo_symbol)
    common = dict(
        issuer=record.issuer if record else None,
        exchange=record.exchange if record else None,
        isin=record.isin if record else None,
        listing_market=record.country if record else None,
        fund_group_id=record.fund_group_id if record else None,
    )

    # Primary adapter first (public, supported yfinance API); the private
    # quoteSummary adapter is only consulted as a fallback when the primary
    # one comes back empty -- see module docstring.
    pub_top, pub_asset_classes, reached_public = _cached_fetch_public(yahoo_symbol)
    holdings = (
        _normalize_public_holdings(ticker, yahoo_symbol, pub_top, pub_asset_classes, retrieved_at, default_asset_type)
        if (pub_top is not None or pub_asset_classes) else []
    )

    reached = reached_public
    if not holdings:
        result, reached_fallback = _cached_fetch_raw(yahoo_symbol)
        reached = reached_public or reached_fallback
        holdings = (
            _normalize_yahoo_holdings(ticker, yahoo_symbol, result, retrieved_at, default_asset_type)
            if result else []
        )

    if holdings:
        snapshot = HoldingsSnapshot(
            etf_ticker=ticker, holdings=holdings, data_date=retrieved_at,
            source="Yahoo Finance", source_url=src_url,
            status=STATUS_UPDATED, retrieved_at=retrieved_at, **common,
        )
        _LAST_GOOD_SNAPSHOT[ticker] = snapshot
        return snapshot

    cached = _LAST_GOOD_SNAPSHOT.get(ticker)
    if cached is not None:
        return HoldingsSnapshot(
            etf_ticker=ticker, holdings=cached.holdings, data_date=cached.data_date,
            source=cached.source, source_url=cached.source_url,
            status=STATUS_CACHED, retrieved_at=retrieved_at, **common,
        )

    status = STATUS_NOT_SUPPORTED if reached else STATUS_UNAVAILABLE
    return HoldingsSnapshot(
        etf_ticker=ticker, holdings=[], data_date=None,
        source="Yahoo Finance", source_url=src_url,
        status=status, retrieved_at=retrieved_at, **common,
    )


def itemized_holdings(snapshot: HoldingsSnapshot) -> List[HoldingRecord]:
    """The subset of a snapshot's holdings that are individually-identified
    securities (excludes aggregate cash/bond/other buckets) -- what
    "Top N Holdings" and concentration metrics should be computed over."""
    return [h for h in snapshot.holdings if not h.is_aggregate]


def total_disclosed_weight(snapshot: HoldingsSnapshot) -> float:
    """Sum of every disclosed row's weight (itemized + aggregate buckets).
    Deliberately does NOT claim this equals 100% of NAV -- most sources only
    disclose the top N holdings, so this is usually well under 100% and the
    UI must say so (PRODUCT SPEC section 10)."""
    return float(sum(h.weight for h in snapshot.holdings))


def search_holdings(snapshot: HoldingsSnapshot, query: str) -> List[HoldingRecord]:
    """Case-insensitive substring match on holding ticker or name."""
    q = (query or "").strip().lower()
    if not q:
        return list(snapshot.holdings)
    return [
        h for h in snapshot.holdings
        if q in h.holding_ticker.lower() or q in (h.holding_name or "").lower()
    ]
