"""
ETF Holdings & Exposure -- Central Holdings Data Service (Round 1: single-ETF).

Architecture:

    Yahoo Finance quoteSummary "topHoldings" module   (source adapter)
                    v
    _normalize_yahoo_holdings()                        (normalized records)
                    v
    _cached_fetch_raw()  [@st.cache_data]               (cached snapshot)
                    v
    get_etf_holdings()                                  (status + fallback)
                    v
    pages/1_ETF_Analysis.py "Holdings & Exposure"        (UI)

No holdings weight in this module is ever hand-typed/fabricated -- every
HoldingRecord traces back to a live Yahoo Finance response, normalized as-is.
When the source has nothing for a ticker (rate-limited, network failure, or
the instrument type genuinely isn't covered by this source), the caller gets
an explicit status instead of invented numbers -- see HoldingsSnapshot.status.

This intentionally reuses yfinance's already-installed, already-authenticated
HTTP session (`yf.Ticker(...)._data.get_raw_json`) rather than bumping the
project's pinned yfinance version: the installed version (see requirements.txt)
predates yfinance's own public `Ticker.funds_data` convenience wrapper, but
the underlying quoteSummary "topHoldings" endpoint it wraps is reachable
through the same crumb/cookie-authenticated request path this app already
depends on for price downloads and `get_etf_info()` -- so this ships without
touching the shared yfinance pin (zero blast radius on price downloads,
Portfolio Optimizer, Investment Simulator, or Risk Analytics).

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

from src.etf_database import to_yahoo_symbol

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
    sector: Optional[str] = None       # per-holding sector, if the source discloses it
    country: Optional[str] = None      # per-holding country, if the source discloses it
    is_aggregate: bool = False         # True for a fund-level bucket (e.g. "CASH")
                                        # rather than one itemized security
    data_date: Optional[str] = None    # "as of" date for this holding's weight (ISO)
    source: str = "Yahoo Finance"
    source_url: Optional[str] = None


@dataclass
class HoldingsSnapshot:
    """The full holdings picture for one ETF at the time it was retrieved."""
    etf_ticker: str
    holdings: List[HoldingRecord] = field(default_factory=list)
    data_date: Optional[str] = None    # None only when status == "unavailable"/"not_supported"
    source: str = "Yahoo Finance"
    source_url: Optional[str] = None
    status: str = STATUS_UNAVAILABLE
    retrieved_at: Optional[str] = None  # when THIS call ran, regardless of data_date


# Process-local "last known good" cache, keyed by ETF ticker. Deliberately
# separate from the @st.cache_data-decorated fetch below: st.cache_data's
# TTL controls how often the SOURCE is hit, but a failed fetch inside that
# TTL window must still be able to fall back to the last successful
# snapshot (PRODUCT SPEC section 4/13) rather than showing "unavailable"
# for up to a full day whenever Yahoo has one bad request.
_LAST_GOOD_SNAPSHOT: Dict[str, HoldingsSnapshot] = {}


def _source_url(yahoo_symbol: str) -> str:
    return f"https://finance.yahoo.com/quote/{yahoo_symbol}/holdings"


def _fetch_yahoo_topholdings_raw(yahoo_symbol: str):
    """Low-level source adapter. Returns (result_dict_or_None, reached_source).

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
    """Cached wrapper around the source adapter -- the ONLY function that
    ever hits the network, and only once per yahoo_symbol per TTL window
    (PRODUCT SPEC section 14: never re-scrape on every Streamlit rerun)."""
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
                               data_date: str) -> List[HoldingRecord]:
    """Turn one Yahoo quoteSummary result into normalized HoldingRecords.
    Never invents a row that isn't backed by a field actually present in
    `result` -- a missing bucket is simply omitted, not zero-filled."""
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
            asset_type="Equity", weight=float(weight),
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


def get_etf_holdings(ticker: str) -> HoldingsSnapshot:
    """Public entry point: the single reusable holdings service every page
    should call (PRODUCT SPEC section 2: "central holdings data model").

    `ticker` is the platform's own display ticker (e.g. "0050" or "00981A"),
    never a raw Yahoo symbol -- management_style/return_type/asset_class are
    never consulted here, so an Active ETF or a leveraged/inverse ETF is
    fetched exactly the same way as a plain passive equity ETF (section 12).
    """
    yahoo_symbol = to_yahoo_symbol(ticker)
    retrieved_at = datetime.now().strftime("%Y-%m-%d")
    src_url = _source_url(yahoo_symbol)

    result, reached = _cached_fetch_raw(yahoo_symbol)
    holdings = _normalize_yahoo_holdings(ticker, yahoo_symbol, result, retrieved_at) if result else []

    if holdings:
        snapshot = HoldingsSnapshot(
            etf_ticker=ticker, holdings=holdings, data_date=retrieved_at,
            source="Yahoo Finance", source_url=src_url,
            status=STATUS_UPDATED, retrieved_at=retrieved_at,
        )
        _LAST_GOOD_SNAPSHOT[ticker] = snapshot
        return snapshot

    cached = _LAST_GOOD_SNAPSHOT.get(ticker)
    if cached is not None:
        return HoldingsSnapshot(
            etf_ticker=ticker, holdings=cached.holdings, data_date=cached.data_date,
            source=cached.source, source_url=cached.source_url,
            status=STATUS_CACHED, retrieved_at=retrieved_at,
        )

    status = STATUS_NOT_SUPPORTED if reached else STATUS_UNAVAILABLE
    return HoldingsSnapshot(
        etf_ticker=ticker, holdings=[], data_date=None,
        source="Yahoo Finance", source_url=src_url,
        status=status, retrieved_at=retrieved_at,
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
