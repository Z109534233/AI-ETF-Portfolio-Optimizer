"""
FX (Foreign Exchange) Conversion Module
========================================
Converts a wide, mixed-currency price DataFrame (e.g. US ETFs in USD,
Taiwan ETFs in TWD, UK ETFs in GBP) into a single common base currency, so
that downstream portfolio math (src/portfolio_optimizer.py's
run_optimization(), which internally does prices_df.pct_change()) can keep
operating on a plain wide price DataFrame completely unchanged -- it never
needs to know currencies were mixed in the first place.

This module is intentionally Streamlit-free and side-effect-light, mirroring
the separation of concerns already used by src/financial_metrics.py and
src/portfolio_optimizer.py (both pure/Streamlit-free; only src/data_loader.py
and page files own Streamlit concerns like st.cache_data/st.warning). If a
caller (e.g. a page) wants to cache get_fx_series()/convert_prices_to_base_
currency() results across reruns, it should wrap the call site itself with
st.cache_data, the same way pages already wrap src/data_loader.py calls --
this module does not import streamlit and never will.

Yahoo Finance FX ticker conventions relied on here
---------------------------------------------------
Yahoo Finance FX tickers follow the pattern "<FROM><TO>=X", quoting units of
<TO> per 1 <FROM>, EXCEPT that USD-quoted pairs conventionally drop the "USD"
base and are just "<TO>=X" (e.g. "TWD=X" means "USD/TWD", i.e. USD->TWD).
This module supports exactly the three currencies the product exposes as a
base-currency choice (TWD, USD, GBP), plus USD/GBP/TWD as native ETF
currencies, so only these direct pairs are ever needed:

    USD -> TWD : "TWD=X"      (direct; Yahoo's USD-base convention)
    GBP -> USD : "GBPUSD=X"   (direct; standard Yahoo FX pair)

There is no direct Yahoo Finance ticker for GBP -> TWD (Yahoo does not list
"GBPTWD=X"), so that conversion -- and any other currency pair that isn't
one of the two direct pairs above -- is COMPOSED as a cross-rate through USD,
which is the standard convention for cross-rates lacking a direct quote:

    GBP -> TWD : GBPUSD=X * TWD=X   (GBP->USD->TWD)
    TWD -> USD : 1 / (TWD=X)
    TWD -> GBP : (1 / TWD=X) * (1 / GBPUSD=X)   i.e. TWD->USD->GBP
    USD -> GBP : 1 / (GBPUSD=X)

_DIRECT_PAIRS below is the single source of truth for which (from, to) pairs
have a real Yahoo ticker; get_fx_series() and required_fx_pairs() both derive
their behavior from it so they can never drift out of sync with each other.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

FX_SOURCE_LABEL = "Yahoo Finance (yfinance) daily FX rates"

# The only direct (from_currency, to_currency) -> Yahoo ticker pairs this
# module knows about. Every other (from, to) combination among the
# currencies the product supports (USD, TWD, GBP) is composed as a cross
# through USD -- see _resolve_fx_path() below. Extending to a new currency
# only requires adding its direct-vs-USD pair here (if Yahoo has one); no
# other function needs to change.
_DIRECT_PAIRS: Dict[Tuple[str, str], str] = {
    ("USD", "TWD"): "TWD=X",
    ("GBP", "USD"): "GBPUSD=X",
}


def _invert_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    a, b = pair
    return (b, a)


def _resolve_fx_path(from_currency: str, to_currency: str) -> Optional[List[Tuple[str, str, bool]]]:
    """Return the sequence of legs needed to convert from_currency ->
    to_currency, as a list of (leg_from, leg_to, invert) tuples, where
    `invert` means "fetch the direct ticker for (leg_to, leg_from) and use
    1/rate" because Yahoo only lists the pair in the opposite direction.

    Returns None if from_currency/to_currency aren't a pair this module
    knows how to resolve (not one of the direct pairs, their inverses, or a
    USD-composed cross of the two direct pairs above). Pure/no network call.
    """
    if from_currency == to_currency:
        return []

    direct = (from_currency, to_currency)
    if direct in _DIRECT_PAIRS:
        return [(from_currency, to_currency, False)]

    inverse = _invert_pair(direct)
    if inverse in _DIRECT_PAIRS:
        return [(to_currency, from_currency, True)]

    # Cross through USD: from_currency -> USD -> to_currency. Only supported
    # when both legs individually resolve (each leg is itself either direct
    # or an inverse of a direct pair).
    if from_currency != "USD" and to_currency != "USD":
        leg1 = _resolve_fx_path(from_currency, "USD")
        leg2 = _resolve_fx_path("USD", to_currency)
        if leg1 is not None and leg2 is not None:
            return leg1 + leg2

    return None


def required_fx_pairs(currencies_present: set, base_currency: str) -> List[Tuple[str, str]]:
    """Pure, network-free helper for the UI layer: given the set of native
    currencies present in a selected portfolio and the chosen base currency,
    return the list of (from_currency, to_currency) pairs that will actually
    need to be fetched/composed, so a page can disclose this upfront (e.g.
    "FX pairs used: USD->TWD (direct), GBP->TWD (via GBP->USD->TWD)")
    before running the real conversion.

    Currencies already equal to base_currency are omitted (identity
    conversion needs no FX pair). Order follows iteration order of
    currencies_present with base_currency's own entry (if present) skipped.
    Does not report whether the data will actually be obtainable at fetch
    time -- it only describes which pairs would need to be fetched.
    """
    pairs = []
    for ccy in currencies_present:
        if ccy == base_currency:
            continue
        pair = (ccy, base_currency)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _download_fx_leg(from_currency: str, to_currency: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetch one direct Yahoo Finance FX leg (from_currency == the direct
    pair's own `from`, i.e. this never inverts -- callers handle inversion).
    Returns a cleaned (NaN-dropped) Series of to_currency-per-from_currency,
    or None if the ticker is unknown or the download fails/returns nothing.
    Never raises -- mirrors src/data_loader.py's "never raise, report
    unavailability via None" idiom for a single download.
    """
    ticker = _DIRECT_PAIRS.get((from_currency, to_currency))
    if ticker is None:
        return None
    try:
        raw = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" in raw.columns:
        col = raw["Close"]
    else:
        return None

    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    col = col.dropna()
    if col.empty:
        return None
    return col


def get_fx_series(from_currency: str, to_currency: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetch a real daily FX rate series (units of `to_currency` per 1
    `from_currency`) over [start_date, end_date] via yfinance, using Yahoo
    Finance's FX ticker conventions -- see the module docstring for exactly
    which direct pairs are used and which crosses are composed through USD.

    Identity case: if from_currency == to_currency, returns a Series of 1.0
    for every business day in the requested range WITHOUT making any network
    call. This is not a fabricated/fallback rate -- 1.0 is the mathematically
    correct, exact conversion rate for a currency to itself, so returning it
    directly (rather than e.g. querying "USDUSD=X", which doesn't exist on
    Yahoo) is the correct behavior, not a shortcut that hides missing data.

    For any other pair, returns None (deliberately -- never a fabricated or
    flat/constant series) if:
      - the pair isn't one this module knows how to resolve (see
        _resolve_fx_path()), or
      - any leg's download genuinely fails/returns no data.
    Callers (see convert_prices_to_base_currency()) must treat None as
    "FX unavailable" and surface/skip accordingly rather than assume success.
    """
    if from_currency == to_currency:
        dates = pd.bdate_range(start=start_date, end=end_date)
        return pd.Series(1.0, index=dates)

    path = _resolve_fx_path(from_currency, to_currency)
    if not path:
        return None

    composed: Optional[pd.Series] = None
    for leg_from, leg_to, invert in path:
        leg_series = _download_fx_leg(leg_from, leg_to, start_date, end_date)
        if leg_series is None:
            return None
        if invert:
            leg_series = 1.0 / leg_series
        composed = leg_series if composed is None else _multiply_aligned(composed, leg_series)

    if composed is None or composed.empty:
        return None
    return composed


def _multiply_aligned(a: pd.Series, b: pd.Series) -> pd.Series:
    """Multiply two FX leg series after forward-filling each onto the union
    of both indices, so composing crosses (e.g. GBP->USD * USD->TWD) doesn't
    lose dates just because the two legs' trading calendars/history don't
    line up exactly (documented rationale for forward-fill: FX markets trade
    ~24/5 but Yahoo's daily series can still have sparse gaps per pair;
    carrying the last known spot rate forward is the standard, defensible
    approximation for a daily-granularity conversion -- see also
    convert_prices_to_base_currency()'s docstring, which applies the same
    reasoning when aligning FX onto the price index)."""
    union_index = a.index.union(b.index)
    a_ff = a.reindex(union_index).ffill()
    b_ff = b.reindex(union_index).ffill()
    result = (a_ff * b_ff).dropna()
    return result


def convert_prices_to_base_currency(
    prices_df: pd.DataFrame,
    ticker_currency_map: Dict[str, str],
    base_currency: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Convert a wide, mixed-currency price DataFrame (columns=tickers,
    index=dates, as returned by src.data_loader.download_etf_data()) into a
    single common base_currency, so the result can be fed straight into
    src.portfolio_optimizer.run_optimization() unchanged.

    ticker_currency_map: {ticker: native_currency}, expected to be built by
    the caller (a page) from src.etf_database.get_etf(ticker).currency for
    each ticker -- this module deliberately does not import etf_database
    itself, both to stay testable in isolation from the ETF universe and to
    keep a clean one-way dependency (pages depend on src.fx, not the other
    way around). Tickers missing from ticker_currency_map are treated as
    already being in base_currency (nothing to convert) rather than being
    silently dropped, since "currency unknown" is not the same failure mode
    as "FX unavailable" that this function is responsible for reporting.

    For each ticker:
      - if its native currency == base_currency, its column is copied
        through unchanged and NO FX network call is made at all (a no-op
        currency is not a reason to touch the network);
      - otherwise, get_fx_series(native, base_currency, ...) is fetched once
        per distinct native currency actually present (not once per
        ticker -- multiple tickers sharing a currency reuse the same FX
        series) and forward-filled onto that ticker's own price index before
        multiplying. Forward-filling is used (rather than e.g. dropping
        price dates FX doesn't have, or interpolating) because both prices
        and FX are daily spot series recorded on their own market's trading
        calendar; carrying the last known FX rate forward onto a price date
        with no same-day FX quote is the standard, conservative convention
        (it never invents a rate that didn't exist on some earlier date).

    A single ticker's FX failure is fully isolated: it never raises, and
    never causes any other ticker's conversion to be skipped. It is instead
    recorded in "unavailable_tickers" and DROPPED from "converted_prices" --
    never left in its native currency and never filled with a fabricated
    rate, since either of those would silently corrupt a shared-currency
    portfolio optimization downstream.

    Returns a dict with:
      converted_prices    : wide DataFrame, base_currency, only successfully
                             converted (or already-native) tickers as columns
      unavailable_tickers  : list[str] of tickers dropped due to FX failure
      fx_source            : str, e.g. "Yahoo Finance (yfinance) daily FX rates"
      conversion_method    : str describing the method used (forward-filled
                             daily spot conversion, cross-rate composition
                             through USD when applicable, base currency)
      base_currency        : str, echoes the requested base_currency
      currency_adjusted     : bool, True iff at least one ticker actually
                             required a real (non-identity) conversion --
                             i.e. False when every ticker was already in
                             base_currency, in which case no FX call was
                             attempted at all.
    """
    converted_columns: Dict[str, pd.Series] = {}
    unavailable_tickers: List[str] = []
    currency_adjusted = False
    fx_cache: Dict[str, Optional[pd.Series]] = {}

    for ticker in prices_df.columns:
        native_currency = ticker_currency_map.get(ticker, base_currency)
        price_col = prices_df[ticker].dropna()

        if native_currency == base_currency:
            converted_columns[ticker] = price_col
            continue

        if native_currency not in fx_cache:
            fx_cache[native_currency] = get_fx_series(native_currency, base_currency, start_date, end_date)
        fx_series = fx_cache[native_currency]

        if fx_series is None or fx_series.empty:
            unavailable_tickers.append(ticker)
            continue

        fx_aligned = fx_series.reindex(price_col.index).ffill().bfill()
        if fx_aligned.isna().any():
            # Even after ffill/bfill some price dates have no usable FX
            # coverage at all (e.g. FX series starts well after prices do) --
            # genuinely unavailable for this ticker, not fabricated.
            unavailable_tickers.append(ticker)
            continue

        converted_columns[ticker] = price_col * fx_aligned
        currency_adjusted = True

    converted_prices = pd.DataFrame(converted_columns) if converted_columns else pd.DataFrame()

    used_crosses = any(
        len(_resolve_fx_path(ccy, base_currency) or []) > 1
        for ccy in fx_cache if ccy != base_currency
    )
    cross_note = (
        " (composing cross-rates through USD where no direct Yahoo Finance pair exists)"
        if used_crosses else ""
    )
    conversion_method = (
        f"Forward-filled daily spot FX conversion to {base_currency}{cross_note}; "
        f"each ticker's native-currency price is multiplied by its currency's "
        f"to-{base_currency}-currency spot rate, forward-filled (then back-filled "
        f"for any leading gap) onto that ticker's own price date index."
    )

    return {
        "converted_prices": converted_prices,
        "unavailable_tickers": unavailable_tickers,
        "fx_source": FX_SOURCE_LABEL,
        "conversion_method": conversion_method,
        "base_currency": base_currency,
        "currency_adjusted": currency_adjusted,
    }
