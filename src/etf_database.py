"""
Global ETF Database
Single source of truth for every ETF the platform knows about, across every
supported market. Currently covers United States, Taiwan, and United Kingdom
(UCITS) ETFs; adding a new market (Japan, Europe, Hong Kong, Canada, ...) is
just a matter of appending more ETFRecord entries below -- no other code in
the project needs to change, since every page reads the ETF universe through
ETFDatabase's region/country-aware accessors rather than any hardcoded list.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ETFRecord:
    """Everything the platform knows about a single ETF."""
    ticker: str              # display ticker, e.g. "0050"
    name: str                # full fund name
    region: str               # broad grouping, e.g. "North America", "Asia Pacific", "Europe"
    country: str              # e.g. "United States", "Taiwan", "United Kingdom"
    currency: str              # e.g. "USD", "TWD", "GBP"
    exchange: str              # e.g. "NYSE Arca", "Taiwan Stock Exchange (TWSE)"
    category: str              # e.g. "Equity", "Fixed Income", "Commodity"
    sector: str                # e.g. "Broad Market", "Technology", "Dividend", "Gold", "Bond"
    benchmark: str              # tracked index, e.g. "S&P 500"
    asset_type: str              # e.g. "Equity ETF", "Bond ETF", "Commodity ETF"
    investment_style: str         # e.g. "Blend", "Growth", "Value/Income"
    yahoo_symbol: str              # the actual Yahoo Finance-fetchable symbol (with exchange suffix)
    expense_ratio: Optional[float] = None  # left as None where not yet populated


# ── Built-in ETF Universe ────────────────────────────────────────────────────
# NOTE: to add a new market, add more ETFRecord rows here (with the correct
# region/country/currency/exchange/yahoo_symbol). No other file needs to
# change -- every page reads this list through ETFDatabase's accessors.
_BUILTIN_RECORDS: List[ETFRecord] = [
    # ── United States ─────────────────────────────────────────────────────
    ETFRecord("VOO", "Vanguard S&P 500 ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Broad Market", "S&P 500", "Equity ETF", "Blend", "VOO"),
    ETFRecord("VTI", "Vanguard Total Stock Market ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Broad Market", "CRSP US Total Market Index", "Equity ETF", "Blend", "VTI"),
    ETFRecord("QQQ", "Invesco QQQ Trust", "North America", "United States", "USD",
              "Nasdaq", "Equity", "Technology", "Nasdaq-100", "Equity ETF", "Growth", "QQQ"),
    ETFRecord("SPY", "SPDR S&P 500 ETF Trust", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Broad Market", "S&P 500", "Equity ETF", "Blend", "SPY"),
    ETFRecord("SCHD", "Schwab U.S. Dividend Equity ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Dividend", "Dow Jones U.S. Dividend 100 Index", "Equity ETF",
              "Value / Income", "SCHD"),
    ETFRecord("BND", "Vanguard Total Bond Market ETF", "North America", "United States", "USD",
              "Nasdaq", "Fixed Income", "Bond", "Bloomberg U.S. Aggregate Float Adjusted Index",
              "Bond ETF", "Income", "BND"),
    ETFRecord("GLD", "SPDR Gold Shares", "North America", "United States", "USD",
              "NYSE Arca", "Commodity", "Gold", "Gold Spot Price", "Commodity ETF", "Alternative", "GLD"),

    # ── Taiwan ────────────────────────────────────────────────────────────
    ETFRecord("0050", "Yuanta/P-shares Taiwan Top 50 ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Broad Market", "Taiwan 50 Index",
              "Equity ETF", "Blend", "0050.TW"),
    ETFRecord("0056", "Yuanta/P-shares Taiwan Dividend Plus ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Dividend", "Taiwan High Dividend Yield Index",
              "Equity ETF", "Value / Income", "0056.TW"),
    ETFRecord("006208", "Fubon Taiwan 50 ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Broad Market", "Taiwan 50 Index",
              "Equity ETF", "Blend", "006208.TW"),
    ETFRecord("00878", "Cathay Sustainable High Dividend ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Dividend",
              "MSCI Taiwan ESG Sustainability High Dividend Yield Index", "Equity ETF",
              "Value / Income", "00878.TW"),
    ETFRecord("00919", "Group Cathay Taiwan High Dividend Tech ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Dividend",
              "Taiwan High Dividend Tech Select 30 Index", "Equity ETF", "Value / Income", "00919.TW"),
    ETFRecord("00929", "Fuh Hwa Taiwan Technology Dividend Optimized ETF", "Asia Pacific", "Taiwan", "TWD",
              "Taiwan Stock Exchange (TWSE)", "Equity", "Dividend",
              "Taiwan Technology Dividend Optimized Index", "Equity ETF", "Value / Income", "00929.TW"),

    # ── United Kingdom (UCITS) ────────────────────────────────────────────
    ETFRecord("VUSA", "Vanguard S&P 500 UCITS ETF", "Europe", "United Kingdom", "GBP",
              "London Stock Exchange (LSE)", "Equity", "Broad Market", "S&P 500",
              "Equity ETF", "Blend", "VUSA.L"),
    ETFRecord("VUAG", "Vanguard S&P 500 UCITS ETF (USD Accumulating)", "Europe", "United Kingdom", "GBP",
              "London Stock Exchange (LSE)", "Equity", "Broad Market", "S&P 500",
              "Equity ETF", "Blend (Accumulating)", "VUAG.L"),
    ETFRecord("EQQQ", "Invesco EQQQ Nasdaq-100 UCITS ETF", "Europe", "United Kingdom", "GBP",
              "London Stock Exchange (LSE)", "Equity", "Technology", "Nasdaq-100",
              "Equity ETF", "Growth", "EQQQ.L"),
    ETFRecord("CSPX", "iShares Core S&P 500 UCITS ETF", "Europe", "United Kingdom", "GBP",
              "London Stock Exchange (LSE)", "Equity", "Broad Market", "S&P 500",
              "Equity ETF", "Blend (Accumulating)", "CSPX.L"),
    ETFRecord("FUSD", "Fidelity US Quality Income UCITS ETF", "Europe", "United Kingdom", "GBP",
              "London Stock Exchange (LSE)", "Equity", "Dividend", "Fidelity US Quality Income Index",
              "Equity ETF", "Value / Income", "FUSD.L"),
]


class ETFDatabase:
    """
    In-memory registry of ETFRecord entries with region/country-aware lookup
    helpers. All pages should go through this class (or the module-level
    convenience functions below) rather than hardcoding ticker lists, so that
    adding a new market never requires touching page code.
    """

    def __init__(self, records: List[ETFRecord]):
        self._records: Dict[str, ETFRecord] = {r.ticker: r for r in records}

    def all(self) -> List[ETFRecord]:
        return list(self._records.values())

    def get(self, ticker: str) -> Optional[ETFRecord]:
        return self._records.get(ticker)

    def all_tickers(self) -> List[str]:
        return list(self._records.keys())

    def regions(self) -> List[str]:
        """Distinct regions, in first-seen order."""
        return list(dict.fromkeys(r.region for r in self._records.values()))

    def countries(self) -> List[str]:
        """Distinct countries, in first-seen order."""
        return list(dict.fromkeys(r.country for r in self._records.values()))

    def countries_by_region(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for r in self._records.values():
            out.setdefault(r.region, [])
            if r.country not in out[r.region]:
                out[r.region].append(r.country)
        return out

    def by_country(self, country: str) -> List[ETFRecord]:
        return [r for r in self._records.values() if r.country == country]

    def tickers_by_country(self, country: str) -> List[str]:
        return [r.ticker for r in self._records.values() if r.country == country]

    def by_region(self, region: str) -> List[ETFRecord]:
        return [r for r in self._records.values() if r.region == region]


# ── Module-level singleton + convenience functions ──────────────────────────
ETF_DATABASE = ETFDatabase(_BUILTIN_RECORDS)


def get_etf(ticker: str) -> Optional[ETFRecord]:
    """Look up a single ETF's full record, or None if it isn't in the database
    (e.g. a free-text custom ticker the user typed in)."""
    return ETF_DATABASE.get(ticker)


def get_countries() -> List[str]:
    """All supported countries, in the order they were registered."""
    return ETF_DATABASE.countries()


def get_regions() -> List[str]:
    """All supported regions, in the order they were registered."""
    return ETF_DATABASE.regions()


def get_tickers_by_country(country: str) -> List[str]:
    """All tickers belonging to a given country (empty list if unknown)."""
    return ETF_DATABASE.tickers_by_country(country)


def get_all_tickers() -> List[str]:
    """Every ticker in the database, across all countries."""
    return ETF_DATABASE.all_tickers()


def get_country(ticker: str) -> Optional[str]:
    """The country of a given ticker, or None if it isn't in the database."""
    record = ETF_DATABASE.get(ticker)
    return record.country if record else None


def to_yahoo_symbol(ticker: str) -> str:
    """
    Map a display ticker to its actual Yahoo Finance-fetchable symbol (e.g.
    '0050' -> '0050.TW', 'VUSA' -> 'VUSA.L'). Tickers not in the database
    (including free-text custom tickers such as 'ARKK') are returned
    unchanged, so this is always safe to call on any ticker string.
    """
    record = ETF_DATABASE.get(ticker)
    return record.yahoo_symbol if record else ticker


def rename_yahoo_columns(df):
    """
    Rename a downloaded-price DataFrame's columns from Yahoo Finance symbols
    back to clean display tickers (e.g. '0050.TW' -> '0050'). Columns that
    don't match any known Yahoo symbol are left unchanged, so this is always
    safe to call even on US-only, already-clean data.
    """
    reverse_map = {r.yahoo_symbol: r.ticker for r in ETF_DATABASE.all()}
    return df.rename(columns=reverse_map)
