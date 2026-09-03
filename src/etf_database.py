"""
Global ETF Database
Single source of truth for every ETF the platform knows about, across every
supported market. Currently covers United States, Taiwan, and United Kingdom
(UCITS) ETFs; adding a new market (Japan, Europe, Hong Kong, Canada, ...) is
just a matter of appending more ETFRecord entries below -- no other code in
the project needs to change, since every page reads the ETF universe through
ETFDatabase's region/country-aware accessors rather than any hardcoded list.

Taiwan ETF universe (2026 market-data architecture fix): this is a curated
LOCAL SNAPSHOT compiled from general market knowledge of TWSE/TPEx-listed
ETF products -- NOT a live scrape of the Taiwan Stock Exchange or Taipei
Exchange, and NOT claimed to be the literal complete ~300+ product Taiwan
ETF universe. Per the architecture requirement that drove this expansion
("do not make every page load scrape TWSE/TPEx -- use a maintained local
metadata snapshot plus a controlled refresh process"), this snapshot
substantially widens coverage (see the module docstring's companion report
for exact counts and category breakdown) while staying easy to extend:
adding a newly-confirmed ETF is one more _tw_etf(...) call below, and nothing
elsewhere in the codebase needs to change.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ETFRecord:
    """Everything the platform knows about a single ETF."""
    ticker: str              # display ticker, e.g. "0050"
    name: str                # full fund name (English, or the fund's own name where no reliable English name exists)
    region: str               # broad grouping, e.g. "North America", "Asia Pacific", "Europe"
    country: str              # LISTING market, e.g. "United States", "Taiwan", "United Kingdom"
    currency: str              # e.g. "USD", "TWD", "GBP"
    exchange: str              # e.g. "NYSE Arca", "Taiwan Stock Exchange (TWSE)", "Taipei Exchange (TPEx)"
    category: str              # e.g. "Equity", "Fixed Income", "Commodity", "Multi-Asset"
    sector: str                # e.g. "Broad Market", "Technology", "Dividend", "Gold", "Bond"
    benchmark: str              # tracked index, e.g. "S&P 500"
    asset_type: str              # e.g. "Equity ETF", "Bond ETF", "Commodity ETF"
    investment_style: str         # e.g. "Blend", "Growth", "Value/Income" (fund style box -- unrelated to active/passive)
    yahoo_symbol: str              # the actual Yahoo Finance-fetchable symbol (with exchange suffix)
    expense_ratio: Optional[float] = None  # left as None where not yet populated
    # -- Fields added by the Taiwan ETF universe expansion (additive only;
    # every field above keeps its original meaning and position, so no
    # existing positional ETFRecord(...) call anywhere needed to change) --
    display_name_zh: Optional[str] = None     # official Chinese fund name, where the ETF trades in a Chinese-speaking market
    issuer: Optional[str] = None                # fund management company, e.g. "Yuanta", "Cathay", "Fubon"
    listing_date: Optional[str] = None            # ISO date string, where known; None if not confirmed
    underlying_market: Optional[str] = None         # what the ETF actually INVESTS in, e.g. "Taiwan", "United States",
                                                     # "Japan", "China/Hong Kong", "Global" -- distinct from `country`
                                                     # (where it's LISTED); a Taiwan-listed S&P 500 ETF has
                                                     # country="Taiwan", underlying_market="United States"
    management_style: Optional[str] = None           # "Active" / "Passive"
    return_type: Optional[str] = None                  # "Standard" / "Leveraged" / "Inverse"
    yahoo_status: str = "valid"                          # "valid" / "unsupported" / "unknown" -- see
                                                          # validate_etf_database()'s docstring: this is the ETF's
                                                          # LISTING/registry status, separate from whether Yahoo
                                                          # Finance happens to be reachable on any given request


def _tw_etf(ticker: str, name_en: str, name_zh: str, category: str, sector: str,
            benchmark: str, asset_type: str, issuer: str,
            underlying_market: str = "Taiwan", management_style: str = "Passive",
            return_type: str = "Standard", investment_style: str = "Blend",
            exchange: str = "TWSE", listing_date: Optional[str] = None) -> "ETFRecord":
    """Construct one Taiwan-listed ETFRecord.

    This is the ONE place that decides the exchange-specific Yahoo Finance
    suffix for a Taiwan-listed security (market-data reliability
    architecture requirement: TWSE-listed securities are ".TW" on Yahoo
    Finance; Taipei Exchange (TPEx, Taiwan's OTC market) securities are
    ".TWO" -- verified against Yahoo Finance's actual Taiwan OTC ticker
    convention). No page or other module ever constructs a Taiwan Yahoo
    symbol itself; everything goes through to_yahoo_symbol() below, which
    just looks up whatever was decided here.

    Every field that's constant across all Taiwan-listed ETFs (region,
    country, currency) is also filled in here once, so it can never drift
    between individual entries in the _TW_RECORDS list.
    """
    if exchange == "TPEx":
        exchange_full = "Taipei Exchange (TPEx)"
        yahoo_suffix = ".TWO"
    else:
        exchange_full = "Taiwan Stock Exchange (TWSE)"
        yahoo_suffix = ".TW"
    return ETFRecord(
        ticker=ticker, name=name_en, region="Asia Pacific", country="Taiwan", currency="TWD",
        exchange=exchange_full, category=category, sector=sector, benchmark=benchmark,
        asset_type=asset_type, investment_style=investment_style,
        yahoo_symbol=f"{ticker}{yahoo_suffix}",
        display_name_zh=name_zh, issuer=issuer, listing_date=listing_date,
        underlying_market=underlying_market, management_style=management_style,
        return_type=return_type, yahoo_status="valid",
    )


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
    ETFRecord("VT", "Vanguard Total World Stock ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Broad Market", "FTSE Global All Cap Index", "Equity ETF", "Blend", "VT"),
    ETFRecord("VXUS", "Vanguard Total International Stock ETF", "North America", "United States", "USD",
              "Nasdaq", "Equity", "Broad Market", "FTSE Global All Cap ex US Index", "Equity ETF", "Blend", "VXUS"),
    ETFRecord("TLT", "iShares 20+ Year Treasury Bond ETF", "North America", "United States", "USD",
              "Nasdaq", "Fixed Income", "Bond", "ICE U.S. Treasury 20+ Year Bond Index", "Bond ETF", "Income", "TLT"),
    ETFRecord("IWM", "iShares Russell 2000 ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Small Cap", "Russell 2000", "Equity ETF", "Blend", "IWM"),
    ETFRecord("XLK", "Technology Select Sector SPDR Fund", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Technology", "Technology Select Sector Index", "Equity ETF", "Growth", "XLK"),
    ETFRecord("XLF", "Financial Select Sector SPDR Fund", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Financials", "Financial Select Sector Index", "Equity ETF", "Blend", "XLF"),
    ETFRecord("XLV", "Health Care Select Sector SPDR Fund", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Healthcare", "Health Care Select Sector Index", "Equity ETF", "Blend", "XLV"),
    ETFRecord("VNQ", "Vanguard Real Estate ETF", "North America", "United States", "USD",
              "NYSE Arca", "Equity", "Real Estate", "MSCI US Investable Market Real Estate 25/50 Index",
              "Equity ETF", "Income", "VNQ"),

    # ── Taiwan (TWSE + TPEx) ─────────────────────────────────────────────
    # See _tw_etf() above for the exchange -> Yahoo-suffix mapping. Grouped
    # by category for readability; category/return_type/management_style
    # are explicit metadata fields (never inferred from ticker suffix --
    # market-data reliability requirement), even though several issuers'
    # own naming conventions happen to correlate (e.g. many, not all,
    # leveraged products end in "L" and inverse in "R" -- coincidental to
    # this data, not how it's classified).

    # -- Broad Market / Taiwan Equity Index (Passive, Standard) --
    _tw_etf("0050", "Yuanta Taiwan Top 50 ETF", "元大台灣50", "Equity", "Broad Market",
            "Taiwan 50 Index", "Equity ETF", "Yuanta", listing_date="2003-06-30"),
    _tw_etf("006208", "Fubon Taiwan 50 ETF", "富邦台50", "Equity", "Broad Market",
            "Taiwan 50 Index", "Equity ETF", "Fubon", listing_date="2012-07-17"),
    _tw_etf("0051", "Yuanta Taiwan Mid-Cap 100 ETF", "元大中型100", "Equity", "Mid Cap",
            "Taiwan Mid-Cap 100 Index", "Equity ETF", "Yuanta"),
    _tw_etf("006203", "Yuanta MSCI Taiwan ETF", "元大MSCI台灣", "Equity", "Broad Market",
            "MSCI Taiwan Index", "Equity ETF", "Yuanta"),
    _tw_etf("006204", "SinoPac Taiwan Weighted ETF", "永豐臺灣加權", "Equity", "Broad Market",
            "Taiwan Capitalization Weighted Stock Index", "Equity ETF", "SinoPac"),
    _tw_etf("00692", "Fubon Taiwan Corporate Governance ETF", "富邦公司治理", "Equity", "Broad Market",
            "Taiwan Corporate Governance 100 Index", "Equity ETF", "Fubon"),
    _tw_etf("00850", "Yuanta Taiwan ESG Sustainability ETF", "元大臺灣ESG永續", "Equity", "Broad Market",
            "Taiwan ESG Sustainability Index", "Equity ETF", "Yuanta"),

    # -- High Dividend (Passive, Standard) --
    _tw_etf("0056", "Yuanta Taiwan Dividend Plus ETF", "元大高股息", "Equity", "Dividend",
            "Taiwan High Dividend Yield Index", "Equity ETF", "Yuanta",
            investment_style="Value / Income", listing_date="2007-12-13"),
    _tw_etf("00878", "Cathay Sustainable High Dividend ETF", "國泰永續高股息", "Equity", "Dividend",
            "MSCI Taiwan ESG Sustainability High Dividend Yield Index", "Equity ETF", "Cathay",
            investment_style="Value / Income", listing_date="2020-07-20"),
    _tw_etf("00919", "Capital Taiwan Top Dividend ETF", "群益台灣精選高息", "Equity", "Dividend",
            "Taiwan Select High Dividend Index", "Equity ETF", "Capital",
            investment_style="Value / Income", listing_date="2022-10-20"),
    _tw_etf("00929", "Fuh Hwa Taiwan Technology Optimized Dividend ETF", "復華台灣科技優息", "Equity", "Dividend",
            "Taiwan Technology Dividend Optimized Index", "Equity ETF", "Fuh Hwa",
            investment_style="Value / Income", listing_date="2023-03-08"),
    _tw_etf("00713", "Yuanta Taiwan Dividend Low Volatility ETF", "元大台灣高息低波", "Equity", "Dividend",
            "Taiwan Dividend Low Volatility Index", "Equity ETF", "Yuanta", investment_style="Value / Income"),
    _tw_etf("00915", "KGI Taiwan Top Dividend 30 ETF", "凱基優選高股息30", "Equity", "Dividend",
            "Taiwan Top Dividend 30 Index", "Equity ETF", "KGI", investment_style="Value / Income"),
    _tw_etf("00934", "CTBC Growth High Dividend ETF", "中信成長高股息", "Equity", "Dividend",
            "Taiwan Growth High Dividend Index", "Equity ETF", "CTBC", investment_style="Value / Income"),
    _tw_etf("00936", "Taishin Sustainable High Dividend Small-Mid ETF", "台新永續高息中小", "Equity", "Dividend",
            "Taiwan Sustainable High Dividend Small-Mid Index", "Equity ETF", "Taishin",
            investment_style="Value / Income"),
    _tw_etf("00939", "Uni-President Taiwan High Dividend Momentum ETF", "統一台灣高息動能", "Equity", "Dividend",
            "Taiwan High Dividend Momentum Index", "Equity ETF", "Uni-President",
            investment_style="Value / Income"),
    _tw_etf("00940", "Yuanta Taiwan Value High Dividend ETF", "元大台灣價值高息", "Equity", "Dividend",
            "Taiwan Value High Dividend Index", "Equity ETF", "Yuanta", investment_style="Value / Income"),

    # -- Technology / Sector Thematic (Passive, Standard) --
    _tw_etf("0052", "Fubon Taiwan Technology ETF", "富邦科技", "Equity", "Technology",
            "Taiwan Information Technology Index", "Equity ETF", "Fubon", investment_style="Growth"),
    _tw_etf("00830", "Cathay Philadelphia Semiconductor ETF", "國泰費城半導體", "Equity", "Semiconductor",
            "PHLX Semiconductor Sector Index", "Equity ETF", "Cathay",
            underlying_market="United States", investment_style="Growth"),
    _tw_etf("00881", "Cathay Taiwan 5G+ ETF", "國泰台灣5G+", "Equity", "Technology",
            "Taiwan 5G+ Communication Technology Index", "Equity ETF", "Cathay", investment_style="Growth"),
    _tw_etf("00891", "CTBC Taiwan Key Semiconductor ETF", "中信關鍵半導體", "Equity", "Semiconductor",
            "Taiwan Key Semiconductor Index", "Equity ETF", "CTBC", investment_style="Growth"),
    _tw_etf("00892", "Fubon Taiwan Semiconductor ETF", "富邦台灣半導體", "Equity", "Semiconductor",
            "Taiwan Semiconductor Index", "Equity ETF", "Fubon", investment_style="Growth"),

    # -- Overseas Equity Underlying (Passive, Standard, Taiwan-LISTED but investing abroad) --
    _tw_etf("00646", "Yuanta S&P 500 ETF", "元大S&P500", "Equity", "Broad Market",
            "S&P 500 Index", "Equity ETF", "Yuanta", underlying_market="United States"),
    _tw_etf("00662", "Fubon NASDAQ ETF", "富邦NASDAQ", "Equity", "Technology",
            "NASDAQ-100 Index", "Equity ETF", "Fubon", underlying_market="United States", investment_style="Growth"),
    _tw_etf("00661", "Yuanta Nikkei 225 ETF", "元大日經225", "Equity", "Broad Market",
            "Nikkei 225 Index", "Equity ETF", "Yuanta", underlying_market="Japan"),
    _tw_etf("006205", "Fubon Shanghai ETF", "富邦上證", "Equity", "Broad Market",
            "SSE 180 Index", "Equity ETF", "Fubon", underlying_market="China/Hong Kong"),
    _tw_etf("00636", "Cathay China A50 ETF", "國泰中國A50", "Equity", "Broad Market",
            "FTSE China A50 Index", "Equity ETF", "Cathay", underlying_market="China/Hong Kong"),

    # -- Active ETFs (management_style="Active") -- classification is via
    # this explicit field, populated from official product metadata, never
    # inferred from the "A" ticker suffix alone (that suffix is common
    # among actively-managed Taiwan ETFs but is supporting evidence only,
    # per the ticker-format fix: a ticker's ALPHABETIC characters, if any,
    # are never used to derive category/management_style/return_type).
    # `name` below is an English GLOSS of the official Chinese fund name
    # for display convenience (matching this file's existing convention
    # for every other Taiwan record), not a claimed official English fund
    # name -- display_name_zh is the authoritative name.
    _tw_etf("00981A", "Uni-President Active Taiwan Growth ETF", "主動統一台股增長", "Equity", "Broad Market",
            "N/A (Actively Managed)", "Equity ETF", "Uni-President",
            management_style="Active", investment_style="Growth"),

    # -- Bond ETFs (overseas-denominated fixed income, Passive, Standard) --
    _tw_etf("00679B", "Yuanta US Treasury 20+ Year ETF", "元大美債20年", "Fixed Income", "Bond",
            "ICE U.S. Treasury 20+ Year Bond Index", "Bond ETF", "Yuanta",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00687B", "Cathay 20+ Year US Treasury ETF", "國泰20年美債", "Fixed Income", "Bond",
            "Bloomberg US Treasury 20+ Year Index", "Bond ETF", "Cathay",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00694B", "Fubon US Treasury 1-3 Year ETF", "富邦美債1-3", "Fixed Income", "Bond",
            "Bloomberg US Treasury 1-3 Year Index", "Bond ETF", "Fubon",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00695B", "Fubon US Treasury 7-10 Year ETF", "富邦美債7-10", "Fixed Income", "Bond",
            "Bloomberg US Treasury 7-10 Year Index", "Bond ETF", "Fubon",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00696B", "Fubon US Treasury 20 Year ETF", "富邦美債20年", "Fixed Income", "Bond",
            "Bloomberg US Treasury 20+ Year Index", "Bond ETF", "Fubon",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00720B", "Yuanta Investment Grade Corporate Bond ETF", "元大投資級公司債", "Fixed Income", "Bond",
            "Bloomberg US Corporate Investment Grade Index", "Bond ETF", "Yuanta",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00751B", "Yuanta AAA-A Corporate Bond ETF", "元大AAA至A公司債", "Fixed Income", "Bond",
            "Bloomberg AAA-A US Corporate Bond Index", "Bond ETF", "Yuanta",
            underlying_market="United States", investment_style="Income"),
    _tw_etf("00761B", "Cathay A-Rated Corporate Bond ETF", "國泰A級公司債", "Fixed Income", "Bond",
            "Bloomberg A Rated US Corporate Bond Index", "Bond ETF", "Cathay",
            underlying_market="United States", investment_style="Income"),

    # -- Leveraged / Inverse (Return Type != Standard) --
    _tw_etf("00631L", "Yuanta Taiwan 50 Leveraged 2X ETF", "元大台灣50正2", "Equity", "Broad Market",
            "Taiwan 50 Index", "Equity ETF", "Yuanta", return_type="Leveraged", investment_style="Aggressive"),
    _tw_etf("00632R", "Yuanta Taiwan 50 Inverse 1X ETF", "元大台灣50反1", "Equity", "Broad Market",
            "Taiwan 50 Index", "Equity ETF", "Yuanta", return_type="Inverse", investment_style="Aggressive"),
    _tw_etf("00633L", "Fubon Shanghai Leveraged 2X ETF", "富邦上證正2", "Equity", "Broad Market",
            "SSE 180 Index", "Equity ETF", "Fubon", underlying_market="China/Hong Kong",
            return_type="Leveraged", investment_style="Aggressive"),
    _tw_etf("00637L", "Yuanta CSI 300 Leveraged 2X ETF", "元大滬深300正2", "Equity", "Broad Market",
            "CSI 300 Index", "Equity ETF", "Yuanta", underlying_market="China/Hong Kong",
            return_type="Leveraged", investment_style="Aggressive"),
    _tw_etf("00663L", "Cathay Taiwan Weighted Leveraged 2X ETF", "國泰臺灣加權正2", "Equity", "Broad Market",
            "Taiwan Capitalization Weighted Stock Index", "Equity ETF", "Cathay",
            return_type="Leveraged", investment_style="Aggressive"),
    _tw_etf("00664R", "Cathay Taiwan Weighted Inverse 1X ETF", "國泰臺灣加權反1", "Equity", "Broad Market",
            "Taiwan Capitalization Weighted Stock Index", "Equity ETF", "Cathay",
            return_type="Inverse", investment_style="Aggressive"),
    _tw_etf("00675L", "Fubon Taiwan Weighted Leveraged 2X ETF", "富邦臺灣加權正2", "Equity", "Broad Market",
            "Taiwan Capitalization Weighted Stock Index", "Equity ETF", "Fubon",
            return_type="Leveraged", investment_style="Aggressive"),

    # -- Commodity (Passive, Standard) --
    _tw_etf("00635U", "Yuanta Gold ETF", "元大黃金", "Commodity", "Gold",
            "LBMA Gold Price", "Commodity ETF", "Yuanta", underlying_market="Global",
            investment_style="Alternative"),

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

    def search(self, query: str, country: Optional[str] = None) -> List[ETFRecord]:
        """Case-insensitive substring search across ticker, English name,
        Chinese display name, and issuer -- optionally scoped to one
        country. Powers the ETF selector's search box (section 6: search by
        ticker / Chinese name / issuer keyword, e.g. "0050", "高股息", "元大")."""
        pool = self.by_country(country) if country else self.all()
        if not query:
            return pool
        q = query.strip().lower()
        if not q:
            return pool
        out = []
        for r in pool:
            haystack = " ".join(filter(None, [
                r.ticker, r.name, r.display_name_zh, r.issuer,
            ])).lower()
            if q in haystack or query.strip() in (r.display_name_zh or ""):
                out.append(r)
        return out


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


def search_etfs(query: str, country: Optional[str] = None) -> List[ETFRecord]:
    """Module-level convenience wrapper around ETFDatabase.search()."""
    return ETF_DATABASE.search(query, country)


def to_yahoo_symbol(ticker: str) -> str:
    """
    Map a display ticker to its actual Yahoo Finance-fetchable symbol (e.g.
    '0050' -> '0050.TW', 'VUSA' -> 'VUSA.L'). Tickers not in the database
    (including free-text custom tickers such as 'ARKK') are returned
    unchanged, so this is always safe to call on any ticker string.

    This is the ONLY place in the project that resolves a display ticker to
    a Yahoo-fetchable symbol -- every page (Portfolio Optimizer, ETF
    Analysis, Investment Simulator, Risk Analytics, Machine Learning)
    calls this function rather than constructing a suffix itself, so the
    TWSE (".TW") vs TPEx (".TWO") distinction (decided once, in _tw_etf()
    above) can never drift or duplicate across pages.
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


def validate_etf_database() -> dict:
    """Data-quality audit of the ETF master universe (section 13: exchange
    listing status and Yahoo data availability are separate concepts --
    this function checks the FORMER only; it never touches the network and
    never removes a record just because Yahoo Finance might be temporarily
    unreachable for it).

    Returns a report dict:
        {
          "total_records": int,
          "issues": [ {ticker, problem}, ... ],   # empty if everything is clean
          "duplicate_tickers": [ticker, ...],
          "yahoo_status_counts": {"valid": n, "unsupported": n, "unknown": n},
        }
    """
    # Duplicate-ticker detection MUST run against the RAW source list, not
    # ETF_DATABASE.all() -- ETFDatabase.__init__ builds a {ticker: record}
    # dict, which silently collapses duplicate tickers (last one wins)
    # before .all() ever sees them, so checking the post-dedup list would
    # never find a duplicate that genuinely exists in _BUILTIN_RECORDS.
    raw_records = _BUILTIN_RECORDS
    issues = []
    seen = {}
    for r in raw_records:
        if not r.ticker or not r.ticker.strip():
            issues.append({"ticker": r.ticker, "problem": "empty ticker"})
            continue
        if not r.name and not r.display_name_zh:
            issues.append({"ticker": r.ticker, "problem": "no display name (name/display_name_zh both empty)"})
        if not r.exchange:
            issues.append({"ticker": r.ticker, "problem": "missing exchange"})
        if not r.yahoo_symbol:
            issues.append({"ticker": r.ticker, "problem": "missing yahoo_symbol"})
        if r.yahoo_status not in ("valid", "unsupported", "unknown"):
            issues.append({"ticker": r.ticker, "problem": f"invalid yahoo_status: {r.yahoo_status!r}"})
        seen[r.ticker] = seen.get(r.ticker, 0) + 1

    duplicate_tickers = [tk for tk, count in seen.items() if count > 1]
    for tk in duplicate_tickers:
        issues.append({"ticker": tk, "problem": "duplicate ticker in master universe"})

    yahoo_status_counts = {"valid": 0, "unsupported": 0, "unknown": 0}
    for r in raw_records:
        if r.yahoo_status in yahoo_status_counts:
            yahoo_status_counts[r.yahoo_status] += 1

    return {
        "total_records": len(raw_records),
        "issues": issues,
        "duplicate_tickers": duplicate_tickers,
        "yahoo_status_counts": yahoo_status_counts,
    }
