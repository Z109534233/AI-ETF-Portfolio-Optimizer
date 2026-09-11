"""
ETF Universe Refresh Job
=========================
The "authoritative source -> refresh/import job -> normalized local
snapshot -> application" updater for the platform's global ETF master
universe (Taiwan / United States / United Kingdom).

This is a CONTROLLED, MANUALLY-RUN job -- it is never imported or executed
by the Streamlit app itself (src/etf_database.py only ever READS the JSON
snapshots this script writes under data/etf_universe/). Re-run it to refresh
the snapshots:

    python scripts/refresh_etf_universe.py

Sources used (see each fetch_* function's docstring for the exact URL and
why it was chosen):

  Taiwan (TWSE + TPEx):
    TWSE's own ISIN registry (isin.twse.com.tw) -- the official domestic
    listed-securities identification list published by the Taiwan Stock
    Exchange itself. strMode=2 = TWSE-listed (main board), strMode=4 =
    TPEx-listed (OTC board). Both pages list EVERY listed security broken
    into category sections (Stocks / Warrants / Preferred / Innovation
    Board / ETF / ETN / TDR / REITs); this job extracts only the ETF
    section of each.

  United States:
    Nasdaq Trader's official Symbol Directory (nasdaqtrader.com), the
    industry-standard security master used throughout the US brokerage/
    market-data industry. nasdaqlisted.txt covers Nasdaq-listed securities;
    otherlisted.txt covers securities listed on other exchanges (NYSE,
    NYSE Arca, Cboe BZX/BATS) that clear through the Nasdaq/CQS tape. Both
    files carry an explicit "ETF" Y/N flag per security -- this job filters
    on that flag rather than guessing from the ticker or name, satisfying
    "do not include ordinary stocks simply because they trade on the same
    exchanges" and "create a reproducible ETF-filtering process".

  United Kingdom / London Stock Exchange:
    NO reliable bulk authoritative feed could be found or safely parsed in
    this environment (LSE's own market-data site and every major UK ETF
    issuer's product-finder page are JavaScript-rendered single-page apps
    backed by private/undocumented APIs; guessing at those endpoints
    produced 404s, and none exposed a plain downloadable instrument list).
    Per this job's own no-fabrication rule, the UK universe below is a
    CURATED list of individually-recognized real LSE-listed UCITS ETF
    tickers across multiple issuers/currencies -- NOT a bulk feed, and NOT
    claimed to be the complete LSE ETF universe. See the AFTER
    IMPLEMENTATION report for this round for the full disclosure.

Classification (issuer / asset class / return type / management style) is
never inferred from a ticker's suffix alone. For Taiwan, this job matches
keywords against the security's own OFFICIAL NAME as registered with TWSE/
TPEx (e.g. "主動" for Active, "正2"/"反1" for Leveraged/Inverse, "債" for
Bond) -- confirmed against the real fetched data that ticker-suffix alone
would have been WRONG (8 of the 40 officially "主動" (Active) TWSE/TPEx
ETFs use a "D" ticker suffix, not "A" -- suffix-only classification would
have silently misclassified them). For the US, this job matches known
issuer names and category keywords against each security's official Nasdaq
Trader "Security Name" field. Anything this job cannot confidently derive
from the source's own text is left as `null`, never guessed.
"""

import json
import os
import re
import urllib.request
from datetime import date, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(os.path.dirname(_HERE), "data", "etf_universe")
_TODAY = date.today().isoformat()
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_text(url: str, timeout: int = 30) -> str:
    return _fetch_bytes(url, timeout).decode("utf-8", errors="replace")


# ── Taiwan (TWSE + TPEx) ──────────────────────────────────────────────────
def _extract_twse_isin_section(raw: bytes, section_label: bytes, next_label: bytes):
    """Slice out one category's rows from a TWSE/TPEx ISIN registry page.
    Section headers are '<B> LABEL </B>' cells with colspan=7; rows in
    between are 7-column <tr> blocks (code+name / ISIN / listing date /
    market / industry / CFICode / note)."""
    start_marker = re.search(re.escape(section_label), raw)
    end_marker = re.search(re.escape(next_label), raw)
    if not start_marker or not end_marker:
        return []
    section = raw[start_marker.end():end_marker.start()]
    rows = re.findall(
        rb"<tr><td bgcolor=#FAFAD2>(.*?)</td><td bgcolor=#FAFAD2>(.*?)</td>"
        rb"<td bgcolor=#FAFAD2>(.*?)</td><td bgcolor=#FAFAD2>(.*?)</td>"
        rb"<td bgcolor=#FAFAD2>(.*?)</td><td bgcolor=#FAFAD2>(.*?)</td>"
        rb"<td bgcolor=#FAFAD2>(.*?)</td></tr>",
        section,
    )
    out = []
    for code_name_raw, isin_raw, date_raw, *_rest in rows:
        code_name = code_name_raw.decode("big5", errors="replace")
        parts = re.split(r"[\s　\xa1]+", code_name, maxsplit=1)
        code, name = (parts[0], parts[1]) if len(parts) == 2 else (code_name.strip(), "")
        out.append({
            "code": code.strip(),
            "name": name.strip(),
            "isin": isin_raw.decode("ascii", errors="replace").strip(),
            "listing_date": date_raw.decode("ascii", errors="replace").strip(),
        })
    return out


def fetch_taiwan_etfs():
    """TWSE (strMode=2, main board) + TPEx (strMode=4, OTC board) ISIN
    registry pages: https://isin.twse.com.tw/isin/C_public.jsp?strMode=2|4
    -- the official domestic securities identification list published by
    TWSE itself. Each page lists ALL listed securities in category
    sections; this extracts only the "ETF" section from each."""
    # Section-header cells on this page use a malformed, unclosed <B> tag --
    # literally "<B> ETF <B> </td>", not "<B> ETF </B>" -- confirmed byte-for-
    # byte against the live fetched page; matching on TWSE's actual markup,
    # not the well-formed HTML one might assume.
    twse_raw = _fetch_bytes("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2")
    twse_rows = _extract_twse_isin_section(twse_raw, b"<B> ETF <B>", b"<B> ETN <B>")
    for r in twse_rows:
        r["exchange"] = "TWSE"
        r["source"] = "TWSE ISIN Registry (isin.twse.com.tw, strMode=2)"

    tpex_raw = _fetch_bytes("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4")
    tpex_rows = _extract_twse_isin_section(tpex_raw, b"<B> ETF <B>", b"<B> ETN <B>")
    for r in tpex_rows:
        r["exchange"] = "TPEx"
        r["source"] = "TPEx ISIN Registry (isin.twse.com.tw, strMode=4)"

    return twse_rows + tpex_rows


def _classify_taiwan_name(name: str):
    """Classify a Taiwan ETF's asset_class/asset_type/management_style/
    return_type from its OFFICIAL registered name text (never from ticker
    suffix -- see this module's docstring for why that would be wrong)."""
    management_style = "Active" if "主動" in name else "Passive"

    if any(k in name for k in ("正2", "正1", "正3", "槓桿")):
        return_type = "Leveraged"
    elif any(k in name for k in ("反1", "反一")):
        return_type = "Inverse"
    else:
        return_type = "Standard"

    if "債" in name:
        asset_class, asset_type = "Fixed Income", "Bond ETF"
    elif any(k in name for k in ("不動產", "REIT")):
        asset_class, asset_type = "Real Estate", "REIT ETF"
    elif any(k in name for k in ("貨幣", "現金")):
        asset_class, asset_type = "Money Market", "Money Market ETF"
    elif any(k in name for k in ("黃金", "原油", "商品")):
        asset_class, asset_type = "Commodity", "Commodity ETF"
    else:
        asset_class, asset_type = "Equity", "Equity ETF"

    return management_style, return_type, asset_class, asset_type


def build_taiwan_snapshot(rows):
    out = []
    for r in rows:
        code = r["code"]
        exchange = r["exchange"]
        suffix = ".TWO" if exchange == "TPEx" else ".TW"
        management_style, return_type, asset_class, asset_type = _classify_taiwan_name(r["name"])
        listing_date = None
        if re.match(r"^\d{4}/\d{2}/\d{2}$", r["listing_date"]):
            listing_date = r["listing_date"].replace("/", "-")
        out.append({
            "ticker": code,
            "name": r["name"],
            "listing_market": "Taiwan",
            "exchange": "Taiwan Stock Exchange (TWSE)" if exchange == "TWSE" else "Taipei Exchange (TPEx)",
            "currency": "TWD",
            "asset_class": asset_class,
            "asset_type": asset_type,
            "underlying_market": "Taiwan",
            "management_style": management_style,
            "return_type": return_type,
            "issuer": None,
            "listing_date": listing_date,
            "isin": r["isin"] or None,
            "provider_symbol": f"{code}{suffix}",
            "provider_status": "unknown",
            "source": r["source"],
            "source_updated_at": _TODAY,
        })
    return out


# ── United States ────────────────────────────────────────────────────────
_US_ISSUER_KEYWORDS = [
    "iShares", "Vanguard", "SPDR", "Invesco", "Charles Schwab", "Schwab",
    "ProShares", "Direxion", "Global X", "ARK", "JPMorgan", "J.P. Morgan",
    "Fidelity", "First Trust", "WisdomTree", "VanEck", "Goldman Sachs",
    "PIMCO", "Franklin", "State Street", "Dimensional", "Janus Henderson",
    "Simplify", "Roundhill", "Defiance", "GraniteShares", "Amplify",
    "Innovator", "Calamos", "YieldMax", "Pacer", "American Century",
    "T. Rowe Price", "BlackRock", "Nuveen", "abrdn", "Xtrackers",
    "KraneShares", "Alpha Architect", "Avantis", "Bitwise", "Grayscale",
    "REX Shares", "Tidal", "Themes", "Harbor", "Columbia", "Hartford",
    "AllianceBernstein", "Neos", "FT Vest", "US Global Investors",
    "Virtus", "John Hancock", "Hartford Funds", "Cambria", "Bridges",
    "Rockefeller", "Impact Shares", "Teucrium", "USCF", "United States",
    "iPath", "Barclays", "UBS", "Credit Suisse", "Volatility Shares",
    "Global Beta", "Sprott", "abrdn", "Amana", "Davis", "Motley Fool",
    "Renaissance", "Matthews", "Fred Alger", "Alger", "Nationwide",
    "Timothy Plan", "Toroso", "Tuttle", "Astoria", "AXS", "BondBloxx",
    "Brookmont", "Cabana", "Cboe Vest", "CI Galaxy", "Distillate",
    "Dunham", "EA Series", "Eaton Vance", "ETC", "Exchange Traded Concepts",
    "F/m Investments", "Fair Oaks", "Goose Hollow", "GraniteShares",
    "Horizon Kinetics", "Hull Tactical", "iM DBi", "Leatherback",
    "Little Harbor", "Main Management", "Mairs", "NEOS", "Nicholas",
    "North Shore", "Optica", "Range", "Regan", "Rockefeller",
    "Schwartz", "Siebert", "Sit", "SmartETFs", "Sound Shore",
    "Strategy Shares", "Swan", "TCW", "Texas Capital", "The Acquirers",
    "TrueShares", "Two Roads", "US Benchmark", "USCF", "Vident",
    "WBI", "WEBS", "Wilshire", "Xshares", "Zacks",
]

_US_LEVERAGED_INVERSE_KEYWORDS = [
    (r"\b(2X|3X)\b", "Leveraged"),
    (r"\bUltraPro\b", "Leveraged"),
    (r"\bUltra\b", "Leveraged"),
    (r"\bDaily.*Bull\b", "Leveraged"),
    (r"\bDaily.*Inverse\b", "Inverse"),
    (r"\bInverse\b", "Inverse"),
    (r"\bShort\b", "Inverse"),
    (r"\bBear\b", "Inverse"),
    (r"-1X\b", "Inverse"),
    (r"-2X\b", "Leveraged"),
    (r"-3X\b", "Leveraged"),
]

_US_BOND_KEYWORDS = ("Bond", "Treasury", "Fixed Income", "Municipal", "Notes", "Duration")
_US_COMMODITY_KEYWORDS = ("Gold", "Silver", "Commodity", "Commodities", "Oil", "Crude", "Futures", "Natural Gas")
_US_REALESTATE_KEYWORDS = ("Real Estate", "REIT")
_US_MULTIASSET_KEYWORDS = ("Multi-Asset", "Multi Asset", "Allocation", "Target Date", "Balanced")

_US_MARKET_KEYWORDS = [
    ("China", "China/Hong Kong"), ("Hong Kong", "China/Hong Kong"),
    ("Japan", "Japan"), ("India", "India"), ("Korea", "South Korea"),
    ("Europe", "Europe"), ("Emerging Markets", "Emerging Markets"),
    ("Brazil", "Brazil"), ("Latin America", "Latin America"),
    ("International", "International (ex-US)"), ("Global", "Global"),
    ("World", "Global"), ("Australia", "Australia"), ("Canada", "Canada"),
    ("Taiwan", "Taiwan"), ("Vietnam", "Vietnam"), ("ASEAN", "Southeast Asia"),
]


def _classify_us_name(name: str):
    issuer = next((k for k in _US_ISSUER_KEYWORDS if k.lower() in name.lower()), None)
    return_type = "Standard"
    for pattern, rtype in _US_LEVERAGED_INVERSE_KEYWORDS:
        if re.search(pattern, name, re.IGNORECASE):
            return_type = rtype
            break
    if any(k.lower() in name.lower() for k in _US_BOND_KEYWORDS):
        asset_class, asset_type = "Fixed Income", "Bond ETF"
    elif any(k.lower() in name.lower() for k in _US_COMMODITY_KEYWORDS):
        asset_class, asset_type = "Commodity", "Commodity ETF"
    elif any(k.lower() in name.lower() for k in _US_REALESTATE_KEYWORDS):
        asset_class, asset_type = "Real Estate", "REIT ETF"
    elif any(k.lower() in name.lower() for k in _US_MULTIASSET_KEYWORDS):
        asset_class, asset_type = "Multi-Asset", "Multi-Asset ETF"
    else:
        asset_class, asset_type = "Equity", "Equity ETF"
    underlying_market = next((v for k, v in _US_MARKET_KEYWORDS if k.lower() in name.lower()), "United States")
    return issuer, return_type, asset_class, asset_type, underlying_market


_US_EXCHANGE_NAMES = {"P": "NYSE Arca", "N": "NYSE", "Z": "Cboe BZX", "A": "NYSE American", "V": "IEX"}


def fetch_us_etfs():
    """Nasdaq Trader Symbol Directory:
    https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt (Nasdaq-listed)
    https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt   (NYSE / NYSE Arca / Cboe BZX)
    The industry-standard official symbol directory; both files carry an
    explicit ETF Y/N flag used as the filter (never a name/ticker guess)."""
    nasdaq_raw = _fetch_text("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")
    nasdaq_lines = [l for l in nasdaq_raw.strip().split("\n") if l and not l.startswith("File Creation Time")]
    nasdaq_header = nasdaq_lines[0].split("|")
    nasdaq_rows = [dict(zip(nasdaq_header, l.split("|"))) for l in nasdaq_lines[1:]]
    nasdaq_etfs = [r for r in nasdaq_rows if r.get("ETF") == "Y" and r.get("Test Issue") == "N"]

    other_raw = _fetch_text("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
    other_lines = [l for l in other_raw.strip().split("\n") if l and not l.startswith("File Creation Time")]
    other_header = other_lines[0].split("|")
    other_rows = [dict(zip(other_header, l.split("|"))) for l in other_lines[1:]]
    other_etfs = [r for r in other_rows if r.get("ETF") == "Y" and r.get("Test Issue") == "N"]

    out = []
    for r in nasdaq_etfs:
        out.append({"symbol": r["Symbol"].strip(), "name": r["Security Name"].strip(),
                     "exchange": "Nasdaq", "source": "Nasdaq Trader Symbol Directory (nasdaqlisted.txt)"})
    for r in other_etfs:
        symbol = r["ACT Symbol"].strip()
        if not symbol or "." in symbol or "$" in symbol:
            continue  # non-plain trading lines (units/warrants/rights) -- not itemized here
        out.append({"symbol": symbol, "name": r["Security Name"].strip(),
                     "exchange": _US_EXCHANGE_NAMES.get(r.get("Exchange", "").strip(), r.get("Exchange", "")),
                     "source": "Nasdaq Trader Symbol Directory (otherlisted.txt)"})
    return out


def build_us_snapshot(rows):
    out = []
    seen = set()
    for r in rows:
        symbol = r["symbol"]
        if symbol in seen:
            continue  # a security should appear in exactly one of the two directories; guard against any overlap
        seen.add(symbol)
        issuer, return_type, asset_class, asset_type, underlying_market = _classify_us_name(r["name"])
        out.append({
            "ticker": symbol,
            "name": r["name"],
            "listing_market": "United States",
            "exchange": r["exchange"],
            "currency": "USD",
            "asset_class": asset_class,
            "asset_type": asset_type,
            "underlying_market": underlying_market,
            # Nasdaq Trader's directory has no active/passive flag, and (unlike
            # Taiwan's "主動" naming convention) US active ETFs have no reliable
            # universal name marker -- left unknown rather than guessed.
            "management_style": None,
            "return_type": return_type,
            "issuer": issuer,
            "listing_date": None,
            "isin": None,
            "provider_symbol": symbol,
            "provider_status": "unknown",
            "source": r["source"],
            "source_updated_at": _TODAY,
        })
    return out


# ── United Kingdom / LSE ─────────────────────────────────────────────────
# CURATED, not a bulk feed -- see module docstring. Each entry is an
# individually-recognized real LSE-listed UCITS ETF trading line. Where the
# same underlying fund has multiple LSE trading lines (different currency
# and/or income vs accumulating share class), each line is its own record
# (own ticker, own yahoo/provider_symbol) but shares a `fund_group_id` so
# the UI can later avoid presenting them as unrelated, unconnected funds
# (PRODUCT SPEC section B5).
_UK_CURATED = [
    # ticker, name, issuer, asset_class, asset_type, underlying_market, currency, fund_group_id, return_type
    ("VUSA", "Vanguard S&P 500 UCITS ETF", "Vanguard", "Equity", "Equity ETF", "United States", "GBP", "vg-sp500", "Standard"),
    ("VUAG", "Vanguard S&P 500 UCITS ETF (USD Accumulating)", "Vanguard", "Equity", "Equity ETF", "United States", "GBP", "vg-sp500", "Standard"),
    ("VWRL", "Vanguard FTSE All-World UCITS ETF", "Vanguard", "Equity", "Equity ETF", "Global", "GBP", "vg-allworld", "Standard"),
    ("VWRP", "Vanguard FTSE All-World UCITS ETF (Accumulating)", "Vanguard", "Equity", "Equity ETF", "Global", "GBP", "vg-allworld", "Standard"),
    ("VFEM", "Vanguard FTSE Emerging Markets UCITS ETF", "Vanguard", "Equity", "Equity ETF", "Emerging Markets", "GBP", None, "Standard"),
    ("VMID", "Vanguard FTSE 250 UCITS ETF", "Vanguard", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("VUKE", "Vanguard FTSE 100 UCITS ETF", "Vanguard", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("VGOV", "Vanguard UK Gilt UCITS ETF", "Vanguard", "Fixed Income", "Bond ETF", "United Kingdom", "GBP", None, "Standard"),
    ("VECP", "Vanguard EUR Corporate Bond UCITS ETF", "Vanguard", "Fixed Income", "Bond ETF", "Europe", "GBP", None, "Standard"),
    ("VERX", "Vanguard FTSE Developed Europe ex UK UCITS ETF", "Vanguard", "Equity", "Equity ETF", "Europe", "GBP", None, "Standard"),
    ("CSPX", "iShares Core S&P 500 UCITS ETF", "iShares", "Equity", "Equity ETF", "United States", "GBP", "is-sp500", "Standard"),
    ("IUSA", "iShares Core S&P 500 UCITS ETF (Dist)", "iShares", "Equity", "Equity ETF", "United States", "GBP", "is-sp500", "Standard"),
    ("ISF", "iShares Core FTSE 100 UCITS ETF", "iShares", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("SWDA", "iShares Core MSCI World UCITS ETF", "iShares", "Equity", "Equity ETF", "Global", "USD", "is-world", "Standard"),
    ("EIMI", "iShares Core MSCI EM IMI UCITS ETF", "iShares", "Equity", "Equity ETF", "Emerging Markets", "USD", None, "Standard"),
    ("IGLT", "iShares Core UK Gilts UCITS ETF", "iShares", "Fixed Income", "Bond ETF", "United Kingdom", "GBP", None, "Standard"),
    ("INXG", "iShares GBP Index-Linked Gilts UCITS ETF", "iShares", "Fixed Income", "Bond ETF", "United Kingdom", "GBP", None, "Standard"),
    ("SGLN", "iShares Physical Gold ETC", "iShares", "Commodity", "Commodity ETF", "Global", "GBP", None, "Standard"),
    ("IUIT", "iShares S&P 500 Information Technology Sector UCITS ETF", "iShares", "Equity", "Equity ETF", "United States", "USD", None, "Standard"),
    ("EQQQ", "Invesco EQQQ Nasdaq-100 UCITS ETF", "Invesco", "Equity", "Equity ETF", "United States", "GBP", "iv-nasdaq100", "Standard"),
    ("SPY5", "Invesco S&P 500 UCITS ETF (Dist)", "Invesco", "Equity", "Equity ETF", "United States", "USD", None, "Standard"),
    ("FTAL", "Invesco FTSE All-Share UCITS ETF", "Invesco", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("FUSD", "Fidelity US Quality Income UCITS ETF", "Fidelity", "Equity", "Equity ETF", "United States", "USD", None, "Standard"),
    ("FGQI", "Fidelity Global Quality Income UCITS ETF", "Fidelity", "Equity", "Equity ETF", "Global", "USD", None, "Standard"),
    ("XDWT", "Xtrackers MSCI World Information Technology UCITS ETF", "Xtrackers (DWS)", "Equity", "Equity ETF", "Global", "USD", None, "Standard"),
    ("XDWD", "Xtrackers MSCI World UCITS ETF", "Xtrackers (DWS)", "Equity", "Equity ETF", "Global", "USD", None, "Standard"),
    ("XESC", "Xtrackers S&P 500 Equal Weight UCITS ETF", "Xtrackers (DWS)", "Equity", "Equity ETF", "United States", "USD", None, "Standard"),
    ("CNDX", "iShares Nasdaq 100 UCITS ETF", "iShares", "Equity", "Equity ETF", "United States", "GBP", None, "Standard"),
    ("CUKX", "iShares FTSE 100 UCITS ETF (Acc)", "iShares", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("IUKD", "iShares UK Dividend UCITS ETF", "iShares", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("AGGG", "iShares Core Global Aggregate Bond UCITS ETF", "iShares", "Fixed Income", "Bond ETF", "Global", "GBP", None, "Standard"),
    ("EMHG", "iShares JP Morgan EM Bond UCITS ETF (GBP Hedged)", "iShares", "Fixed Income", "Bond ETF", "Emerging Markets", "GBP", None, "Standard"),
    ("PHGP", "WisdomTree Physical Gold", "WisdomTree", "Commodity", "Commodity ETF", "Global", "GBP", None, "Standard"),
    ("PHAU", "WisdomTree Physical Gold (USD)", "WisdomTree", "Commodity", "Commodity ETF", "Global", "USD", "wt-gold", "Standard"),
    ("PHSP", "WisdomTree Physical Silver", "WisdomTree", "Commodity", "Commodity ETF", "Global", "GBP", None, "Standard"),
    ("AMEU", "Amundi MSCI Europe UCITS ETF", "Amundi", "Equity", "Equity ETF", "Europe", "EUR", None, "Standard"),
    ("LCUK", "Amundi FTSE 100 UCITS ETF", "Amundi", "Equity", "Equity ETF", "United Kingdom", "GBP", None, "Standard"),
    ("IJPN", "iShares Core MSCI Japan IMI UCITS ETF", "iShares", "Equity", "Equity ETF", "Japan", "GBP", None, "Standard"),
    ("EMIM", "iShares Core MSCI Emerging Markets IMI UCITS ETF", "iShares", "Equity", "Equity ETF", "Emerging Markets", "USD", None, "Standard"),
    ("VJPN", "Vanguard FTSE Japan UCITS ETF", "Vanguard", "Equity", "Equity ETF", "Japan", "GBP", None, "Standard"),
]


def build_uk_snapshot():
    out = []
    for ticker, name, issuer, asset_class, asset_type, underlying_market, currency, fund_group_id, return_type in _UK_CURATED:
        out.append({
            "ticker": ticker,
            "name": name,
            "listing_market": "United Kingdom",
            "exchange": "London Stock Exchange (LSE)",
            "currency": currency,
            "asset_class": asset_class,
            "asset_type": asset_type,
            "underlying_market": underlying_market,
            "management_style": "Passive",
            "return_type": return_type,
            "issuer": issuer,
            "listing_date": None,
            "isin": None,
            "provider_symbol": f"{ticker}.L",
            "provider_status": "unknown",
            "fund_group_id": fund_group_id,
            "source": "Curated (individually verified -- no bulk authoritative LSE feed available)",
            "source_updated_at": _TODAY,
        })
    return out


def main():
    os.makedirs(_OUT_DIR, exist_ok=True)

    print("Fetching Taiwan (TWSE + TPEx) ETF universe...")
    tw_rows = fetch_taiwan_etfs()
    tw_snapshot = build_taiwan_snapshot(tw_rows)
    twse_n = sum(1 for r in tw_snapshot if "TWSE" in r["exchange"])
    tpex_n = sum(1 for r in tw_snapshot if "TPEx" in r["exchange"])
    print(f"  TWSE: {twse_n}  TPEx: {tpex_n}  total: {len(tw_snapshot)}")
    with open(os.path.join(_OUT_DIR, "taiwan.json"), "w", encoding="utf-8") as f:
        json.dump(tw_snapshot, f, ensure_ascii=False, indent=1)

    print("Fetching United States ETF universe...")
    us_rows = fetch_us_etfs()
    us_snapshot = build_us_snapshot(us_rows)
    print(f"  total (deduplicated): {len(us_snapshot)}")
    with open(os.path.join(_OUT_DIR, "united_states.json"), "w", encoding="utf-8") as f:
        json.dump(us_snapshot, f, ensure_ascii=False, indent=1)

    print("Building United Kingdom / LSE ETF universe (curated)...")
    uk_snapshot = build_uk_snapshot()
    print(f"  total (curated): {len(uk_snapshot)}")
    with open(os.path.join(_OUT_DIR, "united_kingdom.json"), "w", encoding="utf-8") as f:
        json.dump(uk_snapshot, f, ensure_ascii=False, indent=1)

    meta = {
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "taiwan_twse": twse_n, "taiwan_tpex": tpex_n, "taiwan_total": len(tw_snapshot),
            "united_states_total": len(us_snapshot),
            "united_kingdom_total": len(uk_snapshot),
        },
    }
    with open(os.path.join(_OUT_DIR, "_refresh_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("Done.", meta["counts"])


if __name__ == "__main__":
    main()
