"""
Live risk-free rate service (Issue #45 item 3; hardened for hosted
production environments -- e.g. Render -- in Issue #48 item 1).

Primary proxy: FRED series DGS3MO -- "Market Yield on U.S. Treasury
Securities at 3-Month Constant Maturity, Quoted on an Investment Basis"
(daily, percent per annum; FRED / Federal Reserve H.15).

This fetches the PUBLIC FRED graph CSV export
(fred.stlouisfed.org/graph/fredgraph.csv) rather than the keyed FRED Web
API v1 -- this is a public data export, not an authenticated API call, and
needs no API key/secret to be committed or configured anywhere.

Secondary proxy: Yahoo Finance ticker ^IRX (13-Week Treasury Bill, quoted
as a discount-basis annualized percent). Used ONLY when live FRED DGS3MO
cannot be obtained -- it is a distinct instrument on a distinct quoting
convention from DGS3MO and is always labeled as a secondary proxy, never
as if it were FRED data.

Every public function here is safe to call with no network access: a
request failure, timeout, or an unparseable/empty response always falls
through to the next source in the chain (FRED -> ^IRX -> the fixed
emergency default), never raises, and never crashes the app. Callers must
show `status` prominently -- a fallback rate must never be presented as if
it were a current live rate, and a secondary proxy must never be presented
as if it were FRED DGS3MO.
"""

import logging
import time

import requests
import streamlit as st

logger = logging.getLogger(__name__)

FRED_SERIES_ID = "DGS3MO"
FRED_SOURCE_NAME = "FRED"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"

# Secondary live proxy (Issue #48 item 1): Yahoo Finance's ^IRX, the
# 13-Week Treasury Bill rate. Never presented as FRED/DGS3MO -- always
# tagged with its own source/series identity below.
IRX_TICKER = "^IRX"
IRX_SOURCE_NAME = "Yahoo Finance"
IRX_FETCH_PERIOD = "5d"

# Unchanged from the app's pre-existing manual default (5%) -- used ONLY
# when BOTH live sources (FRED and the ^IRX secondary proxy) genuinely
# cannot be obtained. This is a fallback of last resort, never a claim
# about the current market rate.
DEFAULT_FALLBACK_RATE = 0.05

# Increased from 5.0s -- hosted platforms (e.g. Render) routinely see
# higher outbound-request latency to fred.stlouisfed.org than local
# development, and a too-tight timeout was the confirmed root cause of
# production always landing on the emergency fallback (Issue #48 item 1).
REQUEST_TIMEOUT_SECONDS = 15.0
CACHE_TTL_SECONDS = 6 * 3600  # DGS3MO updates at most once per trading day

# Bounded retry for transient failures only (timeout, connection error, or
# a 5xx/429 response) -- never for a 4xx client error, which will not
# succeed on retry.
MAX_FETCH_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# A plain, browser-like User-Agent + Accept header pair -- some hosted
# environments' default `python-requests/x.y` User-Agent is more likely to
# be rate-limited or blocked (403) by edge/CDN layers in front of FRED than
# a request that looks like it came from a browser.
_FRED_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
}


def _http_get_fred_csv(timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Isolated network call so tests can mock exactly this boundary --
    the real `requests.get()` is never invoked in the unit test suite.

    Retries transient timeout/connection/5xx/429 failures up to
    MAX_FETCH_ATTEMPTS times with a small bounded linear backoff before
    giving up; a non-retryable HTTP error (e.g. 404) raises immediately.
    Validates the response status and a non-empty body before returning."""
    last_exc = None
    for attempt in range(MAX_FETCH_ATTEMPTS):
        is_last_attempt = attempt == MAX_FETCH_ATTEMPTS - 1
        try:
            response = requests.get(FRED_CSV_URL, timeout=timeout, headers=_FRED_REQUEST_HEADERS)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if is_last_attempt:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue

        if response.status_code in _RETRYABLE_STATUS_CODES and not is_last_attempt:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue

        response.raise_for_status()
        text = response.text
        if not text or not text.strip():
            raise ValueError("FRED response body was empty")
        return text

    # Unreachable in practice (the loop always returns or raises above),
    # but keeps this function's return type honest for static analysis.
    raise last_exc if last_exc else RuntimeError("FRED fetch failed with no captured exception")


def _parse_latest_observation(csv_text: str):
    """Parse the two-column `DATE,DGS3MO` FRED graph CSV export and return
    (observed_date_str, rate_decimal) for the LATEST row whose value is not
    missing, or None if every row is missing ('.', FRED's own missing-data
    marker for holidays/no-observation days) or the text can't be parsed at
    all. Never raises."""
    if not csv_text:
        return None
    lines = [ln.strip() for ln in csv_text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    latest = None
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        observed_date, value_str = parts[0].strip(), parts[1].strip()
        if not value_str or value_str == ".":
            continue
        try:
            rate_pct = float(value_str)
        except ValueError:
            continue
        latest = (observed_date, rate_pct / 100.0)
    return latest


def _fetch_fred(timeout: float = REQUEST_TIMEOUT_SECONDS):
    """Attempt the live FRED DGS3MO fetch. Returns (result_dict, None) on
    success, or (None, reason_str) on failure -- never raises. Logs a
    concise `logging.warning` (exception type + message, no stack trace)
    on failure so hosted-platform logs (e.g. Render Logs) can diagnose
    DNS/timeout/SSL/403/parse failures without exposing internals to
    users."""
    try:
        csv_text = _http_get_fred_csv(timeout=timeout)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        logging.warning("FRED fetch failed: %s", reason)
        return None, reason

    parsed = _parse_latest_observation(csv_text)
    if parsed is None:
        reason = "response contained no usable (non-missing) observation"
        logging.warning("FRED fetch failed: %s", reason)
        return None, reason

    observed_date, rate = parsed
    return {
        "rate": rate,
        "observed_date": observed_date,
        "series_id": FRED_SERIES_ID,
        "source": FRED_SOURCE_NAME,
        "status": "live",
        "reason": None,
    }, None


def _http_get_irx_history(period: str = IRX_FETCH_PERIOD):
    """Isolated network call for the secondary ^IRX live proxy so tests can
    mock exactly this boundary -- the real `yfinance` download is never
    invoked in the unit test suite. Imports yfinance lazily so importing
    this module never requires network access or a working yfinance
    install just to read the fallback constants."""
    import yfinance as yf

    return yf.download(IRX_TICKER, period=period, progress=False, auto_adjust=False, threads=False)


def _parse_latest_irx_observation(history):
    """Extract (observed_date_str, rate_decimal) from a yfinance history
    DataFrame for ^IRX, using the latest valid (non-NaN) Close observation
    in the short recent window. ^IRX is quoted as a percent (e.g. 4.20
    meaning 4.20%), so the value is divided by 100 to convert to the same
    decimal convention as the FRED path (e.g. 0.0420). Returns None if the
    frame is empty/missing or every Close observation is NaN. Never
    raises."""
    if history is None or getattr(history, "empty", True):
        return None
    if "Close" not in history.columns:
        return None
    closes = history["Close"]
    if hasattr(closes, "columns"):  # defensive: a stray MultiIndex column
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    if closes.empty:
        return None

    last_index = closes.index[-1]
    observed_date = last_index.strftime("%Y-%m-%d") if hasattr(last_index, "strftime") else str(last_index)
    rate_pct = float(closes.iloc[-1])
    return observed_date, rate_pct / 100.0


def _fetch_irx_secondary():
    """Attempt the secondary ^IRX live proxy fetch. Returns (result_dict,
    None) on success, or (None, reason_str) on failure -- never raises.
    Logs a concise `logging.warning` on failure, mirroring _fetch_fred()."""
    try:
        history = _http_get_irx_history()
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        logging.warning("Secondary risk-free-rate proxy (Yahoo Finance %s) fetch failed: %s", IRX_TICKER, reason)
        return None, reason

    parsed = _parse_latest_irx_observation(history)
    if parsed is None:
        reason = "response contained no usable Close observation"
        logging.warning("Secondary risk-free-rate proxy (Yahoo Finance %s) fetch failed: %s", IRX_TICKER, reason)
        return None, reason

    observed_date, rate = parsed
    return {
        "rate": rate,
        "observed_date": observed_date,
        "series_id": IRX_TICKER,
        "source": IRX_SOURCE_NAME,
        # Deliberately NOT "live" -- "secondary_live" so callers/captions
        # can never conflate this with a primary FRED DGS3MO observation
        # (Issue #48 item 1: never label ^IRX as FRED DGS3MO).
        "status": "secondary_live",
        "reason": None,
    }, None


def _fallback(reason: str) -> dict:
    return {
        "rate": DEFAULT_FALLBACK_RATE,
        "observed_date": None,
        "series_id": FRED_SERIES_ID,
        "source": FRED_SOURCE_NAME,
        "status": "fallback",
        "reason": reason,
    }


def get_risk_free_rate(timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict:
    """Fetch the current risk-free-rate reference, trying each source in
    order and never raising:

        1. Live FRED DGS3MO (primary).
        2. Live Yahoo Finance ^IRX, a secondary 13-week T-bill proxy.
        3. The fixed DEFAULT_FALLBACK_RATE emergency default.

    Always returns a dict:

        {"rate": float (decimal, e.g. 0.0411), "observed_date": "YYYY-MM-DD" or None,
         "series_id": "DGS3MO" | "^IRX", "source": "FRED" | "Yahoo Finance",
         "status": "live" | "secondary_live" | "fallback", "reason": str or None}

    `status="live"` is FRED DGS3MO. `status="secondary_live"` is the ^IRX
    proxy -- callers must show this distinctly from FRED, never as if it
    were the primary series. `status="fallback"` means BOTH live sources
    failed -- callers must show this as a stale/default rate, never as if
    it were current.

    The `timeout` argument only applies to the FRED attempt (the ^IRX path
    is a separate library call with its own internal timeout handling).

    This function itself is NOT Streamlit-cached (see
    get_cached_risk_free_rate() for the cached wrapper pages actually call)
    so it can be unit-tested directly with mocked _http_get_fred_csv() /
    _http_get_irx_history(), with no real network access.
    """
    fred_result, fred_reason = _fetch_fred(timeout=timeout)
    if fred_result is not None:
        return fred_result

    irx_result, irx_reason = _fetch_irx_secondary()
    if irx_result is not None:
        return irx_result

    combined_reason = f"FRED DGS3MO: {fred_reason}; Yahoo Finance {IRX_TICKER}: {irx_reason}"
    logging.warning("All live risk-free-rate sources failed, using emergency fallback: %s", combined_reason)
    return _fallback(combined_reason)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_cached_risk_free_rate() -> dict:
    """Streamlit-cached wrapper -- the ONE function every page should call
    so a Streamlit rerun never re-hits FRED/^IRX on every rerun, and every
    page reads the exact same rate/observed_date/status within a cache
    window (cross-page consistency, Issue #45 item 3)."""
    return get_risk_free_rate()


def is_manual_override(rf: dict, selected_rate: float) -> bool:
    """True if `selected_rate` (decimal, e.g. a page's risk-free-rate
    slider value) differs from the fetched/fallback `rf['rate']` at the
    slider's own display precision (2 decimal places as a percentage).

    Every page's slider is initialized to `round(rf['rate'] * 100, 2)`, so
    an untouched slider compares equal here; moving it away from that
    initial value (the only way a page's risk_free_rate can differ from
    rf['rate']) is what makes the assumption a manual override rather than
    the live/fallback FRED value (Issue #46 review item 2)."""
    return round(selected_rate * 100, 2) != round(rf["rate"] * 100, 2)


def format_rate_provenance(rf: dict, lang: str = "en", selected_rate: float = None) -> str:
    """Compact, dynamic provenance string for methodology/sidebar captions.
    The returned string always states the applicable rate's percentage
    EXACTLY ONCE -- callers must never wrap it in another string that also
    prints the same percentage again, or the result nests/duplicates it
    (Issue #48 item 2). Forms:

    - Live FRED: "X.XX% · FRED DGS3MO · YYYY-MM-DD".
    - Secondary ^IRX proxy: "X.XX% · Yahoo Finance ^IRX · YYYY-MM-DD ·
      secondary proxy" (localized), truthfully labeled as a proxy, never
      as FRED.
    - Emergency fallback: an explicit fallback disclosure with no as-of
      date (none exists).

    `selected_rate` (decimal) is the ACTUAL rate a page will use (e.g. its
    risk-free-rate slider's current value) after the user may have moved it
    away from the fetched/fallback default it was initialized to. When
    given and it differs from `rf['rate']` (see is_manual_override()), the
    returned caption states the assumption is a manual override -- NOT
    sourced from either live source -- and shows the reference rate/date
    separately, so a moved slider is never captioned as if it were still
    the live reference value (Issue #46 review item 2)."""
    pct = f"{rf['rate'] * 100:.2f}%"
    status = rf.get("status")
    if status == "live" and rf.get("observed_date"):
        reference_desc = f"{pct} · {rf['source']} {rf['series_id']} · {rf['observed_date']}"
    elif status == "secondary_live" and rf.get("observed_date"):
        if lang == "zh-TW":
            reference_desc = f"{pct} · {rf['source']} {rf['series_id']} · {rf['observed_date']} · 次要代理指標"
        else:
            reference_desc = f"{pct} · {rf['source']} {rf['series_id']} · {rf['observed_date']} · secondary proxy"
    elif lang == "zh-TW":
        reference_desc = f"{pct}（預設備用值，無法取得即時 {rf['series_id']} 資料）"
    else:
        reference_desc = f"{pct} (fallback default -- live {rf['series_id']} data unavailable)"

    if selected_rate is not None and is_manual_override(rf, selected_rate):
        selected_pct = f"{selected_rate * 100:.2f}%"
        if lang == "zh-TW":
            return f"{selected_pct}（使用者手動調整，非即時數值；參考值：{reference_desc}）"
        return f"{selected_pct} (manually overridden -- not from a live source; reference: {reference_desc})"

    return reference_desc


__all__ = [
    "FRED_SERIES_ID", "FRED_SOURCE_NAME", "IRX_TICKER", "IRX_SOURCE_NAME",
    "DEFAULT_FALLBACK_RATE", "get_risk_free_rate", "get_cached_risk_free_rate",
    "format_rate_provenance", "is_manual_override",
]
