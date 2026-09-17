"""
Live risk-free rate service (Issue #45 item 3).

Primary proxy: FRED series DGS3MO -- "Market Yield on U.S. Treasury
Securities at 3-Month Constant Maturity, Quoted on an Investment Basis"
(daily, percent per annum; FRED / Federal Reserve H.15).

This fetches the PUBLIC FRED graph CSV export
(fred.stlouisfed.org/graph/fredgraph.csv) rather than the keyed FRED Web
API v1 -- this is a public data export, not an authenticated API call, and
needs no API key/secret to be committed or configured anywhere.

Every public function here is safe to call with no network access: a
request failure, timeout, or an unparseable/empty response always falls
back to DEFAULT_FALLBACK_RATE with status="fallback" and a stated reason,
never raises, and never crashes the app. Callers must show `status`
prominently -- a fallback rate must never be presented as if it were a
current live rate.
"""

import requests
import streamlit as st

FRED_SERIES_ID = "DGS3MO"
FRED_SOURCE_NAME = "FRED"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"

# Unchanged from the app's pre-existing manual default (5%) -- used ONLY
# when a live FRED observation genuinely cannot be obtained. This is a
# fallback of last resort, never a claim about the current market rate.
DEFAULT_FALLBACK_RATE = 0.05

REQUEST_TIMEOUT_SECONDS = 5.0
CACHE_TTL_SECONDS = 6 * 3600  # DGS3MO updates at most once per trading day


def _http_get_fred_csv(timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Isolated network call so tests can mock exactly this boundary --
    the real `requests.get()` is never invoked in the unit test suite."""
    response = requests.get(
        FRED_CSV_URL, timeout=timeout,
        headers={"User-Agent": "ai-etf-portfolio-optimizer/1.0"},
    )
    response.raise_for_status()
    return response.text


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
    """Fetch the latest non-missing DGS3MO observation. Always returns a
    dict, never raises:

        {"rate": float (decimal, e.g. 0.0411), "observed_date": "YYYY-MM-DD" or None,
         "series_id": "DGS3MO", "source": "FRED",
         "status": "live" | "fallback", "reason": str or None}

    `status="fallback"` means live FRED data could not be obtained (network
    failure, timeout, or no usable observation in the response) -- callers
    must show this as a stale/default rate, never as if it were current.

    This function itself is NOT Streamlit-cached (see
    get_cached_risk_free_rate() for the cached wrapper pages actually call)
    so it can be unit-tested directly with a mocked _http_get_fred_csv(),
    with no real network access.
    """
    try:
        csv_text = _http_get_fred_csv(timeout=timeout)
    except Exception as e:
        return _fallback(f"FRED request failed: {type(e).__name__}: {e}")

    parsed = _parse_latest_observation(csv_text)
    if parsed is None:
        return _fallback("FRED response contained no usable (non-missing) observation")

    observed_date, rate = parsed
    return {
        "rate": rate,
        "observed_date": observed_date,
        "series_id": FRED_SERIES_ID,
        "source": FRED_SOURCE_NAME,
        "status": "live",
        "reason": None,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_cached_risk_free_rate() -> dict:
    """Streamlit-cached wrapper -- the ONE function every page should call
    so a Streamlit rerun never re-hits FRED on every rerun, and every page
    reads the exact same rate/observed_date/status within a cache window
    (cross-page consistency, Issue #45 item 3)."""
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
    """Compact, dynamic provenance string for methodology/sidebar captions,
    e.g. "X.XX% · FRED DGS3MO · YYYY-MM-DD" when live (the actual rate and
    date are whatever `rf` holds -- never hard-coded here), or an explicit
    fallback disclosure when the live fetch failed.

    `selected_rate` (decimal) is the ACTUAL rate a page will use (e.g. its
    risk-free-rate slider's current value) after the user may have moved it
    away from the fetched/fallback default it was initialized to. When
    given and it differs from `rf['rate']` (see is_manual_override()), the
    returned caption states the assumption is a manual override -- NOT
    FRED-sourced -- and shows the FRED reference rate/date separately, so a
    moved slider is never captioned as if it were still the live FRED
    value (Issue #46 review item 2)."""
    pct = f"{rf['rate'] * 100:.2f}%"
    if rf.get("status") == "live" and rf.get("observed_date"):
        fred_desc = f"{pct} · {rf['source']} {rf['series_id']} · {rf['observed_date']}"
    elif lang == "zh-TW":
        fred_desc = f"{pct}（預設備用值，無法取得即時 {rf['series_id']} 資料）"
    else:
        fred_desc = f"{pct} (fallback default -- live {rf['series_id']} data unavailable)"

    if selected_rate is not None and is_manual_override(rf, selected_rate):
        selected_pct = f"{selected_rate * 100:.2f}%"
        if lang == "zh-TW":
            return f"{selected_pct}（使用者手動調整，非即時 FRED 數值；FRED 參考值：{fred_desc}）"
        return f"{selected_pct} (manually overridden -- not FRED-sourced; FRED reference: {fred_desc})"

    return fred_desc


__all__ = [
    "FRED_SERIES_ID", "FRED_SOURCE_NAME", "DEFAULT_FALLBACK_RATE",
    "get_risk_free_rate", "get_cached_risk_free_rate", "format_rate_provenance",
    "is_manual_override",
]
