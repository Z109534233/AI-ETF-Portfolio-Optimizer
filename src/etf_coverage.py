"""
ETF price-coverage audit snapshot schema + parser (Issue #45 item 8).

Separates "registered/searchable ETF universe size" (len(get_all_tickers()))
from "verified fetchable from the live price provider". The latter is ONLY
ever a real, reproducible measurement produced by
scripts/audit_etf_price_coverage.py -- never a number computed, guessed, or
invented at app-startup or import time. When no snapshot file exists yet,
callers must show that plainly rather than fabricating a count.
"""

import json
import os

COVERAGE_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "etf_universe", "price_coverage_summary.json",
)

REQUIRED_FIELDS = (
    "provider", "audited_at", "universe_size", "attempted_count",
    "available_count", "unavailable_count", "status",
)

VALID_STATUSES = ("COMPLETE", "PARTIAL")


def load_coverage_snapshot(path: str = None):
    """Load and validate a price-coverage snapshot JSON file. Returns None
    if the file doesn't exist, isn't valid JSON, is missing a required
    field, or has an unrecognized `status` -- callers must treat None as
    "no snapshot available yet" and never fabricate a count in its place.
    """
    path = path or COVERAGE_SNAPSHOT_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not all(field in data for field in REQUIRED_FIELDS):
        return None
    if data["status"] not in VALID_STATUSES:
        return None
    return data


def coverage_display_stat(snapshot, lang: str = "en"):
    """Compact (value, label) tuple for Home's low-emphasis coverage stat,
    or None if no valid snapshot exists (Home must not show ANY coverage
    count in that case). `snapshot` is load_coverage_snapshot()'s return
    value.

    - COMPLETE audit -> "已驗證價格資料 X / Y" / "Verified Price Coverage X / Y"
      (Y = the full registered universe size at audit time).
    - PARTIAL audit -> "價格覆蓋抽查 X / N（部分驗證）" / "Price Coverage Sample
      X / N (Partial)" (N = tickers actually attempted so far, not the full
      universe).
    """
    if snapshot is None:
        return None
    available = snapshot["available_count"]
    attempted = snapshot["attempted_count"]
    if snapshot["status"] == "COMPLETE":
        label = "已驗證價格資料" if lang == "zh-TW" else "Verified Price Coverage"
        return f"{available} / {snapshot['universe_size']}", label
    label = "價格覆蓋抽查（部分驗證）" if lang == "zh-TW" else "Price Coverage Sample (Partial)"
    return f"{available} / {attempted}", label


def coverage_unavailable_caption(lang: str = "en") -> str:
    """Caption shown on Home when no coverage snapshot exists at all --
    never displayed alongside a real coverage_display_stat()."""
    if lang == "zh-TW":
        return "價格覆蓋尚未完成全量驗證。"
    return "Price coverage has not yet been fully verified."


__all__ = [
    "COVERAGE_SNAPSHOT_PATH", "REQUIRED_FIELDS", "VALID_STATUSES",
    "load_coverage_snapshot", "coverage_display_stat", "coverage_unavailable_caption",
]
