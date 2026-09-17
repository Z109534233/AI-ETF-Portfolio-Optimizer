#!/usr/bin/env python3
"""
Audit which ETFs in the registered universe (src.etf_database) actually
return usable price data from the live provider (Yahoo Finance via
yfinance), using the SAME to_yahoo_symbol() mapping/provider semantics the
app itself uses (Issue #45 item 8).

This is NOT run automatically by the app -- it is a standalone,
chunkable, resumable, rate-limit-friendly CLI tool a maintainer runs
manually (or on a slow schedule) to produce a reproducible
data/etf_universe/price_coverage_summary.json snapshot. Auditing the full
registered universe (thousands of tickers) against a live network takes a
long time and must never happen on every app startup.

Usage:
    python scripts/audit_etf_price_coverage.py --start 0 --limit 500 --sleep 0.5
    python scripts/audit_etf_price_coverage.py --resume --limit 500  # continue a PARTIAL snapshot

Status categories per ticker (a transient failure is never treated as
mathematically proven "ticker unsupported"):
    available       -- at least one usable close price was returned
    no_data         -- the request succeeded but returned no rows (the
                        provider has no data for this symbol/window)
    transient_error -- the request raised an exception (timeout, rate
                        limit, connection error, etc.) -- NOT proof the
                        ticker is unsupported, just that this attempt failed
    not_attempted   -- not yet reached in this run (present in a PARTIAL
                        snapshot's registered universe, absent from
                        ticker_status)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etf_database import get_all_tickers, get_country, to_yahoo_symbol  # noqa: E402

DEFAULT_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "etf_universe", "price_coverage_summary.json",
)

DEFAULT_LOOKBACK_DAYS = 10
PROVIDER_NAME = "Yahoo Finance (yfinance)"


def attempt_fetch(yahoo_symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> str:
    """One ticker's fetch attempt against the live provider. Returns
    "available" / "no_data" / "transient_error" -- never raises. Imports
    yfinance lazily so this module can be imported (e.g. by tests exercising
    build_snapshot()) without requiring network access or the yfinance
    package's import-time side effects."""
    try:
        import yfinance as yf
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days * 3)  # pad for weekends/holidays
        data = yf.download(
            tickers=yahoo_symbol,
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False, threads=False,
        )
        if data is None or data.empty:
            return "no_data"
        return "available"
    except Exception:
        return "transient_error"


def load_existing_snapshot(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return None


def build_snapshot(tickers_with_market: list, results: dict, universe_size: int,
                    complete: bool, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                    audited_at: str = None) -> dict:
    """Pure aggregation of per-ticker status into the machine-readable
    snapshot schema -- deterministic and testable without any network
    access. `tickers_with_market` is [(ticker, market_or_None), ...] for
    the FULL registered universe (not just this run's batch);
    `results` is {ticker: status} for every ticker attempted so far
    (across all runs, if resuming).
    """
    per_market = {}
    available_count = 0
    unavailable_count = 0
    attempted_count = 0
    for ticker, market in tickers_with_market:
        status = results.get(ticker, "not_attempted")
        bucket = per_market.setdefault(market or "Unknown", {"attempted": 0, "available": 0, "unavailable": 0})
        if status == "not_attempted":
            continue
        attempted_count += 1
        bucket["attempted"] += 1
        if status == "available":
            available_count += 1
            bucket["available"] += 1
        else:
            unavailable_count += 1
            bucket["unavailable"] += 1

    return {
        "provider": PROVIDER_NAME,
        "audited_at": audited_at or datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "universe_size": universe_size,
        "attempted_count": attempted_count,
        "available_count": available_count,
        "unavailable_count": unavailable_count,
        "per_market": per_market,
        "status": "COMPLETE" if complete else "PARTIAL",
        "ticker_status": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=0, help="Start index into the registered universe (ignored with --resume)")
    parser.add_argument("--limit", type=int, default=200, help="Max number of NEW tickers to attempt this run")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests (rate-limit friendly)")
    parser.add_argument("--resume", action="store_true", help="Continue from the existing snapshot's ticker_status instead of starting over")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--out", type=str, default=DEFAULT_SNAPSHOT_PATH)
    args = parser.parse_args(argv)

    all_tickers = get_all_tickers()
    universe_size = len(all_tickers)
    tickers_with_market = [(tk, get_country(tk)) for tk in all_tickers]

    existing = load_existing_snapshot(args.out) if args.resume else None
    results = dict(existing.get("ticker_status", {})) if existing else {}

    remaining = [tk for tk in all_tickers if tk not in results]
    batch = remaining[:args.limit] if args.resume else all_tickers[args.start:args.start + args.limit]

    for ticker in batch:
        if ticker in results:
            continue
        yahoo_symbol = to_yahoo_symbol(ticker)
        results[ticker] = attempt_fetch(yahoo_symbol, args.lookback_days)
        if args.sleep > 0:
            time.sleep(args.sleep)

    complete = all(tk in results for tk in all_tickers)
    snapshot = build_snapshot(tickers_with_market, results, universe_size, complete, args.lookback_days)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(
        f"Wrote {args.out}: status={snapshot['status']} "
        f"available={snapshot['available_count']} attempted={snapshot['attempted_count']} "
        f"universe_size={universe_size}"
    )
    return snapshot


if __name__ == "__main__":
    main()
