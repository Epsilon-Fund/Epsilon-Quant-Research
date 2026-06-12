#!/usr/bin/env python3
"""Block S5f — overnight gap-filler for the SPCX PM-PDF dashboard.

The live monitor (`spcx_pm_pdf_monitor.py`) only writes a parquet shard while it is
actually running, so any window where nobody was polling (you were asleep) is a hole in
the dashboard time series. Both upstream venues, however, expose *historical* endpoints
the live monitor never calls:

  * Hyperliquid `candleSnapshot` → the xyz:SPCX perp mark, OHLC, 24/7.
  * Polymarket CLOB `/prices-history` → each strike/bucket token's midpoint time series.

This script replays those histories onto a regular time grid, reconstructs a synthetic
snapshot at each grid point (same dict shape `build_snapshot` produces live), and runs it
through the SAME `analyze()` + `log_parquet()` path the live monitor uses. The shards land
in the same directory with filenames keyed off the historical timestamp, so:

  * the dashboard's `backfill_from_parquet(days=N)` picks them up with zero changes;
  * re-running is idempotent (same timestamp → same filename → overwrite);
  * the math (survivor fit, EV, percentiles) is byte-identical to a live poll.

LOOKAHEAD-FREE by construction: the snapshot at grid time t forward-fills each token's
last quote with timestamp <= t. No future information ever leaks into a past point.

FIDELITY CAVEAT (be honest about it): live polls read the best executable BID/ASK off the
live book; this backfill only has the CLOB *midpoint* history, so it sets bid = ask = mid.
For an overnight TREND view that is fine, but a backfilled point is a midpoint
reconstruction, a notch lower fidelity than a live best-bid/ask poll. Spot is NOT
backfilled — SPCX is not a listed equity until listing day, so there is no overnight price.

Usage (from polymarket/research, PYTHONPATH=.):
    uv run python scripts/spcx_backfill_history.py --hours 48            # fill last 48h
    uv run python scripts/spcx_backfill_history.py --hours 48 --dry-run  # report only
    uv run python scripts/spcx_backfill_history.py \
        --start 2026-06-10T17:00 --end 2026-06-11T07:00 --step-min 10

Then relaunch the dashboard with a backfill window that covers it:
    uv run python scripts/spcx_pm_pdf_monitor.py --serve 8642 --backfill-days 2 ...
"""
from __future__ import annotations

import argparse
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import httpx

from scripts.spcx_pm_pdf_monitor import (
    CLOB_BASE,
    HL_INFO_URL,
    PARQUET_LOG_DIR,
    analyze,
    fetch_pm_metadata,
    log_parquet,
)

HL_COIN = "xyz:SPCX"


# --------------------------------------------------------------------------------------
# Historical fetch layer (read-only)
# --------------------------------------------------------------------------------------
def fetch_hl_candles(start_s: int, end_s: int, interval: str = "5m",
                     timeout: float = 30.0) -> list[tuple[float, float]]:
    """Hyperliquid candle closes for the perp → sorted [(ts_sec, close)]. Empty on failure."""
    try:
        r = httpx.post(HL_INFO_URL, timeout=timeout, json={
            "type": "candleSnapshot",
            "req": {"coin": HL_COIN, "interval": interval,
                    "startTime": start_s * 1000, "endTime": end_s * 1000}})
        r.raise_for_status()
        rows = r.json()
    except Exception as e:  # noqa: BLE001 — degrade, the perp line just stays empty
        print(f"[hl] candle fetch failed: {e!r}", file=sys.stderr)
        return []
    out = [(row["t"] / 1000.0, float(row["c"])) for row in rows if "t" in row]
    out.sort()
    return out


def fetch_token_history(token: str, start_s: int, end_s: int, fidelity_min: int = 10,
                        timeout: float = 30.0) -> list[tuple[float, float]]:
    """CLOB midpoint history for one outcome token → sorted [(ts_sec, price)]. Empty on
    failure (that token simply has no backfilled quote; the strike is dropped if <4 remain)."""
    try:
        r = httpx.get(f"{CLOB_BASE}/prices-history", timeout=timeout,
                      params={"market": token, "startTs": start_s, "endTs": end_s,
                              "fidelity": fidelity_min})
        r.raise_for_status()
        hist = r.json().get("history", [])
    except Exception as e:  # noqa: BLE001
        print(f"[clob] history fetch failed for {token[:14]}…: {e!r}", file=sys.stderr)
        return []
    out = [(float(p["t"]), float(p["p"])) for p in hist if "t" in p and "p" in p]
    out.sort()
    return out


# --------------------------------------------------------------------------------------
# Lookahead-free forward-fill + synthetic snapshot assembly
# --------------------------------------------------------------------------------------
def existing_shard_epochs(out_dir: Path) -> list[float]:
    """Epoch-seconds of every shard already on disk (live polls or prior backfill), sorted.
    Used to skip grid points already covered, so we fill only genuine holes and never stack
    a midpoint reconstruction on top of a denser live best-bid/ask poll."""
    out = []
    for p in out_dir.glob("poll_*.parquet"):
        stamp = p.stem.removeprefix("poll_")[:15]  # YYYYMMDDTHHMMSS
        try:
            out.append(datetime.strptime(stamp, "%Y%m%dT%H%M%S")
                       .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    out.sort()
    return out


def _covered(existing: list[float], t: float, tol_s: float) -> bool:
    """True if some existing shard lies within ±tol_s of grid time t."""
    if not existing:
        return False
    i = bisect_right(existing, t)
    for j in (i - 1, i):
        if 0 <= j < len(existing) and abs(existing[j] - t) <= tol_s:
            return True
    return False


def _asof(series: list[tuple[float, float]], t: float) -> float | None:
    """Last value with timestamp <= t (forward-fill). None if nothing yet — never peeks
    ahead, so a grid point only ever sees quotes that existed at or before it."""
    if not series:
        return None
    i = bisect_right(series, (t, float("inf")))
    return series[i - 1][1] if i > 0 else None


def synth_snapshot(meta: dict, ladder_hist: dict, bucket_hist: dict,
                   noipo_hist: list, hl_candles: list, t: float) -> dict:
    """Reconstruct one poll at grid time `t` in the exact shape `build_snapshot` emits.
    Midpoint history gives a single price per token, so bid == ask == mid."""
    snap = {"fetched_at_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(),
            "ladder": [], "buckets": [], "no_ipo": None, "hl": None}
    for d in meta["ladder"]:
        p = _asof(ladder_hist.get(d["token"], []), t)
        snap["ladder"].append({"strike_t": d["strike_t"], "bid": p, "ask": p})
    for d in meta["buckets"]:
        p = _asof(bucket_hist.get(d["token"], []), t)
        snap["buckets"].append({"label": d["label"], "lo": d["lo"], "hi": d["hi"],
                                "bid": p, "ask": p})
    if meta.get("no_ipo"):
        p = _asof(noipo_hist, t)
        snap["no_ipo"] = {"bid": p, "ask": p}
    mark = _asof(hl_candles, t)
    if mark is not None:
        snap["hl"] = {"mark": mark, "mid": mark, "funding_hourly": 0.0}
    return snap


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def backfill(start_s: int, end_s: int, step_s: int = 600, fidelity_min: int = 10,
             interval: str = "5m", dry_run: bool = False, fill_all: bool = False,
             out_dir: Path = PARQUET_LOG_DIR) -> dict:
    """Walk a regular grid from start→end, reconstruct + analyze + log each point.
    Returns a summary dict. Grid points with <4 usable ladder strikes are skipped; unless
    `fill_all`, points already covered by an existing shard (within ±step/2) are skipped too,
    so only genuine holes get filled."""
    meta = fetch_pm_metadata()
    print(f"[meta] {len(meta['ladder'])} ladder strikes, {len(meta['buckets'])} buckets",
          file=sys.stderr)

    ladder_hist = {d["token"]: fetch_token_history(d["token"], start_s, end_s, fidelity_min)
                   for d in meta["ladder"]}
    bucket_hist = {d["token"]: fetch_token_history(d["token"], start_s, end_s, fidelity_min)
                   for d in meta["buckets"]}
    noipo_hist = (fetch_token_history(meta["no_ipo"]["token"], start_s, end_s, fidelity_min)
                  if meta.get("no_ipo") else [])
    hl_candles = fetch_hl_candles(start_s, end_s, interval)
    print(f"[hist] perp {len(hl_candles)} candles; "
          f"ladder pts {[len(v) for v in ladder_hist.values()]}", file=sys.stderr)

    existing = [] if fill_all else existing_shard_epochs(out_dir)
    tol_s = step_s / 2.0

    written, skipped, covered, paths = 0, 0, 0, []
    t = float(start_s)
    while t <= end_s:
        if _covered(existing, t, tol_s):
            covered += 1            # a real/prior shard already sits here — leave it alone
            t += step_s
            continue
        snap = synth_snapshot(meta, ladder_hist, bucket_hist, noipo_hist, hl_candles, t)
        try:
            rep = analyze(snap, basis="mid")
        except RuntimeError:
            skipped += 1            # <4 strikes priced this early in history
            t += step_s
            continue
        if not dry_run:
            paths.append(log_parquet(snap, rep, out_dir=out_dir))
        written += 1
        t += step_s

    return {"written": written, "skipped": skipped, "covered": covered,
            "first": paths[0].name if paths else None,
            "last": paths[-1].name if paths else None,
            "span_utc": (datetime.fromtimestamp(start_s, timezone.utc).strftime("%m-%d %H:%M"),
                         datetime.fromtimestamp(end_s, timezone.utc).strftime("%m-%d %H:%M"))}


def _parse_iso(s: str) -> int:
    """Accept 'YYYY-MM-DDTHH:MM' (assumed UTC) or a full ISO string → epoch seconds."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=48.0,
                    help="fill the last N hours up to now (ignored if --start given)")
    ap.add_argument("--start", help="ISO start (UTC if no tz), e.g. 2026-06-10T17:00")
    ap.add_argument("--end", help="ISO end (UTC if no tz); default now")
    ap.add_argument("--step-min", type=float, default=10.0,
                    help="grid spacing in minutes (default 10)")
    ap.add_argument("--fidelity", type=int, default=10,
                    help="CLOB prices-history fidelity in minutes (default 10)")
    ap.add_argument("--interval", default="5m", help="HL candle interval (default 5m)")
    ap.add_argument("--dry-run", action="store_true",
                    help="reconstruct + analyze but write no shards (report only)")
    ap.add_argument("--fill-all", action="store_true",
                    help="also write grid points already covered by an existing shard "
                         "(default skips them so only overnight holes get filled)")
    args = ap.parse_args(argv)

    now = int(time.time())
    end_s = _parse_iso(args.end) if args.end else now
    start_s = _parse_iso(args.start) if args.start else end_s - int(args.hours * 3600)
    if start_s >= end_s:
        ap.error("start must be before end")

    summ = backfill(start_s, end_s, step_s=int(args.step_min * 60),
                    fidelity_min=args.fidelity, interval=args.interval,
                    dry_run=args.dry_run, fill_all=args.fill_all)
    tag = "(dry-run, nothing written)" if args.dry_run else "written"
    print(f"\n[backfill] {summ['written']} shards {tag}; {summ['covered']} already covered "
          f"(left as-is); {summ['skipped']} skipped (too few strikes priced)")
    print(f"           window {summ['span_utc'][0]} → {summ['span_utc'][1]} UTC, "
          f"grid {args.step_min:g} min")
    if summ["first"]:
        print(f"           {summ['first']} … {summ['last']}")
    if not args.dry_run:
        print("\nRelaunch the dashboard to load it:\n"
              "  uv run python scripts/spcx_pm_pdf_monitor.py --serve 8642 "
              "--backfill-days 2 --parquet-log --spot-ws alpaca \\\n"
              "      --html data/analysis/spcx_convergence/pm_pdf_dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
