"""MM NegRisk basket-consistency live poller.

Read-only scanner over the full active Polymarket NegRisk universe:
  1. Pulls every active negRisk event from Gamma (paginated) -> universe snapshot.
  2. Every cycle, batch-fetches the YES-side CLOB book for every leg via
     POST https://clob.polymarket.com/books.
  3. Per event per cycle, records sum-of-best-asks (cost to BUY every leg) and
     sum-of-best-bids (proceeds to SELL every leg via mint), depth at best,
     and per-leg detail whenever the event is within a wide diagnostic band of
     a violation (so violation episodes have full leg-level depth context).

Output (append-only JSONL, one dir per run):
  data/live_clob/mm_negrisk_consistency_scan/<run_id>/
    universe_<ts>.json     -- full event/leg metadata snapshot (refreshed ~45min)
    cycles.jsonl           -- one row per (cycle, event)
    cycle_meta.jsonl       -- one row per cycle (timing, coverage, errors)
    manifest.json          -- run metadata

No orders, no auth, no writes outside the run dir. Designed to be killed at
any time; every line is self-contained.

Run from polymarket/research:
  PYTHONPATH=. uv run python scripts/mm_negrisk_consistency_poll.py --hours 6
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events?closed=false&active=true&limit=100&offset={offset}"
CLOB_BOOKS = "https://clob.polymarket.com/books"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; epsilon-quant read-only scanner)", "Content-Type": "application/json"}
BATCH = 100
# Record per-leg books whenever |ask_sum - 1| or |bid_sum - 1| is inside this
# band, not only on strict violation, so near-miss dynamics are inspectable.
DETAIL_BAND = 0.03


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def http_get_json(url: str, timeout: int = 25, retries: int = 3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def http_post_json(url: str, payload, timeout: int = 25, retries: int = 3):
    body = json.dumps(payload).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=HEADERS, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_universe(max_offset: int = 20000):
    """All active negRisk events with per-leg YES token ids and fee/lockup metadata."""
    events = []
    offset = 0
    while offset <= max_offset:
        page = http_get_json(GAMMA_EVENTS.format(offset=offset))
        if not page:
            break
        for e in page:
            if not e.get("negRisk"):
                continue
            legs = []
            for m in e.get("markets", []):
                tids = m.get("clobTokenIds")
                if isinstance(tids, str):
                    try:
                        tids = json.loads(tids)
                    except Exception:
                        tids = None
                if not tids:
                    continue
                legs.append(
                    {
                        "market_id": str(m.get("id")),
                        "slug": m.get("slug"),
                        "yes_token": tids[0],
                        "active": m.get("active"),
                        "closed": m.get("closed"),
                        "fees_enabled": m.get("feesEnabled"),
                        "taker_fee_bps": m.get("takerBaseFee"),
                        "maker_fee_bps": m.get("makerBaseFee"),
                        "end_date": m.get("endDate"),
                        "group_item_title": m.get("groupItemTitle"),
                    }
                )
            if legs:
                events.append(
                    {
                        "event_id": str(e.get("id")),
                        "event_slug": e.get("slug"),
                        "title": e.get("title"),
                        "neg_risk_augmented": e.get("negRiskAugmented"),
                        "enable_neg_risk": e.get("enableNegRisk"),
                        "liquidity": e.get("liquidity"),
                        "end_date": e.get("endDate"),
                        "n_legs": len(legs),
                        "legs": legs,
                    }
                )
        offset += 100
        time.sleep(0.1)
    return events


def best_level(levels):
    """CLOB book sides are sorted worst->best; best level is the last entry."""
    if not levels:
        return None, None
    lv = levels[-1]
    return float(lv["price"]), float(lv["size"])


def run(hours: float, cycle_seconds: float, out_root: Path):
    run_id = "mm_negrisk_consistency_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe()
    uni_ts = utcnow()
    (out_dir / f"universe_{uni_ts.replace(':', '').replace('.', '')}.json").write_text(json.dumps(universe))
    token_to_event = {}
    for ev in universe:
        for leg in ev["legs"]:
            token_to_event[leg["yes_token"]] = (ev["event_id"], leg["market_id"])

    manifest = {
        "run_id": run_id,
        "started_at": utcnow(),
        "purpose": "read-only NegRisk basket consistency scan: sum-of-best-asks / sum-of-best-bids per event over time, with depth",
        "cycle_seconds": cycle_seconds,
        "planned_hours": hours,
        "n_events": len(universe),
        "n_legs": sum(e["n_legs"] for e in universe),
        "detail_band": DETAIL_BAND,
        "no_orders": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{utcnow()}] run {run_id}: {manifest['n_events']} events / {manifest['n_legs']} legs", flush=True)

    cycles_f = open(out_dir / "cycles.jsonl", "a")
    meta_f = open(out_dir / "cycle_meta.jsonl", "a")

    deadline = time.time() + hours * 3600
    last_universe_refresh = time.time()
    cycle_idx = 0

    while time.time() < deadline:
        cycle_start = time.time()
        cycle_idx += 1

        if time.time() - last_universe_refresh > 45 * 60:
            try:
                universe = fetch_universe()
                uni_ts = utcnow()
                (out_dir / f"universe_{uni_ts.replace(':', '').replace('.', '')}.json").write_text(json.dumps(universe))
                last_universe_refresh = time.time()
                print(f"[{utcnow()}] universe refreshed: {len(universe)} events", flush=True)
            except Exception as exc:
                print(f"[{utcnow()}] universe refresh failed: {exc}", flush=True)

        all_tokens = [leg["yes_token"] for ev in universe for leg in ev["legs"]]
        books = {}
        n_req_err = 0
        for i in range(0, len(all_tokens), BATCH):
            chunk = all_tokens[i : i + BATCH]
            try:
                resp = http_post_json(CLOB_BOOKS, [{"token_id": t} for t in chunk])
                for b in resp or []:
                    books[b["asset_id"]] = b
            except Exception:
                n_req_err += 1
            time.sleep(0.12)

        recv_ts = utcnow()
        n_rows = 0
        for ev in universe:
            leg_rows = []
            ask_sum = 0.0
            bid_sum = 0.0
            n_ask_missing = 0
            n_book_missing = 0
            min_ask_depth_usd = None
            min_bid_depth_usd = None
            for leg in ev["legs"]:
                b = books.get(leg["yes_token"])
                if b is None:
                    n_book_missing += 1
                    n_ask_missing += 1
                    leg_rows.append({"m": leg["market_id"], "bb": None, "bs": None, "ba": None, "as": None})
                    continue
                bb, bbs = best_level(b.get("bids"))
                ba, bas = best_level(b.get("asks"))
                if ba is None:
                    n_ask_missing += 1
                else:
                    ask_sum += ba
                    ask_depth = ba * bas
                    if min_ask_depth_usd is None or ask_depth < min_ask_depth_usd:
                        min_ask_depth_usd = ask_depth
                if bb is not None:
                    bid_sum += bb
                    bid_depth = bb * bbs
                    if min_bid_depth_usd is None or bid_depth < min_bid_depth_usd:
                        min_bid_depth_usd = bid_depth
                leg_rows.append({"m": leg["market_id"], "bb": bb, "bs": bbs, "ba": ba, "as": bas})

            complete_asks = n_ask_missing == 0
            row = {
                "ts": recv_ts,
                "cycle": cycle_idx,
                "event_id": ev["event_id"],
                "n_legs": ev["n_legs"],
                "n_book_missing": n_book_missing,
                "n_ask_missing": n_ask_missing,
                "ask_sum": round(ask_sum, 6) if complete_asks else None,
                "bid_sum": round(bid_sum, 6),
                "min_ask_depth_usd": round(min_ask_depth_usd, 2) if (complete_asks and min_ask_depth_usd is not None) else None,
                "min_bid_depth_usd": round(min_bid_depth_usd, 2) if min_bid_depth_usd is not None else None,
            }
            in_band = (complete_asks and abs(ask_sum - 1.0) <= DETAIL_BAND) or (abs(bid_sum - 1.0) <= DETAIL_BAND)
            if in_band:
                row["legs"] = leg_rows
            cycles_f.write(json.dumps(row) + "\n")
            n_rows += 1

        cycles_f.flush()
        elapsed = time.time() - cycle_start
        meta = {
            "ts": recv_ts,
            "cycle": cycle_idx,
            "n_tokens_requested": len(all_tokens),
            "n_books_returned": len(books),
            "n_batch_errors": n_req_err,
            "n_event_rows": n_rows,
            "sweep_seconds": round(elapsed, 1),
        }
        meta_f.write(json.dumps(meta) + "\n")
        meta_f.flush()
        print(f"[{recv_ts}] cycle {cycle_idx}: {len(books)}/{len(all_tokens)} books, sweep {elapsed:.0f}s, batch_err {n_req_err}", flush=True)

        sleep_left = cycle_seconds - (time.time() - cycle_start)
        if sleep_left > 0:
            time.sleep(sleep_left)

    cycles_f.close()
    meta_f.close()
    print(f"[{utcnow()}] done: {cycle_idx} cycles", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--cycle-seconds", type=float, default=120.0)
    ap.add_argument("--out-root", default="data/live_clob/mm_negrisk_consistency_scan")
    args = ap.parse_args()
    run(args.hours, args.cycle_seconds, Path(args.out_root))
