#!/usr/bin/env python3
"""Gate + L1-state scan over the real VPS L2 Parquet archive (one replay, two outputs).

For every ``{date}/{universe}`` slice under ``--data-dir`` this runs the REQUIRED
capture-quality gate (``mm_eval.capture_gate`` — BBA checksum lead-lag-aware, ≤5s
staleness, trade-in-spread coherence, heartbeat gap inference) and, in the SAME
replay pass, records the reconstructed top-of-book every time L1 changes:

    {out}/{date}/{universe}/gate.json                 gate verdict + aggregate stats
    {out}/{date}/{universe}/gate_per_token.csv        per-token classification counts
    {out}/{date}/{universe}/l1_states_{k}.parquet     L1-change series (append-only shards)

The L1-state shards are the substrate for (a) the book-measured cost helper
(``lib.book_costs`` — spread/depth READ from the book at a timestamp, never estimated)
and (b) the dali-descended feature recomputation (TOB imbalance, OFI, microprice).

Columns of ``l1_states``: ``timestamp_ms`` (exchange ms — lookahead-free key),
``asset_id``, ``market``, ``best_bid``, ``best_ask``, ``bid_size``, ``ask_size``,
``stale`` (BookTracker's ≤5s/gap/anchor flag at that event). One row per change of
that 5-tuple per token.

Run (from ``polymarket/research``):
    PYTHONPATH=. uv run python scripts/mm_real_l2_gate_scan.py \
        --data-dir data/l2_parquet_full --out-dir data/analysis/real_l2 --workers 5

Workers are sized to RAM, not cores (CODEX anti-pattern rule): each worker holds one
hour-chunk of events (~0.5-3M objects, ~1-3 GB peak on the busiest politics hours).
Parquet shards are append-only; re-running a completed slice is skipped unless
``--force``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLUSH_ROWS = 5_000_000


class L1StateRecorder:
    """Append a row whenever a token's (bid_p, bid_s, ask_p, ask_s, stale) changes."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.last: dict[str, tuple] = {}
        self.rows: list[tuple] = []
        self.shard = 0
        self.total = 0

    def __call__(self, ev, state) -> None:
        if ev.type not in ("book", "price_change"):
            return
        bid = state.bids[0] if state.bids else (None, None)
        ask = state.asks[0] if state.asks else (None, None)
        key = (bid[0], bid[1], ask[0], ask[1], state.stale)
        if self.last.get(ev.token_id) == key:
            return
        self.last[ev.token_id] = key
        self.rows.append((
            ev.ts_exchange, ev.token_id, str(ev.payload.get("market") or ""),
            bid[0], bid[1], ask[0], ask[1], bool(state.stale),
        ))
        if len(self.rows) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pandas as pd

        df = pd.DataFrame(self.rows, columns=[
            "timestamp_ms", "asset_id", "market",
            "best_bid", "bid_size", "best_ask", "ask_size", "stale",
        ])
        path = self.out_dir / f"l1_states_{self.shard:03d}.parquet"
        df.to_parquet(path, index=False)   # new file every flush — append-only invariant
        self.total += len(df)
        self.rows.clear()
        self.shard += 1


def process_slice(data_dir: str, out_dir: str, date: str, universe: str) -> dict:
    from mm_eval.capture_gate import CaptureQualityGate

    src = Path(data_dir) / date / universe
    out = Path(out_dir) / date / universe
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = L1StateRecorder(out)
    gate = CaptureQualityGate()
    result = gate.run_slice(src, slice_id=f"{date}/{universe}", on_state=rec)
    rec.flush()
    result.to_json(out / "gate.json")

    import pandas as pd

    per_tok = pd.DataFrame([
        {"asset_id": tok, **asdict(s)} for tok, s in result.per_token.items()
    ])
    per_tok.to_csv(out / "gate_per_token.csv", index=False)
    (out / "DONE").write_text("")
    dt = time.time() - t0
    return {
        "slice": f"{date}/{universe}", "verdict": result.verdict,
        "fresh_clean_pct": round(result.overall.fresh_clean_pct, 2),
        "raw_clean_pct": round(result.overall.raw_clean_pct, 2),
        "stale_pct": round(result.overall.stale_pct, 2),
        "trade_in_spread_pct": round(result.overall.trade_in_spread_pct, 2),
        "checks": result.overall.checks, "l1_rows": rec.total,
        "gaps_inferred": result.gaps_inferred, "secs": round(dt, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "l2_parquet_full")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "analysis" / "real_l2")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--dates", nargs="*", default=None, help="subset of dates (default: all)")
    ap.add_argument("--universes", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tasks: list[tuple[str, str]] = []
    for date_dir in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        if args.dates and date_dir.name not in args.dates:
            continue
        for uni_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if args.universes and uni_dir.name not in args.universes:
                continue
            done = args.out_dir / date_dir.name / uni_dir.name / "DONE"
            if done.exists() and not args.force:
                continue
            tasks.append((date_dir.name, uni_dir.name))

    print(f"{len(tasks)} slices to process with {args.workers} workers", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_slice, str(args.data_dir), str(args.out_dir), d, u): (d, u)
            for d, u in tasks
        }
        for fut in as_completed(futs):
            d, u = futs[fut]
            try:
                s = fut.result()
            except Exception as exc:  # noqa: BLE001 — surface, don't die
                s = {"slice": f"{d}/{u}", "verdict": "ERROR", "error": str(exc)}
            summaries.append(s)
            print(json.dumps(s), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scan_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"wrote {args.out_dir / 'scan_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
