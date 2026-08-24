#!/usr/bin/env python3
"""Dali-descended feature recomputation on the real VPS L2 (gated, book-measured).

Consumes the ``l1_states_*.parquet`` shards written by ``mm_real_l2_gate_scan.py``
(reconstructed top-of-book behind the required capture-quality gate) plus the raw
``trades_*`` shards, and writes one feature Parquet per ``{date}/{universe}`` slice:

    {out}/{date}/{universe}/features.parquet

One row per fresh L1 change per token, with the dali-descended features computed by
the SAME definitions the dali era used, but on measured book state:

* ``tob_imbalance``      = (bid_size - ask_size) / (bid_size + ask_size)
* ``microprice``         = (best_bid*ask_size + best_ask*bid_size) / (bid_size+ask_size)
* ``micro_dev``          = microprice - mid   (reversion-to-microprice distance)
* ``ofi_event``          = Cont-style L1 order-flow imbalance for the state change
* ``ofi_5s`` / ``ofi_60s``  rolling event-time sums of ``ofi_event``
* ``spread``, ``mid``, ``touch_depth``  (book-measured costs, Tier-2 rule 3)
* ``fwd_mid_5s/30s/60s`` forward mid at +h seconds (as-of last fresh state <= t+h;
  NaN when the next fresh state is beyond the horizon — lookahead-free)
* ``trade_rate_60s``     trailing 60s trade count for the token (activity control)

Gate enforcement is fail-closed: a slice whose ``gate.json`` verdict is not
PASS/MARGINAL is skipped with an error line (Tier-2 rule 2 — nothing gets a feature
panel until it passes).

Run (from ``polymarket/research``):
    PYTHONPATH=. uv run python scripts/mm_real_l2_features.py \
        --real-l2-dir data/analysis/real_l2 --raw-dir data/l2_parquet_full --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HORIZONS_S = (5, 30, 60)


def _features_for_asset(g, trades_ts) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    g = g.sort_values("timestamp_ms").reset_index(drop=True)
    bid = g["best_bid"].to_numpy(float)
    ask = g["best_ask"].to_numpy(float)
    bsz = g["bid_size"].to_numpy(float)
    asz = g["ask_size"].to_numpy(float)
    ts = g["timestamp_ms"].to_numpy(np.int64)

    denom = bsz + asz
    with np.errstate(invalid="ignore", divide="ignore"):
        tob = np.where(denom > 0, (bsz - asz) / denom, np.nan)
        micro = np.where(denom > 0, (bid * asz + ask * bsz) / denom, np.nan)
    mid = (bid + ask) / 2.0
    spread = ask - bid

    # Cont et al. L1 OFI over consecutive fresh states
    pb, pa = np.roll(bid, 1), np.roll(ask, 1)
    pqb, pqa = np.roll(bsz, 1), np.roll(asz, 1)
    e_bid = np.where(bid >= pb, bsz, 0.0) - np.where(bid <= pb, pqb, 0.0)
    e_ask = np.where(ask <= pa, asz, 0.0) - np.where(ask >= pa, pqa, 0.0)
    ofi = e_bid - e_ask
    ofi[0] = np.nan   # no predecessor state

    out = pd.DataFrame({
        "timestamp_ms": ts,
        "asset_id": g["asset_id"].to_numpy(),
        "market": g["market"].to_numpy(),
        "best_bid": bid, "best_ask": ask, "bid_size": bsz, "ask_size": asz,
        "mid": mid, "spread": spread, "touch_depth": denom,
        "tob_imbalance": tob, "microprice": micro, "micro_dev": micro - mid,
        "ofi_event": ofi,
    })

    # rolling event-time sums (windows in ms, inclusive)
    for win_s, col in ((5, "ofi_5s"), (60, "ofi_60s")):
        w = win_s * 1000
        left = np.searchsorted(ts, ts - w, side="left")
        cs = np.nancumsum(np.nan_to_num(ofi))
        cs = np.concatenate([[0.0], cs])
        out[col] = cs[np.arange(len(ts)) + 1] - cs[left]

    # forward mids: as-of the last state <= t + h (fresh states only — g is pre-filtered)
    for h in HORIZONS_S:
        target = ts + h * 1000
        idx = np.searchsorted(ts, target, side="right") - 1
        fwd = np.full(len(ts), np.nan)
        valid = (idx > np.arange(len(ts))) & (target <= ts[-1] if len(ts) else False)
        fwd[valid] = mid[idx[valid]]
        out[f"fwd_mid_{h}s"] = fwd

    # trailing 60s trade count for the token
    if trades_ts is not None and len(trades_ts):
        t_sorted = np.sort(trades_ts)
        hi = np.searchsorted(t_sorted, ts, side="right")
        lo = np.searchsorted(t_sorted, ts - 60_000, side="left")
        out["trade_rate_60s"] = (hi - lo).astype(float)
    else:
        out["trade_rate_60s"] = 0.0
    return out


def process_slice(real_l2_dir: str, raw_dir: str, date: str, universe: str) -> dict:
    import duckdb
    import pandas as pd

    sdir = Path(real_l2_dir) / date / universe
    gate = json.loads((sdir / "gate.json").read_text())
    if gate["verdict"] not in ("PASS", "MARGINAL"):
        return {"slice": f"{date}/{universe}", "status": f"GATE-{gate['verdict']}"}

    t0 = time.time()
    con = duckdb.connect()
    states = con.execute(
        "SELECT timestamp_ms, asset_id, market, best_bid, bid_size, best_ask, ask_size "
        "FROM read_parquet($g) WHERE NOT stale AND best_bid IS NOT NULL AND best_ask IS NOT NULL "
        "AND bid_size IS NOT NULL AND ask_size IS NOT NULL ORDER BY asset_id, timestamp_ms",
        {"g": [str(sdir / "l1_states_*.parquet")]},
    ).df()
    trades = con.execute(
        "SELECT timestamp_ms, asset_id FROM read_parquet($g)",
        {"g": [str(Path(raw_dir) / date / universe / "trades_*.parquet")]},
    ).df()
    con.close()

    trades_by_asset = {aid: g["timestamp_ms"].to_numpy("int64") for aid, g in trades.groupby("asset_id")}
    parts = []
    for aid, g in states.groupby("asset_id", sort=False):
        parts.append(_features_for_asset(g, trades_by_asset.get(aid)))
    feats = pd.concat(parts, ignore_index=True)
    out_path = sdir / "features.parquet"
    feats.to_parquet(out_path, index=False)
    return {
        "slice": f"{date}/{universe}", "status": "OK", "rows": len(feats),
        "assets": feats["asset_id"].nunique(), "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-l2-dir", type=Path, default=ROOT / "data" / "analysis" / "real_l2")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "l2_parquet_full")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tasks = []
    for date_dir in sorted(p for p in args.real_l2_dir.iterdir() if p.is_dir()):
        for uni_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if not (uni_dir / "DONE").exists():
                continue
            if (uni_dir / "features.parquet").exists() and not args.force:
                continue
            tasks.append((date_dir.name, uni_dir.name))
    print(f"{len(tasks)} slices", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_slice, str(args.real_l2_dir), str(args.raw_dir), d, u): (d, u)
                for d, u in tasks}
        for fut in as_completed(futs):
            d, u = futs[fut]
            try:
                print(json.dumps(fut.result()), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"slice": f"{d}/{u}", "status": "ERROR", "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
