#!/usr/bin/env python3
"""Dali class-B passive/maker re-tests through the institutional MM fill engine.

Re-runs the dali-era passive framings — the ones [[pm_dali_workflow_revision_decision]]
Tier 1 marks REOPEN-ELIGIBLE because they were killed under queue-blind fill proxies —
on the real VPS L2, through ``mm_engine``'s latency-gated :class:`FillSimulator` under
the FULL queue-model bracket (Optimistic / Prob(0.5) / RiskAverse). Nothing here writes
a new fill proxy: entries fill only when a real trade prints through the coded queue
models, net of a realistic 200ms round-trip.

Framings (each an engine Strategy, one 1-contract order at a time per market):

* ``mid_continuation``  (A14c/A14h): at extreme TOB imbalance, post at the rounded mid
  on the signal side (long posts a bid, short posts an ask); withdraw if unfilled after
  W; after a fill, hold and exit analytically (below).
* ``touch_fade``        (A18 passive reversion-to-microprice): at extreme TOB imbalance,
  JOIN the heavy-side touch queue (signal>0 -> post at best_bid; signal<0 -> at
  best_ask) and earn the drift toward microprice. The queue bracket bites here.
* ``touch_fade_ofi``    (P2's passive fade, OFI flavor): same posting rule keyed on the
  rolling 5s L1 order-flow imbalance instead of the TOB state.

Signal thresholds are the per-asset q90 of |signal| from the gated feature shards
(``mm_real_l2_features.py``) — the same in-capture descriptive conditioning the dali
blocks used (not an optimized parameter).

Exits are the dali grids' ``forced_taker`` convention, book-measured: a long exits at
the **best_bid**, a short at the **best_ask**, read from the gated L1 states at
``t_fill + H`` (H in {5, 30, 60}s) through :class:`lib.book_costs.BookCostIndex`
(<=5s staleness; uncovered exits are dropped AND counted). Episodes are blocked
in-engine until ``t_fill + max(H)`` so every H shares one non-overlapping episode set.
Fees: the captured schedule — fee_rate_bps == 0 on every politics/esports trade in the
window (verified), so ``FeeModel.fee_free_model()`` is the documentary truth, and there
is no rebate to flatter the result.

Outputs per (date, universe): ``{out}/{date}/{universe}/retest_episodes.parquet`` (one
row per episode x H x queue model x framing) and a printed per-run summary. Aggregation
across slices + market-cluster CIs happen in the findings pass.

STANDING CAVEAT carried by every number this produces: no queue model is calibrated
against our own fills yet (``calibrate(live_fills)`` is the Join-2 stub), so a positive
here is a *reopen-warranting candidate*, not an edge; a negative across the whole
bracket is an upgraded (robust) closure.

Run (from ``polymarket/research``):
    PYTHONPATH=. uv run python scripts/mm_dali_retest_engine.py \
        --dates 2026-06-20 2026-06-24 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_TRADES = 300          # dali's own per-market bar for the retest universe
MIN_THRESHOLD_ROWS = 500  # fresh L1 rows required to trust a per-asset q90
W_MS = 5_000              # fill window (dali's middle W)
H_GRID_S = (5, 30, 60)
BLOCK_MS = max(H_GRID_S) * 1000
TICK = 0.001
ROUND_TRIP_MS = 200.0     # realistic default; Join-2 measured ~160ms one-config


class DaliPassiveRetest:
    """Signal-triggered one-shot passive entries with W-withdrawal + H-blocking."""

    def __init__(self, mode: str, thresholds: dict[str, float], *,
                 w_ms: int = W_MS, block_ms: int = BLOCK_MS, size: float = 1.0) -> None:
        assert mode in ("mid_continuation", "touch_fade", "touch_fade_ofi")
        self.mode = mode
        self.thresholds = thresholds
        self.w_ms = int(w_ms)
        self.block_ms = int(block_ms)
        self.size = float(size)
        self.state: dict[str, dict] = {}
        self.signals_posted = 0
        # per-token L1 history for the OFI flavor
        self._prev_l1: dict[str, tuple] = {}
        self._ofi_win: dict[str, deque] = {}

    # --- signals -----------------------------------------------------------------
    @staticmethod
    def _tob(book) -> float | None:
        if not book.bids or not book.asks:
            return None
        bsz, asz = book.bids[0][1], book.asks[0][1]
        denom = bsz + asz
        return (bsz - asz) / denom if denom > 0 else None

    def _ofi_5s(self, book) -> float | None:
        """Cont L1 OFI accumulated over a trailing 5s of book states (per token)."""
        if not book.bids or not book.asks:
            return None
        tok = book.token_id
        cur = (book.bids[0][0], book.bids[0][1], book.asks[0][0], book.asks[0][1])
        prev = self._prev_l1.get(tok)
        self._prev_l1[tok] = cur
        if prev is None:
            return None
        b, qb, a, qa = cur
        pb, pqb, pa, pqa = prev
        e = ((qb if b >= pb else 0.0) - (pqb if b <= pb else 0.0)) \
            - ((qa if a <= pa else 0.0) - (pqa if a >= pa else 0.0))
        win = self._ofi_win.setdefault(tok, deque())
        win.append((book.ts_exchange, e))
        cutoff = book.ts_exchange - 5_000
        while win and win[0][0] < cutoff:
            win.popleft()
        return sum(x for _, x in win)

    # --- entry pricing -----------------------------------------------------------
    def _entry(self, book, side: str) -> float | None:
        bb, ba = book.bids[0][0], book.asks[0][0]
        if self.mode == "mid_continuation":
            px = round(round((bb + ba) / 2.0 / TICK) * TICK, 6)
            # clamp inside the spread (never cross); 1-tick books degrade to joining touch
            if side == "BUY":
                px = min(px, round(ba - TICK, 6))
                return max(px, bb) if px > 0 else None
            px = max(px, round(bb + TICK, 6))
            return min(px, ba) if px < 1 else None
        # touch_fade*: JOIN the heavy-side touch queue
        return bb if side == "BUY" else ba

    # --- Strategy protocol ---------------------------------------------------------
    def quote(self, book, inventory: float, params: dict) -> list:
        from mm_engine.interfaces import Order

        tok = book.token_id
        ts = book.ts_exchange
        st = self.state.setdefault(tok, {"prev_inv": 0.0, "posted_ts": None, "blocked_until": 0})

        if self.mode == "touch_fade_ofi":
            sig = self._ofi_5s(book)       # keep the rolling window warm on every event
        else:
            sig = self._tob(book)

        if inventory != st["prev_inv"]:
            # a fill routed on this event: episode opens, block re-entry for max(H)
            st["prev_inv"] = inventory
            if st["posted_ts"] is not None:
                st["posted_ts"] = None
                st["blocked_until"] = ts + self.block_ms
            return []

        if st["posted_ts"] is not None:
            if book.stale or ts - st["posted_ts"] > self.w_ms:
                st["posted_ts"] = None          # withdraw unfilled entry
                return []
            return [st["order"]]

        if book.stale or ts < st["blocked_until"] or not book.bids or not book.asks:
            return []
        thr = self.thresholds.get(tok)
        if thr is None or sig is None or abs(sig) < thr:
            return []
        side = "BUY" if sig > 0 else "SELL"
        px = self._entry(book, side)
        if px is None or not (0.0 < px < 1.0):
            return []
        order = Order(token_id=tok, side=side, price=px, size=self.size, tag=self.mode)
        st["order"] = order
        st["posted_ts"] = ts
        st["signal_value"] = float(sig)
        self.signals_posted += 1
        return [order]


def _hour_chunks(raw_dir: Path, keep: set[str]):
    """Yield hour-chunk event lists filtered to ``keep`` assets (DuckDB-side filter).

    Same canonical builders + ordering as the engine's Parquet adapter; chunked so a
    50M-event day never materializes (the memory-thrash lesson from the first run).
    """
    import duckdb

    from mm_engine.events import bba_event, book_event, price_change_event, trade_event
    from mm_engine.feeds._merge import order_and_interleave
    from mm_engine.feeds.replay_parquet import SCHEMA, _available_cols, _maybe_json_levels
    from mm_eval.capture_gate import _shard_hours

    con = duckdb.connect(config={"threads": 2})
    assets = sorted(keep)
    try:
        for hour in _shard_hours(raw_dir):
            events = []

            def rows(table: str) -> list[tuple]:
                files = sorted(str(p) for p in raw_dir.glob(f"{table}_*_{hour}.parquet"))
                if not files:
                    return []
                have = _available_cols(con, files)
                proj = ", ".join(c if c in have else f"NULL AS {c}" for c in SCHEMA[table])
                return con.execute(
                    f"SELECT {proj} FROM read_parquet($f) WHERE asset_id IN (SELECT unnest($a))",
                    {"f": files, "a": assets},
                ).fetchall()

            for ts, rcv_at, rcv_ns, aid, mkt, bids, asks in rows("book"):
                events.append(book_event(asset_id=aid, market=mkt,
                                         bids=_maybe_json_levels(bids), asks=_maybe_json_levels(asks),
                                         ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns))
            for ts, rcv_at, rcv_ns, aid, mkt, price, size, side in rows("trades"):
                events.append(trade_event(asset_id=aid, market=mkt, price=price, side=side, size=size,
                                          ts_exchange=ts, ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns))
            for ts, rcv_at, rcv_ns, aid, mkt, price, side, size in rows("price_change"):
                events.append(price_change_event(asset_id=aid, market=mkt, price=price, side=side,
                                                 size=size, ts_exchange=ts, ts_local_iso=rcv_at,
                                                 ts_monotonic_ns=rcv_ns))
            for ts, rcv_at, rcv_ns, aid, mkt, bbid, bask, bsz, asz in rows("bba"):
                events.append(bba_event(asset_id=aid, market=mkt, best_bid=bbid, best_ask=bask,
                                        bid_size=bsz, ask_size=asz, ts_exchange=ts,
                                        ts_local_iso=rcv_at, ts_monotonic_ns=rcv_ns))
            if events:
                yield list(order_and_interleave(events, []))
    finally:
        con.close()


def process_slice(raw_root: str, real_l2_root: str, date: str, universe: str) -> dict:
    import duckdb
    import numpy as np
    import pandas as pd

    from lib.book_costs import BookCostIndex
    from mm_engine.engine import run_engine
    from mm_engine.fees import FeeModel
    from mm_engine.latency_models import ConstantLatency
    from mm_engine.queue_models import OptimisticQueue, ProbQueue, RiskAverseQueue
    from mm_eval.capture_gate import CaptureQualityError

    raw_dir = Path(raw_root) / date / universe
    sdir = Path(real_l2_root) / date / universe
    gate = json.loads((sdir / "gate.json").read_text())
    if gate["verdict"] not in ("PASS", "MARGINAL"):
        raise CaptureQualityError(f"{date}/{universe}: gate verdict {gate['verdict']}")

    t0 = time.time()
    con = duckdb.connect()
    # one token per market (mirror books), markets with >= MIN_TRADES trades
    sel = con.execute(
        "SELECT market, asset_id, count(*) AS n FROM read_parquet($g) GROUP BY 1, 2",
        {"g": [str(raw_dir / "trades_*.parquet")]},
    ).df()
    sel = sel.sort_values("n", ascending=False).drop_duplicates("market")
    sel = sel[sel["n"] >= MIN_TRADES]
    keep = set(sel["asset_id"].astype(str))
    # per-asset |signal| q90 thresholds from the gated features
    feats = con.execute(
        "SELECT asset_id, tob_imbalance, ofi_5s FROM read_parquet($g) "
        "WHERE asset_id IN (SELECT unnest($a))",
        {"g": [str(sdir / "features.parquet")], "a": sorted(keep)},
    ).df()
    con.close()
    thr_tob: dict[str, float] = {}
    thr_ofi: dict[str, float] = {}
    for aid, g in feats.groupby("asset_id"):
        tob = g["tob_imbalance"].replace([np.inf, -np.inf], np.nan).dropna()
        ofi = g["ofi_5s"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tob) >= MIN_THRESHOLD_ROWS:
            thr_tob[str(aid)] = float(tob.abs().quantile(0.90))
        if len(ofi) >= MIN_THRESHOLD_ROWS:
            q = float(ofi.abs().quantile(0.90))
            if q > 0:
                thr_ofi[str(aid)] = q
    keep &= set(thr_tob) | set(thr_ofi)
    if not keep:
        return {"slice": f"{date}/{universe}", "status": "NO-ELIGIBLE-MARKETS"}

    asset_market = dict(zip(sel["asset_id"].astype(str), sel["market"].astype(str)))
    cost_idx = BookCostIndex.load(sdir, asset_ids=sorted(keep))

    from mm_engine.book import BookTracker
    from mm_engine.orders import OrderManager
    from mm_engine.telemetry import Telemetry

    models = {
        "optimistic": OptimisticQueue,
        "prob": lambda: ProbQueue(0.5),
        "risk_averse": RiskAverseQueue,
    }
    framings = {
        "mid_continuation": thr_tob,
        "touch_fade": thr_tob,
        "touch_fade_ofi": thr_ofi,
    }
    # persistent per-config state; each hour chunk streams through every config, so a
    # full day of events is never resident (memory-thrash lesson from the first run)
    configs = {}
    for framing, thresholds in framings.items():
        if not thresholds:
            continue
        for model_name, model_factory in models.items():
            configs[(framing, model_name)] = {
                "strategy": DaliPassiveRetest(framing, thresholds),
                "queue_model": model_factory(),
                "tracker": BookTracker(),
                "om": OrderManager(),
                "tele": Telemetry.in_memory(),
                "latency": ConstantLatency(ROUND_TRIP_MS),
            }
    fee_model = FeeModel.fee_free_model()
    for chunk in _hour_chunks(raw_dir, keep):
        for cfg in configs.values():
            run_engine(
                chunk,
                strategy=cfg["strategy"],
                queue_model=cfg["queue_model"],
                latency_model=cfg["latency"],
                fee_model=fee_model,
                tracker=cfg["tracker"],
                order_manager=cfg["om"],
                telemetry=cfg["tele"],
            )

    rows: list[dict] = []
    run_stats: list[dict] = []
    for (framing, model_name), cfg in configs.items():
        fills = cfg["tele"].fills.records
        strat = cfg["strategy"]
        # episodes: first fill per posting; the strategy blocks re-entry for BLOCK_MS,
        # so grouping fills by (token, client_id) is unambiguous.
        episodes: dict[tuple, dict] = {}
        for f in fills:
            key = (f["token_id"], f["client_id"])
            ep = episodes.setdefault(key, {
                "t_fill": f["ts_exchange"], "side": f["side"], "price": f["price"],
                "qty": 0.0, "token_id": f["token_id"],
            })
            ep["qty"] += f["qty"]
        dropped_exits = 0
        for ep in episodes.values():
            direction = 1.0 if ep["side"] == "BUY" else -1.0
            for h in H_GRID_S:
                q = cost_idx.quote(ep["token_id"], ep["t_fill"] + h * 1000)
                if q is None or q.best_bid is None or q.best_ask is None:
                    dropped_exits += 1
                    continue
                exit_px = q.best_bid if direction > 0 else q.best_ask
                pnl_c = direction * (exit_px - ep["price"]) * 100.0
                rows.append({
                    "date": date, "universe": universe, "framing": framing,
                    "queue_model": model_name, "h_s": h,
                    "token_id": ep["token_id"],
                    "market": asset_market.get(ep["token_id"], ""),
                    "t_fill_ms": ep["t_fill"], "side": ep["side"],
                    "entry_px": ep["price"], "exit_px": exit_px,
                    "qty": ep["qty"], "pnl_cents_per_contract": pnl_c,
                })
        run_stats.append({
            "framing": framing, "model": model_name,
            "signals_posted": strat.signals_posted, "episodes": len(episodes),
            "fills": len(fills), "dropped_exits": dropped_exits,
        })

    out = pd.DataFrame(rows)
    out_path = sdir / "retest_episodes.parquet"
    out.to_parquet(out_path, index=False)
    return {
        "slice": f"{date}/{universe}", "status": "OK", "markets": len(keep),
        "episode_rows": len(out), "runs": run_stats, "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "l2_parquet_full")
    ap.add_argument("--real-l2-dir", type=Path, default=ROOT / "data" / "analysis" / "real_l2")
    ap.add_argument("--dates", nargs="*", required=True)
    ap.add_argument("--universes", nargs="*", default=["politics_negrisk", "esports"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks = [(d, u) for d in args.dates for u in args.universes
             if (args.real_l2_dir / d / u / "features.parquet").exists()]
    print(f"{len(tasks)} slices", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_slice, str(args.raw_dir), str(args.real_l2_dir), d, u): (d, u)
                for d, u in tasks}
        for fut in as_completed(futs):
            d, u = futs[fut]
            try:
                print(json.dumps(fut.result()), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"slice": f"{d}/{u}", "status": "ERROR", "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
