"""Task-5.1 setup — universe selection, Gamma metadata, lead-in cohorts, CPCV folds.

Runs the data-side prerequisites of the Task-5.1 redesign, and prints the code-verified
coverage statement the PRD requires (date range + market count per category) BEFORE any
strategy runs:

1. **Coverage verification** over the full R2 clone (`~/epsilon_l2_full`).
2. **Token selection** — top-K most-traded quotable tokens per universe (Task-4 rule:
   ≥150 prints, avg price 5–95¢), on the FULL sample.
3. **Gamma metadata** for every selected condition id (endDate = τ anchor; NegRisk
   event id = the whole-market split unit; resolution payoffs) — cached, `closed=true`
   retry for resolved markets.
4. **Leakage-safe lead-in cohort features** per event group (`mm_eval.cpcv.lead_in_features`
   — first min(24h, 25% of span), never the evaluation window), cohort assignment
   (aggressiveness × liquidity), and cohort-balanced CPCV fold assignment.

Outputs (committed-path caches; regenerable):
    data/markets/mm_task5_1_market_meta.json      Gamma metadata per condition id
    data/markets/mm_task5_1_selection.json        selected tokens + spans per universe
    data/analysis/csv_outputs/market_making/mm_task5_1_groups.parquet   features+cohorts+folds

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_1_setup.py [--top-k 20] [--n-folds 6] [--force-meta]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mm_eval import cpcv
from mm_eval import markets as mk

RESEARCH = Path(__file__).resolve().parents[1]
L2_ROOT = Path.home() / "epsilon_l2_full"
CSV_OUT = RESEARCH / "data/analysis/csv_outputs/market_making"
META_JSON = RESEARCH / "data/markets/mm_task5_1_market_meta.json"
SELECTION_JSON = RESEARCH / "data/markets/mm_task5_1_selection.json"
TASK5_META = RESEARCH / "data/markets/mm_task5_market_meta.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-justiniturregui-Desktop-github-epsilon-quant-research/"
               "b6ea1a3f-cca4-465b-b11c-5ab18e4a749c/scratchpad")
CACHE = SCRATCH / "mm_task5_1_cache"
UNIVERSES = ("politics_negrisk", "esports")


def verify_coverage(con) -> dict:
    """The PRD's code-verified coverage statement (print BEFORE running anything)."""
    out = {}
    print("=== R2 sample coverage (code-verified) ===")
    for u in UNIVERSES:
        files = [str(p) for d in sorted(L2_ROOT.glob(f"*/{u}")) for p in d.glob("trades_*.parquet")]
        lo, hi, n_tok, n_mkt, n_tr = con.execute(
            "SELECT min(timestamp_ms), max(timestamp_ms), count(DISTINCT asset_id), "
            "count(DISTINCT market), count(*) FROM read_parquet(?)", [files]).fetchone()
        f = lambda t: datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        span_d = (hi - lo) / 86400e3
        print(f"  {u}: {f(lo)} → {f(hi)} UTC  span={span_d:.1f}d  "
              f"tokens={n_tok}  markets={n_mkt}  trades={n_tr}")
        out[u] = {"ts_min": int(lo), "ts_max": int(hi), "span_days": span_d,
                  "n_tokens": int(n_tok), "n_markets": int(n_mkt), "n_trades": int(n_tr)}
    return out


# ── Gamma metadata (mirrors mm_task5_fetch_meta with the same gotcha handling) ──
GAMMA = "https://gamma-api.polymarket.com/markets"
KEEP_MARKET = ("question", "conditionId", "slug", "endDate", "startDate", "closed", "active",
               "negRisk", "negRiskMarketID", "umaResolutionStatuses", "outcomes",
               "outcomePrices", "clobTokenIds", "volumeNum", "groupItemTitle")
KEEP_EVENT = ("id", "slug", "title", "negRisk", "negRiskMarketID", "endDate")


def _get(params: dict) -> list:
    import urllib.parse
    import urllib.request
    url = f"{GAMMA}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "epsilon-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_condition(cid: str) -> dict | None:
    # Gamma omits resolved markets from the bare condition_ids query — retry closed=true.
    rows = _get({"condition_ids": cid}) or _get({"condition_ids": cid, "closed": "true"})
    if not rows:
        return None
    m = rows[0]
    rec = {k: m.get(k) for k in KEEP_MARKET}
    evs = m.get("events") or []
    rec["events"] = [{k: e.get(k) for k in KEEP_EVENT} for e in evs]
    ev_id = evs[0]["id"] if evs and evs[0].get("id") else None
    rec["group_id"] = m.get("negRiskMarketID") or ev_id or cid
    return rec


def load_or_fetch_meta(cids: list[str], force: bool) -> dict:
    meta: dict = {}
    if META_JSON.exists() and not force:
        meta = json.loads(META_JSON.read_text())
    if TASK5_META.exists():   # reuse the Task-5 cache for overlapping markets
        for cid, rec in json.loads(TASK5_META.read_text()).items():
            meta.setdefault(cid, rec)
    missing = [c for c in cids if c not in meta]
    print(f"Gamma metadata: {len(cids)} condition ids, {len(missing)} to fetch")
    for i, cid in enumerate(missing, 1):
        rec = fetch_condition(cid)
        if rec is None:
            print(f"  [{i}/{len(missing)}] {cid[:12]}… NOT FOUND")
            continue
        meta[cid] = rec
        print(f"  [{i}/{len(missing)}] {cid[:12]}… end={rec['endDate']} "
              f"group={str(rec['group_id'])[:14]} closed={rec['closed']} "
              f"q=\"{str(rec['question'])[:44]}\"")
        time.sleep(0.25)
    META_JSON.parent.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-folds", type=int, default=6)
    ap.add_argument("--force-meta", action="store_true")
    args = ap.parse_args()
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    coverage = verify_coverage(con)

    selection: dict = {"coverage": coverage, "top_k": args.top_k, "universes": {}}
    all_cids: list[str] = []
    specs_by_u: dict[str, list[mk.MarketSpec]] = {}
    for u in UNIVERSES:
        specs = mk.select_markets(L2_ROOT, u, top_k=args.top_k, con=con)
        specs_by_u[u] = specs
        all_cids.extend(s.market for s in specs)
        print(f"\n[{u}] selected {len(specs)} tokens "
              f"(trades {specs[-1].n_trades}..{specs[0].n_trades})")

    meta = load_or_fetch_meta(sorted(set(all_cids)), args.force_meta)

    for u in UNIVERSES:
        # materialize per-token replay dirs (cached)
        mk.build_compact(L2_ROOT, u, specs_by_u[u], CACHE, con=con)
        for s in specs_by_u[u]:
            mk.materialize_token(s, CACHE, con=con)
        # group map for this universe
        toks = []
        for s in specs_by_u[u]:
            rec = meta.get(s.market, {})
            toks.append({
                "token_id": s.token_id, "market": s.market,
                "group_id": str(rec.get("group_id", s.market)),
                "end_date": rec.get("endDate"),
                "half_spread": s.half_spread, "n_trades": s.n_trades,
                "avg_price": s.avg_price, "question": rec.get("question"),
            })
        selection["universes"][u] = {
            "span": [coverage[u]["ts_min"], coverage[u]["ts_max"]],
            "tokens": toks,
        }
        n_groups = len({t["group_id"] for t in toks})
        print(f"[{u}] {len(toks)} tokens → {n_groups} event groups")

    # lead-in cohort features + folds per universe
    frames = []
    for u in UNIVERSES:
        toks = selection["universes"][u]["tokens"]
        by_group: dict[str, list[Path]] = {}
        for t in toks:
            by_group.setdefault(t["group_id"], []).append(CACHE / u / t["token_id"])
        feats = pd.DataFrame([cpcv.lead_in_features(g, dirs, con)
                              for g, dirs in by_group.items()])
        feats["universe"] = u
        feats = cpcv.assign_cohorts(feats)
        n_folds = min(args.n_folds, len(feats))
        feats = cpcv.assign_folds(feats, n_folds=n_folds)
        feats["n_folds"] = n_folds
        frames.append(feats)
        print(f"\n[{u}] cohorts: "
              f"{feats.groupby(['cohort_aggr', 'cohort_liq']).size().to_dict()}")
        print(feats[["group_id", "n_tokens", "flow_rate", "sweep_share", "spread_c",
                     "vol_c", "avg_price", "cohort_aggr", "cohort_liq", "fold"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    groups_df = pd.concat(frames, ignore_index=True)
    groups_df.to_parquet(CSV_OUT / "mm_task5_1_groups.parquet", index=False)

    SELECTION_JSON.write_text(json.dumps(selection, indent=2))
    print(f"\nwrote {SELECTION_JSON}\nwrote {CSV_OUT / 'mm_task5_1_groups.parquet'}")
    con.close()


if __name__ == "__main__":
    main()
