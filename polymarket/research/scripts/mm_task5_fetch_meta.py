"""Fetch + cache Gamma market metadata for the Task-4/Task-5 token set.

Pulls, for every condition id in the Task-4 verdict CSV (the 24 evaluated tokens' markets):
``endDate`` (the τ anchor Task 5 injects via params), ``negRiskMarketID`` + event id/slug
(the IS/OOS split unit — complementary legs share a resolution fingerprint, so the split
must group by event, never by token), ``closed``/``umaResolutionStatus`` and
``outcomePrices`` + ``clobTokenIds`` (so in-window resolvers can be settled at their actual
payoff in the costed eval).

Cache is written to ``data/markets/mm_task5_market_meta.json`` (durable, committed — the
prior scratchpad cache was session-local and is gone). Reruns are offline once the cache
exists; ``--force`` refreshes.

Usage (from polymarket/research/):
    PYTHONPATH=. uv run python scripts/mm_task5_fetch_meta.py [--force]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
VERDICT_CSV = RESEARCH / "data/analysis/csv_outputs/market_making/mm_validation_verdict.csv"
OUT = RESEARCH / "data/markets/mm_task5_market_meta.json"
GAMMA = "https://gamma-api.polymarket.com/markets"

KEEP_MARKET = ("question", "conditionId", "slug", "endDate", "startDate", "closed", "active",
               "negRisk", "negRiskMarketID", "umaResolutionStatuses", "outcomes",
               "outcomePrices", "clobTokenIds", "volumeNum", "groupItemTitle")
KEEP_EVENT = ("id", "slug", "title", "negRisk", "negRiskMarketID", "endDate")


def _get(params: dict) -> list:
    url = f"{GAMMA}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "epsilon-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_condition(cid: str) -> dict | None:
    # Gamma omits resolved markets from the bare condition_ids query — retry with closed=true.
    rows = _get({"condition_ids": cid}) or _get({"condition_ids": cid, "closed": "true"})
    if not rows:
        return None
    m = rows[0]
    rec = {k: m.get(k) for k in KEEP_MARKET}
    evs = m.get("events") or []
    rec["events"] = [{k: e.get(k) for k in KEEP_EVENT} for e in evs]
    # The split unit: prefer the explicit NegRisk market id, else the (first) event id,
    # else fall back to the condition id itself (a standalone market is its own group).
    ev_id = evs[0]["id"] if evs and evs[0].get("id") else None
    rec["group_id"] = m.get("negRiskMarketID") or ev_id or cid
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        meta = json.loads(OUT.read_text())
        print(f"cache exists: {OUT} ({len(meta)} markets); --force to refresh")
        return

    vd = pd.read_csv(VERDICT_CSV, dtype={"token_id": str})
    cids = sorted(vd["market"].unique())
    print(f"{len(cids)} unique condition ids across {len(vd)} tokens")
    meta: dict[str, dict] = {}
    for i, cid in enumerate(cids, 1):
        rec = fetch_condition(cid)
        if rec is None:
            print(f"  [{i}/{len(cids)}] {cid[:12]}… NOT FOUND")
            continue
        meta[cid] = rec
        print(f"  [{i}/{len(cids)}] {cid[:12]}… end={rec['endDate']} group={str(rec['group_id'])[:14]} "
              f"closed={rec['closed']} q=\"{str(rec['question'])[:48]}\"")
        time.sleep(0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT} ({len(meta)} markets)")

    # group summary — the IS/OOS split units
    groups: dict[str, list[str]] = {}
    for cid, rec in meta.items():
        groups.setdefault(str(rec["group_id"]), []).append(cid)
    print(f"\n{len(groups)} event groups:")
    for g, members in sorted(groups.items()):
        qs = [str(meta[c]["question"])[:40] for c in members]
        print(f"  group {g[:16]}…: {len(members)} markets — {qs}")


if __name__ == "__main__":
    main()
