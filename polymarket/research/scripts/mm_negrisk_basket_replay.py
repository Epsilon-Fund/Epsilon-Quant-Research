"""Millisecond-resolution NegRisk basket-consistency replay over captured books.

Input: filtered capture JSONLs (NegRisk legs only) produced by grepping the
mm_stage1 capture lanes for the leg token ids
(data/analysis/negrisk_replay/filtered/*_negrisk.jsonl).

Method:
  1. Replay each filtered JSONL with the canonical state builder
     (scripts/dali_clob_replay_features.replay) -> per-asset state rows with
     best bid/ask, sizes, completeness, staleness.
  2. Merge rows across capture lanes (both lanes record the same public WS
     feed; duplicate states are harmless for last-state-as-of sampling).
  3. Per event basket, sample every leg's last state on the union of state
     timestamps (K4 discipline: leg counts as live only if its book state is
     complete and book_staleness_seconds <= 5).
  4. Compute conservative basket sums per instant:
       bid_sum_live  = sum of live legs' best bids (missing legs add 0)
                       -> bid_sum_live > 1 is a CONFIRMED sell-all violation
       ask_sum_live  = sum of live legs' best asks
                       -> only meaningful for buy-all when n_live == n_legs_total;
                          otherwise reported as a partial diagnostic
  5. Collapse violations into exchange-time intervals with duration, edge,
     and binding top-of-book depth.

Outputs CSVs under data/analysis/csv_outputs/market_making/ prefixed
mm_negrisk_basket_replay_*.

Run from polymarket/research:
  PYTHONPATH=. uv run python scripts/mm_negrisk_basket_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dali_clob_replay_features as rf  # noqa: E402

FILTERED_DIR = Path("data/analysis/negrisk_replay/filtered")
OUT_DIR = Path("data/analysis/csv_outputs/market_making")
STATE_CACHE = Path("data/analysis/negrisk_replay/state_rows.parquet")
MAX_STALE_S = 5.0
TICK_FLOOR = 0.001

# Basket definitions: event -> total legs and the captured leg slugs.
# buy_diagnostic: emit the tick-floor partial buy-side bound ONLY where the
# missing legs are known deep tails (Fed: the one missing leg per day trades
# 0.1-0.5c). For SPX/continents the missing legs are mid-priced, so the bound
# is meaningless and buy-side is reported as not-evaluable.
BASKETS = {
    "fed-decision-in-june-825": {
        "n_legs_total": 5,
        "buy_diagnostic": True,
        "legs": [
            "will-the-fed-decrease-interest-rates-by-25-bps-after-the-june-2026-meeting",
            "will-the-fed-decrease-interest-rates-by-50-bps-after-the-june-2026-meeting",
            "will-the-fed-increase-interest-rates-by-25-bps-after-the-june-2026-meeting",
            "will-the-fed-increase-interest-rates-by-50-bps-after-the-june-2026-meeting",
            "will-there-be-no-change-in-fed-interest-rates-after-the-june-2026-meeting",
        ],
    },
    "which-continent-will-win-the-world-cup": {
        "n_legs_total": 7,
        "legs": [
            "will-asia-win-the-2026-fifa-world-cup",
            "will-north-america-win-the-2026-fifa-world-cup",
            "will-europe-win-the-2026-fifa-world-cup",
            "will-africa-win-the-2026-fifa-world-cup",
        ],
    },
    "spx-close-dec-2026": {
        "n_legs_total": 6,
        "legs": [
            "spx-close-7000-7500-dec-2026-723",
            "spx-close-7500-8000-dec-2026",
            "spx-close-6000-6500-dec-2026",
        ],
    },
}


def token_metadata() -> dict[str, tuple[str, int]]:
    """asset_id -> (slug, outcome_index). The filtered JSONLs have no manifest
    sidecars, so the canonical replayer leaves slug/outcome_index empty; we
    restore them from the capture-config inventory (markets.json: slug -> tokens,
    token order = clob_token_ids order, index 0 = YES)."""
    inv = json.loads((FILTERED_DIR.parent / "markets.json").read_text())
    out = {}
    for slug, rec in inv.items():
        for i, tok in enumerate(rec["tokens"]):
            out[tok] = (slug, i)
    return out


def build_state_rows() -> pd.DataFrame:
    if STATE_CACHE.exists():
        return pd.read_parquet(STATE_CACHE)
    frames = []
    for p in sorted(FILTERED_DIR.glob("*_negrisk.jsonl")):
        if p.stat().st_size == 0:
            continue
        df = rf.replay(p, 3)
        if len(df):
            df["capture_run"] = p.stem.replace("_negrisk", "")
            frames.append(df)
        print(f"replayed {p.name}: {len(df):,} rows", flush=True)
    if not frames:
        raise SystemExit("no filtered capture rows found")
    out = pd.concat(frames, ignore_index=True)
    keep = [
        "received_at", "exchange_ts", "event_type", "asset_id", "slug", "outcome_index",
        "is_book_state_complete", "book_staleness_seconds",
        "best_bid", "best_bid_size", "best_ask", "best_ask_size", "capture_run",
    ]
    out = out[keep]
    STATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(STATE_CACHE, index=False)
    return out


def yes_leg_rows(df: pd.DataFrame, leg_slugs: list[str]) -> pd.DataFrame:
    """YES-side rows only (outcome_index 0) for the basket legs, deduped across lanes."""
    meta = token_metadata()
    df = df.copy()
    df["slug"] = df["asset_id"].map(lambda a: meta.get(a, ("", None))[0])
    df["outcome_index"] = df["asset_id"].map(lambda a: meta.get(a, ("", None))[1])
    sub = df[(df["slug"].isin(leg_slugs)) & (df["outcome_index"] == 0)].copy()
    sub = sub.dropna(subset=["received_at"])
    sub["ts"] = pd.to_datetime(sub["received_at"], utc=True)
    sub = sub.sort_values("ts")
    sub = sub.drop_duplicates(subset=["slug", "ts", "best_bid", "best_ask", "best_bid_size", "best_ask_size"])
    return sub


def basket_series(sub: pd.DataFrame, leg_slugs: list[str]) -> pd.DataFrame:
    """Sample each leg's last state on the union of timestamps; staleness-gated."""
    union_ts = sub["ts"].drop_duplicates().sort_values()
    grid = pd.DataFrame({"ts": union_ts.reset_index(drop=True)})
    legs_live = pd.Series(0, index=grid.index)
    n_ask_live = pd.Series(0, index=grid.index)
    bid_sum = pd.Series(0.0, index=grid.index)
    ask_sum = pd.Series(0.0, index=grid.index)
    ask_all_present = pd.Series(True, index=grid.index)
    min_bid_depth = pd.Series(np.inf, index=grid.index)
    min_ask_depth = pd.Series(np.inf, index=grid.index)

    for slug in leg_slugs:
        leg = sub[sub["slug"] == slug][["ts", "is_book_state_complete", "best_bid", "best_bid_size", "best_ask", "best_ask_size"]]
        if not len(leg):
            ask_all_present[:] = False
            continue
        leg = leg.sort_values("ts")
        m = pd.merge_asof(grid, leg, on="ts", direction="backward")
        last_seen = pd.merge_asof(grid, leg[["ts"]].assign(seen_ts=leg["ts"]), on="ts", direction="backward")["seen_ts"]
        stale_s = (grid["ts"] - last_seen).dt.total_seconds()
        live = m["is_book_state_complete"].fillna(False) & (stale_s <= MAX_STALE_S)

        has_bid = live & m["best_bid"].notna()
        bid_sum = bid_sum + m["best_bid"].where(has_bid, 0.0)
        depth_b = (m["best_bid"] * m["best_bid_size"]).where(has_bid, np.inf)
        min_bid_depth = np.minimum(min_bid_depth, depth_b)

        has_ask = live & m["best_ask"].notna()
        ask_sum = ask_sum + m["best_ask"].where(has_ask, 0.0)
        ask_all_present = ask_all_present & has_ask
        n_ask_live = n_ask_live + has_ask.astype(int)
        depth_a = (m["best_ask"] * m["best_ask_size"]).where(has_ask, np.inf)
        min_ask_depth = np.minimum(min_ask_depth, depth_a)

        legs_live = legs_live + live.astype(int)

    grid["n_legs_live"] = legs_live
    grid["n_ask_live"] = n_ask_live
    grid["bid_sum_live"] = bid_sum
    grid["ask_sum_partial"] = ask_sum
    grid["ask_sum_live"] = ask_sum.where(ask_all_present, np.nan)
    grid["min_bid_depth_usd"] = min_bid_depth.replace(np.inf, np.nan)
    grid["min_ask_depth_usd"] = min_ask_depth.replace(np.inf, np.nan)
    return grid


def collapse_intervals(grid: pd.DataFrame, flag: pd.Series, edge: pd.Series, depth: pd.Series) -> list[dict]:
    intervals = []
    in_run = False
    start_i = None
    idx = grid.index.to_list()
    for k, i in enumerate(idx + [None]):
        active = i is not None and bool(flag.iloc[k]) if i is not None else False
        if active and not in_run:
            in_run, start_i = True, k
        elif not active and in_run:
            seg = grid.iloc[start_i:k]
            e = edge.iloc[start_i:k]
            d = depth.iloc[start_i:k]
            intervals.append(
                {
                    "start_ts": seg["ts"].iloc[0],
                    "end_ts": seg["ts"].iloc[-1],
                    "duration_ms": (seg["ts"].iloc[-1] - seg["ts"].iloc[0]).total_seconds() * 1000,
                    "n_states": len(seg),
                    "edge_mean_c": float(e.mean() * 100),
                    "edge_max_c": float(e.max() * 100),
                    "binding_depth_usd_median": float(d.median()) if d.notna().any() else np.nan,
                    "n_legs_live_min": int(seg["n_legs_live"].min()),
                }
            )
            in_run = False
    return intervals


def main():
    df = build_state_rows()
    print(f"state rows: {len(df):,} | slugs: {df['slug'].nunique()} | runs: {df['capture_run'].nunique()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_intervals = []
    summaries = []
    for ev_slug, spec in BASKETS.items():
        sub = yes_leg_rows(df, spec["legs"])
        if not len(sub):
            print(f"{ev_slug}: no captured rows")
            continue
        grid = basket_series(sub, spec["legs"])
        n_cap = sub["slug"].nunique()
        n_total = spec["n_legs_total"]
        max_live = int(grid["n_legs_live"].max()) if len(grid) else 0
        full_cov = grid["n_legs_live"] == max_live  # modal best concurrent coverage

        # Sell side: confirmed violations even with missing legs.
        sell_flag = grid["bid_sum_live"] > 1.0
        sell_edge = grid["bid_sum_live"] - 1.0
        sell_iv = collapse_intervals(grid, sell_flag, sell_edge, grid["min_bid_depth_usd"])
        for r in sell_iv:
            r.update({"event": ev_slug, "direction": "sell_all_confirmed"})
        all_intervals += sell_iv

        # Buy side, confirmed: every leg of the FULL event has a live ask.
        buy_conf_flag = (grid["n_ask_live"] == n_total) & (grid["ask_sum_partial"] < 1.0)
        buy_iv = collapse_intervals(grid, buy_conf_flag, 1.0 - grid["ask_sum_partial"], grid["min_ask_depth_usd"])
        for r in buy_iv:
            r.update({"event": ev_slug, "direction": "buy_all_confirmed"})
        all_intervals += buy_iv

        # Buy side, tick-floor diagnostic: exactly one leg short of full, and only
        # for events whose missing leg is a known deep tail (Fed).
        if spec.get("buy_diagnostic"):
            one_missing = grid["n_ask_live"] == n_total - 1
            bound = grid["ask_sum_partial"] + TICK_FLOOR
            diag_flag = one_missing & (bound < 1.0)
            diag_iv = collapse_intervals(grid, diag_flag, 1.0 - bound, grid["min_ask_depth_usd"])
            for r in diag_iv:
                r.update({"event": ev_slug, "direction": "buy_all_partial_bound_diagnostic"})
            all_intervals += diag_iv
            buy_iv = buy_iv + diag_iv

        cov = grid[full_cov]
        summaries.append(
            {
                "event": ev_slug,
                "n_legs_total": n_total,
                "n_legs_captured": n_cap,
                "n_legs_live_max_concurrent": max_live,
                "n_states": len(grid),
                "first_ts": grid["ts"].min(),
                "last_ts": grid["ts"].max(),
                "max_concurrent_coverage_share": float(full_cov.mean()),
                "bid_sum_p50": float(cov["bid_sum_live"].median()) if len(cov) else np.nan,
                "bid_sum_p99": float(cov["bid_sum_live"].quantile(0.99)) if len(cov) else np.nan,
                "bid_sum_max": float(cov["bid_sum_live"].max()) if len(cov) else np.nan,
                "ask_sum_partial_p50": float(cov["ask_sum_partial"].median()) if len(cov) else np.nan,
                "ask_sum_partial_p01": float(cov["ask_sum_partial"].quantile(0.01)) if len(cov) else np.nan,
                "ask_sum_partial_min": float(cov["ask_sum_partial"].min()) if len(cov) else np.nan,
                "n_sell_violations": len(sell_iv),
                "n_buy_rows": len(buy_iv),
            }
        )
        print(f"{ev_slug}: {len(grid):,} states | coverage {float(full_cov.mean()):.1%} | sell-violation intervals {len(sell_iv)}")

    pd.DataFrame(summaries).to_csv(OUT_DIR / "mm_negrisk_basket_replay_summary.csv", index=False)
    iv_df = pd.DataFrame(all_intervals)
    if len(iv_df):
        iv_df = iv_df.sort_values("edge_max_c", ascending=False)
    iv_df.to_csv(OUT_DIR / "mm_negrisk_basket_replay_intervals.csv", index=False)
    print(f"\nintervals: {len(iv_df)}")
    if len(iv_df):
        print(iv_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
