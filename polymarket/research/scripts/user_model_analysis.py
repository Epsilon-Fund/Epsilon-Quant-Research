"""User-calibrated execution model: optimal (p_in, p_out) + PnL at 5% sizing.

Two entry models compared:
  STICKY (default, REALISTIC) — bot posts at p_in cap, doesn't track down.
                                In crashes our high bid is hit FIRST → fill at p_in.
                                Captures adverse selection.
  TRACK_DOWN (UPPER BOUND)     — bot continuously chases best bid in both
                                directions. Fill at the touch. Requires queue
                                priority and sub-second cancel-replace that
                                Polymarket weather markets don't realistically
                                support. Sensitivity check only.

Common: exit at p_out via active sell (hit bid, receive p_out − spread).
Sizing: 5% of INITIAL bankroll per filled trade (non-compounding).

Caches each grid to:
    data/analysis/user_passive_grid_sticky.parquet      (REALISTIC)
    data/analysis/user_passive_grid_track_down.parquet  (UPPER BOUND)

Run:
    python scripts/user_model_analysis.py [--no-grid]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_PKG = Path(__file__).resolve().parents[1]
if str(REPO_PKG) not in sys.path:
    sys.path.insert(0, str(REPO_PKG))

from data_infra import weather_analysis as wa  # noqa: E402


BANKROLL = 10_000.0
SIZE_PCT = 0.05
SPREAD_CENTS = 2.0


def _run_or_load_grid(p_wide, entry_model: str, label: str):
    cache = wa.DEFAULT_DATA_DIR / f"user_passive_grid_{entry_model}.parquet"
    if cache.exists():
        print(f"  [{label}] loaded cache: {cache}")
        return pd.read_parquet(cache)
    print(f"  [{label}] running grid (~7 min)…")
    t0 = time.time()
    grid = wa.grid_user_passive(
        p_wide, bankroll=BANKROLL, size_pct=SIZE_PCT,
        entry_model=entry_model,
        assumed_spread_cents=SPREAD_CENTS,
    )
    grid.to_parquet(cache, index=False)
    print(f"  [{label}] ran in {time.time() - t0:.1f}s, saved {cache}")
    return grid


def _print_top(grid: pd.DataFrame, label: str) -> None:
    valid = grid[grid["n_filled"] > 0].copy()
    print(f"\n  [{label}] full grid sorted by total $PnL desc:")
    sorted_df = valid.sort_values("total_dollar_pnl", ascending=False)
    cols = ["p_in", "p_out", "n_filled", "fill_rate", "p_tp_of_filled",
            "edge_per_filled", "total_dollar_pnl", "pnl_pct_of_initial_bankroll",
            "sharpe_ann", "max_dd_pct"]
    print(sorted_df[cols].round({
        "fill_rate": 3, "p_tp_of_filled": 3, "edge_per_filled": 4,
        "total_dollar_pnl": 1, "pnl_pct_of_initial_bankroll": 2,
        "sharpe_ann": 2, "max_dd_pct": 1,
    }).to_string(index=False))


def main() -> int:
    inst = pd.read_parquet(wa.DEFAULT_DATA_DIR / "weather_tail_per_instance.parquet")
    p_wide = wa.pivot_inst_to_wide(inst)
    print(f"p_wide rows: {len(p_wide):,}    bankroll=${BANKROLL:,.0f}   "
          f"size_pct={SIZE_PCT:.1%}   spread={SPREAD_CENTS}¢")

    print(f"\n=== canonical (0.60, 0.90) — STICKY (realistic) ===")
    t0 = time.time()
    canon_sticky = wa.eval_pair_user_passive(
        p_wide, p_in=0.60, p_out=0.90,
        bankroll=BANKROLL, size_pct=SIZE_PCT,
        entry_model="sticky",
        assumed_spread_cents=SPREAD_CENTS,
    )
    print(f"  ran in {time.time() - t0:.1f}s")
    for k, v in canon_sticky.items():
        print(f"  {k:32s} {v}")

    if "--no-grid" not in sys.argv:
        print(f"\n=== Sticky grid (REALISTIC) ===")
        grid_sticky = _run_or_load_grid(p_wide, "sticky", "sticky")
        _print_top(grid_sticky, "sticky / REALISTIC")

        print(f"\n=== Track-down grid (UPPER BOUND, unrealistic) ===")
        grid_td = _run_or_load_grid(p_wide, "track_down", "track_down")
        _print_top(grid_td, "track_down / UPPER BOUND")

        # Side-by-side gap = execution sensitivity
        print(f"\n=== Sticky vs Track-down gap per cell ===")
        merged = grid_sticky.merge(
            grid_td, on=["p_in", "p_out"], suffixes=("_sticky", "_td"))
        merged["gap_pnl_$"] = merged["total_dollar_pnl_td"] - merged["total_dollar_pnl_sticky"]
        merged["gap_roi_pp"] = (merged["pnl_pct_of_initial_bankroll_td"]
                                  - merged["pnl_pct_of_initial_bankroll_sticky"])
        print(merged[["p_in", "p_out",
                      "total_dollar_pnl_sticky", "total_dollar_pnl_td",
                      "gap_pnl_$", "gap_roi_pp"]].sort_values("gap_pnl_$",
                      ascending=False).round({
                          "total_dollar_pnl_sticky": 0, "total_dollar_pnl_td": 0,
                          "gap_pnl_$": 0, "gap_roi_pp": 1,
                      }).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
