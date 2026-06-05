"""WS-passive execution model: redo PnL assuming bot posts limit orders
at exactly p_in (entry) and p_out (exit), filled only when an aggressive
counter-arrives.

Reports:
  - canonical (p_in=0.60, p_out=0.90) under optimistic + strict exit assumptions
  - comparison to the proxy-based scenarios from Session 2.6
  - 21-pair grid scan; top 5 cells by edge_per_filled and total_pnl_per_entered

Run:
    python scripts/passive_analysis.py [--no-grid]
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


P_IN, P_OUT = 0.60, 0.90


def main() -> int:
    inst = pd.read_parquet(wa.DEFAULT_DATA_DIR / "weather_tail_per_instance.parquet")
    p_wide = wa.pivot_inst_to_wide(inst)
    print(f"p_wide rows: {len(p_wide):,}")

    print(f"\n=== Canonical (p_in={P_IN}, p_out={P_OUT}, policy='all') — WS-passive PnL ===")
    t0 = time.time()
    out, audit = wa.eval_pair(p_wide, P_IN, P_OUT, return_audit=True)
    scen = wa.compute_fill_scenarios(audit)
    print(f"  eval_pair: {time.time() - t0:.1f}s, n_entries={out['n_entries']}")

    aug_opt, sum_opt = wa.passive_pnl_from_audit(scen, exit_passive="optimistic")
    aug_str, sum_str = wa.passive_pnl_from_audit(scen, exit_passive="strict")

    print("\n  --- WS-passive headline ---")
    rows = []
    rows.append({
        "model": "proxy_next_same_dir (taker)",
        "n_trades": len(scen),
        "edge_per_trade":   float(scen["pnl_next_same_dir"].mean()),
        "edge_per_entered": float(scen["pnl_next_same_dir"].mean()),
        "roi_pct":          100 * float(scen["pnl_next_same_dir"].mean()) / P_IN,
    })
    rows.append({
        "model": "proxy_next_opp_dir (maker_best, fallback-dominated)",
        "n_trades": len(scen),
        "edge_per_trade":   float(scen["pnl_next_opp_dir"].mean()),
        "edge_per_entered": float(scen["pnl_next_opp_dir"].mean()),
        "roi_pct":          100 * float(scen["pnl_next_opp_dir"].mean()) / P_IN,
    })
    for label, s in [("WS-passive (optimistic exit)", sum_opt),
                     ("WS-passive (strict exit)",     sum_str)]:
        rows.append({
            "model": label,
            "n_trades": s["n_entry_filled"],
            "edge_per_trade":   s["edge_per_filled"],
            "edge_per_entered": s["edge_per_entered"],
            "roi_pct":          s["roi_per_filled_pct"],
        })
    print(pd.DataFrame(rows).round({
        "edge_per_trade": 4, "edge_per_entered": 4, "roi_pct": 2,
    }).to_string(index=False))

    print("\n  --- WS-passive details ---")
    for label, s in [("optimistic", sum_opt), ("strict", sum_str)]:
        print(f"  {label} exit assumption:")
        print(f"    n_total={s['n_total']}   n_filled={s['n_entry_filled']}   "
              f"fill_rate={s['fill_rate']:.3f}")
        print(f"    outcomes among filled: tp={s['p_tp_of_filled']:.3f}  "
              f"hold_win={s['p_hold_win_of_filled']:.3f}  "
              f"hold_chop={s['p_hold_chop_of_filled']:.3f}")
        print(f"    edge_per_filled = ${s['edge_per_filled']:+.4f}   "
              f"ROI = {s['roi_per_filled_pct']:+.2f}%")
        print(f"    edge_per_entered = ${s['edge_per_entered']:+.4f}   "
              f"total_pnl across {s['n_total']} crosses = ${s['total_pnl_per_entered']:+.2f}")

    if "--no-grid" not in sys.argv:
        print(f"\n\n=== 21-pair grid under WS-passive (optimistic exit) — ~7 min ===")
        t0 = time.time()
        grid = wa.grid_passive(p_wide, exit_passive="optimistic")
        print(f"  grid_passive: {time.time() - t0:.1f}s   {len(grid)} cells")

        print("\n--- top 5 cells by edge_per_filled (ROI on actual trades) ---")
        top_r = grid.sort_values("edge_per_filled", ascending=False).head(5)
        print(top_r[["p_in", "p_out", "n_total", "n_entry_filled", "fill_rate",
                     "p_tp_of_filled", "edge_per_filled", "roi_per_filled_pct",
                     "total_pnl_per_entered"]].round({
                         "fill_rate": 3, "p_tp_of_filled": 3,
                         "edge_per_filled": 4, "roi_per_filled_pct": 2,
                         "total_pnl_per_entered": 1,
                     }).to_string(index=False))

        print("\n--- top 5 cells by total_pnl_per_entered (PnL across all crosses) ---")
        top_t = grid.sort_values("total_pnl_per_entered", ascending=False).head(5)
        print(top_t[["p_in", "p_out", "n_total", "n_entry_filled", "fill_rate",
                     "edge_per_filled", "roi_per_filled_pct",
                     "total_pnl_per_entered"]].round({
                         "fill_rate": 3,
                         "edge_per_filled": 4, "roi_per_filled_pct": 2,
                         "total_pnl_per_entered": 1,
                     }).to_string(index=False))

        # Save grid for downstream use
        out_path = wa.DEFAULT_DATA_DIR / "passive_grid_canonical.parquet"
        grid.to_parquet(out_path, index=False)
        print(f"\n  saved grid → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
