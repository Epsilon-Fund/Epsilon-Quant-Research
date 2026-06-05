"""Proposal B — data-supported subset analysis.

For canonical (p_in=0.60, p_out=0.90, policy='all'):
  - baseline           : full universe
  - taker_filt         : entry_next_same_dir came from real next_fill
  - maker_filt         : entry_next_opp_dir came from real next_fill
  - intersection       : both legs real

For each subset: bucket dist + 3 scenario edges + entry slippage.
Selection bias: top-5 cities baseline vs each subset, hour-of-day, dow,
distance-to-resolution stats.
Then runs the 21-pair grid on the highest-edge mode and reports top 5 cells.

Run:
    python scripts/subset_analysis.py
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


def _print_summary_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    print(df[[
        "label", "n", "p_tp",
        "edge_next_same_dir", "edge_midpoint", "edge_next_opp_dir",
        "roi_next_same_dir_pct", "roi_midpoint_pct", "roi_next_opp_dir_pct",
        "mean_entry_slip_cents",
    ]].round({
        "p_tp": 3,
        "edge_next_same_dir": 4, "edge_midpoint": 4, "edge_next_opp_dir": 4,
        "roi_next_same_dir_pct": 2, "roi_midpoint_pct": 2, "roi_next_opp_dir_pct": 2,
        "mean_entry_slip_cents": 2,
    }).to_string(index=False))


def _print_selection_bias(bias: dict, label: str) -> None:
    print(f"\n  --- selection bias: {label} (n={bias['n_subset']} of {bias['n_baseline']}; "
          f"{bias['subset_share_of_baseline']:.1%}) ---")
    print(f"  top cities (baseline → subset share):")
    base_by = {c["city"]: c["share"] for c in bias["top_cities_baseline"]}
    sub_by  = {c["city"]: c["share"] for c in bias["top_cities_subset"]}
    cities = list(dict.fromkeys(list(base_by.keys()) + list(sub_by.keys())))
    for c in cities[:8]:
        print(f"    {c:42s}  base={base_by.get(c, 0):.3f}  subset={sub_by.get(c, 0):.3f}")
    bh = bias["hours_to_resolution_baseline"]; sh = bias["hours_to_resolution_subset"]
    print(f"  hours-to-resolution at entry (p10/p50/p90):")
    print(f"    baseline: {bh['p10']:.1f} / {bh['median']:.1f} / {bh['p90']:.1f}")
    print(f"    subset  : {sh['p10']:.1f} / {sh['median']:.1f} / {sh['p90']:.1f}")
    print(f"  hour-UTC top3 (baseline → subset):")
    for h in sorted(set(bias["hour_utc_baseline_top3"].keys()) |
                    set(bias["hour_utc_subset_top3"].keys())):
        print(f"    {h:02d}h  base={bias['hour_utc_baseline_top3'].get(h, 0):.3f}  "
              f"subset={bias['hour_utc_subset_top3'].get(h, 0):.3f}")
    print(f"  day-of-week top3 (baseline → subset):")
    for d in sorted(set(bias["dow_baseline_top3"].keys()) |
                    set(bias["dow_subset_top3"].keys())):
        print(f"    {d:10s} base={bias['dow_baseline_top3'].get(d, 0):.3f}  "
              f"subset={bias['dow_subset_top3'].get(d, 0):.3f}")


def main() -> int:
    inst = pd.read_parquet(wa.DEFAULT_DATA_DIR / "weather_tail_per_instance.parquet")
    p_wide = wa.pivot_inst_to_wide(inst)
    print(f"p_wide rows: {len(p_wide):,}")

    print(f"\n=== Running eval_pair at canonical (p_in={P_IN}, p_out={P_OUT}) ===")
    t0 = time.time()
    out, audit = wa.eval_pair(p_wide, P_IN, P_OUT, return_audit=True)
    scen = wa.compute_fill_scenarios(audit)
    print(f"  done in {time.time() - t0:.1f}s, n_entries={out['n_entries']}")

    # Masks
    baseline_mask = pd.Series(True, index=scen.index)
    taker_mask = scen["entry_next_same_dir_source"] == "next_fill"
    maker_mask = scen["entry_next_opp_dir_source"] == "next_fill"
    inter_mask = taker_mask & maker_mask

    print("\n=== 4-row summary table (canonical, p_in=0.60, p_out=0.90) ===")
    rows = [
        wa.subset_pnl_summary(scen, baseline_mask, "baseline",     p_in=P_IN),
        wa.subset_pnl_summary(scen, taker_mask,    "taker_filt",   p_in=P_IN),
        wa.subset_pnl_summary(scen, maker_mask,    "maker_filt",   p_in=P_IN),
        wa.subset_pnl_summary(scen, inter_mask,    "intersection", p_in=P_IN),
    ]
    _print_summary_table(rows)

    print("\n=== Selection bias ===")
    for mask, label in [(taker_mask, "taker_filt"),
                        (maker_mask, "maker_filt"),
                        (inter_mask, "intersection")]:
        bias = wa.subset_selection_bias(scen, p_wide, mask, baseline_mask=baseline_mask)
        _print_selection_bias(bias, label)

    # Pick the highest-edge_midpoint subset for the grid scan.
    best_idx = max(range(1, 4), key=lambda i: rows[i]["edge_midpoint"]
                                              if not pd.isna(rows[i]["edge_midpoint"])
                                              else -1)
    best_label = rows[best_idx]["label"]
    best_mask_builder = {
        "taker_filt":   lambda a: a["entry_next_same_dir_source"] == "next_fill",
        "maker_filt":   lambda a: a["entry_next_opp_dir_source"] == "next_fill",
        "intersection": lambda a: (a["entry_next_same_dir_source"] == "next_fill")
                                  & (a["entry_next_opp_dir_source"] == "next_fill"),
    }[best_label]

    print(f"\n\n=== Grid scan on subset='{best_label}' (21 pairs × ~20s each) ===")
    t0 = time.time()
    grid = wa.grid_subset(p_wide, best_mask_builder)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n--- top 5 cells by edge_midpoint on subset='{best_label}' ---")
    valid = grid[grid["n"] > 0].copy()
    if len(valid):
        top = valid.sort_values("edge_midpoint", ascending=False).head(5)
        print(top[["p_in", "p_out", "n", "subset_share",
                   "p_tp", "edge_next_same_dir", "edge_midpoint", "edge_next_opp_dir",
                   "roi_midpoint_pct"]].round({
                       "subset_share": 3, "p_tp": 3,
                       "edge_next_same_dir": 4, "edge_midpoint": 4, "edge_next_opp_dir": 4,
                       "roi_midpoint_pct": 2,
                   }).to_string(index=False))

        print(f"\n--- top 5 cells by edge_next_same_dir (taker) on subset='{best_label}' ---")
        top_t = valid.sort_values("edge_next_same_dir", ascending=False).head(5)
        print(top_t[["p_in", "p_out", "n", "edge_next_same_dir", "edge_midpoint",
                     "edge_next_opp_dir", "roi_next_same_dir_pct"]].round({
                         "edge_next_same_dir": 4, "edge_midpoint": 4, "edge_next_opp_dir": 4,
                         "roi_next_same_dir_pct": 2,
                     }).to_string(index=False))

        print(f"\n--- top 5 cells by edge_next_opp_dir (maker_best) on subset='{best_label}' ---")
        top_o = valid.sort_values("edge_next_opp_dir", ascending=False).head(5)
        print(top_o[["p_in", "p_out", "n", "edge_next_same_dir", "edge_midpoint",
                     "edge_next_opp_dir", "roi_next_opp_dir_pct"]].round({
                         "edge_next_same_dir": 4, "edge_midpoint": 4, "edge_next_opp_dir": 4,
                         "roi_next_opp_dir_pct": 2,
                     }).to_string(index=False))
    else:
        print("  no cells with n>0 — subset is empty.")

    # Trades-per-year estimate (universe spans 12 months by build_universe filter).
    print("\n--- trades/year estimate (under each subset, canonical p_in=0.60/p_out=0.90) ---")
    for r in rows:
        print(f"    {r['label']:14s} n={r['n']:>5}  ≈ {r['n']:>5} trades/yr "
              f"(universe = 12 months)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
