"""Session 2.6 — compare PnL under next_same_dir / next_opp_dir / midpoint proxies
with slippage diagnostics, time-to-fill distribution, spread estimates, and
crossed-market anomaly fractions.

Default: canonical (p_in=0.60, p_out=0.90, policy='all'). Optional --grid runs
the full 21-pair sweep to identify the top-edge cell.

Run:
    python scripts/compare_fill_scenarios.py            # canonical only
    python scripts/compare_fill_scenarios.py --grid     # canonical + top cell
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_PKG = Path(__file__).resolve().parents[1]
if str(REPO_PKG) not in sys.path:
    sys.path.insert(0, str(REPO_PKG))

from data_infra import weather_analysis as wa  # noqa: E402


CAVEAT = """
============================================================================
CAVEAT (always read alongside the numbers above)
============================================================================
These are PROXIES for execution cost built from realised fills. They are not
literal bid/ask quotes:

  next_same_dir: tends to be WORSE than the true touch — the orderbook is
                 consumed by the leader's fill and re-prints elsewhere within
                 the window.
  next_opp_dir : tends to be BETTER than the true passive price — an actual
                 passive maker order at the touch may not have been filled.
  midpoint     : midpoint of the two proxies (clipped to never be worse than
                 next_same_dir per direction, to avoid the crossed-market
                 anomaly).

The gap between next_same_dir and next_opp_dir is a SENSITIVITY TEST.
Interpret it as a range, not a precise estimate.

Where a leg's fallback_pct > 50%, the scenario edge is dominated by the
constant `fallback_cents` parameter; the "next_*_dir" labelling is mostly
cosmetic for that leg. Read those rows as "constant slippage with extra
labelling," not a data-derived proxy.
============================================================================
"""


def _print_legs_table(legs: list[dict]) -> None:
    rows = []
    for L in legs:
        if L.get("n_total", 0) == 0:
            rows.append({"leg": L["leg"], "dir": L["direction"], "n": 0})
            continue
        rows.append({
            "leg":          L["leg"],
            "dir":          L["direction"],
            "n":            L["n_total"],
            "fallback_pct": L["fallback_pct"],
            "slip_p10":     L["slip_cents_p10"],
            "slip_med":     L["slip_cents_median"],
            "slip_p90":     L["slip_cents_p90"],
            "lag_med_s":    L["lag_median_s"],
            "lag_<30s":     L["lag_under_30s_share"],
            "lag_2-5min":   L["lag_2min_to_5min_share"],
            "shape":        L["shape"],
            "interpretation": L["interpretation"],
        })
    print(pd.DataFrame(rows).round({
        "fallback_pct": 3, "slip_p10": 2, "slip_med": 2, "slip_p90": 2,
        "lag_med_s": 1, "lag_<30s": 3, "lag_2-5min": 3,
    }).to_string(index=False))


def _print_spread_table(spread: list[dict]) -> None:
    rows = []
    for S in spread:
        if S.get("n", 0) == 0:
            rows.append({"leg": S["leg"], "n": 0})
            continue
        rows.append({
            "leg":                  S["leg"],
            "n":                    S["n"],
            "median_signed¢":       S["median_signed_cents"],
            "median_abs¢":          S["median_abs_cents"],
            "p10_signed¢":          S["p10_signed_cents"],
            "p90_signed¢":          S["p90_signed_cents"],
            "crossed_share":        S["crossed_market_share"],
            "flag_thin":            S["flag_thin_markets"],
            "flag_noisy_spread":    S["flag_noisy_spread"],
        })
    print(pd.DataFrame(rows).round({
        "median_signed¢": 2, "median_abs¢": 2,
        "p10_signed¢": 2, "p90_signed¢": 2,
        "crossed_share": 3,
    }).to_string(index=False))


def report_for_pair(p_wide: pd.DataFrame, p_in: float, p_out: float,
                    policy: str = "all") -> dict:
    print(f"\n{'=' * 80}")
    print(f"  p_in={p_in:.2f}  p_out={p_out:.2f}  policy={policy}")
    print(f"  next-fill window=({wa.DEFAULT_MIN_SECONDS}s, {wa.DEFAULT_MAX_SECONDS}s]"
          f"  fallback_cents={wa.DEFAULT_FALLBACK_CENTS}  "
          f"assumed_spread_cents={wa.DEFAULT_SPREAD_CENTS}")
    print(f"{'=' * 80}")

    t0 = time.time()
    summary, audit = wa.eval_pair(
        p_wide, p_in, p_out, policy=policy, return_audit=True,
    )
    print(f"  eval_pair time: {time.time() - t0:.1f}s   n_entries={summary['n_entries']}")

    scen = wa.compute_fill_scenarios(audit)
    diag = wa.slippage_diagnostic(scen, p_in=p_in, p_out=p_out)

    edge_same = float(scen["pnl_next_same_dir"].mean())
    edge_opp  = float(scen["pnl_next_opp_dir"].mean())
    edge_mid  = float(scen["pnl_midpoint"].mean())
    sd_same   = float(scen["pnl_next_same_dir"].std(ddof=1))
    sd_opp    = float(scen["pnl_next_opp_dir"].std(ddof=1))
    sd_mid    = float(scen["pnl_midpoint"].std(ddof=1))

    print("\n  --- HEADLINE: edge per $1 of payout (mean PnL/share across all entries) ---")
    hdr = pd.DataFrame([
        ["next_same_dir", edge_same, 100 * edge_same / p_in, sd_same, edge_same / sd_same if sd_same else float("nan")],
        ["midpoint",      edge_mid,  100 * edge_mid  / p_in, sd_mid,  edge_mid  / sd_mid  if sd_mid  else float("nan")],
        ["next_opp_dir",  edge_opp,  100 * edge_opp  / p_in, sd_opp,  edge_opp  / sd_opp  if sd_opp  else float("nan")],
    ], columns=["scenario", "edge_$", "ROI_%", "sd", "Sharpe_per_trade"])
    print(hdr.round({"edge_$": 4, "ROI_%": 2, "sd": 4, "Sharpe_per_trade": 4}).to_string(index=False))
    lever_cents = (edge_opp - edge_same) * 100
    print(f"\n  OPERATIONAL LEVER (next_opp_dir − next_same_dir): {lever_cents:+.2f}¢/share")
    if diag["any_leg_high_fallback"]:
        print(f"  ⚠️  AT LEAST ONE LEG HAS FALLBACK > 50%. Read scenario edges as ")
        print(f"      'constant slippage with extra labelling' for those legs.")

    print("\n  --- DIAGNOSTIC: per-leg slippage proxy distribution ---")
    _print_legs_table(diag["legs"])

    print("\n  --- DIAGNOSTIC: spread estimate per leg (signed cents) ---")
    print("  entry: natural sign POSITIVE (next_same_dir − next_opp_dir = ASK − BID)")
    print("  exit:  natural sign NEGATIVE (next_same_dir − next_opp_dir = BID − ASK)")
    print("  flag_thin: median |spread| > 5¢; flag_noisy_spread: crossed-market share > 15%")
    _print_spread_table(diag["spread"])

    print("\n  --- bucket-conditional mean PnL ---")
    for bucket in ("tp", "hold_win", "hold_chop"):
        sub = scen[scen["bucket"] == bucket]
        if len(sub) == 0:
            continue
        print(f"     {bucket:10s} n={len(sub):>5}  "
              f"next_same_dir={sub['pnl_next_same_dir'].mean():+.4f}  "
              f"midpoint={sub['pnl_midpoint'].mean():+.4f}  "
              f"next_opp_dir={sub['pnl_next_opp_dir'].mean():+.4f}")

    return {
        "p_in": p_in, "p_out": p_out,
        "n_entries": summary["n_entries"],
        "edge_next_same_dir": edge_same,
        "edge_next_opp_dir":  edge_opp,
        "edge_midpoint":      edge_mid,
        "operational_lever_cents": lever_cents,
        "any_leg_high_fallback": diag["any_leg_high_fallback"],
        "crossed_share_entry": diag["spread"][0].get("crossed_market_share", float("nan")),
        "crossed_share_exit":  diag["spread"][1].get("crossed_market_share", float("nan")),
    }


def _print_window_comparison(cmp: dict) -> None:
    print(f"\n  --- WINDOW COMPARISON: {sorted(cmp['windows'].keys())}s ---")
    rows = []
    for w in sorted(cmp["windows"].keys()):
        d = cmp["windows"][w]
        rows.append({
            "window_s":          d["window_seconds"],
            "fb_e_same":         d["fallback_rate_entry_same_dir"],
            "fb_e_opp":          d["fallback_rate_entry_opp_dir"],
            "fb_x_same":         d["fallback_rate_exit_same_dir"],
            "fb_x_opp":          d["fallback_rate_exit_opp_dir"],
            "edge_same":         d["edge_next_same_dir"],
            "edge_mid":          d["edge_midpoint"],
            "edge_opp":          d["edge_next_opp_dir"],
            "entry_lag_p10s":    d["entry_lag_p10_s"],
            "entry_lag_meds":    d["entry_lag_median_s"],
            "entry_lag_p90s":    d["entry_lag_p90_s"],
        })
    print(pd.DataFrame(rows).round({
        "fb_e_same": 3, "fb_e_opp": 3, "fb_x_same": 3, "fb_x_opp": 3,
        "edge_same": 4, "edge_mid": 4, "edge_opp": 4,
        "entry_lag_p10s": 1, "entry_lag_meds": 1, "entry_lag_p90s": 1,
    }).to_string(index=False))

    d = cmp["delta_short_to_long"]
    print(f"\n  Δ short ({d['short_seconds']}s) → long ({d['long_seconds']}s):")
    print(f"    entry_same_dir fallback drop: {d['fallback_drop_entry_same_dir_pp']:+.2f} pp")
    print(f"    entry_opp_dir  fallback drop: {d['fallback_drop_entry_opp_dir_pp']:+.2f} pp")
    print(f"    edge change   (same_dir): {d['edge_change_same_dir_cents']:+.2f}¢/share")
    print(f"    edge change   (midpoint): {d['edge_change_midpoint_cents']:+.2f}¢/share")
    print(f"    edge change   (opp_dir):  {d['edge_change_opp_dir_cents']:+.2f}¢/share")


def _print_any_fill_diagnostic(diag: dict) -> None:
    print(f"\n  --- TIME-TO-FIRST-OTHER-FILL (any side, any direction) ---")
    print(f"  cap = {diag['max_seconds']}s    n_anchors = {diag['n_anchors']}    "
          f"n_with_any_fill = {diag['n_with_any_fill_within_max']}")
    print(f"  cumulative share of anchors with a fill within X:")
    print(f"     <30s   : {diag['pct_within_30s']*100:>5.1f}%")
    print(f"     <60s   : {diag['pct_within_60s']*100:>5.1f}%")
    print(f"     <120s  : {diag['pct_within_120s']*100:>5.1f}%")
    print(f"     <300s  : {diag['pct_within_300s']*100:>5.1f}%   (= 5-min window cutoff)")
    print(f"     <600s  : {diag['pct_within_600s']*100:>5.1f}%   (= 10-min window cutoff)")
    print(f"     none<={diag['max_seconds']}s: {diag['pct_no_fill_within_max']*100:>5.1f}%")
    print(f"  lag percentiles (over anchors that found a fill):")
    print(f"     p10/p25/p50/p75/p90/p99 = "
          f"{diag['lag_p10_s']:.0f} / {diag['lag_p25_s']:.0f} / "
          f"{diag['lag_median_s']:.0f} / {diag['lag_p75_s']:.0f} / "
          f"{diag['lag_p90_s']:.0f} / {diag['lag_p99_s']:.0f} seconds")


def main() -> int:
    inst = pd.read_parquet(wa.DEFAULT_DATA_DIR / "weather_tail_per_instance.parquet")
    p_wide = wa.pivot_inst_to_wide(inst)
    print(f"p_wide rows: {len(p_wide):,}")

    canonical = report_for_pair(p_wide, p_in=0.60, p_out=0.90, policy="all")

    if "--no-extras" not in sys.argv:
        print(f"\n\n{'#' * 80}")
        print("# ADDITIONAL DIAGNOSTICS — window comparison + raw market activity")
        print(f"{'#' * 80}")

        t0 = time.time()
        win_cmp = wa.compare_windows_diagnostic(
            p_wide, p_in=0.60, p_out=0.90, windows_seconds=(300, 600),
        )
        print(f"  compare_windows_diagnostic: {time.time() - t0:.1f}s")
        _print_window_comparison(win_cmp)

        t0 = time.time()
        any_fill = wa.time_to_first_any_fill_diagnostic(
            p_wide, p_in=0.60, max_seconds=1800,
        )
        print(f"\n  time_to_first_any_fill_diagnostic: {time.time() - t0:.1f}s")
        _print_any_fill_diagnostic(any_fill)

    if "--grid" in sys.argv:
        print(f"\n\n{'#' * 80}")
        print("# Grid scan across all (p_in, p_out) pairs to find top edge_next_same_dir")
        print(f"{'#' * 80}")
        rows = []
        for p_in in wa.BARRIERS_EE:
            for p_out in wa.BARRIERS_EE:
                if not (p_in < p_out):
                    continue
                try:
                    r = report_for_pair(p_wide, p_in, p_out, policy="all")
                    rows.append(r)
                except Exception as e:
                    print(f"  (skipped p_in={p_in} p_out={p_out}: {e})")

        if rows:
            df = pd.DataFrame(rows)
            best = df.loc[df["edge_next_same_dir"].idxmax()]
            print(f"\n\n{'*' * 80}")
            print(f"* TOP CELL by edge_next_same_dir: p_in={best['p_in']:.2f} p_out={best['p_out']:.2f}")
            print(f"*   same_dir={best['edge_next_same_dir']:+.4f}  "
                  f"midpoint={best['edge_midpoint']:+.4f}  "
                  f"opp_dir={best['edge_next_opp_dir']:+.4f}")
            print(f"*   operational lever = {best['operational_lever_cents']:+.2f}¢/share")
            print(f"*   high_fallback_warning = {bool(best['any_leg_high_fallback'])}")
            print(f"{'*' * 80}")

            print("\n--- all cells (top 10 by edge_next_same_dir) ---")
            print(df.sort_values("edge_next_same_dir", ascending=False)
                    .head(10).round(4).to_string(index=False))

    print(CAVEAT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
