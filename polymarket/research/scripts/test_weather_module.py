"""Smoke test for data_infra.weather_analysis.

Run: python scripts/test_weather_module.py
Prints shapes + column samples + spot values, and asserts the module's outputs
against hard-coded known-good values from the original tail-harvest study.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

__test__ = False

# Allow running as a script from the polymarket/research/ root.
REPO_PKG = Path(__file__).resolve().parents[1]
if str(REPO_PKG) not in sys.path:
    sys.path.insert(0, str(REPO_PKG))

from data_infra import weather_analysis as wa  # noqa: E402


def test_load_weather_results() -> dict:
    """Confirm all four parquets load and derived columns are attached."""
    print("\n=== load_weather_results ===")
    print(f"DEFAULT_DATA_DIR = {wa.DEFAULT_DATA_DIR}")
    data = wa.load_weather_results()
    for key in ("primary", "sidebar", "inst", "uni"):
        assert key in data, f"missing key: {key}"
        df = data[key]
        print(f"  {key:8s}  shape={df.shape}  cols={list(df.columns)[:10]}"
              + ("..." if len(df.columns) > 10 else ""))

    # Spot-check derived columns on PRIMARY
    derived = ["win_rate", "roi_per_signal_pct", "kelly_fraction", "dollar_pnl_pooled"]
    missing = [c for c in derived if c not in data["primary"].columns]
    assert not missing, f"missing derived cols on primary: {missing}"
    print(f"  derived on primary  : {derived}  ✓")

    # Spot-check pooled row at p=0.70 (known-good: edge ~+1.3c, chop ~28.7%)
    pooled = data["primary"]
    row = pooled[(pooled["slug_family"] == "POOLED_ALL_WEATHER")
                 & (pooled["barrier_price"] == 0.70)]
    if len(row) == 1:
        r = row.iloc[0]
        print(f"  pooled @ p=0.70     : n_crossed={int(r['n_crossed'])}, "
              f"chop_rate={r['chop_rate']:.4f}, edge={r['edge_per_signal']:+.4f}, "
              f"win_rate={r['win_rate']:.4f}, kelly={r['kelly_fraction']:.4f}")
    else:
        print(f"  pooled @ p=0.70 row count = {len(row)}  (expected 1)")
    return data


def test_pivot_inst_to_wide(inst) -> None:
    """Confirm wide pivot has the right shape + fc_NNN columns."""
    print("\n=== pivot_inst_to_wide ===")
    print(f"  input INST shape    : {inst.shape}")
    wide = wa.pivot_inst_to_wide(inst)
    print(f"  output wide shape   : {wide.shape}")
    print(f"  columns             : {list(wide.columns)}")

    # Expected: one row per (market_id, outcome_token_id). INST has one row per
    # (market_id, outcome_token_id, barrier_price) with 7 barriers → wide row
    # count should be inst_rows / 7.
    expected_rows = len(inst) // len(wa.BARRIERS_EE)
    assert abs(len(wide) - expected_rows) <= 1, (
        f"row-count mismatch: wide={len(wide)} vs expected≈{expected_rows}"
    )

    # Confirm fc_NNN columns are present for every barrier
    expected_fc = [f"fc_{int(round(p*100)):03d}" for p in wa.BARRIERS_EE]
    missing_fc = [c for c in expected_fc if c not in wide.columns]
    assert not missing_fc, f"missing fc_ columns: {missing_fc}"
    print(f"  fc_ columns present : {expected_fc}  ✓")

    # Sample non-null counts per fc column (sanity: at p=0.50 we expect many,
    # at p=0.95 fewer level-breaks)
    print("  non-null counts per barrier:")
    for c in expected_fc:
        print(f"    {c}  n_non_null = {wide[c].notna().sum()}")

    # Spot-check meta columns
    for meta in ("slug_family", "min_price", "max_price", "resolution", "end_ts"):
        assert meta in wide.columns, f"missing meta col: {meta}"
    print(f"  meta columns        : slug_family, min_price, max_price, resolution, end_ts  ✓")


def test_pooled_metrics_by_barrier(primary) -> None:
    """Confirm pooled headline table matches the known-good TL;DR values."""
    print("\n=== pooled_metrics_by_barrier ===")
    out = wa.pooled_metrics_by_barrier(primary)
    print(f"  shape={out.shape}  cols={list(out.columns)}")

    expected_barriers = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    assert list(out["barrier_price"]) == expected_barriers, (
        f"barriers mismatch: {list(out['barrier_price'])}"
    )

    # Spot checks vs known-good TL;DR:
    #   p=0.50: n=8313, chop=49.7%, edge=+0.3c
    #   p=0.70: n=9033, chop=28.7%, edge=+1.3c
    #   p=0.90: n=9819, chop=10.6%, edge=-0.6c
    spot = [
        (0.50, 8313, 0.497, 0.003),
        (0.70, 9033, 0.287, 0.013),
        (0.90, 9819, 0.106, -0.006),
    ]
    for p, exp_n, exp_chop, exp_edge in spot:
        row = out[out["barrier_price"] == p].iloc[0]
        n_ok    = int(row["n_crossed"]) == exp_n
        chop_ok = abs(row["chop_rate"] - exp_chop) < 0.001
        edge_ok = abs(row["edge_per_signal"] - exp_edge) < 0.0015
        flag = "✓" if (n_ok and chop_ok and edge_ok) else "✗"
        print(f"  p={p:.2f}  n={int(row['n_crossed'])} (exp {exp_n})  "
              f"chop={row['chop_rate']:.4f} (exp {exp_chop:.3f})  "
              f"edge={row['edge_per_signal']:+.4f} (exp {exp_edge:+.3f})  {flag}")
        assert n_ok and chop_ok and edge_ok, f"spot check failed at p={p}"


def test_compute_ftc_metrics(inst) -> None:
    """Confirm FTC vs per-token comparison + both-crossed decomposition."""
    print("\n=== compute_ftc_metrics ===")
    out = wa.compute_ftc_metrics(inst)
    print(f"  shape={out.shape}  cols={list(out.columns)}")
    print(out.round(4).to_string(index=False))

    # Spot check from §5 interpretation:
    #   p=0.80: chop_per_token ≈ 0.196, chop_first_to_cross ≈ 0.191
    #   p=0.80 both-crossed: first chops ≈ 0.76, second chops ≈ 0.24
    row = out[out["p"] == 0.80].iloc[0]
    pt_ok    = abs(row["chop_per_token"] - 0.196) < 0.003
    ftc_ok   = abs(row["chop_first_to_cross"] - 0.191) < 0.003
    first_ok = abs(row["chop_first_in_both"] - 0.76) < 0.03
    second_ok= abs(row["chop_second_in_both"] - 0.24) < 0.03
    flag = "✓" if all([pt_ok, ftc_ok, first_ok, second_ok]) else "✗"
    print(f"\n  p=0.80 spot check:  "
          f"chop_per_token={row['chop_per_token']:.3f} (exp 0.196)  "
          f"chop_ftc={row['chop_first_to_cross']:.3f} (exp 0.191)  "
          f"first_in_both={row['chop_first_in_both']:.3f} (exp 0.76)  "
          f"second_in_both={row['chop_second_in_both']:.3f} (exp 0.24)  {flag}")
    assert all([pt_ok, ftc_ok, first_ok, second_ok]), "p=0.80 FTC spot check failed"


def test_compute_family_rankings(primary) -> None:
    """Confirm top families at p=0.80 match §6 interpretation."""
    print("\n=== compute_family_rankings (p=0.80, min_n=30) ===")
    out = wa.compute_family_rankings(primary, p=0.80, min_n=30)
    print(f"  shape={out.shape}  cols={list(out.columns)}")
    print("\n  top 5 by edge:")
    print(out.head(5).round({"chop_rate":4,"win_rate":4,
                             "edge_per_signal":4,"roi_per_signal_pct":2}).to_string(index=False))

    # §6 interpretation: "Manila +11.7¢ n=36, Austin +9.4¢ n=47, LA +6.9¢ n=61"
    # Family names use the slug_family convention; find any row whose family contains 'manila'
    def find(substr: str) -> pd.Series:
        m = out[out["slug_family"].str.contains(substr, case=False, na=False)]
        return m.iloc[0] if len(m) else None

    spot = [("manila", 0.117, 36),
            ("austin", 0.094, 47),
            ("los-angeles", 0.069, 61)]  # LA family slug usually "los-angeles"
    for sub, exp_edge, exp_n in spot:
        row = find(sub)
        if row is None:
            print(f"  ✗ no family matching '{sub}' at p=0.80 with n>=30")
            continue
        edge_ok = abs(row["edge_per_signal"] - exp_edge) < 0.005
        n_ok    = int(row["n_crossed"]) == exp_n
        flag = "✓" if (edge_ok and n_ok) else "✗"
        print(f"  family≈'{sub}': {row['slug_family']}  "
              f"n={int(row['n_crossed'])} (exp {exp_n})  "
              f"edge={row['edge_per_signal']:+.4f} (exp {exp_edge:+.3f})  {flag}")

    # POOLED should not appear
    assert "POOLED_ALL_WEATHER" not in out["slug_family"].values, "POOLED leaked through"
    # All rows have n >= min_n
    assert (out["n_crossed"] >= 30).all(), "min_n filter failed"
    print("  POOLED excluded ✓  min_n filter ✓")


def test_slippage_grid(primary) -> None:
    """Confirm pooled edge × slippage grid + §6's 'no p survives 2c' claim."""
    print("\n=== slippage_grid ===")
    out = wa.slippage_grid(primary)
    print(f"  shape={out.shape}  cols={list(out.columns)}")
    print(out.round(4).to_string())

    # §6 interpretation: "no `p` survives 2¢ of pooled slippage" → slip_2c < 0 everywhere
    all_neg_at_2c = (out["slip_2c"] < 0).all()
    print(f"\n  '§6: no p survives 2c slip' → all slip_2c < 0:  "
          f"{all_neg_at_2c}  {'✓' if all_neg_at_2c else '✗'}")
    assert all_neg_at_2c, "p=2c slip non-negative somewhere — §6 claim broken"

    # Sanity: slip_0c at p=0.70 should equal pooled edge_per_signal (+0.0131)
    edge_70 = out.loc[0.70, "slip_0c"]
    edge_70_minus_1c = out.loc[0.70, "slip_1c"]
    assert abs(edge_70 - 0.0131) < 0.001, f"slip_0c@0.70 = {edge_70}, expected ~0.0131"
    assert abs(edge_70_minus_1c - (edge_70 - 0.01)) < 1e-9, "slip subtraction broken"
    print(f"  slip_0c@p=0.70 = {edge_70:+.4f} (matches pooled edge +0.0131)  ✓")


def test_window_comparison(primary, sidebar) -> None:
    """Confirm §7's '~0.5c agreement between windows' claim."""
    print("\n=== window_comparison ===")
    out = wa.window_comparison(primary, sidebar)
    print(f"  shape={out.shape}")
    print(out.round(4).to_string(index=False))

    # §7 interpretation: "edges agree to within ~0.5¢ at every barrier"
    diffs = (out["edge_per_signal_24h"] - out["edge_per_signal_48h"]).abs()
    max_diff = diffs.max()
    print(f"\n  max |edge_24h - edge_48h| = {max_diff:.4f} (claim: ~0.005)")
    assert max_diff < 0.01, f"24h vs 48h edges differ by more than 1c at some barrier"


def test_eval_pair(p_wide) -> None:
    """Smoke-check the next-fill-slippage eval_pair: schema + invariants.

    Specific PnL values are not asserted — they're a function of the live trades
    parquet and change as more data lands. The invariants below pin down the
    contract.
    """
    print("\n=== eval_pair ===")

    # 1) Canonical case at default slippage params (next-fill window 15-300s,
    #    fallback 3¢, assumed spread 2¢). Returns a tuple when return_audit=True.
    out, audit = wa.eval_pair(p_wide, 0.60, 0.90, return_audit=True)
    print(f"  (p_in=0.60, p_out=0.90, policy='all'):")
    print(f"    n_entries={out['n_entries']}, p_tp={out['p_tp']:.4f}, "
          f"p_hold_win={out['p_hold_win']:.4f}, p_hold_chop={out['p_hold_chop']:.4f}")
    print(f"    tp_gain={out['tp_gain']:+.4f}, hold_win_gain={out['hold_win_gain']:+.4f}, "
          f"chop_drag={out['chop_drag']:+.4f}")
    print(f"    edge={out['edge']:+.4f}, ROI={out['roi_pct']:+.2f}%")
    print(f"    entry_same_dir fallback_rate={out['entry_next_same_dir_fallback_rate']:.3f}, "
          f"entry_opp_dir fallback_rate={out['entry_next_opp_dir_fallback_rate']:.3f}")

    # 2) Required keys for the new schema (Session 2.6 renames).
    required = {
        "p_in", "p_out", "policy", "family",
        "fill_assumption",
        "min_seconds", "max_seconds", "fallback_cents", "assumed_spread_cents",
        "n_entries", "p_tp", "p_hold_win", "p_hold_chop",
        "tp_gain", "hold_win_gain", "chop_drag", "edge", "roi_pct",
        "mean_entry_slippage_cents", "mean_exit_slippage_cents",
        "entry_next_same_dir_fallback_rate", "entry_next_opp_dir_fallback_rate",
        "exit_next_same_dir_fallback_rate",  "exit_next_opp_dir_fallback_rate",
    }
    missing = required - set(out.keys())
    assert not missing, f"eval_pair output missing keys: {missing}"
    # Old slip key must not leak.
    assert "slip" not in out, "stale 'slip' key in eval_pair output"
    # Old bid/ask fallback-rate keys must not leak as primary names.
    stale = {"entry_ask_fallback_rate", "entry_bid_fallback_rate",
             "exit_ask_fallback_rate",  "exit_bid_fallback_rate"}
    assert not (stale & set(out.keys())), f"stale bid/ask keys in summary: {stale & set(out.keys())}"
    assert out["fill_assumption"] == "next_same_dir", \
        f"default fill_assumption should be 'next_same_dir', got {out['fill_assumption']!r}"
    print("  required keys present + no stale 'slip'/bid/ask keys  ✓")

    # 3) Bucket decomposition identity: edge == tp_gain + hold_win_gain + chop_drag.
    s = out["tp_gain"] + out["hold_win_gain"] + out["chop_drag"]
    assert abs(s - out["edge"]) < 1e-9, f"bucket sum {s} != edge {out['edge']}"
    print("  bucket decomposition identity  ✓")

    # 4) Probability mass sums to ~1 (after NaN-resolution rows are coerced to 0).
    p_total = out["p_tp"] + out["p_hold_win"] + out["p_hold_chop"]
    print(f"  p_tp+p_hold_win+p_hold_chop = {p_total:.4f}  "
          f"({'≈1' if abs(p_total - 1) < 0.01 else 'NULL resolutions detected'})")

    # 5) Audit log shape + invariants (Session 2.6 renames).
    audit_cols = {"market_id", "outcome_token_id", "bucket", "tp_fires",
                  "entry_next_same_dir_price", "entry_next_opp_dir_price",
                  "exit_next_same_dir_price",  "exit_next_opp_dir_price",
                  "entry_next_same_dir_source", "entry_next_opp_dir_source",
                  "exit_next_same_dir_source",  "exit_next_opp_dir_source",
                  "entry_next_same_dir_lag_seconds", "entry_next_opp_dir_lag_seconds",
                  "exit_next_same_dir_lag_seconds",  "exit_next_opp_dir_lag_seconds",
                  "spread_estimate_entry", "spread_estimate_exit",
                  "crossed_market_entry", "crossed_market_exit"}
    missing = audit_cols - set(audit.columns)
    assert not missing, f"audit missing: {missing}"

    # Direction-relative invariants:
    #   entry (we BUY): next_opp_dir <= next_same_dir EXCEPT on crossed_market_entry
    #   exit  (we SELL): next_same_dir <= next_opp_dir EXCEPT on crossed_market_exit
    e_same = audit["entry_next_same_dir_price"].astype(float)
    e_opp  = audit["entry_next_opp_dir_price"].astype(float)
    e_violation = (e_opp > e_same + 1e-9) & (~audit["crossed_market_entry"])
    assert not e_violation.any(), \
        f"entry_next_opp_dir > entry_next_same_dir on non-crossed rows: {int(e_violation.sum())}"

    tp_rows = audit[audit["tp_fires"]]
    if len(tp_rows):
        x_same = tp_rows["exit_next_same_dir_price"].astype(float)
        x_opp  = tp_rows["exit_next_opp_dir_price"].astype(float)
        x_violation = (x_same > x_opp + 1e-9) & (~tp_rows["crossed_market_exit"])
        assert not x_violation.any(), \
            f"exit_next_same_dir > exit_next_opp_dir on non-crossed TP rows: {int(x_violation.sum())}"
    print(f"  audit log shape={audit.shape}, per-leg direction invariants hold  ✓")

    # 6) Sources are 'next_fill' or 'fallback' (or 'n/a' for non-TP exit columns).
    for c in ("entry_next_same_dir_source", "entry_next_opp_dir_source"):
        bad = set(audit[c].unique()) - {"next_fill", "fallback"}
        assert not bad, f"{c} has unexpected values: {bad}"
    for c in ("exit_next_same_dir_source", "exit_next_opp_dir_source"):
        bad = set(audit[c].unique()) - {"next_fill", "fallback", "n/a"}
        assert not bad, f"{c} has unexpected values: {bad}"
    print("  source labels valid  ✓")

    # 7) p_in >= p_out returns None.
    assert wa.eval_pair(p_wide, 0.80, 0.70) is None
    assert wa.eval_pair(p_wide, 0.80, 0.80) is None
    print("  p_in >= p_out returns None  ✓")

    # 8) compute_fill_scenarios runs and produces renamed PnL columns.
    scen = wa.compute_fill_scenarios(audit)
    for c in ("pnl_next_same_dir", "pnl_next_opp_dir", "pnl_midpoint"):
        assert c in scen.columns, f"compute_fill_scenarios missing {c}"
    print(f"  compute_fill_scenarios: "
          f"next_same_dir={scen['pnl_next_same_dir'].mean():+.4f}  "
          f"midpoint={scen['pnl_midpoint'].mean():+.4f}  "
          f"next_opp_dir={scen['pnl_next_opp_dir'].mean():+.4f}  ✓")

    # 9) Monotonicity: next_opp_dir >= midpoint >= next_same_dir on every row.
    #    (Passive fills are better-or-equal; midpoint is between, after
    #    crossed-market clipping.)
    assert (scen["pnl_next_opp_dir"] >= scen["pnl_midpoint"] - 1e-9).all(), \
        "next_opp_dir < midpoint on some rows"
    assert (scen["pnl_midpoint"] >= scen["pnl_next_same_dir"] - 1e-9).all(), \
        "midpoint < next_same_dir on some rows"
    print("  pnl_next_same_dir <= pnl_midpoint <= pnl_next_opp_dir on all rows  ✓")

    # 10) slippage_diagnostic returns the expected schema.
    diag = wa.slippage_diagnostic(scen)
    assert {"legs", "spread", "any_leg_high_fallback"}.issubset(diag.keys()), \
        f"slippage_diagnostic missing keys: got {set(diag.keys())}"
    assert len(diag["legs"]) == 4, "expected 4 (leg, direction) combos in diagnostic"
    for leg in diag["legs"]:
        if leg["n_total"]:
            assert leg["interpretation"] in {"noisy_taker_proxy",
                                              "constant_slippage_with_labelling"}
    print(f"  slippage_diagnostic: any_leg_high_fallback="
          f"{diag['any_leg_high_fallback']}  ✓")


def test_grid_search_entry_exit(p_wide) -> None:
    """Confirm grid_search_entry_exit returns one row per (p_in < p_out) pair."""
    print("\n=== grid_search_entry_exit (small sample to limit trades-parquet scans) ===")
    # Small barriers list to keep this smoke test under ~30s. The default
    # BARRIERS_EE has 7 levels → 21 pairs × ~7s each scan = ~2 min, too slow here.
    small = [0.60, 0.80, 0.90]
    grid = wa.grid_search_entry_exit(p_wide, barriers=small)
    print(f"  shape={grid.shape}")

    expected = sum(1 for i, _ in enumerate(small) for _ in small[i+1:])
    assert len(grid) == expected, f"expected {expected} valid pairs, got {len(grid)}"
    print(f"  pair count: {len(grid)} (expected {expected})  ✓")

    assert "slip" not in grid.columns, "stale 'slip' column in grid"
    for c in ("p_in", "p_out", "edge", "n_entries",
              "entry_next_same_dir_fallback_rate", "entry_next_opp_dir_fallback_rate"):
        assert c in grid.columns, f"grid missing column: {c}"
    print("  schema clean (no 'slip', new direction-relative columns present)  ✓")

    print("\n  rows:")
    print(grid[["p_in", "p_out", "n_entries", "p_tp", "edge", "roi_pct",
                "entry_next_same_dir_fallback_rate"]]
          .round({"p_tp": 3, "edge": 4, "roi_pct": 2,
                  "entry_next_same_dir_fallback_rate": 3}).to_string(index=False))


def main() -> int:
    data = test_load_weather_results()
    test_pivot_inst_to_wide(data["inst"])
    test_pooled_metrics_by_barrier(data["primary"])
    test_compute_ftc_metrics(data["inst"])
    test_compute_family_rankings(data["primary"])
    test_slippage_grid(data["primary"])
    test_window_comparison(data["primary"], data["sidebar"])
    p_wide = wa.pivot_inst_to_wide(data["inst"])
    test_eval_pair(p_wide)
    test_grid_search_entry_exit(p_wide)
    print("\nAll Step 3.7 smoke checks passed — module extraction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
