"""Block A1.4i maker-at-mid pyramiding K-cap sweep.

This sidecar uses A14c's maker-at-mid mechanics and A14h's explicit exit
timestamps to test whether the maker thesis survives finite concurrent
position capacity. K=1 must reproduce A14h, and K=infinite must reproduce A14c
before the intermediate K results are reported.

The current-level TOB signal is horizon-invariant, so this script writes one
row per market/K/W/H/exit grid and calibrates against the duplicated
``signal_horizon == 1`` rows from A14c and A14h.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from dali_block_a1_analyze import family_category
from dali_block_a14c_maker_at_mid import (
    BOOTSTRAP_CHUNK_SECONDS,
    BOOTSTRAP_SAMPLES,
    EXIT_CONVENTIONS,
    FILL_WINDOWS,
    HOLD_HORIZONS,
    RNG_SEED,
    ROOT,
    add_adverse_selection,
    assign_top_decile_signals,
    bps,
    load_candidates,
    load_feature_subset,
    markdown_table,
    pct,
    safe_text,
    simulate_entry_fills,
)
from dali_block_a14h_maker_non_overlap import (
    A14C_RESULTS,
    ANALYSIS,
    NOTE as A14H_NOTE,
    OUT_CSV as A14H_RESULTS,
    NOTES,
    apply_non_overlap,
    simulate_exit_with_time,
)


OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14i_pyramiding_results.csv"
NOTE = NOTES / "block_a14i_pyramiding_findings.md"
K_LEVELS: tuple[int | str, ...] = (1, 2, 3, 5, "inf")
REFERENCE_SIGNAL_HORIZON = 1
ROBUST_MIN_FILLS = 5
CALIBRATION_COLUMNS = [
    "n_executed_fills",
    "fill_rate",
    "mean_pnl_bps",
    "median_pnl_bps",
    "win_rate",
    "ci_lo",
    "ci_hi",
]


def k_label(k_cap: int | str) -> str:
    return "inf" if k_cap == "inf" else str(int(k_cap))


def finite_metric_equal(left: pd.Series, right: pd.Series, tol: float = 1e-9) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_nan = left_num.isna() & right_num.isna()
    return both_nan | np.isclose(left_num, right_num, rtol=0.0, atol=tol, equal_nan=True)


def active_count(intervals: list[tuple[int, int]], timestamp_ns: int) -> int:
    return sum(start <= timestamp_ns <= end for start, end in intervals)


def max_concurrent(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    points: list[tuple[int, int]] = []
    for start, end in intervals:
        points.append((start, 1))
        points.append((end, -1))
    # Starts before ends at the same timestamp mirrors the conservative
    # inclusive interval convention used by A14h's non-overlap selector.
    points.sort(key=lambda item: (item[0], -item[1]))
    current = 0
    peak = 0
    for _, delta in points:
        current += delta
        peak = max(peak, current)
    return peak


def valid_candidate_exits(candidate_exits: pd.DataFrame) -> pd.DataFrame:
    if candidate_exits.empty:
        return candidate_exits.copy()
    candidates = candidate_exits[
        candidate_exits["pnl_bps"].replace([np.inf, -np.inf], np.nan).notna()
        & candidate_exits["fill_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & candidate_exits["exit_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if candidates.empty:
        return candidates
    candidates["signal_time_ns"] = candidates["event_time_ns"].astype("int64")
    candidates["fill_time_ns_int"] = candidates["fill_time_ns"].astype("int64")
    candidates["exit_time_ns_int"] = candidates["exit_time_ns"].astype("int64")
    return candidates.sort_values(
        ["fill_time_ns_int", "signal_time_ns", "abs_signal"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def select_k_cap(candidate_exits: pd.DataFrame, k_cap: int | str) -> pd.DataFrame:
    """Select fills under a per-market concurrent-position K cap."""
    candidates = valid_candidate_exits(candidate_exits)
    if candidates.empty:
        return candidates
    if k_cap == "inf":
        out = candidates.copy()
        out["executed_fill_rank"] = np.arange(1, len(out) + 1)
        return out
    if int(k_cap) == 1:
        return apply_non_overlap(candidates)

    k = int(k_cap)
    intervals: list[tuple[int, int]] = []
    keep_positions: list[int] = []
    for pos, row in enumerate(candidates.itertuples(index=False)):
        signal_ns = int(row.signal_time_ns)
        fill_ns = int(row.fill_time_ns_int)
        exit_ns = int(row.exit_time_ns_int)
        if exit_ns <= fill_ns:
            continue
        if active_count(intervals, signal_ns) >= k or active_count(intervals, fill_ns) >= k:
            continue
        keep_positions.append(pos)
        intervals.append((fill_ns, exit_ns))
    if not keep_positions:
        return candidates.iloc[0:0].copy()
    out = candidates.iloc[keep_positions].copy()
    out["executed_fill_rank"] = np.arange(1, len(out) + 1)
    return out


def bootstrap_mean_ci(rows: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = rows[["fill_time_ns", "pnl_bps"]].dropna().copy()
    clean = clean[np.isfinite(clean["pnl_bps"])]
    if len(clean) < ROBUST_MIN_FILLS:
        return math.nan, math.nan
    elapsed = (clean["fill_time_ns"] - clean["fill_time_ns"].min()) / 1_000_000_000.0
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    pnl = clean["pnl_bps"].to_numpy(dtype=float)
    vals: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        vals.append(float(np.nanmean(pnl[idx])))
    if len(vals) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_combo(
    *,
    market_label: str,
    slug: str,
    family: str,
    sample_size_label: str,
    k_cap: int | str,
    fill_window: int,
    hold_horizon: int,
    exit_convention: str,
    n_signals: int,
    executed: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    n = int(len(executed))
    fill_rate = n / n_signals if n_signals else math.nan
    intervals = (
        list(zip(executed["fill_time_ns_int"].astype(int), executed["exit_time_ns_int"].astype(int)))
        if n and "fill_time_ns_int" in executed
        else []
    )
    ci_lo, ci_hi = bootstrap_mean_ci(executed, seed)
    return {
        "market": market_label,
        "slug": slug,
        "family": family,
        "sample_size_label": sample_size_label,
        "k_cap": k_label(k_cap),
        "fill_window_sec": fill_window,
        "hold_sec": hold_horizon,
        "exit_convention": exit_convention,
        "n_signals": n_signals,
        "n_executed_fills": n,
        "fill_rate": fill_rate,
        "mean_pnl_bps": float(executed["pnl_bps"].mean()) if n else math.nan,
        "median_pnl_bps": float(executed["pnl_bps"].median()) if n else math.nan,
        "win_rate": float(executed["pnl_bps"].gt(0).mean()) if n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "max_concurrent_positions_observed": max_concurrent(intervals),
    }


def run_simulation(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_meta = candidates.set_index(["run_id", "market_id"]).to_dict("index")
    total_markets = df[["run_id", "market_id"]].drop_duplicates().shape[0]
    for market_idx, ((run_id, market_id), market) in enumerate(
        df.groupby(["run_id", "market_id"], sort=False),
        start=1,
    ):
        market = market.sort_values(["asset_id", "received_at"]).copy()
        meta = candidate_meta.get((run_id, market_id), {})
        family = str(meta.get("family") or market["family"].replace("", np.nan).dropna().iloc[0])
        category = family_category(family)
        sample_size_label = str(meta.get("sample_size_label", ""))
        slug = (
            str(market["slug"].replace("", np.nan).dropna().iloc[0])
            if market["slug"].astype(bool).any()
            else str(market_id)
        )
        market_label = f"{run_id}:{market_id}"
        signals = assign_top_decile_signals(market)
        n_signals = int(len(signals))
        print(
            f"A14i market {market_idx:02d}/{total_markets:02d} {market_label} "
            f"signals={n_signals:,}",
            flush=True,
        )

        entry_by_window: dict[int, pd.DataFrame] = {}
        for fill_window in FILL_WINDOWS:
            entry = simulate_entry_fills(signals, market, fill_window)
            entry = entry[entry["entry_filled"]].copy()
            if not entry.empty:
                entry = add_adverse_selection(entry, market)
            entry_by_window[fill_window] = entry

        for fill_window in FILL_WINDOWS:
            entry = entry_by_window[fill_window]
            for hold_horizon in HOLD_HORIZONS:
                for exit_convention in EXIT_CONVENTIONS:
                    exited = (
                        simulate_exit_with_time(entry, market, hold_horizon, exit_convention, category)
                        if not entry.empty
                        else entry
                    )
                    for k_idx, k_cap in enumerate(K_LEVELS):
                        executed = select_k_cap(exited, k_cap)
                        seed_offset = 0 if k_cap in (1, "inf") else k_idx * 1_000_000
                        rows.append(
                            summarize_combo(
                                market_label=market_label,
                                slug=slug,
                                family=family,
                                sample_size_label=sample_size_label,
                                k_cap=k_cap,
                                fill_window=fill_window,
                                hold_horizon=hold_horizon,
                                exit_convention=exit_convention,
                                n_signals=n_signals,
                                executed=executed,
                                seed=RNG_SEED
                                + market_idx * 10_000
                                + REFERENCE_SIGNAL_HORIZON * 100
                                + fill_window * 10
                                + hold_horizon
                                + (0 if exit_convention == "exit_symmetric_maker" else 1)
                                + seed_offset,
                            )
                        )

    columns = [
        "market",
        "slug",
        "family",
        "sample_size_label",
        "k_cap",
        "fill_window_sec",
        "hold_sec",
        "exit_convention",
        "n_signals",
        "n_executed_fills",
        "fill_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "max_concurrent_positions_observed",
    ]
    return pd.DataFrame(rows)[columns]


def load_reference(path: Path, *, k_cap: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["signal_horizon"], errors="coerce").eq(REFERENCE_SIGNAL_HORIZON)].copy()
    if k_cap == "inf":
        df = df.rename(columns={"n_filled": "n_executed_fills", "n_signal_events": "n_signals"})
    else:
        df = df.rename(columns={"n_signal_events": "n_signals"})
    df["k_cap"] = k_cap
    keys = ["market", "fill_window_sec", "hold_sec", "exit_convention", "k_cap"]
    return df[keys + CALIBRATION_COLUMNS].drop_duplicates(keys)


def calibration_report(results: pd.DataFrame) -> tuple[bool, str]:
    keys = ["market", "fill_window_sec", "hold_sec", "exit_convention", "k_cap"]
    left = results[keys + CALIBRATION_COLUMNS].copy()
    refs = pd.concat(
        [
            load_reference(A14H_RESULTS, k_cap="1"),
            load_reference(A14C_RESULTS, k_cap="inf"),
        ],
        ignore_index=True,
    )
    merged = left[left["k_cap"].isin(["1", "inf"])].merge(
        refs,
        on=keys,
        suffixes=("_a14i", "_ref"),
        how="outer",
        indicator=True,
    )
    problems: list[str] = []
    if not merged["_merge"].eq("both").all():
        problems.append(f"key mismatch rows={int(merged['_merge'].ne('both').sum())}")
    for col in CALIBRATION_COLUMNS:
        ok = finite_metric_equal(merged[f"{col}_a14i"], merged[f"{col}_ref"])
        if not ok.all():
            bad = merged.loc[~ok, keys + [f"{col}_a14i", f"{col}_ref"]].head(5)
            problems.append(f"{col} mismatch: {bad.to_dict(orient='records')}")
    if problems:
        return False, "\n".join(problems)
    return True, f"K=1 matched {A14H_RESULTS.name}; K=inf matched {A14C_RESULTS.name}."


def robust_positive_count(results: pd.DataFrame) -> int:
    return int(
        (
            results["mean_pnl_bps"].gt(0)
            & results["ci_lo"].gt(0)
            & results["n_executed_fills"].ge(ROBUST_MIN_FILLS)
        ).sum()
    )


def pnl_vs_k_table(results: pd.DataFrame) -> str:
    best = (
        results.sort_values(["market", "k_cap", "mean_pnl_bps", "fill_rate"], ascending=[True, True, False, False])
        .groupby(["market", "k_cap"], as_index=False)
        .head(1)
        .copy()
    )
    rows: list[list[str]] = []
    for market, sub in best.groupby("market", sort=True):
        slug = safe_text(sub["slug"].dropna().iloc[0], 34)
        cells = []
        for k in ["1", "2", "3", "5", "inf"]:
            row = sub[sub["k_cap"].eq(k)]
            if row.empty:
                cells.append("n/a")
            else:
                r = row.iloc[0]
                cells.append(f"{bps(float(r['mean_pnl_bps']))} ({int(r['n_executed_fills'])})")
        rows.append([safe_text(market, 16), slug, *cells])
    return markdown_table(["market", "slug", "K=1", "K=2", "K=3", "K=5", "K=inf"], rows)


def k_summary_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for k, sub in results.groupby("k_cap", sort=False):
        positives = int(sub["mean_pnl_bps"].gt(0).sum())
        robust = robust_positive_count(sub)
        best = sub.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).iloc[0]
        rows.append(
            [
                str(k),
                f"{positives}/{len(sub)}",
                str(robust),
                safe_text(best["market"], 16),
                safe_text(best["slug"], 34),
                f"W={int(best['fill_window_sec'])}, H={int(best['hold_sec'])}, "
                f"{str(best['exit_convention']).replace('exit_', '')}",
                f"{int(best['n_executed_fills']):,}",
                pct(float(best["fill_rate"])),
                bps(float(best["mean_pnl_bps"])),
                f"[{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]",
                str(int(best["max_concurrent_positions_observed"])),
            ]
        )
    return markdown_table(
        ["K", "positive cells", "robust+", "best market", "slug", "best grid", "fills", "fill rate", "mean", "CI", "max conc."],
        rows,
    )


def btc_curve_table(results: pd.DataFrame) -> str:
    target = results[
        results["market"].eq("a0b:2364426")
        & results["fill_window_sec"].eq(10)
        & results["hold_sec"].eq(30)
        & results["exit_convention"].eq("exit_symmetric_maker")
    ].copy()
    if target.empty:
        return "_BTC target cell missing._"
    rows: list[list[str]] = []
    for k in ["1", "2", "3", "5", "inf"]:
        row = target[target["k_cap"].eq(k)].iloc[0]
        rows.append(
            [
                str(k),
                f"{int(row['n_executed_fills']):,}",
                pct(float(row["fill_rate"])),
                bps(float(row["mean_pnl_bps"])),
                f"[{bps(float(row['ci_lo']))}, {bps(float(row['ci_hi']))}]",
                pct(float(row["win_rate"])),
                str(int(row["max_concurrent_positions_observed"])),
            ]
        )
    return markdown_table(["K", "fills", "fill rate", "mean", "CI", "win", "max conc."], rows)


def write_note(results: pd.DataFrame, calibration_text: str) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    robust = robust_positive_count(results)
    positives = int(results["mean_pnl_bps"].gt(0).sum())
    best = results.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).iloc[0]
    robust_df = results[
        results["mean_pnl_bps"].gt(0)
        & results["ci_lo"].gt(0)
        & results["n_executed_fills"].ge(ROBUST_MIN_FILLS)
    ]
    if robust:
        robust_ks = ", ".join(sorted(robust_df["k_cap"].unique(), key=lambda x: (x == "inf", str(x))))
        headline_tail = f"{robust:,} cells cleared the robustness bar, at K values: {robust_ks}."
        recommendation = "specific K-constrained maker strategy is the only surviving Dali candidate; stress it next with queue, latency, cancellation, and inventory sizing."
    else:
        headline_tail = "no cell cleared the robustness bar at any K."
        recommendation = "Dali microstructure is closed on A0/A0b unless A2 is explicitly reframed as episode discovery rather than deployable maker/taker PnL."

    note = f"""---
tags: [dali, block-a14i, pyramiding, results]
---

# Block A1.4i Pyramiding K-Cap Sweep

## Headline

Calibration passed: {calibration_text} Across {len(results):,} market/K/grid cells, {positives:,} had positive mean PnL and {headline_tail} The best cell was K={best['k_cap']} on `{safe_text(best['market'])}` / `{safe_text(best['slug'])}` with W={int(best['fill_window_sec'])}s, H={int(best['hold_sec'])}s, `{best['exit_convention']}`, {int(best['n_executed_fills']):,} fills, mean {bps(float(best['mean_pnl_bps']))}, and CI [{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}].

## Method

- Universe: same as A14c/A14h, A1 markets labeled `primary_read` or `thin_wide_CI`.
- Signal: per-market top absolute decile of current-level TOB imbalance.
- Entry/fill/exit/rebate: identical to A14c, using A14h's explicit exit timestamps for capacity accounting.
- K rule: at most K concurrent open maker positions per market. A candidate fill is accepted only if fewer than K positions are open at both signal time and fill time.
- K sweep: `1`, `2`, `3`, `5`, `inf`. K=1 reproduces A14h; K=inf reproduces A14c.
- Bootstrap: 200 resamples over contiguous 300s blocks on filled PnL.

## K Summary

{k_summary_table(results)}

## BTC-4h A14c Winner Curve

This is the original A14c positive cell, `a0b:2364426`, W=10s, H=30s, `exit_symmetric_maker`.

{btc_curve_table(results)}

## PnL-vs-K Curve Per Market

Each cell shows the best mean PnL for that market and K, with executed fill count in parentheses.

{pnl_vs_k_table(results)}

## Cross-Market Verdict

Intermediate K values explain the overlap artifact rather than rescuing it. If a cell only becomes attractive at high K or infinite K, the edge is capacity-dependent repeated exposure to the same episode, not a clean one-position maker edge. The robustness bar is mean > 0, CI lower > 0, and at least {ROBUST_MIN_FILLS} fills.

## Recommendation

Recommended next action for Justin: {recommendation}
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not A14C_RESULTS.exists():
        raise SystemExit(f"missing A14c results: {A14C_RESULTS}")
    if not A14H_RESULTS.exists():
        raise SystemExit(f"missing A14h results: {A14H_RESULTS}; run A14h first")
    candidates = load_candidates()
    features = load_feature_subset(candidates)
    results = run_simulation(features, candidates)
    ok, calibration_text = calibration_report(results)
    if not ok:
        raise SystemExit(f"calibration failed:\n{calibration_text}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results, calibration_text)
    print(f"calibration ok: {calibration_text}")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
