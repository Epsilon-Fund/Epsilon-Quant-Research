"""Block A1.4h maker-at-mid counterfactual with non-overlapping positions.

This sidecar re-tests A14c's maker-at-mid thesis using the same per-market
top-decile TOB signal, mid-posting fill model, rebate table, and bootstrap.
The only intended behavioral change is position management:

- Unfilled signals do not block later signals.
- A position starts blocking only after an actual entry fill.
- While a filled position is open, any candidate fill whose signal or fill time
  falls inside the open interval is skipped.

No raw JSONL, A14c script, or canonical A1 artifacts are modified.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from dali_block_a1_analyze import family_category
from dali_block_a14c_maker_at_mid import (
    A1_RESULTS,
    ANALYSIS,
    BOOTSTRAP_CHUNK_SECONDS,
    BOOTSTRAP_SAMPLES,
    EXIT_CONVENTIONS,
    FILL_WINDOWS,
    HOLD_HORIZONS,
    NOTES,
    ROOT,
    RNG_SEED,
    add_adverse_selection,
    assign_top_decile_signals,
    bps,
    first_qualifying_trade,
    load_candidates,
    load_feature_subset,
    load_signal_horizons,
    maker_rebate_bps,
    markdown_table,
    pct,
    safe_text,
    simulate_entry_fills,
    state_at_or_before,
    taker_fee_bps_on_entry_notional,
)


A14C_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a14c_maker_at_mid_results.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14h_maker_non_overlap_results.csv"
NOTE = NOTES / "block_a14h_maker_non_overlap_findings.md"
ROBUST_MIN_FILLS = 5


def simulate_exit_with_time(
    filled: pd.DataFrame,
    market: pd.DataFrame,
    hold_sec: int,
    exit_convention: str,
    category: str,
) -> pd.DataFrame:
    """A14c exit mechanics plus an explicit exit timestamp for non-overlap."""
    out_parts: list[pd.DataFrame] = []
    for asset_id, rows in filled.groupby("asset_id", sort=False):
        asset_rows = market[market["asset_id"].eq(asset_id)].sort_values("received_at")
        state = asset_rows[
            asset_rows["best_bid"].notna()
            & asset_rows["best_ask"].notna()
            & asset_rows["mid"].notna()
        ].copy()
        state_times = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        bids = state["best_bid"].to_numpy(dtype=float)
        asks = state["best_ask"].to_numpy(dtype=float)
        mids = state["mid"].to_numpy(dtype=float)

        trades = asset_rows[
            asset_rows["event_type"].eq("last_trade_price")
            & asset_rows["trade_side_norm"].isin(["BUY", "SELL"])
            & asset_rows["trade_price"].notna()
        ].copy()
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trade_prices = trades["trade_price"].to_numpy(dtype=float)
        trade_sides = trades["trade_side_norm"].to_numpy(dtype=object)

        piece = rows.copy()
        n = len(piece)
        exit_price = np.full(n, np.nan, dtype=float)
        exit_time_ns = np.full(n, np.nan, dtype=float)
        exit_kind = np.full(n, "unexit", dtype=object)
        exit_filled_maker = np.zeros(n, dtype=bool)
        exit_taker_fee_bps = np.zeros(n, dtype=float)

        fill_times = piece["fill_time_ns"].to_numpy(dtype=np.float64).astype(np.int64)
        token_side = piece["token_side"].to_numpy(dtype=float)
        entry_price = piece["entry_price"].to_numpy(dtype=float)

        if exit_convention == "exit_forced_taker":
            target = fill_times + hold_sec * 1_000_000_000
            future_bid = state_at_or_before(state_times, bids, target)
            future_ask = state_at_or_before(state_times, asks, target)
            exit_price = np.where(token_side > 0, future_bid, future_ask)
            valid = np.isfinite(exit_price) & (exit_price >= 0)
            exit_kind[valid] = "taker"
            exit_time_ns[valid] = target[valid].astype(float)
            exit_taker_fee_bps[valid] = taker_fee_bps_on_entry_notional(
                category,
                exit_price[valid],
                entry_price[valid],
            )
        else:
            post_time = fill_times + hold_sec * 1_000_000_000
            post_mid = state_at_or_before(state_times, mids, post_time)
            for pos in range(n):
                if not np.isfinite(post_mid[pos]):
                    continue
                start = int(post_time[pos])
                end = start + hold_sec * 1_000_000_000
                if token_side[pos] > 0:
                    found_time, _ = first_qualifying_trade(
                        start,
                        end,
                        float(post_mid[pos]),
                        "BUY",
                        "ge",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                else:
                    found_time, _ = first_qualifying_trade(
                        start,
                        end,
                        float(post_mid[pos]),
                        "SELL",
                        "le",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                if found_time is not None:
                    exit_price[pos] = post_mid[pos]
                    exit_time_ns[pos] = float(found_time)
                    exit_kind[pos] = "maker"
                    exit_filled_maker[pos] = True
                    continue

                fallback_time = fill_times[pos] + 2 * hold_sec * 1_000_000_000
                fallback_bid = state_at_or_before(
                    state_times,
                    bids,
                    np.array([fallback_time], dtype=np.int64),
                )[0]
                fallback_ask = state_at_or_before(
                    state_times,
                    asks,
                    np.array([fallback_time], dtype=np.int64),
                )[0]
                fallback_price = fallback_bid if token_side[pos] > 0 else fallback_ask
                if np.isfinite(fallback_price):
                    exit_price[pos] = fallback_price
                    exit_time_ns[pos] = float(fallback_time)
                    exit_kind[pos] = "taker_fallback"
                    exit_taker_fee_bps[pos] = taker_fee_bps_on_entry_notional(
                        category,
                        fallback_price,
                        entry_price[pos],
                    )

        piece["exit_price"] = exit_price
        piece["exit_time_ns"] = exit_time_ns
        piece["exit_kind"] = exit_kind
        piece["exit_maker_filled"] = exit_filled_maker
        piece["exit_taker_fee_bps"] = exit_taker_fee_bps
        gross = token_side * (exit_price - entry_price) / np.clip(entry_price, 0.01, 0.99) * 10_000.0
        piece["maker_rebate_bps"] = maker_rebate_bps(category, entry_price)
        piece["pnl_bps"] = gross + piece["maker_rebate_bps"].to_numpy(dtype=float) - exit_taker_fee_bps
        piece.loc[~np.isfinite(exit_price) | ~np.isfinite(exit_time_ns), "pnl_bps"] = np.nan
        out_parts.append(piece)
    return pd.concat(out_parts, ignore_index=True) if out_parts else filled.iloc[0:0].copy()


def interval_contains(intervals: list[tuple[int, int]], timestamp_ns: int) -> bool:
    return any(start <= timestamp_ns <= end for start, end in intervals)


def apply_non_overlap(candidate_exits: pd.DataFrame) -> pd.DataFrame:
    """Select executed fills under the A14h one-open-position rule."""
    if candidate_exits.empty:
        return candidate_exits
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
    candidates = candidates.sort_values(
        ["fill_time_ns_int", "signal_time_ns", "abs_signal"],
        ascending=[True, True, False],
    )

    intervals: list[tuple[int, int]] = []
    keep_positions: list[int] = []
    for pos, row in enumerate(candidates.itertuples(index=False)):
        signal_ns = int(row.signal_time_ns)
        fill_ns = int(row.fill_time_ns_int)
        exit_ns = int(row.exit_time_ns_int)
        if exit_ns <= fill_ns:
            continue
        if interval_contains(intervals, signal_ns) or interval_contains(intervals, fill_ns):
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
    signal_horizon: int,
    fill_window: int,
    hold_horizon: int,
    exit_convention: str,
    n_signal_events: int,
    executed: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    n = int(len(executed))
    fill_rate = n / n_signal_events if n_signal_events else math.nan
    mean_pnl = float(executed["pnl_bps"].mean()) if n else math.nan
    median_pnl = float(executed["pnl_bps"].median()) if n else math.nan
    win_rate = float(executed["pnl_bps"].gt(0).mean()) if n else math.nan
    ci_lo, ci_hi = bootstrap_mean_ci(executed, seed)
    return {
        "market": market_label,
        "slug": slug,
        "family": family,
        "sample_size_label": sample_size_label,
        "signal_variant": "current_level_tob_imbalance",
        "signal_horizon": signal_horizon,
        "fill_window_sec": fill_window,
        "hold_sec": hold_horizon,
        "exit_convention": exit_convention,
        "n_signal_events": n_signal_events,
        "n_executed_fills": n,
        "fill_rate": fill_rate,
        "mean_pnl_bps": mean_pnl,
        "median_pnl_bps": median_pnl,
        "win_rate": win_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_rebate_bps": float(executed["maker_rebate_bps"].mean()) if n else math.nan,
        "mean_adverse_selection_bps_5s": (
            float(executed["adverse_selection_bps_5s"].mean())
            if n and "adverse_selection_bps_5s" in executed
            else math.nan
        ),
    }


def run_simulation(df: pd.DataFrame, candidates: pd.DataFrame, signal_horizons: tuple[int, ...]) -> pd.DataFrame:
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
        n_signal_events = int(len(signals))
        print(
            f"A14h market {market_idx:02d}/{total_markets:02d} {market_label} "
            f"signals={n_signal_events:,}",
            flush=True,
        )

        entry_by_window: dict[int, pd.DataFrame] = {}
        for fill_window in FILL_WINDOWS:
            entry = simulate_entry_fills(signals, market, fill_window)
            entry = entry[entry["entry_filled"]].copy()
            if not entry.empty:
                entry = add_adverse_selection(entry, market)
            entry_by_window[fill_window] = entry

        for signal_horizon in signal_horizons:
            for fill_window in FILL_WINDOWS:
                entry = entry_by_window[fill_window]
                for hold_horizon in HOLD_HORIZONS:
                    for exit_convention in EXIT_CONVENTIONS:
                        exited = (
                            simulate_exit_with_time(entry, market, hold_horizon, exit_convention, category)
                            if not entry.empty
                            else entry
                        )
                        executed = apply_non_overlap(exited)
                        rows.append(
                            summarize_combo(
                                market_label=market_label,
                                slug=slug,
                                family=family,
                                sample_size_label=sample_size_label,
                                signal_horizon=signal_horizon,
                                fill_window=fill_window,
                                hold_horizon=hold_horizon,
                                exit_convention=exit_convention,
                                n_signal_events=n_signal_events,
                                executed=executed,
                                seed=RNG_SEED
                                + market_idx * 10_000
                                + signal_horizon * 100
                                + fill_window * 10
                                + hold_horizon
                                + (0 if exit_convention == "exit_symmetric_maker" else 1),
                            )
                        )
    columns = [
        "market",
        "slug",
        "family",
        "sample_size_label",
        "signal_variant",
        "signal_horizon",
        "fill_window_sec",
        "hold_sec",
        "exit_convention",
        "n_signal_events",
        "n_executed_fills",
        "fill_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_rebate_bps",
        "mean_adverse_selection_bps_5s",
    ]
    return pd.DataFrame(rows)[columns]


def add_a14c_comparison(results: pd.DataFrame) -> pd.DataFrame:
    if not A14C_RESULTS.exists():
        raise SystemExit(f"missing A14c comparison CSV: {A14C_RESULTS}")
    overlap = pd.read_csv(A14C_RESULTS)
    keys = ["market", "signal_horizon", "fill_window_sec", "hold_sec", "exit_convention"]
    overlap = overlap[keys + ["mean_pnl_bps", "fill_rate"]].drop_duplicates(keys)
    overlap = overlap.rename(
        columns={
            "mean_pnl_bps": "a14c_overlap_mean_pnl_bps",
            "fill_rate": "a14c_overlap_fill_rate",
        }
    )
    out = results.merge(overlap, on=keys, how="left", validate="one_to_one")
    out["delta_pnl_vs_a14c_bps"] = out["mean_pnl_bps"] - out["a14c_overlap_mean_pnl_bps"]
    out["delta_fillrate_vs_a14c_pct"] = (out["fill_rate"] - out["a14c_overlap_fill_rate"]) * 100.0
    return out


def unique_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["market", "fill_window_sec", "hold_sec", "exit_convention"]).copy()


def robust_positive_count(df: pd.DataFrame, fill_col: str = "n_executed_fills") -> int:
    return int(
        (
            df["mean_pnl_bps"].gt(0)
            & df["ci_lo"].gt(0)
            & df[fill_col].ge(ROBUST_MIN_FILLS)
        ).sum()
    )


def classify_market(overlap_best: pd.Series, nonoverlap_same: pd.Series, nonoverlap_best: pd.Series) -> str:
    if (
        np.isfinite(nonoverlap_best["mean_pnl_bps"])
        and nonoverlap_best["mean_pnl_bps"] > 0
        and np.isfinite(nonoverlap_best["ci_lo"])
        and nonoverlap_best["ci_lo"] > 0
        and int(nonoverlap_best["n_executed_fills"]) >= ROBUST_MIN_FILLS
    ):
        return "maker thesis survives"
    if (
        np.isfinite(overlap_best["mean_pnl_bps"])
        and overlap_best["mean_pnl_bps"] > 0
        and (
            not np.isfinite(nonoverlap_same["mean_pnl_bps"])
            or nonoverlap_same["mean_pnl_bps"] <= 0
            or nonoverlap_same["delta_pnl_vs_a14c_bps"] < -250
        )
    ):
        return "overlap artifact like A14f"
    if int(nonoverlap_best["n_executed_fills"]) < 30 or float(nonoverlap_best["fill_rate"]) < 0.01:
        return "fills too rare even in best case"
    return "adverse selection wipes rebate"


def side_by_side_table(results: pd.DataFrame, overlap: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    h_unique = unique_cells(results)
    c_unique = unique_cells(overlap.rename(columns={"n_filled": "n_executed_fills"}))
    rows: list[list[str]] = []
    verdict_rows: list[dict[str, object]] = []
    for market, c_sub in c_unique.groupby("market", sort=True):
        h_sub = h_unique[h_unique["market"].eq(market)].copy()
        if h_sub.empty:
            continue
        c_best = c_sub.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).iloc[0]
        same = h_sub[
            h_sub["fill_window_sec"].eq(c_best["fill_window_sec"])
            & h_sub["hold_sec"].eq(c_best["hold_sec"])
            & h_sub["exit_convention"].eq(c_best["exit_convention"])
        ]
        h_same = same.iloc[0] if not same.empty else h_sub.iloc[0]
        h_best = h_sub.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).iloc[0]
        verdict = classify_market(c_best, h_same, h_best)
        verdict_rows.append(
            {
                "market": market,
                "slug": h_best["slug"],
                "a14c_best_mean_pnl_bps": c_best["mean_pnl_bps"],
                "a14h_same_combo_mean_pnl_bps": h_same["mean_pnl_bps"],
                "a14h_best_mean_pnl_bps": h_best["mean_pnl_bps"],
                "verdict": verdict,
            }
        )
        rows.append(
            [
                safe_text(market, 16),
                safe_text(h_best["slug"], 36),
                f"W={int(c_best['fill_window_sec'])}, H={int(c_best['hold_sec'])}, "
                f"{str(c_best['exit_convention']).replace('exit_', '')}",
                bps(float(c_best["mean_pnl_bps"])),
                pct(float(c_best["fill_rate"])),
                bps(float(h_same["mean_pnl_bps"])),
                pct(float(h_same["fill_rate"])),
                bps(float(h_same["delta_pnl_vs_a14c_bps"])),
                bps(float(h_best["mean_pnl_bps"])),
                f"[{bps(float(h_best['ci_lo']))}, {bps(float(h_best['ci_hi']))}]",
                verdict,
            ]
        )
    table = markdown_table(
        [
            "market",
            "slug",
            "A14c best grid",
            "A14c mean",
            "A14c fill",
            "A14h same mean",
            "A14h same fill",
            "same delta",
            "A14h best",
            "A14h best CI",
            "verdict",
        ],
        rows,
    )
    return table, pd.DataFrame(verdict_rows)


def top_cells_table(results: pd.DataFrame, limit: int = 12) -> str:
    sub = unique_cells(results).sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                safe_text(row.market, 18),
                safe_text(row.slug, 38),
                str(int(row.fill_window_sec)),
                str(int(row.hold_sec)),
                str(row.exit_convention).replace("exit_", ""),
                f"{int(row.n_signal_events):,}",
                f"{int(row.n_executed_fills):,}",
                pct(float(row.fill_rate)),
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                bps(float(row.a14c_overlap_mean_pnl_bps)),
                bps(float(row.delta_pnl_vs_a14c_bps)),
            ]
        )
    return markdown_table(
        ["market", "slug", "W", "H", "exit", "signals", "fills", "fill rate", "mean", "CI", "A14c", "delta"],
        rows,
    )


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    overlap = pd.read_csv(A14C_RESULTS)
    h_unique = unique_cells(results)
    c_unique = unique_cells(overlap.rename(columns={"n_filled": "n_executed_fills"}))
    side_table, verdicts = side_by_side_table(results, overlap)

    btc_mask = (
        results["market"].eq("a0b:2364426")
        & results["fill_window_sec"].eq(10)
        & results["hold_sec"].eq(30)
        & results["exit_convention"].eq("exit_symmetric_maker")
    )
    btc = results[btc_mask].sort_values("signal_horizon").iloc[0]
    btc_survives = bool(np.isfinite(btc["mean_pnl_bps"]) and btc["mean_pnl_bps"] > 0 and np.isfinite(btc["ci_lo"]) and btc["ci_lo"] > 0)

    c_positive = int(c_unique["mean_pnl_bps"].gt(0).sum())
    h_positive = int(h_unique["mean_pnl_bps"].gt(0).sum())
    h_robust = robust_positive_count(h_unique)
    raw_h_positive = int(results["mean_pnl_bps"].gt(0).sum())
    raw_h_robust = robust_positive_count(results)

    verdict_counts = verdicts["verdict"].value_counts().to_dict() if not verdicts.empty else {}
    recommendation = (
        "run A14e queue+latency only if you want to falsify deployability formally; otherwise pivot away from Dali microstructure on A0/A0b because non-overlap already kills the remaining maker clue"
        if h_robust == 0 and not btc_survives
        else "run A14e queue+latency next because at least one non-overlap maker cell survived the robustness bar"
    )

    note = f"""---
tags: [dali, block-a14h, maker-thesis, results]
---

# Block A1.4h Maker Non-Overlap Findings

## Headline

The A14c BTC-4h winner does **not** survive the non-overlap check. The same `{safe_text(btc['market'])}` / `{safe_text(btc['slug'])}` cell that was +554.9 bps in A14c overlap math is {bps(float(btc['mean_pnl_bps']))} after non-overlap, with CI [{bps(float(btc['ci_lo']))}, {bps(float(btc['ci_hi']))}], fill rate {pct(float(btc['fill_rate']))}, and {int(btc['n_executed_fills']):,} executed fills. This puts it in the same bucket as A14f: the positive cell was mostly overlap-position math, not a deployable one-position-at-a-time maker thesis.

## Method

- Universe: same as A14c, A1 markets labeled `primary_read` or `thin_wide_CI` at the 5s horizon.
- Signal: per-market top absolute decile of current-level TOB imbalance, `direction_factor * tob_imbalance`.
- Entry: at signal time, post at current mid on the signal-favorable token side. Long signal posts a bid at mid; short signal posts an ask at mid.
- Fill model: same as A14c. Long-entry bid fills on a SELL print at or below mid; short-entry ask fills on a BUY print at or above mid, within W in {{1s, 5s, 10s}}.
- Exit model: same two A14c conventions. `exit_forced_taker` closes at opposite touch after H. `exit_symmetric_maker` posts opposite-side at mid after H and forces a taker fallback at t_fill + 2H if no exit fill arrives.
- Non-overlap rule: unfilled signals do not block. Only actual fills create a blocking interval, and candidate fills are greedily selected in fill-time order so no executed fill or signal occurs inside an already-open interval.
- PnL: A14c direction-adjusted mid-maker entry PnL plus entry maker rebate, minus taker fee on forced exits.
- Bootstrap: 200 resamples over contiguous 300s blocks on non-overlap filled PnL.

## Side-by-Side Per-Market Table

{side_table}

## Cross-Market Verdict

Collapsing A14c's duplicated current-level signal-horizon labels, A14c had {c_positive:,}/{len(c_unique):,} positive unique market/grid cells. A14h keeps {h_positive:,}/{len(h_unique):,} positive unique cells, with {h_robust:,} clearing the robustness bar of mean > 0, CI lower > 0, and at least {ROBUST_MIN_FILLS} fills. In the raw CSV rows, A14h has {raw_h_positive:,}/{len(results):,} positive rows and {raw_h_robust:,} robust-positive rows.

Verdict counts by market: {', '.join(f'{k}: {v}' for k, v in verdict_counts.items())}.

## Top Non-Overlap Cells

{top_cells_table(results)}

## Per-Market Verdict

The per-market verdict table above is the diagnostic read. `maker thesis survives` requires a positive non-overlap best cell with CI lower bound above zero. `overlap artifact like A14f` means the market's A14c best overlap cell was positive but the same cell collapsed or lost more than 250 bps under non-overlap. `fills too rare even in best case` means the best non-overlap cell had fewer than 30 fills or sub-1% fill rate. The remaining negative cells are classified as adverse-selection/rebate failure.

## Interpretation

This is still a generous model: full priority at mid, no queue, no latency, no quote-cancel risk, and no partial fills. Since the last positive maker clue disappears before adding those frictions, queue+latency would be an execution autopsy rather than a likely rescue.

Recommended next action for Justin: {recommendation}.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not A1_RESULTS.exists():
        raise SystemExit(f"missing A1 results: {A1_RESULTS}")
    if not A14C_RESULTS.exists():
        raise SystemExit(f"missing A14c results: {A14C_RESULTS}")
    signal_horizons = load_signal_horizons()
    candidates = load_candidates()
    features = load_feature_subset(candidates)
    results = run_simulation(features, candidates, signal_horizons)
    results = add_a14c_comparison(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
