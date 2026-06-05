"""Block A1.4 executable taker round-trip QA.

This sidecar answers a narrower question than the A1 mid-return cost overlay:
for the strongest pre-cost A1 candidates, does the OFI signal survive paying
the executable touch on entry and exit?

It does not modify A1 artifacts or raw JSONL captures.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
A1_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a1_results.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14_executable_taker_results.csv"
NOTE = NOTES / "block_a14_executable_taker_findings.md"

HORIZONS = (5, 10)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260528


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fee_amount(category: str, price: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Polymarket taker fee in dollars per share at price p."""
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = np.clip(price, 0.0, 1.0)
    return params["fee_rate"] * p * (1.0 - p)


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    results["horizon_sec"] = pd.to_numeric(results["horizon_sec"], errors="coerce")
    candidates = results[
        results["horizon_sec"].eq(5)
        & results["verdict"].eq("signal_present_pre_cost")
        & results["sample_size_label"].eq("primary_read")
    ].copy()
    candidates = candidates[
        [
            "run_id",
            "market_id",
            "family",
            "n_classifiable",
            "mean_depth_at_touch",
        ]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no A1 5s primary-read pre-cost candidates found")
    return candidates, results


def load_feature_subset(candidates: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("candidates", candidates[["run_id", "market_id"]])
    query = f"""
        SELECT
            f.run_id,
            f.received_at,
            f.asset_id,
            f.market_id,
            f.family,
            f.slug,
            f.question,
            f.outcome_index,
            f.is_book_state_complete,
            f.best_bid,
            f.best_ask,
            f.best_bid_size,
            f.best_ask_size,
            f.mid,
            f.ofi_combined_event
        FROM read_parquet('{FEATURES}') AS f
        INNER JOIN candidates AS c
            ON f.run_id = c.run_id
           AND f.market_id = c.market_id
        ORDER BY f.run_id, f.market_id, f.asset_id, f.received_at
    """
    df = con.execute(query).df()
    con.close()
    if df.empty:
        raise SystemExit("candidate feature subset is empty")
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    for col in (
        "outcome_index",
        "best_bid",
        "best_ask",
        "best_bid_size",
        "best_ask_size",
        "mid",
        "ofi_combined_event",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    df["family"] = df["family"].fillna("").astype(str)
    df["market_id"] = df["market_id"].fillna("").astype(str)
    df["asset_id"] = df["asset_id"].fillna("").astype(str)
    df["slug"] = df["slug"].fillna("").astype(str)
    df["question"] = df["question"].fillna("").astype(str)
    return df


def future_touch(group: pd.DataFrame, horizon_sec: int) -> tuple[np.ndarray, np.ndarray]:
    times = group["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bid = group["best_bid"].to_numpy(dtype=float)
    ask = group["best_ask"].to_numpy(dtype=float)
    target = times + horizon_sec * 1_000_000_000
    idx = np.searchsorted(times, target, side="right") - 1
    future_bid = np.full(len(group), np.nan, dtype=float)
    future_ask = np.full(len(group), np.nan, dtype=float)
    if len(group) == 0:
        return future_bid, future_ask
    valid = (target <= times[-1]) & (idx >= 0)
    future_bid[valid] = bid[idx[valid]]
    future_ask[valid] = ask[idx[valid]]
    return future_bid, future_ask


def add_signals(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    depth = candidates.set_index(["run_id", "market_id"])["mean_depth_at_touch"].to_dict()
    pieces: list[pd.DataFrame] = []
    for (run_id, market_id, asset_id), group in df.groupby(
        ["run_id", "market_id", "asset_id"],
        sort=False,
    ):
        g = group.sort_values("received_at").copy()
        g["direction_factor"] = np.where(
            g["outcome_index"].fillna(0).astype(int).eq(0),
            1.0,
            -1.0,
        )
        mean_depth = float(depth.get((run_id, market_id), math.nan))
        g["market_mean_depth"] = mean_depth
        g = g.set_index("received_at", drop=False)
        for horizon in HORIZONS:
            roll = g["ofi_combined_event"].fillna(0.0).rolling(f"{horizon}s").sum()
            g[f"ofi_token_{horizon}s"] = roll.to_numpy()
            g[f"ofi_market_{horizon}s"] = (
                g["direction_factor"].to_numpy(dtype=float) * roll.to_numpy(dtype=float)
            )
            g[f"ofi_scaled_{horizon}s"] = g[f"ofi_market_{horizon}s"] / mean_depth
            future_bid, future_ask = future_touch(g, horizon)
            g[f"future_bid_{horizon}s"] = future_bid
            g[f"future_ask_{horizon}s"] = future_ask
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def bootstrap_mean_ci(events: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = events[["received_at", "pnl_bps"]].dropna().copy()
    if len(clean) < 5:
        return math.nan, math.nan
    elapsed = (clean["received_at"] - clean["received_at"].min()).dt.total_seconds()
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


def simulate_market_horizon(
    sub: pd.DataFrame,
    horizon: int,
    category: str,
) -> tuple[pd.DataFrame, float]:
    sig_col = f"ofi_scaled_{horizon}s"
    token_col = f"ofi_token_{horizon}s"
    base = sub[
        sub["is_book_state_complete"]
        & np.isfinite(sub[sig_col])
        & sub[sig_col].ne(0.0)
        & sub["market_mean_depth"].gt(0)
    ].copy()
    if base.empty:
        return base, math.nan
    threshold = float(base[sig_col].abs().quantile(0.90))
    top = base[base[sig_col].abs().ge(threshold)].copy()
    if top.empty:
        return top, threshold

    token_side = np.sign(top[token_col].to_numpy(dtype=float))
    entry_long = top["best_ask"].to_numpy(dtype=float)
    exit_long = top[f"future_bid_{horizon}s"].to_numpy(dtype=float)
    entry_short = top["best_bid"].to_numpy(dtype=float)
    exit_short = top[f"future_ask_{horizon}s"].to_numpy(dtype=float)

    entry_price = np.where(token_side > 0, entry_long, entry_short)
    exit_price = np.where(token_side > 0, exit_long, exit_short)
    quote_ok = (
        np.isfinite(entry_price)
        & np.isfinite(exit_price)
        & (entry_price > 0)
        & (exit_price >= 0)
        & np.isfinite(token_side)
        & (token_side != 0)
    )

    top["signal_side"] = np.where(token_side > 0, "long_asset", "short_asset")
    top["entry_price"] = entry_price
    top["exit_price"] = exit_price
    top["is_fillable"] = quote_ok
    pnl = np.full(len(top), np.nan, dtype=float)
    if quote_ok.any():
        fee_entry = fee_amount(category, entry_price[quote_ok])
        fee_exit = fee_amount(category, exit_price[quote_ok])
        gross = np.where(
            token_side[quote_ok] > 0,
            exit_price[quote_ok] - entry_price[quote_ok],
            entry_price[quote_ok] - exit_price[quote_ok],
        )
        pnl[quote_ok] = (gross - fee_entry - fee_exit) / entry_price[quote_ok] * 10_000.0
    top["pnl_bps"] = pnl
    return top, threshold


def summarize(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    a1_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    overlay = (
        a1_results[
            [
                "run_id",
                "market_id",
                "horizon_sec",
                "edge_after_cost_bps",
            ]
        ]
        .drop_duplicates(["run_id", "market_id", "horizon_sec"])
        .copy()
    )
    overlay["horizon_sec"] = pd.to_numeric(overlay["horizon_sec"], errors="coerce")
    overlay_map = {
        (row.run_id, row.market_id, int(row.horizon_sec)): float(row.edge_after_cost_bps)
        for row in overlay.itertuples(index=False)
        if np.isfinite(row.horizon_sec)
    }

    rows: list[dict[str, object]] = []
    meta: dict[str, dict[str, str]] = {}
    for idx, cand in candidates.reset_index(drop=True).iterrows():
        run_id = str(cand["run_id"])
        market_id = str(cand["market_id"])
        family = str(cand["family"])
        category = family_category(family)
        sub = df[df["run_id"].eq(run_id) & df["market_id"].eq(market_id)].copy()
        slug = str(sub["slug"].replace("", np.nan).dropna().iloc[0]) if sub["slug"].astype(bool).any() else market_id
        question = (
            str(sub["question"].replace("", np.nan).dropna().iloc[0])
            if sub["question"].astype(bool).any()
            else ""
        )
        market_label = f"{run_id}:{market_id}"
        meta[market_label] = {"slug": slug, "question": question, "family": family}
        for horizon in HORIZONS:
            top, _threshold = simulate_market_horizon(sub, horizon, category)
            n_events = int(len(top))
            n_unfillable = int((~top["is_fillable"]).sum()) if n_events else 0
            fillable = top[top.get("is_fillable", False)].copy() if n_events else pd.DataFrame()
            fillable_count = int(len(fillable))
            fillable_rate = fillable_count / n_events if n_events else math.nan
            mean_pnl = float(fillable["pnl_bps"].mean()) if fillable_count else math.nan
            median_pnl = float(fillable["pnl_bps"].median()) if fillable_count else math.nan
            win_rate = float(fillable["pnl_bps"].gt(0).mean()) if fillable_count else math.nan
            ci_lo, ci_hi = bootstrap_mean_ci(fillable, RNG_SEED + idx * 100 + horizon)
            a1_overlay = overlay_map.get((run_id, market_id, horizon), math.nan)
            gap = mean_pnl - a1_overlay if np.isfinite(mean_pnl) and np.isfinite(a1_overlay) else math.nan
            rows.append(
                {
                    "market": market_label,
                    "horizon": horizon,
                    "n_events": n_events,
                    "n_unfillable": n_unfillable,
                    "fillable_rate": fillable_rate,
                    "mean_pnl_bps": mean_pnl,
                    "median_pnl_bps": median_pnl,
                    "win_rate": win_rate,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "a1_overlay_edge_bps": a1_overlay,
                    "gap_bps": gap,
                }
            )
    out = pd.DataFrame(rows)
    preferred = [
        "market",
        "horizon",
        "n_events",
        "n_unfillable",
        "fillable_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "a1_overlay_edge_bps",
        "gap_bps",
    ]
    return out[preferred], meta


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_note(results: pd.DataFrame, meta: dict[str, dict[str, str]]) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    survive = results[results["mean_pnl_bps"].gt(0)].copy()
    total_cells = int(len(results))
    survive_cells = int(len(survive))
    five = results[results["horizon"].eq(5)]
    five_gap = float(five["gap_bps"].mean()) if five["gap_bps"].notna().any() else math.nan
    five_survive = int(five["mean_pnl_bps"].gt(0).sum())
    five_total = int(len(five))

    rows: list[list[str]] = []
    for row in results.sort_values(["horizon", "mean_pnl_bps"], ascending=[True, False]).itertuples(index=False):
        market_meta = meta.get(row.market, {})
        slug = str(market_meta.get("slug", row.market))[:48]
        verdict = "survives executable cost" if np.isfinite(row.mean_pnl_bps) and row.mean_pnl_bps > 0 else "wiped by spread"
        rows.append(
            [
                str(row.market),
                slug.replace("|", "/"),
                str(int(row.horizon)),
                f"{int(row.n_events):,}",
                f"{int(row.n_unfillable):,}",
                pct(float(row.fillable_rate)),
                bps(float(row.mean_pnl_bps)),
                bps(float(row.median_pnl_bps)),
                pct(float(row.win_rate)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                bps(float(row.a1_overlay_edge_bps)),
                bps(float(row.gap_bps)),
                verdict,
            ]
        )

    verdict_lines: list[str] = []
    for market, sub in results.groupby("market", sort=True):
        market_meta = meta.get(market, {})
        slug = str(market_meta.get("slug", market))
        h5 = sub[sub["horizon"].eq(5)]
        h10 = sub[sub["horizon"].eq(10)]
        h5_mean = float(h5["mean_pnl_bps"].iloc[0]) if not h5.empty else math.nan
        h10_mean = float(h10["mean_pnl_bps"].iloc[0]) if not h10.empty else math.nan
        status = "survives executable cost" if np.nanmax([h5_mean, h10_mean]) > 0 else "wiped by spread"
        verdict_lines.append(
            f"- `{market}` ({slug}): {status}; 5s {bps(h5_mean)}, 10s {bps(h10_mean)}."
        )

    note = f"""---
tags: [dali, block-a14, executable-cost, results]
---

# Block A1.4 Executable Taker Findings

## Headline

A1.4 replaces the A1 mid-return cost overlay with an executable touch-to-touch round trip on the six 5s primary-read pre-cost candidates. {survive_cells} of {total_cells} market-horizon cells have positive mean executable PnL after crossing the spread on both entry and exit. At the original A1 5s horizon, {five_survive} of {five_total} candidates survive; the mean gap versus A1's 5s overlay is {bps(five_gap)}, where positive means executable round-trip PnL was better than the A1 overlay. In this run, the top-decile mid-mid signal is real as a descriptive pattern but is not close to surviving a simple taker round trip.

## Method

- Candidate universe: markets with `verdict = signal_present_pre_cost` and `sample_size_label = primary_read` in `block_a1_results.csv` at the 5s horizon.
- Signal: top decile by absolute `OFI_scaled` per market and horizon, where `OFI_scaled = direction_factor * rolling_OFI / mean_depth_at_touch`.
- Horizons: 5s and 10s. A1 has a cost-overlay comparison at 5s; 10s is reported as executable-only because A1 did not produce a 10s cost-overlay row.
- Entry/exit: after the YES/NO direction flip, the action is converted back to the current asset's token side. A long token signal pays `best_ask` at entry and receives future `best_bid`; a short token signal receives `best_bid` at entry and pays future `best_ask`.
- Fees: Polymarket taker fee is applied at entry and exit using the A1 `FEE_BY_CATEGORY` table as dollars per share, then normalized by entry price.
- Fill model: full size at touch, no partial fills, no queue model, and no latency layer.
- Exit quote: last observed book state at or before `t + horizon`, matching A1's forward-state convention.
- Confidence intervals: 200-sample block bootstrap of mean PnL using contiguous 300s clock-time blocks.

## Gap Table

{markdown_table(
        [
            "market",
            "slug",
            "h",
            "events",
            "unfillable",
            "fillable",
            "mean pnl",
            "median pnl",
            "win",
            "mean CI",
            "A1 overlay",
            "gap",
            "verdict",
        ],
        rows,
    )}

## Per-Market Verdicts

{chr(10).join(verdict_lines)}

## Interpretation

This is the executable-cost QA that A1's overlay could not provide. A positive row means the observed mid-mid OFI move was large enough to overcome both entry and exit spread crossing plus taker fees under the simplified touch-fill assumption. A negative row means the apparent mid-return alpha was wiped by executable round-trip cost in this capture window.

The 10s rows should be read as an A1.4 extension, not a direct A1 overlay audit, because the original A1 result table only contains 1s, 5s, 30s, and 300s horizons.

Recommended next action for Justin: do not advance the A1 top-decile signal as a taker edge; make A2 a longer signal-characterization capture with an explicit tight-spread executable-cost screen.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates, a1_results = load_candidates()
    features = load_feature_subset(candidates)
    features = add_signals(features, candidates)
    results, meta = summarize(features, candidates, a1_results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results, meta)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
