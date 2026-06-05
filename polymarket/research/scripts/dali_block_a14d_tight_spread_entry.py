"""Block A1.4d tight-spread-conditional taker entry.

This sidecar tests whether the A1.3/A14 TOB imbalance signal becomes
executable when taker entry is allowed only during tight-spread states.
It does not modify A1/A13/A14 artifacts or raw captures.
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
A13_CONTROL = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_control_buckets.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14d_tight_spread_results.csv"
NOTE = NOTES / "block_a14d_tight_spread_findings.md"

HORIZONS = (5, 30, 300)
SPREAD_THRESHOLDS: tuple[tuple[str, float | None], ...] = (
    ("50", 50.0),
    ("100", 100.0),
    ("200", 200.0),
    ("500", 500.0),
    ("1000", 1000.0),
    ("no_filter", None),
)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260528
MIN_INTERPRETABLE_EVENTS = 30


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


def load_candidates() -> pd.DataFrame:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    results["horizon_sec"] = pd.to_numeric(results["horizon_sec"], errors="coerce")
    candidates = results[results["sample_size_label"].eq("primary_read")].copy()
    candidates = candidates.sort_values("horizon_sec")
    candidates = candidates[
        ["run_id", "market_id", "family", "n_classifiable"]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no primary_read candidates found in block_a1_results.csv")
    return candidates


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
            f.spread,
            f.mid,
            f.tob_imbalance
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
        "spread",
        "mid",
        "tob_imbalance",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("run_id", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    quoted_spread = df["best_ask"] - df["best_bid"]
    spread = df["spread"].where(df["spread"].notna(), quoted_spread)
    df["spread_bps"] = np.where(df["mid"].gt(0), spread / df["mid"] * 10_000.0, np.nan)
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    return df


def future_touch(group: pd.DataFrame, horizon_sec: int) -> tuple[np.ndarray, np.ndarray]:
    times = group["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bid = group["best_bid"].to_numpy(dtype=float)
    ask = group["best_ask"].to_numpy(dtype=float)
    future_bid = np.full(len(group), np.nan, dtype=float)
    future_ask = np.full(len(group), np.nan, dtype=float)
    if len(group) == 0:
        return future_bid, future_ask
    target = times + horizon_sec * 1_000_000_000
    idx = np.searchsorted(times, target, side="right") - 1
    valid = (target <= times[-1]) & (idx >= 0)
    future_bid[valid] = bid[idx[valid]]
    future_ask[valid] = ask[idx[valid]]
    return future_bid, future_ask


def add_market_signals(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "market_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        for horizon in HORIZONS:
            future_bid, future_ask = future_touch(g, horizon)
            g[f"future_bid_{horizon}s"] = future_bid
            g[f"future_ask_{horizon}s"] = future_ask
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    df["is_top_tob_decile"] = False
    base_valid = (
        df["is_book_state_complete"]
        & df["tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
        & df["tob_imbalance_level"].ne(0.0)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["spread_bps"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].ge(df["best_bid"])
    )
    for _, market in df[base_valid].groupby(["run_id", "market_id"], sort=False):
        if len(market) < 10:
            continue
        ranked = market["tob_imbalance_level"].abs().rank(method="first")
        decile = pd.qcut(ranked, 10, labels=False) + 1
        top_idx = decile[decile.eq(10)].index
        df.loc[top_idx, "is_top_tob_decile"] = True
    return df


def bootstrap_mean_ci(events: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = events[["received_at", "pnl_bps"]].dropna().copy()
    if len(clean) < 5:
        return math.nan, math.nan
    elapsed = (clean["received_at"] - clean["received_at"].min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    if len(np.unique(block_id)) < 2:
        return math.nan, math.nan
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
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


def executable_base(sub: pd.DataFrame, horizon: int) -> pd.DataFrame:
    token_side = np.sign(sub["tob_imbalance"].to_numpy(dtype=float))
    entry_long = sub["best_ask"].to_numpy(dtype=float)
    exit_long = sub[f"future_bid_{horizon}s"].to_numpy(dtype=float)
    entry_short = sub["best_bid"].to_numpy(dtype=float)
    exit_short = sub[f"future_ask_{horizon}s"].to_numpy(dtype=float)

    entry_price = np.where(token_side > 0, entry_long, entry_short)
    exit_price = np.where(token_side > 0, exit_long, exit_short)
    ok = (
        sub["is_top_tob_decile"].to_numpy(dtype=bool)
        & np.isfinite(token_side)
        & (token_side != 0)
        & np.isfinite(entry_price)
        & np.isfinite(exit_price)
        & (entry_price > 0)
        & (exit_price >= 0)
        & np.isfinite(sub["spread_bps"].to_numpy(dtype=float))
    )
    out = sub.loc[ok].copy()
    if out.empty:
        out["entry_price"] = []
        out["exit_price"] = []
        out["token_side"] = []
        return out
    out["entry_price"] = entry_price[ok]
    out["exit_price"] = exit_price[ok]
    out["token_side"] = token_side[ok]
    return out


def add_pnl(events: pd.DataFrame, category: str) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        out["pnl_bps"] = pd.Series(dtype=float)
        return out
    token_side = out["token_side"].to_numpy(dtype=float)
    entry = out["entry_price"].to_numpy(dtype=float)
    exit_price = out["exit_price"].to_numpy(dtype=float)
    fee_entry = fee_amount(category, entry)
    fee_exit = fee_amount(category, exit_price)
    gross = np.where(token_side > 0, exit_price - entry, entry - exit_price)
    out["pnl_bps"] = (gross - fee_entry - fee_exit) / entry * 10_000.0
    return out


def summarize(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    meta = (
        df.groupby(["run_id", "market_id"], sort=False)
        .agg(
            slug=("slug", lambda s: str(s.replace("", np.nan).dropna().iloc[0]) if s.astype(bool).any() else ""),
            family=("family", lambda s: str(s.replace("", np.nan).dropna().iloc[0]) if s.astype(bool).any() else ""),
        )
        .reset_index()
    )
    cand = candidates.merge(meta, on=["run_id", "market_id"], how="left", suffixes=("", "_feature"))
    cand["family"] = cand["family_feature"].where(cand["family_feature"].notna(), cand["family"])
    for market_idx, market in cand.reset_index(drop=True).iterrows():
        run_id = str(market["run_id"])
        market_id = str(market["market_id"])
        market_key = f"{run_id}:{market_id}"
        slug = str(market.get("slug") or market_id)
        family = str(market.get("family") or "")
        category = family_category(family)
        sub = df[df["run_id"].eq(run_id) & df["market_id"].eq(market_id)].copy()
        for horizon in HORIZONS:
            base = executable_base(sub, horizon)
            n_total = int(len(base))
            for thresh_idx, (label, threshold) in enumerate(SPREAD_THRESHOLDS):
                if threshold is None:
                    selected = base.copy()
                else:
                    selected = base[base["spread_bps"].le(threshold)].copy()
                selected = add_pnl(selected, category)
                n_selected = int(len(selected))
                fillable_rate = n_selected / n_total if n_total else math.nan
                mean_pnl = float(selected["pnl_bps"].mean()) if n_selected else math.nan
                median_pnl = float(selected["pnl_bps"].median()) if n_selected else math.nan
                win_rate = float(selected["pnl_bps"].gt(0).mean()) if n_selected else math.nan
                ci_lo, ci_hi = bootstrap_mean_ci(
                    selected,
                    RNG_SEED + market_idx * 1000 + horizon * 10 + thresh_idx,
                )
                rows.append(
                    {
                        "market": market_key,
                        "slug": slug,
                        "horizon": horizon,
                        "spread_threshold_bps": label,
                        "n_signal_events_total": n_total,
                        "n_events_passing_filter": n_selected,
                        "fillable_rate": fillable_rate,
                        "mean_pnl_bps": mean_pnl,
                        "median_pnl_bps": median_pnl,
                        "win_rate": win_rate,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def threshold_sort_key(label: str) -> float:
    if label == "no_filter":
        return math.inf
    return float(label)


def summary_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    baseline = results[results["spread_threshold_bps"].eq("no_filter")][
        ["market", "horizon", "mean_pnl_bps"]
    ].rename(columns={"mean_pnl_bps": "baseline_pnl"})
    merged = results.merge(baseline, on=["market", "horizon"], how="left")
    merged["delta_vs_no_filter"] = merged["mean_pnl_bps"] - merged["baseline_pnl"]
    for (horizon, threshold), sub in merged.groupby(["horizon", "spread_threshold_bps"], sort=False):
        total_signal = int(sub["n_signal_events_total"].sum())
        total_passing = int(sub["n_events_passing_filter"].sum())
        trigger = total_passing / total_signal if total_signal else math.nan
        rows.append(
            [
                str(int(horizon)),
                str(threshold),
                f"{int(sub['mean_pnl_bps'].gt(0).sum())}/{len(sub)}",
                pct(trigger),
                pct(float(sub["fillable_rate"].median())),
                bps(float(sub["mean_pnl_bps"].mean())),
                bps(float(sub["delta_vs_no_filter"].mean())),
            ]
        )
    rows = sorted(rows, key=lambda r: (int(r[0]), threshold_sort_key(r[1])))
    return markdown_table(
        [
            "h",
            "threshold",
            "positive cells",
            "event trigger",
            "median market trigger",
            "mean pnl",
            "mean delta vs no_filter",
        ],
        rows,
    )


def survival_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for market, sub in results.groupby("market", sort=True):
        slug = str(sub["slug"].dropna().iloc[0])[:48]
        positive = sub[sub["mean_pnl_bps"].gt(0)].copy()
        if positive.empty:
            verdict = "no threshold tested produces positive PnL"
            best = sub.sort_values("mean_pnl_bps", ascending=False).iloc[0]
        else:
            interpretable = positive[positive["n_events_passing_filter"].ge(MIN_INTERPRETABLE_EVENTS)].copy()
            if interpretable.empty:
                best = positive.sort_values(["mean_pnl_bps", "fillable_rate"], ascending=False).iloc[0]
                verdict = (
                    f"tradeable with spread filter S={best['spread_threshold_bps']} at "
                    f"{int(best['horizon'])}s, but fragile n<{MIN_INTERPRETABLE_EVENTS}"
                )
            else:
                best = interpretable.sort_values(["mean_pnl_bps", "fillable_rate"], ascending=False).iloc[0]
                ci_note = "" if np.isfinite(best["ci_lo"]) and best["ci_lo"] > 0 else ", CI crosses zero"
                verdict = (
                    f"tradeable with spread filter S={best['spread_threshold_bps']} at "
                    f"{int(best['horizon'])}s{ci_note}"
                )
        rows.append(
            [
                market,
                slug.replace("|", "/"),
                str(best["spread_threshold_bps"]),
                str(int(best["horizon"])),
                pct(float(best["fillable_rate"])),
                bps(float(best["mean_pnl_bps"])),
                pct(float(best["win_rate"])),
                verdict,
            ]
        )
    return markdown_table(
        ["market", "slug", "best S", "h", "trigger", "mean pnl", "win", "verdict"],
        rows,
    )


def notable_rows_table(results: pd.DataFrame) -> str:
    sub = results.sort_values(["mean_pnl_bps", "fillable_rate"], ascending=False).head(15)
    rows: list[list[str]] = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                str(row.market),
                str(row.slug)[:42].replace("|", "/"),
                str(int(row.horizon)),
                str(row.spread_threshold_bps),
                f"{int(row.n_events_passing_filter):,}/{int(row.n_signal_events_total):,}",
                pct(float(row.fillable_rate)),
                bps(float(row.mean_pnl_bps)),
                bps(float(row.median_pnl_bps)),
                pct(float(row.win_rate)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
            ]
        )
    return markdown_table(
        ["market", "slug", "h", "S", "events", "trigger", "mean", "median", "win", "CI"],
        rows,
    )


def a13_sanity_table() -> str:
    if not A13_CONTROL.exists():
        return "_A13 control bucket file not found._"
    a13 = pd.read_csv(A13_CONTROL)
    ref = a13[
        a13["signal_variant"].eq("current_level")
        & a13["segment_type"].eq("spread_bucket")
        & a13["decile"].eq(10)
    ].copy()
    if ref.empty:
        return "_No A13 spread-bucket reference rows._"
    rows: list[list[str]] = []
    for bucket, sub in ref.groupby("segment_value", sort=True):
        rows.append(
            [
                str(bucket),
                f"{int(sub['n'].sum()):,}",
                bps(float(sub["mean_spread_bps"].mean())),
                pct(float(sub["hit_rate"].mean())),
                bps(float(sub["directional_return_bps"].mean())),
            ]
        )
    return markdown_table(
        ["A13 spread bucket", "top-decile rows", "mean spread", "mean hit", "mean dir ret"],
        rows,
    )


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    positive = results[results["mean_pnl_bps"].gt(0)].copy()
    interpretable_positive = positive[positive["n_events_passing_filter"].ge(MIN_INTERPRETABLE_EVENTS)].copy()
    total_cells = len(results)
    positive_cells = len(positive)
    if positive.empty:
        headline = (
            "No spread threshold tested flips the fixed-horizon TOB taker simulation positive. "
            "Tight-spread entry improves some rows versus no_filter, but spread-conditional entry alone "
            "is not enough to make this tradeable in A0/A0b."
        )
    elif not interpretable_positive.empty:
        best = interpretable_positive.sort_values(["mean_pnl_bps", "fillable_rate"], ascending=False).iloc[0]
        no_filter = results[
            results["market"].eq(best["market"])
            & results["horizon"].eq(best["horizon"])
            & results["spread_threshold_bps"].eq("no_filter")
        ]
        baseline_text = ""
        if not no_filter.empty and np.isfinite(no_filter["mean_pnl_bps"].iloc[0]):
            baseline = float(no_filter["mean_pnl_bps"].iloc[0])
            baseline_text = (
                f" The no_filter baseline for that same market/horizon is {bps(baseline)}, "
                "so the filter amplifies that one cell rather than creating a broad cross-market flip."
            )
        ci_text = (
            f" CI [{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]"
            if np.isfinite(best["ci_lo"]) and np.isfinite(best["ci_hi"])
            else " CI unavailable"
        )
        headline = (
            f"Positive PnL appears only in {positive['market'].nunique()} market(s), concentrated at "
            f"{', '.join(str(h) + 's' for h in sorted(positive['horizon'].unique()))}. "
            f"The best interpretable positive row is `{best['market']}` at {int(best['horizon'])}s with "
            f"S={best['spread_threshold_bps']} bps: {bps(float(best['mean_pnl_bps']))} mean PnL on "
            f"{int(best['n_events_passing_filter']):,}/{int(best['n_signal_events_total']):,} events "
            f"({pct(float(best['fillable_rate']))} trigger),{ci_text}.{baseline_text} "
            "Spread-conditional entry alone is therefore not a robust tradeability result."
        )
    else:
        best = positive.sort_values(["mean_pnl_bps", "fillable_rate"], ascending=False).iloc[0]
        headline = (
            f"The only positive rows are sparse-trigger artifacts. The best positive cell is "
            f"`{best['market']}` at {int(best['horizon'])}s with "
            f"S={best['spread_threshold_bps']} bps: {bps(float(best['mean_pnl_bps']))} mean PnL, "
            f"triggering on {int(best['n_events_passing_filter']):,}/"
            f"{int(best['n_signal_events_total']):,} top-decile signal events "
            f"({pct(float(best['fillable_rate']))})."
        )

    note = f"""---
tags: [dali, block-a14d, executable-cost, results]
---

# Block A1.4d Tight-Spread Entry Findings

## Headline

{headline} Overall, {positive_cells} of {total_cells} market-horizon-threshold cells have positive mean PnL after taker entry, fixed-horizon taker exit, and fees on both legs.

## Method

- Candidate universe: all `primary_read` markets from `block_a1_results.csv`, not only A1.4's `signal_present_pre_cost` rows.
- Signal: current top-of-book imbalance level, `tob_imbalance_level = direction_factor * tob_imbalance`.
- Deciles: equal-count ranked deciles by `abs(tob_imbalance_level)` within each `(run_id, market_id)`, computed once from current-state rows and reused across horizons. Only decile 10 is traded.
- Entry filter: current `spread_bps <= S`, with `S in {{50, 100, 200, 500, 1000, no_filter}}`.
- Execution: same fixed-horizon A1.4 touch round trip. A long token signal pays `best_ask` and exits at future `best_bid`; a short token signal receives `best_bid` and exits at future `best_ask`.
- Horizons: 5s, 30s, and 300s.
- Fees: A1 `FEE_BY_CATEGORY`, charged on both entry and exit as dollars per share and normalized by entry price.
- Bootstrap: 200-sample block bootstrap of mean PnL with contiguous 300s clock-time blocks.

## Cross-Market Threshold Summary

The `event trigger` column is the pooled share of top-decile signal events that pass the spread filter. The `median market trigger` column avoids one very active market dominating the read. Deltas are versus each market-horizon's `no_filter` baseline.

{summary_table(results)}

## Per-Market Threshold Survival

{survival_table(results)}

## Best Cells

{notable_rows_table(results)}

## A13 Spread Sanity Reference

This is not used for the A14d PnL math; it is a sanity check that the absolute spread thresholds bracket A13's observed top-decile spread regimes.

{a13_sanity_table()}

## Interpretation

Spread filtering is necessary for any executable taker version of this signal, but the fixed-horizon version is still carrying too much spread and adverse movement. If a positive cell appears, it should be treated as a narrow capture-window diagnostic until latency, capacity at touch, and A14b-style exit rules are layered in. If no positive cell appears, the conclusion is cleaner: spread-conditional entry alone does not rescue the top-decile TOB signal.

Recommended next action for Justin: do not treat spread filtering alone as sufficient; combine tight-spread entry with the A14b exit-rule work and require a minimum trigger-rate/capacity screen before A2 trading design.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates = load_candidates()
    features = load_feature_subset(candidates)
    features = add_market_signals(features)
    results = summarize(features, candidates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
