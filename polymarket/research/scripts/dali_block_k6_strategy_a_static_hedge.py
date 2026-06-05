"""K6 Strategy A: passive K-PEG entry, static Binance hedge, hold to resolution.

This is the missing gate before any forward-vol/Kronos bake-off. It reuses the
K-PEG passive-fill proxy and the K6/K3 digital surface, then replaces the
Polymarket exit / continuous hedge with a single entry-time Binance hedge.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table, number, pct


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"

K6_PANEL = ANALYSIS / "k6_vol_gap_panel.parquet"
KPEG_FILLS = ANALYSIS / "kpeg_robustness_fills.parquet"
OUT_TRADES = ANALYSIS / "k6_strategy_a_static_hedge_trades.parquet"
OUT_CSV = ANALYSIS / "csv_outputs" / "options_delta" / "k6_strategy_a_static_hedge.csv"
NOTE = NOTES / "block_k6_strategy_a_static_hedge_findings.md"

CRYPTO_FEE_RATE = 0.07
MAKER_REBATE_SHARE = 0.20
BINANCE_HEDGE_COST_BPS = 6.0
BOOTSTRAP_SAMPLES = 1000
RNG_SEED = 20260531
SPIKE_TAU_SECONDS = 15 * 60
SPIKE_MIN_PRICE = 0.40
SPIKE_MAX_PRICE = 0.60
ROBUST_MIN_TRADES = 5


PANEL_COLS = [
    "market_id",
    "market_slug",
    "asset",
    "ts",
    "source_runs",
    "window_start",
    "window_end",
    "seconds_to_expiry",
    "up_bid",
    "up_ask",
    "down_bid",
    "down_ask",
    "polymarket_mid",
    "binance_spot",
    "binance_close_spot",
    "binance_strike_spot",
    "binance_window_abs_return_bps",
    "binance_resolution_up",
    "chainlink_resolution_up",
    "chainlink_binance_resolution_disagree",
    "resolution_source",
    "digital_delta",
    "abs_z",
    "moneyness_bucket",
    "time_bucket",
    "state_bucket",
    "source_ok_strict",
    "large_static_basis_10c",
    "toxic_near_expiry",
]


def fee_rebate(price: float) -> float:
    p = float(np.clip(price, 0.0, 1.0))
    return MAKER_REBATE_SHARE * CRYPTO_FEE_RATE * p * (1.0 - p)


def bootstrap_ci(trades: pd.DataFrame, col: str) -> tuple[float, float]:
    clean = trades[["market_id", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return math.nan, math.nan
    blocks = [g[col].to_numpy(dtype=float) for _, g in clean.groupby("market_id", sort=False)]
    if len(blocks) == 1:
        val = float(np.nanmean(blocks[0]))
        return val, val
    rng = np.random.default_rng(RNG_SEED + len(clean) + len(col))
    vals = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        vals.append(float(np.nanmean(np.concatenate([blocks[i] for i in idx]))))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def ci_text(lo: float, hi: float) -> str:
    return f"[{cents(lo)}, {cents(hi)}]"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not K6_PANEL.exists():
        raise FileNotFoundError(f"missing {K6_PANEL}; run K6 vol gap first")
    if not KPEG_FILLS.exists():
        raise FileNotFoundError(f"missing {KPEG_FILLS}; run K-PEG robustness first")
    panel = pd.read_parquet(K6_PANEL, columns=PANEL_COLS)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["market_id"] = panel["market_id"].astype(str)
    panel = panel.sort_values(["market_id", "ts"]).reset_index(drop=True)

    fills = pd.read_parquet(KPEG_FILLS)
    fills = fills[fills["category"].eq("Crypto")].copy()
    fills["market_id"] = fills["market_id"].astype(str)
    fills["fill_ts"] = pd.to_datetime(fills["fill_time_ns"].astype("int64"), unit="ns", utc=True)
    fills = fills[fills["market_id"].isin(set(panel["market_id"]))].copy()
    fills = fills.sort_values(["market_id", "fill_ts"]).reset_index(drop=True)
    return panel, fills


def asof_join_entries(panel: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for market_id, f in fills.groupby("market_id", sort=False):
        p = panel[panel["market_id"].eq(market_id)].sort_values("ts").copy()
        if p.empty:
            continue
        # Parquet restores some timestamps as us and some as ns. Timestamp.value
        # is always integer nanoseconds, which keeps the as-of join honest.
        p["ts_key"] = p["ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")
        f = f.copy()
        f["fill_ts_key"] = f["fill_ts"].map(lambda x: pd.Timestamp(x).value).astype("int64")
        joined = pd.merge_asof(
            f.sort_values("fill_ts_key"),
            p,
            left_on="fill_ts_key",
            right_on="ts_key",
            direction="backward",
            suffixes=("", "_panel"),
        )
        pieces.append(joined)
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    df = df[df["ts"].notna()].copy()
    return df.sort_values(["market_id", "fill_ts"]).reset_index(drop=True)


def add_trade_economics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    up_mid = (out["up_bid"].astype(float) + out["up_ask"].astype(float)) / 2.0
    down_mid = (out["down_bid"].astype(float) + out["down_ask"].astype(float)) / 2.0
    asset_mid = out["current_mid"].astype(float)
    out["outcome"] = np.where((asset_mid - up_mid).abs() <= (asset_mid - down_mid).abs(), "up", "down")

    chain = out["chainlink_resolution_up"]
    binance_res = out["binance_resolution_up"]
    resolution_up = np.where(chain.notna(), chain.astype(float), binance_res.astype(float))
    out["resolution_up_used"] = resolution_up.astype(bool)
    up_payoff = out["resolution_up_used"].astype(float)
    out["payoff"] = np.where(out["outcome"].eq("up"), up_payoff, 1.0 - up_payoff)

    out["entry_price_strategy"] = out["entry_price"].astype(float)
    out["token_position"] = out["token_side"].astype(float)
    out["maker_rebate"] = out["entry_price_strategy"].map(fee_rebate)
    out["pm_taker_fee"] = 0.0
    out["pm_leg_pnl"] = out["token_position"] * (out["payoff"] - out["entry_price_strategy"]) + out["maker_rebate"]

    delta_up = out["digital_delta"].astype(float)
    delta_outcome = np.where(out["outcome"].eq("up"), delta_up, -delta_up)
    out["entry_delta_outcome"] = delta_outcome
    out["digital_delta_exposure"] = out["token_position"] * out["entry_delta_outcome"]
    out["hedge_units"] = -out["digital_delta_exposure"]
    entry_spot = out["binance_spot"].astype(float)
    close_spot = out["binance_close_spot"].astype(float)
    out["hedge_pnl"] = out["hedge_units"] * (close_spot - entry_spot)
    out["hedge_entry_cost"] = out["hedge_units"].abs() * entry_spot * BINANCE_HEDGE_COST_BPS / 10_000.0
    out["hedge_exit_cost"] = out["hedge_units"].abs() * close_spot * BINANCE_HEDGE_COST_BPS / 10_000.0
    out["hedge_cost"] = out["hedge_entry_cost"] + out["hedge_exit_cost"]
    out["net_pnl"] = out["pm_leg_pnl"] + out["hedge_pnl"] - out["hedge_cost"]
    out["unhedged_pnl"] = out["pm_leg_pnl"]
    out["hold_seconds"] = (pd.to_datetime(out["window_end"], utc=True) - out["fill_ts"]).dt.total_seconds()

    out["two_sided_book_eligible"] = (
        out[["up_bid", "up_ask", "down_bid", "down_ask"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & out["up_ask"].astype(float).gt(out["up_bid"].astype(float))
        & out["down_ask"].astype(float).gt(out["down_bid"].astype(float))
    )
    out["late_near50_spike_zone"] = (
        out["seconds_to_expiry"].astype(float).le(SPIKE_TAU_SECONDS)
        & out["entry_price_strategy"].between(SPIKE_MIN_PRICE, SPIKE_MAX_PRICE, inclusive="both")
    ) | out["toxic_near_expiry"].fillna(False).astype(bool)
    out["strategy_a_eligible"] = out["two_sided_book_eligible"] & ~out["late_near50_spike_zone"]
    out["strict_source_eligible"] = out["strategy_a_eligible"] & out["source_ok_strict"].fillna(False).astype(bool)
    out["entry_split"] = np.where(out["run_group"].eq("holdout"), "oos_holdout", "is_discovery")
    out["all_split"] = "pooled"
    out["route"] = np.where(
        out["token_position"].gt(0),
        "maker_buy_" + out["outcome"].astype(str),
        "maker_sell_" + out["outcome"].astype(str),
    )
    return out


def build_trades(refresh: bool = False) -> pd.DataFrame:
    if OUT_TRADES.exists() and not refresh:
        print(f"loading {OUT_TRADES}", flush=True)
        return pd.read_parquet(OUT_TRADES)
    panel, fills = read_inputs()
    print(f"loaded panel rows={len(panel):,} markets={panel['market_id'].nunique()}", flush=True)
    print(f"loaded overlapping K-PEG crypto fills={len(fills):,} markets={fills['market_id'].nunique()}", flush=True)
    joined = asof_join_entries(panel, fills)
    trades = add_trade_economics(joined)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(OUT_TRADES, index=False)
    print(f"wrote {OUT_TRADES}", flush=True)
    return trades


def select_non_overlap(trades: pd.DataFrame, *, source_filter: str, split: str, bucket: str) -> pd.DataFrame:
    if source_filter == "strict":
        mask = trades["strict_source_eligible"].fillna(False).astype(bool)
    else:
        mask = trades["strategy_a_eligible"].fillna(False).astype(bool)
    sub = trades[mask].copy()
    if split != "pooled":
        sub = sub[sub["entry_split"].eq(split)].copy()
    if bucket != "all_buckets":
        sub = sub[sub["state_bucket"].eq(bucket)].copy()
    if sub.empty:
        return sub
    # A bucket cell is evaluated as its own restricted strategy. Holding to
    # resolution means at most one fill per market in that restricted strategy.
    sub = sub.sort_values(["fill_ts", "market_id"]).groupby("market_id", as_index=False, sort=False).first()
    return sub.sort_values("fill_ts").reset_index(drop=True)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buckets = ["all_buckets", *sorted(trades["state_bucket"].dropna().astype(str).unique())]
    for source_filter in ("all", "strict"):
        for split in ("pooled", "is_discovery", "oos_holdout"):
            for bucket in buckets:
                sub = select_non_overlap(trades, source_filter=source_filter, split=split, bucket=bucket)
                if sub.empty:
                    continue
                lo, hi = bootstrap_ci(sub, "net_pnl")
                ulo, uhi = bootstrap_ci(sub, "unhedged_pnl")
                rows.append(
                    {
                        "source_filter": source_filter,
                        "sample_split": split,
                        "state_bucket": bucket,
                        "n_trades": int(len(sub)),
                        "n_markets": int(sub["market_id"].nunique()),
                        "mean_net_pnl": float(sub["net_pnl"].mean()),
                        "net_ci_lo": lo,
                        "net_ci_hi": hi,
                        "mean_unhedged_pnl": float(sub["unhedged_pnl"].mean()),
                        "unhedged_ci_lo": ulo,
                        "unhedged_ci_hi": uhi,
                        "mean_pm_leg_pnl": float(sub["pm_leg_pnl"].mean()),
                        "mean_hedge_pnl": float(sub["hedge_pnl"].mean()),
                        "mean_maker_rebate": float(sub["maker_rebate"].mean()),
                        "mean_pm_taker_fee": float(sub["pm_taker_fee"].mean()),
                        "mean_hedge_cost": float(sub["hedge_cost"].mean()),
                        "mean_abs_hedge_notional": float((sub["hedge_units"].abs() * sub["binance_spot"]).mean()),
                        "win_rate": float(sub["net_pnl"].gt(0).mean()),
                        "median_hold_seconds": float(sub["hold_seconds"].median()),
                        "p95_hold_seconds": float(sub["hold_seconds"].quantile(0.95)),
                        "maker_buy_share": float(sub["token_position"].gt(0).mean()),
                        "route_top": str(sub["route"].value_counts().index[0]),
                        "tail_market_share": float(sub["market_id"].value_counts(normalize=True).iloc[0]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}", flush=True)
    return out


def table_rows(summary: pd.DataFrame, split: str, source_filter: str = "strict") -> list[list[str]]:
    sub = summary[
        summary["source_filter"].eq(source_filter)
        & summary["sample_split"].eq(split)
        & summary["state_bucket"].ne("all_buckets")
    ].copy()
    if sub.empty:
        return []
    sub["is_far_late"] = sub["state_bucket"].eq("far_absz_ge1|late_lt30m")
    sub = sub.sort_values(["is_far_late", "net_ci_lo", "mean_net_pnl"], ascending=[False, False, False])
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            [
                str(r["state_bucket"]),
                str(int(r["n_trades"])),
                cents(float(r["mean_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                cents(float(r["mean_unhedged_pnl"])),
                cents(float(r["mean_pm_taker_fee"])),
                cents(float(r["mean_maker_rebate"])),
                cents(float(r["mean_hedge_cost"])),
                pct(float(r["win_rate"])),
                number(float(r["median_hold_seconds"]) / 60.0, 1),
            ]
        )
    return rows


def split_rows(summary: pd.DataFrame, bucket: str = "far_absz_ge1|late_lt30m") -> list[list[str]]:
    rows = []
    sub = summary[
        summary["source_filter"].eq("strict")
        & summary["state_bucket"].eq(bucket)
        & summary["sample_split"].isin(["is_discovery", "oos_holdout", "pooled"])
    ].sort_values("sample_split")
    for _, r in sub.iterrows():
        rows.append(
            [
                str(r["sample_split"]),
                str(int(r["n_trades"])),
                cents(float(r["mean_net_pnl"])),
                ci_text(float(r["net_ci_lo"]), float(r["net_ci_hi"])),
                cents(float(r["mean_unhedged_pnl"])),
                cents(float(r["mean_hedge_cost"])),
                pct(float(r["win_rate"])),
            ]
        )
    return rows


def write_note(trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    gate = summary[
        summary["source_filter"].eq("strict")
        & summary["sample_split"].eq("oos_holdout")
        & summary["state_bucket"].eq("far_absz_ge1|late_lt30m")
    ]
    if gate.empty:
        gate_text = "No strict OOS far/late non-overlap trades survived the Strategy-A filters; gate fails."
        gate_pass = False
    else:
        g = gate.iloc[0]
        gate_pass = bool(float(g["net_ci_lo"]) > 0)
        gate_text = (
            f"Strict OOS far/late: n={int(g['n_trades'])}, mean net {cents(float(g['mean_net_pnl']))}, "
            f"CI {ci_text(float(g['net_ci_lo']), float(g['net_ci_hi']))}."
        )
    decision = (
        "Gate clears: static-hedge Strategy A passes the OOS far/late lower-CI test."
        if gate_pass
        else "Gate fails: do not unblock Kronos/HAR/EWMA forward-vol bake-off."
    )

    eligible = trades[trades["strategy_a_eligible"]]
    strict = trades[trades["strict_source_eligible"]]
    overlap = trades
    note = f"""# Block K6 Strategy A Static Hedge

## Headline

{decision}

{gate_text}

This is the missing static-hedge gate: passive K-PEG maker entries, no Polymarket exit, one Binance hedge set at entry delta and closed at resolution. The primary table uses strict source filtering, K5-style two-sided book eligibility, and excludes the late near-50c spike zone.

## Construction

- Entry source: `data/analysis/kpeg_robustness_fills.parquet`, restricted to crypto-4h fills that overlap the K6/K3 panel.
- Pricing surface: `data/analysis/k6_vol_gap_panel.parquet`.
- Entry is passive maker fill: maker fee is zero; maker rebate is `20% * 0.07 * p * (1-p)`.
- Static hedge: `hedge_units = -entry_token_position * entry_digital_delta`; held unchanged until Binance window close.
- Binance cost: entry plus settlement notional at `{BINANCE_HEDGE_COST_BPS:.1f}bp` each way.
- Spike filter: exclude rows with <=15m to expiry and token price in `[0.40, 0.60]`, plus K6 toxic near-strike/near-expiry rows.
- Non-overlap: each bucket cell is a restricted strategy and takes at most the first eligible fill per market, held to resolution.

Input counts:

- overlapping K-PEG/K6 crypto fills: `{len(overlap):,}`
- K5-style eligible fills before non-overlap: `{len(eligible):,}`
- strict-source eligible fills before non-overlap: `{len(strict):,}`

## OOS Bucket Table

Strict source, OOS holdout only.

{markdown_table(
    ["bucket", "n", "net", "net CI", "unhedged", "PM taker fee", "maker rebate", "hedge cost", "win", "median hold min"],
    table_rows(summary, "oos_holdout", "strict"),
)}

## IS Bucket Table

Strict source, discovery only. This is a lead, not the result.

{markdown_table(
    ["bucket", "n", "net", "net CI", "unhedged", "PM taker fee", "maker rebate", "hedge cost", "win", "median hold min"],
    table_rows(summary, "is_discovery", "strict"),
)}

## Far/Late Split

Decision bucket: `far_absz_ge1|late_lt30m`.

{markdown_table(
    ["sample", "n", "net", "net CI", "unhedged", "hedge cost", "win"],
    split_rows(summary),
)}

## Decision

{decision}

The static hedge removes the continuous-turnover problem from K6, but the OOS far/late gate is judged only on the strict-source, non-overlap bucket lower CI. This note does not run Kronos or any forward-vol model.

Outputs:

- `data/analysis/csv_outputs/options_delta/k6_strategy_a_static_hedge.csv`
- `data/analysis/k6_strategy_a_static_hedge_trades.parquet`
"""
    NOTE.write_text(note, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def main() -> None:
    trades = build_trades(refresh=True)
    summary = summarize(trades)
    write_note(trades, summary)


if __name__ == "__main__":
    main()
