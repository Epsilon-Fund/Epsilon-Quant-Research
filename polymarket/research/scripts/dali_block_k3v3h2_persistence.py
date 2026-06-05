"""Block K3 v3h2: persistence-gated moderate-delta dynamic-basis RV.

Final falsifier for the hedged-basis thesis: remove far/late tail noise,
require the dynamic logit gap to persist, and evaluate taker-in/taker-out
convergence with and without the Binance delta hedge.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dali_block_k3v2_leadlag_causal import cents, markdown_table, number, pct
from dali_block_k3v3h_hedged_basis import (
    ACTION_LATENCIES,
    ANALYSIS,
    BINANCE_HEDGE_COST_BPS,
    BOOTSTRAP_SAMPLES,
    FEATURE_CACHE,
    FUNDING_BPS_PER_8H,
    RNG_SEED,
    SOURCE_MARGIN_BP,
    STATIC_BASIS_CUTOFF,
    bootstrap_mean,
    hedge_path_pnl_arrays,
    load_features,
    market_arrays,
    ns_to_ts,
    taker_fee,
)


NOTES = Path(__file__).resolve().parents[1] / "notes"
OUT_TRADES = ANALYSIS / "csv_outputs" / "options_delta" / "k3v3h2_persistence_trades.csv"
OUT_SUMMARY = ANALYSIS / "csv_outputs" / "options_delta" / "k3v3h2_persistence_summary.csv"
OUT_EXTENDED_LEDGER = ANALYSIS / "csv_outputs" / "options_delta" / "k3v3h_hedged_basis_trades_ext.csv"
PREV_TRADES = ANALYSIS / "csv_outputs" / "options_delta" / "k3v3h_hedged_basis_trades.csv"
NOTE = NOTES / "block_k3v3h2_findings.md"

ENTRY_BANDS = (0.25, 0.35, 0.50, 0.75)
EXIT_BANDS = (0.10, 0.25, 0.50)
PERSISTENCE_SECONDS = (5, 15, 30)
MAX_HOLD_SECONDS = (60, 180, 300, 600)
MIN_ABS_Z = 0.25
MAX_ABS_Z = 1.0
MIN_TAU_SECONDS = 30 * 60
MAX_TAU_SECONDS = 2 * 60 * 60
TARGET_BUCKET = "mid_absz_0.25_1|mid_30m_2h"
MIN_TRADES_FOR_SELECTION = 10
MAX_CONSECUTIVE_GAP_SECONDS = 2.0


@dataclass
class PersistenceTrade:
    strategy_phase: str
    latency_s: int
    entry_band: float
    exit_band: float
    persistence_k_s: int
    max_hold_s: int
    asset: str
    market_slug: str
    state_bucket: str
    route: str
    exit_reason: str
    signal_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    exit_signal_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    hold_seconds: float
    entry_run_seconds: float
    signal_dynamic_gap: float
    entry_dynamic_gap: float
    exit_dynamic_gap: float
    entry_abs_z: float
    exit_abs_z: float
    entry_seconds_to_expiry: float
    exit_seconds_to_expiry: float
    entry_spot: float
    exit_spot: float
    pm_entry_price: float
    pm_exit_price: float
    pm_entry_fee: float
    pm_exit_fee: float
    pm_pnl: float
    hedge_pnl: float
    hedge_turnover_notional: float
    hedge_cost: float
    funding_cost: float
    hedged_net_pnl: float
    naked_net_pnl: float
    source_ok: bool
    source_disagree: bool
    settlement_margin_bp: float
    static_large_entry: bool


def valid_moderate_mid(arr: dict[str, Any]) -> np.ndarray:
    abs_z = arr["abs_z"]
    tau = arr["seconds_to_expiry"]
    gap = arr["dynamic_gap"]
    return (
        np.isfinite(gap)
        & (abs_z >= MIN_ABS_Z)
        & (abs_z <= MAX_ABS_Z)
        & (tau >= MIN_TAU_SECONDS)
        & (tau <= MAX_TAU_SECONDS)
        & (~arr["large_static"])
        & (~arr["toxic"])
    )


def ci_cents_text(lo: float, hi: float) -> str:
    return f"[{cents(lo)}, {cents(hi)}]"


def add_regime_arrays(arr: dict[str, Any]) -> dict[str, Any]:
    arr = dict(arr)
    arr["valid_regime"] = valid_moderate_mid(arr)
    return arr


def persistence_runs(
    gap: np.ndarray,
    ts_ns: np.ndarray,
    entry_band: float,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros(len(gap), dtype=float)
    neg = np.zeros(len(gap), dtype=float)
    if active is None:
        active = np.ones(len(gap), dtype=bool)
    for i, value in enumerate(gap):
        if (not bool(active[i])) or not np.isfinite(value):
            continue
        if i == 0:
            dt = 1.0
            contiguous = False
        else:
            dt = float((ts_ns[i] - ts_ns[i - 1]) / 1_000_000_000)
            contiguous = bool(active[i - 1]) and 0 < dt <= MAX_CONSECUTIVE_GAP_SECONDS
            if not contiguous:
                dt = 1.0
        if value > entry_band:
            if i > 0 and contiguous and gap[i - 1] > entry_band:
                pos[i] = pos[i - 1] + dt
            else:
                pos[i] = 1.0
        elif value < -entry_band:
            if i > 0 and contiguous and gap[i - 1] < -entry_band:
                neg[i] = neg[i - 1] + dt
            else:
                neg[i] = 1.0
    return pos, neg


def exit_reason(arr: dict[str, Any], pos: int, route: str, exit_band: float, entry_ts_ns: int, max_hold_s: int) -> str | None:
    if not bool(arr["valid_regime"][pos]):
        return "regime_exit"
    elapsed = float((arr["ts_ns"][pos] - entry_ts_ns) / 1_000_000_000)
    if elapsed >= max_hold_s:
        return "max_hold"
    gap = float(arr["dynamic_gap"][pos])
    if route == "buy_up" and gap >= -exit_band:
        return "converged"
    if route == "buy_down" and gap <= exit_band:
        return "converged"
    return None


def simulate_market(
    arr: dict[str, Any],
    *,
    latency: int,
    entry_band: float,
    exit_band: float,
    persistence_k: int,
    max_hold_s: int,
) -> list[PersistenceTrade]:
    if not bool(arr["source_ok"]):
        return []
    n = len(arr["ts_ns"])
    if n <= latency + 2:
        return []

    pos_run, neg_run = persistence_runs(arr["dynamic_gap"], arr["ts_ns"], entry_band, arr["valid_regime"])
    signalable = arr["valid_regime"] & ((pos_run >= persistence_k) | (neg_run >= persistence_k))
    candidates = np.flatnonzero(signalable[: max(0, n - latency - 2)])
    trades: list[PersistenceTrade] = []
    next_allowed = 0

    for signal_pos in candidates:
        if signal_pos < next_allowed:
            continue
        gap = float(arr["dynamic_gap"][signal_pos])
        if gap >= entry_band and pos_run[signal_pos] >= persistence_k:
            route = "buy_down"
            entry_run = float(pos_run[signal_pos])
        elif gap <= -entry_band and neg_run[signal_pos] >= persistence_k:
            route = "buy_up"
            entry_run = float(neg_run[signal_pos])
        else:
            continue

        entry_pos = signal_pos + latency
        if entry_pos >= n - 2 or not bool(arr["valid_regime"][entry_pos]):
            continue

        j = entry_pos + 1
        reason = "end_of_market"
        while j < n - latency - 1:
            maybe_reason = exit_reason(arr, j, route, exit_band, int(arr["ts_ns"][entry_pos]), max_hold_s)
            if maybe_reason is not None:
                reason = maybe_reason
                break
            j += 1
        exit_signal_pos = min(j, n - latency - 1)
        exit_pos = min(exit_signal_pos + latency, n - 1)

        if route == "buy_up":
            pm_entry = float(arr["up_ask"][entry_pos])
            pm_exit = float(arr["up_bid"][exit_pos])
        else:
            pm_entry = float(arr["down_ask"][entry_pos])
            pm_exit = float(arr["down_bid"][exit_pos])
        if not all(np.isfinite(x) for x in (pm_entry, pm_exit)):
            next_allowed = exit_pos + 1
            continue

        entry_fee_rate = float(arr["fee_rate"][entry_pos]) if np.isfinite(float(arr["fee_rate"][entry_pos])) else 0.07
        exit_fee_rate = float(arr["fee_rate"][exit_pos]) if np.isfinite(float(arr["fee_rate"][exit_pos])) else entry_fee_rate
        entry_fee = taker_fee(pm_entry, entry_fee_rate)
        exit_fee = taker_fee(pm_exit, exit_fee_rate)
        pm_pnl = pm_exit - exit_fee - pm_entry - entry_fee
        hedge_pnl, turnover, hedge_cost, funding_cost = hedge_path_pnl_arrays(arr, route, entry_pos, exit_pos)
        hedged = pm_pnl + hedge_pnl - hedge_cost - funding_cost
        hold_seconds = float((arr["ts_ns"][exit_pos] - arr["ts_ns"][entry_pos]) / 1_000_000_000)

        trades.append(
            PersistenceTrade(
                strategy_phase="v3h2_persistence",
                latency_s=latency,
                entry_band=entry_band,
                exit_band=exit_band,
                persistence_k_s=persistence_k,
                max_hold_s=max_hold_s,
                asset=arr["asset"],
                market_slug=arr["market_slug"],
                state_bucket=TARGET_BUCKET,
                route=route,
                exit_reason=reason,
                signal_ts=ns_to_ts(arr["ts_ns"][signal_pos]),
                entry_ts=ns_to_ts(arr["ts_ns"][entry_pos]),
                exit_signal_ts=ns_to_ts(arr["ts_ns"][exit_signal_pos]),
                exit_ts=ns_to_ts(arr["ts_ns"][exit_pos]),
                hold_seconds=hold_seconds,
                entry_run_seconds=entry_run,
                signal_dynamic_gap=gap,
                entry_dynamic_gap=float(arr["dynamic_gap"][entry_pos]),
                exit_dynamic_gap=float(arr["dynamic_gap"][exit_pos]),
                entry_abs_z=float(arr["abs_z"][entry_pos]),
                exit_abs_z=float(arr["abs_z"][exit_pos]),
                entry_seconds_to_expiry=float(arr["seconds_to_expiry"][entry_pos]),
                exit_seconds_to_expiry=float(arr["seconds_to_expiry"][exit_pos]),
                entry_spot=float(arr["spot"][entry_pos]),
                exit_spot=float(arr["spot"][exit_pos]),
                pm_entry_price=pm_entry,
                pm_exit_price=pm_exit,
                pm_entry_fee=entry_fee,
                pm_exit_fee=exit_fee,
                pm_pnl=pm_pnl,
                hedge_pnl=hedge_pnl,
                hedge_turnover_notional=turnover,
                hedge_cost=hedge_cost,
                funding_cost=funding_cost,
                hedged_net_pnl=hedged,
                naked_net_pnl=pm_pnl,
                source_ok=True,
                source_disagree=bool(arr["source_disagree"]),
                settlement_margin_bp=float(arr["settlement_margin_bp"]),
                static_large_entry=bool(arr["large_static"][entry_pos]),
            )
        )
        next_allowed = exit_pos + 1
    return trades


def simulate_all(panel: pd.DataFrame) -> pd.DataFrame:
    strict_target = panel[
        panel["source_ok_strict"].fillna(False).astype(bool)
        & panel["abs_z"].between(MIN_ABS_Z, MAX_ABS_Z, inclusive="both")
        & panel["seconds_to_expiry"].between(MIN_TAU_SECONDS, MAX_TAU_SECONDS, inclusive="both")
    ]
    market_slugs = set(strict_target["market_slug"].dropna().unique())
    groups = [add_regime_arrays(market_arrays(g)) for m, g in panel.groupby("market_slug", sort=False) if m in market_slugs]
    combos = [
        (latency, entry, exit_band, persistence_k, max_hold_s)
        for latency in ACTION_LATENCIES
        for entry in ENTRY_BANDS
        for exit_band in EXIT_BANDS
        if exit_band < entry
        for persistence_k in PERSISTENCE_SECONDS
        for max_hold_s in MAX_HOLD_SECONDS
    ]
    trades: list[PersistenceTrade] = []
    for combo_i, (latency, entry, exit_band, persistence_k, max_hold_s) in enumerate(combos, start=1):
        print(
            "sim "
            f"{combo_i}/{len(combos)} latency={latency}s entry={entry} exit={exit_band} "
            f"k={persistence_k}s max_hold={max_hold_s}s",
            flush=True,
        )
        for arr in groups:
            trades.extend(
                simulate_market(
                    arr,
                    latency=latency,
                    entry_band=entry,
                    exit_band=exit_band,
                    persistence_k=persistence_k,
                    max_hold_s=max_hold_s,
                )
            )
    return pd.DataFrame([trade.__dict__ for trade in trades])


def summarize_grid(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["latency_s", "entry_band", "exit_band", "persistence_k_s", "max_hold_s"]
    for key, g in trades.groupby(keys, sort=True):
        if len(g) < 5:
            continue
        h_lo, h_hi = bootstrap_mean(g, "hedged_net_pnl")
        n_lo, n_hi = bootstrap_mean(g, "naked_net_pnl")
        rows.append(
            {
                "latency_s": int(key[0]),
                "entry_band": float(key[1]),
                "exit_band": float(key[2]),
                "persistence_k_s": int(key[3]),
                "max_hold_s": int(key[4]),
                "n_trades": int(len(g)),
                "markets": int(g["market_slug"].nunique()),
                "mean_hedged_pnl": float(g["hedged_net_pnl"].mean()),
                "hedged_ci_lo": h_lo,
                "hedged_ci_hi": h_hi,
                "mean_naked_pnl": float(g["naked_net_pnl"].mean()),
                "naked_ci_lo": n_lo,
                "naked_ci_hi": n_hi,
                "win_rate_hedged": float((g["hedged_net_pnl"] > 0).mean()),
                "win_rate_naked": float((g["naked_net_pnl"] > 0).mean()),
                "median_hold_seconds": float(g["hold_seconds"].median()),
                "p90_hold_seconds": float(g["hold_seconds"].quantile(0.90)),
                "p95_hold_seconds": float(g["hold_seconds"].quantile(0.95)),
                "mean_entry_run_seconds": float(g["entry_run_seconds"].mean()),
                "mean_pm_pnl": float(g["pm_pnl"].mean()),
                "mean_hedge_pnl": float(g["hedge_pnl"].mean()),
                "mean_hedge_cost": float(g["hedge_cost"].mean()),
                "mean_pm_fee": float((g["pm_entry_fee"] + g["pm_exit_fee"]).mean()),
                "mean_funding_cost": float(g["funding_cost"].mean()),
                "top_market_share": float(g["market_slug"].value_counts(normalize=True).iloc[0]),
                "converged_share": float(g["exit_reason"].eq("converged").mean()),
                "max_hold_share": float(g["exit_reason"].eq("max_hold").mean()),
                "regime_exit_share": float(g["exit_reason"].eq("regime_exit").mean()),
            }
        )
    return pd.DataFrame(rows)


def select_row(summary: pd.DataFrame, pnl_prefix: str) -> pd.Series | None:
    if summary.empty:
        return None
    ci_col = f"{pnl_prefix}_ci_lo"
    mean_col = f"mean_{pnl_prefix}_pnl"
    eligible = summary[summary["n_trades"] >= MIN_TRADES_FOR_SELECTION].copy()
    if eligible.empty:
        return None
    positive = eligible[eligible[ci_col] > 0]
    if not positive.empty:
        return positive.sort_values([ci_col, mean_col, "n_trades"], ascending=[False, False, False]).iloc[0]
    return eligible.sort_values([ci_col, mean_col, "n_trades"], ascending=[False, False, False]).iloc[0]


def selected_trades(trades: pd.DataFrame, row: pd.Series | None) -> pd.DataFrame:
    if row is None or trades.empty:
        return pd.DataFrame()
    return trades[
        trades["latency_s"].eq(int(row["latency_s"]))
        & trades["entry_band"].eq(float(row["entry_band"]))
        & trades["exit_band"].eq(float(row["exit_band"]))
        & trades["persistence_k_s"].eq(int(row["persistence_k_s"]))
        & trades["max_hold_s"].eq(int(row["max_hold_s"]))
    ].copy()


def latency_table(trades: pd.DataFrame, row: pd.Series | None) -> pd.DataFrame:
    if row is None or trades.empty:
        return pd.DataFrame()
    mask = (
        trades["entry_band"].eq(float(row["entry_band"]))
        & trades["exit_band"].eq(float(row["exit_band"]))
        & trades["persistence_k_s"].eq(int(row["persistence_k_s"]))
        & trades["max_hold_s"].eq(int(row["max_hold_s"]))
    )
    rows = []
    for latency, g in trades[mask].groupby("latency_s", sort=True):
        h_lo, h_hi = bootstrap_mean(g, "hedged_net_pnl")
        n_lo, n_hi = bootstrap_mean(g, "naked_net_pnl")
        rows.append(
            {
                "latency_s": int(latency),
                "n_trades": int(len(g)),
                "mean_hedged_pnl": float(g["hedged_net_pnl"].mean()),
                "hedged_ci_lo": h_lo,
                "hedged_ci_hi": h_hi,
                "mean_naked_pnl": float(g["naked_net_pnl"].mean()),
                "naked_ci_lo": n_lo,
                "naked_ci_hi": n_hi,
                "median_hold_seconds": float(g["hold_seconds"].median()),
                "p95_hold_seconds": float(g["hold_seconds"].quantile(0.95)),
                "mean_hedge_cost": float(g["hedge_cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def top_rows(summary: pd.DataFrame, sort_prefix: str, n: int = 10) -> list[list[str]]:
    if summary.empty:
        return []
    ci_col = f"{sort_prefix}_ci_lo"
    mean_col = f"mean_{sort_prefix}_pnl"
    top = summary.sort_values([ci_col, mean_col, "n_trades"], ascending=[False, False, False]).head(n)
    rows: list[list[str]] = []
    for _, row in top.iterrows():
        rows.append(
            [
                str(int(row["latency_s"])),
                number(float(row["entry_band"]), 2),
                number(float(row["exit_band"]), 2),
                str(int(row["persistence_k_s"])),
                str(int(row["max_hold_s"])),
                str(int(row["n_trades"])),
                str(int(row["markets"])),
                cents(float(row["mean_hedged_pnl"])),
                ci_cents_text(float(row["hedged_ci_lo"]), float(row["hedged_ci_hi"])),
                cents(float(row["mean_naked_pnl"])),
                ci_cents_text(float(row["naked_ci_lo"]), float(row["naked_ci_hi"])),
                number(float(row["median_hold_seconds"]), 1),
                number(float(row["p95_hold_seconds"]), 1),
                pct(float(row["converged_share"])),
                pct(float(row["top_market_share"])),
            ]
        )
    return rows


def format_latency_rows(table: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in table.iterrows():
        rows.append(
            [
                str(int(row["latency_s"])),
                str(int(row["n_trades"])),
                cents(float(row["mean_hedged_pnl"])),
                ci_cents_text(float(row["hedged_ci_lo"]), float(row["hedged_ci_hi"])),
                cents(float(row["mean_naked_pnl"])),
                ci_cents_text(float(row["naked_ci_lo"]), float(row["naked_ci_hi"])),
                number(float(row["median_hold_seconds"]), 1),
                number(float(row["p95_hold_seconds"]), 1),
                cents(float(row["mean_hedge_cost"])),
            ]
        )
    return rows


def write_extended_ledger(trades: pd.DataFrame) -> None:
    if PREV_TRADES.exists():
        prev = pd.read_csv(PREV_TRADES)
        prev["strategy_phase"] = prev.get("strategy_phase", "v3h_original")
        combined = pd.concat([prev, trades], ignore_index=True, sort=False)
    else:
        combined = trades.copy()
    combined.to_csv(OUT_EXTENDED_LEDGER, index=False)


def panel_counts(panel: pd.DataFrame) -> dict[str, Any]:
    target = (
        panel["abs_z"].between(MIN_ABS_Z, MAX_ABS_Z, inclusive="both")
        & panel["seconds_to_expiry"].between(MIN_TAU_SECONDS, MAX_TAU_SECONDS, inclusive="both")
    )
    strict = panel["source_ok_strict"].fillna(False).astype(bool)
    static_ok = ~panel["large_static_basis_10c"].fillna(False).astype(bool)
    return {
        "rows_total": int(len(panel)),
        "rows_target": int(target.sum()),
        "rows_target_strict": int((target & strict).sum()),
        "rows_target_strict_static_ok": int((target & strict & static_ok).sum()),
        "markets_total": int(panel["market_slug"].nunique()),
        "markets_target_strict": int(panel.loc[target & strict, "market_slug"].nunique()),
        "static_large_share_target_strict": float((target & strict & ~static_ok).sum() / max(1, (target & strict).sum())),
    }


def write_note(panel: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    counts = panel_counts(panel)
    hedged_row = select_row(summary, "hedged")
    naked_row = select_row(summary, "naked")
    hedged_sel = selected_trades(trades, hedged_row)
    naked_sel = selected_trades(trades, naked_row)
    hedged_lat = latency_table(trades, hedged_row)
    naked_lat = latency_table(trades, naked_row)

    if hedged_row is None:
        headline = "No moderate-delta persistence-gated trades were generated."
        verdict = "The gate produced no evaluable sample."
    elif float(hedged_row["hedged_ci_lo"]) > 0:
        headline = "The moderate-delta persistence-gated hedged RV clears zero after costs in-sample."
        verdict = (
            f"Best hedged regime: latency {int(hedged_row['latency_s'])}s, entry "
            f"{number(float(hedged_row['entry_band']), 2)}, exit {number(float(hedged_row['exit_band']), 2)}, "
            f"k={int(hedged_row['persistence_k_s'])}s, max_hold={int(hedged_row['max_hold_s'])}s, "
            f"mean {cents(float(hedged_row['mean_hedged_pnl']))} CI "
            f"{ci_cents_text(float(hedged_row['hedged_ci_lo']), float(hedged_row['hedged_ci_hi']))}."
        )
    else:
        headline = "The moderate-delta persistence-gated hedged RV does not clear zero after costs."
        verdict = (
            f"Best hedged regime is still strictly negative: latency {int(hedged_row['latency_s'])}s, entry "
            f"{number(float(hedged_row['entry_band']), 2)}, exit {number(float(hedged_row['exit_band']), 2)}, "
            f"k={int(hedged_row['persistence_k_s'])}s, max_hold={int(hedged_row['max_hold_s'])}s, "
            f"mean {cents(float(hedged_row['mean_hedged_pnl']))} CI "
            f"{ci_cents_text(float(hedged_row['hedged_ci_lo']), float(hedged_row['hedged_ci_hi']))}."
        )

    naked_verdict = "No naked selected row."
    if naked_row is not None:
        naked_verdict = (
            f"Best naked counterpart: latency {int(naked_row['latency_s'])}s, entry "
            f"{number(float(naked_row['entry_band']), 2)}, exit {number(float(naked_row['exit_band']), 2)}, "
            f"k={int(naked_row['persistence_k_s'])}s, max_hold={int(naked_row['max_hold_s'])}s, "
            f"mean {cents(float(naked_row['mean_naked_pnl']))} CI "
            f"{ci_cents_text(float(naked_row['naked_ci_lo']), float(naked_row['naked_ci_hi']))}."
        )

    hold_text = "No selected hedged trades."
    if not hedged_sel.empty:
        med = float(hedged_sel["hold_seconds"].median())
        p90 = float(hedged_sel["hold_seconds"].quantile(0.90))
        p95 = float(hedged_sel["hold_seconds"].quantile(0.95))
        all_med = float(trades["hold_seconds"].median()) if not trades.empty else math.nan
        all_p95 = float(trades["hold_seconds"].quantile(0.95)) if not trades.empty else math.nan
        minute_scale = med >= 60.0
        hold_text = (
            f"Selected hedged hold distribution: median {number(med, 1)}s, p90 {number(p90, 1)}s, "
            f"p95 {number(p95, 1)}s. "
            f"Across all H2 trades, median hold is {number(all_med, 1)}s and p95 is {number(all_p95, 1)}s. "
            f"{'This is minute-scale on the selected median.' if minute_scale else 'The selected median is still not minute-scale, so the gate only partially removed short-lived blips.'}"
        )

    exit_rows = []
    if not hedged_sel.empty:
        for reason, share in hedged_sel["exit_reason"].value_counts(normalize=True).sort_index().items():
            exit_rows.append([str(reason), pct(float(share))])

    component_rows = []
    for label, row, sel in [("best_hedged", hedged_row, hedged_sel), ("best_naked", naked_row, naked_sel)]:
        if row is None or sel.empty:
            continue
        component_rows.append(
            [
                label,
                str(int(row["n_trades"])),
                cents(float(sel["pm_pnl"].mean())),
                cents(float(sel["hedge_pnl"].mean())),
                cents(float(sel["hedge_cost"].mean())),
                cents(float(sel["funding_cost"].mean())),
                cents(float((sel["pm_entry_fee"] + sel["pm_exit_fee"]).mean())),
                cents(float(sel["hedged_net_pnl"].mean())),
                cents(float(sel["naked_net_pnl"].mean())),
            ]
        )

    latency_comment = "Latency robustness could not be evaluated."
    if not hedged_lat.empty and {1, 10}.issubset(set(hedged_lat["latency_s"].astype(int))):
        lat1 = hedged_lat[hedged_lat["latency_s"].eq(1)].iloc[0]
        lat10 = hedged_lat[hedged_lat["latency_s"].eq(10)].iloc[0]
        latency_comment = (
            f"For the selected hedged parameter set, hedged mean changes from {cents(float(lat1['mean_hedged_pnl']))} "
            f"at 1s to {cents(float(lat10['mean_hedged_pnl']))} at 10s; naked changes from "
            f"{cents(float(lat1['mean_naked_pnl']))} to {cents(float(lat10['mean_naked_pnl']))}."
        )

    source_note = (
        f"Rows available: {counts['rows_target']:,} in the forced moderate/mid regime, "
        f"{counts['rows_target_strict']:,} after strict Chainlink-vs-Binance source filter, and "
        f"{counts['rows_target_strict_static_ok']:,} after excluding large static basis at the row level. "
        f"Static-large share inside target+strict is {pct(counts['static_large_share_target_strict'])}."
    )

    text = f"""# Block K3 v3h2 Persistence-Gated Findings

Generated: {datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")}

## Headline

{headline} {verdict}

{naked_verdict}

{hold_text}

This is the intended falsifier for K3 v3-H: it **only** enters `{TARGET_BUCKET}` (`|z|` in [{number(MIN_ABS_Z, 2)}, {number(MAX_ABS_Z, 2)}], tau in 30m-2h), requires the same-signed dynamic logit gap to persist for k in {PERSISTENCE_SECONDS}, excludes far/late/tail rows entirely, applies the strict source-basis filter, and excludes large static basis at entry. Costs include Polymarket taker on entry+exit, Binance hedge turnover at {number(BINANCE_HEDGE_COST_BPS, 1)}bp, and funding at {number(FUNDING_BPS_PER_8H, 1)}bp per 8h.

## Best Hedged Regimes

{markdown_table(["lat", "entry", "exit", "k_s", "max_s", "trades", "mkts", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "med_hold", "p95_hold", "conv", "top_mkt"], top_rows(summary, "hedged"))}

## Best Naked Regimes

{markdown_table(["lat", "entry", "exit", "k_s", "max_s", "trades", "mkts", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "med_hold", "p95_hold", "conv", "top_mkt"], top_rows(summary, "naked"))}

## Latency Robustness

Same selected hedged parameter set across action latencies.

{latency_comment}

{markdown_table(["latency_s", "trades", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "med_hold_s", "p95_hold_s", "hedge_cost"], format_latency_rows(hedged_lat))}

Same selected naked parameter set across action latencies.

{markdown_table(["latency_s", "trades", "hedged_mean", "hedged_CI", "naked_mean", "naked_CI", "med_hold_s", "p95_hold_s", "hedge_cost"], format_latency_rows(naked_lat))}

## Cost Components

{markdown_table(["selection", "trades", "pm_pnl", "hedge_pnl", "hedge_cost", "funding", "pm_fees", "hedged_net", "naked_net"], component_rows)}

## Exit Mix

Selected hedged regime exit reasons:

{markdown_table(["exit_reason", "share"], exit_rows)}

## Source And Regime Filter

{source_note}

The strict source filter is the same hard filter as v3-H: no Chainlink-vs-Binance direction disagreement and Binance settlement margin >= {number(SOURCE_MARGIN_BP, 1)}bp. Source-risk windows are not counted as alpha.

## Method

- Signal: `dynamic_logit_gap = (pm_logit - fair_logit) - causal_static_logit_gap`.
- Entry: buy `UP` when the gap is below `-entry_band`; buy `DOWN` when above `entry_band`; require the same sign to persist for `k` sampled seconds before the signal.
- Exit: convergence to the exit band, max hold, or forced flatten when the row leaves the moderate/mid regime.
- No overlap: one open trade per market/config; the next entry search resumes after the exit fill.
- Hedge: optional Binance digital-delta hedge is rebalanced every second; naked and hedged PnL are both reported.
- CI: cluster bootstrap by market, {BOOTSTRAP_SAMPLES} samples.

## Outputs

- H2 trade ledger: `data/analysis/csv_outputs/options_delta/k3v3h2_persistence_trades.csv`
- H2 summary grid: `data/analysis/csv_outputs/options_delta/k3v3h2_persistence_summary.csv`
- Extended v3-H ledger: `data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades_ext.csv`
- Feature cache reused: `data/analysis/cache/k3v3h_panel_features.parquet`
- Repro script: `scripts/dali_block_k3v3h2_persistence.py`
"""
    NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {NOTE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-base", action="store_true", help="Rebuild base parquet from K3 v2 CSV.")
    parser.add_argument("--refresh-features", action="store_true", help="Recompute v3-H feature cache.")
    args = parser.parse_args()

    panel = load_features(refresh_base=args.refresh_base, refresh_features=args.refresh_features)
    print(f"using feature cache {FEATURE_CACHE}", flush=True)
    trades = simulate_all(panel)
    if trades.empty:
        summary = pd.DataFrame()
    else:
        trades.to_csv(OUT_TRADES, index=False)
        summary = summarize_grid(trades)
        summary.to_csv(OUT_SUMMARY, index=False)
        write_extended_ledger(trades)
        print(f"wrote {OUT_TRADES}", flush=True)
        print(f"wrote {OUT_SUMMARY}", flush=True)
        print(f"wrote {OUT_EXTENDED_LEDGER}", flush=True)
    write_note(panel, trades, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
