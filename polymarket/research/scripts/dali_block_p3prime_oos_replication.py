"""P3' A0c out-of-sample replication of the P2 deep-book reversion whiff.

This intentionally replaces the earlier P3 fill-probability build. It only
re-tests the single P2 survivor family:

- passive fade-to-current weighted mid/microprice
- ofi_5s bottom-decile and both-tails
- geopolitics markets
- discovery-set deepest-depth regime
- W in {1, 10}, timeout in {5, 10, 30, 60}, target in {micro, half}

The A0/A0b feature parquet is treated as discovery. A0c is replayed into a
separate append-only feature parquet and never appended in place to the A1
feature file.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import family_category
from dali_block_a1_replay_batch import RunSpec, atomic_write_parquet, replay_run
from dali_block_p2_reversion import (
    display_path,
    execute_non_overlap,
    passive_entry_candidates,
)


ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
DISCOVERY_FEATURES = ANALYSIS / "block_a1_features.parquet"
A0C_FEATURES = ANALYSIS / "block_a0c_features.parquet"
A0C_RUN_DIR = ROOT / "data" / "live_clob" / "block_a0c" / "block_a0c_targeted_20260529_morning"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "p3prime_oos_replication.csv"
NOTE = NOTES / "block_p3prime_oos_findings.md"

FILL_WINDOWS = (1, 10)
TARGET_TYPES = ("micro_price", "half_to_micro_price")
TIMEOUT_SECONDS = (5, 10, 30, 60)
MIN_CI_N = 5
REPLICATION_MIN_N = 30
DEPLOYABLE_FILL_RATE = 0.02
BOOTSTRAP_CHUNK_SECONDS = 300


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def safe_text(value: object, max_len: int = 52) -> str:
    text = str(value if value is not None else "").replace("|", "/").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def active_a0c_capture_processes() -> list[str]:
    proc = subprocess.run(
        ["ps", "-axo", "pid,ppid,stat,etime,command"],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if "dali_a0c_finalize_when_done.py" in line:
            continue
        if "block_a0c_targeted_20260529_morning" in line and "dali_block_a0_capture.py" in line:
            lines.append(line.strip())
    return lines


def discovery_run_ids() -> list[str]:
    con = duckdb.connect()
    rows = con.execute(
        "SELECT DISTINCT run_id FROM read_parquet(?) ORDER BY 1",
        [str(DISCOVERY_FEATURES)],
    ).fetchall()
    con.close()
    return [str(row[0]) for row in rows]


def discovery_depth_threshold() -> float:
    con = duckdb.connect()
    query = """
        WITH base AS (
            SELECT
                run_id,
                market_id,
                best_bid_size + best_ask_size AS touch_depth
            FROM read_parquet(?)
            WHERE is_book_state_complete
              AND best_bid_size IS NOT NULL
              AND best_ask_size IS NOT NULL
              AND best_bid_size + best_ask_size > 0
        ),
        depth AS (
            SELECT
                *,
                avg(touch_depth) OVER (PARTITION BY run_id, market_id) AS market_mean_depth
            FROM base
        )
        SELECT quantile_cont(touch_depth / market_mean_depth, 0.90) AS q90_relative_depth
        FROM depth
        WHERE market_mean_depth > 0
    """
    value = con.execute(query, [str(DISCOVERY_FEATURES)]).fetchone()[0]
    con.close()
    if value is None or not np.isfinite(float(value)):
        raise SystemExit("could not compute discovery relative-depth q90")
    return float(value)


def maybe_replay_a0c_features(rebuild: bool = False) -> None:
    if A0C_FEATURES.exists() and not rebuild:
        return
    if not A0C_RUN_DIR.exists():
        raise SystemExit(f"A0c run directory not found: {display_path(A0C_RUN_DIR)}")
    active = active_a0c_capture_processes()
    if active:
        print(
            "A0c capture still active; replaying current completed snapshot into append-only feature parquet.",
            flush=True,
        )
        for line in active:
            print(f"active: {line}", flush=True)
    spec = RunSpec(run_id="a0c", run_dir=A0C_RUN_DIR)
    df = replay_run(spec, top_n=5)
    df = df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)
    atomic_write_parquet(df, A0C_FEATURES)
    print(f"wrote {display_path(A0C_FEATURES)} rows={len(df):,}", flush=True)


def load_oos_features(path: Path, depth_threshold: float) -> pd.DataFrame:
    cols = [
        "run_id",
        "received_at",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "is_book_state_complete",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_price",
        "trade_side",
        "last_trade_side",
        "trade_size",
    ]
    con = duckdb.connect()
    df = con.execute(f"SELECT {', '.join(cols)} FROM read_parquet(?)", [str(path)]).df()
    con.close()
    if df.empty:
        raise SystemExit(f"A0c feature panel is empty: {display_path(path)}")

    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    numeric = [
        "outcome_index",
        "best_bid",
        "best_bid_size",
        "best_ask",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
        "trade_price",
        "trade_size",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    df["market"] = df["run_id"] + ":" + df["market_id"]
    df["category"] = df["family"].map(family_category)
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df["touch_depth"] = df[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    df["spread_bps"] = np.where(
        df["mid"].gt(0) & df["spread"].notna(),
        df["spread"] / df["mid"] * 10_000.0,
        np.nan,
    )
    size_sum = df["best_bid_size"] + df["best_ask_size"]
    df["weighted_mid"] = np.where(
        size_sum.gt(0)
        & df["best_bid"].notna()
        & df["best_ask"].notna()
        & df["best_bid_size"].notna()
        & df["best_ask_size"].notna(),
        (df["best_ask"] * df["best_bid_size"] + df["best_bid"] * df["best_ask_size"]) / size_sum,
        df["mid"] + 0.5 * df["spread"] * df["tob_imbalance"],
    )
    df["weighted_mid"] = df["weighted_mid"].clip(lower=0.0, upper=1.0)
    df["trade_side_norm"] = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    market_depth = (
        df.groupby(["run_id", "market_id"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    df = df.merge(market_depth, on=["run_id", "market_id"], how="left")
    df["relative_depth"] = df["touch_depth"] / df["market_mean_depth"]
    df["depth_regime"] = np.where(
        df["relative_depth"].ge(depth_threshold),
        "discovery_depth_q90_deep",
        "not_deep",
    )
    return df.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)


def add_ofi_signal(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    groups = list(df.groupby(["run_id", "market_id", "asset_id"], sort=False))
    for idx, (_, group) in enumerate(groups, start=1):
        if idx % 20 == 0:
            print(f"ofi features {idx:,}/{len(groups):,}", flush=True)
        g = group.sort_values("received_at").copy()
        g["ofi_combined_event"] = g["ofi_combined_event"].fillna(0.0)
        mean_depth = float(g["market_mean_depth"].replace([np.inf, -np.inf], np.nan).mean())
        if not np.isfinite(mean_depth) or mean_depth <= 0:
            mean_depth = 1.0
        g = g.set_index("received_at", drop=False)
        g["signal_ofi_5s"] = (
            g["direction_factor"].to_numpy(dtype=float)
            * g["ofi_combined_event"].rolling("5s").sum().to_numpy(dtype=float)
            / mean_depth
        )
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def build_oos_events(df: pd.DataFrame) -> pd.DataFrame:
    quote_ok = (
        df["category"].eq("Geopolitics")
        & df["is_book_state_complete"]
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["weighted_mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["spread"].replace([np.inf, -np.inf], np.nan).gt(0)
        & df["signal_ofi_5s"].replace([np.inf, -np.inf], np.nan).notna()
        & df["signal_ofi_5s"].ne(0.0)
    )
    valid = df.loc[quote_ok].copy()
    if valid.empty:
        raise SystemExit("no valid A0c geopolitics OFI rows")

    base_cols = [
        "run_id",
        "market_id",
        "market",
        "received_at",
        "asset_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "category",
        "direction_factor",
        "best_bid",
        "best_ask",
        "spread",
        "spread_bps",
        "mid",
        "weighted_mid",
        "touch_depth",
        "relative_depth",
        "depth_regime",
        "signal_ofi_5s",
    ]
    pieces: list[pd.DataFrame] = []
    for _, group in valid[base_cols].groupby(["run_id", "market_id"], sort=False):
        if len(group) < 20:
            continue
        q10 = float(group["signal_ofi_5s"].quantile(0.10))
        q90 = float(group["signal_ofi_5s"].quantile(0.90))
        tails = group[group["signal_ofi_5s"].le(q10) | group["signal_ofi_5s"].ge(q90)].copy()
        tails = tails[tails["depth_regime"].eq("discovery_depth_q90_deep")].copy()
        if tails.empty:
            continue
        tails["signal_variant"] = "ofi_5s"
        tails["signal_tail"] = np.where(
            tails["signal_ofi_5s"].le(q10),
            "bottom_decile",
            "top_decile",
        )
        tails["signal_value"] = tails["signal_ofi_5s"].astype(float)
        tails["abs_signal"] = tails["signal_value"].abs()
        tails["signal_decile_threshold_low"] = q10
        tails["signal_decile_threshold_high"] = q90
        tails["token_side"] = -np.sign(tails["signal_value"].to_numpy(dtype=float)) * tails[
            "direction_factor"
        ].to_numpy(dtype=float)
        tails = tails[tails["token_side"].isin([-1.0, 1.0])].copy()
        pieces.append(tails.drop(columns=["signal_ofi_5s"]))
    if not pieces:
        raise SystemExit("no A0c deep-depth geopolitics tail events")
    events = pd.concat(pieces, ignore_index=True)
    events["event_time_ns"] = events["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events.sort_values(["market", "event_time_ns", "asset_id"]).reset_index(drop=True)


def augment_tails(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return pd.concat([df, df.assign(signal_tail="both_tails")], ignore_index=True)


def normal_block_ci(rows: pd.DataFrame) -> tuple[float, float]:
    clean = rows[["entry_time_ns_int", "pnl_bps"]].dropna().copy()
    clean = clean[np.isfinite(clean["pnl_bps"])]
    if len(clean) < MIN_CI_N:
        return math.nan, math.nan
    mean = float(clean["pnl_bps"].mean())
    elapsed = (clean["entry_time_ns_int"] - clean["entry_time_ns_int"].min()) / 1_000_000_000.0
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    block_means = pd.Series(clean["pnl_bps"].to_numpy(dtype=float)).groupby(block_id).mean()
    if len(block_means) >= 2:
        se = float(block_means.std(ddof=1) / math.sqrt(len(block_means)))
    else:
        se = float(clean["pnl_bps"].std(ddof=1) / math.sqrt(len(clean)))
    if not np.isfinite(se):
        return math.nan, math.nan
    return mean - 1.96 * se, mean + 1.96 * se


def count_groups(rows: pd.DataFrame, value_name: str) -> pd.DataFrame:
    src = augment_tails(rows)
    configs = [
        ("a0c_geopolitics_deep_all", []),
        ("a0c_geopolitics_deep_market", ["market", "slug", "family"]),
    ]
    parts = []
    for segment_type, cols in configs:
        group_cols = ["signal_variant", "signal_tail", *cols]
        g = src.groupby(group_cols, dropna=False).size().reset_index(name=value_name)
        g["segment_type"] = segment_type
        g["market"] = g["market"] if "market" in g else "ALL"
        g["slug"] = g["slug"] if "slug" in g else "ALL"
        g["family"] = g["family"] if "family" in g else "Geopolitics"
        parts.append(g[["signal_variant", "signal_tail", "segment_type", "market", "slug", "family", value_name]])
    return pd.concat(parts, ignore_index=True)


GROUP_KEYS = ["signal_variant", "signal_tail", "segment_type", "market", "slug", "family"]


def stat_groups(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=GROUP_KEYS)
    src = augment_tails(rows)
    configs = [
        ("a0c_geopolitics_deep_all", []),
        ("a0c_geopolitics_deep_market", ["market", "slug", "family"]),
    ]
    out_rows: list[dict[str, object]] = []
    for segment_type, cols in configs:
        group_cols = ["signal_variant", "signal_tail", *cols]
        for key, sub in src.groupby(group_cols, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_map = dict(zip(group_cols, key_tuple, strict=True))
            ci_lo, ci_hi = normal_block_ci(sub)
            counts = sub["exit_reason"].value_counts().to_dict()
            out_rows.append(
                {
                    "signal_variant": key_map["signal_variant"],
                    "signal_tail": key_map["signal_tail"],
                    "segment_type": segment_type,
                    "market": key_map.get("market", "ALL"),
                    "slug": key_map.get("slug", "ALL"),
                    "family": key_map.get("family", "Geopolitics"),
                    "n_executed": int(len(sub)),
                    "mean_pnl_bps": float(sub["pnl_bps"].mean()),
                    "median_pnl_bps": float(sub["pnl_bps"].median()),
                    "win_rate": float(sub["pnl_bps"].gt(0).mean()),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "mean_target_edge_bps": float(sub["target_edge_bps"].mean()),
                    "mean_gross_bps": float(sub["gross_bps"].mean()),
                    "mean_entry_rebate_bps": float(sub["entry_rebate_bps"].mean()),
                    "mean_exit_fee_bps": float(sub["exit_fee_bps"].mean()),
                    "mean_hold_seconds": float(sub["hold_seconds"].mean()),
                    "target_reached_rate": float(sub["exit_reason"].eq("target_reached").mean()),
                    "adverse_stop_rate": float(sub["exit_reason"].eq("adverse_stop").mean()),
                    "timeout_rate": float(sub["exit_reason"].eq("timeout").mean()),
                    "n_target_reached": int(counts.get("target_reached", 0)),
                    "n_adverse_stop": int(counts.get("adverse_stop", 0)),
                    "n_timeout": int(counts.get("timeout", 0)),
                }
            )
    return pd.DataFrame(out_rows)


def summarize_combo(
    signals: pd.DataFrame,
    entries: pd.DataFrame,
    executed: pd.DataFrame,
    *,
    fill_window: int,
    target_type: str,
    timeout_sec: int,
    depth_threshold: float,
) -> pd.DataFrame:
    denominators = count_groups(signals, "n_signal_events")
    filled = count_groups(entries[entries["entry_filled"]], "n_entry_filled_raw")
    stats = stat_groups(executed)
    out = denominators.merge(filled, on=GROUP_KEYS, how="left").merge(stats, on=GROUP_KEYS, how="left")
    out["n_entry_filled_raw"] = out["n_entry_filled_raw"].fillna(0).astype(int)
    out["n_executed"] = out["n_executed"].fillna(0).astype(int)
    out["raw_entry_fill_rate"] = out["n_entry_filled_raw"] / out["n_signal_events"].replace(0, np.nan)
    out["executed_fill_rate"] = out["n_executed"] / out["n_signal_events"].replace(0, np.nan)
    out["nonoverlap_keep_rate_after_fill"] = out["n_executed"] / out["n_entry_filled_raw"].replace(0, np.nan)
    out["sample_split"] = "oos_a0c"
    out["execution_mode"] = "P"
    out["fill_window_sec"] = fill_window
    out["target_type"] = target_type
    out["timeout_sec"] = timeout_sec
    out["depth_filter"] = "discovery_relative_depth_q90"
    out["discovery_relative_depth_q90"] = depth_threshold
    out["ci_positive"] = out["ci_lo"].gt(0) & out["n_executed"].ge(MIN_CI_N)
    out["replication_n_pass"] = out["n_executed"].ge(REPLICATION_MIN_N)
    out["deployable_fill_pass"] = out["executed_fill_rate"].ge(DEPLOYABLE_FILL_RATE)
    out["replication_pass"] = out["ci_positive"] & out["replication_n_pass"] & out["deployable_fill_pass"]
    out["verdict"] = np.where(
        out["replication_pass"],
        "replicates",
        np.where(
            out["n_executed"].lt(REPLICATION_MIN_N),
            "fails_n_lt_30",
            np.where(out["executed_fill_rate"].lt(DEPLOYABLE_FILL_RATE), "fails_fill_lt_2pct", "fails_ci"),
        ),
    )
    front = [
        "sample_split",
        "execution_mode",
        "signal_variant",
        "signal_tail",
        "segment_type",
        "market",
        "slug",
        "family",
        "fill_window_sec",
        "target_type",
        "timeout_sec",
        "depth_filter",
        "discovery_relative_depth_q90",
        "n_signal_events",
        "n_entry_filled_raw",
        "n_executed",
        "raw_entry_fill_rate",
        "executed_fill_rate",
        "nonoverlap_keep_rate_after_fill",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "ci_positive",
        "replication_n_pass",
        "deployable_fill_pass",
        "replication_pass",
        "verdict",
    ]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest]


def run_oos_grid(df: pd.DataFrame, events: pd.DataFrame, depth_threshold: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    market_groups = {market: sub.copy() for market, sub in df.groupby("market", sort=False)}
    entries_by_window: dict[int, pd.DataFrame] = {}
    for fill_window in FILL_WINDOWS:
        parts = []
        for market, market_events in events.groupby("market", sort=False):
            parts.append(passive_entry_candidates(market_events, market_groups[market], fill_window))
        entries_by_window[fill_window] = pd.concat(parts, ignore_index=True) if parts else events.iloc[0:0]

    total = len(FILL_WINDOWS) * len(TARGET_TYPES) * len(TIMEOUT_SECONDS)
    idx = 0
    for fill_window, entries in entries_by_window.items():
        for target_type in TARGET_TYPES:
            for timeout_sec in TIMEOUT_SECONDS:
                idx += 1
                print(
                    f"P3' OOS grid {idx}/{total}: W={fill_window}s target={target_type} timeout={timeout_sec}s",
                    flush=True,
                )
                executed_parts = []
                for market, market_entries in entries.groupby("market", sort=False):
                    executed = execute_non_overlap(
                        market_entries,
                        market_groups[market],
                        target_type,
                        timeout_sec,
                        "P",
                    )
                    if not executed.empty:
                        executed_parts.append(executed)
                executed_all = pd.concat(executed_parts, ignore_index=True) if executed_parts else entries.iloc[0:0]
                rows.append(
                    summarize_combo(
                        events,
                        entries,
                        executed_all,
                        fill_window=fill_window,
                        target_type=target_type,
                        timeout_sec=timeout_sec,
                        depth_threshold=depth_threshold,
                    )
                )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def top_table(results: pd.DataFrame, limit: int = 16) -> str:
    sub = results[results["n_executed"].gt(0)].copy()
    if sub.empty:
        return "_No executed OOS cells._"
    sub = sub.sort_values(["replication_pass", "ci_lo", "mean_pnl_bps", "executed_fill_rate"], ascending=False).head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                row.segment_type.replace("a0c_geopolitics_deep_", ""),
                safe_text(row.market, 18),
                safe_text(row.slug, 34),
                row.signal_tail,
                f"W={int(row.fill_window_sec)}",
                str(row.target_type).replace("_", " "),
                f"{int(row.timeout_sec)}s",
                f"{int(row.n_signal_events):,}",
                f"{int(row.n_executed):,}",
                pct(float(row.executed_fill_rate)),
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                row.verdict,
            ]
        )
    return markdown_table(
        ["segment", "market", "slug", "tail", "fill", "target", "timeout", "signals", "exec", "fill rate", "mean", "CI", "verdict"],
        rows,
    )


def write_note(
    results: pd.DataFrame,
    *,
    run_ids: list[str],
    depth_threshold: float,
    oos_rows: int,
    oos_markets: int,
    event_count: int,
    active_process_count: int,
) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    candidates = results[results["n_executed"].ge(MIN_CI_N)].copy()
    pass_rows = candidates[candidates["replication_pass"]].copy()
    ci_pos = candidates[candidates["ci_positive"]].copy()
    robust_ci = candidates[candidates["ci_positive"] & candidates["replication_n_pass"]].copy()
    deployable_ci = robust_ci[robust_ci["deployable_fill_pass"]].copy()
    best = (
        results[results["n_executed"].gt(0)]
        .sort_values(["ci_lo", "mean_pnl_bps", "executed_fill_rate"], ascending=False)
        .head(1)
    )
    best_text = "No A0c deep-book geopolitics executions were generated."
    if not best.empty:
        r = best.iloc[0]
        best_text = (
            f"Best OOS row by CI lower bound: `{safe_text(r['segment_type'])}` / `{safe_text(r['market'])}` "
            f"/ `{safe_text(r['signal_tail'])}`, W={int(r['fill_window_sec'])}s, target `{r['target_type']}`, "
            f"timeout={int(r['timeout_sec'])}s, n={int(r['n_executed'])}, fill "
            f"{pct(float(r['executed_fill_rate']))}, mean {bps(float(r['mean_pnl_bps']))}, "
            f"CI [{bps(float(r['ci_lo']))}, {bps(float(r['ci_hi']))}], verdict `{r['verdict']}`."
        )

    if pass_rows.empty:
        headline = (
            "Fails OOS: no A0c deep-book geopolitics cell meets CI lower > 0, n >= 30, "
            "and fill rate >= 2%. Under the preregistered rule, the local Dali microstructure "
            "signal is closed; go to P6."
        )
    else:
        headline = (
            f"Replicates: {len(pass_rows):,} A0c deep-book geopolitics rows meet CI lower > 0, "
            f"n >= 30, and fill rate >= 2%. P4 depth/MLOFI features are justified."
        )

    note = f"""---
tags: [dali, p3prime, oos, micro-price, reversion, results]
---

# P3' A0c OOS Reversion Replication Findings

## Headline

{headline}

{best_text}

## Decision Rule

- Preregistered replication bar: CI lower > 0, n >= {REPLICATION_MIN_N}, and fill rate >= {pct(DEPLOYABLE_FILL_RATE)} on A0c-only deep-book geopolitics cells.
- Rows clearing CI lower > 0: {len(ci_pos):,}
- Rows clearing CI lower > 0 and n >= {REPLICATION_MIN_N}: {len(robust_ci):,}
- Rows clearing CI lower > 0, n >= {REPLICATION_MIN_N}, and fill >= {pct(DEPLOYABLE_FILL_RATE)}: {len(deployable_ci):,}

## OOS Top Rows

{top_table(results)}

## Method

- Discovery feature panel: `{display_path(DISCOVERY_FEATURES)}` with run IDs `{', '.join(run_ids)}`. Confirmed A0c is absent.
- A0c feature panel: `{display_path(A0C_FEATURES)}`. This is append-only separate output; `block_a1_features.parquet` was not mutated.
- A0c rows loaded: {oos_rows:,} across {oos_markets:,} markets. A0c deep geopolitics tail events tested: {event_count:,}.
- Active A0c capture processes observed at replay time: {active_process_count}. If nonzero, the feature parquet is a snapshot of available shards at replay time.
- Deep-book filter: A0/A0b discovery relative-depth q90 = `{depth_threshold:.6g}`; A0c rows must satisfy `relative_depth >= q90`.
- Signal: `ofi_5s = direction_factor * rolling_sum(ofi_combined_event, 5s) / market_mean_depth`.
- Trigger: per-market OFI bottom/top deciles; the preregistered decision focuses on `bottom_decile` and `both_tails`. The grid is not retuned on A0c.
- Execution: passive fade at the touch, P2 fill proxy, entry maker rebate, taker exit to bid/ask, non-overlap after actual fill.
- Grid: W in {FILL_WINDOWS}, target in {TARGET_TYPES}, timeout in {TIMEOUT_SECONDS}.
- CI: normal interval over contiguous 300s block means of non-overlap executed PnL.

## Output

- `{display_path(OUT_CSV)}`
"""
    NOTE.write_text(note, encoding="utf-8")


def parse_args() -> tuple[bool]:
    rebuild = "--rebuild-a0c-features" in sys.argv
    return (rebuild,)


def main() -> int:
    (rebuild,) = parse_args()
    if not DISCOVERY_FEATURES.exists():
        raise SystemExit(f"missing discovery feature parquet: {display_path(DISCOVERY_FEATURES)}")
    run_ids = discovery_run_ids()
    if any(str(run_id).lower().startswith("a0c") for run_id in run_ids):
        raise SystemExit(
            f"A0c already present in discovery feature parquet; strict OOS split broken: {run_ids}"
        )
    print(f"discovery run IDs: {run_ids}", flush=True)
    depth_threshold = discovery_depth_threshold()
    print(f"discovery relative-depth q90: {depth_threshold:.6g}", flush=True)

    active_before = active_a0c_capture_processes()
    maybe_replay_a0c_features(rebuild=rebuild)
    print(f"loading A0c features from {display_path(A0C_FEATURES)}", flush=True)
    df = load_oos_features(A0C_FEATURES, depth_threshold)
    print(f"A0c feature rows: {len(df):,}; markets: {df['market'].nunique():,}", flush=True)
    df = add_ofi_signal(df)
    events = build_oos_events(df)
    print(f"A0c deep geopolitics tail events: {len(events):,}", flush=True)
    results = run_oos_grid(df, events, depth_threshold)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(
        results,
        run_ids=run_ids,
        depth_threshold=depth_threshold,
        oos_rows=len(df),
        oos_markets=df["market"].nunique(),
        event_count=len(events),
        active_process_count=len(active_before),
    )
    print(f"results: {display_path(OUT_CSV)} ({len(results):,} rows)")
    print(f"note: {display_path(NOTE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
