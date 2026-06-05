"""A0c strict holdout retest for the Dali cells that looked promising.

This script mutates only the analysis feature parquet requested by the prompt:
``data/analysis/block_a1_features.parquet`` is extended append-only with A0c
main and crypto-roll runs when those run IDs are absent. All model/threshold
calibration remains restricted to A0/A0b discovery rows.
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category
from dali_block_a1_replay_batch import atomic_write_parquet, display_path, enrich_features, scan_shard_metadata
from dali_block_p2_reversion import execute_non_overlap, passive_entry_candidates
from scripts.dali_clob_replay_features import replay


ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
MAIN_FEATURES = ANALYSIS / "block_a0c_features.parquet"
ROLL_FEATURES = ANALYSIS / "block_a0c_roll_features.parquet"
MAIN_RUN_DIR = ROOT / "data" / "live_clob" / "block_a0c" / "block_a0c_targeted_20260529_morning"
ROLL_RUN_DIR = ROOT / "data" / "live_clob" / "block_a0c_crypto_roll" / "block_a0c_crypto_roll_20260529_morning"
MAIN_FINAL_NOTE = NOTES / "block_a0c_capture_status_final.md"
ROLL_FINAL_NOTE = NOTES / "block_a0c_crypto_roll_status_final.md"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "a0c_holdout_retest_surface.csv"
NOTE = NOTES / "block_a0c_holdout_retest_findings.md"

DISCOVERY_RUNS = ("a0", "a0b")
MAIN_RUN_ID = "a0c"
ROLL_RUN_ID = "a0c_roll"
STALE_BOOK_MAX_SECONDS = 5.0
BOOTSTRAP_CHUNK_SECONDS = 300
BOOTSTRAP_SAMPLES = 500
RNG_SEED = 20260530
A1_3_TOB_HIT_POINT = 0.737
A_BAR_MIN_N = 30
A_BAR_MIN_WINDOWS = 3
B_BAR_MIN_N = 30
B_BAR_MIN_FILL = 0.02
FOUR_HOURS_SECONDS = 4 * 60 * 60
EXCLUDE_SLUG_FRAGMENT = "will-jd-vance"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def safe_text(value: object, max_len: int = 48) -> str:
    text = str(value if value is not None else "").replace("|", "/").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def stable_seed(*parts: object) -> int:
    return RNG_SEED + int(zlib.crc32("|".join(map(str, parts)).encode("utf-8")) % 100_000)


def ns_from_ts(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[ns]").astype("int64")


def fee_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.0, 1.0))
    return float(params["fee_rate"] * p * (1.0 - p))


def raw_jsonl_shards(run_dir: Path, *, recursive: bool) -> list[Path]:
    iterator = run_dir.rglob("*.jsonl") if recursive else run_dir.glob("*.jsonl")
    return sorted(
        path
        for path in iterator
        if path.name not in {"capture_gaps.jsonl", "roll_supervisor.jsonl"}
    )


def scan_raw_event_counts(run_dir: Path, *, recursive: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in raw_jsonl_shards(run_dir, recursive=recursive):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                counts[str(rec.get("event_type") or "unknown")] += 1
    return counts


def event_counts_from_note(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*?)\n```", text, flags=re.S)
    if not match:
        return {}
    data = json.loads(match.group(1))
    return {str(k): int(v) for k, v in data.items()}


def replay_paths(run_id: str, run_dir: Path, *, recursive: bool, out: Path, top_n: int = 5) -> pd.DataFrame:
    shards = raw_jsonl_shards(run_dir, recursive=recursive)
    if not shards:
        raise SystemExit(f"no JSONL shards under {display_path(run_dir)}")
    pieces: list[pd.DataFrame] = []
    for idx, shard in enumerate(shards, start=1):
        metadata, resolved_at = scan_shard_metadata(shard)
        df = replay(shard, top_n=top_n)
        df = enrich_features(
            df,
            run_id=run_id,
            shard=shard,
            metadata=metadata,
            resolved_at=resolved_at,
        )
        pieces.append(df)
        print(f"[{run_id}] {idx:02d}/{len(shards):02d} {display_path(shard)} -> {len(df):,} rows", flush=True)
    out_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    out_df = out_df.sort_values(["run_id", "asset_id", "received_at"]).reset_index(drop=True)
    atomic_write_parquet(out_df, out)
    print(f"wrote {display_path(out)} rows={len(out_df):,}", flush=True)
    return out_df


def parquet_run_counts(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        """
        SELECT
            run_id,
            count(*) AS n_rows,
            count(distinct market_id) AS n_markets,
            min(received_at) AS first_ts,
            max(received_at) AS last_ts
        FROM read_parquet(?)
        GROUP BY 1
        ORDER BY 1
        """,
        [str(path)],
    ).df()
    con.close()
    return df


def append_missing_a0c_runs() -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if not FEATURES.exists():
        raise SystemExit(f"missing discovery feature panel: {display_path(FEATURES)}")

    con = duckdb.connect()
    existing_runs = {
        str(row[0])
        for row in con.execute("SELECT DISTINCT run_id FROM read_parquet(?) ORDER BY 1", [str(FEATURES)]).fetchall()
    }
    con.close()

    if MAIN_RUN_ID not in existing_runs and not MAIN_FEATURES.exists():
        replay_paths(MAIN_RUN_ID, MAIN_RUN_DIR, recursive=False, out=MAIN_FEATURES)
    if ROLL_RUN_ID not in existing_runs and not ROLL_FEATURES.exists():
        replay_paths(ROLL_RUN_ID, ROLL_RUN_DIR, recursive=True, out=ROLL_FEATURES)

    append_paths: list[Path] = []
    if MAIN_RUN_ID not in existing_runs:
        append_paths.append(MAIN_FEATURES)
    if ROLL_RUN_ID not in existing_runs:
        append_paths.append(ROLL_FEATURES)

    if append_paths:
        union_sql = [f"SELECT * FROM read_parquet('{FEATURES}')"]
        union_sql.extend(f"SELECT * FROM read_parquet('{path}')" for path in append_paths)
        query = "\nUNION ALL\n".join(union_sql)
        with tempfile.NamedTemporaryFile(dir=FEATURES.parent, suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            con = duckdb.connect()
            con.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT PARQUET)")
            con.close()
            tmp_path.replace(FEATURES)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"appended {', '.join(path.name for path in append_paths)} into {display_path(FEATURES)}", flush=True)
    else:
        print("canonical feature panel already contains a0c/a0c_roll; no append needed", flush=True)

    confirmations = {
        "main_final_note": event_counts_from_note(MAIN_FINAL_NOTE),
        "main_raw_scan": dict(scan_raw_event_counts(MAIN_RUN_DIR, recursive=False)),
        "roll_final_note": event_counts_from_note(ROLL_FINAL_NOTE),
        "roll_raw_scan": dict(scan_raw_event_counts(ROLL_RUN_DIR, recursive=True)),
    }
    return parquet_run_counts(FEATURES), confirmations


def load_feature_rows(run_ids: tuple[str, ...]) -> pd.DataFrame:
    cols = [
        "run_id",
        "shard",
        "received_at",
        "exchange_ts",
        "event_type",
        "asset_id",
        "market_id",
        "family",
        "slug",
        "question",
        "outcome_index",
        "is_book_state_complete",
        "book_staleness_seconds",
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
    placeholders = ", ".join(["?"] * len(run_ids))
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT {', '.join(cols)}
        FROM read_parquet(?)
        WHERE run_id IN ({placeholders})
          AND coalesce(lower(slug), '') NOT LIKE ?
        """,
        [str(FEATURES), *run_ids, f"%{EXCLUDE_SLUG_FRAGMENT}%"],
    ).df()
    con.close()
    if df.empty:
        raise SystemExit(f"no feature rows loaded for {run_ids}")
    return prepare_features(df)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["received_at"] = pd.to_datetime(out["received_at"], utc=True)
    out["exchange_ts"] = pd.to_datetime(out["exchange_ts"], utc=True, errors="coerce")
    out["event_ts"] = out["exchange_ts"].where(out["exchange_ts"].notna(), out["received_at"])
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question", "trade_side", "last_trade_side"):
        out[col] = out[col].fillna("").astype(str)
    numeric = [
        "outcome_index",
        "book_staleness_seconds",
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
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_book_state_complete"] = out["is_book_state_complete"].fillna(False).astype(bool)
    out["market"] = out["run_id"] + ":" + out["market_id"]
    out["category"] = out["family"].map(family_category)
    out["direction_factor"] = np.where(out["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    out["trade_side_norm"] = (
        out["trade_side"]
        .where(out["trade_side"].ne(""), out["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    out["touch_depth"] = out[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
    depth = (
        out.groupby(["run_id", "market_id"], as_index=False)["touch_depth"]
        .mean()
        .rename(columns={"touch_depth": "market_mean_depth"})
    )
    out = out.merge(depth, on=["run_id", "market_id"], how="left")
    out["relative_depth"] = out["touch_depth"] / out["market_mean_depth"]
    size_sum = out["best_bid_size"] + out["best_ask_size"]
    out["weighted_mid"] = np.where(
        size_sum.gt(0),
        (out["best_ask"] * out["best_bid_size"] + out["best_bid"] * out["best_ask_size"]) / size_sum,
        np.nan,
    )
    return out.sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    groups = list(df.groupby(["run_id", "market_id", "asset_id"], sort=False))
    for idx, (_, group) in enumerate(groups, start=1):
        if idx % 40 == 0:
            print(f"signal features {idx}/{len(groups)}", flush=True)
        g = group.sort_values("event_ts").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g = g.set_index("event_ts", drop=False)
        ofi = g["ofi_combined_event"].fillna(0.0).rolling("5s").sum()
        denom = g["market_mean_depth"].replace(0.0, np.nan)
        g["ofi_5s"] = g["direction_factor"] * ofi / denom
        g["abs_tob_imbalance_level"] = g["tob_imbalance_level"].abs()
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True).sort_values(["run_id", "market_id", "asset_id", "event_ts"]).reset_index(drop=True)


def valid_quote_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_book_state_complete"]
        & df["event_ts"].notna()
        & df["book_staleness_seconds"].le(STALE_BOOK_MAX_SECONDS)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_bid"].ge(0.0)
        & df["best_ask"].le(1.0)
        & df["best_ask"].ge(df["best_bid"])
    )


def discovery_thresholds() -> tuple[pd.DataFrame, float]:
    discovery = add_signals(load_feature_rows(DISCOVERY_RUNS))
    valid = discovery[valid_quote_mask(discovery)].copy()
    crypto = valid[valid["family"].eq("crypto_4h_up_down")]
    geo = valid[valid["category"].eq("Geopolitics")]
    rows: list[dict[str, object]] = []
    for scope, sub in (("crypto_4h_up_down", crypto), ("geopolitics", geo), ("all", valid)):
        for signal in ("ofi_5s", "tob_imbalance_level"):
            vals = sub[signal].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "threshold_scope": scope,
                    "signal_variant": signal,
                    "q10": float(vals.quantile(0.10)),
                    "q90": float(vals.quantile(0.90)),
                    "abs_q90": float(vals.abs().quantile(0.90)),
                    "n_threshold_rows": int(len(vals)),
                }
            )
    depth_base = valid[valid["relative_depth"].replace([np.inf, -np.inf], np.nan).notna()]
    depth_q90 = float(depth_base["relative_depth"].quantile(0.90))
    return pd.DataFrame(rows), depth_q90


def threshold_row(thresholds: pd.DataFrame, scope: str, signal: str) -> pd.Series:
    match = thresholds[thresholds["threshold_scope"].eq(scope) & thresholds["signal_variant"].eq(signal)]
    if match.empty:
        match = thresholds[thresholds["threshold_scope"].eq("all") & thresholds["signal_variant"].eq(signal)]
    if match.empty:
        raise SystemExit(f"missing discovery threshold for {scope}/{signal}")
    return match.iloc[0]


def crypto_window_metadata(slug: str) -> tuple[str, int, int] | None:
    match = re.match(r"^(btc|eth|sol)-updown-4h-(\d+)$", str(slug))
    if not match:
        return None
    start = int(match.group(2))
    return match.group(1), start, start + FOUR_HOURS_SECONDS


def crypto_trade_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_id, market_id), group in df.groupby(["run_id", "market_id"], sort=False):
        slug = str(group["slug"].replace("", np.nan).dropna().iloc[0]) if group["slug"].astype(bool).any() else ""
        meta = crypto_window_metadata(slug)
        if meta is None:
            continue
        symbol, start, end = meta
        rows.append(
            {
                "run_id": run_id,
                "market_id": market_id,
                "market": f"{run_id}:{market_id}",
                "slug": slug,
                "symbol": symbol,
                "window_start_epoch": start,
                "window_end_epoch": end,
                "last_trade_events": int(group["event_type"].eq("last_trade_price").sum()),
            }
        )
    return pd.DataFrame(rows)


def quote_arrays_by_asset(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for asset_id, group in df[valid_quote_mask(df)].groupby("asset_id", sort=False):
        g = group.sort_values("event_ts").reset_index(drop=True)
        out[str(asset_id)] = {
            "times": ns_from_ts(g["event_ts"]),
            "bid": g["best_bid"].to_numpy(dtype=float),
            "ask": g["best_ask"].to_numpy(dtype=float),
            "mid": g["mid"].to_numpy(dtype=float),
            "direction_factor": g["direction_factor"].to_numpy(dtype=float),
        }
    return out


def target_exit_index(times: np.ndarray, target_ns: int) -> int | None:
    idx = int(np.searchsorted(times, target_ns, side="right") - 1)
    if idx < 0:
        return None
    return idx


def binary_candidate_events(market_df: pd.DataFrame, signal: str, tail: str, thr: pd.Series) -> pd.DataFrame:
    sub = market_df[valid_quote_mask(market_df) & market_df[signal].replace([np.inf, -np.inf], np.nan).notna()].copy()
    if sub.empty:
        return sub
    if tail == "top_decile":
        sub = sub[sub[signal].ge(float(thr["q90"]))].copy()
    elif tail == "bottom_decile":
        sub = sub[sub[signal].le(float(thr["q10"]))].copy()
    else:
        raise ValueError(tail)
    if sub.empty:
        return sub
    sub["signal_value"] = sub[signal].astype(float)
    sub["signal_tail"] = tail
    sub["signal_variant"] = signal
    sub["signal_sign"] = np.sign(sub["signal_value"])
    sub = sub[sub["signal_sign"].isin([-1.0, 1.0])].copy()
    sub["token_side"] = sub["signal_sign"] * sub["direction_factor"]
    sub["abs_signal"] = sub["signal_value"].abs()
    sub["event_time_ns"] = ns_from_ts(sub["event_ts"])
    return (
        sub.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False])
        .drop_duplicates(["event_time_ns"], keep="first")
        .reset_index(drop=True)
    )


def simulate_binary_trades(
    market_df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_type: str,
    window_end_epoch: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    arrays = quote_arrays_by_asset(market_df)
    category = family_category(str(market_df["family"].replace("", np.nan).dropna().iloc[0]))
    next_available = -1
    trades: list[dict[str, object]] = []
    window_end_ns = window_end_epoch * 1_000_000_000
    for row in events.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False]).itertuples(index=False):
        entry_ns = int(row.event_time_ns)
        if entry_ns <= next_available:
            continue
        target_ns = entry_ns + 300 * 1_000_000_000 if horizon_type == "fixed_300s" else window_end_ns
        if target_ns <= entry_ns:
            continue
        state = arrays.get(str(row.asset_id))
        if state is None:
            continue
        times = state["times"]
        entry_idx = int(np.searchsorted(times, entry_ns, side="right") - 1)
        exit_idx = target_exit_index(times, target_ns)
        if entry_idx < 0 or exit_idx is None or exit_idx <= entry_idx:
            continue
        side = float(row.token_side)
        entry_px = float(state["ask"][entry_idx] if side > 0 else state["bid"][entry_idx])
        exit_px = float(state["bid"][exit_idx] if side > 0 else state["ask"][exit_idx])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
            continue
        gross = side * (exit_px - entry_px)
        fees = fee_amount(category, entry_px) + fee_amount(category, exit_px)
        pnl = (gross - fees) / float(np.clip(entry_px, 0.01, 0.99)) * 10_000.0
        exit_ns = int(times[exit_idx]) if horizon_type == "fixed_300s" else target_ns
        trades.append(
            {
                "entry_time_ns": entry_ns,
                "entry_time": pd.Timestamp(row.event_ts),
                "exit_time_ns": exit_ns,
                "hold_seconds": (target_ns - entry_ns) / 1_000_000_000.0,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "pnl_bps": float(pnl),
                "signal_value": float(row.signal_value),
            }
        )
        next_available = target_ns
    return pd.DataFrame(trades)


def block_bootstrap_ci(values: np.ndarray, times_ns: np.ndarray, *, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    mask = np.isfinite(values)
    values = values[mask]
    times_ns = times_ns[mask]
    if len(values) < 2:
        return math.nan, math.nan
    block_id = ((times_ns - int(times_ns.min())) // int(BOOTSTRAP_CHUNK_SECONDS * 1_000_000_000)).astype(int)
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        samples.append(float(np.mean(values[idx])))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_pnl_trades(
    trades: pd.DataFrame,
    *,
    n_signal_events: int,
    seed: int,
) -> dict[str, object]:
    if trades.empty:
        return {
            "n_signal_events": int(n_signal_events),
            "n_executed": 0,
            "mean_pnl_bps": math.nan,
            "median_pnl_bps": math.nan,
            "win_rate": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "mean_hold_seconds": math.nan,
        }
    ci_lo, ci_hi = block_bootstrap_ci(
        trades["pnl_bps"].to_numpy(dtype=float),
        trades["entry_time_ns"].to_numpy(dtype=np.int64),
        seed=seed,
    )
    return {
        "n_signal_events": int(n_signal_events),
        "n_executed": int(len(trades)),
        "mean_pnl_bps": float(trades["pnl_bps"].mean()),
        "median_pnl_bps": float(trades["pnl_bps"].median()),
        "win_rate": float(trades["pnl_bps"].gt(0).mean()),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_hold_seconds": float(trades["hold_seconds"].mean()),
    }


def retest_a_binary(crypto: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = crypto_trade_counts(crypto)
    markets = counts[counts["last_trade_events"].ge(300)].copy()
    rows: list[dict[str, object]] = []
    trade_rows: list[pd.DataFrame] = []
    for idx, m in enumerate(markets.sort_values(["window_start_epoch", "symbol"]).itertuples(index=False), start=1):
        print(f"Retest A market {idx}/{len(markets)}: {m.slug}", flush=True)
        market_df = crypto[crypto["market_id"].eq(m.market_id) & crypto["run_id"].eq(m.run_id)].copy()
        for signal in ("ofi_5s", "tob_imbalance_level"):
            thr = threshold_row(thresholds, "crypto_4h_up_down", signal)
            for tail in ("top_decile", "bottom_decile"):
                events = binary_candidate_events(market_df, signal, tail, thr)
                for horizon in ("fixed_300s", "boundary_4h"):
                    trades = simulate_binary_trades(market_df, events, horizon_type=horizon, window_end_epoch=int(m.window_end_epoch))
                    if not trades.empty:
                        tr = trades.copy()
                        tr["retest"] = "A_binary"
                        tr["market"] = m.market
                        tr["slug"] = m.slug
                        tr["symbol"] = m.symbol
                        tr["window_start_epoch"] = int(m.window_start_epoch)
                        tr["window_end_epoch"] = int(m.window_end_epoch)
                        tr["signal_variant"] = signal
                        tr["signal_tail"] = tail
                        tr["horizon"] = horizon
                        trade_rows.append(tr)
                    rows.append(
                        {
                            "retest": "A_binary",
                            "sample_split": "a0c_roll_holdout",
                            "row_scope": "window",
                            "execution_mode": "taker",
                            "market": m.market,
                            "slug": m.slug,
                            "family": str(market_df["family"].replace("", np.nan).dropna().iloc[0]),
                            "symbol": m.symbol,
                            "window_start_epoch": int(m.window_start_epoch),
                            "window_end_epoch": int(m.window_end_epoch),
                            "signal_variant": signal,
                            "signal_tail": tail,
                            "horizon": horizon,
                            "target_type": "",
                            "timeout_sec": np.nan,
                            "fill_window_sec": np.nan,
                            "n_distinct_windows": 1,
                            "last_trade_events": int(m.last_trade_events),
                            "raw_entry_fill_rate": np.nan,
                            "executed_fill_rate": np.nan,
                            **summarize_pnl_trades(
                                trades,
                                n_signal_events=len(events),
                                seed=stable_seed("A", m.market, signal, tail, horizon),
                            ),
                        }
                    )
    all_trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    if not all_trades.empty:
        for (signal, tail, horizon), trades in all_trades.groupby(["signal_variant", "signal_tail", "horizon"], sort=False):
            n_signal = sum(
                row["n_signal_events"]
                for row in rows
                if row["signal_variant"] == signal and row["signal_tail"] == tail and row["horizon"] == horizon
            )
            n_windows = int(trades["window_start_epoch"].nunique())
            row = {
                "retest": "A_binary",
                "sample_split": "a0c_roll_holdout",
                "row_scope": "pooled",
                "execution_mode": "taker",
                "market": "ALL",
                "slug": "ALL",
                "family": "crypto_4h_up_down",
                "symbol": "ALL",
                "window_start_epoch": np.nan,
                "window_end_epoch": np.nan,
                "signal_variant": signal,
                "signal_tail": tail,
                "horizon": horizon,
                "target_type": "",
                "timeout_sec": np.nan,
                "fill_window_sec": np.nan,
                "n_distinct_windows": n_windows,
                "last_trade_events": int(markets["last_trade_events"].sum()),
                "raw_entry_fill_rate": np.nan,
                "executed_fill_rate": np.nan,
                **summarize_pnl_trades(
                    trades,
                    n_signal_events=int(n_signal),
                    seed=stable_seed("A", "pooled", signal, tail, horizon),
                ),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, all_trades
    out["bar_pass"] = (
        out["row_scope"].eq("pooled")
        & out["ci_lo"].gt(0)
        & out["n_executed"].ge(A_BAR_MIN_N)
        & out["n_distinct_windows"].ge(A_BAR_MIN_WINDOWS)
    )
    out["verdict"] = np.where(out["bar_pass"], "replicated", "failed_OOS")
    return out, all_trades


def market_for_p2(main: pd.DataFrame, market_id: str) -> pd.DataFrame:
    sub = main[main["market_id"].eq(market_id)].copy()
    keep_trade = sub["event_type"].eq("last_trade_price") & sub["trade_price"].notna()
    keep_quote = valid_quote_mask(sub)
    out = sub[keep_trade | keep_quote].copy()
    stale_trade = keep_trade.loc[out.index] & ~keep_quote.loc[out.index]
    out.loc[stale_trade, "is_book_state_complete"] = False
    out["received_at"] = out["event_ts"]
    out = out.sort_values(["asset_id", "received_at"]).reset_index(drop=True)
    return out


def p2_events_for_market(market_df: pd.DataFrame, q10: float, depth_q90: float) -> pd.DataFrame:
    sub = market_df[
        valid_quote_mask(market_df)
        & market_df["relative_depth"].ge(depth_q90)
        & market_df["ofi_5s"].replace([np.inf, -np.inf], np.nan).notna()
        & market_df["ofi_5s"].le(q10)
    ].copy()
    if sub.empty:
        return sub
    sub["signal_variant"] = "ofi_5s"
    sub["signal_tail"] = "bottom_decile"
    sub["signal_value"] = sub["ofi_5s"].astype(float)
    sub["abs_signal"] = sub["signal_value"].abs()
    sub["token_side"] = -np.sign(sub["signal_value"]) * sub["direction_factor"]
    sub = sub[sub["token_side"].isin([-1.0, 1.0])].copy()
    sub["event_time_ns"] = ns_from_ts(sub["event_ts"])
    sub["event_id"] = np.arange(len(sub), dtype=np.int64)
    sub["received_at"] = sub["event_ts"]
    return sub.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False]).reset_index(drop=True)


def summarize_b_executions(
    executed: pd.DataFrame,
    *,
    n_signal_events: int,
    n_raw_filled: int,
    seed: int,
) -> dict[str, object]:
    if not executed.empty:
        summary_trades = executed.copy()
        if "entry_time_ns_int" in summary_trades.columns:
            summary_trades["entry_time_ns"] = summary_trades["entry_time_ns_int"].astype(np.int64)
    else:
        summary_trades = executed
    base = summarize_pnl_trades(
        summary_trades,
        n_signal_events=n_signal_events,
        seed=seed,
    )
    base["n_entry_filled_raw"] = int(n_raw_filled)
    base["raw_entry_fill_rate"] = n_raw_filled / n_signal_events if n_signal_events else math.nan
    base["executed_fill_rate"] = base["n_executed"] / n_signal_events if n_signal_events else math.nan
    if not executed.empty:
        base["target_reached_rate"] = float(executed["exit_reason"].eq("target_reached").mean())
        base["timeout_rate"] = float(executed["exit_reason"].eq("timeout").mean())
        base["adverse_stop_rate"] = float(executed["exit_reason"].eq("adverse_stop").mean())
    else:
        base["target_reached_rate"] = math.nan
        base["timeout_rate"] = math.nan
        base["adverse_stop_rate"] = math.nan
    return base


def retest_b_reversion(main: pd.DataFrame, thresholds: pd.DataFrame, depth_q90: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    executed_parts: list[pd.DataFrame] = []
    market_ids = []
    for market_id, group in main.groupby("market_id", sort=False):
        if valid_quote_mask(group).any() and group.loc[valid_quote_mask(group), "relative_depth"].ge(depth_q90).any():
            market_ids.append(market_id)
        elif group["slug"].str.contains("strait-of-hormuz", case=False, na=False).any():
            market_ids.append(market_id)
    market_ids = sorted(set(market_ids))
    for idx, market_id in enumerate(market_ids, start=1):
        group = main[main["market_id"].eq(market_id)].copy()
        slug = str(group["slug"].replace("", np.nan).dropna().iloc[0]) if group["slug"].astype(bool).any() else market_id
        category = family_category(str(group["family"].replace("", np.nan).dropna().iloc[0]))
        scope = "geopolitics" if category == "Geopolitics" else "all"
        q10 = float(threshold_row(thresholds, scope, "ofi_5s")["q10"])
        events = p2_events_for_market(group, q10, depth_q90)
        p2_market = market_for_p2(main, market_id)
        print(f"Retest B market {idx}/{len(market_ids)}: {slug} events={len(events):,}", flush=True)
        candidates = passive_entry_candidates(events, p2_market, fill_window_sec=1) if not events.empty else events
        n_raw_filled = int(candidates["entry_filled"].sum()) if not candidates.empty and "entry_filled" in candidates else 0
        for target_type in ("micro_price", "half_to_micro_price"):
            for timeout in (5, 10, 30, 60):
                executed = (
                    execute_non_overlap(candidates, p2_market, target_type, timeout, "P")
                    if not candidates.empty
                    else pd.DataFrame()
                )
                if not executed.empty:
                    ex = executed.copy()
                    ex["market"] = f"{MAIN_RUN_ID}:{market_id}"
                    ex["slug"] = slug
                    ex["target_type"] = target_type
                    ex["timeout_sec"] = timeout
                    executed_parts.append(ex)
                rows.append(
                    {
                        "retest": "B_reversion",
                        "sample_split": "a0c_main_holdout",
                        "row_scope": "market",
                        "execution_mode": "passive",
                        "market": f"{MAIN_RUN_ID}:{market_id}",
                        "slug": slug,
                        "family": str(group["family"].replace("", np.nan).dropna().iloc[0]) if group["family"].astype(bool).any() else "",
                        "symbol": "",
                        "window_start_epoch": np.nan,
                        "window_end_epoch": np.nan,
                        "signal_variant": "ofi_5s",
                        "signal_tail": "bottom_decile",
                        "horizon": "",
                        "target_type": target_type,
                        "timeout_sec": timeout,
                        "fill_window_sec": 1,
                        "n_distinct_windows": np.nan,
                        "last_trade_events": int(group["event_type"].eq("last_trade_price").sum()),
                        **summarize_b_executions(
                            executed,
                            n_signal_events=len(events),
                            n_raw_filled=n_raw_filled,
                            seed=stable_seed("B", market_id, target_type, timeout),
                        ),
                    }
                )
    all_exec = pd.concat(executed_parts, ignore_index=True) if executed_parts else pd.DataFrame()
    if not all_exec.empty:
        for (target_type, timeout), trades in all_exec.groupby(["target_type", "timeout_sec"], sort=False):
            matching = [r for r in rows if r["target_type"] == target_type and r["timeout_sec"] == timeout]
            n_signal = int(sum(r["n_signal_events"] for r in matching))
            n_raw = int(sum(r["n_entry_filled_raw"] for r in matching))
            rows.append(
                {
                    "retest": "B_reversion",
                    "sample_split": "a0c_main_holdout",
                    "row_scope": "pooled",
                    "execution_mode": "passive",
                    "market": "ALL",
                    "slug": "ALL",
                    "family": "ALL",
                    "symbol": "",
                    "window_start_epoch": np.nan,
                    "window_end_epoch": np.nan,
                    "signal_variant": "ofi_5s",
                    "signal_tail": "bottom_decile",
                    "horizon": "",
                    "target_type": target_type,
                    "timeout_sec": int(timeout),
                    "fill_window_sec": 1,
                    "n_distinct_windows": np.nan,
                    "last_trade_events": int(main["event_type"].eq("last_trade_price").sum()),
                    **summarize_b_executions(
                        trades,
                        n_signal_events=n_signal,
                        n_raw_filled=n_raw,
                        seed=stable_seed("B", "pooled", target_type, timeout),
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bar_pass"] = out["ci_lo"].gt(0) & out["n_executed"].ge(B_BAR_MIN_N) & out["executed_fill_rate"].ge(B_BAR_MIN_FILL)
    out["verdict"] = np.where(out["bar_pass"], "replicated", "artifact_confirmed")
    return out


def simulate_tob_hits(market_df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    arrays = quote_arrays_by_asset(market_df)
    next_available = -1
    rows: list[dict[str, object]] = []
    for row in events.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False]).itertuples(index=False):
        entry_ns = int(row.event_time_ns)
        if entry_ns <= next_available:
            continue
        target_ns = entry_ns + 5 * 1_000_000_000
        state = arrays.get(str(row.asset_id))
        if state is None:
            continue
        times = state["times"]
        entry_idx = int(np.searchsorted(times, entry_ns, side="right") - 1)
        exit_idx = target_exit_index(times, target_ns)
        if entry_idx < 0 or exit_idx is None or exit_idx <= entry_idx:
            continue
        direction_factor = float(row.direction_factor)
        current_mid = float(state["mid"][entry_idx])
        future_mid = float(state["mid"][exit_idx])
        current_dir_mid = current_mid if direction_factor > 0 else 1.0 - current_mid
        future_dir_mid = future_mid if direction_factor > 0 else 1.0 - future_mid
        if current_dir_mid <= 0 or not np.isfinite(future_dir_mid):
            continue
        ret_bps = (future_dir_mid - current_dir_mid) / current_dir_mid * 10_000.0
        signal_sign = float(np.sign(row.signal_value))
        rows.append(
            {
                "entry_time_ns": entry_ns,
                "entry_time": pd.Timestamp(row.event_ts),
                "future_return_bps": float(ret_bps),
                "hit": float(signal_sign * ret_bps > 0),
                "signal_value": float(row.signal_value),
            }
        )
        next_available = target_ns
    return pd.DataFrame(rows)


def tob_hit_events(market_df: pd.DataFrame, abs_q90: float) -> pd.DataFrame:
    sub = market_df[
        valid_quote_mask(market_df)
        & market_df["tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    sub = sub[sub["tob_imbalance_level"].abs().ge(abs_q90)].copy()
    if sub.empty:
        return sub
    sub["signal_value"] = sub["tob_imbalance_level"].astype(float)
    sub["abs_signal"] = sub["signal_value"].abs()
    sub["event_time_ns"] = ns_from_ts(sub["event_ts"])
    return (
        sub.sort_values(["event_time_ns", "abs_signal"], ascending=[True, False])
        .drop_duplicates(["event_time_ns"], keep="first")
        .reset_index(drop=True)
    )


def summarize_hits(hits: pd.DataFrame, *, n_signal_events: int, seed: int) -> dict[str, object]:
    if hits.empty:
        return {
            "n_signal_events": int(n_signal_events),
            "n_executed": 0,
            "hit_rate": math.nan,
            "hit_ci_lo": math.nan,
            "hit_ci_hi": math.nan,
            "mean_directional_return_bps": math.nan,
        }
    lo, hi = block_bootstrap_ci(
        hits["hit"].to_numpy(dtype=float),
        hits["entry_time_ns"].to_numpy(dtype=np.int64),
        seed=seed,
    )
    return {
        "n_signal_events": int(n_signal_events),
        "n_executed": int(len(hits)),
        "hit_rate": float(hits["hit"].mean()),
        "hit_ci_lo": lo,
        "hit_ci_hi": hi,
        "mean_directional_return_bps": float(hits["future_return_bps"].mean()),
    }


def retest_c_tob_hit(crypto: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    counts = crypto_trade_counts(crypto)
    markets = counts[counts["last_trade_events"].ge(300)].copy()
    thr = threshold_row(thresholds, "crypto_4h_up_down", "tob_imbalance_level")
    abs_q90 = float(thr["abs_q90"])
    rows: list[dict[str, object]] = []
    hit_parts: list[pd.DataFrame] = []
    for idx, m in enumerate(markets.sort_values(["window_start_epoch", "symbol"]).itertuples(index=False), start=1):
        print(f"Retest C market {idx}/{len(markets)}: {m.slug}", flush=True)
        market_df = crypto[crypto["market_id"].eq(m.market_id) & crypto["run_id"].eq(m.run_id)].copy()
        events = tob_hit_events(market_df, abs_q90)
        hits = simulate_tob_hits(market_df, events)
        if not hits.empty:
            h = hits.copy()
            h["market"] = m.market
            h["slug"] = m.slug
            h["symbol"] = m.symbol
            h["window_start_epoch"] = int(m.window_start_epoch)
            hit_parts.append(h)
        rows.append(
            {
                "retest": "C_tob_hit",
                "sample_split": "a0c_roll_holdout",
                "row_scope": "window",
                "execution_mode": "diagnostic",
                "market": m.market,
                "slug": m.slug,
                "family": "crypto_4h_up_down",
                "symbol": m.symbol,
                "window_start_epoch": int(m.window_start_epoch),
                "window_end_epoch": int(m.window_end_epoch),
                "signal_variant": "tob_imbalance_level",
                "signal_tail": "top_abs_decile",
                "horizon": "hit_5s",
                "target_type": "",
                "timeout_sec": np.nan,
                "fill_window_sec": np.nan,
                "n_distinct_windows": 1,
                "last_trade_events": int(m.last_trade_events),
                "a13_point_hit_rate": A1_3_TOB_HIT_POINT,
                **summarize_hits(hits, n_signal_events=len(events), seed=stable_seed("C", m.market)),
            }
        )
    all_hits = pd.concat(hit_parts, ignore_index=True) if hit_parts else pd.DataFrame()
    if not all_hits.empty:
        n_signal = int(sum(r["n_signal_events"] for r in rows))
        rows.append(
            {
                "retest": "C_tob_hit",
                "sample_split": "a0c_roll_holdout",
                "row_scope": "pooled",
                "execution_mode": "diagnostic",
                "market": "ALL",
                "slug": "ALL",
                "family": "crypto_4h_up_down",
                "symbol": "ALL",
                "window_start_epoch": np.nan,
                "window_end_epoch": np.nan,
                "signal_variant": "tob_imbalance_level",
                "signal_tail": "top_abs_decile",
                "horizon": "hit_5s",
                "target_type": "",
                "timeout_sec": np.nan,
                "fill_window_sec": np.nan,
                "n_distinct_windows": int(all_hits["window_start_epoch"].nunique()),
                "last_trade_events": int(markets["last_trade_events"].sum()),
                "a13_point_hit_rate": A1_3_TOB_HIT_POINT,
                **summarize_hits(all_hits, n_signal_events=n_signal, seed=stable_seed("C", "pooled")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ci_contains_a13_point"] = out["hit_ci_lo"].le(A1_3_TOB_HIT_POINT) & out["hit_ci_hi"].ge(A1_3_TOB_HIT_POINT)
    out["delta_vs_a13_hit_rate"] = out["hit_rate"] - A1_3_TOB_HIT_POINT
    out["bar_pass"] = out["row_scope"].eq("pooled") & out["ci_contains_a13_point"] & out["n_executed"].ge(A_BAR_MIN_N)
    out["verdict"] = np.where(
        out["bar_pass"],
        "replicated",
        np.where(out["hit_ci_hi"].lt(A1_3_TOB_HIT_POINT), "artifact_confirmed", "failed_OOS"),
    )
    return out


def align_surface_columns(frames: list[pd.DataFrame]) -> pd.DataFrame:
    ordered = [
        "retest",
        "sample_split",
        "row_scope",
        "verdict",
        "bar_pass",
        "execution_mode",
        "market",
        "slug",
        "family",
        "symbol",
        "window_start_epoch",
        "window_end_epoch",
        "signal_variant",
        "signal_tail",
        "horizon",
        "target_type",
        "timeout_sec",
        "fill_window_sec",
        "n_distinct_windows",
        "last_trade_events",
        "n_signal_events",
        "n_entry_filled_raw",
        "n_executed",
        "raw_entry_fill_rate",
        "executed_fill_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_hold_seconds",
        "hit_rate",
        "hit_ci_lo",
        "hit_ci_hi",
        "a13_point_hit_rate",
        "delta_vs_a13_hit_rate",
        "ci_contains_a13_point",
        "mean_directional_return_bps",
        "target_reached_rate",
        "timeout_rate",
        "adverse_stop_rate",
    ]
    out = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True, sort=False)
    for col in ordered:
        if col not in out.columns:
            out[col] = np.nan
    extra = [col for col in out.columns if col not in ordered]
    return out[ordered + extra]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def top_rows_table(df: pd.DataFrame, retest: str, n: int = 10) -> str:
    sub = df[df["retest"].eq(retest)].copy()
    if sub.empty:
        return "_No rows._"
    if retest == "C_tob_hit":
        sub = sub.sort_values(["row_scope", "hit_ci_lo", "hit_rate"], ascending=[True, False, False]).head(n)
        rows = [
            [
                row.row_scope,
                safe_text(row.slug),
                row.signal_tail,
                int(row.n_executed),
                pct(float(row.hit_rate)),
                f"[{pct(float(row.hit_ci_lo))}, {pct(float(row.hit_ci_hi))}]",
                pct(float(row.delta_vs_a13_hit_rate)),
                row.verdict,
            ]
            for row in sub.itertuples(index=False)
        ]
        return markdown_table(["scope", "slug", "tail", "n", "hit", "CI", "delta vs A1.3", "verdict"], rows)
    sub = sub.sort_values(["row_scope", "ci_lo", "mean_pnl_bps"], ascending=[True, False, False]).head(n)
    rows = []
    for row in sub.itertuples(index=False):
        fill = pct(float(row.executed_fill_rate)) if np.isfinite(float(row.executed_fill_rate)) else ""
        rows.append(
            [
                row.row_scope,
                safe_text(row.slug),
                row.signal_variant,
                row.signal_tail,
                row.horizon or row.target_type,
                int(row.n_executed),
                fill,
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                row.verdict,
            ]
        )
    return markdown_table(["scope", "slug", "signal", "tail", "hold/target", "n", "fill", "mean", "CI", "verdict"], rows)


def write_note(surface: pd.DataFrame, run_counts: pd.DataFrame, confirmations: dict[str, dict[str, int]], thresholds: pd.DataFrame, depth_q90: float) -> None:
    a_pass = bool(surface[surface["retest"].eq("A_binary")]["bar_pass"].fillna(False).any())
    b_pass = bool(surface[surface["retest"].eq("B_reversion")]["bar_pass"].fillna(False).any())
    c_pooled = surface[(surface["retest"].eq("C_tob_hit")) & (surface["row_scope"].eq("pooled"))].copy()
    if not c_pooled.empty and bool(c_pooled["bar_pass"].fillna(False).any()):
        c_status = "replicated"
    elif not c_pooled.empty and bool(c_pooled["hit_ci_hi"].lt(A1_3_TOB_HIT_POINT).all()):
        c_status = "artifact-confirmed"
    else:
        c_status = "failed-OOS"
    a_status = "replicated" if a_pass else "failed-OOS"
    b_status = "replicated" if b_pass else "artifact-confirmed"
    close_line = (
        "Retest A and B both fail; the local Dali signal is closed with no remaining needs-more-data caveat."
        if not a_pass and not b_pass
        else "At least one executable retest cleared its bar; escalate the surviving track only."
    )
    a_window = surface[(surface["retest"].eq("A_binary")) & (surface["row_scope"].eq("window"))]
    a_universe = a_window[["slug", "symbol"]].drop_duplicates() if not a_window.empty else pd.DataFrame()
    symbols = ", ".join(sorted(str(s) for s in a_universe["symbol"].dropna().unique())) if not a_universe.empty else "none"

    run_rows = [
        [
            row.run_id,
            f"{int(row.n_rows):,}",
            int(row.n_markets),
            str(row.first_ts)[:19],
            str(row.last_ts)[:19],
        ]
        for row in run_counts.itertuples(index=False)
    ]
    confirm_rows = []
    for label, note_key, scan_key in (
        ("main_a0c", "main_final_note", "main_raw_scan"),
        ("crypto_roll", "roll_final_note", "roll_raw_scan"),
    ):
        note_counts = confirmations.get(note_key, {})
        scan_counts = confirmations.get(scan_key, {})
        for event_type in ("book", "price_change", "best_bid_ask", "last_trade_price"):
            confirm_rows.append(
                [
                    label,
                    event_type,
                    f"{note_counts.get(event_type, 0):,}",
                    f"{scan_counts.get(event_type, 0):,}",
                    "yes" if note_counts.get(event_type, 0) == scan_counts.get(event_type, 0) else "NO",
                ]
            )

    threshold_rows = [
        [
            row.threshold_scope,
            row.signal_variant,
            f"{float(row.q10):.6g}",
            f"{float(row.q90):.6g}",
            f"{float(row.abs_q90):.6g}",
            f"{int(row.n_threshold_rows):,}",
        ]
        for row in thresholds.itertuples(index=False)
    ]

    lines = [
        "---",
        "tags: [dali, a0c, holdout, oos, retest, results]",
        "---",
        "",
        "# A0c Holdout Retest Findings",
        "",
        "## Headline",
        "",
        f"- Retest A binary-bet / 4h-boundary: **{a_status}** under CI lower > 0, n >= {A_BAR_MIN_N}, and >= {A_BAR_MIN_WINDOWS} windows.",
        f"- Retest B passive deep-book fade: **{b_status}** under CI lower > 0, n >= {B_BAR_MIN_N}, and fill >= {pct(B_BAR_MIN_FILL)}.",
        f"- Retest C TOB hit-rate diagnostic: **{c_status}** versus the A1.3 73.7% point estimate.",
        f"- {close_line}",
        f"- Retest A/C crypto-roll universe after the >=300-trade gate: {len(a_universe)} windows ({symbols}). ETH/SOL peer windows were captured but did not meet the preregistered trade-count gate.",
        "",
        "## Retest A Top Rows",
        "",
        top_rows_table(surface, "A_binary", n=12),
        "",
        "## Retest B Top Rows",
        "",
        top_rows_table(surface, "B_reversion", n=12),
        "",
        "## Retest C Rows",
        "",
        top_rows_table(surface, "C_tob_hit", n=12),
        "",
        "## Feature Panel Append Check",
        "",
        markdown_table(["run", "feature rows", "markets", "first", "last"], run_rows),
        "",
        "Raw JSONL event counts match the final A0c notes:",
        "",
        markdown_table(["capture", "event", "final note", "raw scan", "match"], confirm_rows),
        "",
        "## Discovery Thresholds",
        "",
        f"Deep-book relative-depth q90 from A0/A0b discovery rows: `{depth_q90:.6g}`.",
        "",
        markdown_table(["scope", "signal", "q10", "q90", "abs q90", "n"], threshold_rows),
        "",
        "## Method",
        "",
        "- A0/A0b are discovery only; A0c main and A0c crypto_roll are strict holdout rows.",
        "- A0c main is tagged `a0c`; crypto_roll is tagged `a0c_roll` in `data/analysis/block_a1_features.parquet`.",
        "- `will-jd-vance` is excluded from all retests.",
        "- Event ordering and horizons use `exchange_ts` when present, falling back to `received_at`.",
        "- Entry and exit quote states require `is_book_state_complete` and `book_staleness_seconds <= 5`.",
        "- Confidence intervals are 300s clock-block bootstrap intervals.",
        "- Retest A uses taker entry/exit, net of `FEE_BY_CATEGORY`, with non-overlap per market.",
        "- Retest B uses the P2 passive fill proxy with W=1s, maker entry rebate, taker exit, and non-overlap after fill.",
        "- Retest C is PnL-independent and non-overlap at the 5s horizon.",
        "",
        "## Outputs",
        "",
        f"- `{display_path(OUT_CSV)}`",
        f"- `{display_path(NOTE)}`",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_counts, confirmations = append_missing_a0c_runs()
    thresholds, depth_q90 = discovery_thresholds()
    print("loading A0c holdout rows", flush=True)
    main_df = add_signals(load_feature_rows((MAIN_RUN_ID,)))
    crypto_df = add_signals(load_feature_rows((ROLL_RUN_ID,)))
    print(f"main holdout rows: {len(main_df):,}; crypto_roll rows: {len(crypto_df):,}", flush=True)
    a_rows, _ = retest_a_binary(crypto_df, thresholds)
    b_rows = retest_b_reversion(main_df, thresholds, depth_q90)
    c_rows = retest_c_tob_hit(crypto_df, thresholds)
    surface = align_surface_columns([a_rows, b_rows, c_rows])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(OUT_CSV, index=False)
    write_note(surface, run_counts, confirmations, thresholds, depth_q90)
    print(f"wrote {display_path(OUT_CSV)} rows={len(surface):,}", flush=True)
    print(f"wrote {display_path(NOTE)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
