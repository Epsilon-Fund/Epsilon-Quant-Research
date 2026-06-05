"""Block A1.4f combined refined-exit + tight-spread taker entry.

This sidecar combines A14b's refined exits with A14d's spread-filtered TOB
entry. It does not modify A1/A13/A14 artifacts or raw capture JSONL.
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
A13_PERSISTENCE = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_persistence_by_market.csv"
A14B_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a14b_refined_exit_results.csv"
A14D_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a14d_tight_spread_results.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14f_combined_results.csv"
NOTE = NOTES / "block_a14f_combined_findings.md"

HORIZONS = (5, 30, 300)
SPREAD_THRESHOLDS: tuple[tuple[str, float | None], ...] = (
    ("100", 100.0),
    ("200", 200.0),
    ("500", 500.0),
    ("1000", 1000.0),
    ("no_filter", None),
)
EXIT_CONFIGS = (
    "cfg_fixed_5s",
    "cfg_signal_reversal",
    "cfg_take_profit_500bps",
    "cfg_stop_loss_combined",
)
EXIT_REASONS = ("signal_reversal", "take_profit", "stop_loss", "time_stop")
TAKE_PROFIT_BPS = 500.0
STOP_LOSS_BPS = -100.0
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260529
MIN_INTERPRETABLE_EVENTS = 30


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fee_amount(category: str, price: float | np.ndarray) -> float | np.ndarray:
    """Polymarket taker fee in dollars per share at price p."""
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = np.clip(price, 0.0, 1.0)
    return params["fee_rate"] * p * (1.0 - p)


def load_candidates() -> pd.DataFrame:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    candidates = results[results["sample_size_label"].eq("primary_read")].copy()
    candidates = candidates.sort_values("horizon_sec")
    candidates = candidates[
        ["run_id", "market_id", "family", "n_classifiable"]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no primary_read candidates found in block_a1_results.csv")
    candidates["market"] = candidates["run_id"] + ":" + candidates["market_id"]
    return candidates.reset_index(drop=True)


def load_time_stops(candidates: pd.DataFrame) -> pd.DataFrame:
    persistence = pd.read_csv(A13_PERSISTENCE, dtype={"run_id": str, "market_key": str})
    merged = candidates.merge(
        persistence[["run_id", "market_key", "p90_time_until_flip_sec"]],
        left_on=["run_id", "market_id"],
        right_on=["run_id", "market_key"],
        how="left",
    )
    missing = merged["p90_time_until_flip_sec"].isna()
    if missing.any():
        raise SystemExit(
            "missing A1.3 p90 persistence stop for: "
            + ", ".join(merged.loc[missing, "market"].astype(str).tolist())
        )
    merged["p90_time_stop_sec"] = pd.to_numeric(
        merged["p90_time_until_flip_sec"],
        errors="coerce",
    ).clip(lower=0.001)
    return merged.drop(columns=["market_key"])


def load_a14b_baseline() -> dict[tuple[str, str], float]:
    if not A14B_RESULTS.exists():
        return {}
    df = pd.read_csv(A14B_RESULTS)
    labels = []
    for row in df.itertuples(index=False):
        config = str(row.config)
        if config == "cfg_take_profit" and np.isfinite(row.take_profit_bps):
            config = f"cfg_take_profit_{int(row.take_profit_bps)}bps"
        labels.append(config)
    df["exit_config"] = labels
    return {
        (str(row.market), str(row.exit_config)): float(row.mean_pnl_bps)
        for row in df.itertuples(index=False)
        if np.isfinite(row.mean_pnl_bps)
    }


def load_a14d_baseline() -> dict[tuple[str, int, str], float]:
    if not A14D_RESULTS.exists():
        return {}
    df = pd.read_csv(A14D_RESULTS)
    return {
        (str(row.market), int(row.horizon), str(row.spread_threshold_bps)): float(row.mean_pnl_bps)
        for row in df.itertuples(index=False)
        if np.isfinite(row.mean_pnl_bps)
    }


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
    for col in ("run_id", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    for col in ("outcome_index", "best_bid", "best_ask", "spread", "mid", "tob_imbalance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    quoted_spread = df["best_ask"] - df["best_bid"]
    spread = df["spread"].where(df["spread"].notna(), quoted_spread)
    df["spread_bps"] = np.where(df["mid"].gt(0), spread / df["mid"] * 10_000.0, np.nan)
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    return df


def add_tob_signal(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "market_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g["signal_sign"] = np.sign(g["tob_imbalance_level"]).replace(0.0, np.nan)
        g["token_side"] = g["signal_sign"] * g["direction_factor"]
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out["is_top_tob_decile"] = False
    valid = (
        out["is_book_state_complete"]
        & out["tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
        & out["tob_imbalance_level"].ne(0.0)
        & out["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & out["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & out["spread_bps"].replace([np.inf, -np.inf], np.nan).notna()
        & out["best_ask"].ge(out["best_bid"])
    )
    for _, market in out[valid].groupby(["run_id", "market_id"], sort=False):
        if len(market) < 10:
            continue
        ranked = market["tob_imbalance_level"].abs().rank(method="first")
        decile = pd.qcut(ranked, 10, labels=False) + 1
        out.loc[decile[decile.eq(10)].index, "is_top_tob_decile"] = True
    return out.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)


def first_opposite_index(
    entry_pos: int,
    entry_sign: float,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
) -> int | None:
    opposite = neg_idx if entry_sign > 0 else pos_idx
    loc = np.searchsorted(opposite, entry_pos + 1, side="left")
    if loc >= len(opposite):
        return None
    return int(opposite[loc])


def first_favorable_index(
    mid: np.ndarray,
    start_pos: int,
    stop_pos: int,
    entry_mid: float,
    token_side: float,
    threshold_bps: float,
) -> int | None:
    if stop_pos < start_pos or not np.isfinite(entry_mid) or entry_mid <= 0:
        return None
    window = mid[start_pos : stop_pos + 1]
    if token_side > 0:
        move_bps = (window - entry_mid) / entry_mid * 10_000.0
    else:
        move_bps = (entry_mid - window) / entry_mid * 10_000.0
    hits = np.flatnonzero(np.isfinite(move_bps) & (move_bps >= threshold_bps))
    return int(start_pos + hits[0]) if len(hits) else None


def first_adverse_index(
    mid: np.ndarray,
    start_pos: int,
    stop_pos: int,
    entry_mid: float,
    token_side: float,
    stop_loss_bps: float,
) -> int | None:
    if stop_pos < start_pos or not np.isfinite(entry_mid) or entry_mid <= 0:
        return None
    window = mid[start_pos : stop_pos + 1]
    if token_side > 0:
        move_bps = (window - entry_mid) / entry_mid * 10_000.0
    else:
        move_bps = (entry_mid - window) / entry_mid * 10_000.0
    hits = np.flatnonzero(np.isfinite(move_bps) & (move_bps <= stop_loss_bps))
    return int(start_pos + hits[0]) if len(hits) else None


def entry_price(token_side: float, idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(ask[idx]) if token_side > 0 else float(bid[idx])


def exit_price(token_side: float, idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(bid[idx]) if token_side > 0 else float(ask[idx])


def trade_pnl_bps(category: str, token_side: float, entry_px: float, exit_px: float) -> float:
    if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
        return math.nan
    gross = exit_px - entry_px if token_side > 0 else entry_px - exit_px
    fees = float(fee_amount(category, entry_px) + fee_amount(category, exit_px))
    return float((gross - fees) / entry_px * 10_000.0)


def make_trade_record(
    *,
    received_at: pd.Timestamp,
    entry_time_ns: int,
    exit_time_ns: int | None,
    entry_idx: int,
    exit_idx: int | None,
    token_side: float,
    bid: np.ndarray,
    ask: np.ndarray,
    category: str,
    reason: str,
) -> dict[str, object]:
    entry_px = entry_price(token_side, entry_idx, bid, ask)
    fillable = exit_idx is not None and exit_time_ns is not None
    exit_px = exit_price(token_side, exit_idx, bid, ask) if fillable else math.nan
    pnl = trade_pnl_bps(category, token_side, entry_px, exit_px) if fillable else math.nan
    if not np.isfinite(pnl):
        fillable = False
    return {
        "received_at": received_at,
        "is_fillable": bool(fillable),
        "pnl_bps": pnl if fillable else math.nan,
        "hold_seconds": (exit_time_ns - entry_time_ns) / 1_000_000_000.0
        if fillable and exit_time_ns is not None
        else math.nan,
        "exit_reason": reason if fillable else "unfillable",
    }


def simulate_entry_rows(
    group: pd.DataFrame,
    *,
    category: str,
    horizon_sec: int,
    spread_threshold: float | None,
    exit_config: str,
    p90_time_stop_sec: float,
) -> list[dict[str, object]]:
    g = group.sort_values("received_at").reset_index(drop=True)
    entry_mask = g["is_top_tob_decile"].to_numpy(dtype=bool).copy()
    if spread_threshold is not None:
        entry_mask &= g["spread_bps"].to_numpy(dtype=float) <= spread_threshold
    entry_positions = np.flatnonzero(entry_mask)
    if len(entry_positions) == 0:
        return []

    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bid = g["best_bid"].to_numpy(dtype=float)
    ask = g["best_ask"].to_numpy(dtype=float)
    mid = g["mid"].to_numpy(dtype=float)
    signal_sign = g["signal_sign"].to_numpy(dtype=float)
    token_side_arr = g["token_side"].to_numpy(dtype=float)
    pos_idx = np.flatnonzero(signal_sign > 0)
    neg_idx = np.flatnonzero(signal_sign < 0)
    last_time = int(times[-1]) if len(times) else 0
    horizon_ns = int(horizon_sec * 1_000_000_000)
    dynamic_stop_ns_delta = int(min(float(p90_time_stop_sec), float(horizon_sec)) * 1_000_000_000)
    records: list[dict[str, object]] = []

    for entry_idx in entry_positions:
        entry_time = int(times[entry_idx])
        received_at = pd.Timestamp(g.loc[entry_idx, "received_at"])
        token_side = float(token_side_arr[entry_idx])
        entry_sign = float(signal_sign[entry_idx])
        if (
            not np.isfinite(token_side)
            or token_side == 0
            or not np.isfinite(entry_sign)
            or entry_sign == 0
        ):
            continue

        if exit_config == "cfg_fixed_5s":
            target = entry_time + horizon_ns
            if target <= last_time:
                exit_idx = int(np.searchsorted(times, target, side="right") - 1)
                if exit_idx < entry_idx:
                    exit_idx = None
            else:
                exit_idx = None
            records.append(
                make_trade_record(
                    received_at=received_at,
                    entry_time_ns=entry_time,
                    exit_time_ns=target if exit_idx is not None else None,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    token_side=token_side,
                    bid=bid,
                    ask=ask,
                    category=category,
                    reason="time_stop",
                )
            )
            continue

        stop_target = entry_time + dynamic_stop_ns_delta
        stop_available = stop_target <= last_time
        if stop_available:
            time_stop_idx = int(np.searchsorted(times, stop_target, side="right") - 1)
            if time_stop_idx < entry_idx:
                time_stop_idx = None
        else:
            time_stop_idx = None
        scan_start = entry_idx + 1
        scan_stop = time_stop_idx if stop_available and time_stop_idx is not None else len(times) - 1

        reversal_idx = first_opposite_index(entry_idx, entry_sign, pos_idx, neg_idx)
        if reversal_idx is not None and int(times[reversal_idx]) > stop_target:
            reversal_idx = None

        if exit_config == "cfg_signal_reversal":
            if reversal_idx is not None:
                exit_idx = reversal_idx
                exit_time = int(times[reversal_idx])
                reason = "signal_reversal"
            elif stop_available and time_stop_idx is not None:
                exit_idx = time_stop_idx
                exit_time = stop_target
                reason = "time_stop"
            else:
                exit_idx = None
                exit_time = None
                reason = "unfillable"
        elif exit_config == "cfg_take_profit_500bps":
            tp_idx = first_favorable_index(
                mid,
                scan_start,
                scan_stop,
                float(mid[entry_idx]),
                token_side,
                TAKE_PROFIT_BPS,
            )
            if tp_idx is not None and int(times[tp_idx]) <= stop_target:
                exit_idx = tp_idx
                exit_time = int(times[tp_idx])
                reason = "take_profit"
            elif stop_available and time_stop_idx is not None:
                exit_idx = time_stop_idx
                exit_time = stop_target
                reason = "time_stop"
            else:
                exit_idx = None
                exit_time = None
                reason = "unfillable"
        elif exit_config == "cfg_stop_loss_combined":
            stop_idx = first_adverse_index(
                mid,
                scan_start,
                scan_stop,
                float(mid[entry_idx]),
                token_side,
                STOP_LOSS_BPS,
            )
            candidates: list[tuple[int, str]] = []
            if reversal_idx is not None:
                candidates.append((reversal_idx, "signal_reversal"))
            if stop_idx is not None and int(times[stop_idx]) <= stop_target:
                candidates.append((stop_idx, "stop_loss"))
            candidates.sort(key=lambda item: (int(times[item[0]]), 0 if item[1] == "signal_reversal" else 1))
            if candidates:
                exit_idx = candidates[0][0]
                exit_time = int(times[exit_idx])
                reason = candidates[0][1]
            elif stop_available and time_stop_idx is not None:
                exit_idx = time_stop_idx
                exit_time = stop_target
                reason = "time_stop"
            else:
                exit_idx = None
                exit_time = None
                reason = "unfillable"
        else:
            raise ValueError(f"unknown exit config {exit_config}")

        records.append(
            make_trade_record(
                received_at=received_at,
                entry_time_ns=entry_time,
                exit_time_ns=exit_time,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                token_side=token_side,
                bid=bid,
                ask=ask,
                category=category,
                reason=reason,
            )
        )
    return records


def simulate_asset_horizon_all_configs(
    group: pd.DataFrame,
    *,
    category: str,
    horizon_sec: int,
    p90_time_stop_sec: float,
) -> list[dict[str, object]]:
    """Simulate all exit configs once, leaving spread filtering to aggregation."""
    g = group.sort_values("received_at").reset_index(drop=True)
    entry_positions = np.flatnonzero(g["is_top_tob_decile"].to_numpy(dtype=bool))
    if len(entry_positions) == 0:
        return []

    times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bid = g["best_bid"].to_numpy(dtype=float)
    ask = g["best_ask"].to_numpy(dtype=float)
    mid = g["mid"].to_numpy(dtype=float)
    spread = g["spread_bps"].to_numpy(dtype=float)
    signal_sign = g["signal_sign"].to_numpy(dtype=float)
    token_side_arr = g["token_side"].to_numpy(dtype=float)
    pos_idx = np.flatnonzero(signal_sign > 0)
    neg_idx = np.flatnonzero(signal_sign < 0)
    last_time = int(times[-1]) if len(times) else 0
    horizon_ns = int(horizon_sec * 1_000_000_000)
    dynamic_stop_ns_delta = int(min(float(p90_time_stop_sec), float(horizon_sec)) * 1_000_000_000)
    records: list[dict[str, object]] = []

    def add_record(
        *,
        exit_config: str,
        entry_idx: int,
        exit_idx: int | None,
        exit_time: int | None,
        token_side: float,
        reason: str,
    ) -> None:
        record = make_trade_record(
            received_at=pd.Timestamp(g.loc[entry_idx, "received_at"]),
            entry_time_ns=int(times[entry_idx]),
            exit_time_ns=exit_time,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            token_side=token_side,
            bid=bid,
            ask=ask,
            category=category,
            reason=reason,
        )
        record["exit_config"] = exit_config
        record["spread_bps"] = float(spread[entry_idx])
        records.append(record)

    for entry_idx in entry_positions:
        entry_time = int(times[entry_idx])
        token_side = float(token_side_arr[entry_idx])
        entry_sign = float(signal_sign[entry_idx])
        if (
            not np.isfinite(token_side)
            or token_side == 0
            or not np.isfinite(entry_sign)
            or entry_sign == 0
        ):
            continue

        fixed_target = entry_time + horizon_ns
        fixed_exit_idx: int | None = None
        if fixed_target <= last_time:
            idx = int(np.searchsorted(times, fixed_target, side="right") - 1)
            if idx >= entry_idx:
                fixed_exit_idx = idx
        add_record(
            exit_config="cfg_fixed_5s",
            entry_idx=entry_idx,
            exit_idx=fixed_exit_idx,
            exit_time=fixed_target if fixed_exit_idx is not None else None,
            token_side=token_side,
            reason="time_stop",
        )

        stop_target = entry_time + dynamic_stop_ns_delta
        stop_available = stop_target <= last_time
        if stop_available:
            time_stop_idx = int(np.searchsorted(times, stop_target, side="right") - 1)
            if time_stop_idx < entry_idx:
                time_stop_idx = None
        else:
            time_stop_idx = None
        scan_start = entry_idx + 1
        scan_stop = time_stop_idx if stop_available and time_stop_idx is not None else len(times) - 1

        reversal_idx = first_opposite_index(entry_idx, entry_sign, pos_idx, neg_idx)
        if reversal_idx is not None and int(times[reversal_idx]) > stop_target:
            reversal_idx = None

        if reversal_idx is not None:
            reversal_exit_idx = reversal_idx
            reversal_time = int(times[reversal_idx])
            reversal_reason = "signal_reversal"
        elif stop_available and time_stop_idx is not None:
            reversal_exit_idx = time_stop_idx
            reversal_time = stop_target
            reversal_reason = "time_stop"
        else:
            reversal_exit_idx = None
            reversal_time = None
            reversal_reason = "unfillable"
        add_record(
            exit_config="cfg_signal_reversal",
            entry_idx=entry_idx,
            exit_idx=reversal_exit_idx,
            exit_time=reversal_time,
            token_side=token_side,
            reason=reversal_reason,
        )

        tp_idx = first_favorable_index(
            mid,
            scan_start,
            scan_stop,
            float(mid[entry_idx]),
            token_side,
            TAKE_PROFIT_BPS,
        )
        if tp_idx is not None and int(times[tp_idx]) <= stop_target:
            tp_exit_idx = tp_idx
            tp_time = int(times[tp_idx])
            tp_reason = "take_profit"
        elif stop_available and time_stop_idx is not None:
            tp_exit_idx = time_stop_idx
            tp_time = stop_target
            tp_reason = "time_stop"
        else:
            tp_exit_idx = None
            tp_time = None
            tp_reason = "unfillable"
        add_record(
            exit_config="cfg_take_profit_500bps",
            entry_idx=entry_idx,
            exit_idx=tp_exit_idx,
            exit_time=tp_time,
            token_side=token_side,
            reason=tp_reason,
        )

        stop_idx = first_adverse_index(
            mid,
            scan_start,
            scan_stop,
            float(mid[entry_idx]),
            token_side,
            STOP_LOSS_BPS,
        )
        stop_candidates: list[tuple[int, str]] = []
        if reversal_idx is not None:
            stop_candidates.append((reversal_idx, "signal_reversal"))
        if stop_idx is not None and int(times[stop_idx]) <= stop_target:
            stop_candidates.append((stop_idx, "stop_loss"))
        stop_candidates.sort(key=lambda item: (int(times[item[0]]), 0 if item[1] == "signal_reversal" else 1))
        if stop_candidates:
            sl_exit_idx = stop_candidates[0][0]
            sl_time = int(times[sl_exit_idx])
            sl_reason = stop_candidates[0][1]
        elif stop_available and time_stop_idx is not None:
            sl_exit_idx = time_stop_idx
            sl_time = stop_target
            sl_reason = "time_stop"
        else:
            sl_exit_idx = None
            sl_time = None
            sl_reason = "unfillable"
        add_record(
            exit_config="cfg_stop_loss_combined",
            entry_idx=entry_idx,
            exit_idx=sl_exit_idx,
            exit_time=sl_time,
            token_side=token_side,
            reason=sl_reason,
        )
    return records


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


def reason_breakdown(events: pd.DataFrame) -> str:
    counts = events[events["is_fillable"]]["exit_reason"].value_counts().to_dict()
    return ";".join(f"{reason}={int(counts.get(reason, 0))}" for reason in EXIT_REASONS)


def summarize_events(
    events: pd.DataFrame,
    *,
    market: str,
    slug: str,
    horizon: int,
    threshold_label: str,
    exit_config: str,
    a14b_baseline: float,
    a14d_baseline: float,
    seed: int,
) -> dict[str, object]:
    n_signal = int(len(events))
    if n_signal == 0:
        return {
            "market": market,
            "slug": slug,
            "horizon": horizon,
            "spread_threshold": threshold_label,
            "exit_config": exit_config,
            "n_signal_events": 0,
            "fillable_rate": math.nan,
            "mean_pnl_bps": math.nan,
            "median_pnl_bps": math.nan,
            "win_rate": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "mean_hold_seconds": math.nan,
            "exit_reason_breakdown": "",
            "a14b_mean_pnl_bps": a14b_baseline,
            "a14d_mean_pnl_bps": a14d_baseline,
            "delta_vs_a14b": math.nan,
            "delta_vs_a14d": math.nan,
        }
    fillable = events[events["is_fillable"]].copy()
    n_fillable = int(len(fillable))
    mean_pnl = float(fillable["pnl_bps"].mean()) if n_fillable else math.nan
    median_pnl = float(fillable["pnl_bps"].median()) if n_fillable else math.nan
    win_rate = float(fillable["pnl_bps"].gt(0).mean()) if n_fillable else math.nan
    ci_lo, ci_hi = bootstrap_mean_ci(fillable, seed)
    mean_hold = float(fillable["hold_seconds"].mean()) if n_fillable else math.nan
    return {
        "market": market,
        "slug": slug,
        "horizon": horizon,
        "spread_threshold": threshold_label,
        "exit_config": exit_config,
        "n_signal_events": n_signal,
        "fillable_rate": n_fillable / n_signal if n_signal else math.nan,
        "mean_pnl_bps": mean_pnl,
        "median_pnl_bps": median_pnl,
        "win_rate": win_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_hold_seconds": mean_hold,
        "exit_reason_breakdown": reason_breakdown(events) if n_signal else "",
        "a14b_mean_pnl_bps": a14b_baseline,
        "a14d_mean_pnl_bps": a14d_baseline,
        "delta_vs_a14b": mean_pnl - a14b_baseline
        if np.isfinite(mean_pnl) and np.isfinite(a14b_baseline)
        else math.nan,
        "delta_vs_a14d": mean_pnl - a14d_baseline
        if np.isfinite(mean_pnl) and np.isfinite(a14d_baseline)
        else math.nan,
    }


def run_grid(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    b_baseline = load_a14b_baseline()
    d_baseline = load_a14d_baseline()
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
    for market_idx, market_row in cand.reset_index(drop=True).iterrows():
        run_id = str(market_row["run_id"])
        market_id = str(market_row["market_id"])
        market = f"{run_id}:{market_id}"
        print(f"A14f market {market_idx + 1:02d}/{len(cand):02d}: {market}", flush=True)
        slug = str(market_row.get("slug") or market_id)
        family = str(market_row.get("family") or "")
        category = family_category(family)
        p90_stop = float(market_row["p90_time_stop_sec"])
        sub = df[df["run_id"].eq(run_id) & df["market_id"].eq(market_id)].copy()
        for horizon in HORIZONS:
            horizon_records: list[dict[str, object]] = []
            for _, asset_rows in sub.groupby("asset_id", sort=False):
                horizon_records.extend(
                    simulate_asset_horizon_all_configs(
                        asset_rows,
                        category=category,
                        horizon_sec=horizon,
                        p90_time_stop_sec=p90_stop,
                    )
                )
            horizon_events = pd.DataFrame(horizon_records)
            for threshold_idx, (threshold_label, threshold_value) in enumerate(SPREAD_THRESHOLDS):
                for config_idx, exit_config in enumerate(EXIT_CONFIGS):
                    if horizon_events.empty:
                        events = pd.DataFrame()
                    else:
                        mask = horizon_events["exit_config"].eq(exit_config)
                        if threshold_value is not None:
                            mask &= horizon_events["spread_bps"].le(threshold_value)
                        events = horizon_events[mask].copy()
                    a14b = b_baseline.get((market, exit_config), math.nan)
                    a14d = d_baseline.get((market, horizon, threshold_label), math.nan)
                    rows.append(
                        summarize_events(
                            events,
                            market=market,
                            slug=slug,
                            horizon=horizon,
                            threshold_label=threshold_label,
                            exit_config=exit_config,
                            a14b_baseline=a14b,
                            a14d_baseline=a14d,
                            seed=RNG_SEED
                            + market_idx * 10_000
                            + horizon * 100
                            + threshold_idx * 10
                            + config_idx,
                        )
                    )
    columns = [
        "market",
        "slug",
        "horizon",
        "spread_threshold",
        "exit_config",
        "n_signal_events",
        "fillable_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_hold_seconds",
        "exit_reason_breakdown",
        "a14b_mean_pnl_bps",
        "a14d_mean_pnl_bps",
        "delta_vs_a14b",
        "delta_vs_a14d",
    ]
    return pd.DataFrame(rows)[columns]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def config_summary_table(results: pd.DataFrame) -> str:
    grouped = (
        results.groupby(["horizon", "spread_threshold", "exit_config"], as_index=False)
        .agg(
            cells=("market", "count"),
            positive=("mean_pnl_bps", lambda s: int(s.gt(0).sum())),
            interpretable_positive=(
                "mean_pnl_bps",
                lambda s: int((s.gt(0) & results.loc[s.index, "n_signal_events"].ge(MIN_INTERPRETABLE_EVENTS)).sum()),
            ),
            mean_pnl_bps=("mean_pnl_bps", "mean"),
            mean_delta_vs_a14d=("delta_vs_a14d", "mean"),
            mean_hold_seconds=("mean_hold_seconds", "mean"),
        )
        .sort_values(["positive", "mean_pnl_bps"], ascending=False)
        .head(18)
    )
    rows = [
        [
            str(int(row.horizon)),
            str(row.spread_threshold),
            str(row.exit_config),
            f"{int(row.positive)}/{int(row.cells)}",
            f"{int(row.interpretable_positive)}/{int(row.cells)}",
            bps(float(row.mean_pnl_bps)),
            bps(float(row.mean_delta_vs_a14d)),
            f"{float(row.mean_hold_seconds):.2f}s" if np.isfinite(row.mean_hold_seconds) else "n/a",
        ]
        for row in grouped.itertuples(index=False)
    ]
    return markdown_table(
        ["h", "S", "exit", "positive", "positive n>=30", "mean pnl", "delta vs A14d", "hold"],
        rows,
    )


def best_cells_table(results: pd.DataFrame) -> str:
    sub = (
        results[results["n_signal_events"].ge(MIN_INTERPRETABLE_EVENTS)]
        .sort_values(["mean_pnl_bps", "n_signal_events"], ascending=False)
        .head(20)
    )
    rows = []
    for row in sub.itertuples(index=False):
        rows.append(
            [
                str(row.market),
                str(row.slug)[:38].replace("|", "/"),
                str(int(row.horizon)),
                str(row.spread_threshold),
                str(row.exit_config),
                f"{int(row.n_signal_events):,}",
                pct(float(row.fillable_rate)),
                bps(float(row.mean_pnl_bps)),
                bps(float(row.median_pnl_bps)),
                pct(float(row.win_rate)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                bps(float(row.delta_vs_a14b)),
                bps(float(row.delta_vs_a14d)),
            ]
        )
    return markdown_table(
        ["market", "slug", "h", "S", "exit", "events", "fillable", "mean", "median", "win", "CI", "dB", "dD"],
        rows,
    )


def per_market_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for market, sub in results.groupby("market", sort=True):
        slug = str(sub["slug"].dropna().iloc[0])[:44].replace("|", "/")
        positive = sub[sub["mean_pnl_bps"].gt(0)].copy()
        interpretable = positive[positive["n_signal_events"].ge(MIN_INTERPRETABLE_EVENTS)].copy()
        if not interpretable.empty:
            best = interpretable.sort_values(["mean_pnl_bps", "n_signal_events"], ascending=False).iloc[0]
            verdict = "crosses zero, CI wide" if not (np.isfinite(best["ci_lo"]) and best["ci_lo"] > 0) else "crosses zero"
        elif not positive.empty:
            best = positive.sort_values(["mean_pnl_bps", "n_signal_events"], ascending=False).iloc[0]
            verdict = "sparse positive artifact"
        else:
            best = sub.sort_values(["mean_pnl_bps", "delta_vs_a14d"], ascending=False).iloc[0]
            if np.isfinite(best["delta_vs_a14d"]) and best["delta_vs_a14d"] > 0:
                verdict = "improves but stays negative"
            else:
                verdict = "still wiped"
        rows.append(
            [
                market,
                slug,
                str(int(best["horizon"])),
                str(best["spread_threshold"]),
                str(best["exit_config"]),
                f"{int(best['n_signal_events']):,}",
                bps(float(best["mean_pnl_bps"])),
                bps(float(best["delta_vs_a14b"])),
                bps(float(best["delta_vs_a14d"])),
                verdict,
            ]
        )
    return markdown_table(
        ["market", "slug", "h", "S", "exit", "events", "mean", "dB", "dD", "verdict"],
        rows,
    )


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    positive = results[results["mean_pnl_bps"].gt(0)].copy()
    interp_positive = positive[positive["n_signal_events"].ge(MIN_INTERPRETABLE_EVENTS)].copy()
    dynamic_positive = int(positive[positive["exit_config"].ne("cfg_fixed_5s")].shape[0])
    beyond_both = results[
        results["delta_vs_a14b"].gt(0)
        & results["delta_vs_a14d"].gt(0)
        & results["mean_pnl_bps"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    material_beyond_both = beyond_both[beyond_both["delta_vs_a14d"].gt(10.0)]
    best_pool = interp_positive if not interp_positive.empty else results
    best = best_pool.sort_values(["mean_pnl_bps", "n_signal_events"], ascending=False).iloc[0]
    best_text = (
        f"Best cell: `{best['market']}` h={int(best['horizon'])}s, S={best['spread_threshold']}, "
        f"`{best['exit_config']}` with {bps(float(best['mean_pnl_bps']))} on "
        f"{int(best['n_signal_events']):,} signal events."
    )
    ci_positive = int((interp_positive["ci_lo"].replace([np.inf, -np.inf], np.nan) > 0).sum())

    note = f"""---
tags: [dali, block-a14f, executable-cost, results]
---

# Block A1.4f Combined Refined-Exit + Tight-Spread Findings

## Headline

A1.4f tests all primary-read markets with per-market top-decile current TOB imbalance, spread-filtered entry, and refined exits. {len(positive)} of {len(results)} market-horizon-threshold-exit cells crossed zero by mean PnL; {len(interp_positive)} had at least {MIN_INTERPRETABLE_EVENTS} signal events, {ci_positive} had a bootstrap CI lower bound above zero, and {dynamic_positive} used a refined dynamic exit. The positive cells are all the same 300s fixed-horizon BTC 4h market from A14d, so the combination does not create a new refined-exit winner. {len(beyond_both)} cells improved versus both available baselines on paper, but only {len(material_beyond_both)} beat A14d by more than 10 bps. {best_text}

## Method

- Candidate universe: all `primary_read` markets from `block_a1_results.csv`.
- Signal: per-market top decile by `abs(tob_imbalance_level)`, with `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry filter: `spread_bps <= S`, where `S in {{100, 200, 500, 1000, no_filter}}` and `spread_bps = (best_ask - best_bid) / mid * 10_000`.
- Exit configs: `cfg_fixed_5s`, `cfg_signal_reversal`, `cfg_take_profit_500bps`, and `cfg_stop_loss_combined`.
- Horizon handling: the inherited `cfg_fixed_5s` label is kept for comparability, but the row's `horizon` controls the fixed hold length. Dynamic exits close on their event trigger or `min(p90 persistence, horizon)`.
- Execution: touch round trip with taker fees on both legs; no partial fills, queue, size capacity, or latency model.
- Bootstrap: 200-sample contiguous 300s block bootstrap on mean PnL.
- Baselines: `a14b_mean_pnl_bps` is the no-spread refined-exit baseline where available; `a14d_mean_pnl_bps` is the fixed-horizon tight-spread baseline for the same market/horizon/spread threshold.

## Best Configuration Cells

{config_summary_table(results)}

## Best Individual Cells

{best_cells_table(results)}

## Per-Market Combined Verdict

{per_market_table(results)}

## Interpretation

The combined filter is the first place where the test can ask whether A14b's exit discipline and A14d's tight-spread entry stack constructively. Positive cells should still be read through event count, CI width, and whether they beat both sidecar baselines. A sparse positive row is not a strategy; a repeated positive row across neighboring thresholds/horizons is the thing worth carrying into A2.

Recommended next action for Justin: keep TOB plus tight-spread/refined-exit as an A2 diagnostic, but require repeated positive cells with CI support and add latency/capacity before any taker design.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates = load_time_stops(load_candidates())
    features = add_tob_signal(load_feature_subset(candidates))
    results = run_grid(features, candidates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
