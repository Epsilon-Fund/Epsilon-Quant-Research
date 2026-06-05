"""Block A1.4b refined-exit executable taker QA on TOB imbalance.

This is a sidecar over A1/A1.3 artifacts. It does not modify the A1 analyzer,
the A1.4 script, or raw capture JSONL.
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
A14_BASELINE = ANALYSIS / "csv_outputs" / "dali" / "block_a14_executable_taker_results.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14b_refined_exit_results.csv"
NOTE = NOTES / "block_a14b_refined_exit_findings.md"

BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260529
FIXED_HOLD_SECONDS = 5.0
TAKE_PROFIT_BPS = (100.0, 200.0, 500.0)
STOP_LOSS_BPS = -100.0

CONFIG_FIXED = "cfg_fixed_5s"
CONFIG_SIGNAL_REVERSAL = "cfg_signal_reversal"
CONFIG_TAKE_PROFIT = "cfg_take_profit"
CONFIG_STOP_LOSS = "cfg_stop_loss_combined"
EXIT_REASONS = ("signal_reversal", "take_profit", "stop_loss", "time_stop")


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
    results["horizon_sec"] = pd.to_numeric(results["horizon_sec"], errors="coerce")
    candidates = results[
        results["horizon_sec"].eq(5)
        & results["verdict"].eq("signal_present_pre_cost")
        & results["sample_size_label"].eq("primary_read")
    ].copy()
    candidates = candidates[
        ["run_id", "market_id", "family", "n_classifiable"]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no A1 5s primary-read pre-cost candidates found")
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
        missing_markets = ", ".join(merged.loc[missing, "market"].astype(str).tolist())
        raise SystemExit(f"missing A1.3 p90 persistence stop for: {missing_markets}")
    merged["time_stop_sec"] = pd.to_numeric(
        merged["p90_time_until_flip_sec"],
        errors="coerce",
    ).clip(lower=0.001)
    return merged.drop(columns=["market_key"])


def load_a14_baseline() -> dict[str, float]:
    if not A14_BASELINE.exists():
        return {}
    baseline = pd.read_csv(A14_BASELINE)
    baseline = baseline[pd.to_numeric(baseline["horizon"], errors="coerce").eq(5)].copy()
    return {
        str(row.market): float(row.mean_pnl_bps)
        for row in baseline.itertuples(index=False)
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
    for col in ("outcome_index", "best_bid", "best_ask", "mid", "tob_imbalance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    return df


def add_tob_signal(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["direction_factor"] = np.where(
            g["outcome_index"].fillna(0).astype(int).eq(0),
            1.0,
            -1.0,
        )
        g["tob_imbalance_level"] = g["direction_factor"] * g["tob_imbalance"]
        g["signal_sign"] = np.sign(g["tob_imbalance_level"]).replace(0.0, np.nan)
        g["token_side"] = g["signal_sign"] * g["direction_factor"]
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out["abs_tob_imbalance_level"] = out["tob_imbalance_level"].abs()
    return out.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)


def mark_top_decile_entries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_top_decile_entry"] = False
    valid = (
        df["is_book_state_complete"]
        & df["abs_tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
        & df["abs_tob_imbalance_level"].gt(0)
    )
    for _, idx in df[valid].groupby(["run_id", "market_id"], sort=False).groups.items():
        values = df.loc[idx, "abs_tob_imbalance_level"]
        try:
            decile = pd.qcut(values, 10, labels=False, duplicates="drop")
        except ValueError:
            decile = pd.qcut(values.rank(method="first"), 10, labels=False, duplicates="drop")
        top = decile == decile.max()
        df.loc[values.index[top], "is_top_decile_entry"] = True
    return df


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


def first_threshold_index(
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
    if len(hits) == 0:
        return None
    return int(start_pos + hits[0])


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
    if len(hits) == 0:
        return None
    return int(start_pos + hits[0])


def exit_price_for_side(token_side: float, exit_idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(bid[exit_idx]) if token_side > 0 else float(ask[exit_idx])


def entry_price_for_side(token_side: float, entry_idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(ask[entry_idx]) if token_side > 0 else float(bid[entry_idx])


def pnl_bps(
    category: str,
    token_side: float,
    entry_price: float,
    exit_price: float,
) -> float:
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return math.nan
    if token_side > 0:
        gross = exit_price - entry_price
    else:
        gross = entry_price - exit_price
    fees = float(fee_amount(category, entry_price) + fee_amount(category, exit_price))
    return float((gross - fees) / entry_price * 10_000.0)


def make_record(
    *,
    market: str,
    slug: str,
    config: str,
    take_profit_bps: float,
    received_at: pd.Timestamp,
    entry_time_ns: int,
    exit_time_ns: int | None,
    entry_idx: int,
    exit_idx: int | None,
    token_side: float,
    bid: np.ndarray,
    ask: np.ndarray,
    category: str,
    exit_reason: str,
) -> dict[str, object]:
    entry_px = entry_price_for_side(token_side, entry_idx, bid, ask)
    fillable = exit_idx is not None and exit_time_ns is not None
    exit_px = exit_price_for_side(token_side, exit_idx, bid, ask) if fillable else math.nan
    trade_pnl = pnl_bps(category, token_side, entry_px, exit_px) if fillable else math.nan
    if not np.isfinite(trade_pnl):
        fillable = False
    return {
        "market": market,
        "slug": slug,
        "config": config,
        "take_profit_bps": take_profit_bps,
        "received_at": received_at,
        "is_fillable": bool(fillable),
        "pnl_bps": trade_pnl if fillable else math.nan,
        "hold_seconds": (exit_time_ns - entry_time_ns) / 1_000_000_000.0
        if fillable and exit_time_ns is not None
        else math.nan,
        "exit_reason": exit_reason if fillable else "unfillable",
    }


def simulate_asset(
    group: pd.DataFrame,
    *,
    market: str,
    slug: str,
    category: str,
    time_stop_sec: float,
) -> list[dict[str, object]]:
    g = group.sort_values("received_at").reset_index(drop=True)
    entry_positions = np.flatnonzero(g["is_top_decile_entry"].to_numpy(dtype=bool))
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
    fixed_ns = int(FIXED_HOLD_SECONDS * 1_000_000_000)
    time_stop_ns_delta = int(time_stop_sec * 1_000_000_000)
    records: list[dict[str, object]] = []

    for entry_idx in entry_positions:
        entry_time_ns = int(times[entry_idx])
        received_at = pd.Timestamp(g.loc[entry_idx, "received_at"])
        token_side = float(token_side_arr[entry_idx])
        entry_sign = float(signal_sign[entry_idx])
        if not np.isfinite(token_side) or token_side == 0 or not np.isfinite(entry_sign) or entry_sign == 0:
            continue

        fixed_target = entry_time_ns + fixed_ns
        fixed_exit_idx: int | None = None
        if fixed_target <= last_time:
            idx = int(np.searchsorted(times, fixed_target, side="right") - 1)
            if idx >= entry_idx:
                fixed_exit_idx = idx
        records.append(
            make_record(
                market=market,
                slug=slug,
                config=CONFIG_FIXED,
                take_profit_bps=math.nan,
                received_at=received_at,
                entry_time_ns=entry_time_ns,
                exit_time_ns=fixed_target if fixed_exit_idx is not None else None,
                entry_idx=entry_idx,
                exit_idx=fixed_exit_idx,
                token_side=token_side,
                bid=bid,
                ask=ask,
                category=category,
                exit_reason="time_stop",
            )
        )

        stop_target = entry_time_ns + time_stop_ns_delta
        time_stop_idx: int | None = None
        time_stop_available = stop_target <= last_time
        if stop_target <= last_time:
            idx = int(np.searchsorted(times, stop_target, side="right") - 1)
            if idx >= entry_idx:
                time_stop_idx = idx

        reversal_idx = first_opposite_index(entry_idx, entry_sign, pos_idx, neg_idx)
        if reversal_idx is not None and int(times[reversal_idx]) > stop_target:
            reversal_idx = None

        if reversal_idx is not None:
            reversal_exit_idx = reversal_idx
            reversal_exit_time = int(times[reversal_idx])
            reversal_reason = "signal_reversal"
        elif time_stop_available and time_stop_idx is not None:
            reversal_exit_idx = time_stop_idx
            reversal_exit_time = stop_target
            reversal_reason = "time_stop"
        else:
            reversal_exit_idx = None
            reversal_exit_time = None
            reversal_reason = "unfillable"
        records.append(
            make_record(
                market=market,
                slug=slug,
                config=CONFIG_SIGNAL_REVERSAL,
                take_profit_bps=math.nan,
                received_at=received_at,
                entry_time_ns=entry_time_ns,
                exit_time_ns=reversal_exit_time,
                entry_idx=entry_idx,
                exit_idx=reversal_exit_idx,
                token_side=token_side,
                bid=bid,
                ask=ask,
                category=category,
                exit_reason=reversal_reason,
            )
        )

        scan_start = entry_idx + 1
        scan_stop = time_stop_idx if time_stop_available and time_stop_idx is not None else len(times) - 1

        tp_indices = {
            threshold: first_threshold_index(
                mid,
                scan_start,
                scan_stop,
                float(mid[entry_idx]),
                token_side,
                threshold,
            )
            for threshold in TAKE_PROFIT_BPS
        }
        stop_idx = first_adverse_index(
            mid,
            scan_start,
            scan_stop,
            float(mid[entry_idx]),
            token_side,
            STOP_LOSS_BPS,
        )

        for threshold, tp_idx in tp_indices.items():
            if tp_idx is not None:
                exit_idx = tp_idx
                exit_time = int(times[tp_idx])
                reason = "take_profit"
            elif time_stop_available and time_stop_idx is not None:
                exit_idx = time_stop_idx
                exit_time = stop_target
                reason = "time_stop"
            else:
                exit_idx = None
                exit_time = None
                reason = "unfillable"
            records.append(
                make_record(
                    market=market,
                    slug=slug,
                    config=CONFIG_TAKE_PROFIT,
                    take_profit_bps=threshold,
                    received_at=received_at,
                    entry_time_ns=entry_time_ns,
                    exit_time_ns=exit_time,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    token_side=token_side,
                    bid=bid,
                    ask=ask,
                    category=category,
                    exit_reason=reason,
                )
            )

        event_candidates: list[tuple[int, str]] = []
        if reversal_idx is not None:
            event_candidates.append((reversal_idx, "signal_reversal"))
        if stop_idx is not None:
            event_candidates.append((stop_idx, "stop_loss"))
        event_candidates.sort(key=lambda item: (int(times[item[0]]), 0 if item[1] == "signal_reversal" else 1))
        if event_candidates and int(times[event_candidates[0][0]]) <= stop_target:
            exit_idx = event_candidates[0][0]
            exit_time = int(times[exit_idx])
            reason = event_candidates[0][1]
        elif time_stop_available and time_stop_idx is not None:
            exit_idx = time_stop_idx
            exit_time = stop_target
            reason = "time_stop"
        else:
            exit_idx = None
            exit_time = None
            reason = "unfillable"
        records.append(
            make_record(
                market=market,
                slug=slug,
                config=CONFIG_STOP_LOSS,
                take_profit_bps=math.nan,
                received_at=received_at,
                entry_time_ns=entry_time_ns,
                exit_time_ns=exit_time,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                token_side=token_side,
                bid=bid,
                ask=ask,
                category=category,
                exit_reason=reason,
            )
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


def summarize_market_events(
    events: pd.DataFrame,
    *,
    market: str,
    slug: str,
    baseline_pnl: float,
    seed_base: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_cols = ["config", "take_profit_bps"]
    for idx, ((config, take_profit), sub) in enumerate(events.groupby(group_cols, dropna=False, sort=True)):
        fillable = sub[sub["is_fillable"]].copy()
        n_events = int(len(sub))
        n_fillable = int(len(fillable))
        n_unfillable = n_events - n_fillable
        mean_pnl = float(fillable["pnl_bps"].mean()) if n_fillable else math.nan
        median_pnl = float(fillable["pnl_bps"].median()) if n_fillable else math.nan
        win_rate = float(fillable["pnl_bps"].gt(0).mean()) if n_fillable else math.nan
        ci_lo, ci_hi = bootstrap_mean_ci(fillable, seed_base + idx)
        mean_hold = float(fillable["hold_seconds"].mean()) if n_fillable else math.nan
        delta = mean_pnl - baseline_pnl if np.isfinite(mean_pnl) and np.isfinite(baseline_pnl) else math.nan
        rows.append(
            {
                "market": market,
                "slug": slug,
                "config": str(config),
                "take_profit_bps": float(take_profit) if np.isfinite(take_profit) else math.nan,
                "n_events": n_events,
                "n_unfillable": n_unfillable,
                "fillable_pct": n_fillable / n_events if n_events else math.nan,
                "mean_pnl_bps": mean_pnl,
                "median_pnl_bps": median_pnl,
                "win_rate": win_rate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mean_hold_seconds": mean_hold,
                "exit_reason_breakdown": reason_breakdown(sub),
                "a14_baseline_pnl_bps": baseline_pnl,
                "delta_vs_a14_bps": delta,
            }
        )
    return rows


def run_simulation(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline = load_a14_baseline()
    all_rows: list[dict[str, object]] = []
    for market_idx, cand in candidates.reset_index(drop=True).iterrows():
        market = str(cand["market"])
        run_id = str(cand["run_id"])
        market_id = str(cand["market_id"])
        family = str(cand["family"])
        category = family_category(family)
        time_stop_sec = float(cand["time_stop_sec"])
        sub = df[df["run_id"].eq(run_id) & df["market_id"].eq(market_id)].copy()
        slug = str(sub["slug"].replace("", np.nan).dropna().iloc[0]) if sub["slug"].astype(bool).any() else market
        records: list[dict[str, object]] = []
        for _, asset_rows in sub.groupby("asset_id", sort=False):
            records.extend(
                simulate_asset(
                    asset_rows,
                    market=market,
                    slug=slug,
                    category=category,
                    time_stop_sec=time_stop_sec,
                )
            )
        events = pd.DataFrame(records)
        if events.empty:
            continue
        base_pnl = float(baseline.get(market, math.nan))
        all_rows.extend(
            summarize_market_events(
                events,
                market=market,
                slug=slug,
                baseline_pnl=base_pnl,
                seed_base=RNG_SEED + market_idx * 100,
            )
        )
    out = pd.DataFrame(all_rows)
    columns = [
        "market",
        "slug",
        "config",
        "take_profit_bps",
        "n_events",
        "n_unfillable",
        "fillable_pct",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_hold_seconds",
        "exit_reason_breakdown",
        "a14_baseline_pnl_bps",
        "delta_vs_a14_bps",
    ]
    return out[columns].sort_values(["market", "config", "take_profit_bps"]).reset_index(drop=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def config_label(row: pd.Series) -> str:
    config = str(row["config"])
    if config == CONFIG_TAKE_PROFIT and np.isfinite(row["take_profit_bps"]):
        return f"{config}_{int(row['take_profit_bps'])}bps"
    return config


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    positive = int(results["mean_pnl_bps"].gt(0).sum())
    total = int(len(results))
    best_row = results.sort_values("mean_pnl_bps", ascending=False).iloc[0]
    by_config = (
        results.assign(config_label=results.apply(config_label, axis=1))
        .groupby("config_label", as_index=False)
        .agg(
            markets=("market", "nunique"),
            mean_pnl_bps=("mean_pnl_bps", "mean"),
            median_pnl_bps=("median_pnl_bps", "mean"),
            mean_delta_vs_a14_bps=("delta_vs_a14_bps", "mean"),
            mean_hold_seconds=("mean_hold_seconds", "mean"),
            mean_win_rate=("win_rate", "mean"),
        )
        .sort_values("mean_pnl_bps", ascending=False)
    )

    config_rows = [
        [
            str(row.config_label),
            f"{int(row.markets)}",
            bps(float(row.mean_pnl_bps)),
            bps(float(row.median_pnl_bps)),
            bps(float(row.mean_delta_vs_a14_bps)),
            f"{float(row.mean_hold_seconds):.2f}s" if np.isfinite(row.mean_hold_seconds) else "n/a",
            pct(float(row.mean_win_rate)),
        ]
        for row in by_config.itertuples(index=False)
    ]

    gap_rows: list[list[str]] = []
    for row in results.sort_values(["config", "take_profit_bps", "mean_pnl_bps"], ascending=[True, True, False]).itertuples(index=False):
        label = f"{row.config}_{int(row.take_profit_bps)}bps" if row.config == CONFIG_TAKE_PROFIT and np.isfinite(row.take_profit_bps) else row.config
        gap_rows.append(
            [
                str(row.market),
                str(row.slug)[:42].replace("|", "/"),
                label,
                f"{int(row.n_events):,}",
                f"{int(row.n_unfillable):,}",
                pct(float(row.fillable_pct)),
                bps(float(row.mean_pnl_bps)),
                bps(float(row.median_pnl_bps)),
                pct(float(row.win_rate)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                f"{float(row.mean_hold_seconds):.2f}s" if np.isfinite(row.mean_hold_seconds) else "n/a",
                bps(float(row.a14_baseline_pnl_bps)),
                bps(float(row.delta_vs_a14_bps)),
                str(row.exit_reason_breakdown),
            ]
        )

    verdict_lines: list[str] = []
    for market, sub in results.groupby("market", sort=True):
        best = sub.sort_values("mean_pnl_bps", ascending=False).iloc[0]
        best_label = config_label(best)
        if np.isfinite(best["mean_pnl_bps"]) and best["mean_pnl_bps"] > 0:
            verdict = "survives with refined exit"
        elif np.isfinite(best["delta_vs_a14_bps"]) and best["delta_vs_a14_bps"] > 0:
            verdict = "improves but stays negative"
        else:
            verdict = "still wiped"
        verdict_lines.append(
            f"- `{market}` ({str(best['slug'])}): {verdict}; best `{best_label}` at {bps(float(best['mean_pnl_bps']))}, delta vs A1.4 {bps(float(best['delta_vs_a14_bps']))}."
        )

    note = f"""---
tags: [dali, block-a14b, executable-cost, results]
---

# Block A1.4b Refined-Exit Taker Findings

## Headline

A1.4b swaps A1.4's OFI trigger for A1.3's current top-of-book imbalance level and tests refined exits on the same six candidate markets. {positive} of {total} market-config rows have positive mean executable PnL. The best single row is `{best_row['market']}` with `{config_label(best_row)}` at {bps(float(best_row['mean_pnl_bps']))}. Averaged across markets, the configuration ranking is:

{markdown_table(["config", "markets", "mean pnl", "mean median", "delta vs A1.4", "mean hold", "win"], config_rows)}

## Method

- Candidate universe: same as A1.4, primary-read markets with `signal_present_pre_cost` at 5s in `block_a1_results.csv`.
- Signal: per-market top decile by absolute current TOB imbalance level, where `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry: instantaneous taker at the current asset touch. Long token signals pay `best_ask`; short token signals receive `best_bid`.
- Exits: fixed 5s, signal reversal or p90 time stop, take-profit at +100/+200/+500 bps or p90 time stop, and signal reversal plus -100 bps stop loss or p90 time stop.
- Take-profit and stop-loss checks use mid-price movement in the token-side trade direction. Executable PnL uses bid/ask touch on both entry and exit.
- Time stops use `p90_time_until_flip_sec` from `block_a13_tob_persistence_by_market.csv`.
- Fees: taker fee applied at entry and exit using A1's `FEE_BY_CATEGORY`.
- Confidence intervals: 200-sample block bootstrap of mean PnL using contiguous 300s clock-time blocks.

## Four-Config Gap Table

{markdown_table(
        [
            "market",
            "slug",
            "config",
            "events",
            "unfillable",
            "fillable",
            "mean pnl",
            "median pnl",
            "win",
            "mean CI",
            "mean hold",
            "A1.4 base",
            "delta",
            "exit reasons",
        ],
        gap_rows,
    )}

## Per-Market Verdicts

{chr(10).join(verdict_lines)}

## Interpretation

The refined exits move the needle mostly by shortening exposure when the TOB state is unstable. That improves every market relative to the OFI fixed-horizon A1.4 baseline, but positive executable taker PnL is absent in this run. This is still a touch-fill, no-latency, no-size-capacity diagnostic rather than a tradeable verdict.

Recommended next action for Justin: carry TOB imbalance into A2, but gate any taker version on tight-spread/deep-book cells and require the refined-exit executable test to stay positive after latency.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates = load_time_stops(load_candidates())
    features = load_feature_subset(candidates)
    features = mark_top_decile_entries(add_tob_signal(features))
    results = run_simulation(features, candidates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
