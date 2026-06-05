"""Block K2 logit-space Avellaneda-Stoikov quoting simulator.

Research-only sidecar. Uses pooled A1 feature captures as one in-sample set,
then optimizes a lightweight logit-space market-making simulator with Optuna.

The simulator intentionally remains queue-blind and small-lot: every qualifying
fill is one unit, entry is passive maker, and exit is taker after a short hold
or before resolution. It is a ceiling test, not a deployability claim.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k2_quoting_sim.csv"
NOTE = NOTES / "block_k2_quoting_findings.md"

RUN_POOL = ("a0", "a0b", "a0c", "a0c_roll")
EXCLUDED_SLUG_SUBSTRINGS = ("will-jd-vance",)
FILL_WINDOW_SEC = 5
HOLD_SEC = 60
RESOLUTION_BUFFER_SEC = 10
ROLLING_SIGMA_WINDOW = "300s"
N_TRIALS_STAGE_A = 80
N_TRIALS_STAGE_B = 80
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260530
ROBUST_MIN_FILLS = 30
CONTRARIAN_SKEW_X = 0.03

FEE_BY_CATEGORY = {
    "Crypto": {"fee_rate": 0.07, "maker_rebate_pct": 0.20},
    "Sports": {"fee_rate": 0.03, "maker_rebate_pct": 0.25},
    "Finance": {"fee_rate": 0.04, "maker_rebate_pct": 0.25},
    "Politics": {"fee_rate": 0.04, "maker_rebate_pct": 0.25},
    "Economics": {"fee_rate": 0.05, "maker_rebate_pct": 0.25},
    "Culture": {"fee_rate": 0.05, "maker_rebate_pct": 0.25},
    "Weather": {"fee_rate": 0.05, "maker_rebate_pct": 0.25},
    "Tech": {"fee_rate": 0.04, "maker_rebate_pct": 0.25},
    "Other": {"fee_rate": 0.05, "maker_rebate_pct": 0.25},
    "Geopolitics": {"fee_rate": 0.00, "maker_rebate_pct": 0.00},
}


@dataclass(frozen=True)
class Params:
    gamma: float
    base_spread_bps: float
    inventory_cap: int
    widening_strength: float


@dataclass
class SimResult:
    fills: pd.DataFrame
    summary: dict[str, Any]


def family_category(family: object) -> str:
    fam = str(family or "").lower()
    if "crypto" in fam:
        return "Crypto"
    if "sport" in fam or "nba" in fam or "ucl" in fam or "nhl" in fam:
        return "Sports"
    if "geopolitics" in fam or "iran" in fam:
        return "Geopolitics"
    if "politics" in fam:
        return "Politics"
    if "equity" in fam or "stock" in fam or "finance" in fam:
        return "Finance"
    if "weather" in fam:
        return "Weather"
    if "culture" in fam:
        return "Culture"
    if "econom" in fam:
        return "Economics"
    if "ai" in fam or "tech" in fam or "product" in fam:
        return "Tech"
    return "Other"


def fee_segment(category: str) -> str:
    return "fee_free" if category == "Geopolitics" else "fee_enabled"


def logit(p: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(p, 0.001, 0.999)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def safe_text(value: object, limit: int = 60) -> str:
    text = str(value or "").replace("|", "/")
    return text[:limit]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def state_at_or_before(
    state_times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(state_times, target_times, side="right") - 1
    out = np.full(len(target_times), np.nan, dtype=float)
    valid = (idx >= 0) & (idx < len(values))
    out[valid] = values[idx[valid]]
    return out


def bootstrap_mean_ci(rows: pd.DataFrame, value_col: str, seed: int) -> tuple[float, float]:
    clean = rows[["market_id", "fill_time_ns", value_col]].dropna().reset_index(drop=True)
    clean = clean[np.isfinite(clean[value_col])]
    if len(clean) < 5:
        return math.nan, math.nan

    block_labels: list[str] = []
    for market_id, piece in clean.groupby("market_id", sort=False):
        elapsed = (piece["fill_time_ns"] - piece["fill_time_ns"].min()) / 1_000_000_000.0
        buckets = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int)
        block_labels.extend([f"{market_id}:{bucket}" for bucket in buckets])
    clean["block_id"] = block_labels

    block_groups = [idx.to_numpy() for _, idx in clean.groupby("block_id", sort=False).groups.items()]
    if len(block_groups) < 2:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    vals = clean[value_col].to_numpy(dtype=float)
    means: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_blocks = rng.integers(0, len(block_groups), size=len(block_groups))
        idx = np.concatenate([block_groups[i] for i in sampled_blocks])
        means.append(float(np.nanmean(vals[idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def load_market_list(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    run_values = ", ".join(f"'{run}'" for run in RUN_POOL)
    excluded = " AND ".join(
        f"lower(coalesce(slug, '')) NOT LIKE '%{needle}%'" for needle in EXCLUDED_SLUG_SUBSTRINGS
    )
    query = f"""
        SELECT
            market_id,
            any_value(slug) AS slug,
            any_value(question) AS question,
            any_value(family) AS family,
            string_agg(DISTINCT run_id, ', ' ORDER BY run_id) AS runs,
            count(*) AS n_rows,
            sum(CASE WHEN event_type = 'last_trade_price' THEN 1 ELSE 0 END) AS n_trade_rows
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({run_values})
          AND {excluded}
        GROUP BY market_id
        ORDER BY market_id
    """
    return con.execute(query).df()


def load_market_features(con: duckdb.DuckDBPyConnection, market_id: str) -> pd.DataFrame:
    run_values = ", ".join(f"'{run}'" for run in RUN_POOL)
    excluded = " AND ".join(
        f"lower(coalesce(slug, '')) NOT LIKE '%{needle}%'" for needle in EXCLUDED_SLUG_SUBSTRINGS
    )
    query = f"""
        SELECT
            run_id,
            received_at,
            event_type,
            asset_id,
            market_id,
            family,
            slug,
            question,
            outcome_index,
            is_book_state_complete,
            best_bid,
            best_ask,
            mid,
            tob_imbalance,
            trade_price,
            trade_side,
            last_trade_side,
            trade_size,
            transaction_hash,
            market_resolved_at
        FROM read_parquet('{FEATURES}')
        WHERE run_id IN ({run_values})
          AND market_id = ?
          AND {excluded}
        ORDER BY asset_id, received_at
    """
    df = con.execute(query, [market_id]).df()
    if df.empty:
        return df
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    for col in ("outcome_index", "best_bid", "best_ask", "mid", "tob_imbalance", "trade_price", "trade_size"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_side_norm"] = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    df["transaction_hash"] = df["transaction_hash"].fillna("").astype(str)
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    return df


def add_quote_state_metrics(quote_state: pd.DataFrame) -> pd.DataFrame:
    q = quote_state.copy()
    q["quote_time_ns"] = q["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    q["x_mid"] = logit(q["mid"].to_numpy(dtype=float))
    q["signal_current"] = q["direction_factor"].to_numpy(dtype=float) * q["tob_imbalance"].fillna(0.0).to_numpy(dtype=float)
    q["abs_signal"] = np.abs(q["signal_current"].to_numpy(dtype=float))
    q90 = float(np.nanquantile(q["abs_signal"], 0.90)) if len(q) else math.nan
    q["signal_threshold"] = q90 if np.isfinite(q90) and q90 > 0 else math.inf

    pieces: list[pd.DataFrame] = []
    for _, piece in q.groupby("asset_id", sort=False):
        g = piece.sort_values("received_at").copy()
        g = g.set_index("received_at", drop=False)
        dx = g["x_mid"].diff()
        dt = g["quote_time_ns"].diff() / 1_000_000_000.0
        step_var = (dx * dx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        step_time = dt.replace([np.inf, -np.inf], np.nan).clip(lower=0.001).fillna(0.001)
        var_sum = step_var.rolling(ROLLING_SIGMA_WINDOW, min_periods=2).sum()
        time_sum = step_time.rolling(ROLLING_SIGMA_WINDOW, min_periods=2).sum()
        g["sigma2_per_sec"] = (var_sum / time_sum).replace([np.inf, -np.inf], np.nan)
        median_sigma = float(g["sigma2_per_sec"].median()) if g["sigma2_per_sec"].notna().any() else 1e-5
        g["sigma2_per_sec"] = g["sigma2_per_sec"].fillna(median_sigma).clip(1e-7, 0.05)
        pieces.append(g.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True) if pieces else q


def candidate_trades_for_market(df: pd.DataFrame, meta: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    family = str(meta["family"] or "")
    category = family_category(family)
    resolved_at = df["market_resolved_at"].dropna()
    resolution_ns = int(resolved_at.min().to_datetime64().astype("datetime64[ns]").astype("int64")) if not resolved_at.empty else 0

    out_parts: list[pd.DataFrame] = []
    n_quote_events = 0
    for asset_id, asset_rows in df.groupby("asset_id", sort=False):
        asset_rows = asset_rows.sort_values("received_at").copy()
        state = asset_rows[
            asset_rows["is_book_state_complete"].fillna(False)
            & asset_rows["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["mid"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["best_bid"].ge(0.0)
            & asset_rows["best_ask"].le(1.0)
            & asset_rows["best_ask"].gt(asset_rows["best_bid"])
            & asset_rows["mid"].gt(0.0)
            & asset_rows["mid"].lt(1.0)
        ].copy()
        quote_state = state[state["event_type"].isin(["book", "price_change"])].copy()
        quote_state = quote_state.drop_duplicates(["received_at", "best_bid", "best_ask", "mid"])
        n_quote_events += int(len(quote_state))
        if quote_state.empty:
            continue
        quote_state = add_quote_state_metrics(quote_state)
        quote_times = quote_state["quote_time_ns"].to_numpy(dtype=np.int64)
        quote_mid = quote_state["mid"].to_numpy(dtype=float)
        quote_x = quote_state["x_mid"].to_numpy(dtype=float)
        quote_sigma2 = quote_state["sigma2_per_sec"].to_numpy(dtype=float)
        quote_signal = quote_state["signal_current"].to_numpy(dtype=float)
        quote_signal_threshold = quote_state["signal_threshold"].to_numpy(dtype=float)
        quote_direction_factor = quote_state["direction_factor"].to_numpy(dtype=float)

        state_times = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        state_bid = state["best_bid"].to_numpy(dtype=float)
        state_ask = state["best_ask"].to_numpy(dtype=float)

        trades = asset_rows[
            asset_rows["event_type"].eq("last_trade_price")
            & asset_rows["trade_side_norm"].isin(["BUY", "SELL"])
            & asset_rows["trade_price"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["trade_price"].between(0.0, 1.0, inclusive="both")
            & asset_rows["trade_size"].fillna(0).gt(0)
        ].copy()
        if trades.empty:
            continue
        hash_mask = trades["transaction_hash"].ne("")
        with_hash = trades[hash_mask].drop_duplicates(
            ["asset_id", "transaction_hash", "trade_price", "trade_side_norm", "trade_size"]
        )
        no_hash = trades[~hash_mask].drop_duplicates(
            ["asset_id", "received_at", "trade_price", "trade_side_norm", "trade_size"]
        )
        trades = pd.concat([with_hash, no_hash], ignore_index=True).sort_values("received_at")
        if trades.empty:
            continue

        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        quote_idx = np.searchsorted(quote_times, trade_times, side="right") - 1
        valid = quote_idx >= 0
        if not valid.any():
            continue
        valid_idx = np.clip(quote_idx, 0, len(quote_times) - 1)
        quote_age_ns = trade_times - quote_times[valid_idx]
        valid &= quote_age_ns >= 0
        valid &= quote_age_ns <= FILL_WINDOW_SEC * 1_000_000_000
        if resolution_ns:
            valid &= trade_times < resolution_ns - RESOLUTION_BUFFER_SEC * 1_000_000_000
        if not valid.any():
            continue

        fill_times = trade_times[valid]
        exit_target = fill_times + HOLD_SEC * 1_000_000_000
        if resolution_ns:
            latest_exit = resolution_ns - RESOLUTION_BUFFER_SEC * 1_000_000_000
            exit_target = np.minimum(exit_target, latest_exit)
        valid_exit = exit_target > fill_times
        if not valid_exit.any():
            continue

        valid_positions = np.flatnonzero(valid)[valid_exit]
        valid_quote_idx = quote_idx[valid_positions]
        fill_times = trade_times[valid_positions]
        exit_target = exit_target[valid_exit]
        exit_bid = state_at_or_before(state_times, state_bid, exit_target)
        exit_ask = state_at_or_before(state_times, state_ask, exit_target)
        exit_valid = np.isfinite(exit_bid) & np.isfinite(exit_ask) & (exit_ask >= exit_bid)
        if not exit_valid.any():
            continue

        valid_positions = valid_positions[exit_valid]
        valid_quote_idx = valid_quote_idx[exit_valid]
        fill_times = fill_times[exit_valid]
        exit_target = exit_target[exit_valid]
        piece = trades.iloc[valid_positions].copy()
        piece["quote_time_ns"] = quote_times[valid_quote_idx]
        piece["fill_time_ns"] = fill_times
        piece["quote_age_sec"] = (fill_times - quote_times[valid_quote_idx]) / 1_000_000_000.0
        piece["quote_mid"] = quote_mid[valid_quote_idx]
        piece["quote_x_mid"] = quote_x[valid_quote_idx]
        piece["sigma2_per_sec"] = quote_sigma2[valid_quote_idx]
        piece["quote_signal"] = quote_signal[valid_quote_idx]
        piece["quote_signal_threshold"] = quote_signal_threshold[valid_quote_idx]
        piece["direction_factor_at_quote"] = quote_direction_factor[valid_quote_idx]
        piece["exit_time_ns"] = exit_target
        piece["exit_bid"] = exit_bid[exit_valid]
        piece["exit_ask"] = exit_ask[exit_valid]
        piece["seconds_to_resolution"] = (
            (resolution_ns - fill_times) / 1_000_000_000.0 if resolution_ns else np.nan
        )
        out_parts.append(piece)

    candidates = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    meta_out = {
        "market_id": str(meta["market_id"]),
        "slug": str(meta["slug"] or meta["market_id"]),
        "question": str(meta["question"] or ""),
        "family": family,
        "category": category,
        "runs": str(meta["runs"] or ""),
        "resolution_ns": resolution_ns,
        "n_quote_events": n_quote_events,
        "n_quote_orders": n_quote_events * 2,
    }
    if candidates.empty:
        return candidates, meta_out
    candidates["market_id"] = meta_out["market_id"]
    candidates["slug"] = meta_out["slug"]
    candidates["family"] = family
    candidates["category"] = category
    candidates["fee_segment"] = fee_segment(category)
    candidates["resolution_ns"] = resolution_ns
    candidates["is_crypto_4h"] = str(family).lower().find("crypto_4h") >= 0
    candidates = candidates.sort_values(["asset_id", "fill_time_ns", "quote_time_ns"]).reset_index(drop=True)
    return candidates, meta_out


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    markets = load_market_list(con)
    candidates: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    for idx, meta in enumerate(markets.to_dict("records"), start=1):
        df = load_market_features(con, str(meta["market_id"]))
        if df.empty:
            continue
        print(
            f"K2 precompute {idx:02d}/{len(markets):02d} {meta['market_id']} rows={len(df):,}",
            flush=True,
        )
        cand, meta_out = candidate_trades_for_market(df, pd.Series(meta))
        metas.append(meta_out)
        if not cand.empty:
            candidates.append(cand)
    con.close()
    pool = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    meta_df = pd.DataFrame(metas)
    if not pool.empty:
        pool = pool.sort_values(["market_id", "asset_id", "fill_time_ns", "quote_time_ns"]).reset_index(drop=True)
    return pool, meta_df


def quote_for_candidate(row: Any, params: Params, q: int, stage: str) -> tuple[float, float, float, float, float]:
    p = float(np.clip(row.quote_mid, 0.001, 0.999))
    x = float(row.quote_x_mid)
    sigma2 = float(np.clip(row.sigma2_per_sec, 1e-7, 0.05))
    if np.isfinite(row.seconds_to_resolution):
        tau_sec = min(HOLD_SEC, max(float(row.seconds_to_resolution) - RESOLUTION_BUFFER_SEC, 1.0))
    else:
        tau_sec = HOLD_SEC
    rx = x - q * params.gamma * sigma2 * tau_sec

    base_half_p = params.base_spread_bps / 10_000.0 * p
    base_delta_x = base_half_p / max(p * (1.0 - p), 1e-5)
    delta_x = base_delta_x + 0.5 * params.gamma * sigma2 * tau_sec

    widen_x = 0.0
    if bool(row.is_crypto_4h) and np.isfinite(row.seconds_to_resolution):
        near_50 = math.exp(-((p - 0.5) / 0.15) ** 2)
        time_left = max(float(row.seconds_to_resolution), 60.0)
        pressure = max(0.0, 1.0 - time_left / (4.0 * 3600.0)) ** 2
        pressure /= math.sqrt(max(time_left / 3600.0, 0.05))
        widen_x = params.widening_strength * near_50 * pressure
        delta_x += widen_x

    contrarian_x = 0.0
    if stage == "contrarian":
        abs_signal = abs(float(row.quote_signal))
        threshold = float(row.quote_signal_threshold)
        if np.isfinite(threshold) and abs_signal >= threshold and threshold > 0:
            token_signal_side = math.copysign(1.0, float(row.quote_signal)) * float(row.direction_factor_at_quote)
            contrarian_x = -CONTRARIAN_SKEW_X * token_signal_side
            rx += contrarian_x

    bid = float(sigmoid(rx - delta_x))
    ask = float(sigmoid(rx + delta_x))
    floor = base_half_p
    bid = min(bid, p - floor)
    ask = max(ask, p + floor)
    bid = float(np.clip(bid, 0.001, 0.999))
    ask = float(np.clip(ask, 0.001, 0.999))
    if bid >= ask:
        mid = 0.5 * (bid + ask)
        bid = float(np.clip(mid - floor, 0.001, 0.999))
        ask = float(np.clip(mid + floor, 0.001, 0.999))
    return bid, ask, delta_x, widen_x, contrarian_x


def taker_fee_bps(category: str, price: float, entry_price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.001, 0.999))
    denom = float(np.clip(entry_price, 0.01, 0.99))
    return params["fee_rate"] * p * (1.0 - p) / denom * 10_000.0


def maker_rebate_bps(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.001, 0.999))
    denom = float(np.clip(price, 0.01, 0.99))
    return params["fee_rate"] * p * (1.0 - p) * params["maker_rebate_pct"] / denom * 10_000.0


def simulate_asset(asset_candidates: pd.DataFrame, params: Params, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    open_lots: list[dict[str, Any]] = []
    cap = int(max(params.inventory_cap, 1))

    for row in asset_candidates.itertuples(index=False):
        now = int(row.fill_time_ns)
        if open_lots:
            open_lots = [lot for lot in open_lots if int(lot["exit_time_ns"]) > now]
        q = int(sum(int(lot["token_side"]) for lot in open_lots))
        bid, ask, delta_x, widen_x, contrarian_x = quote_for_candidate(row, params, q, stage)

        side = str(row.trade_side_norm)
        trade_price = float(row.trade_price)
        token_side = 0
        entry_price = math.nan
        exit_price = math.nan
        if side == "SELL" and q < cap and trade_price <= bid + 1e-12:
            token_side = 1
            entry_price = bid
            exit_price = float(row.exit_bid)
        elif side == "BUY" and q > -cap and trade_price >= ask - 1e-12:
            token_side = -1
            entry_price = ask
            exit_price = float(row.exit_ask)

        if token_side == 0 or not np.isfinite(exit_price) or not np.isfinite(entry_price):
            continue
        if entry_price <= 0 or exit_price < 0:
            continue

        category = str(row.category)
        gross_bps = token_side * (exit_price - entry_price) / max(entry_price, 0.01) * 10_000.0
        rebate_bps = maker_rebate_bps(category, entry_price)
        exit_fee_bps = taker_fee_bps(category, exit_price, entry_price)
        net_bps = gross_bps + rebate_bps - exit_fee_bps
        lot = {
            "exit_time_ns": int(row.exit_time_ns),
            "token_side": token_side,
        }
        open_lots.append(lot)
        rows.append(
            {
                "market_id": str(row.market_id),
                "slug": str(row.slug),
                "asset_id": str(row.asset_id),
                "family": str(row.family),
                "category": category,
                "fee_segment": str(row.fee_segment),
                "stage": stage,
                "fill_time_ns": now,
                "exit_time_ns": int(row.exit_time_ns),
                "quote_mid": float(row.quote_mid),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_price": trade_price,
                "maker_side": "BUY" if token_side > 0 else "SELL",
                "inventory_before": q,
                "quote_bid": bid,
                "quote_ask": ask,
                "delta_x": delta_x,
                "widen_x": widen_x,
                "contrarian_x": contrarian_x,
                "gross_bps": gross_bps,
                "rebate_bps": rebate_bps,
                "exit_fee_bps": exit_fee_bps,
                "net_pnl_bps": net_bps,
            }
        )
    return rows


def simulate(pool: pd.DataFrame, params: Params, stage: str, trial_number: int | None = None) -> SimResult:
    if pool.empty:
        return SimResult(pd.DataFrame(), summarize_fills(pd.DataFrame(), stage, params, trial_number))
    fill_rows: list[dict[str, Any]] = []
    for _, asset_candidates in pool.groupby(["market_id", "asset_id"], sort=False):
        fill_rows.extend(simulate_asset(asset_candidates, params, stage))
    fills = pd.DataFrame(fill_rows)
    return SimResult(fills, summarize_fills(fills, stage, params, trial_number))


def summarize_fills(
    fills: pd.DataFrame,
    stage: str,
    params: Params,
    trial_number: int | None = None,
) -> dict[str, Any]:
    n = int(len(fills))
    mean = float(fills["net_pnl_bps"].mean()) if n else math.nan
    std = float(fills["net_pnl_bps"].std(ddof=1)) if n > 1 else math.nan
    sharpe_like = mean / std * math.sqrt(n) if n > 1 and std > 0 and np.isfinite(mean) else math.nan
    total = float(fills["net_pnl_bps"].sum()) if n else 0.0
    win_rate = float(fills["net_pnl_bps"].gt(0).mean()) if n else math.nan
    return {
        "row_type": "trial",
        "stage": stage,
        "trial_number": -1 if trial_number is None else int(trial_number),
        "gamma": params.gamma,
        "base_spread_bps": params.base_spread_bps,
        "inventory_cap": params.inventory_cap,
        "widening_strength": params.widening_strength,
        "n_fills": n,
        "mean_net_pnl_bps": mean,
        "median_net_pnl_bps": float(fills["net_pnl_bps"].median()) if n else math.nan,
        "total_net_pnl_bps_units": total,
        "std_net_pnl_bps": std,
        "sharpe_like": sharpe_like,
        "win_rate": win_rate,
        "mean_gross_bps": float(fills["gross_bps"].mean()) if n else math.nan,
        "mean_rebate_bps": float(fills["rebate_bps"].mean()) if n else math.nan,
        "mean_exit_fee_bps": float(fills["exit_fee_bps"].mean()) if n else math.nan,
        "ci_lo": math.nan,
        "ci_hi": math.nan,
        "clears_zero": False,
        "objective": objective_from_summary(n, mean, std),
    }


def objective_from_summary(n: int, mean: float, std: float) -> float:
    if n < ROBUST_MIN_FILLS or not np.isfinite(mean):
        return -1e6 + n
    if not np.isfinite(std) or std <= 0:
        return mean
    return mean / std * math.sqrt(n)


def suggest_params(trial: optuna.Trial) -> Params:
    return Params(
        gamma=trial.suggest_float("gamma", 0.001, 1.0, log=True),
        base_spread_bps=trial.suggest_float("base_spread_bps", 5.0, 500.0, log=True),
        inventory_cap=trial.suggest_int("inventory_cap", 1, 5),
        widening_strength=trial.suggest_float("widening_strength", 0.0, 0.35),
    )


def optimize_stage(pool: pd.DataFrame, stage: str, n_trials: int) -> tuple[Params, pd.DataFrame, SimResult]:
    trial_rows: list[dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RNG_SEED + (0 if stage == "zero_skew" else 1000))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = simulate(pool, params, stage, trial.number)
        row = result.summary
        trial_rows.append(row)
        return float(row["objective"])

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = Params(**study.best_params)
    best_result = simulate(pool, best_params, stage, -1)
    trials = pd.DataFrame(trial_rows).sort_values("objective", ascending=False)
    return best_params, trials, best_result


def add_ci_to_summary(summary: dict[str, Any], fills: pd.DataFrame, seed: int) -> dict[str, Any]:
    out = dict(summary)
    ci_lo, ci_hi = bootstrap_mean_ci(fills, "net_pnl_bps", seed) if not fills.empty else (math.nan, math.nan)
    out["ci_lo"] = ci_lo
    out["ci_hi"] = ci_hi
    out["clears_zero"] = bool(
        int(out["n_fills"]) >= ROBUST_MIN_FILLS
        and np.isfinite(out["mean_net_pnl_bps"])
        and out["mean_net_pnl_bps"] > 0
        and np.isfinite(ci_lo)
        and ci_lo > 0
    )
    out["row_type"] = "selected"
    return out


def category_rows(fills: pd.DataFrame, selected_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if fills.empty:
        return rows
    for category, sub in fills.groupby("category", sort=True):
        n = int(len(sub))
        std = float(sub["net_pnl_bps"].std(ddof=1)) if n > 1 else math.nan
        mean = float(sub["net_pnl_bps"].mean()) if n else math.nan
        sharpe_like = mean / std * math.sqrt(n) if n > 1 and std > 0 and np.isfinite(mean) else math.nan
        ci_lo, ci_hi = bootstrap_mean_ci(sub, "net_pnl_bps", RNG_SEED + len(rows) * 11)
        rows.append(
            {
                **{k: selected_summary[k] for k in ["stage", "gamma", "base_spread_bps", "inventory_cap", "widening_strength"]},
                "row_type": "category",
                "trial_number": -1,
                "category": category,
                "fee_segment": fee_segment(str(category)),
                "n_fills": n,
                "mean_net_pnl_bps": mean,
                "median_net_pnl_bps": float(sub["net_pnl_bps"].median()) if n else math.nan,
                "total_net_pnl_bps_units": float(sub["net_pnl_bps"].sum()) if n else 0.0,
                "std_net_pnl_bps": std,
                "sharpe_like": sharpe_like,
                "win_rate": float(sub["net_pnl_bps"].gt(0).mean()) if n else math.nan,
                "mean_gross_bps": float(sub["gross_bps"].mean()) if n else math.nan,
                "mean_rebate_bps": float(sub["rebate_bps"].mean()) if n else math.nan,
                "mean_exit_fee_bps": float(sub["exit_fee_bps"].mean()) if n else math.nan,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "clears_zero": bool(n >= ROBUST_MIN_FILLS and np.isfinite(ci_lo) and ci_lo > 0),
                "objective": sharpe_like,
            }
        )
    return rows


def compare_incremental(stage_a: pd.DataFrame, stage_b: pd.DataFrame) -> dict[str, Any]:
    a = stage_a["net_pnl_bps"].to_numpy(dtype=float) if not stage_a.empty else np.array([])
    b = stage_b["net_pnl_bps"].to_numpy(dtype=float) if not stage_b.empty else np.array([])
    return {
        "row_type": "incremental",
        "stage": "contrarian_minus_zero_skew",
        "trial_number": -1,
        "gamma": math.nan,
        "base_spread_bps": math.nan,
        "inventory_cap": math.nan,
        "widening_strength": math.nan,
        "n_fills": int(len(b) - len(a)),
        "mean_net_pnl_bps": float(np.nanmean(b) - np.nanmean(a)) if len(a) and len(b) else math.nan,
        "median_net_pnl_bps": float(np.nanmedian(b) - np.nanmedian(a)) if len(a) and len(b) else math.nan,
        "total_net_pnl_bps_units": float(np.nansum(b) - np.nansum(a)),
        "std_net_pnl_bps": math.nan,
        "sharpe_like": math.nan,
        "win_rate": math.nan,
        "mean_gross_bps": math.nan,
        "mean_rebate_bps": math.nan,
        "mean_exit_fee_bps": math.nan,
        "ci_lo": math.nan,
        "ci_hi": math.nan,
        "clears_zero": False,
        "objective": math.nan,
    }


def result_rows(
    trials_a: pd.DataFrame,
    selected_a: dict[str, Any],
    fills_a: pd.DataFrame,
    trials_b: pd.DataFrame,
    selected_b: dict[str, Any],
    fills_b: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(trials_a.to_dict("records"))
    rows.extend(trials_b.to_dict("records"))
    rows.append(selected_a)
    rows.append(selected_b)
    rows.extend(category_rows(fills_a, selected_a))
    rows.extend(category_rows(fills_b, selected_b))
    rows.append(compare_incremental(fills_a, fills_b))
    out = pd.DataFrame(rows)
    if "category" not in out:
        out["category"] = ""
    if "fee_segment" not in out:
        out["fee_segment"] = ""
    order = [
        "row_type",
        "stage",
        "category",
        "fee_segment",
        "trial_number",
        "gamma",
        "base_spread_bps",
        "inventory_cap",
        "widening_strength",
        "n_fills",
        "mean_net_pnl_bps",
        "median_net_pnl_bps",
        "total_net_pnl_bps_units",
        "std_net_pnl_bps",
        "sharpe_like",
        "win_rate",
        "mean_gross_bps",
        "mean_rebate_bps",
        "mean_exit_fee_bps",
        "ci_lo",
        "ci_hi",
        "clears_zero",
        "objective",
    ]
    for col in order:
        if col not in out:
            out[col] = np.nan
    return out[order]


def write_note(results: pd.DataFrame, pool: pd.DataFrame, selected_a: dict[str, Any], selected_b: dict[str, Any]) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    a_clears = bool(selected_a["clears_zero"])
    b_clears = bool(selected_b["clears_zero"])
    gate_dead = not a_clears and not b_clears and max(float(selected_a["ci_hi"]), float(selected_b["ci_hi"])) <= 0
    gate = "maker thesis dead on this universe" if gate_dead else "maker thesis remains alive for K3/K2 stress"
    if gate_dead:
        headline = (
            "The optimized logit-space A-S ceiling is negative in pooled IS. "
            "Both zero-skew and small-contrarian stages have 95% CIs entirely below zero, "
            "so the K2 pre-commit rule kills the maker thesis on this universe."
        )
        interpretation = (
            "Unlike K1's generous midpoint spread-capture decomposition, the logit A-S quote has to earn fills away "
            "from mid and then pay taker exit costs. That optimized ceiling is still below zero. The small contrarian "
            "fade is mildly less bad, but it does not rescue the thesis."
        )
    else:
        headline = (
            "The optimized logit-space A-S ceiling remains positive in pooled IS. "
            "At least one optimized stage clears zero, so the maker thesis survives K2 as a ceiling test."
        )
        interpretation = (
            "This remains a ceiling test: no queue priority, no partial fill sizing, no quote cancellation latency, "
            "and one-unit fills. It says the logit-space maker thesis is not killed before those deployment stresses."
        )

    selected_rows: list[list[str]] = []
    for row in (selected_a, selected_b):
        selected_rows.append(
            [
                str(row["stage"]),
                f"{float(row['gamma']):.5f}",
                bps(float(row["base_spread_bps"])),
                str(int(row["inventory_cap"])),
                f"{float(row['widening_strength']):.4f}",
                f"{int(row['n_fills']):,}",
                bps(float(row["mean_net_pnl_bps"])),
                f"[{bps(float(row['ci_lo']))}, {bps(float(row['ci_hi']))}]",
                f"{float(row['sharpe_like']):.2f}",
                "yes" if bool(row["clears_zero"]) else "no",
            ]
        )

    inc = results[results["row_type"].eq("incremental")].iloc[0]
    cat = results[results["row_type"].eq("category")].copy()
    cat_rows: list[list[str]] = []
    for row in cat.sort_values(["stage", "fee_segment", "category"]).itertuples(index=False):
        cat_rows.append(
            [
                str(row.stage),
                str(row.fee_segment),
                str(row.category),
                f"{int(row.n_fills):,}",
                bps(float(row.mean_net_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                f"{float(row.sharpe_like):.2f}" if np.isfinite(row.sharpe_like) else "n/a",
                "yes" if bool(row.clears_zero) else "no",
            ]
        )

    note = f"""---
tags: [dali, block-k2, logit-as, optuna, maker-sim]
---

# Block K2 Logit-Space Quoting Findings

## Headline

{headline}

Stage B's fixed small contrarian fade has incremental total PnL of {bps(float(inc.total_net_pnl_bps_units))} units and incremental mean PnL of {bps(float(inc.mean_net_pnl_bps))} versus stage A. Pre-commit gate result: **{gate}**.

## Selected Optuna Fits

{markdown_table(
    ["stage", "gamma", "base half-spread", "cap", "widen", "fills", "mean", "95% CI", "Sharpe-like", "clears"],
    selected_rows,
)}

## Category Breakdown

{markdown_table(
    ["stage", "segment", "category", "fills", "mean", "95% CI", "Sharpe-like", "clears"],
    cat_rows,
)}

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled `{', '.join(RUN_POOL)}` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance`.
- Candidate fills: A1.4h-style passive proxy with a {FILL_WINDOW_SEC}s quote freshness window. A bid fills only on a real SELL print at or below our modeled bid; an ask fills only on a real BUY print at or above our modeled ask.
- Logit A-S quote: `x=logit(p)`, `rx=x-q*gamma*sigma^2*tau`, half-spread in logit units is a probability-floor-derived base term plus `0.5*gamma*sigma^2*tau`, then bid/ask are mapped back with sigmoid.
- Spread floor: optimized `base_spread_bps`, interpreted as displayed half-spread floor in bps of entry mid.
- Inventory cap: optimized integer cap, per asset. Quotes that would add beyond the cap are disabled.
- Crypto-4h widening: optimized `widening_strength * near_50(p) * near_resolution(tau)` added to the logit half-spread for `crypto_4h` markets.
- Volatility: rolling 300s logit-mid realized variance per second from quote-state updates.
- Exits: each fill is closed as taker after {HOLD_SEC}s or before resolution minus a {RESOLUTION_BUFFER_SEC}s buffer. Taker exit fee is charged. Entry maker rebate uses the K1 fee table.
- Stage A: no directional skew.
- Stage B: same parameter search but with a fixed small contrarian skew of {CONTRARIAN_SKEW_X:.2f} logit units when the A1 current TOB signal is in its market-level top absolute decile.
- Objective: pooled fill-level `mean(net_bps) / std(net_bps) * sqrt(n)`, with fewer than {ROBUST_MIN_FILLS} fills penalized.
- CI: 500 bootstrap resamples over contiguous 300s fill-time blocks within market.

## Interpretation

{interpretation}

The simulator is still generous: no queue priority, no partial fill sizing, no quote cancellation latency, and one-unit fills. Because even this optimized ceiling is negative, adding realistic queue and latency costs would only worsen it.

Inputs precomputed: {len(pool):,} candidate trade rows across {pool['market_id'].nunique() if not pool.empty else 0:,} markets.

CSV: `data/analysis/csv_outputs/market_making/k2_quoting_sim.csv`.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(f"missing features parquet: {FEATURES}")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    pool, _meta = build_candidate_pool()
    if pool.empty:
        raise SystemExit("candidate pool is empty")
    print(f"K2 candidate pool: {len(pool):,} rows, {pool['market_id'].nunique():,} markets")

    params_a, trials_a, best_a = optimize_stage(pool, "zero_skew", N_TRIALS_STAGE_A)
    selected_a = add_ci_to_summary(best_a.summary, best_a.fills, RNG_SEED + 1)
    print(
        f"K2 stage A best: mean={selected_a['mean_net_pnl_bps']:.2f} "
        f"CI=[{selected_a['ci_lo']:.2f}, {selected_a['ci_hi']:.2f}] fills={selected_a['n_fills']}",
        flush=True,
    )

    params_b, trials_b, best_b = optimize_stage(pool, "contrarian", N_TRIALS_STAGE_B)
    selected_b = add_ci_to_summary(best_b.summary, best_b.fills, RNG_SEED + 2)
    print(
        f"K2 stage B best: mean={selected_b['mean_net_pnl_bps']:.2f} "
        f"CI=[{selected_b['ci_lo']:.2f}, {selected_b['ci_hi']:.2f}] fills={selected_b['n_fills']}",
        flush=True,
    )

    results = result_rows(trials_a, selected_a, best_a.fills, trials_b, selected_b, best_b.fills)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results, pool, selected_a, selected_b)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
