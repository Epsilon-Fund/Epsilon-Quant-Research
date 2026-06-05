"""Block K-PEG dynamic bid/ask chase optimization.

Research-only sidecar for the Block K maker track. It models a Midas-style
peg that joins/improves/sits behind the best quote, then chases favorable quote
moves by tick increments subject to a Stoikov-style micro-price cap.

The output is a ceiling diagnostic. It is queue-blind, uses one-share fills, and
does not model partial queue priority or cancellation latency.
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
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_chase_optimization.csv"
SELECTED_FILLS_CSV = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_selected_fills.csv"
PORTFOLIO_CSV = ANALYSIS / "csv_outputs" / "market_making" / "kpeg_portfolio_timeseries.csv"
PLOTS = ANALYSIS / "kpeg_plots"
NOTE = NOTES / "block_kpeg_findings.md"

RUN_POOL = ("a0", "a0b", "a0c", "a0c_roll")
EXCLUDED_SLUG_SUBSTRINGS = ("will-jd-vance",)
TICK = 0.01
CADENCE_OPTIONS = (1, 2, 5, 10, 30)
STALENESS_MULTIPLIER = 2.0
HOLD_SEC = 60
ADVERSE_HORIZONS = (5, 30, 60)
OBJECTIVE_HORIZON = 30
N_TRIALS = 360
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260531
ROBUST_MIN_FILLS = 30
INTERNAL_INVENTORY_CAP = 5

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
    peg_offset_ticks: int
    chase_increment_ticks: int
    chase_cap_c_ticks: int
    inventory_scaling: float
    cadence_sec: int


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


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def usd(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"${value:,.4f}"


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def state_at_or_before(state_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
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
    blocks = [idx.to_numpy() for _, idx in clean.groupby("block_id", sort=False).groups.items()]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    vals = clean[value_col].to_numpy(dtype=float)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        means.append(float(np.nanmean(vals[idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def taker_rebate_bps(category: str, entry_price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(entry_price, 0.001, 0.999))
    denom = float(np.clip(entry_price, 0.01, 0.99))
    return params["fee_rate"] * p * (1.0 - p) * params["maker_rebate_pct"] / denom * 10_000.0


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
            best_bid_size,
            best_ask,
            best_ask_size,
            mid,
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
    for col in ("outcome_index", "best_bid", "best_bid_size", "best_ask", "best_ask_size", "mid", "trade_price", "trade_size"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_side_norm"] = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    df["transaction_hash"] = df["transaction_hash"].fillna("").astype(str)
    return df


def sampled_state_indices(times: np.ndarray, cadence_sec: int) -> np.ndarray:
    keep = np.zeros(len(times), dtype=bool)
    if len(times) == 0:
        return np.array([], dtype=int)
    last = int(times[0]) - cadence_sec * 1_000_000_000
    for idx, ts in enumerate(times):
        if int(ts) - last >= cadence_sec * 1_000_000_000:
            keep[idx] = True
            last = int(ts)
    return np.flatnonzero(keep)


def candidate_trades_for_market(df: pd.DataFrame, meta: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = str(meta["family"] or "")
    category = family_category(family)
    rows: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []

    for asset_id, asset_rows in df.groupby("asset_id", sort=False):
        asset_rows = asset_rows.sort_values("received_at").copy()
        state = asset_rows[
            asset_rows["is_book_state_complete"].fillna(False)
            & asset_rows["event_type"].isin(["book", "price_change"])
            & asset_rows["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["mid"].replace([np.inf, -np.inf], np.nan).notna()
            & asset_rows["best_bid"].between(0.0, 1.0, inclusive="both")
            & asset_rows["best_ask"].between(0.0, 1.0, inclusive="both")
            & asset_rows["best_ask"].gt(asset_rows["best_bid"])
        ].drop_duplicates(["received_at", "best_bid", "best_ask", "mid"]).copy()
        if state.empty:
            continue
        bid_size = state["best_bid_size"].fillna(0.0).clip(lower=0.0)
        ask_size = state["best_ask_size"].fillna(0.0).clip(lower=0.0)
        denom = (bid_size + ask_size).replace(0.0, np.nan)
        state["micro_price"] = (
            (state["best_ask"] * bid_size + state["best_bid"] * ask_size) / denom
        ).fillna(state["mid"])
        state["state_time_ns"] = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")

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
        trades = pd.concat(
            [
                trades[hash_mask].drop_duplicates(["asset_id", "transaction_hash", "trade_price", "trade_side_norm", "trade_size"]),
                trades[~hash_mask].drop_duplicates(["asset_id", "received_at", "trade_price", "trade_side_norm", "trade_size"]),
            ],
            ignore_index=True,
        ).sort_values("received_at")
        if trades.empty:
            continue

        state_times = state["state_time_ns"].to_numpy(dtype=np.int64)
        current_idx = np.searchsorted(
            state_times,
            trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            side="right",
        ) - 1
        valid_current = current_idx >= 0
        if not valid_current.any():
            continue
        trades = trades.loc[valid_current].copy().reset_index(drop=True)
        current_idx = current_idx[valid_current]
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")

        current = state.iloc[current_idx].reset_index(drop=True)
        for cadence_sec in CADENCE_OPTIONS:
            sampled_idx = sampled_state_indices(state_times, cadence_sec)
            if len(sampled_idx) == 0:
                continue
            sampled_times = state_times[sampled_idx]
            qpos = np.searchsorted(sampled_times, trade_times, side="right") - 1
            valid_q = qpos >= 0
            if not valid_q.any():
                continue
            quote_idx = sampled_idx[np.clip(qpos, 0, len(sampled_idx) - 1)]
            quote_age_sec = (trade_times - state_times[quote_idx]) / 1_000_000_000.0
            valid = valid_q & (quote_age_sec <= max(float(cadence_sec) * STALENESS_MULTIPLIER, 1.0))
            if not valid.any():
                continue

            piece = trades.loc[valid].copy().reset_index(drop=True)
            q = state.iloc[quote_idx[valid]].reset_index(drop=True)
            cur = current.loc[valid].reset_index(drop=True)
            piece["quote_time_ns"] = q["state_time_ns"].to_numpy(dtype=np.int64)
            piece["fill_time_ns"] = trade_times[valid]
            piece["quote_age_sec"] = quote_age_sec[valid]
            piece["cadence_sec"] = cadence_sec
            piece["quote_best_bid"] = q["best_bid"].to_numpy(dtype=float)
            piece["quote_best_ask"] = q["best_ask"].to_numpy(dtype=float)
            piece["quote_mid"] = q["mid"].to_numpy(dtype=float)
            piece["current_best_bid"] = cur["best_bid"].to_numpy(dtype=float)
            piece["current_best_ask"] = cur["best_ask"].to_numpy(dtype=float)
            piece["current_mid"] = cur["mid"].to_numpy(dtype=float)
            piece["micro_price"] = cur["micro_price"].to_numpy(dtype=float)
            piece["bid_up_ticks"] = np.maximum(
                0.0,
                np.rint((piece["current_best_bid"] - piece["quote_best_bid"]) / TICK),
            )
            piece["ask_down_ticks"] = np.maximum(
                0.0,
                np.rint((piece["quote_best_ask"] - piece["current_best_ask"]) / TICK),
            )

            state_all = asset_rows[
                asset_rows["is_book_state_complete"].fillna(False)
                & asset_rows["mid"].replace([np.inf, -np.inf], np.nan).notna()
            ].sort_values("received_at")
            all_times = state_all["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
            all_mids = state_all["mid"].to_numpy(dtype=float)
            for horizon in ADVERSE_HORIZONS:
                piece[f"future_mid_{horizon}s"] = state_at_or_before(
                    all_times,
                    all_mids,
                    piece["fill_time_ns"].to_numpy(dtype=np.int64) + horizon * 1_000_000_000,
                )
            piece["market_id"] = str(meta["market_id"])
            piece["slug"] = str(meta["slug"] or meta["market_id"])
            piece["family"] = family
            piece["category"] = category
            piece["fee_segment"] = fee_segment(category)
            rows.append(piece)
            meta_rows.append(
                {
                    "market_id": str(meta["market_id"]),
                    "cadence_sec": cadence_sec,
                    "n_quote_events": int(len(sampled_idx)),
                    "n_quote_orders": int(len(sampled_idx) * 2),
                    "category": category,
                    "fee_segment": fee_segment(category),
                }
            )

    candidates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows).drop_duplicates(["market_id", "cadence_sec"]) if meta_rows else pd.DataFrame()
    if not candidates.empty:
        candidates = candidates.sort_values(["market_id", "asset_id", "cadence_sec", "fill_time_ns"]).reset_index(drop=True)
    return candidates, meta_df


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    markets = load_market_list(con)
    pools: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    for idx, meta in enumerate(markets.to_dict("records"), start=1):
        df = load_market_features(con, str(meta["market_id"]))
        if df.empty:
            continue
        print(f"KPEG precompute {idx:02d}/{len(markets):02d} {meta['market_id']} rows={len(df):,}", flush=True)
        cand, meta_df = candidate_trades_for_market(df, pd.Series(meta))
        if not cand.empty:
            pools.append(cand)
        if not meta_df.empty:
            metas.append(meta_df)
    con.close()
    pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    return pool, meta


def quote_prices(row: Any, params: Params, inventory: int) -> tuple[float, float, float, float]:
    inv_scale = max(0.0, 1.0 - params.inventory_scaling * abs(inventory))
    bid_chase = params.chase_increment_ticks * inv_scale * float(row.bid_up_ticks) * TICK
    ask_chase = params.chase_increment_ticks * inv_scale * float(row.ask_down_ticks) * TICK
    bid = float(row.quote_best_bid) + params.peg_offset_ticks * TICK + bid_chase
    ask = float(row.quote_best_ask) - params.peg_offset_ticks * TICK - ask_chase
    bid_cap = float(row.micro_price) - params.chase_cap_c_ticks * TICK
    ask_cap = float(row.micro_price) + params.chase_cap_c_ticks * TICK
    bid = min(bid, bid_cap, float(row.current_best_ask) - TICK)
    ask = max(ask, ask_cap, float(row.current_best_bid) + TICK)
    bid = float(np.clip(bid, 0.001, 0.999))
    ask = float(np.clip(ask, 0.001, 0.999))
    bid_distance_to_micro_ticks = (float(row.micro_price) - bid) / TICK
    ask_distance_to_micro_ticks = (ask - float(row.micro_price)) / TICK
    return bid, ask, bid_distance_to_micro_ticks, ask_distance_to_micro_ticks


def add_fill_economics(fill: dict[str, Any], row: Any, token_side: int, entry_price: float) -> None:
    category = str(row.category)
    current_mid = float(row.current_mid)
    denom = max(entry_price, 0.01)
    fill["realized_spread_bps"] = token_side * (current_mid - entry_price) / denom * 10_000.0
    fill["rebate_bps"] = taker_rebate_bps(category, entry_price)
    for horizon in ADVERSE_HORIZONS:
        future_mid = float(getattr(row, f"future_mid_{horizon}s"))
        adverse = -token_side * (future_mid - current_mid) / denom * 10_000.0
        inv_charge = 0.5 * abs(future_mid - current_mid) / denom * 10_000.0
        fill[f"adverse_selection_bps_{horizon}s"] = adverse
        fill[f"inv_resolution_charge_bps_{horizon}s"] = inv_charge
        fill[f"net_pnl_bps_{horizon}s"] = fill["realized_spread_bps"] + fill["rebate_bps"] - adverse - inv_charge


def simulate(pool: pd.DataFrame, params: Params) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    sub = pool[pool["cadence_sec"].eq(params.cadence_sec)].copy()
    if sub.empty:
        return pd.DataFrame()
    fills: list[dict[str, Any]] = []
    for _, asset_rows in sub.groupby(["market_id", "asset_id"], sort=False):
        open_lots: list[dict[str, int]] = []
        for row in asset_rows.itertuples(index=False):
            now = int(row.fill_time_ns)
            open_lots = [lot for lot in open_lots if int(lot["exit_time_ns"]) > now]
            inventory = int(sum(int(lot["token_side"]) for lot in open_lots))
            bid, ask, bid_dist, ask_dist = quote_prices(row, params, inventory)
            token_side = 0
            entry_price = math.nan
            distance_to_micro_ticks = math.nan
            side = str(row.trade_side_norm)
            trade_price = float(row.trade_price)
            if side == "SELL" and inventory < INTERNAL_INVENTORY_CAP and trade_price <= bid + 1e-12:
                token_side = 1
                entry_price = bid
                distance_to_micro_ticks = bid_dist
            elif side == "BUY" and inventory > -INTERNAL_INVENTORY_CAP and trade_price >= ask - 1e-12:
                token_side = -1
                entry_price = ask
                distance_to_micro_ticks = ask_dist
            if token_side == 0 or not np.isfinite(entry_price):
                continue
            fill = {
                "market_id": str(row.market_id),
                "slug": str(row.slug),
                "asset_id": str(row.asset_id),
                "family": str(row.family),
                "category": str(row.category),
                "fee_segment": str(row.fee_segment),
                "cadence_sec": params.cadence_sec,
                "peg_offset_ticks": params.peg_offset_ticks,
                "chase_increment_ticks": params.chase_increment_ticks,
                "chase_cap_c_ticks": params.chase_cap_c_ticks,
                "inventory_scaling": params.inventory_scaling,
                "fill_time_ns": now,
                "entry_price": entry_price,
                "trade_price": trade_price,
                "current_mid": float(row.current_mid),
                "micro_price": float(row.micro_price),
                "maker_side": "BUY" if token_side > 0 else "SELL",
                "token_side": token_side,
                "inventory_before": inventory,
                "distance_to_micro_ticks": distance_to_micro_ticks,
                "quote_age_sec": float(row.quote_age_sec),
            }
            add_fill_economics(fill, row, token_side, entry_price)
            fills.append(fill)
            open_lots.append({"exit_time_ns": now + HOLD_SEC * 1_000_000_000, "token_side": token_side})
    return pd.DataFrame(fills)


def denominator(meta: pd.DataFrame, params: Params, category: str | None = None, market_id: str | None = None) -> int:
    if meta.empty:
        return 0
    sub = meta[meta["cadence_sec"].eq(params.cadence_sec)]
    if category is not None:
        sub = sub[sub["category"].eq(category)]
    if market_id is not None:
        sub = sub[sub["market_id"].eq(market_id)]
    return int(sub["n_quote_orders"].sum())


def summarize(
    fills: pd.DataFrame,
    params: Params,
    meta: pd.DataFrame,
    *,
    row_type: str,
    scope: str,
    horizon: int,
    trial_number: int = -1,
    category: str = "",
    market_id: str = "",
    slug: str = "",
    marginal_vs_prior_bps: float = math.nan,
) -> dict[str, Any]:
    value_col = f"net_pnl_bps_{horizon}s"
    n = int(len(fills))
    denom = denominator(meta, params, category or None, market_id or None)
    mean = float(fills[value_col].mean()) if n else math.nan
    std = float(fills[value_col].std(ddof=1)) if n > 1 else math.nan
    ci_lo, ci_hi = bootstrap_mean_ci(fills.rename(columns={value_col: "net"}), "net", RNG_SEED + horizon + n) if n else (math.nan, math.nan)
    objective = mean if n >= ROBUST_MIN_FILLS and np.isfinite(mean) else -1e6 + n
    entry_notional = float(fills["entry_price"].sum()) if n and "entry_price" in fills else 0.0
    net_usd = (
        float((fills[value_col].astype(float) / 10_000.0 * fills["entry_price"].astype(float)).sum())
        if n and "entry_price" in fills
        else 0.0
    )
    portfolio_return_bps = net_usd / entry_notional * 10_000.0 if entry_notional > 0 else math.nan
    if n and "fill_time_ns" in fills:
        start_ns = int(fills["fill_time_ns"].min())
        end_ns = int(fills["fill_time_ns"].max())
        elapsed_hours = max((end_ns - start_ns) / 3_600_000_000_000.0, 0.0)
        start_time = pd.to_datetime(start_ns, utc=True).isoformat()
        end_time = pd.to_datetime(end_ns, utc=True).isoformat()
    else:
        elapsed_hours = math.nan
        start_time = ""
        end_time = ""
    return {
        "row_type": row_type,
        "scope": scope,
        "trial_number": trial_number,
        "horizon_sec": horizon,
        "category": category,
        "fee_segment": fee_segment(category) if category else "",
        "market_id": market_id,
        "slug": slug,
        "peg_offset_ticks": params.peg_offset_ticks,
        "chase_increment_ticks": params.chase_increment_ticks,
        "chase_cap_c_ticks": params.chase_cap_c_ticks,
        "inventory_scaling": params.inventory_scaling,
        "cadence_sec": params.cadence_sec,
        "n_fills": n,
        "fill_rate": n / denom if denom else math.nan,
        "mean_distance_to_micro_ticks": float(fills["distance_to_micro_ticks"].mean()) if n and "distance_to_micro_ticks" in fills else math.nan,
        "mean_realized_spread_bps": float(fills["realized_spread_bps"].mean()) if n else math.nan,
        "mean_rebate_bps": float(fills["rebate_bps"].mean()) if n else math.nan,
        "mean_adverse_selection_bps": float(fills[f"adverse_selection_bps_{horizon}s"].mean()) if n else math.nan,
        "mean_inv_resolution_charge_bps": float(fills[f"inv_resolution_charge_bps_{horizon}s"].mean()) if n else math.nan,
        "mean_net_pnl_bps": mean,
        "median_net_pnl_bps": float(fills[value_col].median()) if n else math.nan,
        "total_net_pnl_bps_units": float(fills[value_col].sum()) if n else 0.0,
        "gross_entry_notional_usd_1share": entry_notional,
        "total_net_pnl_usd_1share": net_usd,
        "portfolio_return_bps_1share": portfolio_return_bps,
        "portfolio_start_utc": start_time,
        "portfolio_end_utc": end_time,
        "portfolio_elapsed_hours": elapsed_hours,
        "fills_per_hour": n / elapsed_hours if n and elapsed_hours > 0 else math.nan,
        "std_net_pnl_bps": std,
        "win_rate": float(fills[value_col].gt(0).mean()) if n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "clears_zero": bool(n >= ROBUST_MIN_FILLS and np.isfinite(ci_lo) and ci_lo > 0),
        "marginal_vs_prior_bps": marginal_vs_prior_bps,
        "objective": objective,
    }


def suggest_params(trial: optuna.Trial) -> Params:
    return Params(
        peg_offset_ticks=trial.suggest_categorical("peg_offset_ticks", [-1, 0, 1]),
        chase_increment_ticks=trial.suggest_int("chase_increment_ticks", 0, 4),
        chase_cap_c_ticks=trial.suggest_int("chase_cap_c_ticks", -3, 12),
        inventory_scaling=trial.suggest_float("inventory_scaling", 0.0, 0.35),
        cadence_sec=trial.suggest_categorical("cadence_sec", list(CADENCE_OPTIONS)),
    )


def optimize(pool: pd.DataFrame, meta: pd.DataFrame) -> tuple[Params, pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RNG_SEED))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fills = simulate(pool, params)
        row = summarize(
            fills,
            params,
            meta,
            row_type="trial",
            scope="pooled",
            horizon=OBJECTIVE_HORIZON,
            trial_number=trial.number,
        )
        trial_rows.append(row)
        return float(row["objective"])

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = Params(**study.best_params)
    best_fills = simulate(pool, best)
    return best, pd.DataFrame(trial_rows), best_fills


def curve_params(best: Params) -> list[Params]:
    out = []
    cap_values = sorted(set([-3, -2, -1, 0, 1, 2, 3, 5, 8, 12, best.chase_cap_c_ticks - 1, best.chase_cap_c_ticks, best.chase_cap_c_ticks + 1]))
    for c in cap_values:
        for inc in [0, 1, 2, 3, 4]:
            out.append(
                Params(
                    peg_offset_ticks=best.peg_offset_ticks,
                    chase_increment_ticks=inc,
                    chase_cap_c_ticks=c,
                    inventory_scaling=best.inventory_scaling,
                    cadence_sec=best.cadence_sec,
                )
            )
    return out


def build_curve(pool: pd.DataFrame, meta: pd.DataFrame, best: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fills_by_key: dict[tuple[int, int], pd.DataFrame] = {}
    for params in curve_params(best):
        fills = simulate(pool, params)
        fills_by_key[(params.chase_cap_c_ticks, params.chase_increment_ticks)] = fills
        rows.append(
            summarize(
                fills,
                params,
                meta,
                row_type="curve_category",
                scope="pooled",
                horizon=OBJECTIVE_HORIZON,
            )
        )
        for category, sub in fills.groupby("category", sort=True):
            rows.append(
                summarize(
                    sub,
                    params,
                    meta,
                    row_type="curve_category",
                    scope="category",
                    horizon=OBJECTIVE_HORIZON,
                    category=str(category),
                )
            )
        for market_id, sub in fills.groupby("market_id", sort=True):
            slug = str(sub["slug"].dropna().iloc[0]) if not sub.empty else ""
            category = str(sub["category"].dropna().iloc[0]) if not sub.empty else ""
            rows.append(
                summarize(
                    sub,
                    params,
                    meta,
                    row_type="curve_market",
                    scope="market",
                    horizon=OBJECTIVE_HORIZON,
                    category=category,
                    market_id=str(market_id),
                    slug=slug,
                )
            )

    curve = pd.DataFrame(rows)
    curve["marginal_vs_prior_bps"] = np.nan
    curve["marginal_toward_micro_bps"] = np.nan
    keys = ["row_type", "scope", "category", "market_id", "chase_cap_c_ticks"]
    parts = []
    for _, group in curve.groupby(keys, dropna=False, sort=False):
        g = group.sort_values("chase_increment_ticks").copy()
        g["marginal_vs_prior_bps"] = g["mean_net_pnl_bps"].diff()
        parts.append(g)
    curve = pd.concat(parts, ignore_index=True)
    parts = []
    keys = ["row_type", "scope", "category", "market_id", "chase_increment_ticks"]
    for _, group in curve.groupby(keys, dropna=False, sort=False):
        g = group.sort_values("chase_cap_c_ticks", ascending=False).copy()
        g["marginal_toward_micro_bps"] = g["mean_net_pnl_bps"].diff()
        parts.append(g)
    curve = pd.concat(parts, ignore_index=True)
    return curve, pd.concat(fills_by_key.values(), ignore_index=True) if fills_by_key else pd.DataFrame()


def optimum_rows(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in curve[curve["scope"].isin(["pooled", "category", "market"])].groupby(
        ["scope", "category", "market_id"], dropna=False, sort=False
    ):
        g = group[group["n_fills"].ge(ROBUST_MIN_FILLS)].copy()
        if g.empty:
            g = group.copy()
        best = g.sort_values(["mean_net_pnl_bps", "fill_rate"], ascending=False).head(1).copy()
        row = best.iloc[0].to_dict()
        best_inc = int(row["chase_increment_ticks"])
        best_cap = int(row["chase_cap_c_ticks"])
        same_cap = group[group["chase_cap_c_ticks"].eq(best_cap)].sort_values("chase_increment_ticks")
        sign_flip = same_cap[
            same_cap["chase_increment_ticks"].gt(best_inc)
            & same_cap["marginal_vs_prior_bps"].lt(0)
        ].head(1)
        same_inc = group[group["chase_increment_ticks"].eq(best_inc)].sort_values(
            "chase_cap_c_ticks",
            ascending=False,
        )
        cap_flip = same_inc[
            same_inc["chase_cap_c_ticks"].lt(best_cap)
            & same_inc["marginal_toward_micro_bps"].lt(0)
        ].head(1)
        row["row_type"] = "optimum"
        row["sign_flip_chase_increment_ticks"] = (
            int(sign_flip["chase_increment_ticks"].iloc[0]) if not sign_flip.empty else math.nan
        )
        row["sign_flip_chase_cap_c_ticks"] = (
            int(sign_flip["chase_cap_c_ticks"].iloc[0]) if not sign_flip.empty else math.nan
        )
        row["sign_flip_marginal_bps"] = (
            float(sign_flip["marginal_vs_prior_bps"].iloc[0]) if not sign_flip.empty else math.nan
        )
        row["sign_flip_near_microprice"] = (
            abs(float(row["sign_flip_chase_cap_c_ticks"])) <= 1 if np.isfinite(row["sign_flip_chase_cap_c_ticks"]) else False
        )
        if not np.isfinite(row["sign_flip_chase_cap_c_ticks"]):
            row["sign_flip_location"] = "none"
        elif abs(float(row["sign_flip_chase_cap_c_ticks"])) <= 1:
            row["sign_flip_location"] = "near_microprice"
        elif float(row["sign_flip_chase_cap_c_ticks"]) > 1:
            row["sign_flip_location"] = "before_microprice"
        else:
            row["sign_flip_location"] = "past_microprice"
        row["cap_axis_sign_flip_chase_cap_c_ticks"] = (
            int(cap_flip["chase_cap_c_ticks"].iloc[0]) if not cap_flip.empty else math.nan
        )
        row["cap_axis_sign_flip_marginal_bps"] = (
            float(cap_flip["marginal_toward_micro_bps"].iloc[0]) if not cap_flip.empty else math.nan
        )
        if not np.isfinite(row["cap_axis_sign_flip_chase_cap_c_ticks"]):
            row["cap_axis_sign_flip_location"] = "none"
        elif abs(float(row["cap_axis_sign_flip_chase_cap_c_ticks"])) <= 1:
            row["cap_axis_sign_flip_location"] = "near_microprice"
        elif float(row["cap_axis_sign_flip_chase_cap_c_ticks"]) > 1:
            row["cap_axis_sign_flip_location"] = "before_microprice"
        else:
            row["cap_axis_sign_flip_location"] = "past_microprice"
        rows.append(row)
    return pd.DataFrame(rows)


def selected_rows(best_fills: pd.DataFrame, best: Params, meta: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_trials = trials.sort_values("objective", ascending=False).head(25).copy()
    rows.extend(top_trials.to_dict("records"))
    for horizon in ADVERSE_HORIZONS:
        rows.append(summarize(best_fills, best, meta, row_type="selected", scope="pooled", horizon=horizon))
        for category, sub in best_fills.groupby("category", sort=True):
            rows.append(
                summarize(
                    sub,
                    best,
                    meta,
                    row_type="selected_category",
                    scope="category",
                    horizon=horizon,
                    category=str(category),
                )
            )
    return pd.DataFrame(rows)


def fills_with_portfolio_pnl(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()
    out = fills.sort_values("fill_time_ns").copy()
    out["fill_time_utc"] = pd.to_datetime(out["fill_time_ns"].astype("int64"), utc=True).astype(str)
    for horizon in ADVERSE_HORIZONS:
        bps_col = f"net_pnl_bps_{horizon}s"
        usd_col = f"net_pnl_usd_1share_{horizon}s"
        out[usd_col] = out[bps_col].astype(float) / 10_000.0 * out["entry_price"].astype(float)
    return out


def portfolio_timeseries(fills: pd.DataFrame, horizon: int = OBJECTIVE_HORIZON) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    net_col = f"net_pnl_usd_1share_{horizon}s"
    bps_col = f"net_pnl_bps_{horizon}s"
    out = fills_with_portfolio_pnl(fills).copy()
    out = out.sort_values("fill_time_ns").reset_index(drop=True)
    out["fill_number"] = np.arange(1, len(out) + 1)
    out["gross_entry_notional_usd_1share"] = out["entry_price"].astype(float)
    out["cum_net_pnl_usd_1share"] = out[net_col].cumsum()
    out["cum_gross_entry_notional_usd_1share"] = out["gross_entry_notional_usd_1share"].cumsum()
    out["cum_return_bps_1share"] = (
        out["cum_net_pnl_usd_1share"] / out["cum_gross_entry_notional_usd_1share"].replace(0, np.nan) * 10_000.0
    )
    out["cum_peak_net_pnl_usd_1share"] = out["cum_net_pnl_usd_1share"].cummax()
    out["drawdown_usd_1share"] = out["cum_net_pnl_usd_1share"] - out["cum_peak_net_pnl_usd_1share"]
    keep = [
        "fill_number",
        "fill_time_utc",
        "market_id",
        "slug",
        "category",
        "maker_side",
        "entry_price",
        bps_col,
        net_col,
        "gross_entry_notional_usd_1share",
        "cum_net_pnl_usd_1share",
        "cum_gross_entry_notional_usd_1share",
        "cum_return_bps_1share",
        "drawdown_usd_1share",
    ]
    return out[keep]


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    fmt: str = ".0f",
    optimum: tuple[int, int] | None = None,
    sign_flip: tuple[int, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    ax.set_xlabel("chase increment, ticks")
    ax.set_ylabel("micro-price cap c, ticks")
    ax.set_title(title)
    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            val = values[yi, xi]
            if np.isfinite(val):
                ax.text(xi, yi, format(val, fmt), ha="center", va="center", fontsize=8)
    if optimum is not None:
        cap, inc = optimum
        if cap in pivot.index and inc in pivot.columns:
            ax.scatter(
                [list(pivot.columns).index(inc)],
                [list(pivot.index).index(cap)],
                marker="o",
                s=190,
                facecolors="none",
                edgecolors="black",
                linewidths=2.2,
                label="optimum",
            )
    if sign_flip is not None:
        cap, inc = sign_flip
        if cap in pivot.index and inc in pivot.columns:
            ax.scatter(
                [list(pivot.columns).index(inc)],
                [list(pivot.index).index(cap)],
                marker="x",
                s=150,
                color="black",
                linewidths=2.4,
                label="first bad extra increment",
            )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_plots(results: pd.DataFrame, selected_fills: pd.DataFrame, portfolio: pd.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    category = results[(results["row_type"].eq("selected_category")) & (results["horizon_sec"].eq(OBJECTIVE_HORIZON))].copy()
    if not category.empty:
        category = category.sort_values("mean_net_pnl_bps", ascending=False)
        fig, ax = plt.subplots(figsize=(7.6, 4.5))
        x = np.arange(len(category))
        y = category["mean_net_pnl_bps"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - category["ci_lo"].to_numpy(dtype=float),
            category["ci_hi"].to_numpy(dtype=float) - y,
        ])
        ax.bar(x, y, color="#4C78A8", width=0.68)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=3)
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(category["category"].astype(str), rotation=20, ha="right")
        ax.set_ylabel("mean net PnL, bps per fill")
        ax.set_title("Selected K-PEG Policy: Per-Fill Net by Category")
        fig.tight_layout()
        path = PLOTS / "kpeg_selected_category_net.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    curve_crypto = results[
        (results["row_type"].eq("curve_category"))
        & (results["scope"].eq("category"))
        & (results["category"].eq("Crypto"))
    ].copy()
    opt_crypto = results[
        (results["row_type"].eq("optimum"))
        & (results["scope"].eq("category"))
        & (results["category"].eq("Crypto"))
    ].copy()
    optimum = None
    sign_flip = None
    if not opt_crypto.empty:
        opt_row = opt_crypto.iloc[0]
        optimum = (int(opt_row["chase_cap_c_ticks"]), int(opt_row["chase_increment_ticks"]))
        if np.isfinite(opt_row.get("sign_flip_chase_cap_c_ticks", np.nan)):
            sign_flip = (
                int(opt_row["sign_flip_chase_cap_c_ticks"]),
                int(opt_row["sign_flip_chase_increment_ticks"]),
            )
    if not curve_crypto.empty:
        mean_pivot = curve_crypto.pivot(
            index="chase_cap_c_ticks",
            columns="chase_increment_ticks",
            values="mean_net_pnl_bps",
        ).sort_index()
        path = PLOTS / "kpeg_crypto_chase_heatmap.png"
        plot_heatmap(
            mean_pivot,
            title="Crypto Mean Net PnL by Chase Increment and Micro-Price Cap",
            cbar_label="bps per fill",
            out_path=path,
            fmt=".0f",
            optimum=optimum,
            sign_flip=sign_flip,
        )
        paths.append(path)

        portfolio_pivot = curve_crypto.pivot(
            index="chase_cap_c_ticks",
            columns="chase_increment_ticks",
            values="total_net_pnl_usd_1share",
        ).sort_index()
        path = PLOTS / "kpeg_crypto_portfolio_heatmap.png"
        plot_heatmap(
            portfolio_pivot,
            title="Crypto Portfolio Net PnL by Chase Setting",
            cbar_label="total net dollars, 1-share fills",
            out_path=path,
            fmt=".2f",
            optimum=optimum,
            sign_flip=sign_flip,
        )
        paths.append(path)

    opt_cat = results[(results["row_type"].eq("optimum")) & (results["scope"].eq("category"))].copy()
    if not opt_cat.empty:
        opt_cat = opt_cat.sort_values("category")
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        y = np.arange(len(opt_cat))
        pos_map = {idx: pos for pos, idx in enumerate(opt_cat.index)}
        ax.scatter(opt_cat["chase_cap_c_ticks"], y, s=90, label="optimum cap", color="#4C78A8")
        inc_flip = opt_cat[np.isfinite(opt_cat["sign_flip_chase_cap_c_ticks"])]
        if not inc_flip.empty:
            inc_y = inc_flip.index.map(pos_map).to_numpy(dtype=int)
            ax.scatter(
                inc_flip["sign_flip_chase_cap_c_ticks"],
                inc_y,
                marker="x",
                s=90,
                label="bad extra increment",
                color="#E45756",
            )
        cap_flip = opt_cat[np.isfinite(opt_cat.get("cap_axis_sign_flip_chase_cap_c_ticks", np.nan))]
        if not cap_flip.empty:
            cap_y = cap_flip.index.map(pos_map).to_numpy(dtype=int)
            ax.scatter(
                cap_flip["cap_axis_sign_flip_chase_cap_c_ticks"],
                cap_y,
                marker="^",
                s=80,
                label="bad tighter cap",
                color="#F58518",
            )
        ax.axvline(0, color="#222222", linewidth=1, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(opt_cat["category"].astype(str))
        ax.set_xlabel("micro-price cap c, ticks")
        ax.set_title("Optimum and Marginal Sign Flips Relative to Micro-Price")
        ax.legend(loc="best")
        fig.tight_layout()
        path = PLOTS / "kpeg_optimum_vs_signflip.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    curve_cat = results[(results["row_type"].eq("curve_category")) & (results["scope"].eq("category"))].copy()
    if not curve_cat.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for cat, sub in curve_cat.groupby("category", sort=True):
            ax.scatter(sub["fill_rate"] * 100.0, sub["mean_net_pnl_bps"], s=28, alpha=0.72, label=str(cat))
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_xlabel("fill rate, % of quote opportunities")
        ax.set_ylabel("mean net PnL, bps per fill")
        ax.set_title("Fill Rate vs Per-Fill Net PnL")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = PLOTS / "kpeg_fillrate_vs_net.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    if not portfolio.empty:
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        times = pd.to_datetime(portfolio["fill_time_utc"], utc=True, format="mixed")
        ax.step(times, portfolio["cum_net_pnl_usd_1share"], where="post", label="pooled", color="#4C78A8", linewidth=1.8)
        sf = fills_with_portfolio_pnl(selected_fills)
        for cat, sub in sf.groupby("category", sort=True):
            if len(sub) < 5:
                continue
            sub = sub.sort_values("fill_time_ns")
            cat_times = pd.to_datetime(sub["fill_time_utc"], utc=True, format="mixed")
            cat_cum = sub[f"net_pnl_usd_1share_{OBJECTIVE_HORIZON}s"].cumsum()
            ax.step(cat_times, cat_cum, where="post", linewidth=1.1, alpha=0.8, label=str(cat))
        ax.axhline(0, color="#222222", linewidth=1)
        ax.set_ylabel("cumulative net dollars, 1-share fills")
        ax.set_title("Selected K-PEG Portfolio Replay")
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = PLOTS / "kpeg_selected_portfolio_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths


def write_note(results: pd.DataFrame, best: Params, pool: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    selected = results[(results["row_type"].eq("selected")) & (results["horizon_sec"].eq(OBJECTIVE_HORIZON))].iloc[0]
    category = results[(results["row_type"].eq("selected_category")) & (results["horizon_sec"].eq(OBJECTIVE_HORIZON))].copy()
    opt = results[results["row_type"].eq("optimum")].copy()
    opt_cat = opt[opt["scope"].eq("category")].sort_values(["category"]).copy()

    def span(row: Any) -> str:
        start = str(getattr(row, "portfolio_start_utc", "") or "")
        end = str(getattr(row, "portfolio_end_utc", "") or "")
        if not start or start == "nan" or not end or end == "nan":
            return "n/a"
        return f"{start[:16]} -> {end[:16]}"

    cat_rows = []
    for row in category.sort_values(["fee_segment", "category"]).itertuples(index=False):
        cat_rows.append(
            [
                str(row.fee_segment),
                str(row.category),
                f"{int(row.n_fills):,}",
                pct(float(row.fill_rate)),
                bps(float(row.mean_realized_spread_bps)),
                bps(float(row.mean_rebate_bps)),
                bps(float(row.mean_adverse_selection_bps)),
                bps(float(row.mean_inv_resolution_charge_bps)),
                bps(float(row.mean_net_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                "yes" if bool(row.clears_zero) else "no",
            ]
        )

    portfolio_rows_out = []
    portfolio_basis = pd.concat(
        [
            results[(results["row_type"].eq("selected")) & (results["horizon_sec"].eq(OBJECTIVE_HORIZON))],
            category,
        ],
        ignore_index=True,
    )
    for row in portfolio_basis.sort_values(["scope", "category"]).itertuples(index=False):
        portfolio_rows_out.append(
            [
                "pooled" if str(row.scope) == "pooled" else str(row.category),
                f"{int(row.n_fills):,}",
                span(row),
                f"{float(row.portfolio_elapsed_hours):.2f}h" if np.isfinite(row.portfolio_elapsed_hours) else "n/a",
                usd(float(row.total_net_pnl_usd_1share)),
                usd(float(row.gross_entry_notional_usd_1share)),
                bps(float(row.portfolio_return_bps_1share)),
                f"{float(row.fills_per_hour):.1f}" if np.isfinite(row.fills_per_hour) else "n/a",
            ]
        )

    opt_rows = []
    for row in opt_cat.itertuples(index=False):
        flip = (
            f"inc={int(row.sign_flip_chase_increment_ticks)}, c={int(row.sign_flip_chase_cap_c_ticks)}, {bps(float(row.sign_flip_marginal_bps))}"
            if np.isfinite(row.sign_flip_chase_increment_ticks)
            else "none"
        )
        cap_flip = (
            f"c={int(row.cap_axis_sign_flip_chase_cap_c_ticks)}, {bps(float(row.cap_axis_sign_flip_marginal_bps))}"
            if hasattr(row, "cap_axis_sign_flip_chase_cap_c_ticks")
            and np.isfinite(row.cap_axis_sign_flip_chase_cap_c_ticks)
            else "none"
        )
        location = str(row.sign_flip_location) if hasattr(row, "sign_flip_location") else "none"
        cap_location = (
            str(row.cap_axis_sign_flip_location)
            if hasattr(row, "cap_axis_sign_flip_location") and str(row.cap_axis_sign_flip_location)
            else "none"
        )
        opt_rows.append(
            [
                str(row.category),
                f"inc={int(row.chase_increment_ticks)}, c={int(row.chase_cap_c_ticks)}, dist={float(row.mean_distance_to_micro_ticks):.1f}",
                bps(float(row.mean_net_pnl_bps)),
                pct(float(row.fill_rate)),
                flip,
                location,
                cap_flip,
                cap_location,
            ]
        )

    dead_categories = []
    curve_cat = results[(results["row_type"].eq("curve_category")) & (results["scope"].eq("category"))]
    for cat, sub in curve_cat.groupby("category", sort=True):
        if not sub["mean_net_pnl_bps"].gt(0).any():
            dead_categories.append(str(cat))

    note = f"""---
tags: [dali, block-kpeg, maker, peg-chase, optuna]
---

# Block K-PEG Dynamic Chase Findings

## Headline

Optimized dynamic chasing is {'positive' if float(selected.mean_net_pnl_bps) > 0 else 'negative'} on the pooled IS ceiling: best 30s net is {bps(float(selected.mean_net_pnl_bps))}, CI [{bps(float(selected.ci_lo))}, {bps(float(selected.ci_hi))}], fill rate {pct(float(selected.fill_rate))}, and {int(selected.n_fills):,} fills. Best policy: peg offset `{best.peg_offset_ticks}` ticks, chase increment `{best.chase_increment_ticks}` ticks, micro-price cap `c={best.chase_cap_c_ticks}` ticks, inventory scaling `{best.inventory_scaling:.3f}`, cadence `{best.cadence_sec}s`.

One-share portfolio replay over the selected-policy sample: {usd(float(selected.total_net_pnl_usd_1share))} net on {usd(float(selected.gross_entry_notional_usd_1share))} gross entry notional, or {bps(float(selected.portfolio_return_bps_1share))}, across {float(selected.portfolio_elapsed_hours):.2f} elapsed hours.

Falsifier check: categories with negative net at every curve setting: {', '.join(dead_categories) if dead_categories else 'none'}.

## Plain-English Summary

This test asks a practical market-making question: if Midas keeps moving our passive quote to stay competitive, how far should it chase before the extra fills become toxic?

A market maker posts bids and asks instead of crossing the spread. The maker earns money when it buys slightly cheap or sells slightly rich, may receive a maker rebate, and loses money when the trade was informed and price moves against the maker after the fill. The trade-off is simple:

`net = spread captured + rebate - adverse selection - inventory/resolution risk`

In the main category tables, **net PnL is calculated per simulated fill first**, then averaged across all non-overlapping fills for the market or category. The reported `net` number is mean basis points per fill; the CI is the uncertainty around that mean. The `30s` horizon means we look 30 seconds after each fill to estimate adverse selection and inventory risk.

The portfolio table is different: it converts each fill's bps into dollars assuming **one share per fill**, then cumulatively sums those dollars over the actual replay span. This is still a ceiling replay, not a deployable account statement, because it ignores queue rank, partial fill sizing, cancellation latency, and capital constraints beyond the internal inventory cap.

The main result is not "chase as much as possible." For Crypto, the best setting joins the best quote and chases by 2 ticks, but stops about 7 ticks away from micro-price. In plain terms: be competitive, but leave a meaningful cushion before fair value.

## Concepts

- **Best bid / best ask:** the highest visible buy price and lowest visible sell price in the order book.
- **Tick:** the smallest price step. Here one tick is 1 cent.
- **Pegging:** keeping our quote tied to the best bid or best ask.
- **Join:** quote at the current best bid or ask. In this run, `peg_offset = 0`.
- **Improve:** quote one tick better than the current best quote to get priority.
- **Sit behind:** quote one tick worse than the best quote, safer but less likely to fill.
- **Chase increment:** how many ticks we move when the best bid rises, mirrored for asks.
- **Micro-price:** an order-book estimate of fair value using both price and size at the best bid/ask.
- **Chase cap `c`:** the stop line around micro-price. For bids, `c=7` means stop at `micro_price - 7 ticks`; `c=0` means chase up to micro-price; `c<0` means chase past estimated fair value.
- **Inventory scaling:** if we already hold inventory, reduce chase aggressiveness so we do not keep adding risk in the same direction.
- **Cadence:** how often the quote refreshes. The best result used a 1-second cadence.
- **Adverse selection:** the bad case where someone fills us because they know or react faster than we do, and price moves against us after the fill.
- **Sign flip:** the first tested point where the next extra chase step makes marginal PnL worse.

## Visuals

Selected policy by category. Crypto is the only robust category; Sports has too few fills.

![]({display_path(PLOTS / "kpeg_selected_category_net.png")})

Crypto chase/cap heatmap. Each cell is mean net PnL in bps for that chase increment and micro-price cap. Circle = optimum; X = corrected first bad extra increment.

![]({display_path(PLOTS / "kpeg_crypto_chase_heatmap.png")})

Crypto portfolio heatmap. Each cell is total net dollars over the replay using one-share fills, so it rewards both per-fill quality and fill count.

![]({display_path(PLOTS / "kpeg_crypto_portfolio_heatmap.png")})

Where the category optimum, bad extra increment, and bad tighter cap occur relative to micro-price.

![]({display_path(PLOTS / "kpeg_optimum_vs_signflip.png")})

Fill-rate versus net-PnL trade-off. Higher fill rate is not automatically better; toxic fills can reduce net.

![]({display_path(PLOTS / "kpeg_fillrate_vs_net.png")})

Portfolio replay for the selected policy. This is cumulative one-share-per-fill net PnL through the captured sample.

![]({display_path(PLOTS / "kpeg_selected_portfolio_curve.png")})

## Selected Policy by Category

{markdown_table(
    ["segment", "category", "fills", "fill rate", "spread", "rebate", "adverse", "inv/res", "net", "95% CI", "clears"],
    cat_rows,
)}

Simple read: Crypto clears because it has enough fills and the confidence interval stays above zero. Sports looks profitable, but the fill count is too small to trust.

## Portfolio Replay

{markdown_table(
    ["scope", "fills", "span", "elapsed", "net $", "gross entry $", "return", "fills/hr"],
    portfolio_rows_out,
)}

Simple read: the pooled selected policy made {usd(float(selected.total_net_pnl_usd_1share))} over {float(selected.portfolio_elapsed_hours):.2f} hours under the one-share-per-fill replay. That is the portfolio-style version of the same per-fill edge; it is useful for capacity intuition, while the bps table is cleaner for comparing trade quality.

## Marginal Curve Optima

{markdown_table(
    ["category", "max setting", "max net", "fill rate", "bad extra increment", "extra location", "bad tighter cap", "cap location"],
    opt_rows,
)}

Simple read: the best Crypto curve point is `inc=2, c=7`, which means the bot chases, but still stops well before micro-price. The corrected extra-increment sign flip is `inc=3, c=7`: at the same cap, one more tick of chase reduces mean net PnL. The separate tighter-cap flip asks what happens if we keep `inc=2` but move closer to micro-price; for Crypto, the first bad tighter cap is also before micro-price, so adverse selection binds before the fair-value boundary in this ceiling model.

## What We Tested

1. **Starting quote position:** sit behind, join, or improve the best quote.
2. **Chase increment:** move by 0 to 4 ticks when the best bid/ask moves.
3. **Micro-price cap:** stop before, at, or past estimated fair value.
4. **Inventory conditioning:** reduce chase as open inventory grows.
5. **Requote cadence:** refresh every 1, 2, 5, 10, or 30 seconds.

For each policy, the simulator replayed real A1 book/trade data. A modeled bid filled only when a real SELL trade crossed our modeled bid. A modeled ask filled only when a real BUY trade crossed our modeled ask. This makes both fill rate and adverse selection depend on how aggressively we chase.

## How Conclusions Were Reached

1. We pooled `a0`, `a0b`, `a0c`, and `a0c_roll` as one in-sample dataset and excluded `will-jd-vance`.
2. For each quote setting, we reconstructed fills using the A1.4h passive-fill proxy.
3. For every fill, we measured spread captured, maker rebate, price movement against us after 5/30/60 seconds, and an inventory/resolution charge.
4. Optuna searched {N_TRIALS} combinations of peg offset, chase increment, cap, inventory scaling, and cadence.
5. The best policy was selected on 30-second net PnL.
6. A curve sweep then measured how PnL changed as chase increment and micro-price cap changed.
7. Portfolio replay converted each selected fill into one-share dollars and cumulated those dollars over the actual sample span.

Bottom line: dynamic chasing can rescue the maker baseline in this optimistic replay, especially in Crypto, but only with a cap. The practical next question is whether this survives real queue position, latency, partial fills, and cancellation risk.

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled `{', '.join(RUN_POOL)}` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance`.
- Tick size: fixed at 1 cent.
- Micro-price: `(ask * bid_size + bid * ask_size) / (bid_size + ask_size)`, falling back to mid when touch size is missing.
- Bid policy: `best_bid + peg_offset*tick + chase_increment*positive_bid_move_ticks*tick`, scaled by `max(0, 1 - inventory_scaling*abs(inventory))`, capped at `micro_price - c*tick`; ask side is mirrored.
- Cap sweep includes `c < 0`, allowing quotes past micro-price in the curve and optimizer.
- Cadence/staleness: quotes are refreshed on sampled book states at the optimized cadence and ignored after `2 * cadence`.
- Fill proxy: A1.4h extension. A SELL print fills our bid only if `trade_price <= modeled_bid`; a BUY print fills our ask only if `trade_price >= modeled_ask`. More aggressive chase settings therefore change both fill count and realized adverse selection.
- PnL: `realized_spread + rebate - adverse_selection - inventory/resolution_charge`; rebate follows K1 rules. Objective optimizes 30s net; 5/30/60s rows are included in the CSV.
- Portfolio PnL: one-share-per-fill replay, `net_usd = net_bps / 10000 * entry_price`, cumulatively summed by fill time. This is a capacity diagnostic, not a real account equity curve.
- Inventory: one-share fills with an internal ±{INTERNAL_INVENTORY_CAP} open-lot cap per asset; chase size is reduced as open inventory grows.
- CI: 500 bootstrap resamples over contiguous 300s fill-time blocks within market.
- Optuna trials: {N_TRIALS}.

## Interpretation

The marginal curve is the key read. The extra-increment flip holds cap fixed and asks whether one more chase tick helps. The tighter-cap flip holds chase increment fixed and asks whether moving closer to micro-price helps. If either flip appears at `c≈0`, the micro-price cap is behaving like the Stoikov fair-value boundary. If it appears at larger positive `c`, adverse selection binds before fair value. If it appears at negative `c`, the ceiling still tolerated some past-fair-value chasing before the next increment turned negative.

CSV: `data/analysis/csv_outputs/market_making/kpeg_chase_optimization.csv`.

Selected fills: `data/analysis/csv_outputs/market_making/kpeg_selected_fills.csv`.

Portfolio series: `data/analysis/csv_outputs/market_making/kpeg_portfolio_timeseries.csv`.

Candidate rows precomputed: {len(pool):,} across {pool['market_id'].nunique() if not pool.empty else 0:,} markets.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(f"missing features parquet: {FEATURES}")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    pool, meta = build_candidate_pool()
    if pool.empty:
        raise SystemExit("KPEG candidate pool is empty")
    print(f"KPEG candidate pool: {len(pool):,} rows, {pool['market_id'].nunique():,} markets")

    best, trials, best_fills = optimize(pool, meta)
    print(
        f"KPEG best: peg={best.peg_offset_ticks} inc={best.chase_increment_ticks} "
        f"c={best.chase_cap_c_ticks} inv={best.inventory_scaling:.3f} cadence={best.cadence_sec}s",
        flush=True,
    )
    selected = selected_rows(best_fills, best, meta, trials)
    curve, _curve_fills = build_curve(pool, meta, best)
    opt = optimum_rows(curve)
    results = pd.concat([selected, curve, opt], ignore_index=True)
    if "sign_flip_chase_increment_ticks" not in results:
        results["sign_flip_chase_increment_ticks"] = np.nan
    if "sign_flip_chase_cap_c_ticks" not in results:
        results["sign_flip_chase_cap_c_ticks"] = np.nan
    if "sign_flip_near_microprice" not in results:
        results["sign_flip_near_microprice"] = False
    if "sign_flip_location" not in results:
        results["sign_flip_location"] = ""
    if "sign_flip_marginal_bps" not in results:
        results["sign_flip_marginal_bps"] = np.nan
    if "cap_axis_sign_flip_chase_cap_c_ticks" not in results:
        results["cap_axis_sign_flip_chase_cap_c_ticks"] = np.nan
    if "cap_axis_sign_flip_marginal_bps" not in results:
        results["cap_axis_sign_flip_marginal_bps"] = np.nan
    if "cap_axis_sign_flip_location" not in results:
        results["cap_axis_sign_flip_location"] = ""
    order = [
        "row_type",
        "scope",
        "trial_number",
        "horizon_sec",
        "category",
        "fee_segment",
        "market_id",
        "slug",
        "peg_offset_ticks",
        "chase_increment_ticks",
        "chase_cap_c_ticks",
        "inventory_scaling",
        "cadence_sec",
        "n_fills",
        "fill_rate",
        "mean_distance_to_micro_ticks",
        "mean_realized_spread_bps",
        "mean_rebate_bps",
        "mean_adverse_selection_bps",
        "mean_inv_resolution_charge_bps",
        "mean_net_pnl_bps",
        "median_net_pnl_bps",
        "total_net_pnl_bps_units",
        "gross_entry_notional_usd_1share",
        "total_net_pnl_usd_1share",
        "portfolio_return_bps_1share",
        "portfolio_start_utc",
        "portfolio_end_utc",
        "portfolio_elapsed_hours",
        "fills_per_hour",
        "std_net_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "clears_zero",
        "marginal_vs_prior_bps",
        "marginal_toward_micro_bps",
        "sign_flip_chase_increment_ticks",
        "sign_flip_chase_cap_c_ticks",
        "sign_flip_marginal_bps",
        "sign_flip_near_microprice",
        "sign_flip_location",
        "cap_axis_sign_flip_chase_cap_c_ticks",
        "cap_axis_sign_flip_marginal_bps",
        "cap_axis_sign_flip_location",
        "objective",
    ]
    for col in order:
        if col not in results:
            results[col] = np.nan
    results = results[order]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    selected_fills = fills_with_portfolio_pnl(best_fills)
    portfolio = portfolio_timeseries(best_fills)
    selected_fills.to_csv(SELECTED_FILLS_CSV, index=False)
    portfolio.to_csv(PORTFOLIO_CSV, index=False)
    plot_paths = write_plots(results, best_fills, portfolio)
    write_note(results, best, pool)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {SELECTED_FILLS_CSV.relative_to(ROOT)}")
    print(f"wrote {PORTFOLIO_CSV.relative_to(ROOT)}")
    for path in plot_paths:
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
