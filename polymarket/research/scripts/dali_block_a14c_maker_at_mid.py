"""Block A1.4c maker-at-mid counterfactual.

This sidecar tests whether A1.3's current-level TOB imbalance signal survives
as a best-case maker thesis: post at mid, assume full priority, and only count
fills when a real taker print reaches the quote.

Rebate table used here, derived from ``FEE_BY_CATEGORY`` fee rates in
``dali_block_a1_analyze.py`` and the Block A1.4c prompt:

- Crypto: 20% of counterparty taker fee equivalent
- Sports: 25%
- Finance: 50%
- Politics/Economics/Culture/Weather/Tech/Other: 25%
- Geopolitics: 0%

No raw JSONL or canonical A1 artifacts are modified.
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
A13_DECILES = ANALYSIS / "csv_outputs" / "dali" / "block_a13_tob_decile_aggregate.csv"
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a14c_maker_at_mid_results.csv"
NOTE = NOTES / "block_a14c_maker_at_mid_findings.md"

FILL_WINDOWS = (1, 5, 10)
HOLD_HORIZONS = (5, 30)
ADVERSE_HORIZONS = (1, 5, 30)
EXIT_CONVENTIONS = ("exit_symmetric_maker", "exit_forced_taker")
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260529

REBATE_PCT_BY_CATEGORY = {
    "Crypto": 0.20,
    "Sports": 0.25,
    "Finance": 0.50,
    "Politics": 0.25,
    "Economics": 0.25,
    "Culture": 0.25,
    "Weather": 0.25,
    "Tech": 0.25,
    "Other": 0.25,
    "Geopolitics": 0.0,
}


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def safe_text(value: object, limit: int = 54) -> str:
    text = str(value or "").replace("|", "/")
    return text[:limit]


def taker_fee_amount(category: str, price: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = np.clip(price, 0.0, 1.0)
    return params["fee_rate"] * p * (1.0 - p)


def maker_rebate_bps(category: str, price: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    fee = taker_fee_amount(category, price)
    denom = np.clip(price, 0.01, 0.99)
    pct_rebate = REBATE_PCT_BY_CATEGORY.get(category, REBATE_PCT_BY_CATEGORY["Other"])
    return fee * pct_rebate / denom * 10_000.0


def taker_fee_bps_on_entry_notional(
    category: str,
    exit_price: np.ndarray | pd.Series | float,
    entry_price: np.ndarray | pd.Series | float,
) -> np.ndarray | pd.Series | float:
    fee = taker_fee_amount(category, exit_price)
    denom = np.clip(entry_price, 0.01, 0.99)
    return fee / denom * 10_000.0


def load_signal_horizons() -> tuple[int, ...]:
    if not A13_DECILES.exists():
        return (1, 5, 30, 300)
    a13 = pd.read_csv(A13_DECILES)
    current = a13[a13["signal_variant"].eq("current_level")].copy()
    horizons = sorted(current["horizon_sec"].dropna().astype(int).unique())
    return tuple(horizons) or (1, 5, 30, 300)


def load_candidates() -> pd.DataFrame:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    results["horizon_sec"] = pd.to_numeric(results["horizon_sec"], errors="coerce")
    candidates = results[
        results["horizon_sec"].eq(5)
        & results["sample_size_label"].isin(["primary_read", "thin_wide_CI"])
    ].copy()
    candidates = candidates[
        ["run_id", "market_id", "family", "n_classifiable", "sample_size_label"]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no primary_read/thin_wide_CI A1 candidates found")
    return candidates


def load_feature_subset(candidates: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("candidates", candidates[["run_id", "market_id"]])
    query = f"""
        SELECT
            f.run_id,
            f.received_at,
            f.event_type,
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
            f.tob_imbalance,
            f.trade_price,
            f.trade_side,
            f.last_trade_side,
            f.trade_size
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
    for col in ("run_id", "event_type", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    for col in (
        "outcome_index",
        "best_bid",
        "best_ask",
        "mid",
        "tob_imbalance",
        "trade_price",
        "trade_size",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    df["trade_side_norm"] = (
        df["trade_side"]
        .fillna(df["last_trade_side"])
        .fillna("")
        .astype(str)
        .str.upper()
    )
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df = df.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)
    df["tob_imbalance_ffill"] = df.groupby(["run_id", "asset_id"], sort=False)["tob_imbalance"].ffill()
    df["tob_signal_current"] = df["direction_factor"] * df["tob_imbalance_ffill"]
    return df


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


def first_qualifying_trade(
    start_ns: int,
    end_ns: int,
    quote_price: float,
    desired_trade_side: str,
    price_relation: str,
    trade_times: np.ndarray,
    trade_prices: np.ndarray,
    trade_sides: np.ndarray,
) -> tuple[int | None, float | None]:
    left = int(np.searchsorted(trade_times, start_ns, side="left"))
    right = int(np.searchsorted(trade_times, end_ns, side="right"))
    if left >= right:
        return None, None
    for idx in range(left, right):
        if trade_sides[idx] != desired_trade_side:
            continue
        px = trade_prices[idx]
        if not np.isfinite(px):
            continue
        if price_relation == "le" and px <= quote_price + 1e-12:
            return int(trade_times[idx]), float(px)
        if price_relation == "ge" and px >= quote_price - 1e-12:
            return int(trade_times[idx]), float(px)
    return None, None


def assign_top_decile_signals(market: pd.DataFrame) -> pd.DataFrame:
    valid = market[
        market["is_book_state_complete"]
        & market["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & market["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & market["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & market["tob_signal_current"].replace([np.inf, -np.inf], np.nan).notna()
        & market["tob_signal_current"].ne(0)
    ].copy()
    if len(valid) < 10:
        return valid.iloc[0:0].copy()
    valid["abs_signal"] = valid["tob_signal_current"].abs()
    try:
        valid["signal_decile"] = pd.qcut(valid["abs_signal"], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return valid.iloc[0:0].copy()
    top_decile = valid["signal_decile"].max()
    out = valid[valid["signal_decile"].eq(top_decile)].copy()
    out["token_side"] = np.sign(out["tob_signal_current"]) * out["direction_factor"]
    out = out[out["token_side"].isin([-1.0, 1.0])].copy()
    out["event_time_ns"] = out["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    out["entry_price"] = out["mid"].astype(float)
    return out


def simulate_entry_fills(signals: pd.DataFrame, market: pd.DataFrame, fill_window_sec: int) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for asset_id, events in signals.groupby("asset_id", sort=False):
        asset_rows = market[market["asset_id"].eq(asset_id)].sort_values("received_at")
        trades = asset_rows[
            asset_rows["event_type"].eq("last_trade_price")
            & asset_rows["trade_side_norm"].isin(["BUY", "SELL"])
            & asset_rows["trade_price"].notna()
        ].copy()
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trade_prices = trades["trade_price"].to_numpy(dtype=float)
        trade_sides = trades["trade_side_norm"].to_numpy(dtype=object)

        piece = events.copy()
        fill_time = np.full(len(piece), np.nan, dtype=float)
        was_filled = np.zeros(len(piece), dtype=bool)
        for pos, row in enumerate(piece.itertuples(index=False)):
            start = int(row.event_time_ns)
            end = start + fill_window_sec * 1_000_000_000
            if row.token_side > 0:
                found_time, _ = first_qualifying_trade(
                    start,
                    end,
                    float(row.entry_price),
                    "SELL",
                    "le",
                    trade_times,
                    trade_prices,
                    trade_sides,
                )
            else:
                found_time, _ = first_qualifying_trade(
                    start,
                    end,
                    float(row.entry_price),
                    "BUY",
                    "ge",
                    trade_times,
                    trade_prices,
                    trade_sides,
                )
            if found_time is not None:
                fill_time[pos] = found_time
                was_filled[pos] = True
        piece["entry_filled"] = was_filled
        piece["fill_time_ns"] = fill_time
        out_parts.append(piece)
    return pd.concat(out_parts, ignore_index=True) if out_parts else signals.iloc[0:0].copy()


def add_adverse_selection(filled: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for asset_id, rows in filled.groupby("asset_id", sort=False):
        asset_rows = market[market["asset_id"].eq(asset_id)].sort_values("received_at")
        state = asset_rows[asset_rows["mid"].notna()].copy()
        state_times = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        mids = state["mid"].to_numpy(dtype=float)
        piece = rows.copy()
        for horizon in ADVERSE_HORIZONS:
            target = piece["fill_time_ns"].to_numpy(dtype=np.float64).astype(np.int64) + horizon * 1_000_000_000
            future_mid = state_at_or_before(state_times, mids, target)
            piece[f"adverse_selection_bps_{horizon}s"] = (
                piece["token_side"].to_numpy(dtype=float)
                * (future_mid - piece["entry_price"].to_numpy(dtype=float))
                / np.clip(piece["entry_price"].to_numpy(dtype=float), 0.01, 0.99)
                * 10_000.0
            )
        out_parts.append(piece)
    return pd.concat(out_parts, ignore_index=True) if out_parts else filled


def simulate_exit(
    filled: pd.DataFrame,
    market: pd.DataFrame,
    hold_sec: int,
    exit_convention: str,
    category: str,
) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for asset_id, rows in filled.groupby("asset_id", sort=False):
        asset_rows = market[market["asset_id"].eq(asset_id)].sort_values("received_at")
        state = asset_rows[
            asset_rows["best_bid"].notna()
            & asset_rows["best_ask"].notna()
            & asset_rows["mid"].notna()
        ].copy()
        state_times = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        bids = state["best_bid"].to_numpy(dtype=float)
        asks = state["best_ask"].to_numpy(dtype=float)
        mids = state["mid"].to_numpy(dtype=float)

        trades = asset_rows[
            asset_rows["event_type"].eq("last_trade_price")
            & asset_rows["trade_side_norm"].isin(["BUY", "SELL"])
            & asset_rows["trade_price"].notna()
        ].copy()
        trade_times = trades["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        trade_prices = trades["trade_price"].to_numpy(dtype=float)
        trade_sides = trades["trade_side_norm"].to_numpy(dtype=object)

        piece = rows.copy()
        n = len(piece)
        exit_price = np.full(n, np.nan, dtype=float)
        exit_kind = np.full(n, "unexit", dtype=object)
        exit_filled_maker = np.zeros(n, dtype=bool)
        exit_taker_fee_bps = np.zeros(n, dtype=float)

        fill_times = piece["fill_time_ns"].to_numpy(dtype=np.float64).astype(np.int64)
        token_side = piece["token_side"].to_numpy(dtype=float)
        entry_price = piece["entry_price"].to_numpy(dtype=float)

        if exit_convention == "exit_forced_taker":
            target = fill_times + hold_sec * 1_000_000_000
            future_bid = state_at_or_before(state_times, bids, target)
            future_ask = state_at_or_before(state_times, asks, target)
            exit_price = np.where(token_side > 0, future_bid, future_ask)
            valid = np.isfinite(exit_price) & (exit_price >= 0)
            exit_kind[valid] = "taker"
            exit_taker_fee_bps[valid] = taker_fee_bps_on_entry_notional(
                category,
                exit_price[valid],
                entry_price[valid],
            )
        else:
            post_time = fill_times + hold_sec * 1_000_000_000
            post_mid = state_at_or_before(state_times, mids, post_time)
            for pos in range(n):
                if not np.isfinite(post_mid[pos]):
                    continue
                start = int(post_time[pos])
                end = start + hold_sec * 1_000_000_000
                if token_side[pos] > 0:
                    found_time, _ = first_qualifying_trade(
                        start,
                        end,
                        float(post_mid[pos]),
                        "BUY",
                        "ge",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                else:
                    found_time, _ = first_qualifying_trade(
                        start,
                        end,
                        float(post_mid[pos]),
                        "SELL",
                        "le",
                        trade_times,
                        trade_prices,
                        trade_sides,
                    )
                if found_time is not None:
                    exit_price[pos] = post_mid[pos]
                    exit_kind[pos] = "maker"
                    exit_filled_maker[pos] = True
                    continue

                fallback_time = fill_times[pos] + 2 * hold_sec * 1_000_000_000
                fallback_bid = state_at_or_before(state_times, bids, np.array([fallback_time], dtype=np.int64))[0]
                fallback_ask = state_at_or_before(state_times, asks, np.array([fallback_time], dtype=np.int64))[0]
                fallback_price = fallback_bid if token_side[pos] > 0 else fallback_ask
                if np.isfinite(fallback_price):
                    exit_price[pos] = fallback_price
                    exit_kind[pos] = "taker_fallback"
                    exit_taker_fee_bps[pos] = taker_fee_bps_on_entry_notional(
                        category,
                        fallback_price,
                        entry_price[pos],
                    )

        piece["exit_price"] = exit_price
        piece["exit_kind"] = exit_kind
        piece["exit_maker_filled"] = exit_filled_maker
        piece["exit_taker_fee_bps"] = exit_taker_fee_bps
        gross = token_side * (exit_price - entry_price) / np.clip(entry_price, 0.01, 0.99) * 10_000.0
        piece["maker_rebate_bps"] = maker_rebate_bps(category, entry_price)
        piece["pnl_bps"] = gross + piece["maker_rebate_bps"].to_numpy(dtype=float) - exit_taker_fee_bps
        piece.loc[~np.isfinite(exit_price), "pnl_bps"] = np.nan
        out_parts.append(piece)
    return pd.concat(out_parts, ignore_index=True) if out_parts else filled.iloc[0:0].copy()


def bootstrap_mean_ci(rows: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = rows[["fill_time_ns", "pnl_bps"]].dropna().copy()
    clean = clean[np.isfinite(clean["pnl_bps"])]
    if len(clean) < 5:
        return math.nan, math.nan
    elapsed = (clean["fill_time_ns"] - clean["fill_time_ns"].min()) / 1_000_000_000.0
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


def summarize_combo(
    market_label: str,
    slug: str,
    family: str,
    sample_size_label: str,
    signal_horizon: int,
    fill_window: int,
    hold_horizon: int,
    exit_convention: str,
    n_signal_events: int,
    entry_filled: pd.DataFrame,
    exited: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    n_filled = int(len(entry_filled))
    fill_rate = n_filled / n_signal_events if n_signal_events else math.nan
    n_exited = int(exited["pnl_bps"].notna().sum()) if not exited.empty else 0
    mean_pnl = float(exited["pnl_bps"].mean()) if n_exited else math.nan
    median_pnl = float(exited["pnl_bps"].median()) if n_exited else math.nan
    win_rate = float(exited["pnl_bps"].gt(0).mean()) if n_exited else math.nan
    ci_lo, ci_hi = bootstrap_mean_ci(exited, seed)
    return {
        "market": market_label,
        "slug": slug,
        "family": family,
        "sample_size_label": sample_size_label,
        "signal_variant": "current_level_tob_imbalance",
        "signal_horizon": signal_horizon,
        "fill_window_sec": fill_window,
        "hold_sec": hold_horizon,
        "exit_convention": exit_convention,
        "n_signal_events": n_signal_events,
        "n_filled": n_filled,
        "n_unfilled": int(n_signal_events - n_filled),
        "fill_rate": fill_rate,
        "n_exited": n_exited,
        "n_exit_maker": int(exited["exit_maker_filled"].sum()) if "exit_maker_filled" in exited else 0,
        "n_exit_forced": int(exited["exit_kind"].isin(["taker", "taker_fallback"]).sum()) if "exit_kind" in exited else 0,
        "exit_maker_fill_rate": (
            float(exited["exit_maker_filled"].mean())
            if "exit_maker_filled" in exited and len(exited)
            else math.nan
        ),
        "mean_pnl_bps": mean_pnl,
        "median_pnl_bps": median_pnl,
        "win_rate": win_rate,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_rebate_bps": float(exited["maker_rebate_bps"].mean()) if n_exited else math.nan,
        "mean_adverse_selection_bps_1s": (
            float(entry_filled["adverse_selection_bps_1s"].mean()) if n_filled else math.nan
        ),
        "mean_adverse_selection_bps_5s": (
            float(entry_filled["adverse_selection_bps_5s"].mean()) if n_filled else math.nan
        ),
        "mean_adverse_selection_bps_30s": (
            float(entry_filled["adverse_selection_bps_30s"].mean()) if n_filled else math.nan
        ),
    }


def run_simulation(df: pd.DataFrame, candidates: pd.DataFrame, signal_horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_meta = candidates.set_index(["run_id", "market_id"]).to_dict("index")
    for market_idx, ((run_id, market_id), market) in enumerate(
        df.groupby(["run_id", "market_id"], sort=False),
        start=1,
    ):
        market = market.sort_values(["asset_id", "received_at"]).copy()
        meta = candidate_meta.get((run_id, market_id), {})
        family = str(meta.get("family") or market["family"].replace("", np.nan).dropna().iloc[0])
        category = family_category(family)
        sample_size_label = str(meta.get("sample_size_label", ""))
        slug = (
            str(market["slug"].replace("", np.nan).dropna().iloc[0])
            if market["slug"].astype(bool).any()
            else str(market_id)
        )
        market_label = f"{run_id}:{market_id}"
        signals = assign_top_decile_signals(market)
        n_signal_events = int(len(signals))
        print(
            f"{market_idx:02d}/{df[['run_id','market_id']].drop_duplicates().shape[0]:02d} "
            f"{market_label} signals={n_signal_events:,}",
            flush=True,
        )

        entry_by_window: dict[int, pd.DataFrame] = {}
        for fill_window in FILL_WINDOWS:
            entry = simulate_entry_fills(signals, market, fill_window)
            entry = entry[entry["entry_filled"]].copy()
            if not entry.empty:
                entry = add_adverse_selection(entry, market)
            entry_by_window[fill_window] = entry

        for signal_horizon in signal_horizons:
            for fill_window in FILL_WINDOWS:
                entry = entry_by_window[fill_window]
                for hold_horizon in HOLD_HORIZONS:
                    for exit_convention in EXIT_CONVENTIONS:
                        exited = (
                            simulate_exit(entry, market, hold_horizon, exit_convention, category)
                            if not entry.empty
                            else entry
                        )
                        rows.append(
                            summarize_combo(
                                market_label,
                                slug,
                                family,
                                sample_size_label,
                                signal_horizon,
                                fill_window,
                                hold_horizon,
                                exit_convention,
                                n_signal_events,
                                entry,
                                exited,
                                RNG_SEED
                                + market_idx * 10_000
                                + signal_horizon * 100
                                + fill_window * 10
                                + hold_horizon
                                + (0 if exit_convention == "exit_symmetric_maker" else 1),
                            )
                        )
    out = pd.DataFrame(rows)
    order = [
        "market",
        "slug",
        "family",
        "sample_size_label",
        "signal_variant",
        "signal_horizon",
        "fill_window_sec",
        "hold_sec",
        "exit_convention",
        "n_signal_events",
        "n_filled",
        "n_unfilled",
        "fill_rate",
        "n_exited",
        "n_exit_maker",
        "n_exit_forced",
        "exit_maker_fill_rate",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_rebate_bps",
        "mean_adverse_selection_bps_1s",
        "mean_adverse_selection_bps_5s",
        "mean_adverse_selection_bps_30s",
    ]
    return out[order]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def best_grid_table(results: pd.DataFrame) -> str:
    unique = results.drop_duplicates(["market", "fill_window_sec", "hold_sec", "exit_convention"]).copy()
    grid = (
        unique.groupby(["fill_window_sec", "hold_sec", "exit_convention"], as_index=False)
        .agg(
            mean_pnl_bps=("mean_pnl_bps", "mean"),
            median_pnl_bps=("median_pnl_bps", "median"),
            mean_fill_rate=("fill_rate", "mean"),
            total_filled=("n_filled", "sum"),
            positive_markets=("mean_pnl_bps", lambda s: int((s > 0).sum())),
            cells=("market", "count"),
        )
        .sort_values(["mean_pnl_bps", "mean_fill_rate"], ascending=False)
    )
    rows = []
    for row in grid.itertuples(index=False):
        rows.append([
            str(int(row.fill_window_sec)),
            str(int(row.hold_sec)),
            str(row.exit_convention),
            bps(float(row.mean_pnl_bps)),
            bps(float(row.median_pnl_bps)),
            pct(float(row.mean_fill_rate)),
            f"{int(row.total_filled):,}",
            f"{int(row.positive_markets)}/{int(row.cells)}",
        ])
    return markdown_table(
        ["fill W", "hold H", "exit", "mean pnl", "median pnl", "fill rate", "fills", "positive cells"],
        rows,
    )


def per_market_verdicts(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    unique = results.drop_duplicates(["market", "fill_window_sec", "hold_sec", "exit_convention"]).copy()
    best = (
        unique.sort_values(["market", "mean_pnl_bps", "fill_rate"], ascending=[True, False, False])
        .groupby("market", as_index=False)
        .head(1)
        .copy()
    )
    verdicts = []
    verdict_rows = []
    for row in best.itertuples(index=False):
        if row.n_filled < 30 or row.fill_rate < 0.01:
            verdict = "fills too rare"
        elif np.isfinite(row.mean_pnl_bps) and row.mean_pnl_bps > 0 and row.mean_adverse_selection_bps_5s > -50:
            verdict = "maker thesis lives"
        else:
            verdict = "adverse selection wipes rebate"
        verdict_rows.append({**row._asdict(), "verdict": verdict})
        verdicts.append(
            f"- `{row.market}` ({safe_text(row.slug)}): {verdict}; best cell W={int(row.fill_window_sec)}s, "
            f"H={int(row.hold_sec)}s, {row.exit_convention}, mean {bps(float(row.mean_pnl_bps))}, "
            f"fill {pct(float(row.fill_rate))}, 5s adverse {bps(float(row.mean_adverse_selection_bps_5s))}."
        )
    return "\n".join(verdicts), pd.DataFrame(verdict_rows)


def top_cells_table(results: pd.DataFrame, limit: int = 16) -> str:
    unique = results.drop_duplicates(["market", "fill_window_sec", "hold_sec", "exit_convention"]).copy()
    sub = unique.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).head(limit)
    rows = []
    for row in sub.itertuples(index=False):
        rows.append([
            safe_text(row.market, 18),
            safe_text(row.slug, 42),
            "current",
            str(int(row.fill_window_sec)),
            str(int(row.hold_sec)),
            str(row.exit_convention).replace("exit_", ""),
            f"{int(row.n_signal_events):,}",
            f"{int(row.n_filled):,}",
            pct(float(row.fill_rate)),
            bps(float(row.mean_pnl_bps)),
            bps(float(row.mean_adverse_selection_bps_5s)),
        ])
    return markdown_table(
        ["market", "slug", "signal", "W", "H", "exit", "signals", "fills", "fill rate", "mean pnl", "5s adverse"],
        rows,
    )


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    unique = results.drop_duplicates(["market", "fill_window_sec", "hold_sec", "exit_convention"]).copy()
    raw_rows = int(len(results))
    total_cells = int(len(unique))
    positive_cells = int(unique["mean_pnl_bps"].gt(0).sum())
    best = unique.sort_values(["mean_pnl_bps", "fill_rate"], ascending=False).iloc[0]
    best_exit = str(best["exit_convention"])
    verdict_text, verdict_df = per_market_verdicts(results)
    lives = int(verdict_df["verdict"].eq("maker thesis lives").sum()) if not verdict_df.empty else 0
    rare = int(verdict_df["verdict"].eq("fills too rare").sum()) if not verdict_df.empty else 0
    wiped = int(verdict_df["verdict"].eq("adverse selection wipes rebate").sum()) if not verdict_df.empty else 0

    note = f"""---
tags: [dali, block-a14c, maker-thesis, results]
---

# Block A1.4c Maker-at-Mid Findings

## Headline

The maker-at-mid counterfactual is materially more interesting than taker execution, but it is a best-case maker assumption. The CSV has {raw_rows:,} rows because it preserves the four A1.3 current-level horizon labels; collapsing those duplicate labels leaves {total_cells:,} unique market/grid cells, of which {positive_cells:,} have positive mean PnL. The best cell is `{safe_text(best['market'])}` / `{safe_text(best['slug'])}` with W={int(best['fill_window_sec'])}s, H={int(best['hold_sec'])}s, `{best_exit}`, mean {bps(float(best['mean_pnl_bps']))}, fill rate {pct(float(best['fill_rate']))}, and 5s signed adverse selection {bps(float(best['mean_adverse_selection_bps_5s']))}. Per-market verdicts: {lives} maker-thesis-live, {rare} fills-too-rare, {wiped} adverse-selection-wipes-rebate.

## Method

- Universe: A1 markets labeled `primary_read` or `thin_wide_CI` at the 5s horizon.
- Signal: A1.3 current-level TOB imbalance, `direction_factor * tob_imbalance`, using per-market top absolute decile.
- Signal horizon: carried from A1.3 current-level horizons. Because current-level TOB is a state variable, the signal rows are horizon-invariant; repeated horizon labels are included for traceability.
- Entry: post at current mid on the signal-favorable token side. Long token posts bid at mid; short token posts ask at mid.
- Fill windows: 1s, 5s, 10s. Unfilled signal events are retained in `n_unfilled` and `fill_rate`.
- Exit conventions: `exit_forced_taker` closes at the opposite touch after H; `exit_symmetric_maker` posts opposite-side at mid after H and, if not filled within H, forces a taker close at t_fill + 2H.
- Fees/rebates: entry maker rebate is credited once. Taker fee is charged on forced-taker exits and on symmetric-maker fallback exits.
- Queue model: none. Inside-spread quotes are assumed to have full priority, so these are best-case maker numbers.
- Bootstrap: 200 resamples over contiguous 300s blocks on filled PnL.

## Grid Winners

{best_grid_table(results)}

## Top Cells

{top_cells_table(results)}

## Per-Market Verdicts

{verdict_text}

## Interpretation

Positive maker-at-mid cells mean the spread capture plus entry rebate beat adverse selection under full-priority inside-spread posting. They do not imply deployability: queue priority, quote cancellation, touch size, latency, and inventory constraints are still absent. Low fill-rate rows should be treated as opportunity diagnostics, not scalable PnL estimates.

Recommended next action for Justin: A2 should include a maker-focused branch, but only with queue/latency instrumentation and a minimum fill-rate screen rather than a pure OFI taker thesis.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    signal_horizons = load_signal_horizons()
    candidates = load_candidates()
    features = load_feature_subset(candidates)
    results = run_simulation(features, candidates, signal_horizons)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
