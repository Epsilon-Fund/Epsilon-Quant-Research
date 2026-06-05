"""Block K1 maker-economics decomposition.

Research-only sidecar for the Block K maker pivot gate. It reconstructs
zero-skew passive maker fills from the A1 live feature parquet, using the same
basic A1.4h fill evidence: a passive quote is considered filled only when a
real taker print crosses a previously observed quote within the fill window.

No directional signal is used here. The simulated maker is always symmetric:
both token sides are quoted at the current mid. Economics are then decomposed
into spread capture, maker rebate, adverse selection, and an inventory /
resolution mark-to-market charge.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
OUT_CSV = ANALYSIS / "csv_outputs" / "market_making" / "k1_maker_economics.csv"
NOTE = NOTES / "block_k1_maker_economics_findings.md"

RUN_POOL = ("a0", "a0b", "a0c", "a0c_roll")
EXCLUDED_SLUG_SUBSTRINGS = ("will-jd-vance",)
FILL_WINDOWS = (1, 5, 10)
ADVERSE_HORIZONS = (5, 30, 60)
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260530
ROBUST_MIN_FILLS = 30

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


def family_category(family: object) -> str:
    """Map local research family labels to Block K fee categories."""
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


def safe_text(value: object, limit: int = 60) -> str:
    text = str(value or "").replace("|", "/")
    return text[:limit]


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


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


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
            spread,
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
    for col in ("best_bid", "best_ask", "spread", "mid", "trade_price", "trade_size"):
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


def candidate_fills_for_market(df: pd.DataFrame, fill_window_sec: int) -> tuple[pd.DataFrame, int, int]:
    """Return raw passive fills plus quote-event/order opportunity counts."""
    raw_candidates: list[pd.DataFrame] = []
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
        # Quote opportunities are book states. Trade rows can carry a book
        # snapshot, but using them as prior quote states leaks the fill itself.
        quote_state = state[state["event_type"].isin(["book", "price_change"])].copy()
        quote_state = quote_state.drop_duplicates(["received_at", "best_bid", "best_ask", "mid"])
        n_quote_events += int(len(quote_state))
        if quote_state.empty:
            continue

        quote_times = quote_state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        quote_mid = quote_state["mid"].to_numpy(dtype=float)
        quote_bid = quote_state["best_bid"].to_numpy(dtype=float)
        quote_ask = quote_state["best_ask"].to_numpy(dtype=float)

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
        idx = np.searchsorted(quote_times, trade_times, side="right") - 1
        valid = idx >= 0
        if not valid.any():
            continue

        quote_age_ns = trade_times - quote_times[np.clip(idx, 0, len(quote_times) - 1)]
        valid &= quote_age_ns >= 0
        valid &= quote_age_ns <= fill_window_sec * 1_000_000_000

        trade_px = trades["trade_price"].to_numpy(dtype=float)
        trade_side = trades["trade_side_norm"].to_numpy(dtype=object)
        entry_mid = quote_mid[np.clip(idx, 0, len(quote_mid) - 1)]
        crossed_mid = np.where(
            trade_side == "SELL",
            trade_px <= entry_mid + 1e-12,
            trade_px >= entry_mid - 1e-12,
        )
        valid &= crossed_mid
        if not valid.any():
            continue

        piece = trades.loc[valid].copy()
        valid_idx = idx[valid]
        piece["quote_time_ns"] = quote_times[valid_idx]
        piece["fill_time_ns"] = trade_times[valid]
        piece["quote_age_sec"] = quote_age_ns[valid] / 1_000_000_000.0
        piece["entry_price"] = entry_mid[valid]
        piece["quote_best_bid"] = quote_bid[valid_idx]
        piece["quote_best_ask"] = quote_ask[valid_idx]
        piece["quote_spread"] = piece["quote_best_ask"] - piece["quote_best_bid"]
        piece["maker_side"] = np.where(piece["trade_side_norm"].eq("SELL"), "BUY", "SELL")
        piece["token_side"] = np.where(piece["maker_side"].eq("BUY"), 1.0, -1.0)
        raw_candidates.append(piece)

    candidates = pd.concat(raw_candidates, ignore_index=True) if raw_candidates else df.iloc[0:0].copy()
    return candidates, n_quote_events, n_quote_events * 2


def add_future_mids(candidates: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out_parts: list[pd.DataFrame] = []
    for asset_id, rows in candidates.groupby("asset_id", sort=False):
        state = market[
            market["asset_id"].eq(asset_id)
            & market["is_book_state_complete"].fillna(False)
            & market["mid"].replace([np.inf, -np.inf], np.nan).notna()
            & market["mid"].between(0.0, 1.0, inclusive="both")
        ].sort_values("received_at")
        piece = rows.copy()
        if state.empty:
            for horizon in ADVERSE_HORIZONS:
                piece[f"future_mid_{horizon}s"] = np.nan
            out_parts.append(piece)
            continue
        state = state.drop_duplicates(["received_at", "mid"])
        state_times = state["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        mids = state["mid"].to_numpy(dtype=float)
        fill_times = piece["fill_time_ns"].to_numpy(dtype=np.int64)
        for horizon in ADVERSE_HORIZONS:
            target = fill_times + horizon * 1_000_000_000
            piece[f"future_mid_{horizon}s"] = state_at_or_before(state_times, mids, target)
        out_parts.append(piece)
    return pd.concat(out_parts, ignore_index=True)


def apply_non_overlap(candidates: pd.DataFrame, horizon_sec: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    future_col = f"future_mid_{horizon_sec}s"
    valid = candidates[
        candidates["fill_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & candidates["quote_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & candidates[future_col].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if valid.empty:
        return valid
    valid["exit_time_ns"] = valid["fill_time_ns"].astype("int64") + horizon_sec * 1_000_000_000
    valid = valid.sort_values(["fill_time_ns", "quote_time_ns", "asset_id"])

    intervals: list[tuple[int, int]] = []
    keep_positions: list[int] = []
    for pos, row in enumerate(valid.itertuples(index=False)):
        quote_ns = int(row.quote_time_ns)
        fill_ns = int(row.fill_time_ns)
        exit_ns = int(row.exit_time_ns)
        if exit_ns <= fill_ns:
            continue
        blocked = any(start <= quote_ns <= end or start <= fill_ns <= end for start, end in intervals)
        if blocked:
            continue
        keep_positions.append(pos)
        intervals.append((fill_ns, exit_ns))
    if not keep_positions:
        return valid.iloc[0:0].copy()
    out = valid.iloc[keep_positions].copy()
    out["executed_fill_rank"] = np.arange(1, len(out) + 1)
    return out


def add_economics(executed: pd.DataFrame, category: str, horizon_sec: int) -> pd.DataFrame:
    if executed.empty:
        return executed
    out = executed.copy()
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = out["entry_price"].astype(float).clip(0.01, 0.99)
    future_mid = out[f"future_mid_{horizon_sec}s"].astype(float)
    token_side = out["token_side"].astype(float)

    out["spread_capture_bps"] = out["quote_spread"].astype(float).clip(lower=0.0) / 2.0 / p * 10_000.0
    taker_fee_usd = params["fee_rate"] * p * (1.0 - p)
    out["rebate_bps"] = taker_fee_usd * params["maker_rebate_pct"] / p * 10_000.0
    # Positive adverse-selection values are costs; negative values are favorable.
    out[f"adverse_selection_bps_{horizon_sec}s"] = (
        -token_side * (future_mid - out["entry_price"].astype(float)) / p * 10_000.0
    )
    abs_move_bps = (future_mid - out["entry_price"].astype(float)).abs() / p * 10_000.0
    # Realized inventory / resolution risk proxy. It charges half the absolute
    # post-fill mid move, including settlement-adjacent jumps visible in A1.
    out["inv_resolution_charge_bps"] = 0.5 * abs_move_bps
    out["net_maker_pnl_bps"] = (
        out["spread_capture_bps"]
        + out["rebate_bps"]
        - out[f"adverse_selection_bps_{horizon_sec}s"]
        - out["inv_resolution_charge_bps"]
    )
    out["fee_rate"] = params["fee_rate"]
    out["rebate_pct"] = params["maker_rebate_pct"]
    out["adverse_horizon_sec"] = horizon_sec
    return out


def summarize_group(
    rows: pd.DataFrame,
    *,
    row_type: str,
    market_id: str,
    slug: str,
    question: str,
    family: str,
    category: str,
    runs: str,
    fill_window_sec: int,
    horizon_sec: int,
    n_quote_events: int,
    n_quote_orders: int,
    n_raw_fills: int,
    seed: int,
) -> dict[str, object]:
    n = int(len(rows))
    fill_rate = n / n_quote_orders if n_quote_orders else math.nan
    ci_lo, ci_hi = bootstrap_mean_ci(rows, "net_maker_pnl_bps", seed) if n else (math.nan, math.nan)
    mean_net = float(rows["net_maker_pnl_bps"].mean()) if n else math.nan
    return {
        "row_type": row_type,
        "fee_segment": fee_segment(category),
        "market_id": market_id,
        "slug": slug,
        "question": question,
        "family": family,
        "category": category,
        "runs": runs,
        "fill_window_sec": fill_window_sec,
        "adverse_horizon_sec": horizon_sec,
        "n_quote_events": int(n_quote_events),
        "n_quote_orders": int(n_quote_orders),
        "n_raw_candidate_fills": int(n_raw_fills),
        "n_executed_fills": n,
        "fill_rate": fill_rate,
        "mean_spread_capture_bps": float(rows["spread_capture_bps"].mean()) if n else math.nan,
        "mean_rebate_bps": float(rows["rebate_bps"].mean()) if n else math.nan,
        f"mean_adverse_selection_bps_{horizon_sec}s": (
            float(rows[f"adverse_selection_bps_{horizon_sec}s"].mean()) if n else math.nan
        ),
        "mean_inv_resolution_charge_bps": (
            float(rows["inv_resolution_charge_bps"].mean()) if n else math.nan
        ),
        "mean_net_maker_pnl_bps": mean_net,
        "median_net_maker_pnl_bps": float(rows["net_maker_pnl_bps"].median()) if n else math.nan,
        "win_rate": float(rows["net_maker_pnl_bps"].gt(0).mean()) if n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "clears_zero": bool(
            n >= ROBUST_MIN_FILLS
            and np.isfinite(mean_net)
            and mean_net > 0
            and np.isfinite(ci_lo)
            and ci_lo > 0
        ),
        "fee_rate": float(rows["fee_rate"].mean()) if n else FEE_BY_CATEGORY[category]["fee_rate"],
        "rebate_pct": float(rows["rebate_pct"].mean()) if n else FEE_BY_CATEGORY[category]["maker_rebate_pct"],
        "risk_charge_method": "0.5 * abs(mid_horizon - entry_mid) / entry_mid",
    }


def category_summaries(market_rows: list[pd.DataFrame], market_summaries: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not market_rows:
        return out
    fills = pd.concat(market_rows, ignore_index=True)
    for (category, fill_window, horizon), rows in fills.groupby(
        ["category", "fill_window_sec", "adverse_horizon_sec"], sort=True
    ):
        summary_basis = market_summaries[
            market_summaries["row_type"].eq("market")
            & market_summaries["category"].eq(category)
            & market_summaries["fill_window_sec"].eq(fill_window)
            & market_summaries["adverse_horizon_sec"].eq(horizon)
        ]
        n_quote_events = int(summary_basis["n_quote_events"].sum())
        n_quote_orders = int(summary_basis["n_quote_orders"].sum())
        n_raw_fills = int(summary_basis["n_raw_candidate_fills"].sum())
        family = "category_aggregate"
        out.append(
            summarize_group(
                rows,
                row_type="category",
                market_id=f"category:{category}",
                slug=f"category:{category}",
                question="",
                family=family,
                category=str(category),
                runs=", ".join(RUN_POOL),
                fill_window_sec=int(fill_window),
                horizon_sec=int(horizon),
                n_quote_events=n_quote_events,
                n_quote_orders=n_quote_orders,
                n_raw_fills=n_raw_fills,
                seed=RNG_SEED + int(fill_window) * 100 + int(horizon),
            )
        )
    return out


def write_note(results: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    canonical = results[
        results["row_type"].eq("category")
        & results["fill_window_sec"].eq(5)
        & results["adverse_horizon_sec"].eq(30)
    ].copy()
    all_category = results[results["row_type"].eq("category")].copy()
    market = results[results["row_type"].eq("market")].copy()
    cleared = all_category[all_category["clears_zero"]].sort_values(
        ["mean_net_maker_pnl_bps", "fill_rate"], ascending=False
    )
    gate = "PROCEED to K2" if not cleared.empty else "STOP: maker pivot does not start"

    canonical_rows: list[list[str]] = []
    for row in canonical.sort_values(["fee_segment", "category"]).itertuples(index=False):
        adverse_col = f"mean_adverse_selection_bps_{int(row.adverse_horizon_sec)}s"
        canonical_rows.append(
            [
                str(row.fee_segment),
                str(row.category),
                f"{int(row.n_executed_fills):,}",
                pct(float(row.fill_rate)),
                bps(float(row.mean_spread_capture_bps)),
                bps(float(row.mean_rebate_bps)),
                bps(float(getattr(row, adverse_col))),
                bps(float(row.mean_inv_resolution_charge_bps)),
                bps(float(row.mean_net_maker_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                "yes" if bool(row.clears_zero) else "no",
            ]
        )

    best_market_rows: list[list[str]] = []
    best_markets = (
        market.sort_values(["market_id", "mean_net_maker_pnl_bps"], ascending=[True, False])
        .groupby("market_id", as_index=False)
        .head(1)
        .sort_values(["mean_net_maker_pnl_bps", "fill_rate"], ascending=False)
        .head(15)
    )
    for row in best_markets.itertuples(index=False):
        best_market_rows.append(
            [
                safe_text(row.market_id, 12),
                safe_text(row.slug, 42),
                str(row.category),
                str(int(row.fill_window_sec)),
                str(int(row.adverse_horizon_sec)),
                f"{int(row.n_executed_fills):,}",
                pct(float(row.fill_rate)),
                bps(float(row.mean_net_maker_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                "yes" if bool(row.clears_zero) else "no",
            ]
        )

    segment_rows: list[list[str]] = []
    for row in all_category.sort_values(["fee_segment", "category", "fill_window_sec", "adverse_horizon_sec"]).itertuples(index=False):
        segment_rows.append(
            [
                str(row.fee_segment),
                str(row.category),
                str(int(row.fill_window_sec)),
                str(int(row.adverse_horizon_sec)),
                f"{int(row.n_executed_fills):,}",
                pct(float(row.fill_rate)),
                bps(float(row.mean_net_maker_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                "yes" if bool(row.clears_zero) else "no",
            ]
        )

    if cleared.empty:
        headline = (
            "No category clears zero after the K1 baseline maker decomposition. "
            "Under the gate rule, the maker pivot does not start."
        )
    else:
        cats = ", ".join(sorted(cleared["category"].unique()))
        headline = (
            f"At least one category clears zero ({cats}) under the K1 baseline maker decomposition. "
            "Under the gate rule, proceed to K2."
        )

    note = f"""---
tags: [dali, block-k1, maker-economics, gate]
---

# Block K1 Maker-Economics Findings

## Headline

{headline}

Canonical read: zero directional skew, 5s fill window, 30s adverse-selection horizon. Gate result: **{gate}**.

## Canonical Category Table

{markdown_table(
    [
        "segment",
        "category",
        "fills",
        "fill rate",
        "spread",
        "rebate",
        "adverse",
        "inv/res",
        "net",
        "95% CI",
        "clears",
    ],
    canonical_rows,
)}

## Full Category Grid

{markdown_table(
    ["segment", "category", "fill W", "horizon", "fills", "fill rate", "net", "95% CI", "clears"],
    segment_rows,
)}

## Best Market Rows

{markdown_table(
    ["market", "slug", "category", "fill W", "horizon", "fills", "fill rate", "net", "95% CI", "clears"],
    best_market_rows,
)}

## Method

- Data: `data/analysis/block_a1_features.parquet`, pooled runs `{', '.join(RUN_POOL)}` as one in-sample set. No holdout split.
- Exclusion: slugs containing `will-jd-vance` were removed because of the known quote-noise issue.
- Zero skew: no OFI/TFI/TOB directional signal is used. The simulated maker quotes both sides symmetrically at the current mid.
- Fill proxy: A1.4h-style passive evidence. A bid fill requires a real SELL print at or below the prior mid within W; an ask fill requires a real BUY print at or above the prior mid within W. W is reported for 1s, 5s, and 10s.
- Non-overlap: after an actual fill, the market is blocked until `fill_time + adverse_horizon`. Unfilled quote opportunities do not block later fills.
- Rebate: `feeRate * p * (1-p) * rebate_pct`, normalized by entry price. Rebate pct is 20% for Crypto, 25% for other fee-enabled categories, and 0% for Geopolitics.
- Decomposition: `net = spread_capture + rebate - adverse_selection - inv_resolution_charge`.
- Spread capture: half the quoted spread at the prior book state, normalized by entry mid. This keeps K1 as a generous full-priority maker gate.
- Adverse selection: signed mark-to-market loss from entry mid to future mid at 5/30/60s. Positive values are costs.
- Inventory/resolution charge: `0.5 * abs(future_mid - entry_mid) / entry_mid`, using the same future mid path; settlement-adjacent jumps are included when visible in A1. A1 does not carry final resolution prices, so no synthetic resolution payoff is added.
- CI: 500 bootstrap resamples of contiguous 300s fill-time blocks, blocked within market.

## Interpretation

Geopolitics is the fee-free negative control: it has no rebate cushion. Fee-enabled categories receive the official rebate cushion, but the gate only clears if the category mean is positive, the lower CI is above zero, and there are at least {ROBUST_MIN_FILLS} non-overlap fills.

Important caveat: because fee-free Geopolitics also clears, the K1 pass is not a pure rebate-cushion proof. It is mostly a generous spread-capture/full-priority maker baseline. Treat the gate result as permission to proceed to K2 simulation and queue/capacity stress, not as a deployable market-making result.

CSV: `data/analysis/csv_outputs/market_making/k1_maker_economics.csv`.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(f"missing features parquet: {FEATURES}")

    con = duckdb.connect()
    market_list = load_market_list(con)
    if market_list.empty:
        raise SystemExit("no markets found for K1 run pool")

    market_summaries: list[dict[str, object]] = []
    executed_parts: list[pd.DataFrame] = []

    for market_idx, meta in enumerate(market_list.itertuples(index=False), start=1):
        market_id = str(meta.market_id)
        df = load_market_features(con, market_id)
        if df.empty:
            continue
        family = str(meta.family or "")
        category = family_category(family)
        slug = str(meta.slug or market_id)
        question = str(meta.question or "")
        runs = str(meta.runs or "")
        print(
            f"K1 market {market_idx:02d}/{len(market_list):02d} {market_id} "
            f"{category} rows={len(df):,}",
            flush=True,
        )

        for fill_window in FILL_WINDOWS:
            candidates, n_quote_events, n_quote_orders = candidate_fills_for_market(df, fill_window)
            candidates = add_future_mids(candidates, df)
            n_raw = int(len(candidates))
            for horizon in ADVERSE_HORIZONS:
                executed = apply_non_overlap(candidates, horizon)
                executed = add_economics(executed, category, horizon)
                if not executed.empty:
                    executed["category"] = category
                    executed["fill_window_sec"] = fill_window
                    executed["adverse_horizon_sec"] = horizon
                    executed_parts.append(
                        executed[
                            [
                                "market_id",
                                "asset_id",
                                "category",
                                "fill_window_sec",
                                "adverse_horizon_sec",
                                "fill_time_ns",
                                "spread_capture_bps",
                                "rebate_bps",
                                f"adverse_selection_bps_{horizon}s",
                                "inv_resolution_charge_bps",
                                "net_maker_pnl_bps",
                                "fee_rate",
                                "rebate_pct",
                            ]
                        ].rename(columns={f"adverse_selection_bps_{horizon}s": "adverse_selection_bps"})
                    )
                market_summaries.append(
                    summarize_group(
                        executed,
                        row_type="market",
                        market_id=market_id,
                        slug=slug,
                        question=question,
                        family=family,
                        category=category,
                        runs=runs,
                        fill_window_sec=fill_window,
                        horizon_sec=horizon,
                        n_quote_events=n_quote_events,
                        n_quote_orders=n_quote_orders,
                        n_raw_fills=n_raw,
                        seed=RNG_SEED + market_idx * 10_000 + fill_window * 100 + horizon,
                    )
                )

    con.close()

    market_summary_df = pd.DataFrame(market_summaries)
    executed_for_categories: list[pd.DataFrame] = []
    if executed_parts:
        fills = pd.concat(executed_parts, ignore_index=True)
        for (category, fill_window, horizon), rows in fills.groupby(
            ["category", "fill_window_sec", "adverse_horizon_sec"], sort=False
        ):
            renamed = rows.rename(
                columns={"adverse_selection_bps": f"adverse_selection_bps_{int(horizon)}s"}
            ).copy()
            executed_for_categories.append(renamed)

    category_rows = category_summaries(executed_for_categories, market_summary_df)
    results = pd.concat([market_summary_df, pd.DataFrame(category_rows)], ignore_index=True)

    # Keep a stable, wide CSV schema even though the adverse-selection mean is
    # horizon-specific in each row.
    for horizon in ADVERSE_HORIZONS:
        col = f"mean_adverse_selection_bps_{horizon}s"
        if col not in results:
            results[col] = np.nan
    order = [
        "row_type",
        "fee_segment",
        "market_id",
        "slug",
        "question",
        "family",
        "category",
        "runs",
        "fill_window_sec",
        "adverse_horizon_sec",
        "n_quote_events",
        "n_quote_orders",
        "n_raw_candidate_fills",
        "n_executed_fills",
        "fill_rate",
        "mean_spread_capture_bps",
        "mean_rebate_bps",
        "mean_adverse_selection_bps_5s",
        "mean_adverse_selection_bps_30s",
        "mean_adverse_selection_bps_60s",
        "mean_inv_resolution_charge_bps",
        "mean_net_maker_pnl_bps",
        "median_net_maker_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "clears_zero",
        "fee_rate",
        "rebate_pct",
        "risk_charge_method",
    ]
    results = results[order].sort_values(
        ["row_type", "fee_segment", "category", "market_id", "fill_window_sec", "adverse_horizon_sec"]
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
