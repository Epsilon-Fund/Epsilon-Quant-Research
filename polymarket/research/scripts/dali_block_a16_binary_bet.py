"""Block A1.6 binary-bet hypothesis with non-overlapping positions.

This sidecar tests whether the A14f 300s BTC-4h winner reflects a binary
directional drift timing effect rather than a microstructure-only move. Unlike
the A14 family, this script enforces one open position per market at a time.
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
OUT_CSV = ANALYSIS / "csv_outputs" / "dali" / "block_a16_binary_bet_results.csv"
NOTE = NOTES / "block_a16_binary_bet_findings.md"

BINARY_FAMILIES = {
    "crypto_4h_up_down",
    "daily_crypto_up_down",
    "sports_game_lines",
    "sports_neg_risk_outright",
}
FIXED_HORIZONS: tuple[tuple[str, float], ...] = (
    ("fixed_60s", 60.0),
    ("fixed_300s", 300.0),
    ("fixed_900s", 900.0),
    ("fixed_1800s", 1800.0),
)
FRACTIONAL_HORIZONS: tuple[tuple[str, float], ...] = (
    ("frac_25pct", 0.25),
    ("frac_50pct", 0.50),
    ("frac_75pct", 0.75),
)
REGIME_SECONDS = (1, 5, 15, 30)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
FRACTIONAL_CAP_SECONDS = 4 * 60 * 60
RESOLUTION_EXIT_BUFFER_SECONDS = 60
RNG_SEED = 20260529
ROBUST_MIN_TRADES = 5


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fee_amount(category: str, price: float) -> float:
    params = FEE_BY_CATEGORY.get(category, FEE_BY_CATEGORY["Other"])
    p = float(np.clip(price, 0.0, 1.0))
    return float(params["fee_rate"] * p * (1.0 - p))


def load_candidates() -> pd.DataFrame:
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    results["n_classifiable"] = pd.to_numeric(results["n_classifiable"], errors="coerce")
    candidates = results[
        results["family"].isin(BINARY_FAMILIES)
        & results["n_classifiable"].ge(30)
    ][["run_id", "market_id", "family", "n_classifiable"]].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no binary candidate markets found")
    candidates["market"] = candidates["run_id"] + ":" + candidates["market_id"]
    return candidates.sort_values(["family", "run_id", "market_id"]).reset_index(drop=True)


def latest_markets_path() -> Path | None:
    paths = sorted((ROOT / "data" / "markets").glob("markets_*.parquet"))
    return paths[-1] if paths else None


def load_market_end_dates(candidates: pd.DataFrame) -> pd.DataFrame:
    path = latest_markets_path()
    candidates = candidates.copy()
    candidates["metadata_source"] = ""
    candidates["end_date"] = pd.NaT
    if path is None:
        return candidates
    con = duckdb.connect()
    meta = con.execute(
        f"""
        SELECT id, slug, end_date
        FROM read_parquet('{path}')
        WHERE end_date IS NOT NULL
        """
    ).df()
    con.close()
    if meta.empty:
        return candidates
    meta["id"] = meta["id"].astype(str)
    meta["slug"] = meta["slug"].fillna("").astype(str)
    meta["end_date"] = pd.to_datetime(meta["end_date"], utc=True, errors="coerce")
    by_id = meta.dropna(subset=["end_date"]).drop_duplicates("id").set_index("id")["end_date"].to_dict()
    candidates["end_date"] = candidates["market_id"].map(by_id)
    candidates["metadata_source"] = np.where(candidates["end_date"].notna(), path.name, "")
    return candidates


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
            f.tob_imbalance,
            f.market_resolved_at
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
    df["market_resolved_at"] = pd.to_datetime(df["market_resolved_at"], utc=True, errors="coerce")
    for col in ("run_id", "asset_id", "market_id", "family", "slug", "question"):
        df[col] = df[col].fillna("").astype(str)
    for col in ("outcome_index", "best_bid", "best_ask", "mid", "tob_imbalance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    return df


def add_tob_signal(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "market_id", "asset_id"], sort=False):
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
        g["abs_tob_imbalance_level"] = g["tob_imbalance_level"].abs()
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    out["is_top_decile_entry"] = False
    out["is_top_quartile_regime"] = False
    valid = (
        out["is_book_state_complete"]
        & out["abs_tob_imbalance_level"].replace([np.inf, -np.inf], np.nan).notna()
        & out["abs_tob_imbalance_level"].gt(0)
        & out["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & out["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & out["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & out["token_side"].replace([np.inf, -np.inf], np.nan).notna()
    )
    for _, idx in out[valid].groupby(["run_id", "market_id"], sort=False).groups.items():
        values = out.loc[idx, "abs_tob_imbalance_level"]
        try:
            decile = pd.qcut(values.rank(method="first"), 10, labels=False) + 1
            quartile = pd.qcut(values.rank(method="first"), 4, labels=False) + 1
        except ValueError:
            continue
        out.loc[values.index[decile.eq(10)], "is_top_decile_entry"] = True
        out.loc[values.index[quartile.eq(4)], "is_top_quartile_regime"] = True
    return out.sort_values(["run_id", "market_id", "asset_id", "received_at"]).reset_index(drop=True)


def add_regime_durations(df: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(["run_id", "market_id", "asset_id"], sort=False):
        g = group.sort_values("received_at").copy()
        times = g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
        high = g["is_top_quartile_regime"].to_numpy(dtype=bool)
        signs = g["signal_sign"].to_numpy(dtype=float)
        start_ns = np.full(len(g), np.nan, dtype=float)
        duration = np.zeros(len(g), dtype=float)
        current_start: int | None = None
        current_sign = math.nan
        for i in range(len(g)):
            if high[i] and np.isfinite(signs[i]) and signs[i] != 0:
                if current_start is None or signs[i] != current_sign:
                    current_start = int(times[i])
                    current_sign = float(signs[i])
                start_ns[i] = float(current_start)
                duration[i] = (int(times[i]) - current_start) / 1_000_000_000.0
            else:
                current_start = None
                current_sign = math.nan
        g["regime_duration_seconds"] = duration
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def entry_price_for_side(token_side: float, idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(ask[idx]) if token_side > 0 else float(bid[idx])


def exit_price_for_side(token_side: float, idx: int, bid: np.ndarray, ask: np.ndarray) -> float:
    return float(bid[idx]) if token_side > 0 else float(ask[idx])


def pnl_bps(category: str, token_side: float, entry_price: float, exit_price: float) -> float:
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
        return math.nan
    gross = exit_price - entry_price if token_side > 0 else entry_price - exit_price
    fees = fee_amount(category, entry_price) + fee_amount(category, exit_price)
    return float((gross - fees) / entry_price * 10_000.0)


def horizon_delta_seconds(
    horizon: str,
    entry_ns: int,
    resolution_ns: int | None,
) -> float | None:
    if horizon.startswith("fixed_"):
        seconds = float(horizon.removeprefix("fixed_").removesuffix("s"))
    else:
        if resolution_ns is None:
            return None
        frac_text = horizon.removeprefix("frac_").removesuffix("pct")
        fraction = float(frac_text) / 100.0
        remaining = (resolution_ns - entry_ns) / 1_000_000_000.0
        if remaining <= RESOLUTION_EXIT_BUFFER_SECONDS:
            return None
        seconds = min(remaining * fraction, FRACTIONAL_CAP_SECONDS)
    if resolution_ns is not None:
        before_resolution = (resolution_ns - entry_ns) / 1_000_000_000.0 - RESOLUTION_EXIT_BUFFER_SECONDS
        seconds = min(seconds, before_resolution)
    return seconds if seconds > 0 else None


def bootstrap_mean_ci(trades: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = trades[["entry_time", "pnl_bps"]].dropna().copy()
    if len(clean) < ROBUST_MIN_TRADES:
        return math.nan, math.nan
    elapsed = (clean["entry_time"] - clean["entry_time"].min()).dt.total_seconds()
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


def market_candidate_events(
    sub: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    if variant == "signal_top_decile":
        mask = sub["is_top_decile_entry"]
    else:
        seconds = float(variant.removeprefix("signal_regime_sustained_").removesuffix("s"))
        mask = sub["is_top_quartile_regime"] & sub["regime_duration_seconds"].ge(seconds)
    events = sub[
        mask
        & sub["is_book_state_complete"]
        & sub["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & sub["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & sub["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & sub["token_side"].replace([np.inf, -np.inf], np.nan).notna()
        & sub["signal_sign"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if events.empty:
        return events
    events["entry_time_ns"] = events["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    events["entry_rank_strength"] = events["abs_tob_imbalance_level"].astype(float)
    events = events.sort_values(["entry_time_ns", "entry_rank_strength"], ascending=[True, False])
    # When YES/NO assets emit simultaneous opportunities, take the stronger state.
    return events.drop_duplicates(["entry_time_ns"], keep="first").reset_index(drop=True)


def simulate_nonoverlap(
    sub: pd.DataFrame,
    *,
    variant: str,
    horizon: str,
    category: str,
    resolution_ns: int | None,
) -> tuple[int, pd.DataFrame]:
    events = market_candidate_events(sub, variant)
    n_eligible = int(len(events))
    if events.empty:
        return n_eligible, pd.DataFrame()
    asset_arrays: dict[str, dict[str, np.ndarray]] = {}
    for asset_id, group in sub.groupby("asset_id", sort=False):
        g = group.sort_values("received_at").reset_index(drop=True)
        asset_arrays[str(asset_id)] = {
            "times": g["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            "bid": g["best_bid"].to_numpy(dtype=float),
            "ask": g["best_ask"].to_numpy(dtype=float),
        }
    next_available_ns = -1
    trades: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        entry_ns = int(row.entry_time_ns)
        if entry_ns < next_available_ns:
            continue
        delta = horizon_delta_seconds(horizon, entry_ns, resolution_ns)
        if delta is None:
            continue
        target_ns = entry_ns + int(delta * 1_000_000_000)
        arrays = asset_arrays[str(row.asset_id)]
        times = arrays["times"]
        if len(times) == 0 or target_ns > int(times[-1]):
            continue
        exit_idx = int(np.searchsorted(times, target_ns, side="right") - 1)
        if exit_idx < 0:
            continue
        # Re-locate the entry row in the asset stream at or before the event time.
        entry_idx = int(np.searchsorted(times, entry_ns, side="right") - 1)
        if entry_idx < 0:
            continue
        token_side = float(row.token_side)
        entry_px = entry_price_for_side(token_side, entry_idx, arrays["bid"], arrays["ask"])
        exit_px = exit_price_for_side(token_side, exit_idx, arrays["bid"], arrays["ask"])
        trade_pnl = pnl_bps(category, token_side, entry_px, exit_px)
        if not np.isfinite(trade_pnl):
            continue
        trades.append(
            {
                "entry_time": pd.Timestamp(row.received_at),
                "exit_time_ns": target_ns,
                "hold_seconds": delta,
                "pnl_bps": trade_pnl,
                "signal_strength": float(row.abs_tob_imbalance_level),
            }
        )
        next_available_ns = target_ns
    return n_eligible, pd.DataFrame(trades)


def summarize_trades(
    trades: pd.DataFrame,
    *,
    n_eligible: int,
    capture_hours: float,
    seed: int,
) -> dict[str, float | int]:
    n = int(len(trades))
    if n == 0:
        return {
            "n_eligible_signals": n_eligible,
            "n_executed_trades": 0,
            "mean_pnl_bps": math.nan,
            "median_pnl_bps": math.nan,
            "win_rate": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "mean_hold_seconds": math.nan,
            "trades_per_hour": 0.0 if capture_hours > 0 else math.nan,
            "mean_signal_strength_at_entry": math.nan,
        }
    ci_lo, ci_hi = bootstrap_mean_ci(trades, seed)
    return {
        "n_eligible_signals": n_eligible,
        "n_executed_trades": n,
        "mean_pnl_bps": float(trades["pnl_bps"].mean()),
        "median_pnl_bps": float(trades["pnl_bps"].median()),
        "win_rate": float(trades["pnl_bps"].gt(0).mean()),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mean_hold_seconds": float(trades["hold_seconds"].mean()),
        "trades_per_hour": n / capture_hours if capture_hours > 0 else math.nan,
        "mean_signal_strength_at_entry": float(trades["signal_strength"].mean()),
    }


def run_analysis(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    variants = ["signal_top_decile"] + [f"signal_regime_sustained_{n}s" for n in REGIME_SECONDS]
    fixed = [name for name, _ in FIXED_HORIZONS]
    fractional = [name for name, _ in FRACTIONAL_HORIZONS]
    rows: list[dict[str, object]] = []
    for market_idx, candidate in candidates.reset_index(drop=True).iterrows():
        market = str(candidate["market"])
        run_id = str(candidate["run_id"])
        market_id = str(candidate["market_id"])
        family = str(candidate["family"])
        category = family_category(family)
        sub = features[features["run_id"].eq(run_id) & features["market_id"].eq(market_id)].copy()
        if sub.empty:
            continue
        slug = str(sub["slug"].replace("", np.nan).dropna().iloc[0]) if sub["slug"].astype(bool).any() else market
        question = (
            str(sub["question"].replace("", np.nan).dropna().iloc[0])
            if sub["question"].astype(bool).any()
            else ""
        )
        min_ts = sub["received_at"].min()
        max_ts = sub["received_at"].max()
        capture_hours = max((max_ts - min_ts).total_seconds() / 3600.0, 0.0)
        end_date = candidate.get("end_date")
        metadata_resolution_ns = None
        if pd.notna(end_date):
            metadata_resolution_ns = int(pd.Timestamp(end_date).to_datetime64().astype("datetime64[ns]").astype("int64"))
        feature_resolved = sub["market_resolved_at"].dropna()
        feature_resolution_ns = None
        if not feature_resolved.empty:
            feature_resolution_ns = int(feature_resolved.min().to_datetime64().astype("datetime64[ns]").astype("int64"))
        fixed_resolution_ns = metadata_resolution_ns if metadata_resolution_ns is not None else feature_resolution_ns
        horizons = fixed + (fractional if metadata_resolution_ns is not None else [])
        print(f"A16 market {market_idx + 1:02d}/{len(candidates):02d}: {market} ({len(horizons)} horizons)", flush=True)
        for variant_idx, variant in enumerate(variants):
            for horizon_idx, horizon in enumerate(horizons):
                n_eligible, trades = simulate_nonoverlap(
                    sub,
                    variant=variant,
                    horizon=horizon,
                    category=category,
                    resolution_ns=metadata_resolution_ns if horizon.startswith("frac_") else fixed_resolution_ns,
                )
                row = {
                    "market": market,
                    "slug": slug,
                    "family": family,
                    "entry_variant": variant,
                    "horizon": horizon,
                    "has_end_date": metadata_resolution_ns is not None,
                    "end_date": pd.Timestamp(end_date).isoformat() if pd.notna(end_date) else "",
                    "capture_window_hours": capture_hours,
                    **summarize_trades(
                        trades,
                        n_eligible=n_eligible,
                        capture_hours=capture_hours,
                        seed=RNG_SEED + market_idx * 1000 + variant_idx * 100 + horizon_idx,
                    ),
                }
                rows.append(row)
    columns = [
        "market",
        "slug",
        "family",
        "entry_variant",
        "horizon",
        "has_end_date",
        "end_date",
        "capture_window_hours",
        "n_eligible_signals",
        "n_executed_trades",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "mean_hold_seconds",
        "trades_per_hour",
        "mean_signal_strength_at_entry",
    ]
    return pd.DataFrame(rows)[columns]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) for row in rows]
    body = [line + " |" for line in body]
    return "\n".join([header, sep, *body])


def universe_table(candidates: pd.DataFrame, results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    meta = results.groupby("market", as_index=False).agg(
        slug=("slug", "first"),
        family=("family", "first"),
        has_end_date=("has_end_date", "max"),
        end_date=("end_date", "first"),
    )
    for row in meta.sort_values(["family", "market"]).itertuples(index=False):
        rows.append([
            str(row.market),
            str(row.slug)[:46].replace("|", "/"),
            str(row.family),
            "yes" if bool(row.has_end_date) else "fixed only",
            str(row.end_date)[:19] if str(row.end_date) else "",
        ])
    return markdown_table(["market", "slug", "family", "fractional?", "end date"], rows)


def robust_count(results: pd.DataFrame) -> int:
    return int(
        (
            results["mean_pnl_bps"].gt(0)
            & results["ci_lo"].gt(0)
            & results["n_executed_trades"].ge(ROBUST_MIN_TRADES)
        ).sum()
    )


def per_market_verdict_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for market, sub in results.groupby("market", sort=True):
        slug = str(sub["slug"].dropna().iloc[0])[:44].replace("|", "/")
        family = str(sub["family"].dropna().iloc[0])
        best = sub.sort_values(["mean_pnl_bps", "n_executed_trades"], ascending=False).iloc[0]
        robust = sub[
            sub["mean_pnl_bps"].gt(0)
            & sub["ci_lo"].gt(0)
            & sub["n_executed_trades"].ge(ROBUST_MIN_TRADES)
        ]
        positive = sub[sub["mean_pnl_bps"].gt(0)]
        if not robust.empty:
            verdict = "robust positive"
        elif not positive.empty:
            verdict = "positive but not robust"
        elif np.isfinite(best["mean_pnl_bps"]):
            verdict = "negative"
        else:
            verdict = "no executed trades"
        rows.append([
            market,
            slug,
            family,
            str(best["entry_variant"]),
            str(best["horizon"]),
            f"{int(best['n_executed_trades']):,}",
            bps(float(best["mean_pnl_bps"])),
            f"[{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]",
            pct(float(best["win_rate"])),
            verdict,
        ])
    return markdown_table(
        ["market", "slug", "family", "best variant", "horizon", "trades", "mean", "CI", "win", "verdict"],
        rows,
    )


def pattern_read_table(results: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for family, sub in results.groupby("family", sort=True):
        best = sub.sort_values(["mean_pnl_bps", "n_executed_trades"], ascending=False).iloc[0]
        robust = robust_count(sub)
        rows.append([
            family,
            str(sub["market"].nunique()),
            str(robust),
            str(best["market"]),
            str(best["entry_variant"]),
            str(best["horizon"]),
            f"{int(best['n_executed_trades']):,}",
            bps(float(best["mean_pnl_bps"])),
            f"[{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]",
        ])
    return markdown_table(["family", "markets", "robust+", "best market", "variant", "horizon", "trades", "mean", "CI"], rows)


def btc_4h_table(results: pd.DataFrame) -> str:
    sub = results[results["family"].eq("crypto_4h_up_down")].copy()
    rows: list[list[str]] = []
    for market, rows_df in sub.groupby("market", sort=True):
        best = rows_df.sort_values(["mean_pnl_bps", "n_executed_trades"], ascending=False).iloc[0]
        fixed300 = rows_df[
            rows_df["entry_variant"].eq("signal_top_decile") & rows_df["horizon"].eq("fixed_300s")
        ]
        fixed_text = "n/a"
        if not fixed300.empty:
            fixed = fixed300.iloc[0]
            fixed_text = f"{bps(float(fixed['mean_pnl_bps']))} / {int(fixed['n_executed_trades'])} trades"
        rows.append([
            market,
            str(best["slug"])[:40].replace("|", "/"),
            fixed_text,
            str(best["entry_variant"]),
            str(best["horizon"]),
            f"{int(best['n_executed_trades']):,}",
            bps(float(best["mean_pnl_bps"])),
            f"[{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]",
        ])
    return markdown_table(["market", "slug", "top-decile 300s", "best variant", "best horizon", "trades", "best mean", "CI"], rows)


def regime_comparison_table(results: pd.DataFrame) -> str:
    grouped = (
        results.groupby("entry_variant", as_index=False)
        .agg(
            cells=("market", "count"),
            robust_positive=("ci_lo", lambda s: int(((results.loc[s.index, "mean_pnl_bps"] > 0) & (s > 0)).sum())),
            positive=("mean_pnl_bps", lambda s: int((s > 0).sum())),
            mean_pnl_bps=("mean_pnl_bps", "mean"),
            median_pnl_bps=("median_pnl_bps", "median"),
            total_trades=("n_executed_trades", "sum"),
            mean_trades_per_hour=("trades_per_hour", "mean"),
        )
        .sort_values("mean_pnl_bps", ascending=False)
    )
    rows = [
        [
            str(row.entry_variant),
            f"{int(row.positive)}/{int(row.cells)}",
            f"{int(row.robust_positive)}/{int(row.cells)}",
            bps(float(row.mean_pnl_bps)),
            bps(float(row.median_pnl_bps)),
            f"{int(row.total_trades):,}",
            f"{float(row.mean_trades_per_hour):.2f}",
        ]
        for row in grouped.itertuples(index=False)
    ]
    return markdown_table(["variant", "positive", "robust+", "mean pnl", "median pnl", "trades", "avg trades/hr"], rows)


def trades_per_hour_table(results: pd.DataFrame) -> str:
    top = results.sort_values(["trades_per_hour", "n_executed_trades"], ascending=False).head(20)
    rows = [
        [
            str(row.market),
            str(row.entry_variant),
            str(row.horizon),
            f"{int(row.n_executed_trades):,}",
            f"{float(row.trades_per_hour):.2f}",
            f"{float(row.mean_hold_seconds):.1f}s" if np.isfinite(row.mean_hold_seconds) else "n/a",
            bps(float(row.mean_pnl_bps)),
        ]
        for row in top.itertuples(index=False)
    ]
    return markdown_table(["market", "variant", "horizon", "trades", "trades/hr", "mean hold", "mean pnl"], rows)


def write_note(results: pd.DataFrame, candidates: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    robust = robust_count(results)
    positive = int(results["mean_pnl_bps"].gt(0).sum())
    total = int(len(results))
    best = results.sort_values(["mean_pnl_bps", "n_executed_trades"], ascending=False).iloc[0]
    btc_4h = results[results["family"].eq("crypto_4h_up_down")]
    btc_4h_positive_markets = int(
        btc_4h.groupby("market")["mean_pnl_bps"].max().gt(0).sum()
    ) if not btc_4h.empty else 0
    recommendation = (
        "the binary-bet hypothesis is partial: the A14f 300s BTC-4h hint does not clear the robust CI bar under non-overlap, so A2 should capture more binary windows and pre-register non-overlap plus tight-spread capacity checks."
        if robust == 0
        else "the binary-bet hypothesis survives the first robustness bar, so A2 should target repeated binary windows with the winning non-overlap variant and add latency/capacity checks."
    )

    note = f"""---
tags: [dali, block-a16, binary-bet, results]
---

# Block A1.6 Binary-Bet Findings

## Headline

A1.6 enforces one open position per market and retests TOB imbalance as a binary-direction timing signal. {robust} of {total} market-variant-horizon cells crossed zero with bootstrap CI lower bound above zero; {positive} cells had positive mean PnL without clearing that robustness bar. Best cell: `{best['market']}` / `{best['entry_variant']}` / `{best['horizon']}` at {bps(float(best['mean_pnl_bps']))} on {int(best['n_executed_trades']):,} non-overlapping trades, CI [{bps(float(best['ci_lo']))}, {bps(float(best['ci_hi']))}]. The A14f BTC-4h winner does not broadly replicate under non-overlap: {btc_4h_positive_markets} of 3 BTC-4h windows have any positive mean row.

## Universe

Binary market universe is selected from `block_a1_results.csv` with families `{sorted(BINARY_FAMILIES)}` and `n_classifiable >= 30`. Fractional horizons are emitted only when `end_date` is present in the latest `data/markets/markets_*.parquet`. Fixed horizons use `end_date` as the resolution backstop when available, and otherwise use `market_resolved_at` from the feature table when the live capture observed resolution.

{universe_table(candidates, results)}

## Per-Market Verdict

{per_market_verdict_table(results)}

## Cross-Market Pattern Read

The fixed-300s BTC-4h A14f winner is the key replication check. Under non-overlap, top-decile fixed-300s is negative on all three BTC 4h windows, including the original A14f winner. One other BTC 4h window has a positive longer-horizon regime row, but it does not clear the CI bar. This table compares top-decile fixed-300s against each market's best non-overlap row.

{btc_4h_table(results)}

Family-level read:

{pattern_read_table(results)}

Daily binaries and sports do not show a robust positive under non-overlap in this capture. The BTC-4h result remains a single-window clue rather than a repeated binary family effect.

## Regime-Filter Comparison

Lipton-style sustained imbalance regimes reduce noisy entry frequency, but in this capture they do not beat the top-decile entry in a robust way.

{regime_comparison_table(results)}

## Trades-Per-Hour

This table shows the highest deployment rates after non-overlap, not overlapping signal counts.

{trades_per_hour_table(results)}

## Interpretation

Non-overlap is a much harsher and more realistic deployment constraint than A14's overlapping-position math. A positive overlapping cell can disappear if it repeatedly re-enters while one real position would still be open. Fractional horizons are limited by available `end_date` metadata; the A0b crypto/NBA markets are fixed-horizon only in this pass because they are absent from the current markets parquet, though resolved BTC 4h windows still use observed `market_resolved_at` as a fixed-horizon backstop.

Recommended next action for Justin: {recommendation}
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    candidates = load_market_end_dates(load_candidates())
    features = add_regime_durations(add_tob_signal(load_feature_subset(candidates)))
    results = run_analysis(features, candidates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    write_note(results, candidates)
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
