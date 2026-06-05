"""Block A1.7 LightGBM selector on engineered TOB/OFI features.

This sidecar tests whether a pooled Tier 2 LightGBM model can select a
high-confidence subset of A1 TOB/OFI signal events that survives executable
touch round-trip cost under a one-position-at-a-time non-overlap rule.

No Optuna or random shuffling is used. Hyperparameters are intentionally fixed:
max_depth=6, num_leaves=31, learning_rate=0.05, n_estimators=200, and
early_stopping_rounds=20 on the pooled validation slice.
"""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from dali_block_a1_analyze import FEE_BY_CATEGORY, family_category


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data" / "analysis"
NOTES = ROOT / "notes"
FEATURES = ANALYSIS / "block_a1_features.parquet"
A1_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a1_results.csv"
OUT_FEATURES = ANALYSIS / "block_a17_lightgbm_features.parquet"
OUT_RESULTS = ANALYSIS / "csv_outputs" / "dali" / "block_a17_lightgbm_results.csv"
OUT_IMPORTANCE = ANALYSIS / "csv_outputs" / "dali" / "block_a17_feature_importance.csv"
NOTE = NOTES / "block_a17_lightgbm_findings.md"

THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_CHUNK_SECONDS = 300
RNG_SEED = 20260529
HOLD_SECONDS = 5
MIN_TEST_EVENTS = 500
EARLY_STOPPING_ROUNDS = 20
LGBM_PARAMS = {
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "objective": "binary",
    "random_state": RNG_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
MODEL_FEATURES = [
    "ofi_5s",
    "tob_imbalance_level_instant",
    "tob_imbalance_level_5s_mean",
    "spread_bps_instant",
    "spread_bps_5s_mean",
    "depth_at_touch",
    "depth_relative",
    "market_id",
]


def ensure_lightgbm():
    """Import LightGBM, re-execing once on macOS if libomp is only bundled by sklearn."""
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

        return lgb, LGBMClassifier
    except OSError as exc:
        if "libomp" not in str(exc) or os.environ.get("A17_LGBM_REEXEC") == "1":
            raise

        lib_dirs: list[Path] = []
        try:
            import importlib.util

            spec = importlib.util.find_spec("sklearn")
            if spec and spec.origin:
                lib_dirs.append(Path(spec.origin).resolve().parent / ".dylibs")
        except Exception:
            pass
        lib_dirs.extend(
            [
                Path("/opt/homebrew/opt/libomp/lib"),
                Path("/usr/local/opt/libomp/lib"),
                Path("/Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib"),
            ]
        )
        for lib_dir in lib_dirs:
            if (lib_dir / "libomp.dylib").exists():
                existing = os.environ.get("DYLD_LIBRARY_PATH", "")
                os.environ["DYLD_LIBRARY_PATH"] = (
                    str(lib_dir) if not existing else f"{lib_dir}{os.pathsep}{existing}"
                )
                os.environ["A17_LGBM_REEXEC"] = "1"
                os.execv(sys.executable, [sys.executable, *sys.argv])
        raise


def bps(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} bps"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def safe_text(value: object, max_len: int = 44) -> str:
    text = str(value if value is not None else "").replace("|", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def load_candidates() -> pd.DataFrame:
    if not A1_RESULTS.exists():
        raise SystemExit(f"missing A1 results: {A1_RESULTS}")
    results = pd.read_csv(A1_RESULTS, dtype={"run_id": str, "market_id": str})
    candidates = results[results["sample_size_label"].eq("primary_read")].copy()
    candidates = candidates[
        ["run_id", "market_id", "family", "n_classifiable", "mean_depth_at_touch"]
    ].drop_duplicates(["run_id", "market_id"])
    if candidates.empty:
        raise SystemExit("no primary_read markets found in block_a1_results.csv")
    candidates["market"] = candidates["run_id"] + ":" + candidates["market_id"]
    return candidates.sort_values(["run_id", "market_id"]).reset_index(drop=True)


def load_feature_subset(candidates: pd.DataFrame) -> pd.DataFrame:
    if not FEATURES.exists():
        raise SystemExit(f"missing A1 features: {FEATURES}")
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
            f.best_bid_size,
            f.best_ask_size,
            f.spread,
            f.mid,
            f.tob_imbalance,
            f.ofi_combined_event
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
    numeric = [
        "outcome_index",
        "best_bid",
        "best_ask",
        "best_bid_size",
        "best_ask_size",
        "spread",
        "mid",
        "tob_imbalance",
        "ofi_combined_event",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    return df


def future_values(group: pd.DataFrame, cols: Iterable[str], horizon_sec: int) -> dict[str, np.ndarray]:
    times = group["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    out = {col: np.full(len(group), np.nan, dtype=float) for col in cols}
    if len(group) == 0:
        return out
    target = times + horizon_sec * 1_000_000_000
    idx = np.searchsorted(times, target, side="right") - 1
    valid = (target <= times[-1]) & (idx >= 0)
    for col in cols:
        values = group[col].to_numpy(dtype=float)
        out[col][valid] = values[idx[valid]]
    return out


def engineer_features(raw: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    depth_map = candidates.set_index(["run_id", "market_id"])["mean_depth_at_touch"].to_dict()
    family_map = candidates.set_index(["run_id", "market_id"])["family"].to_dict()
    pieces: list[pd.DataFrame] = []
    total_groups = raw[["run_id", "market_id", "asset_id"]].drop_duplicates().shape[0]

    for group_idx, ((run_id, market_id, asset_id), group) in enumerate(
        raw.groupby(["run_id", "market_id", "asset_id"], sort=False),
        start=1,
    ):
        if group_idx % 10 == 0:
            print(f"engineer asset group {group_idx:,}/{total_groups:,}", flush=True)
        g = group.sort_values("received_at").copy()
        g["market"] = g["run_id"] + ":" + g["market_id"]
        g["direction_factor"] = np.where(g["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
        g["tob_imbalance"] = g["tob_imbalance"].ffill()
        g["ofi_combined_event"] = g["ofi_combined_event"].fillna(0.0)
        g["depth_at_touch"] = g[["best_bid_size", "best_ask_size"]].sum(axis=1, min_count=2)
        g["market_mean_depth_at_touch"] = float(depth_map.get((run_id, market_id), math.nan))
        if not np.isfinite(g["market_mean_depth_at_touch"].iloc[0]) or g[
            "market_mean_depth_at_touch"
        ].iloc[0] <= 0:
            g["market_mean_depth_at_touch"] = float(g["depth_at_touch"].mean())
        g["depth_relative"] = g["depth_at_touch"] / g["market_mean_depth_at_touch"]
        g["spread_bps_instant"] = np.where(
            g["mid"].gt(0) & g["spread"].notna(),
            g["spread"] / g["mid"] * 10_000.0,
            np.nan,
        )
        g["tob_imbalance_level_instant"] = g["direction_factor"] * g["tob_imbalance"]
        g["directional_mid"] = np.where(g["direction_factor"].gt(0), g["mid"], 1.0 - g["mid"])

        g = g.set_index("received_at", drop=False)
        g["ofi_5s"] = (
            g["direction_factor"].to_numpy(dtype=float)
            * g["ofi_combined_event"].rolling(f"{HOLD_SECONDS}s").sum().to_numpy(dtype=float)
        )
        g["tob_imbalance_level_5s_mean"] = (
            g["tob_imbalance_level_instant"].rolling(f"{HOLD_SECONDS}s").mean().to_numpy(dtype=float)
        )
        g["spread_bps_5s_mean"] = (
            g["spread_bps_instant"].rolling(f"{HOLD_SECONDS}s").mean().to_numpy(dtype=float)
        )
        g = g.reset_index(drop=True)

        future = future_values(g, ["mid", "best_bid", "best_ask"], HOLD_SECONDS)
        g["future_mid_5s"] = future["mid"]
        g["future_bid_5s"] = future["best_bid"]
        g["future_ask_5s"] = future["best_ask"]
        future_directional_mid = np.where(
            g["direction_factor"].gt(0),
            g["future_mid_5s"],
            1.0 - g["future_mid_5s"],
        )
        g["future_5s_directional_mid_return"] = np.where(
            g["directional_mid"].gt(0) & np.isfinite(future_directional_mid),
            (future_directional_mid - g["directional_mid"]) / g["directional_mid"] * 10_000.0,
            np.nan,
        )
        g["direction_correct"] = (
            np.sign(g["ofi_5s"]) == np.sign(g["future_5s_directional_mid_return"])
        ).astype(float)
        zero_or_missing = (
            ~np.isfinite(g["ofi_5s"])
            | ~np.isfinite(g["future_5s_directional_mid_return"])
            | np.sign(g["ofi_5s"]).eq(0)
            | np.sign(g["future_5s_directional_mid_return"]).eq(0)
        )
        g.loc[zero_or_missing, "direction_correct"] = np.nan
        g["family"] = str(family_map.get((run_id, market_id), g["family"].iloc[0] or ""))
        g["category"] = family_category(g["family"].iloc[0])
        pieces.append(g)

    df = pd.concat(pieces, ignore_index=True)
    feature_cols = [
        "is_book_state_complete",
        "best_bid",
        "best_ask",
        "mid",
        "future_bid_5s",
        "future_ask_5s",
        *MODEL_FEATURES[:-1],
        "direction_correct",
    ]
    model_ok = df["is_book_state_complete"].fillna(False)
    for col in feature_cols:
        if col == "is_book_state_complete":
            continue
        model_ok &= np.isfinite(df[col])
    df = df[model_ok].copy()
    df = df[df["depth_at_touch"].gt(0) & df["depth_relative"].gt(0)].copy()
    df["split"] = ""
    for market, idx in df.groupby("market", sort=False).groups.items():
        ordered = df.loc[idx].sort_values(["received_at", "asset_id"]).index.to_numpy()
        n = len(ordered)
        train_end = int(math.floor(n * 2.0 / 3.0))
        valid_end = int(math.floor(n * 5.0 / 6.0))
        df.loc[ordered[:train_end], "split"] = "train"
        df.loc[ordered[train_end:valid_end], "split"] = "validation"
        df.loc[ordered[valid_end:], "split"] = "test"
    df["event_time_ns"] = df["received_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    df["exit_time_ns"] = df["event_time_ns"] + HOLD_SECONDS * 1_000_000_000
    return df.sort_values(["run_id", "market_id", "received_at", "asset_id"]).reset_index(drop=True)


def fee_amount(category: pd.Series, price: np.ndarray) -> np.ndarray:
    fee_rate = category.map(lambda c: FEE_BY_CATEGORY.get(c, FEE_BY_CATEGORY["Other"])["fee_rate"])
    p = np.clip(price.astype(float), 0.0, 1.0)
    return fee_rate.to_numpy(dtype=float) * p * (1.0 - p)


def compute_pnl(events: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    out = events.copy()
    signal = out[signal_col].to_numpy(dtype=float)
    direction_factor = out["direction_factor"].to_numpy(dtype=float)
    token_side = np.sign(signal) * direction_factor
    entry_price = np.where(token_side > 0, out["best_ask"], out["best_bid"]).astype(float)
    exit_price = np.where(token_side > 0, out["future_bid_5s"], out["future_ask_5s"]).astype(float)
    valid = (
        np.isfinite(token_side)
        & (token_side != 0)
        & np.isfinite(entry_price)
        & np.isfinite(exit_price)
        & (entry_price > 0)
        & (exit_price >= 0)
    )
    gross = np.where(token_side > 0, exit_price - entry_price, entry_price - exit_price)
    pnl = np.full(len(out), np.nan, dtype=float)
    if valid.any():
        fees = fee_amount(out.loc[valid, "category"], entry_price[valid]) + fee_amount(
            out.loc[valid, "category"],
            exit_price[valid],
        )
        pnl[valid] = (gross[valid] - fees) / entry_price[valid] * 10_000.0
    out["token_side"] = token_side
    out["entry_price"] = entry_price
    out["exit_price"] = exit_price
    out["pnl_bps"] = pnl
    out["is_fillable"] = valid
    return out


def apply_non_overlap(events: pd.DataFrame, score_col: str, strength_col: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    clean = events[
        events["pnl_bps"].replace([np.inf, -np.inf], np.nan).notna()
        & events["event_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
        & events["exit_time_ns"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if clean.empty:
        return clean
    clean = clean.sort_values(
        ["event_time_ns", score_col, strength_col],
        ascending=[True, False, False],
    )
    keep: list[int] = []
    open_until = -1
    for pos, row in enumerate(clean.itertuples(index=False)):
        entry_ns = int(row.event_time_ns)
        exit_ns = int(row.exit_time_ns)
        if exit_ns <= entry_ns:
            continue
        if entry_ns <= open_until:
            continue
        keep.append(pos)
        open_until = exit_ns
    if not keep:
        return clean.iloc[0:0].copy()
    out = clean.iloc[keep].copy()
    out["executed_trade_rank"] = np.arange(1, len(out) + 1)
    return out


def block_bootstrap_mean_ci(trades: pd.DataFrame, seed: int) -> tuple[float, float]:
    clean = trades[["received_at", "pnl_bps"]].dropna().copy()
    clean = clean[np.isfinite(clean["pnl_bps"])]
    if len(clean) < 5:
        return math.nan, math.nan
    elapsed = (clean["received_at"] - clean["received_at"].min()).dt.total_seconds()
    block_id = (elapsed // BOOTSTRAP_CHUNK_SECONDS).astype(int).to_numpy()
    blocks = [np.flatnonzero(block_id == bid) for bid in np.unique(block_id)]
    if len(blocks) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = clean["pnl_bps"].to_numpy(dtype=float)
    stats: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        stats.append(float(np.nanmean(values[idx])))
    if len(stats) < 20:
        return math.nan, math.nan
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_trades(trades: pd.DataFrame, seed: int) -> dict[str, float | int]:
    n = int(len(trades))
    ci_lo, ci_hi = block_bootstrap_mean_ci(trades, seed)
    return {
        "n_trades_non_overlap": n,
        "mean_pnl_bps": float(trades["pnl_bps"].mean()) if n else math.nan,
        "median_pnl_bps": float(trades["pnl_bps"].median()) if n else math.nan,
        "win_rate": float(trades["pnl_bps"].gt(0).mean()) if n else math.nan,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def top_decile_events(events: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    values = events[signal_col].abs().replace([np.inf, -np.inf], np.nan)
    valid = events[values.notna() & values.gt(0)].copy()
    if valid.empty:
        return valid
    threshold = float(valid[signal_col].abs().quantile(0.90))
    return valid[valid[signal_col].abs().ge(threshold)].copy()


def fit_model(features: pd.DataFrame):
    lgb, LGBMClassifier = ensure_lightgbm()
    frame = features.copy()
    frame["market_id"] = frame["market_id"].astype("category")
    for col in MODEL_FEATURES:
        if col != "market_id":
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
    train = frame[frame["split"].eq("train")].copy()
    valid = frame[frame["split"].eq("validation")].copy()
    if train.empty or valid.empty:
        raise SystemExit("train/validation split is empty")
    y_train = train["direction_correct"].astype(int)
    y_valid = valid["direction_correct"].astype(int)
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        train[MODEL_FEATURES],
        y_train,
        eval_set=[(valid[MODEL_FEATURES], y_valid)],
        eval_metric="binary_logloss",
        categorical_feature=["market_id"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    pred = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    features = features.copy()
    features["model_prob"] = pred
    return model, features


def feature_importance(model) -> pd.DataFrame:
    booster = model.booster_
    out = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split_count": booster.feature_importance(importance_type="split"),
        }
    )
    return out.sort_values(["gain", "split_count"], ascending=False).reset_index(drop=True)


def calibration_table(features: pd.DataFrame) -> pd.DataFrame:
    test = features[features["split"].eq("test") & features["model_prob"].notna()].copy()
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
    labels = ["<0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", ">=0.80"]
    test["prob_bin"] = pd.cut(
        test["model_prob"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows: list[dict[str, object]] = []
    for label, sub in test.groupby("prob_bin", observed=False):
        n = int(len(sub))
        rows.append(
            {
                "prob_bin": str(label),
                "n": n,
                "mean_pred_prob": float(sub["model_prob"].mean()) if n else math.nan,
                "actual_hit_rate": float(sub["direction_correct"].mean()) if n else math.nan,
                "mean_abs_ofi_5s": float(sub["ofi_5s"].abs().mean()) if n else math.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_thresholds(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_meta = candidates.set_index("market").to_dict("index")
    feature_groups = {market: group.copy() for market, group in features.groupby("market", sort=False)}
    empty_features = features.iloc[0:0].copy()
    for market_idx, market in enumerate(candidates["market"].astype(str)):
        market_events = feature_groups.get(market, empty_features)
        test = market_events[market_events["split"].eq("test")].copy()
        slug = (
            str(test["slug"].replace("", np.nan).dropna().iloc[0])
            if not test.empty and test["slug"].astype(bool).any()
            else str(market_events["slug"].replace("", np.nan).dropna().iloc[0])
            if not market_events.empty and market_events["slug"].astype(bool).any()
            else str(market)
        )
        insufficient = len(test) < MIN_TEST_EVENTS
        if insufficient:
            tob_mean = math.nan
            ofi_mean = math.nan
        else:
            tob_base = compute_pnl(top_decile_events(test, "tob_imbalance_level_instant"), "tob_imbalance_level_instant")
            tob_trades = apply_non_overlap(
                tob_base.assign(rule_score=tob_base["tob_imbalance_level_instant"].abs()),
                "rule_score",
                "rule_score",
            )
            ofi_base = compute_pnl(top_decile_events(test, "ofi_5s"), "ofi_5s")
            ofi_trades = apply_non_overlap(
                ofi_base.assign(rule_score=ofi_base["ofi_5s"].abs()),
                "rule_score",
                "rule_score",
            )
            tob_mean = float(tob_trades["pnl_bps"].mean()) if len(tob_trades) else math.nan
            ofi_mean = float(ofi_trades["pnl_bps"].mean()) if len(ofi_trades) else math.nan

        for threshold in THRESHOLDS:
            eligible = test[test["model_prob"].gt(threshold)].copy()
            status = "insufficient_test_data" if insufficient else "ok"
            if insufficient:
                summary = {
                    "n_trades_non_overlap": 0,
                    "mean_pnl_bps": math.nan,
                    "median_pnl_bps": math.nan,
                    "win_rate": math.nan,
                    "ci_lo": math.nan,
                    "ci_hi": math.nan,
                }
            else:
                ml_events = compute_pnl(eligible, "ofi_5s")
                ml_events["model_strength"] = ml_events["ofi_5s"].abs()
                trades = apply_non_overlap(ml_events, "model_prob", "model_strength")
                summary = summarize_trades(
                    trades,
                    RNG_SEED + market_idx * 1000 + int(round(threshold * 100)),
                )
            mean_pnl = float(summary["mean_pnl_bps"])
            rows.append(
                {
                    "market": market,
                    "slug": slug,
                    "threshold": threshold,
                    "n_eligible_test_events": int(len(eligible)),
                    "n_trades_non_overlap": int(summary["n_trades_non_overlap"]),
                    "mean_pnl_bps": mean_pnl,
                    "median_pnl_bps": summary["median_pnl_bps"],
                    "win_rate": summary["win_rate"],
                    "ci_lo": summary["ci_lo"],
                    "ci_hi": summary["ci_hi"],
                    "tob_baseline_mean_pnl_bps": tob_mean,
                    "ofi_baseline_mean_pnl_bps": ofi_mean,
                    "delta_vs_tob_baseline": mean_pnl - tob_mean
                    if np.isfinite(mean_pnl) and np.isfinite(tob_mean)
                    else math.nan,
                    "delta_vs_ofi_baseline": mean_pnl - ofi_mean
                    if np.isfinite(mean_pnl) and np.isfinite(ofi_mean)
                    else math.nan,
                    "status": status,
                    "test_split_events": int(len(test)),
                    "family": str(candidate_meta.get(market, {}).get("family", "")),
                }
            )
    columns = [
        "market",
        "slug",
        "threshold",
        "n_eligible_test_events",
        "n_trades_non_overlap",
        "mean_pnl_bps",
        "median_pnl_bps",
        "win_rate",
        "ci_lo",
        "ci_hi",
        "tob_baseline_mean_pnl_bps",
        "ofi_baseline_mean_pnl_bps",
        "delta_vs_tob_baseline",
        "delta_vs_ofi_baseline",
        "status",
        "test_split_events",
        "family",
    ]
    return pd.DataFrame(rows)[columns]


def best_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, sub in results.groupby("market", sort=True):
        ok = sub[sub["status"].eq("ok")].copy()
        if ok.empty or ok["mean_pnl_bps"].notna().sum() == 0:
            rows.append(sub.iloc[0])
            continue
        rows.append(ok.sort_values(["mean_pnl_bps", "n_trades_non_overlap"], ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def verdict_for_row(row: pd.Series) -> str:
    if row["status"] != "ok":
        return "insufficient_test_data"
    mean = float(row["mean_pnl_bps"])
    ci_lo = float(row["ci_lo"])
    tob_delta = float(row["delta_vs_tob_baseline"])
    ofi_delta = float(row["delta_vs_ofi_baseline"])
    if (
        np.isfinite(mean)
        and np.isfinite(ci_lo)
        and np.isfinite(tob_delta)
        and np.isfinite(ofi_delta)
        and mean > 0
        and ci_lo > 0
        and tob_delta > 0
        and ofi_delta > 0
    ):
        return "beats both baselines robustly"
    if np.isfinite(mean) and mean > 0 and np.isfinite(tob_delta) and np.isfinite(ofi_delta) and tob_delta > 0 and ofi_delta > 0:
        return "beats both on mean only"
    if np.isfinite(mean) and mean > 0:
        return "positive mean, not better than both"
    if int(row["n_trades_non_overlap"]) == 0:
        return "no threshold trades"
    return "no executable edge"


def write_note(results: pd.DataFrame, importance: pd.DataFrame, calibration: pd.DataFrame) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    best = best_rows(results)
    best["verdict"] = best.apply(verdict_for_row, axis=1)
    robust_mask = (
        results["status"].eq("ok")
        & results["mean_pnl_bps"].gt(0)
        & results["ci_lo"].gt(0)
        & results["delta_vs_tob_baseline"].gt(0)
        & results["delta_vs_ofi_baseline"].gt(0)
    )
    robust = results[robust_mask].copy()
    mean_only = results[
        results["status"].eq("ok")
        & results["mean_pnl_bps"].gt(0)
        & results["delta_vs_tob_baseline"].gt(0)
        & results["delta_vs_ofi_baseline"].gt(0)
    ].copy()
    headline = (
        f"Yes: {len(robust):,} market-threshold cells beat both rule-based baselines with CI lower bound above zero."
        if not robust.empty
        else "No: no market-threshold cell beat both rule-based baselines with CI lower bound above zero."
    )

    verdict_rows: list[list[str]] = []
    for row in best.itertuples(index=False):
        verdict_rows.append(
            [
                safe_text(row.market, 18),
                safe_text(row.slug, 38),
                "n/a" if not np.isfinite(row.threshold) else f"{float(row.threshold):.2f}",
                f"{int(row.test_split_events):,}",
                f"{int(row.n_eligible_test_events):,}",
                f"{int(row.n_trades_non_overlap):,}",
                bps(float(row.mean_pnl_bps)),
                f"[{bps(float(row.ci_lo))}, {bps(float(row.ci_hi))}]",
                bps(float(row.tob_baseline_mean_pnl_bps)),
                bps(float(row.ofi_baseline_mean_pnl_bps)),
                bps(float(row.delta_vs_tob_baseline)),
                bps(float(row.delta_vs_ofi_baseline)),
                str(row.verdict),
            ]
        )

    importance_rows: list[list[str]] = []
    for row in importance.itertuples(index=False):
        importance_rows.append(
            [
                str(row.feature),
                f"{float(row.gain):.1f}",
                f"{int(row.split_count):,}",
            ]
        )

    calibration_rows: list[list[str]] = []
    for row in calibration.itertuples(index=False):
        calibration_rows.append(
            [
                str(row.prob_bin),
                f"{int(row.n):,}",
                pct(float(row.mean_pred_prob)),
                pct(float(row.actual_hit_rate)),
                f"{float(row.mean_abs_ofi_5s):.2f}" if np.isfinite(row.mean_abs_ofi_5s) else "n/a",
            ]
        )

    high_bins = calibration[calibration["prob_bin"].isin(["0.70-0.75", "0.75-0.80", ">=0.80"])]
    high_calibrated = bool((high_bins["n"].fillna(0) >= 100).any() and high_bins["actual_hit_rate"].max() > 0.55)
    if not robust.empty:
        recommendation = "expand to medium-scope A1.7"
    elif not mean_only.empty and high_calibrated:
        recommendation = "pivot to maker-side ML"
    else:
        recommendation = "no Tier 2 edge found"

    note = f"""---
tags: [dali, block-a17, lightgbm, results]
---

# Block A1.7 LightGBM Findings

## Headline

{headline}

A1.7 used a single pooled LightGBM classifier across all `primary_read` markets, with `market_id` as a categorical feature and walk-forward per-market splits of 2/3 train, 1/6 validation, and 1/6 test. Hyperparameters were fixed at `max_depth=6`, `num_leaves=31`, `learning_rate=0.05`, `n_estimators=200`, and `early_stopping_rounds=20`; no Optuna or random shuffling was used. Test-set deployment used only `model_prob > P` for P in {THRESHOLDS}, a 5s hold, touch round-trip entry and exit, taker fees on both legs, and one non-overlapping position per market.

## Per-Market Verdict

{markdown_table(
        [
            "market",
            "slug",
            "best P",
            "test events",
            "eligible",
            "trades",
            "ML mean",
            "ML CI",
            "TOB base",
            "OFI base",
            "delta TOB",
            "delta OFI",
            "verdict",
        ],
        verdict_rows,
    )}

## Feature Importance

{markdown_table(["feature", "gain", "split count"], importance_rows)}

## Probability Calibration Check

{markdown_table(["pred prob bin", "n", "mean predicted", "actual hit", "mean abs OFI"], calibration_rows)}

The calibration table is computed on pooled test-set events only. `actual hit` is the realized `direction_correct` rate, so threshold interpretation is meaningful only where the bin has enough samples and actual hit rate increases with predicted probability.

## Recommendation

{recommendation}.
"""
    NOTE.write_text(note, encoding="utf-8")


def main() -> None:
    ensure_lightgbm()
    print("load primary_read candidates", flush=True)
    candidates = load_candidates()
    print(f"load feature subset for {len(candidates)} markets", flush=True)
    raw = load_feature_subset(candidates)
    print(f"engineer A1.7 features from {len(raw):,} raw rows", flush=True)
    features = engineer_features(raw, candidates)
    print(f"model-eligible events: {len(features):,}", flush=True)
    print("fit pooled LightGBM", flush=True)
    model, features = fit_model(features)
    importance = feature_importance(model)
    calibration = calibration_table(features)
    print("evaluate threshold sweep with non-overlap", flush=True)
    results = evaluate_thresholds(features, candidates)

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_FEATURES, index=False)
    results.to_csv(OUT_RESULTS, index=False)
    importance.to_csv(OUT_IMPORTANCE, index=False)
    write_note(results, importance, calibration)
    print(f"wrote {OUT_FEATURES.relative_to(ROOT)}")
    print(f"wrote {OUT_RESULTS.relative_to(ROOT)}")
    print(f"wrote {OUT_IMPORTANCE.relative_to(ROOT)}")
    print(f"wrote {NOTE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
