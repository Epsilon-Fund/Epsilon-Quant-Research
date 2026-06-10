"""Block B TFI hit-rate re-run partitioned by emit path (sub-task 2).

Background: CTF Exchange `_matchOrders` emits the internal active leg of a
two-sided match with maker = takerOrder.maker and taker = address(this).
Block B's "operator-filtered" lift (equity_index 58.8% @300s, crypto 52%)
dropped exactly those internal-leg rows. This script partitions the same
cached family fills into:

  - match_orders_leg : taker is one of the 4 exchange-internal-leg contracts
                       (the wallet in `maker` was the AGGRESSOR)
  - single_sided     : every other row (conventional maker=passive /
                       taker=aggressor semantics)

plus two reconciliation populations that reproduce the original Block B rows:

  - all_fills_recon        : the cached mixed-population eval parquet
  - operator_removed_recon : the original OPERATOR_ADDRESSES maker/taker filter
                             (= single_sided minus MM-bot/HFT-touched rows)

Before recomputing, it sanity-checks `historical_to_aggressor()` semantics on
the match_orders partition: on internal-leg rows `maker_side` is the
AGGRESSOR's token side (the maker column holds the active order signer), so
the inverse_maker_side convention is sign-INVERTED on that subset. The check
is empirical: within transactions that contain an internal leg, the internal
leg's maker_side should be the inverse of its sibling normal legs' maker_side
and its usd_amount should equal the sum of the sibling legs.

Methodology is identical to scripts/dali_tfi_deep_dive.py (imported, not
re-implemented): same market-second bars, same lookahead-free forward
merge_asof, same filter defaults (min |signal| $25, future gap <= 300s,
exclude last 600s before market end), same magnitude buckets and bootstrap
CIs, plus Wilson binomial CIs on hit rates. All inputs are cached/local —
no new capture, no Goldsky backfill.

Run:
    cd polymarket/research
    PYTHONPATH=. uv run python scripts/copytrade_tfi_emit_path_partition.py

Outputs (data/analysis/csv_outputs/copytrade/):
    copytrade_tfi_emit_path_partition.csv      all buckets/horizons/conventions
    copytrade_tfi_emit_path_headline.csv       top-decile @300s summary
    copytrade_tfi_match_orders_sign_check.csv  per-family bundle sign check
    copytrade_tfi_emit_path_paired_by_market.csv  per-market composition control
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_infra.operator_denylist import EXCHANGE_INTERNAL_LEG, OPERATOR_ADDRESSES
from scripts.dali_tfi_deep_dive import (
    DEFAULT_HORIZONS,
    DEFAULT_INPUTS,
    FamilyInput,
    assign_magnitude_bucket,
    filter_eval,
    metric_rows,
    read_candidates,
    read_eval,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "analysis" / "csv_outputs" / "copytrade"
OUT_ALL = OUT_DIR / "copytrade_tfi_emit_path_partition.csv"
OUT_HEADLINE = OUT_DIR / "copytrade_tfi_emit_path_headline.csv"
OUT_SIGN_CHECK = OUT_DIR / "copytrade_tfi_match_orders_sign_check.csv"
OUT_PAIRED = OUT_DIR / "copytrade_tfi_emit_path_paired_by_market.csv"

# Same defaults as dali_tfi_deep_dive.parse_args()
MIN_SIGNAL_USD = 25.0
MAX_FUTURE_GAP_SECONDS = 300
EXCLUDE_LAST_SECONDS = 600
MIN_OBS = 100
BOOTSTRAP_SAMPLES = 300
HEADLINE_HORIZON = 300


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_ci(hits: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, returned in percent."""
    if n <= 0:
        return (np.nan, np.nan)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100.0 * (centre - half), 100.0 * (centre + half))


def load_fills(inp: FamilyInput) -> pd.DataFrame:
    cols = [
        "timestamp", "market_id", "maker", "taker", "maker_side",
        "price", "usd_amount", "transaction_hash",
    ]
    fills = pd.read_parquet(inp.fills_path, columns=cols)
    fills["market_id"] = fills["market_id"].astype(str)
    fills["maker"] = fills["maker"].str.lower()
    fills["taker"] = fills["taker"].str.lower()
    fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True, errors="coerce")
    fills = fills[fills["price"].notna() & fills["usd_amount"].gt(0)].copy()
    fills["is_internal_leg"] = fills["taker"].isin(EXCHANGE_INTERNAL_LEG)
    return fills


def build_eval_from_fills(inp: FamilyInput, fills: pd.DataFrame) -> pd.DataFrame:
    """Identical bar/eval construction to dali_tfi_deep_dive.
    build_eval_from_filtered_fills(), but takes an already-filtered fills
    frame instead of hardcoding the operator filter."""
    fills = fills.copy()
    fills["signed_piece"] = np.where(
        fills["maker_side"].eq("BUY"), fills["usd_amount"], -fills["usd_amount"]
    )
    fills["price_x_usd"] = fills["price"] * fills["usd_amount"]
    bars = (
        fills.groupby(["market_id", "timestamp"], dropna=False)
        .agg(
            n_fills=("price", "size"),
            n_txs=("transaction_hash", "nunique"),
            gross_usd=("usd_amount", "sum"),
            signed_maker_usd=("signed_piece", "sum"),
            price_x_usd=("price_x_usd", "sum"),
        )
        .reset_index()
        .rename(columns={"timestamp": "second_ts"})
    )
    bars["vwap_price"] = bars["price_x_usd"] / bars["gross_usd"]
    candidates = read_candidates(inp)[["market_id", "candidate_end_ts"]]
    bars = bars.merge(candidates, on="market_id", how="left")
    bars = bars.rename(columns={"candidate_end_ts": "end_ts"})
    bars = bars[bars["vwap_price"].between(0.01, 0.99)].sort_values(
        ["market_id", "second_ts"]
    )

    eval_parts = []
    for horizon in DEFAULT_HORIZONS:
        for market_id, g in bars.groupby("market_id", sort=False):
            base = g.copy()
            base["target_ts"] = base["second_ts"] + pd.to_timedelta(horizon, unit="s")
            future = g[["second_ts", "vwap_price"]].rename(
                columns={"second_ts": "future_ts", "vwap_price": "future_vwap_price"}
            )
            merged = pd.merge_asof(
                base.sort_values("target_ts"),
                future.sort_values("future_ts"),
                left_on="target_ts",
                right_on="future_ts",
                direction="forward",
            )
            merged["market_id"] = market_id
            merged["horizon_seconds"] = horizon
            merged["future_gap_seconds"] = (
                merged["future_ts"] - merged["target_ts"]
            ).dt.total_seconds()
            merged["future_price_change"] = (
                merged["future_vwap_price"] - merged["vwap_price"]
            )
            eval_parts.append(merged)
    if not eval_parts:
        return pd.DataFrame()
    out = pd.concat(eval_parts, ignore_index=True)
    out["family"] = inp.family
    out["family_label"] = inp.label
    out["abs_signal_usd"] = out["signed_maker_usd"].abs()
    out["seconds_to_end"] = (out["end_ts"] - out["second_ts"]).dt.total_seconds()
    meta = read_candidates(inp)
    return out.merge(meta, on="market_id", how="left")


# ---------------------------------------------------------------------------
# historical_to_aggressor() sanity check on the match_orders partition
# ---------------------------------------------------------------------------
def sign_check(inp: FamilyInput, fills: pd.DataFrame) -> dict[str, object]:
    """Within (transaction_hash, market_id) bundles containing an internal
    leg, check that the internal leg's maker_side is the INVERSE of the
    sibling normal legs' maker_side and that its usd_amount equals the sum
    of the siblings. If so, on internal-leg rows maker_side is the
    aggressor's own side, so historical_to_aggressor() (which inverts
    maker_side) is sign-inverted on that subset."""
    internal_tx = fills.loc[fills["is_internal_leg"], ["transaction_hash", "market_id"]]
    keys = set(map(tuple, internal_tx.drop_duplicates().to_numpy()))
    sub = fills[
        fills.set_index(["transaction_hash", "market_id"]).index.isin(keys)
    ].copy()

    rows = []
    for (_, _), g in sub.groupby(["transaction_hash", "market_id"], sort=False):
        internal = g[g["is_internal_leg"]]
        normal = g[~g["is_internal_leg"]]
        if len(internal) != 1 or len(normal) == 0:
            continue
        i = internal.iloc[0]
        usd_ratio = float(normal["usd_amount"].sum()) / float(i["usd_amount"])
        all_inverse = bool((normal["maker_side"] != i["maker_side"]).all())
        rows.append((usd_ratio, all_inverse))
    if not rows:
        return {
            "family_label": inp.label,
            "n_internal_rows": int(fills["is_internal_leg"].sum()),
            "n_checkable_bundles": 0,
        }
    arr = pd.DataFrame(rows, columns=["usd_ratio", "all_sides_inverse"])
    return {
        "family_label": inp.label,
        "n_internal_rows": int(fills["is_internal_leg"].sum()),
        "n_checkable_bundles": int(len(arr)),
        "pct_sides_inverse": 100.0 * float(arr["all_sides_inverse"].mean()),
        "median_usd_ratio_normal_over_internal": float(arr["usd_ratio"].median()),
        "pct_usd_match_within_2pct": 100.0
        * float(arr["usd_ratio"].between(0.98, 1.02).mean()),
    }


# ---------------------------------------------------------------------------
# Main partitioned recompute
# ---------------------------------------------------------------------------
def partition_metrics(
    inp: FamilyInput, evals_cached: pd.DataFrame, fills: pd.DataFrame
) -> pd.DataFrame:
    partitions: dict[str, pd.DataFrame] = {}
    # Reconciliation 1: mixed population from the cached eval parquet
    partitions["all_fills_recon"] = evals_cached
    # Reconciliation 2: the original operator filter (maker AND taker clean)
    op_mask = ~fills["maker"].isin(OPERATOR_ADDRESSES) & ~fills["taker"].isin(
        OPERATOR_ADDRESSES
    )
    partitions["operator_removed_recon"] = build_eval_from_fills(inp, fills[op_mask])
    # The pre-registered emit-path partitions
    partitions["match_orders_leg"] = build_eval_from_fills(
        inp, fills[fills["is_internal_leg"]]
    )
    partitions["single_sided"] = build_eval_from_fills(
        inp, fills[~fills["is_internal_leg"]]
    )

    rows: list[dict[str, object]] = []
    for name, ev in partitions.items():
        if ev.empty:
            continue
        sub = filter_eval(
            ev,
            exclude_last_seconds=EXCLUDE_LAST_SECONDS,
            min_signal_usd=MIN_SIGNAL_USD,
            max_future_gap_seconds=MAX_FUTURE_GAP_SECONDS,
            horizons=DEFAULT_HORIZONS,
        )
        sub["emit_partition"] = name
        sub = assign_magnitude_bucket(sub, ["family", "horizon_seconds"])
        rows.extend(
            metric_rows(
                sub,
                group_cols=[
                    "family",
                    "family_label",
                    "emit_partition",
                    "horizon_seconds",
                    "magnitude_bucket",
                ],
                component="emit_path_partition",
                min_obs=MIN_OBS,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Wilson binomial CI on the hit rate (n_obs independent by non-overlap of
    # the per-second bars; same unit of observation as the original).
    wl, wh = zip(
        *[
            wilson_ci(r["hit_rate_pct"] / 100.0 * r["n_obs"], r["n_obs"])
            for _, r in out.iterrows()
        ]
    )
    out["hit_rate_wilson_low_pct"] = wl
    out["hit_rate_wilson_high_pct"] = wh
    # Which convention is the CORRECT token-side aggressor for this partition:
    # internal-leg rows carry the aggressor's own side in maker_side.
    out["corrected_aggressor"] = np.where(
        out["emit_partition"].eq("match_orders_leg"),
        out["sign_convention"].eq("maker_side"),
        out["sign_convention"].eq("inverse_maker_side"),
    )
    return out


# ---------------------------------------------------------------------------
# Composition control: per-market paired comparison at the headline horizon
# ---------------------------------------------------------------------------
def paired_by_market(
    inp: FamilyInput, fills: pd.DataFrame, min_market_obs: int = 30
) -> pd.DataFrame:
    """Per-market hit rates at 300s, all qualifying bars (no magnitude
    bucketing — per-market top-decile cells are too thin). Two comparisons:

      raw_inverse  : both partitions scored with inverse_maker_side (the
                     Block B convention). If the single_sided lift survives
                     within markets, it is not market-composition.
      corrected    : single_sided scored with inverse_maker_side,
                     match_orders_leg scored with maker_side (its true
                     aggressor convention). If the gap closes here, the lift
                     is a sign/attribution artifact of the internal legs, not
                     a 'cleaner population' effect."""
    per_part = {}
    for name, mask in [
        ("single_sided", ~fills["is_internal_leg"]),
        ("match_orders_leg", fills["is_internal_leg"]),
    ]:
        ev = build_eval_from_fills(inp, fills[mask])
        if ev.empty:
            continue
        sub = filter_eval(
            ev,
            exclude_last_seconds=EXCLUDE_LAST_SECONDS,
            min_signal_usd=MIN_SIGNAL_USD,
            max_future_gap_seconds=MAX_FUTURE_GAP_SECONDS,
            horizons=[HEADLINE_HORIZON],
        )
        sign = np.sign(sub["signed_maker_usd"].to_numpy(dtype=float))
        fut = sub["future_price_change"].to_numpy(dtype=float)
        sub["hit_inverse"] = (sign * -1.0 * fut) > 0
        sub["hit_maker"] = (sign * fut) > 0
        g = sub.groupby("market_id").agg(
            n=("hit_inverse", "size"),
            hit_inverse=("hit_inverse", "mean"),
            hit_maker=("hit_maker", "mean"),
        )
        per_part[name] = g[g["n"] >= min_market_obs]

    if len(per_part) < 2:
        return pd.DataFrame()
    j = per_part["single_sided"].join(
        per_part["match_orders_leg"], lsuffix="_ss", rsuffix="_mo", how="inner"
    )
    if j.empty:
        return pd.DataFrame()

    def boot_mean_ci(vals: np.ndarray, seed: int = 7) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        n = len(vals)
        means = rng.choice(vals, size=(2000, n), replace=True).mean(axis=1)
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    rows = []
    for comp, col_mo in [("raw_inverse", "hit_inverse_mo"), ("corrected", "hit_maker_mo")]:
        diff = (j["hit_inverse_ss"] - j[col_mo]).to_numpy(dtype=float) * 100.0
        lo, hi = boot_mean_ci(diff)
        rows.append(
            {
                "family_label": inp.label,
                "comparison": comp,
                "n_paired_markets": int(len(j)),
                "mean_within_market_hit_gap_pp": float(diff.mean()),
                "gap_ci_low_pp": lo,
                "gap_ci_high_pp": hi,
                "mean_hit_single_sided_pct": 100.0 * float(j["hit_inverse_ss"].mean()),
                "mean_hit_match_orders_pct": 100.0 * float(j[col_mo].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = [inp for inp in DEFAULT_INPUTS if inp.eval_path.exists() and inp.fills_path.exists()]
    if not inputs:
        raise SystemExit("no cached Dali TFI inputs found")

    all_rows, sign_rows, paired_rows = [], [], []
    for inp in inputs:
        log(f"family {inp.label}: loading fills")
        fills = load_fills(inp)
        log(
            f"  {len(fills):,} fills; internal legs "
            f"{int(fills['is_internal_leg'].sum()):,} "
            f"({100.0 * fills['is_internal_leg'].mean():.1f}%)"
        )
        log("  sign check on match_orders bundles")
        sign_rows.append(sign_check(inp, fills))
        log("  partitioned metric recompute")
        evals_cached = read_eval(inp)
        m = partition_metrics(inp, evals_cached, fills)
        if not m.empty:
            all_rows.append(m)
        log("  per-market paired composition control")
        p = paired_by_market(inp, fills)
        if not p.empty:
            paired_rows.append(p)

    metrics = pd.concat(all_rows, ignore_index=True)
    metrics.to_csv(OUT_ALL, index=False)
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(OUT_SIGN_CHECK, index=False)
    paired = (
        pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    )
    paired.to_csv(OUT_PAIRED, index=False)

    headline = metrics[
        metrics["magnitude_bucket"].eq("top_decile")
        & metrics["horizon_seconds"].eq(HEADLINE_HORIZON)
    ][
        [
            "family_label", "emit_partition", "sign_convention",
            "corrected_aggressor", "n_obs", "hit_rate_pct",
            "hit_rate_wilson_low_pct", "hit_rate_wilson_high_pct",
            "hit_rate_ci_low_pct", "hit_rate_ci_high_pct",
            "mean_return_cents", "net_edge_after_1tick_cents",
        ]
    ].sort_values(["family_label", "emit_partition", "sign_convention"])
    headline.to_csv(OUT_HEADLINE, index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n=== sign check (match_orders bundles) ===")
        print(sign_df.round(2).to_string(index=False))
        print("\n=== headline: top-decile @300s ===")
        print(headline.round(3).to_string(index=False))
        print("\n=== per-market paired composition control (300s, all bars) ===")
        print(paired.round(2).to_string(index=False))

    for p in [OUT_ALL, OUT_HEADLINE, OUT_SIGN_CHECK, OUT_PAIRED]:
        log(f"written: {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
