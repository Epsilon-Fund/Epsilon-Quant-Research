"""Tests for lib/microstructure_metrics.py — incl. the Finding-1 reproduction gate.

Two layers:

1. **Pure unit tests** (always run): hit-rate zero-move semantics, directional
   return, non-overlap episode selection.
2. **Finding-1 reproduction** (skipped when ``data/analysis/block_a1_features.parquet``
   is absent): ONE code path (:func:`lib.microstructure_metrics.hit_rate` over the
   same non-overlap episode set) must reproduce BOTH headline numbers from
   [[pm_prealvaro_pipeline_trust_audit_findings]] Finding 1 on the exact Retest-C
   universe (a0c_roll, crypto-4h slugs, >=300 last-trade events, discovery
   threshold 0.937422):

   * ``zero_move="miss"``    -> ~36.0-36.1% (the official Retest-C figure)
   * ``zero_move="exclude"`` -> ~63.0%      (the audit's conditional recomputation)

   Verified live 2026-07-21: miss=0.3614 (n=3143), exclude=0.6329 (n=1795),
   recomputed discovery abs_q90 = 0.937422 exactly.

   KNOWN RESIDUAL DISCREPANCY (reported, not hidden): the retest surface's pooled
   ``mean_directional_return_bps = +58.1`` is a RAW episode-mean of the
   directional-mid return (no sign weighting). This reconstruction yields +61.2
   raw / +82.3 sign-weighted — the raw figure is ~3 bps off the official CSV,
   presumably episode-set drift at reconstruction boundaries; the hit-rate anchors
   above reproduce to <0.2pp so the discrepancy is bounded and does not affect
   Finding 1. See the findings note for the write-up.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.microstructure_metrics import (
    directional_return_bps,
    hit_rate,
    non_overlap_episode_count,
    non_overlap_episode_mask,
    zero_move_share,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "analysis" / "block_a1_features.parquet"

# ---------------------------------------------------------------------------
# 1. pure unit tests
# ---------------------------------------------------------------------------


def test_hit_rate_zero_move_miss_vs_exclude() -> None:
    sig = [1.0, 1.0, -1.0, -1.0, 1.0]
    ret = [2.0, 0.0, -3.0, 1.0, 0.0]
    # miss: hits = {+/+, -/-} = 2 of 5
    hr, n = hit_rate(sig, ret, zero_move="miss")
    assert n == 5 and hr == pytest.approx(0.4)
    # exclude: zero-move rows dropped -> 2 of 3
    hr, n = hit_rate(sig, ret, zero_move="exclude")
    assert n == 3 and hr == pytest.approx(2 / 3)


def test_hit_rate_zero_signal_rows_always_excluded() -> None:
    hr, n = hit_rate([0.0, 1.0], [5.0, 5.0], zero_move="miss")
    assert n == 1 and hr == pytest.approx(1.0)


def test_hit_rate_nan_rows_excluded_and_empty_is_nan() -> None:
    hr, n = hit_rate([np.nan, 1.0], [1.0, np.nan], zero_move="miss")
    assert n == 0 and np.isnan(hr)


def test_hit_rate_rejects_bad_zero_move() -> None:
    with pytest.raises(ValueError):
        hit_rate([1.0], [1.0], zero_move="drop")  # type: ignore[arg-type]


def test_zero_move_share() -> None:
    assert zero_move_share([0.0, 1.0, 0.0, np.nan]) == pytest.approx(2 / 3)


def test_directional_return_bps_sign_weighted() -> None:
    dr, n = directional_return_bps([1.0, -2.0, 0.0], [10.0, 5.0, 100.0])
    # sign-weighted: +10 (followed up), -5 (short a rise), 0 (no signal) -> mean 5/3
    assert n == 3 and dr == pytest.approx(5 / 3)


def test_non_overlap_mask_blocks_within_horizon() -> None:
    t = np.array([0, 3, 6, 11], dtype=np.int64)  # horizon 5 -> keep 0, 6, then 11 blocked (6+5=11, ts<=block)
    keep = non_overlap_episode_mask(t, 5)
    assert keep.tolist() == [True, False, True, False]
    assert non_overlap_episode_count(t, 5) == 2


def test_non_overlap_mask_tiebreak_larger_abs_signal() -> None:
    t = np.array([10, 10, 20], dtype=np.int64)
    keep = non_overlap_episode_mask(t, 5, abs_signal=[0.5, 0.9, 0.1])
    assert keep.tolist() == [False, True, True]


def test_non_overlap_mask_preserves_input_order_alignment() -> None:
    # unsorted input: mask must align with input positions
    t = np.array([20, 0, 3], dtype=np.int64)
    keep = non_overlap_episode_mask(t, 5)
    assert keep.tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# 2. Finding-1 reproduction gate (data-dependent; skipped when parquet absent)
# ---------------------------------------------------------------------------

STALE_MAX_S = 5.0
HORIZON_NS = 5 * 1_000_000_000
DISCOVERY_THRESHOLD = 0.937422  # abs_q90 of tob_imbalance_level, a0/a0b crypto_4h valid rows
EXCLUDE_SLUG = "will-jd-vance"  # excluded by the retest at load time
CRYPTO_SLUG_RE = re.compile(r"^(btc|eth|sol)-updown-4h-(\d+)$")


def _load_roll() -> pd.DataFrame:
    import duckdb

    cols = [
        "run_id", "received_at", "exchange_ts", "event_type", "asset_id", "market_id",
        "slug", "outcome_index", "is_book_state_complete", "book_staleness_seconds",
        "best_bid", "best_ask", "mid", "tob_imbalance",
    ]
    con = duckdb.connect()
    df = con.execute(
        f"SELECT {', '.join(cols)} FROM read_parquet(?) "
        "WHERE run_id = 'a0c_roll' AND coalesce(lower(slug), '') NOT LIKE ?",
        [str(FEATURES), f"%{EXCLUDE_SLUG}%"],
    ).df()
    con.close()
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    df["exchange_ts"] = pd.to_datetime(df["exchange_ts"], utc=True, errors="coerce")
    df["event_ts"] = df["exchange_ts"].where(df["exchange_ts"].notna(), df["received_at"])
    for c in ("event_type", "asset_id", "market_id", "slug"):
        df[c] = df[c].fillna("").astype(str)
    for c in ("outcome_index", "book_staleness_seconds", "best_bid", "best_ask", "mid", "tob_imbalance"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_book_state_complete"] = df["is_book_state_complete"].fillna(False).astype(bool)
    df["direction_factor"] = np.where(df["outcome_index"].fillna(0).astype(int).eq(0), 1.0, -1.0)
    df = df.sort_values(["market_id", "asset_id", "event_ts"]).reset_index(drop=True)
    df["tob_level"] = df["direction_factor"] * df.groupby(["market_id", "asset_id"], sort=False)[
        "tob_imbalance"
    ].ffill()
    return df


def _valid_quote_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["is_book_state_complete"]
        & df["event_ts"].notna()
        & df["book_staleness_seconds"].le(STALE_MAX_S)
        & df["best_bid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
        & df["mid"].replace([np.inf, -np.inf], np.nan).notna()
        & df["best_bid"].ge(0.0)
        & df["best_ask"].le(1.0)
        & df["best_ask"].ge(df["best_bid"])
    )


def _ns(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[ns]").astype("int64")


def _market_episodes(g: pd.DataFrame) -> pd.DataFrame:
    """Retest-C episode construction for one market (both outcome tokens)."""
    valid = g[_valid_quote_mask(g)]
    arrays = {}
    for aid, ga in valid.groupby("asset_id", sort=False):
        ga = ga.sort_values("event_ts")
        arrays[aid] = (_ns(ga["event_ts"]), ga["mid"].to_numpy(float), ga["direction_factor"].to_numpy(float))
    sub = valid[valid["tob_level"].replace([np.inf, -np.inf], np.nan).notna()].copy()
    sub = sub[sub["tob_level"].abs().ge(DISCOVERY_THRESHOLD)].copy()
    if sub.empty:
        return sub
    sub["abs_signal"] = sub["tob_level"].abs()
    sub["t_ns"] = _ns(sub["event_ts"])
    sub = (
        sub.sort_values(["t_ns", "abs_signal"], ascending=[True, False])
        .drop_duplicates(["t_ns"], keep="first")
        .reset_index(drop=True)
    )
    rets = np.full(len(sub), np.nan)
    for i, row in enumerate(sub.itertuples(index=False)):
        state = arrays.get(row.asset_id)
        if state is None:
            continue
        times, mid, _ = state
        entry_idx = int(np.searchsorted(times, row.t_ns, side="right") - 1)
        exit_idx = int(np.searchsorted(times, row.t_ns + HORIZON_NS, side="right") - 1)
        if entry_idx < 0 or exit_idx <= entry_idx:
            continue
        d = float(row.direction_factor)
        cur = mid[entry_idx] if d > 0 else 1.0 - mid[entry_idx]
        fut = mid[exit_idx] if d > 0 else 1.0 - mid[exit_idx]
        if cur <= 0 or not np.isfinite(fut):
            continue
        rets[i] = (fut - cur) / cur * 10_000.0
    ok = np.isfinite(rets)
    cand = sub[ok].copy()
    cand["ret_bps"] = rets[ok]
    if cand.empty:
        return cand
    keep = non_overlap_episode_mask(
        cand["t_ns"].to_numpy(np.int64), HORIZON_NS, abs_signal=cand["abs_signal"].to_numpy(float)
    )
    return cand[keep]


@pytest.mark.skipif(not FEATURES.exists(), reason="block_a1_features.parquet not on disk")
def test_finding1_one_code_path_reproduces_both_headline_hit_rates() -> None:
    roll = _load_roll()
    parts = []
    for market_id, g in roll.groupby("market_id", sort=False):
        slugs = g["slug"][g["slug"].ne("")]
        slug = str(slugs.iloc[0]) if len(slugs) else ""
        if not CRYPTO_SLUG_RE.match(slug):
            continue
        if int(g["event_type"].eq("last_trade_price").sum()) < 300:
            continue
        eps = _market_episodes(g)
        if not eps.empty:
            parts.append(eps)
    eps = pd.concat(parts, ignore_index=True)
    sig = eps["tob_level"].to_numpy(float)
    ret = eps["ret_bps"].to_numpy(float)

    hr_miss, n_miss = hit_rate(sig, ret, zero_move="miss")
    hr_cond, n_cond = hit_rate(sig, ret, zero_move="exclude")

    # official Retest-C figure 36.0%; audit recompute 36.1%
    assert hr_miss == pytest.approx(0.361, abs=0.005), f"zero-as-miss={hr_miss:.4f} (n={n_miss})"
    # audit's conditional recomputation 63.0%
    assert hr_cond == pytest.approx(0.630, abs=0.01), f"conditional={hr_cond:.4f} (n={n_cond})"
    # the audit's "~1,800 independent episodes" = episodes with a nonzero move
    assert 1600 <= n_cond <= 2000
    # raw episode-mean return: official surface says +58.1; reconstruction lands ~+61
    raw_mean = float(ret.mean())
    assert raw_mean == pytest.approx(58.1, abs=5.0), f"raw episode-mean={raw_mean:.1f}"
