"""Tests for the SPCX overnight gap-filler (Block S5f).

Acceptance criteria encoded here:
  (i)   the forward-fill is LOOKAHEAD-FREE — a grid point only ever sees quotes whose
        timestamp is <= that point (the core invariant: no future leakage into the past);
  (ii)  gap detection skips grid points already covered by an existing shard;
  (iii) a synthetic snapshot is byte-compatible with the live pipeline — it runs through
        the real analyze()/log_parquet() and yields a shard the dashboard backfill reads;
  (iv)  the run is idempotent — re-running over the same window overwrites, never duplicates.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from scripts import spcx_backfill_history as bf
from scripts.spcx_pm_pdf_monitor import DashboardState


# ---------------------------------------------------------------- (i) lookahead-free
def test_asof_is_lookahead_free():
    series = [(100.0, 0.9), (200.0, 0.8), (300.0, 0.7)]
    assert bf._asof(series, 50.0) is None        # before the first quote — nothing known
    assert bf._asof(series, 100.0) == 0.9        # exactly at a quote
    assert bf._asof(series, 150.0) == 0.9        # forward-fill the last <= t (never 0.8)
    assert bf._asof(series, 299.0) == 0.8        # still the 200s quote, not the 300s future
    assert bf._asof(series, 10_000.0) == 0.7     # well past the end → last known
    assert bf._asof([], 123.0) is None


# ---------------------------------------------------------------- (ii) gap detection
def test_covered_detects_only_nearby_shards():
    existing = [1000.0, 5000.0]
    assert bf._covered(existing, 1000.0, tol_s=300.0) is True   # exact
    assert bf._covered(existing, 1200.0, tol_s=300.0) is True   # within ±300
    assert bf._covered(existing, 1400.0, tol_s=300.0) is False  # 400s away → genuine hole
    assert bf._covered([], 1000.0, tol_s=300.0) is False


# ---------------------------------------------------------------- shared synthetic feed
_LADDER = [{"strike_t": k, "token": f"L{i}", "question": f">{k}T"}
           for i, k in enumerate([1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0])]
_BUCKETS = [{"label": "1.5-2.0T", "lo": 1.5, "hi": 2.0, "token": "B0", "question": "x"}]


def _fake_meta():
    return {"ladder": list(_LADDER), "buckets": list(_BUCKETS), "no_ipo": None}


def _fake_token_history(token, start_s, end_s, fidelity_min=10, timeout=30.0):
    # monotone-decreasing survivor across strikes, two timestamps inside the window
    base = {"L0": 0.97, "L1": 0.90, "L2": 0.78, "L3": 0.60,
            "L4": 0.42, "L5": 0.22, "L6": 0.06, "B0": 0.18}[token]
    return [(start_s + 60.0, base), (start_s + 3600.0, base * 0.95)]


def _fake_candles(start_s, end_s, interval="5m", timeout=30.0):
    return [(start_s + 60.0, 150.0), (start_s + 3600.0, 158.0)]


def _patch_feeds(monkeypatch):
    monkeypatch.setattr(bf, "fetch_pm_metadata", _fake_meta)
    monkeypatch.setattr(bf, "fetch_token_history", _fake_token_history)
    monkeypatch.setattr(bf, "fetch_hl_candles", _fake_candles)


# ------------------------------------------------- (iii) schema parity with live pipeline
def test_synth_snapshot_runs_through_real_analyze():
    meta = _fake_meta()
    lad = {d["token"]: _fake_token_history(d["token"], 0, 7200) for d in meta["ladder"]}
    buc = {d["token"]: _fake_token_history(d["token"], 0, 7200) for d in meta["buckets"]}
    snap = bf.synth_snapshot(meta, lad, buc, [], _fake_candles(0, 7200), t=4000.0)
    # exact shape the live build_snapshot emits
    assert set(snap) == {"fetched_at_utc", "ladder", "buckets", "no_ipo", "hl"}
    assert snap["ladder"][0]["bid"] == snap["ladder"][0]["ask"]  # midpoint → bid==ask
    assert snap["hl"]["mark"] == 158.0
    from scripts.spcx_pm_pdf_monitor import analyze
    rep = analyze(snap, basis="mid")
    assert rep["stats_primary"]["mean_ps"] > 0
    assert np.all(np.diff(rep["fit"]["S"]) <= 1e-9)  # survivor still monotone


def test_backfill_writes_loadable_shards(tmp_path, monkeypatch):
    _patch_feeds(monkeypatch)
    summ = bf.backfill(0, 7200, step_s=600, out_dir=tmp_path)
    assert summ["written"] > 0
    shards = sorted(tmp_path.glob("poll_*.parquet"))
    assert len(shards) == summ["written"]
    # the dashboard backfill machinery reads exactly these columns
    cols = set(pq.read_table(shards[0]).column_names)
    assert {"poll_ts", "mean_ps", "median_ps", "p_win_offer", "hl_mark"} <= cols


# ------------------------------------------------- (iv) idempotency + gap skipping
def test_backfill_is_idempotent(tmp_path, monkeypatch):
    _patch_feeds(monkeypatch)
    s1 = bf.backfill(0, 7200, step_s=600, out_dir=tmp_path)
    n_after_first = len(list(tmp_path.glob("poll_*.parquet")))
    # second pass: every grid point is now covered by the shards we just wrote
    s2 = bf.backfill(0, 7200, step_s=600, out_dir=tmp_path)
    n_after_second = len(list(tmp_path.glob("poll_*.parquet")))
    assert n_after_second == n_after_first       # no duplicates created
    assert s2["written"] == 0 and s2["covered"] >= s1["written"]


def test_fill_all_overrides_gap_skip(tmp_path, monkeypatch):
    _patch_feeds(monkeypatch)
    bf.backfill(0, 7200, step_s=600, out_dir=tmp_path)
    s2 = bf.backfill(0, 7200, step_s=600, out_dir=tmp_path, fill_all=True)
    assert s2["written"] > 0 and s2["covered"] == 0  # --fill-all ignores existing coverage
