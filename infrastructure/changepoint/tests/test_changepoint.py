"""
Tests for the causal changepoint detector.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python -m pytest infrastructure/changepoint/tests -q

Covers the DoD: no-lookahead online stability, recovery of injected breaks,
false-positive-rate bound on pure noise, detection lag, Cohen's κ, append-only
IO, and the integration hooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infrastructure.changepoint import (
    LiveDetector,
    append_changepoints,
    breaks_from_stream,
    changepoint_features,
    cohens_kappa,
    detection_metrics,
    embargo_indices_from_breaks,
    fresh_break_gate,
    kappa_vs_transitions,
    make_detector,
    run_detector,
)
from infrastructure.changepoint import offline

DETECTOR_NAMES = ["cusum", "page_hinkley", "bocpd"]


# ── THE invariant: appending future rows must not change any past output ─────
@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_no_lookahead_online_stability(name):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 500)
    x[250:] += 3.0          # a real break — exercises the interesting path
    full = run_detector(x, name=name)
    for k in (120, 251, 300, 480):
        pref = run_detector(x[:k], name=name)
        f = full.iloc[:k]
        assert (f["cp_flag"].to_numpy() == pref["cp_flag"].to_numpy()).all(), (name, k)
        assert (f["run_length_mode"].to_numpy() == pref["run_length_mode"].to_numpy()).all()
        assert np.allclose(f["change_prob"].to_numpy(), pref["change_prob"].to_numpy())
        assert np.allclose(f["statistic"].to_numpy(), pref["statistic"].to_numpy())


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_live_hook_matches_batch(name):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 300)
    x[150:] -= 2.5
    batch = run_detector(x, name=name)
    live = LiveDetector(name)
    rows = [live.update(i, x[i]) for i in range(len(x))]
    assert [r["cp_flag"] for r in rows] == list(batch["cp_flag"].to_numpy())
    assert [r["run_length_mode"] for r in rows] == [int(v) for v in batch["run_length_mode"]]


# ── recovers injected breaks ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["page_hinkley", "bocpd"])
def test_recovers_mean_shifts(name):
    x, tb = offline.make_mean_shifts(1200, [300, 600, 900], [0, 2.5, -1.5, 1.0], seed=1)
    s = run_detector(x, name=name)
    m = detection_metrics(tb, [int(b) for b in breaks_from_stream(s)], len(x), tolerance=25)
    assert m["recall"] >= 0.8, m
    assert m["median_lag"] <= 20, m


def test_bocpd_recovers_variance_shifts():
    # variance-only regimes are invisible to mean detectors; BOCPD must catch them
    x, tb = offline.make_var_shifts(1200, [400, 800], [0.5, 2.5, 0.8], seed=2)
    s = run_detector(x, name="bocpd")
    m = detection_metrics(tb, [int(b) for b in breaks_from_stream(s)], len(x), tolerance=30)
    assert m["recall"] >= 0.8, m


# ── false-positive rate on pure noise below a stated bound ───────────────────
@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_fpr_on_noise_below_bound(name):
    fars = []
    for seed in range(8):
        x, _ = offline.make_noise(2000, seed=seed)
        s = run_detector(x, name=name)
        m = detection_metrics([], [int(b) for b in breaks_from_stream(s)], len(x), tolerance=25)
        fars.append(m["far_per_1000"])
    # stated bound: fewer than 5 false alarms per 1000 stationary bars
    assert np.mean(fars) < 5.0, (name, np.mean(fars))


# ── Cohen's kappa vs regime transitions ──────────────────────────────────────
def test_kappa_finite_and_bocpd_recovers_transitions():
    x, labels, _ = offline.make_markov_switching(3000, seed=0)
    s = run_detector(x, name="bocpd")
    k = kappa_vs_transitions(labels, s["cp_flag"].to_numpy(), tolerance=15)
    assert np.isfinite(k["kappa"])
    # BOCPD should recover at least half of regime transitions within 15 bars
    assert k["transition_recall"] >= 0.5, k


def test_cohens_kappa_units():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 5000)
    c = rng.integers(0, 2, 5000)
    assert abs(cohens_kappa(b, c)) < 0.1   # independent -> ~chance


# ── append-only IO ───────────────────────────────────────────────────────────
def test_append_only_changepoints(tmp_path):
    x, _ = offline.make_mean_shifts(200, [100], [0, 3.0], seed=4)
    ts = pd.date_range("2024-01-01", periods=200, freq="D")
    s1 = run_detector(x[:120], name="bocpd", timestamps=ts[:120])
    p = append_changepoints(s1, tmp_path / "cp.parquet")
    n1 = len(pd.read_parquet(p))
    s2 = run_detector(x, name="bocpd", timestamps=ts)   # superset (more bars)
    append_changepoints(s2, p)
    out = pd.read_parquet(p)
    assert len(out) == 200 and len(out) >= n1            # grew, never shrank
    assert out["ts"].is_unique                            # no duplicate timestamps
    assert out["ts"].is_monotonic_increasing


# ── integration hooks ────────────────────────────────────────────────────────
def test_embargo_indices_from_breaks():
    idx = pd.RangeIndex(100)
    pos = embargo_indices_from_breaks(idx, [50], embargo_bars=3, symmetric=True)
    assert set(pos) == {47, 48, 49, 50, 51, 52, 53}
    fwd = embargo_indices_from_breaks(idx, [50], embargo_bars=3, symmetric=False)
    assert set(fwd) == {50, 51, 52, 53}


def test_changepoint_features_causal_shape():
    x, _ = offline.make_mean_shifts(300, [150], [0, 3.0], seed=5)
    ts = pd.date_range("2024-01-01", periods=300, freq="D")
    feats = changepoint_features(x, timestamps=ts, name="bocpd")
    assert list(feats.columns) == ["cp_change_prob", "cp_run_length", "cp_flag", "cp_bars_since"]
    assert len(feats) == 300 and feats.index.equals(pd.Index(ts, name="ts"))
    assert not feats.isna().any().any()
    assert (feats["cp_change_prob"] >= 0).all() and (feats["cp_change_prob"] <= 1).all()


def test_fresh_break_gate_blocks_after_break():
    x, _ = offline.make_mean_shifts(200, [100], [0, 4.0], seed=6)
    s = run_detector(x, name="bocpd")
    gate = fresh_break_gate(s, cooldown=5)
    flagged = np.where(s["cp_flag"].to_numpy())[0]
    if len(flagged):
        b = int(flagged[0])
        assert not gate.iloc[b]                           # blocked at the break
        assert not gate.iloc[min(len(gate) - 1, b + 1)]   # and just after
    assert gate.dtype == bool


# ── misc ─────────────────────────────────────────────────────────────────────
def test_make_detector_unknown():
    with pytest.raises(KeyError):
        make_detector("ema_nonsense")


def test_ruptures_offline_recovers_break_if_installed():
    if not offline.has_ruptures():
        pytest.skip("ruptures not installed (offline-only validation dep)")
    x, _ = offline.make_mean_shifts(400, [200], [0, 4.0], seed=8)
    bkps = offline.ruptures_offline(x, pen=10)
    assert any(abs(b - 200) <= 15 for b in bkps), bkps
