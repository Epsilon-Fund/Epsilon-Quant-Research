"""
Gates for the calibration scoring engine.

1. Brier–Murphy decomposition reconciles to the total Brier on synthetic data.
2. Reliability separates a well- vs a deliberately over-confident forecaster.
3. Recalibration (both backends) reduces calibration error out-of-sample.
4. Markets layer, Spiegelhalter Z, ledger reader (all resolution paths), CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from lemma.calibrate import core as C


# ── synthetic forecasters ───────────────────────────────────────────────────────

def _well_calibrated(n=4000, seed=0):
    """Forecasts drawn over [0,1]; outcomes ~ Bernoulli(p) → calibrated by construction."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(float)
    return p, y


def _overconfident(n=4000, seed=1):
    """
    True chance is mild (centred on 0.5) but the forecaster reports extreme
    probabilities — classic over-confidence. Outcomes follow the TRUE prob, so
    the reported forecasts are systematically too sure.
    """
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.30, 0.70, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    reported = np.clip(0.5 + (true_p - 0.5) * 3.0, 0.01, 0.99)
    return reported, y


def _write_ledger(root: Path) -> Path:
    """A tiny synthetic ledger in the events.jsonl convention."""
    ledger = root / "ledger"
    (ledger / "forecasts").mkdir(parents=True)
    recs = [
        {"type": "forecast_created", "id": "sf-2026-001"},
        {"type": "scored", "id": "sf-2026-001", "timestamp": "2026-06-29T00:00:00+00:00",
         "final_probability": 0.7, "outcome": 1, "brier": 0.09},
        {"type": "scored", "id": "sf-2026-002", "timestamp": "2026-06-29T00:01:00+00:00",
         "final_probability": 0.4, "outcome": 0, "brier": 0.16},
    ]
    (ledger / "forecasts" / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    return ledger


# ── Gate 1: Murphy decomposition reconciles ──────────────────────────────────────

def test_brier_decomposition_reconciles():
    # (a) continuous forecasts, unique-value grouping → EXACT identity
    p, y = _well_calibrated()
    d = C.murphy_decomposition(p, y, n_bins=None)
    lhs = d["brier"]
    rhs = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert abs(lhs - rhs) < 1e-12, f"exact identity broke: {lhs} vs {rhs}"
    assert abs(d["residual"]) < 1e-12

    # (b) discrete forecasts aligned to bin edges → binned form reconciles exactly too
    rng = np.random.default_rng(7)
    levels = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    pp = rng.choice(levels, size=5000)
    yy = (rng.uniform(size=pp.size) < pp).astype(float)
    db = C.murphy_decomposition(pp, yy, n_bins=10)
    assert abs(db["residual"]) < 1e-9, f"binned residual too large: {db['residual']}"

    # (c) binned continuous always reconciles via the residual term
    dc = C.murphy_decomposition(p, y, n_bins=10)
    recon = dc["reliability"] - dc["resolution"] + dc["uncertainty"] + dc["residual"]
    assert abs(dc["brier"] - recon) < 1e-12


# ── Gate 2: reliability separates well- vs over-confident ─────────────────────────

def test_reliability_separates_overconfident():
    pw, yw = _well_calibrated()
    po, yo = _overconfident()
    ece_w, ece_o = C.ece(pw, yw), C.ece(po, yo)
    rel_w = C.murphy_decomposition(pw, yw, n_bins=10)["reliability"]
    rel_o = C.murphy_decomposition(po, yo, n_bins=10)["reliability"]
    assert ece_o > ece_w + 0.05, f"ECE failed to separate: well={ece_w:.3f} over={ece_o:.3f}"
    assert rel_o > rel_w, f"reliability term failed to separate: {rel_w:.4f} vs {rel_o:.4f}"
    assert C.brier_score(po, yo) > C.brier_score(pw, yw)


def test_reliability_table_has_wilson_bands():
    p, y = _well_calibrated()
    tbl = C.reliability_table(p, y, n_bins=10)
    assert {"prob_bucket", "mean_pred_prob", "actual_freq", "n", "ci_lo", "ci_hi"} \
        <= set(tbl.columns)
    # bands must bracket the observed frequency
    assert (tbl["ci_lo"] <= tbl["actual_freq"]).all()
    assert (tbl["actual_freq"] <= tbl["ci_hi"]).all()


# ── Gate 3: recalibration helps out-of-sample ─────────────────────────────────────

def test_recalibration_reduces_error():
    po, yo = _overconfident(n=6000, seed=3)
    cut = po.size // 2
    base_ece = C.ece(po[cut:], yo[cut:])
    for backend in ("numpy", "auto"):
        iso = C.isotonic_recalibrate(po[:cut], yo[:cut], po[cut:], backend=backend)
        plt = C.platt_recalibrate(po[:cut], yo[:cut], po[cut:], backend=backend)
        assert C.ece(iso, yo[cut:]) < base_ece, f"isotonic({backend}) did not help"
        assert C.ece(plt, yo[cut:]) < base_ece, f"platt({backend}) did not help"


# ── Gate 4: markets layer / Z / ledger / CLI ─────────────────────────────────────

def test_markets_layer():
    assert abs(C.implied_prob_decimal([2.0])[0] - 0.5) < 1e-12
    assert abs(C.implied_prob_american([-110])[0] - (110 / 210)) < 1e-12
    fair = C.devig([0.55, 0.52])
    assert abs(fair.sum() - 1.0) < 1e-12
    rng = np.random.default_rng(5)
    n = 2000
    true_p = rng.uniform(0.2, 0.8, n)
    outcome = (rng.uniform(size=n) < true_p).astype(float)
    implied = np.clip(true_p + 0.03, 0, 0.999)
    res = C.realized_edge(true_p, implied, outcome)
    assert res["n_bets"] >= 0


def test_spiegelhalter_z():
    pw, yw = _well_calibrated(seed=11)
    zw = C.spiegelhalter_z(pw, yw)
    assert abs(zw["z"]) < 3.0, f"well-calibrated should not be rejected: Z={zw['z']:.2f}"
    pb = np.clip(pw + 0.2, 0, 1)
    zb = C.spiegelhalter_z(pb, yw)
    assert abs(zb["z"]) > abs(zw["z"]), "bias should raise |Z|"


def test_ledger_reader_all_resolution_paths():
    with tempfile.TemporaryDirectory() as d:
        ledger = _write_ledger(Path(d))
        expected_brier = ((0.7 - 1) ** 2 + 0.4 ** 2) / 2

        # (1) explicit ledger_dir argument
        df = C.load_scored_forecasts(ledger_dir=ledger)
        assert list(df["id"]) == ["sf-2026-001", "sf-2026-002"]
        assert abs(C.brier_score(df["prob"], df["label"]) - expected_brier) < 1e-12

        # (2) $SF_LEDGER_DIR
        old = os.environ.get("SF_LEDGER_DIR")
        os.environ["SF_LEDGER_DIR"] = str(ledger)
        try:
            assert len(C.load_scored_forecasts()) == 2
        finally:
            if old is None:
                os.environ.pop("SF_LEDGER_DIR", None)
            else:
                os.environ["SF_LEDGER_DIR"] = old

        # (3) book name via a caller-supplied books mapping
        df3 = C.load_scored_forecasts("mybook", books={"mybook": ledger})
        assert len(df3) == 2

        # (4) nothing resolvable → refuses to guess
        try:
            C.resolve_ledger_dir()
            raise AssertionError("resolve_ledger_dir should have raised")
        except SystemExit:
            pass


def test_cli_score_json_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ledger = _write_ledger(Path(d))
        out = subprocess.run(
            [sys.executable, "-m", "lemma.calibrate.cli",
             "--ledger", str(ledger), "score", "--json"],
            capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        payload = json.loads(out.stdout)
        assert payload["n"] == 2
        assert abs(payload["brier"] - ((0.7 - 1) ** 2 + 0.4 ** 2) / 2) < 1e-12


def test_skills_installer_lists_bundle():
    out = subprocess.run(
        [sys.executable, "-m", "lemma.calibrate.skills", "list"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "calibrate:" in out.stdout
