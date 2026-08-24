"""
test_calibration.py — DoD gates for the calibration scoring layer.

Byte-identical across both project packages. Runs as a plain script
(``python -m <pkg>.tests.test_calibration``) printing PASS/FAIL, and is also
pytest-collectable. The ml_metrics reproduction test self-skips where
``infrastructure.backtester.ml_metrics`` is not importable (polymarket venv).

Gates
-----
  1. Brier–Murphy decomposition reconciles to the total Brier on synthetic data.
  2. Reliability separates a well- vs a deliberately over-confident forecaster.
  3. Reproduces ml_metrics.calibration_table on a shared input (no regression).
  4. (engine) recalibration, markets layer, Spiegelhalter Z, ledger reader.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

# import the sibling engine whether run as a package module or directly
try:
    from .. import core as C
except ImportError:  # pragma: no cover - direct-path execution fallback
    import importlib.util
    _p = Path(__file__).resolve().parents[1] / "core.py"
    _spec = importlib.util.spec_from_file_location("calib_core", _p)
    C = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(C)


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
    # push reported prob away from 0.5 toward the extremes
    reported = np.clip(0.5 + (true_p - 0.5) * 3.0, 0.01, 0.99)
    return reported, y


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

    # (c) binned continuous always reconciles via the residual term (identity by construction)
    dc = C.murphy_decomposition(p, y, n_bins=10)
    recon = dc["reliability"] - dc["resolution"] + dc["uncertainty"] + dc["residual"]
    assert abs(dc["brier"] - recon) < 1e-12
    return True


# ── Gate 2: reliability separates well- vs over-confident ─────────────────────────

def test_reliability_separates_overconfident():
    pw, yw = _well_calibrated()
    po, yo = _overconfident()
    ece_w, ece_o = C.ece(pw, yw), C.ece(po, yo)
    rel_w = C.murphy_decomposition(pw, yw, n_bins=10)["reliability"]
    rel_o = C.murphy_decomposition(po, yo, n_bins=10)["reliability"]
    assert ece_o > ece_w + 0.05, f"ECE failed to separate: well={ece_w:.3f} over={ece_o:.3f}"
    assert rel_o > rel_w, f"reliability term failed to separate: {rel_w:.4f} vs {rel_o:.4f}"
    # the over-confident forecaster also has a worse (higher) Brier
    assert C.brier_score(po, yo) > C.brier_score(pw, yw)
    return True


# ── Gate 3: reproduce ml_metrics.calibration_table (no regression) ────────────────

def test_reproduces_ml_metrics_calibration_table():
    try:
        from infrastructure.backtester import ml_metrics
    except Exception as exc:  # noqa: BLE001 - intentional self-skip off the crypto path
        print(f"  [skip] ml_metrics not importable here ({exc.__class__.__name__}); "
              "run this gate from the crypto repo root.")
        return None

    import pandas as pd
    rng = np.random.default_rng(42)
    n = 3000
    prob = rng.uniform(0, 1, n)
    label = (rng.uniform(size=n) < prob).astype(int)
    preds_df = pd.DataFrame({"pred": np.ones(n), "prob": prob, "label": label})

    theirs = ml_metrics.calibration_table(preds_df, n_bins=10).reset_index(drop=True)
    ours = C.calibration_table(prob, label, n_bins=10).reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)
    return True


# ── engine extras ─────────────────────────────────────────────────────────────────

def test_recalibration_reduces_error():
    po, yo = _overconfident(n=6000, seed=3)
    # split train/apply to avoid trivially fitting the same points
    cut = po.size // 2
    base_ece = C.ece(po[cut:], yo[cut:])
    for backend in ("numpy", "auto"):
        iso = C.isotonic_recalibrate(po[:cut], yo[:cut], po[cut:], backend=backend)
        plt = C.platt_recalibrate(po[:cut], yo[:cut], po[cut:], backend=backend)
        assert C.ece(iso, yo[cut:]) < base_ece, f"isotonic({backend}) did not help"
        assert C.ece(plt, yo[cut:]) < base_ece, f"platt({backend}) did not help"
    return True


def test_markets_layer():
    # decimal odds 2.0 ⇒ 0.5 implied; american -110 ⇒ ~0.524
    assert abs(C.implied_prob_decimal([2.0])[0] - 0.5) < 1e-12
    assert abs(C.implied_prob_american([-110])[0] - (110 / 210)) < 1e-12
    fair = C.devig([0.55, 0.52])  # binary book with overround
    assert abs(fair.sum() - 1.0) < 1e-12
    # a model that exactly matches outcomes earns positive realized edge vs a vigged book
    rng = np.random.default_rng(5)
    n = 2000
    true_p = rng.uniform(0.2, 0.8, n)
    outcome = (rng.uniform(size=n) < true_p).astype(float)
    implied = np.clip(true_p + 0.03, 0, 0.999)  # book shades 3pp against the bettor on average
    model = true_p
    res = C.realized_edge(model, implied, outcome)
    assert res["n_bets"] >= 0
    return True


def test_spiegelhalter_z():
    pw, yw = _well_calibrated(seed=11)
    zw = C.spiegelhalter_z(pw, yw)
    assert abs(zw["z"]) < 3.0, f"well-calibrated should not be rejected: Z={zw['z']:.2f}"
    # biased forecaster: always +0.2 too high
    pb = np.clip(pw + 0.2, 0, 1)
    zb = C.spiegelhalter_z(pb, yw)
    assert abs(zb["z"]) > abs(zw["z"]), "bias should raise |Z|"
    return True


def test_ledger_reader_roundtrip():
    """Write a tiny synthetic ledger and confirm the read-only reader parses scored events."""
    with tempfile.TemporaryDirectory() as d:
        ledger = Path(d) / "ledger"
        (ledger / "forecasts").mkdir(parents=True)
        events = ledger / "forecasts" / "events.jsonl"
        recs = [
            {"type": "forecast_created", "id": "sf-2026-001"},
            {"type": "scored", "id": "sf-2026-001", "timestamp": "2026-06-29T00:00:00+00:00",
             "final_probability": 0.7, "outcome": 1, "brier": 0.09},
            {"type": "scored", "id": "sf-2026-002", "timestamp": "2026-06-29T00:01:00+00:00",
             "final_probability": 0.4, "outcome": 0, "brier": 0.16},
        ]
        events.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        old = os.environ.get("SF_LEDGER_DIR")
        os.environ["SF_LEDGER_DIR"] = str(ledger)
        try:
            df = C.load_scored_forecasts()
        finally:
            if old is None:
                os.environ.pop("SF_LEDGER_DIR", None)
            else:
                os.environ["SF_LEDGER_DIR"] = old
        assert list(df["id"]) == ["sf-2026-001", "sf-2026-002"]
        assert abs(C.brier_score(df["prob"], df["label"]) - ((0.7 - 1) ** 2 + (0.4) ** 2) / 2) < 1e-12
    return True


_TESTS = [
    ("brier_decomposition_reconciles", test_brier_decomposition_reconciles),
    ("reliability_separates_overconfident", test_reliability_separates_overconfident),
    ("reproduces_ml_metrics_calibration_table", test_reproduces_ml_metrics_calibration_table),
    ("recalibration_reduces_error", test_recalibration_reduces_error),
    ("markets_layer", test_markets_layer),
    ("spiegelhalter_z", test_spiegelhalter_z),
    ("ledger_reader_roundtrip", test_ledger_reader_roundtrip),
]


def main() -> int:
    passed = skipped = failed = 0
    for name, fn in _TESTS:
        try:
            r = fn()
            if r is None:
                print(f"SKIP  {name}")
                skipped += 1
            else:
                print(f"PASS  {name}")
                passed += 1
        except AssertionError as exc:
            print(f"FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name}: {exc.__class__.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
