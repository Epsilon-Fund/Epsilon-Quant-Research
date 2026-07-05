"""End-to-end demo of rigorkit-calibrate on two synthetic forecasters.

Runs in a couple of seconds with no arguments and no data files:

    python examples/demo.py

What it shows, in order:

1. Two seeded synthetic forecasters over the same kind of world:
   WELL-CALIBRATED (outcomes drawn at the stated probability) and
   OVER-CONFIDENT (true chance is mild, reported probabilities are extreme).
2. The full scorecard for each — Brier + Murphy decomposition, log-loss,
   ECE/MCE, Spiegelhalter's Z — and what separates them.
3. Isotonic recalibration fixing the over-confident forecaster OUT-OF-SAMPLE
   (fit on one half, applied to the other).
4. The markets layer: de-vigging book odds and tracking realized edge over
   resolved markets.
5. If matplotlib is installed (`pip install "rigorkit-calibrate[plot]"`), a
   reliability diagram PNG comparing all three curves.

Everything is seeded and deterministic. Only numpy + pandas are required.
"""
from __future__ import annotations

import numpy as np

from rigorkit.calibrate import (
    brier_score,
    devig,
    ece,
    implied_prob_decimal,
    isotonic_recalibrate,
    log_loss,
    murphy_decomposition,
    realized_edge,
    spiegelhalter_z,
)

SEED = 7
N = 6000


def make_forecasters(n=N, seed=SEED):
    """One calibrated and one over-confident forecaster over Bernoulli worlds."""
    rng = np.random.default_rng(seed)
    # well-calibrated: outcomes happen at exactly the stated probability
    p_good = rng.uniform(0.02, 0.98, n)
    y_good = (rng.uniform(size=n) < p_good).astype(float)
    # over-confident: the world is mild (p in [0.30, 0.70]) but the forecaster
    # reports 3x-amplified deviations from 50% — too sure in both directions
    true_p = rng.uniform(0.30, 0.70, n)
    y_over = (rng.uniform(size=n) < true_p).astype(float)
    p_over = np.clip(0.5 + (true_p - 0.5) * 3.0, 0.01, 0.99)
    return (p_good, y_good), (p_over, y_over)


def scorecard(name: str, p, y) -> None:
    m = murphy_decomposition(p, y, n_bins=10)
    z = spiegelhalter_z(p, y)
    verdict = "REJECTED (|Z|>1.96)" if abs(z["z"]) > 1.96 else "not rejected"
    print(f"\n{name}")
    print(f"  Brier {m['brier']:.4f} = reliability {m['reliability']:.4f}"
          f" - resolution {m['resolution']:.4f} + uncertainty {m['uncertainty']:.4f}"
          f" (+ residual {m['residual']:.1e})")
    print(f"  log-loss {log_loss(p, y):.4f}   ECE {ece(p, y):.4f}   "
          f"Spiegelhalter Z {z['z']:+.2f} -> calibration {verdict}")


def main() -> None:
    (pg, yg), (po, yo) = make_forecasters()
    print(f"two synthetic forecasters, {N} forecasts each (seed {SEED})")

    # --- 1+2. score both -------------------------------------------------------
    scorecard("WELL-CALIBRATED forecaster (outcomes drawn at stated prob):", pg, yg)
    scorecard("OVER-CONFIDENT forecaster (mild world, extreme reports):", po, yo)
    print("\n  read: the over-confident forecaster is punished in the RELIABILITY"
          "\n  term (calibration), while its RESOLUTION (discrimination) stays high"
          "\n  — the decomposition separates the two failure modes.")

    # --- 3. recalibration, honestly out-of-sample ------------------------------
    cut = N // 2
    p_fixed = isotonic_recalibrate(po[:cut], yo[:cut], po[cut:])
    print(f"\nisotonic recalibration (fit on half 1, applied to half 2):")
    print(f"  ECE  {ece(po[cut:], yo[cut:]):.4f} -> {ece(p_fixed, yo[cut:]):.4f}")
    print(f"  Brier {brier_score(po[cut:], yo[cut:]):.4f} -> "
          f"{brier_score(p_fixed, yo[cut:]):.4f}")

    # --- 4. markets layer -------------------------------------------------------
    rng = np.random.default_rng(SEED + 1)
    n_mkt = 2000
    true_p = rng.uniform(0.2, 0.8, n_mkt)
    outcome = (rng.uniform(size=n_mkt) < true_p).astype(float)
    # a book quoting the truth shaded 3pp against the bettor, with vig
    implied = np.clip(true_p + 0.03, 0.01, 0.99)
    res = realized_edge(true_p, implied, outcome)
    fair = devig(implied_prob_decimal([2.10, 1.85]))
    print(f"\nmarkets layer:")
    print(f"  devig([1/2.10, 1/1.85]) -> {fair.round(4).tolist()} (sums to 1)")
    print(f"  model-vs-shaded-book over {res['n_markets']} resolved markets: "
          f"{res['n_bets']} bets, expected edge {res['expected_edge']:+.4f}, "
          f"realized edge {res['realized_edge']:+.4f}")

    # --- 5. reliability diagram (optional extra) --------------------------------
    try:
        from rigorkit.calibrate import reliability_diagram
        out = reliability_diagram(
            {"well-calibrated": (pg, yg),
             "over-confident": (po, yo),
             "recalibrated": (np.asarray(p_fixed), yo[cut:])},
            "reliability_demo.png",
        )
        print(f"\nreliability diagram written to {out} — the over-confident curve"
              f"\nbows below the diagonal at high p and above it at low p; the"
              f"\nrecalibrated curve sits back on the diagonal.")
    except ImportError:
        print("\n(matplotlib not installed — skipping the reliability diagram; "
              "pip install \"rigorkit-calibrate[plot]\")")


if __name__ == "__main__":
    main()
