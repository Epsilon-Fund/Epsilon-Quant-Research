"""WIRED-BUT-DORMANT overfitting apparatus — Deflated Sharpe / CPCV + an OOS split.

Per the task brief, the apparatus is **built and wired** but reports **no verdict** on the
symmetric quoter, because both gates are *vacuous* on a single-config, 0-parameter strategy:

* **Deflated Sharpe (trial-deflation)** deflates the best-of-N selected Sharpe by the
  expected max Sharpe under the null of N trials. With **N = 1** (one fixed config, nothing
  selected), ``expected_max_sharpe_null(1, ·) = 0`` → the haircut SR\\* is **0** → the deflated
  Sharpe equals the raw Sharpe. There is literally nothing to deflate. It becomes meaningful only
  when a **parameterized** strategy is selected across trials (Task 5).
* **CPCV / PBO** needs a ``T × N`` candidate-return matrix (``N ≥ 2`` configs). With one config
  there is no matrix → PBO is undefined.
* A strict **OOS split** only *bites* once something is **fit** (``ProbQueue.f`` at Join 2, or
  strategy params at Task 5). On a fixed quoter, IS and OOS differ only by sampling, so a "fail"
  there would be noise, not overfitting (``brain/CODEX.md`` realism rule 1).

So this module exposes the apparatus (it imports the shared, engine-agnostic
``infrastructure/validation/overfitting_audit.py`` — pure numpy/pandas, the same harness the
momentum book uses) and a function that *demonstrates* the vacuity numerically, but the report
prints "DORMANT — no verdict", as instructed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Repo root = three levels up from polymarket/research/mm_eval/. overfitting_audit.py is shared,
# engine-agnostic validation infra (NOT the crypto trading code) — the task directs us to wire it.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_overfitting_audit():
    """Import the shared ``infrastructure.validation.overfitting_audit`` (path-shimmed). May raise."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from infrastructure.validation import overfitting_audit  # noqa: E402
    return overfitting_audit


@dataclass(frozen=True)
class DormantOverfitting:
    """The honest 'apparatus wired, no verdict' object for the symmetric quoter."""

    wired: bool
    n_trials: int                       # 1 — a single fixed config
    sr_star_haircut: float              # the deflation haircut — 0 when N=1 (nothing to deflate)
    pbo_available: bool                 # False — needs N>=2 configs
    oos_split_available: bool           # True — apparatus exists, but only bites once FIT
    note: str

    def to_markdown(self) -> str:
        return (
            f"**Overfitting apparatus: WIRED, DORMANT (no verdict).** "
            f"n_trials = {self.n_trials} → Deflated-Sharpe haircut SR\\* = {self.sr_star_haircut:.3f} "
            f"(nothing to deflate on a 0-parameter quoter); PBO/CPCV available = {self.pbo_available} "
            f"(needs ≥2 candidate configs); OOS-split apparatus = {self.oos_split_available} "
            f"(only bites once a parameter is FIT — ProbQueue.f at Join 2, or strategy params at "
            f"Task 5). {self.note}"
        )


def dormant_status(selected_window_returns=None) -> DormantOverfitting:
    """Demonstrate (don't assert) the vacuity: with N=1 the DSR haircut is exactly 0.

    If a per-window return series is supplied, we *call the real harness* with ``n_trials=1`` and
    ``var_trial_sr=0`` to show ``SR* = 0`` numerically (the apparatus runs; it just has nothing to
    deflate). No verdict object is returned — by instruction.
    """
    haircut = 0.0
    note = "Becomes live when Task 5 selects a parameterized strategy across trials."
    try:
        oa = load_overfitting_audit()
        # expected_max_sharpe_null(1, var) == 0 for any var (n_trials<=1 guard) — show it.
        haircut = float(oa.expected_max_sharpe_null(1, 1.0))
        if selected_window_returns is not None and len(selected_window_returns) >= 3:
            # A single-config DSR call: SR* is 0, so deflated == observed (demonstration only).
            res = oa.deflated_sharpe_ratio(
                np.asarray(selected_window_returns, dtype=float),
                n_trials=1, periods_per_year=365.0, var_trial_sr=0.0,
            )
            note += (f" (demo on pooled per-window PnL: SR*_ann={res.sr_star_ann:.3f}, "
                     f"deflated==raw by construction).")
    except Exception as exc:  # pragma: no cover - import/availability guard
        return DormantOverfitting(False, 1, 0.0, False, True,
                                  f"shared overfitting_audit unavailable ({exc}); apparatus not loaded.")
    return DormantOverfitting(True, 1, haircut, pbo_available=False, oos_split_available=True, note=note)


@dataclass(frozen=True)
class OOSSplit:
    """A time-ordered IS/OOS split apparatus — dormant on a fixed quoter (reported, not gated)."""

    is_frac: float
    split_ts: int
    n_is: int
    n_oos: int
    note: str = ("OOS split is wired but DORMANT: it only bites once a parameter is fit. On the "
                 "0-parameter symmetric quoter an IS/OOS gap is sampling noise, not overfitting.")


def build_oos_split(fill_ts: np.ndarray, *, is_frac: float = 0.7, span: tuple[int, int] | None = None) -> OOSSplit:
    """Construct (but do not gate on) a time-ordered IS/OOS split over a token's fills."""
    ts = np.asarray(fill_ts, dtype=np.int64)
    if ts.size == 0:
        return OOSSplit(is_frac, 0, 0, 0)
    t0, t1 = span or (int(ts.min()), int(ts.max()))
    split_ts = int(t0 + (t1 - t0) * is_frac)
    return OOSSplit(is_frac, split_ts, int((ts <= split_ts).sum()), int((ts > split_ts).sum()))
