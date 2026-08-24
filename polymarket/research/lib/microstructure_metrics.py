"""Shared microstructure metric definitions — the ONE owned code path.

Why this module exists
----------------------
The pre-Alvaro pipeline trust audit ([[pm_prealvaro_pipeline_trust_audit_findings]]
Finding 1) proved that the lineage's most-quoted result — "TOB hit rate collapsed
73.7% -> 36.0% OOS" — was a false comparison between two *different* metrics computed
by two different scripts with no shared owner:

* ``dali_block_a13_tob_imbalance.py`` computed hit rate on **overlapping event rows,
  excluding zero-move windows from the denominator** (a *conditional* hit rate:
  "when the mid moves at all, does it move our way?").
* ``dali_block_a0c_holdout_retest.py`` (Retest C) computed hit rate on
  **non-overlapping 5s episodes, counting zero-move episodes as misses**
  ("how often does an independent episode pay within the horizon?").

Both are legitimate quantities; comparing one to the other is not. The standing rule
from [[pm_dali_workflow_revision_decision]] Tier 2 item 1: hit rate, directional
return, and non-overlap episode counting are defined HERE, once, and imported by
both discovery and any retest. No metric is redefined at a call site.

Definitions
-----------
**Hit rate** (:func:`hit_rate`): fraction of observations where the realized move
agrees in sign with the signal. Zero-move handling is an explicit, named parameter:

* ``zero_move="miss"`` (default) — a zero realized move counts as a miss. This is
  the executable-relevance reading: a signal that fires and the price never moves
  did not pay you. Conservative; matches the Retest-C construction. This is the
  default because a gate/monetization decision should not silently condition away
  the no-move outcome.
* ``zero_move="exclude"`` — zero-move observations are dropped from the numerator
  AND denominator. This is the *conditional* descriptive reading ("when it moves,
  which way?"); matches the A13 discovery construction. Use it only when the
  question is explicitly conditional, and always report the zero-move share
  alongside (:func:`zero_move_share`).

Rows where the signal itself has zero sign are excluded under BOTH settings — a
signless observation can neither hit nor miss.

**Directional return** (:func:`directional_return_bps`): mean of
``sign(signal) * realized_return_bps`` over all observations passed in (zero-move
rows contribute 0, zero-signal rows contribute 0 — they dilute, deliberately: this
is the per-observation expectation, not a conditional mean).

**Non-overlap episodes** (:func:`non_overlap_episode_mask`): the Retest-C rule.
Events are ordered by time (tie-break: larger ``|signal|`` first); an event starts
an episode only if its timestamp is strictly after the previous episode's blocking
horizon (``entry + horizon``); the episode then blocks everything up to
``entry + horizon``. Overlapping evaluation windows of a persistent state are the
other half of the 73.7% disaster — one persistent-imbalance state spanned ~40
overlapping rows in A13's n=299,864.

Provenance of the constructions replicated here (documentary, class C):
``scripts/dali_block_a13_tob_imbalance.py::hit_and_directional_return`` (lines
265-276) and ``scripts/dali_block_a0c_holdout_retest.py::simulate_tob_hits`` /
``tob_hit_events`` (lines 842-898). The reproduction gate lives in
``tests/test_microstructure_metrics.py``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

ZeroMove = Literal["miss", "exclude"]

#: Default zero-move handling. "miss" is the conservative, executable-relevance
#: reading (a fired signal with no realized move did not pay). See module docstring.
DEFAULT_ZERO_MOVE: ZeroMove = "miss"


def _as_float_array(x) -> np.ndarray:
    out = np.asarray(x, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {out.shape}")
    return out


def hit_rate(
    signal,
    realized_return,
    *,
    zero_move: ZeroMove = DEFAULT_ZERO_MOVE,
) -> tuple[float, int]:
    """Sign-agreement hit rate with explicit zero-move handling.

    Parameters
    ----------
    signal : array-like
        Signal values (only the sign is used). Zero-sign rows are always excluded.
    realized_return : array-like
        Realized directional move over the evaluation horizon (any unit; only the
        sign is used). NaN/inf rows are excluded.
    zero_move : ``"miss"`` (default) or ``"exclude"``
        ``"miss"``: a zero move counts as a miss (denominator keeps the row).
        ``"exclude"``: zero-move rows are dropped from numerator and denominator
        (the *conditional* hit rate).

    Returns
    -------
    (hit_rate, n_evaluated) — ``hit_rate`` is NaN when ``n_evaluated == 0``.
    """
    if zero_move not in ("miss", "exclude"):
        raise ValueError(f"zero_move must be 'miss' or 'exclude', got {zero_move!r}")
    x = _as_float_array(signal)
    y = _as_float_array(realized_return)
    if x.shape != y.shape:
        raise ValueError(f"signal/return length mismatch: {x.shape} vs {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    sx = np.sign(x)
    base = finite & (sx != 0)
    if zero_move == "exclude":
        base = base & (np.sign(y) != 0)
    n = int(base.sum())
    if n == 0:
        return float("nan"), 0
    hits = sx[base] * y[base] > 0
    return float(hits.mean()), n


def zero_move_share(realized_return) -> float:
    """Share of finite observations whose realized move is exactly zero.

    Report this next to any ``zero_move="exclude"`` hit rate so the conditional
    number cannot masquerade as an unconditional one.
    """
    y = _as_float_array(realized_return)
    finite = np.isfinite(y)
    if not finite.any():
        return float("nan")
    return float((y[finite] == 0).mean())


def directional_return_bps(signal, realized_return_bps) -> tuple[float, int]:
    """Mean of ``sign(signal) * realized_return_bps`` over finite observations.

    Zero-signal and zero-move rows contribute 0 (they dilute the mean toward
    zero on purpose — this is the per-observation expectation of following the
    signal, not a conditional mean).

    Returns ``(mean_bps, n)``; NaN mean when no finite rows.
    """
    x = _as_float_array(signal)
    y = _as_float_array(realized_return_bps)
    if x.shape != y.shape:
        raise ValueError(f"signal/return length mismatch: {x.shape} vs {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(finite.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.sign(x[finite]) * y[finite]).mean()), n


def non_overlap_episode_mask(
    times_ns,
    horizon_ns: int,
    *,
    abs_signal=None,
) -> np.ndarray:
    """Boolean mask selecting the non-overlapping episode starts (Retest-C rule).

    Events are processed in order of ``(times_ns, -abs_signal)`` — earliest first,
    ties broken by larger ``|signal|`` (so the strongest coincident signal claims
    the episode). An event is kept iff its timestamp is **strictly greater** than
    the previous kept event's ``time + horizon_ns``; a kept event blocks everything
    up to and including ``time + horizon_ns``.

    Parameters
    ----------
    times_ns : array-like of int64
        Event timestamps in nanoseconds (need not be pre-sorted).
    horizon_ns : int
        Blocking horizon in nanoseconds (e.g. ``5 * 10**9`` for the 5s rule).
    abs_signal : optional array-like
        Tie-break magnitude. Omitted -> ties keep input order.

    Returns
    -------
    Boolean mask aligned with the INPUT order (True = episode start).
    """
    t = np.asarray(times_ns, dtype=np.int64)
    if t.ndim != 1:
        raise ValueError(f"expected 1-D times, got shape {t.shape}")
    horizon_ns = int(horizon_ns)
    if horizon_ns < 0:
        raise ValueError(f"horizon_ns must be >= 0, got {horizon_ns}")
    if abs_signal is not None:
        a = _as_float_array(abs_signal)
        if a.shape != t.shape:
            raise ValueError(f"times/abs_signal length mismatch: {t.shape} vs {a.shape}")
        order = np.lexsort((-a, t))  # time asc, then larger |signal| first
    else:
        order = np.argsort(t, kind="stable")
    keep = np.zeros(len(t), dtype=bool)
    next_available = np.iinfo(np.int64).min
    for idx in order:
        ts = t[idx]
        if ts <= next_available:
            continue
        keep[idx] = True
        next_available = ts + horizon_ns
    return keep


def non_overlap_episode_count(times_ns, horizon_ns: int, *, abs_signal=None) -> int:
    """Number of independent (non-overlapping) episodes under the Retest-C rule."""
    return int(non_overlap_episode_mask(times_ns, horizon_ns, abs_signal=abs_signal).sum())
