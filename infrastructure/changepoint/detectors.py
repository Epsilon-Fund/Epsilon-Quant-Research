"""
detectors.py — causal, lookahead-free online structural-break detectors.

Three detectors, one interface. Every detector consumes the series ONE bar at a
time via `update(x_t)`; its state at time t is a pure function of x_1..x_t, so
appending future bars can never change a past output (the no-lookahead
invariant, asserted in tests).

  * CUSUM        — two-sided cumulative-sum control chart. O(1)/bar. Live first line.
  * PageHinkley  — Page-Hinkley mean-shift test. O(1)/bar. Live first line.
  * BOCPD        — Bayesian Online Changepoint Detection (Adams & MacKay 2007):
                   a run-length posterior with a Student-t (Normal-Gamma
                   conjugate) predictive. O(Rmax)/bar, bounded. Gives a true
                   `change_prob` = P(run length just reset | data so far).

Pure numpy + stdlib `math` — no scipy/sklearn in the online path, so the live
hook stays dependency-light and portable. `ruptures` (OFFLINE only) lives in
offline.py and is never imported here.

This module is the crypto instance (infrastructure/). A Polymarket instance, if
ever needed, gets its OWN copy under polymarket/research/ — never cross-import
(brain/CODEX.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepResult:
    """One bar of detector output (the per-bar schema, sans timestamp)."""
    cp_flag: bool          # did a structural break fire at this bar?
    run_length_mode: int   # bars since the current regime began (MAP run length)
    change_prob: float     # P(a change just happened) in [0,1]
    statistic: float       # detector's raw test statistic (CUSUM/PH value, or P(r=0))


class _Welford:
    """Causal running mean/variance (Welford). Reset on a detected break so the
    baseline reflects only the CURRENT regime — never future data."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0


# ════════════════════════════════════════════════════════════════════════════
# CUSUM — two-sided cumulative sum on causally-standardised input
# ════════════════════════════════════════════════════════════════════════════
class CUSUM:
    """Two-sided CUSUM control chart.

    The input is standardised by a causal mean/std estimated over the CURRENT
    regime only (Welford, reset on each alarm). We accumulate
        S+ = max(0, S+ + z - k),   S- = max(0, S- - z - k)
    where z is the standardised observation and k is the slack (half the shift
    size we want to catch, in std units). An alarm fires when max(S+,S-) >= h,
    after which S± and the baseline reset so the detector adapts to the new level.

    Parameters
    ----------
    k        : slack / allowance in std units (default 0.5 → catches ~1σ shifts)
    h        : decision threshold (default 5.0; higher → fewer false alarms, more lag)
    warmup   : bars used to seed the baseline before any alarm can fire
    min_std  : floor on the std estimate to avoid divide-by-zero on flat input
    """

    name = "cusum"

    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 20,
                 min_std: float = 1e-9) -> None:
        self.k = float(k)
        self.h = float(h)
        self.warmup = int(warmup)
        self.min_std = float(min_std)
        self.reset_all()

    def reset_all(self) -> None:
        self.t = 0
        self.last_cp_t = 0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._stats = _Welford()

    def update(self, x: float) -> StepResult:
        self.t += 1
        x = float(x)
        std = max(self._stats.std, self.min_std)
        mean = self._stats.mean
        # standardise against the current-regime baseline (data <= t-1 only,
        # then fold x in — folding-in first or last does not leak the future)
        z = (x - mean) / std if self._stats.n >= 2 else 0.0
        self._stats.add(x)

        cp = False
        if self.t - self.last_cp_t >= self.warmup and self._stats.n >= 2:
            self.s_pos = max(0.0, self.s_pos + z - self.k)
            self.s_neg = max(0.0, self.s_neg - z - self.k)
            stat = max(self.s_pos, self.s_neg)
            if stat >= self.h:
                cp = True
                self.last_cp_t = self.t
                self.s_pos = 0.0
                self.s_neg = 0.0
                self._stats.reset()
        stat = max(self.s_pos, self.s_neg)
        change_prob = float(min(1.0, stat / self.h)) if self.h > 0 else 0.0
        return StepResult(cp, self.t - self.last_cp_t, change_prob, float(stat))


# ════════════════════════════════════════════════════════════════════════════
# Page-Hinkley — mean-shift test, two-sided
# ════════════════════════════════════════════════════════════════════════════
class PageHinkley:
    """Page-Hinkley test for an abrupt change in the mean.

    Tracks the cumulative deviation from the running mean with a magnitude
    tolerance `delta`; the test statistic is the gap between the cumulative
    sum and its running extremum. Two-sided (increase and decrease). O(1)/bar.

    Parameters
    ----------
    delta    : magnitude tolerance (ignore drifts smaller than this, in std units)
    lam      : alarm threshold λ (default 8.0)
    warmup   : bars to seed the baseline before alarms can fire
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.5, lam: float = 8.0, warmup: int = 20,
                 min_std: float = 1e-9) -> None:
        self.delta = float(delta)
        self.lam = float(lam)
        self.warmup = int(warmup)
        self.min_std = float(min_std)
        self.reset_all()

    def reset_all(self) -> None:
        self.t = 0
        self.last_cp_t = 0
        self.m_inc = 0.0   # cumulative (z - delta)
        self.min_inc = 0.0
        self.m_dec = 0.0   # cumulative (z + delta)
        self.max_dec = 0.0
        self._stats = _Welford()

    def update(self, x: float) -> StepResult:
        self.t += 1
        x = float(x)
        std = max(self._stats.std, self.min_std)
        mean = self._stats.mean
        z = (x - mean) / std if self._stats.n >= 2 else 0.0
        self._stats.add(x)

        cp = False
        stat = 0.0
        if self.t - self.last_cp_t >= self.warmup and self._stats.n >= 2:
            self.m_inc += z - self.delta
            self.min_inc = min(self.min_inc, self.m_inc)
            ph_inc = self.m_inc - self.min_inc          # detects upward shift

            self.m_dec += z + self.delta
            self.max_dec = max(self.max_dec, self.m_dec)
            ph_dec = self.max_dec - self.m_dec          # detects downward shift

            stat = max(ph_inc, ph_dec)
            if stat >= self.lam:
                cp = True
                self.last_cp_t = self.t
                self.m_inc = self.min_inc = 0.0
                self.m_dec = self.max_dec = 0.0
                self._stats.reset()
        change_prob = float(min(1.0, stat / self.lam)) if self.lam > 0 else 0.0
        return StepResult(cp, self.t - self.last_cp_t, change_prob, float(stat))


# ════════════════════════════════════════════════════════════════════════════
# BOCPD — Bayesian Online Changepoint Detection (Adams & MacKay 2007)
# ════════════════════════════════════════════════════════════════════════════
def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray,
                      alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Log posterior-predictive of a Normal-Gamma model (Murphy 2007, eq. 99):
    Student-t with df=2α, loc=μ, scale²=β(κ+1)/(ακ). Vectorised over run lengths."""
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    scale2 = np.maximum(scale2, 1e-300)
    z2 = (x - mu) ** 2 / scale2
    return (
        _lgamma_vec((df + 1.0) / 2.0)
        - _lgamma_vec(df / 2.0)
        - 0.5 * np.log(df * math.pi)
        - 0.5 * np.log(scale2)
        - ((df + 1.0) / 2.0) * np.log1p(z2 / df)
    )


_lgamma_vec = np.vectorize(math.lgamma, otypes=[float])


class BOCPD:
    """Bayesian Online Changepoint Detection with a constant hazard and a
    Normal-Gamma (unknown mean AND variance) observation model.

    Maintains the run-length posterior P(r_t | x_1:t). At each bar:
      * `change_prob` = P(r_t = 0 | x_1:t)  — posterior mass on "a change just
        happened" (a genuine probability, unlike CUSUM/PH's bounded statistic).
      * `run_length_mode` = argmax_r P(r_t = r)  — the MAP run length.
      * `cp_flag` fires when the MAP run length COLLAPSES (resets) after having
        grown past `min_segment` — the canonical BOCPD detection rule — or,
        if `prob_threshold` is set, when change_prob ≥ that threshold.

    The run-length vector is truncated to `rmax` (tail mass folded into the last
    cell) so cost is bounded O(rmax)/bar — the BOCPD is the richer second line;
    CUSUM/PH are the O(1) first line.

    Parameters
    ----------
    hazard_lambda : expected run length 1/H (constant hazard H = 1/λ). Default 250.
    mu0,kappa0,alpha0,beta0 : Normal-Gamma prior. Defaults are weak/standardised.
    rmax          : max tracked run length (truncation). Default 300.
    min_segment   : MAP run length must exceed this before a collapse counts as a break.
    prob_threshold: if not None, flag on change_prob ≥ threshold instead of MAP-collapse.
    """

    name = "bocpd"

    def __init__(self, hazard_lambda: float = 250.0, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0,
                 rmax: int = 300, min_segment: int = 10,
                 prob_threshold: float | None = None) -> None:
        self.hazard = 1.0 / float(hazard_lambda)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.rmax = int(rmax)
        self.min_segment = int(min_segment)
        self.prob_threshold = prob_threshold
        self.reset_all()

    def reset_all(self) -> None:
        self.t = 0
        self.last_cp_t = 0
        self._prev_mode = 0
        # run-length posterior, starts certain at r=0
        self.rl = np.array([1.0])
        # per-run-length Normal-Gamma params
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])

    def update(self, x: float) -> StepResult:
        self.t += 1
        x = float(x)

        logpred = _student_t_logpdf(x, self.mu, self.kappa, self.alpha, self.beta)
        pred = np.exp(logpred - logpred.max())   # scale-stable; constant cancels in norm

        growth = self.rl * pred * (1.0 - self.hazard)        # r -> r+1
        cp_mass = float(np.sum(self.rl * pred * self.hazard)) # all r -> 0

        new_rl = np.empty(len(self.rl) + 1)
        new_rl[0] = cp_mass
        new_rl[1:] = growth
        s = new_rl.sum()
        new_rl = new_rl / s if s > 0 else np.concatenate([[1.0], np.zeros(len(self.rl))])

        # update sufficient statistics: run length r+1 extends run r with x
        new_mu = np.empty_like(new_rl)
        new_kappa = np.empty_like(new_rl)
        new_alpha = np.empty_like(new_rl)
        new_beta = np.empty_like(new_rl)
        new_mu[0], new_kappa[0] = self.mu0, self.kappa0
        new_alpha[0], new_beta[0] = self.alpha0, self.beta0
        new_mu[1:] = (self.kappa * self.mu + x) / (self.kappa + 1.0)
        new_kappa[1:] = self.kappa + 1.0
        new_alpha[1:] = self.alpha + 0.5
        new_beta[1:] = self.beta + 0.5 * self.kappa * (x - self.mu) ** 2 / (self.kappa + 1.0)

        # truncate to rmax (fold tail mass into the last cell, keep its params)
        if len(new_rl) > self.rmax + 1:
            keep = self.rmax + 1
            tail = float(new_rl[keep:].sum())
            new_rl = new_rl[:keep].copy()
            new_rl[-1] += tail
            new_mu = new_mu[:keep]
            new_kappa = new_kappa[:keep]
            new_alpha = new_alpha[:keep]
            new_beta = new_beta[:keep]
            ssum = new_rl.sum()
            if ssum > 0:
                new_rl = new_rl / ssum

        self.rl, self.mu, self.kappa = new_rl, new_mu, new_kappa
        self.alpha, self.beta = new_alpha, new_beta

        mode = int(np.argmax(self.rl))
        change_prob = float(self.rl[0])

        if self.prob_threshold is not None:
            cp = change_prob >= self.prob_threshold
        else:
            # MAP run length collapsed after a real run -> a break
            cp = (mode < self._prev_mode) and (self._prev_mode >= self.min_segment)
        if cp:
            self.last_cp_t = self.t
        self._prev_mode = mode
        return StepResult(cp, mode, change_prob, change_prob)


DETECTORS = {"cusum": CUSUM, "page_hinkley": PageHinkley, "bocpd": BOCPD}


def make_detector(name: str, **kwargs):
    if name not in DETECTORS:
        raise KeyError(f"unknown detector '{name}'. Known: {sorted(DETECTORS)}")
    return DETECTORS[name](**kwargs)
