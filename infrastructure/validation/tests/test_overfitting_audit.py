"""
Unit tests for infrastructure/validation/overfitting_audit.py.

Two end-to-end fixtures anchor the suite:
- OVERFIT: pure-noise search (no true edge anywhere). Selecting the IS-best
  config must be flagged: deflated Sharpe <= 0, PBO ~ 0.5, verdict != PASS.
- GENUINE: heterogeneous real edge (configs differ in true mean, the best are
  genuinely good). The audit must let it through: DSR prob > 0.5, PBO < 0.5,
  Reality-Check p small, verdict PASS.

Run from repo root:  ./.venv/bin/python -m pytest infrastructure/validation/tests/ -q
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from overfitting_audit import (  # noqa: E402
    _norm_cdf,
    _norm_ppf,
    annualized_sharpe,
    deflated_sharpe_ratio,
    effective_n_trials,
    expected_max_sharpe_null,
    make_null_ohlcv,
    min_track_record_length,
    pbo_cscv,
    per_bar_sharpe,
    probabilistic_sharpe_ratio,
    run_overfitting_audit,
    sharpe_moments,
    stationary_bootstrap_indices,
    stationary_bootstrap_sharpe_ci,
    synthetic_null_mc,
    whites_reality_check,
)

PPY = 365.0  # daily crypto convention


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

# T sized so the OOS half (~2k bars) matches a real CPCV daily-path length —
# short OOS windows widen the Sharpe CI enough to flunk the materiality gate
# even for a real edge.
T, N = 4096, 120
T_IS = T // 2


@pytest.fixture(scope="module")
def overfit_search():
    """Pure noise: N configs, zero true mean. Selection = best IS (first half)."""
    rng = np.random.default_rng(7)
    m = rng.normal(0.0, 0.01, size=(T, N))
    is_sharpe = m[:T_IS].mean(0) / m[:T_IS].std(0, ddof=1)
    best = int(np.argmax(is_sharpe))
    oos = pd.Series(m[T_IS:, best])
    return m, oos, best


@pytest.fixture(scope="module")
def genuine_search():
    """
    Genuine edge: every config has a real positive mean (mu_i in
    [0.001, 0.0025], best ann Sharpe ~ 4.8 at sd=0.01) with enough spread that
    IS ranking carries real information. Note the DSR haircut grows with
    cross-trial SR dispersion, so a fixture whose configs span no-edge-to-edge
    would (correctly) be haircut into the ground — the pass case needs a
    strong common edge, which is what a healthy strategy family looks like.
    """
    rng = np.random.default_rng(11)
    mus = np.linspace(0.001, 0.0025, N)
    m = rng.normal(0.0, 0.01, size=(T, N)) + mus
    is_sharpe = m[:T_IS].mean(0) / m[:T_IS].std(0, ddof=1)
    best = int(np.argmax(is_sharpe))
    oos = pd.Series(m[T_IS:, best])
    return m, oos, best


# ──────────────────────────────────────────────────────────────────────────────
# Normal-distribution helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_norm_ppf_known_values():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.025) == pytest.approx(-1.959964, abs=1e-5)


def test_norm_cdf_inverts_ppf():
    for p in (0.01, 0.25, 0.5, 0.9, 0.999):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-7)


def test_norm_ppf_rejects_bounds():
    with pytest.raises(ValueError):
        _norm_ppf(0.0)
    with pytest.raises(ValueError):
        _norm_ppf(1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_sharpe_matches_stack_convention():
    # mean/std(ddof=1) * sqrt(ppy) — same as performance_metrics.py
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.003, -0.001])
    expected = r.mean() / r.std() * math.sqrt(PPY)  # pandas std is ddof=1
    assert annualized_sharpe(r, PPY) == pytest.approx(float(expected), rel=1e-12)


def test_flat_series_sharpe_is_zero():
    assert per_bar_sharpe(np.zeros(10)) == 0.0


def test_sharpe_moments_normal_data():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=100_000)
    sr, skew, kurt, t = sharpe_moments(r)
    assert t == 100_000
    assert skew == pytest.approx(0.0, abs=0.05)
    assert kurt == pytest.approx(3.0, abs=0.1)


def test_too_few_observations_raises():
    with pytest.raises(ValueError):
        per_bar_sharpe([0.01, 0.02])


def test_psr_positive_sr_above_half():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, size=1000)
    assert probabilistic_sharpe_ratio(r, 0.0) > 0.5


def test_psr_matches_hand_formula_for_normal_returns():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, size=500)
    sr, skew, kurt, t = sharpe_moments(r)
    z = (sr - 0.02) * math.sqrt(t - 1) / math.sqrt(
        1 - skew * sr + (kurt - 1) / 4 * sr**2)
    assert probabilistic_sharpe_ratio(r, 0.02) == pytest.approx(_norm_cdf(z), abs=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Expected-max / N_eff / MinTRL
# ──────────────────────────────────────────────────────────────────────────────

def test_expected_max_sharpe_hand_value():
    # V=1, N=10: (1-g)*z(1-1/10) + g*z(1-1/(10e))
    g = 0.5772156649015329
    expected = (1 - g) * _norm_ppf(1 - 1 / 10) + g * _norm_ppf(1 - 1 / (10 * math.e))
    assert expected_max_sharpe_null(10, 1.0) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_monotone_and_degenerate():
    assert expected_max_sharpe_null(1, 1.0) == 0.0
    assert expected_max_sharpe_null(100, 0.0) == 0.0
    a = expected_max_sharpe_null(10, 0.01)
    b = expected_max_sharpe_null(400, 0.01)
    c = expected_max_sharpe_null(400, 0.04)
    assert 0 < a < b < c


def test_effective_n_trials_extremes():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, size=(500, 1))
    identical = np.repeat(base, 20, axis=1)
    assert effective_n_trials(identical) == pytest.approx(1.0, abs=0.05)
    independent = rng.normal(0, 0.01, size=(500, 20))
    assert effective_n_trials(independent) > 15
    with pytest.raises(ValueError):
        effective_n_trials(np.zeros(10))


def test_min_track_record_length():
    rng = np.random.default_rng(4)
    r = rng.normal(0.002, 0.01, size=1000)
    assert min_track_record_length(r, 0.05) > 0
    assert math.isinf(min_track_record_length(r, 10.0))
    # bigger gap to benchmark -> shorter required track record
    assert min_track_record_length(r, 0.01) < min_track_record_length(r, 0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Stationary bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def test_bootstrap_indices_valid_and_deterministic():
    rng = np.random.default_rng(5)
    idx = stationary_bootstrap_indices(100, 10.0, rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100
    # determinism: same fresh rng state -> same draw
    a = stationary_bootstrap_indices(100, 10.0, np.random.default_rng(9))
    b = stationary_bootstrap_indices(100, 10.0, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_bootstrap_sharpe_ci_covers_truth():
    rng = np.random.default_rng(6)
    true_sr_ann = 0.001 / 0.01 * math.sqrt(PPY)  # ~1.91
    r = rng.normal(0.001, 0.01, size=2000)
    lo, hi, draws = stationary_bootstrap_sharpe_ci(r, PPY, n_boot=400, seed=0)
    assert lo < hi
    assert lo < true_sr_ann < hi
    assert draws.shape == (400,)


# ──────────────────────────────────────────────────────────────────────────────
# DSR
# ──────────────────────────────────────────────────────────────────────────────

def test_dsr_requires_dispersion_input():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(np.random.default_rng(0).normal(size=100), 400, PPY)


def test_dsr_overfit_fails_gate(overfit_search):
    m, oos, _ = overfit_search
    sds = m.std(0, ddof=1)
    trial_sharpes = m.mean(0) / sds
    d = deflated_sharpe_ratio(oos, N, PPY, trial_sharpes=trial_sharpes)
    # no true edge: OOS SR ~ 0, haircut SR* > 0 -> deflated < 0, prob < 0.5
    assert d.sr_star_ann > 0
    assert d.deflated_sr_ann < 0
    assert d.dsr_prob < 0.5
    assert not d.passes


def test_dsr_genuine_passes_gate(genuine_search):
    m, oos, _ = genuine_search
    trial_sharpes = m.mean(0) / m.std(0, ddof=1)
    d = deflated_sharpe_ratio(oos, N, PPY, trial_sharpes=trial_sharpes)
    assert d.deflated_sr_ann > 0
    assert d.dsr_prob > 0.95
    assert d.passes


def test_dsr_n_eff_override_recorded(genuine_search):
    m, oos, _ = genuine_search
    trial_sharpes = m.mean(0) / m.std(0, ddof=1)
    d = deflated_sharpe_ratio(oos, N, PPY, trial_sharpes=trial_sharpes, n_eff=10)
    assert d.n_eff == 10
    assert d.n_trials == N
    d_full = deflated_sharpe_ratio(oos, N, PPY, trial_sharpes=trial_sharpes)
    assert d.sr_star_ann < d_full.sr_star_ann  # fewer effective trials -> smaller haircut


# ──────────────────────────────────────────────────────────────────────────────
# PBO / CSCV
# ──────────────────────────────────────────────────────────────────────────────

def test_pbo_overfit_not_confidently_low(overfit_search):
    # Pure noise: IS ranking carries no OOS info, so PBO is centered near 0.5
    # but is itself high-variance across realisations (the CSCV combinations
    # are dependent). The robust guarantee is that noise never looks
    # confidently skilful: PBO stays well above the 0.2 "good" bar, and the
    # median logit shows no real IS->OOS rank persistence.
    m, _, _ = overfit_search
    res = pbo_cscv(m, n_blocks=16)
    assert res.pbo > 0.2
    assert res.n_combinations == 12870
    assert abs(res.logit_quartiles[1]) < 1.5


def test_pbo_genuine_low(genuine_search):
    m, _, _ = genuine_search
    res = pbo_cscv(m, n_blocks=16)
    assert res.pbo < 0.2
    assert res.passes
    assert res.p_oos_loss < 0.1
    # sensitivity runs agree on the order of magnitude
    for v in res.sensitivity.values():
        assert v < 0.35


def test_pbo_validation_errors():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((10, 5)), n_blocks=16)


def test_pbo_handles_nans(genuine_search):
    m, _, _ = genuine_search
    m2 = m.copy()
    m2[5, 3] = np.nan
    res = pbo_cscv(m2, n_blocks=8, sensitivity_blocks=())
    assert np.isfinite(res.pbo)


# ──────────────────────────────────────────────────────────────────────────────
# Reality Check
# ──────────────────────────────────────────────────────────────────────────────

def test_rc_genuine_significant(genuine_search):
    m, _, _ = genuine_search
    res = whites_reality_check(m, n_boot=400, seed=0)
    assert res.p_value < 0.05
    assert res.significant
    assert res.mc_se > 0


def test_rc_overfit_not_significant(overfit_search):
    m, _, _ = overfit_search
    res = whites_reality_check(m, n_boot=400, seed=0)
    assert res.p_value > 0.05
    assert not res.significant


def test_rc_reports_both_variants(overfit_search):
    m, _, _ = overfit_search
    res = whites_reality_check(m, n_boot=200, seed=1)
    assert 0.0 <= res.p_value <= 1.0
    assert 0.0 <= res.p_value_raw <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end audit + verdict
# ──────────────────────────────────────────────────────────────────────────────

def test_audit_overfit_flagged(overfit_search):
    m, oos, _ = overfit_search
    v = run_overfitting_audit(
        selected_oos_returns=oos, n_trials=N, periods_per_year=PPY,
        trial_returns=m, label="synthetic overfit", n_boot=400, seed=0,
    )
    assert v.verdict == "FLAG-FOR-REVIEW"
    assert not v.gates["dsr_gt_0"]
    assert not v.gates["haircut_lci_material"]


def test_audit_genuine_passes(genuine_search):
    m, oos, _ = genuine_search
    v = run_overfitting_audit(
        selected_oos_returns=oos, n_trials=N, periods_per_year=PPY,
        trial_returns=m, label="synthetic genuine edge", n_boot=400, seed=0,
    )
    assert v.verdict == "PASS"
    assert all(v.gates.values())
    assert v.rc.significant


def test_audit_without_matrix_is_insufficient():
    rng = np.random.default_rng(8)
    v = run_overfitting_audit(
        selected_oos_returns=rng.normal(0.001, 0.01, 500),
        n_trials=400, periods_per_year=PPY, n_boot=100,
    )
    assert v.verdict == "INSUFFICIENT-DATA"
    assert v.pbo is None and v.rc is None


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic-null Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(rets, start=100.0):
    close = start * np.cumprod(1 + rets)
    idx = pd.date_range("2021-01-01", periods=len(close), freq="D")
    open_ = np.concatenate(([start], close[:-1]))
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.01,   # valid bars: H above both O and C
        "Low": np.minimum(open_, close) * 0.99,
        "Close": close,
        "Volume": np.full(len(close), 1000.0),
    }, index=idx)


def _ar1_returns(phi, t, seed, mu=0.0005, sigma=0.015):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, t)
    r = np.empty(t)
    r[0] = eps[0]
    for i in range(1, t):
        r[i] = mu + phi * (r[i - 1] - mu) + eps[i]
    return r


def _one_bar_momentum_sharpe(df):
    # long next bar iff this bar's close-to-close return was positive
    r = df["Close"].pct_change().values[1:]
    strat = r[1:] * (r[:-1] > 0)
    sd = strat.std(ddof=1)
    return float(strat.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def test_null_ohlcv_validity_and_determinism():
    df = _make_ohlcv(_ar1_returns(0.3, 800, seed=0))
    s1 = make_null_ohlcv(df, 10.0, rng=np.random.default_rng(5))
    s2 = make_null_ohlcv(df, 10.0, rng=np.random.default_rng(5))
    assert s1.index.equals(df.index) and len(s1) == len(df)
    assert s1.equals(s2)  # deterministic given the rng state
    # bar-internal OHLC relations survive resampling
    assert (s1["High"] >= s1[["Open", "Close"]].max(axis=1) - 1e-12).all()
    assert (s1["Low"] <= s1[["Open", "Close"]].min(axis=1) + 1e-12).all()
    assert (s1["Close"] > 0).all()
    assert s1["Close"].iloc[0] == df["Close"].iloc[0]


def test_null_ohlcv_preserves_drift_and_demean_removes_it():
    df = _make_ohlcv(_ar1_returns(0.0, 3000, seed=1, mu=0.002))
    real_drift = np.mean(np.log(df["Close"].values[1:] / df["Close"].values[:-1]))
    drifts = [np.mean(np.log(s["Close"].values[1:] / s["Close"].values[:-1]))
              for s in (make_null_ohlcv(df, 10.0, rng=np.random.default_rng(k)) for k in range(20))]
    assert np.mean(drifts) == pytest.approx(real_drift, abs=0.0005)
    dm = make_null_ohlcv(df, 10.0, rng=np.random.default_rng(0), demean=True)
    dm_drift = np.mean(np.log(dm["Close"].values[1:] / dm["Close"].values[:-1]))
    assert abs(dm_drift) < abs(real_drift) / 4


def test_null_ohlcv_joint_preserves_cross_correlation():
    rng = np.random.default_rng(2)
    common = rng.normal(0, 0.01, 1500)
    ra = common + rng.normal(0, 0.004, 1500)
    rb = common + rng.normal(0, 0.004, 1500)
    dfa, dfb = _make_ohlcv(ra), _make_ohlcv(rb)
    out = make_null_ohlcv({"a": dfa, "b": dfb}, 10.0, rng=np.random.default_rng(3))
    sa = out["a"]["Close"].pct_change().dropna()
    sb = out["b"]["Close"].pct_change().dropna()
    assert sa.corr(sb) > 0.7  # joint blocks keep the assets co-moving
    # misaligned indices must be rejected
    with pytest.raises(ValueError):
        make_null_ohlcv({"a": dfa, "b": dfb.iloc[:-1]}, 10.0)


def test_synthetic_null_mc_detects_real_timing_signal():
    # AR(1) momentum: a 1-bar trend rule has real timing skill; IID-resampled
    # nulls (mean_block_len=1) destroy the autocorrelation it exploits.
    df = _make_ohlcv(_ar1_returns(0.5, 2000, seed=4))
    res = synthetic_null_mc(df, _one_bar_momentum_sharpe, n_paths=120,
                            mean_block_len=1.0, seed=2000)
    assert res.real_stat > res.q95
    assert res.passes
    assert res.percentile > 0.95


def test_synthetic_null_mc_no_signal_not_flagged():
    # IID returns: the same rule has no timing skill; the real stat should sit
    # inside the null distribution, not above its 95th percentile.
    df = _make_ohlcv(_ar1_returns(0.0, 2000, seed=6))
    res = synthetic_null_mc(df, _one_bar_momentum_sharpe, n_paths=120,
                            mean_block_len=1.0, seed=3000)
    assert not res.passes
    assert 0.0 < res.percentile < 0.95
    assert res.n_paths == 120


def test_markdown_report_fragment(genuine_search):
    m, oos, _ = genuine_search
    v = run_overfitting_audit(
        selected_oos_returns=oos, n_trials=N, periods_per_year=PPY,
        trial_returns=m, label="md test", n_boot=200, seed=0,
    )
    md = v.to_markdown()
    for token in ("Verdict", "Deflated Sharpe", "PBO", "Reality-Check",
                  "Post-haircut", "md test"):
        assert token in md
