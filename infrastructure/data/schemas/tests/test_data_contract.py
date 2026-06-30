"""
Negative + positive + drift + gate tests for the crypto data-contract layer.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python -m pytest infrastructure/data/schemas/tests -q

Every injected violation (schema/dtype/range/monotonicity/cadence/finite/
lookahead/append-only) must be CAUGHT; a known-good shard must produce ~0
false positives; synthetic drift must be detected.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from infrastructure.data.schemas import core
from infrastructure.data.schemas.contracts import CRYPTO_OHLCV_DAILY


# ── helpers ──────────────────────────────────────────────────────────────────
def clean_ohlcv(n: int = 400, start="2020-01-01") -> pl.DataFrame:
    t0 = datetime.fromisoformat(start).replace(tzinfo=None)
    times = [t0 + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 1.0
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    vol = np.abs(rng.normal(1e6, 2e5, n)) + 1.0
    return pl.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": vol, "Time": pl.Series(times, dtype=pl.Datetime("ns")),
    })


def run(df: pl.DataFrame, tmp_path, *, as_of=None, set_reference=False):
    p = tmp_path / "shard.parquet"
    df.write_parquet(p)
    return core.run_contract(CRYPTO_OHLCV_DAILY, [p], as_of=as_of,
                             monitor_dir=tmp_path / "mon", set_reference=set_reference)


def checks(res):
    return {v.check for v in res.violations if v.severity == "error"}


# ── positive / false-positive ────────────────────────────────────────────────
def test_clean_passes_no_false_positives(tmp_path):
    res = run(clean_ohlcv(), tmp_path)
    assert res.passed, res.violations
    assert [v for v in res.violations if v.severity == "error"] == []


def test_runs_under_time_budget(tmp_path):
    res = run(clean_ohlcv(2000), tmp_path)
    assert res.elapsed_s < 2.0


# ── schema / dtype / range ───────────────────────────────────────────────────
def test_dtype_violation_caught(tmp_path):
    df = clean_ohlcv().with_columns(pl.col("Open").cast(pl.Utf8))
    res = run(df, tmp_path)
    assert not res.passed and "schema" in checks(res)


def test_negative_price_range_caught(tmp_path):
    df = clean_ohlcv()
    df[0, "Close"] = -5.0
    res = run(df, tmp_path)
    assert not res.passed and "schema" in checks(res)


def test_negative_volume_caught(tmp_path):
    df = clean_ohlcv()
    df[3, "Volume"] = -1.0
    res = run(df, tmp_path)
    assert not res.passed and "schema" in checks(res)


def test_missing_column_caught(tmp_path):
    df = clean_ohlcv().drop("Close")
    res = run(df, tmp_path)
    assert not res.passed and "schema" in checks(res)


# ── monotonicity ─────────────────────────────────────────────────────────────
def test_out_of_order_timestamps_caught(tmp_path):
    df = clean_ohlcv()
    times = df["Time"].to_list()
    times[10], times[11] = times[11], times[10]
    df = df.with_columns(pl.Series("Time", times, dtype=pl.Datetime("ns")))
    res = run(df, tmp_path)
    assert not res.passed and "monotonic_ts" in checks(res)


def test_duplicate_timestamps_caught(tmp_path):
    df = clean_ohlcv()
    times = df["Time"].to_list()
    times[20] = times[19]
    df = df.with_columns(pl.Series("Time", times, dtype=pl.Datetime("ns")))
    res = run(df, tmp_path)
    assert not res.passed
    # duplicate ts trips strict monotonic and/or cadence-dup; both are errors
    assert checks(res) & {"monotonic_ts", "cadence"}


# ── cadence (missing bars: warn under tolerance, error beyond) ────────────────
def test_small_gap_is_warning_not_failure(tmp_path):
    df = clean_ohlcv(400)
    df = df.filter(pl.arange(0, df.height) != 200)  # drop one bar -> 0.25%
    res = run(df, tmp_path)
    assert res.passed  # warning only
    assert any(v.check == "cadence" and v.severity == "warn" for v in res.violations)


def test_large_gap_fails(tmp_path):
    df = clean_ohlcv(400)
    keep = [i for i in range(400) if not (150 <= i < 170)]  # drop 20 bars -> ~5%
    df = df[keep]
    res = run(df, tmp_path)
    assert not res.passed and "cadence" in checks(res)


# ── finite ───────────────────────────────────────────────────────────────────
def test_nan_caught(tmp_path):
    df = clean_ohlcv()
    df[5, "High"] = float("nan")
    res = run(df, tmp_path)
    assert not res.passed and "finite" in checks(res)


def test_inf_caught(tmp_path):
    df = clean_ohlcv()
    df[7, "Low"] = float("inf")
    res = run(df, tmp_path)
    assert not res.passed and "finite" in checks(res)


# ── cross-column OHLC sanity ─────────────────────────────────────────────────
def test_high_below_low_caught(tmp_path):
    df = clean_ohlcv()
    df[9, "High"] = df[9, "Low"] - 1.0
    res = run(df, tmp_path)
    assert not res.passed and "row_rule" in checks(res)


# ── lookahead ────────────────────────────────────────────────────────────────
def test_future_dated_row_caught(tmp_path):
    df = clean_ohlcv(100)
    future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
    times = df["Time"].to_list()
    times[-1] = future
    df = df.with_columns(pl.Series("Time", times, dtype=pl.Datetime("ns")))
    res = run(df, tmp_path, as_of=datetime.now(timezone.utc))
    assert not res.passed and "lookahead" in checks(res)


# ── drift ────────────────────────────────────────────────────────────────────
def test_synthetic_drift_detected(tmp_path):
    base = clean_ohlcv(500)
    r1 = run(base, tmp_path, set_reference=True)  # establish reference
    assert all(d.flag == "baseline" for d in r1.drift)
    # shift Volume distribution 3x -> large PSI / KS
    shifted = base.with_columns((pl.col("Volume") * 3.0 + 5e5).alias("Volume"))
    r2 = run(shifted, tmp_path)
    vol = [d for d in r2.drift if d.column == "Volume"][0]
    assert vol.flag in ("large", "moderate", "ks_significant"), vol
    assert r2.passed  # drift never blocks the gate


def test_identical_data_no_drift_flag(tmp_path):
    base = clean_ohlcv(500)
    run(base, tmp_path, set_reference=True)
    r2 = run(base, tmp_path)
    vol = [d for d in r2.drift if d.column == "Volume"][0]
    assert vol.flag == "stable", vol  # regression: identical data must not flag


# ── engine: PSI / KS units ───────────────────────────────────────────────────
def test_ks_identical_pvalue_one():
    a = np.random.default_rng(0).normal(0, 1, 2000)
    d, p = core.ks_2samp(a, a)
    assert d == 0.0 and p == 1.0


def test_ks_shift_significant():
    rng = np.random.default_rng(1)
    d, p = core.ks_2samp(rng.normal(0, 1, 3000), rng.normal(0.6, 1, 3000))
    assert p < 1e-10 and d > 0.2


def test_psi_monotone():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    assert core.psi(ref, ref) < 0.01
    assert core.psi(ref, rng.normal(1.0, 1, 5000)) > core.psi(ref, rng.normal(0.2, 1, 5000))


# ── append-only (engine API; the crypto OHLCV contract has none by design) ────
def test_append_only_shard_mutation_caught(tmp_path):
    from infrastructure.data.schemas.core import AppendOnlyRule, Contract
    c = Contract(name="t_shard", description="", append_only=AppendOnlyRule(mode="shard"))
    mon = tmp_path / "mon"
    s1 = tmp_path / "s1.parquet"
    s2 = tmp_path / "s2.parquet"
    clean_ohlcv(50).write_parquet(s1)
    clean_ohlcv(50, start="2021-01-01").write_parquet(s2)
    # first pass records the manifest
    assert core.check_append_only(c, [s1, s2], mon) == []
    # mutate s1 in place (same row count, different bytes) -> must be caught
    clean_ohlcv(50).with_columns((pl.col("Close") + 1).alias("Close")).write_parquet(s1)
    v = core.check_append_only(c, [s1, s2], mon)
    assert any(x.check == "append_only" for x in v), v


def test_append_only_shard_disappearance_caught(tmp_path):
    from infrastructure.data.schemas.core import AppendOnlyRule, Contract
    c = Contract(name="t_shard2", description="", append_only=AppendOnlyRule(mode="shard"))
    mon = tmp_path / "mon"
    s1 = tmp_path / "s1.parquet"
    s2 = tmp_path / "s2.parquet"
    clean_ohlcv(30).write_parquet(s1)
    clean_ohlcv(30, start="2021-01-01").write_parquet(s2)
    core.check_append_only(c, [s1, s2], mon)
    v = core.check_append_only(c, [s1], mon)  # s2 vanished
    assert any(x.check == "append_only" for x in v)


def test_append_only_row_superset_violation_caught(tmp_path):
    from infrastructure.data.schemas.core import AppendOnlyRule, Contract
    c = Contract(name="t_row", description="",
                 append_only=AppendOnlyRule(mode="row_superset", key_columns=("Time",)))
    mon = tmp_path / "mon"
    f = tmp_path / "growing.parquet"
    df = clean_ohlcv(100)
    df.write_parquet(f)
    assert core.check_append_only(c, [f], mon) == []
    df.head(100).write_parquet(f)              # same -> ok
    assert core.check_append_only(c, [f], mon) == []
    df.head(90).write_parquet(f)               # dropped 10 prior rows -> violation
    v = core.check_append_only(c, [f], mon)
    assert any(x.check == "append_only" for x in v), v


# ── fail-closed gate ──────────────────────────────────────────────────────────
def test_enforce_raises_on_failure(tmp_path):
    df = clean_ohlcv()
    df[0, "Close"] = -1.0
    res = run(df, tmp_path)
    with pytest.raises(core.DataContractError):
        core.enforce(res, CRYPTO_OHLCV_DAILY, tmp_path / "mon", mode="enforce")


def test_enforce_warn_does_not_raise(tmp_path):
    df = clean_ohlcv()
    df[0, "Close"] = -1.0
    res = run(df, tmp_path)
    core.enforce(res, CRYPTO_OHLCV_DAILY, tmp_path / "mon", mode="warn")  # no raise


def test_enforce_pass_does_not_raise(tmp_path):
    res = run(clean_ohlcv(), tmp_path)
    core.enforce(res, CRYPTO_OHLCV_DAILY, tmp_path / "mon", mode="enforce")


def test_report_written_and_readable(tmp_path):
    df = clean_ohlcv()
    df[0, "Close"] = -1.0
    res = run(df, tmp_path)
    rep = core.write_report(res, CRYPTO_OHLCV_DAILY, tmp_path / "mon")
    text = rep.read_text()
    assert "Data-contract report" in text and "FAIL" in text and "## Summary" in text
