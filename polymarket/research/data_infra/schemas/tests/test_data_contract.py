"""
Negative + positive + drift + gate tests for the Polymarket data-contract layer.

Run from polymarket/research/:
    PYTHONPATH=. uv run python -m pytest data_infra/schemas/tests -q

Covers what the crypto suite cannot: lowercase-0x address enforcement, the L2
best_ask>=best_bid / spread-consistency cross-column rules, append-only shard
integrity on the real pm_trades contract, and lookahead on fills.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from data_infra.schemas import core
from data_infra.schemas.contracts import PM_L2_BBA, PM_TRADES


# ── builders matching the real schemas ───────────────────────────────────────
def clean_trades(n: int = 200) -> pl.DataFrame:
    t0 = datetime(2025, 1, 1)
    return pl.DataFrame({
        "timestamp": pl.Series([t0 + timedelta(seconds=i) for i in range(n)], dtype=pl.Datetime("us")),
        "market_id": ["627493"] * n,
        "condition_id": ["0x" + "a" * 64] * n,
        "neg_risk": [False] * n,
        "maker": ["0x" + "b" * 40] * n,
        "taker": ["0x" + "c" * 40] * n,
        "maker_asset_id": ["0"] * n,
        "taker_asset_id": ["112969404185540697506005921597"] * n,
        "usd_amount": [1.0 + i * 0.01 for i in range(n)],
        "token_amount": [2.0] * n,
        "price": [0.4 + 0.2 * (i % 2) for i in range(n)],
        "maker_side": (["BUY", "SELL"] * n)[:n],
        "transaction_hash": ["0x" + "d" * 64] * n,
    })


def clean_bba(n: int = 200) -> pl.DataFrame:
    t0_ms = 1782203350000
    bid = np.clip(0.3 + np.random.default_rng(0).normal(0, 0.01, n), 0.01, 0.98)
    ask = bid + 0.002
    # received_at must be >= server timestamp_ms (data is received after it happens)
    recv = [datetime.fromtimestamp((t0_ms + i) / 1000 + 0.05, tz=timezone.utc)
            .isoformat().replace("+00:00", "Z") for i in range(n)]
    return pl.DataFrame({
        "timestamp_ms": [t0_ms + i for i in range(n)],
        "received_at": recv,
        "received_ns": [3873701380000000 + i * 1000 for i in range(n)],
        "universe": ["politics_negrisk"] * n,
        "asset_id": ["88403451835230950222"] * n,
        "market": ["0x" + "e" * 60] * n,
        "best_bid": bid, "best_ask": ask, "spread": ask - bid,
    })


def run(contract, df, tmp_path, *, as_of=None, set_reference=False):
    p = tmp_path / "shard.parquet"
    df.write_parquet(p)
    return core.run_contract(contract, [p], as_of=as_of, monitor_dir=tmp_path / "mon",
                             set_reference=set_reference)


def errs(res):
    return {v.check for v in res.violations if v.severity == "error"}


# ── positive ─────────────────────────────────────────────────────────────────
def test_clean_trades_pass(tmp_path):
    res = run(PM_TRADES, clean_trades(), tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert res.passed, res.violations


def test_clean_bba_pass(tmp_path):
    res = run(PM_L2_BBA, clean_bba(), tmp_path)
    assert res.passed, res.violations


# ── lowercase 0x addresses (the PM-specific invariant) ───────────────────────
def test_uppercase_address_caught(tmp_path):
    df = clean_trades()
    df[0, "maker"] = "0xABCDEF" + "0" * 34   # checksummed/upper -> not lowercase
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "address" in errs(res)


def test_non_hex_address_caught(tmp_path):
    df = clean_trades()
    df[1, "taker"] = "0xZZZZ" + "0" * 36
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "address" in errs(res)


def test_missing_0x_prefix_caught(tmp_path):
    df = clean_trades()
    df[2, "transaction_hash"] = "d" * 64       # no 0x prefix
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "address" in errs(res)


# ── schema / range / dtype ───────────────────────────────────────────────────
def test_price_out_of_range_caught(tmp_path):
    df = clean_trades()
    df[0, "price"] = 1.5
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "schema" in errs(res)


def test_bad_side_value_caught(tmp_path):
    df = clean_trades()
    df[0, "maker_side"] = "HOLD"
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "schema" in errs(res)


def test_negative_usd_caught(tmp_path):
    df = clean_trades()
    df[0, "usd_amount"] = -1.0
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "schema" in errs(res)


def test_missing_column_caught(tmp_path):
    df = clean_trades().drop("price")
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "schema" in errs(res)


# ── finite ───────────────────────────────────────────────────────────────────
def test_inf_caught(tmp_path):
    df = clean_trades()
    df[0, "token_amount"] = float("inf")
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert not res.passed and "finite" in errs(res)


# ── lookahead (no future-dated fills) ────────────────────────────────────────
def test_future_fill_caught(tmp_path):
    df = clean_trades(50)
    ts = df["timestamp"].to_list()
    ts[-1] = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
    df = df.with_columns(pl.Series("timestamp", ts, dtype=pl.Datetime("us")))
    res = run(PM_TRADES, df, tmp_path, as_of=datetime.now(timezone.utc))
    assert not res.passed and "lookahead" in errs(res)


# ── L2 cross-column rules ────────────────────────────────────────────────────
def test_bba_crossed_book_caught(tmp_path):
    df = clean_bba()
    df[5, "best_ask"] = df[5, "best_bid"] - 0.01   # ask < bid, both > 0
    res = run(PM_L2_BBA, df, tmp_path)
    assert not res.passed and "row_rule" in errs(res)


def test_bba_inconsistent_spread_caught(tmp_path):
    df = clean_bba()
    df[7, "spread"] = 0.5   # != ask-bid
    res = run(PM_L2_BBA, df, tmp_path)
    assert not res.passed and "row_rule" in errs(res)


# ── append-only on the real pm_trades contract ───────────────────────────────
def test_trades_append_only_mutation_caught(tmp_path):
    mon = tmp_path / "mon"
    s1 = tmp_path / "trades_delta_shard_a.parquet"
    s2 = tmp_path / "trades_delta_shard_b.parquet"
    clean_trades(100).write_parquet(s1)
    clean_trades(100).write_parquet(s2)
    assert core.check_append_only(PM_TRADES, [s1, s2], mon) == []  # records manifest
    clean_trades(100).with_columns((pl.col("price") * 0 + 0.9).alias("price")).write_parquet(s1)
    v = core.check_append_only(PM_TRADES, [s1, s2], mon)
    assert any(x.check == "append_only" for x in v), v


# ── drift ────────────────────────────────────────────────────────────────────
def test_synthetic_price_drift_detected(tmp_path):
    import dataclasses
    # clone without append-only so re-writing the same shard path (to shift the
    # distribution) is not itself flagged as an in-place mutation.
    c = dataclasses.replace(PM_TRADES, name="pm_trades_drifttest", append_only=None)
    base = clean_trades(500)
    run(c, base, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc), set_reference=True)
    shifted = base.with_columns(pl.Series("price", np.clip(
        np.random.default_rng(0).uniform(0.8, 0.99, base.height), 0, 1)))
    r2 = run(c, shifted, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    price = [d for d in r2.drift if d.column == "price"][0]
    assert price.flag in ("large", "moderate", "ks_significant"), price
    assert r2.passed  # drift never blocks


# ── fail-closed gate ──────────────────────────────────────────────────────────
def test_enforce_raises_on_failure(tmp_path):
    df = clean_trades()
    df[0, "maker"] = "0xUPPER" + "0" * 34
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    with pytest.raises(core.DataContractError):
        core.enforce(res, PM_TRADES, tmp_path / "mon", mode="enforce")


def test_enforce_off_does_not_raise(tmp_path):
    df = clean_trades()
    df[0, "maker"] = "0xUPPER" + "0" * 34
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    core.enforce(res, PM_TRADES, tmp_path / "mon", mode="off")  # no raise


def test_report_written(tmp_path):
    df = clean_trades()
    df[0, "price"] = 2.0
    res = run(PM_TRADES, df, tmp_path, as_of=datetime(2025, 6, 1, tzinfo=timezone.utc))
    rep = core.write_report(res, PM_TRADES, tmp_path / "mon")
    assert "Data-contract report" in rep.read_text()
