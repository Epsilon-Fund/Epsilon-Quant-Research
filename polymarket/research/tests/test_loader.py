"""Tests for epsilon_data — including the anti-drift test.

Run from polymarket/research with the package importable:
    PYTHONPATH=. .venv/Scripts/python -m pytest tests/test_loader.py -q

The anti-drift test is the important one: it proves the loader returns exactly what a raw
parquet read returns. If the loader ever quietly diverges from the data, everything built on
it is wrong and nothing else would catch it.
"""
from __future__ import annotations
import hashlib
import pathlib

import pandas as pd
import pytest

import epsilon_data as ed
from epsilon_data import _internal as _i


def _sample_assets(n=5):
    cat = ed.catalog(apply_exclusions=False)
    cat = cat[cat["n_l1_events"] > 0].sort_values("n_l1_events", ascending=False)
    # a mix: busiest few + a small one
    picks = list(cat["asset_id"].head(n - 1)) + list(cat["asset_id"].tail(1))
    return list(dict.fromkeys(picks))


def _raw_l1(asset_id: str, universe: str) -> pd.DataFrame:
    glob = _i._uri("l1", f"universe={universe}", "*", "*.parquet")
    c = _i.con()
    try:
        return c.execute(
            f"SELECT * FROM read_parquet('{glob}') WHERE CAST(asset_id AS VARCHAR)='{asset_id}' "
            "ORDER BY timestamp_ms, received_ns"
        ).df()
    finally:
        c.close()


def _checksum(series: pd.Series) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(series.reset_index(drop=True), index=False).values.tobytes()).hexdigest()


def test_anti_drift_l1():
    """loader load_l1 == raw parquet read: row count, first & last row, checksum over mid."""
    assets = _sample_assets(5)
    assert assets, "no assets with l1 data"
    for aid in assets:
        uni = _i.token_row(aid)["universe"]
        raw = _raw_l1(aid, uni)
        got = ed.load_l1(aid)
        assert len(got) == len(raw), f"row count differs for {aid[:12]}: {len(got)} vs {len(raw)}"
        if len(raw) == 0:
            continue
        # same first/last row on the price columns
        for col in ("timestamp_ms", "received_ns", "best_bid", "best_ask", "mid"):
            assert got[col].iloc[0] == raw[col].iloc[0], f"first-row {col} differs for {aid[:12]}"
            assert got[col].iloc[-1] == raw[col].iloc[-1], f"last-row {col} differs for {aid[:12]}"
        # checksum over the price column
        assert _checksum(got["mid"]) == _checksum(raw["mid"]), f"mid checksum differs for {aid[:12]}"


def test_ids_are_strings():
    cat = ed.catalog(apply_exclusions=False)
    for col in ("asset_id", "condition_id"):
        assert cat[col].dtype == "string" or cat[col].map(type).eq(str).all(), f"{col} not strings"
    l1 = ed.load_l1(_sample_assets(1)[0])
    assert l1["asset_id"].map(type).eq(str).all()


def test_resolve_roundtrip():
    row = ed.catalog(apply_exclusions=False).iloc[0]
    assert ed.resolve(row["asset_id"]) == row["asset_id"]
    assert ed.resolve(row["path"]) == row["asset_id"]


def test_ts_column_is_utc():
    l1 = ed.load_l1(_sample_assets(1)[0])
    assert "ts" in l1.columns
    assert str(l1["ts"].dt.tz) == "UTC"


def test_load_pair_two_sides_sum_near_one():
    # pick a resolved politics market (Yes/No) with l1 on both sides
    cat = ed.catalog(universe="politics_negrisk", apply_exclusions=False)
    cat = cat[cat["n_l1_events"] > 500]
    cid = cat["condition_id"].value_counts()
    cid = cid[cid == 2].index[0]
    pair = ed.load_pair(cid)
    assert set(pair.columns) == {0, 1}
    both = pair.dropna()
    assert len(both) > 0
    s = (both[0] + both[1])
    # complementary: median sum within a few cents of 1
    assert abs(s.median() - 1.0) < 0.05, f"pair median sum {s.median():.3f} not ~1"


def test_search_finds_fed():
    r = ed.search("fed")
    assert len(r) > 0
    assert r["question"].str.lower().str.contains("fed").any()


def test_reconciliation_holds():
    rec = ed.reconciliation()
    assert rec["match"].all(), f"reconciliation broke: {rec.to_dict('records')}"


def test_audit_market_structure():
    aid = _sample_assets(1)[0]
    r = ed.audit_market(aid)
    assert r.verdict in ("looks fine", "worth a look", "recommend excluding")
    assert r.checks and all(c.level in ("ok", "note", "bad") for c in r.checks)
    assert r.reason
    # audit NEVER writes
    d = ed.to_dict() if hasattr(ed, "to_dict") else None


def test_audit_never_writes(tmp_path):
    import epsilon_data as ed
    from epsilon_data import config as cfg
    aid = _sample_assets(1)[0]
    before = pathlib.Path(cfg.data_root()) / "exclusions.csv"
    txt0 = before.read_text() if before.exists() else ""
    ed.audit_market(aid)  # must not touch the file
    txt1 = before.read_text() if before.exists() else ""
    assert txt0 == txt1, "audit_market wrote to exclusions.csv — it must never do that"


def test_write_exclusion_roundtrip(tmp_path):
    import shutil, epsilon_data as ed
    from epsilon_data import config as cfg, _internal as _i
    root = pathlib.Path(cfg.data_root())
    excl = root / "exclusions.csv"
    backup = excl.read_text() if excl.exists() else None
    try:
        aid = _sample_assets(1)[0]
        ed.write_exclusion(aid, "market", "unit-test", "pytest")
        assert aid in ed.catalog(apply_exclusions=False).set_index("asset_id").index
        # excluded by default now
        assert aid not in set(ed.catalog(apply_exclusions=True).asset_id)
    finally:
        if backup is not None:
            excl.write_text(backup)
        else:
            excl.unlink(missing_ok=True)
        _i._exclusions_cached.cache_clear()
