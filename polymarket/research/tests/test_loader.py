"""Tests for epsilon_data — including the anti-drift test.

Run from polymarket/research with the package importable:
    PYTHONPATH=. uv run pytest tests/test_loader.py -q

The anti-drift test is the important one: it proves the LOADER returns exactly what a raw
parquet read returns — same rows, same order, every column, l1 and trades. If the loader ever
quietly diverges from the data, everything built on it is wrong and nothing else would catch it.

What it is NOT: a check on the data itself. Its "raw" side is the built library, not the
capture, so it cannot see a dropped trade, a wrong dedup rule, part-boundary duplicates or a
mapping error. Those need the raw archive — see scripts/audit_checks/.
"""
from __future__ import annotations
import hashlib
import pathlib

import pandas as pd
import pytest

import epsilon_data as ed
from epsilon_data import _internal as _i


# How many tokens per universe the anti-drift test samples, and the seed that picks them.
# The sample is random-but-deterministic on purpose: the old test always took the 4 busiest
# tokens plus the single smallest, so drift confined to any of the other 30,767 passed green.
ANTI_DRIFT_N_PER_UNIVERSE = 20
ANTI_DRIFT_SEED = 3


def _sample_assets(n=5):
    cat = ed.catalog(apply_exclusions=False)
    cat = cat[cat["n_l1_events"] > 0].sort_values("n_l1_events", ascending=False)
    # a mix: busiest few + a small one
    picks = list(cat["asset_id"].head(n - 1)) + list(cat["asset_id"].tail(1))
    return list(dict.fromkeys(picks))


def _anti_drift_sample(n_per_universe=ANTI_DRIFT_N_PER_UNIVERSE, seed=ANTI_DRIFT_SEED):
    """n seeded-random tokens per universe, with both an l1 and a trades tape to compare."""
    cat = ed.catalog(apply_exclusions=False)
    cat = cat[(cat["n_l1_events"] > 0) & (cat["n_trades"] > 0)]
    frames = []
    for uni in sorted(cat["universe"].unique()):
        u = cat[cat["universe"] == uni]
        frames.append(u.sample(min(n_per_universe, len(u)), random_state=seed))
    return pd.concat(frames)[["asset_id", "universe"]]


def _raw_tape(kind: str, asset_id: str, universe: str) -> pd.DataFrame:
    """A plain parquet read of one token's tape — the loader-independent side of the comparison."""
    glob = _i._uri(kind, f"universe={universe}", "*", "*.parquet")
    c = _i.con()
    try:
        return c.execute(
            f"SELECT * FROM read_parquet('{glob}') WHERE CAST(asset_id AS VARCHAR)='{asset_id}' "
            "ORDER BY timestamp_ms, received_ns"
        ).df()
    finally:
        c.close()


def _raw_l1(asset_id: str, universe: str) -> pd.DataFrame:
    return _raw_tape("l1", asset_id, universe)


def _checksum(series: pd.Series) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(series.reset_index(drop=True), index=False).values.tobytes()).hexdigest()


def _frame_checksum(df: pd.DataFrame, cols) -> str:
    d = df[list(cols)].reset_index(drop=True)
    return hashlib.md5(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()


@pytest.mark.parametrize("kind, loader", [("l1", ed.load_l1), ("trades", ed.load_trades)])
def test_anti_drift(kind, loader):
    """The loader returns exactly what a raw parquet read returns.

    Widened from the original 5-token / mid-column-only / l1-only version (audit 2026-09,
    04_audit_plan.md §3, reproduced by scripts/audit_checks/check_anti_drift_full.py):
      * 20 seeded-random tokens per universe instead of the 4 busiest + 1 smallest
      * a checksum over EVERY column, not just `mid`
      * trades as well as l1 — `load_trades` previously had no anti-drift test at all
    """
    samp = _anti_drift_sample()
    assert len(samp) > 0, "no tokens with both l1 and trades"
    for r in samp.itertuples():
        raw = _raw_tape(kind, r.asset_id, r.universe)
        got = loader(r.asset_id)
        assert len(got) == len(raw), \
            f"{kind} row count differs for {r.asset_id[:12]}: {len(got)} vs {len(raw)}"
        if len(raw) == 0:
            continue
        # every raw column, in the raw table's own order. The loader adds derived columns (`ts`);
        # it must not change any column it passes through.
        cols = list(raw.columns)
        missing = [c for c in cols if c not in got.columns]
        assert not missing, f"{kind} loader dropped columns {missing} for {r.asset_id[:12]}"
        a = got[cols].copy()
        b = raw[cols].copy()
        a["asset_id"] = a["asset_id"].astype(str)
        b["asset_id"] = b["asset_id"].astype(str)
        assert _frame_checksum(a, cols) == _frame_checksum(b, cols), \
            f"{kind} all-column checksum differs for {r.asset_id[:12]} ({r.universe})"


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


def test_load_pair_aligns_both_sides_of_one_market():
    """load_pair's contract: the two outcome_index columns of ONE condition_id, mids aligned on a
    common UTC index, with .attrs carrying the labels.

    This deliberately does NOT assert col0 + col1 ~= 1. That sum is an ALGEBRAIC IDENTITY of
    Polymarket's quoting design (NO_bid = 1 - YES_ask and NO_ask = 1 - YES_bid, so
    mid_NO == 1 - mid_YES exactly), measured at a worst deviation of 1.1e-16 over 162,041 points.
    An assertion that cannot fail is not evidence about the data — see audit 2026-09,
    04_audit_plan.md §1.1. The identity itself is documented in epsilon_data/README.md; the check
    below only pins the loader's shape and alignment.
    """
    cat = ed.catalog(universe="politics_negrisk", apply_exclusions=False)
    cat = cat[cat["n_l1_events"] > 500]
    counts = cat["condition_id"].value_counts()
    cid = counts[counts == 2].index[0]
    pair = ed.load_pair(cid)

    assert set(pair.columns) == {0, 1}
    assert pair.attrs["condition_id"] == str(cid)
    assert set(pair.attrs["labels"]) == {0, 1}
    assert str(pair.index.tz) == "UTC"
    assert pair.index.is_monotonic_increasing
    both = pair.dropna()
    assert len(both) > 0, "the two sides never overlap in time"
    # mids are dollars in [0, 1] on both sides
    for col in (0, 1):
        assert both[col].between(0.0, 1.0).all(), f"outcome {col} has mids outside [0, 1]"
    # the aligned index is the union of the two tapes' seconds, not one side's
    sides = ed.catalog(apply_exclusions=False)
    sides = sides[sides["condition_id"] == str(cid)]
    assert len(sides) == 2
    for _, row in sides.iterrows():
        tape = ed.load_l1(row["asset_id"])
        assert len(tape) > 0
        assert tape["ts"].dt.floor("1s").isin(pair.index).all(), \
            f"outcome {row['outcome_index']} has timestamps missing from the aligned index"


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
    # (that audit_market never writes is asserted in test_audit_never_writes below)


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
    """write_exclusion writes a well-formed, re-readable exclusions.csv — into tmp_path.

    This test used to take `tmp_path` and ignore it: it appended to the REAL
    data/research_v1/exclusions.csv and restored it in a `finally`, so a hard kill mid-test left a
    mutated data file behind (audit 2026-09, 04_audit_plan.md §3). `write_exclusion` already
    accepts `root=`, so the fixture is all it ever needed.
    """
    from epsilon_data import config as cfg
    from epsilon_data import _internal as _i

    real = pathlib.Path(cfg.data_root()) / "exclusions.csv"
    real_before = real.read_text(encoding="utf-8") if real.exists() else None

    aids = _sample_assets(3)[:2]
    line = ed.write_exclusion(aids[0], "market", "unit-test", "pytest", root=tmp_path)
    excl = tmp_path / "exclusions.csv"
    assert excl.exists(), "write_exclusion did not create exclusions.csv under root="
    assert line.split(",")[0] == aids[0]

    # a second call appends and does not repeat the header
    ed.write_exclusion(aids[1], "market", "unit-test", "pytest", root=tmp_path)
    text = excl.read_text(encoding="utf-8")
    assert text.count("asset_id,scope,reason,date,who") == 1
    assert _i._parse_exclusions_text(text) == set(aids)

    # the shipped data file was not touched
    real_after = real.read_text(encoding="utf-8") if real.exists() else None
    assert real_after == real_before, "write_exclusion(root=tmp_path) touched the real data root"


def test_catalog_applies_exclusions(monkeypatch):
    """The filtering half of the exclusion contract, without writing any file: a token in
    excluded_ids() is marked in catalog(apply_exclusions=False) and dropped by default."""
    from epsilon_data import _internal as _i

    aid = _sample_assets(1)[0]
    monkeypatch.setattr(_i, "excluded_ids", lambda: frozenset({aid}))
    marked = ed.catalog(apply_exclusions=False).set_index("asset_id")
    assert aid in marked.index
    assert bool(marked.loc[aid, "excluded"]), "excluded token not marked when apply_exclusions=False"
    assert aid not in set(ed.catalog(apply_exclusions=True)["asset_id"]), \
        "excluded token not dropped when apply_exclusions=True"
