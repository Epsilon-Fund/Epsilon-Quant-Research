#!/usr/bin/env python
"""What tests/test_loader.py::test_anti_drift_l1 does NOT cover, run once:
  * load_trades vs raw parquet (the shipped suite has NO trades anti-drift test)
  * ALL columns checksummed, not just `mid` (shipped test: 5 tokens, first/last row, md5 of mid only)
  * a larger, seeded-random sample (shipped test: 4 busiest + 1 smallest)
Raw side = DuckDB read of the library parquet, ordered (timestamp_ms, received_ns) — same as the test's _raw_l1.

    python scripts/audit_checks/check_anti_drift_full.py [n=40] [seed=3]
"""
import sys, hashlib, json
sys.path.insert(0, ".")
import duckdb, pandas as pd
import epsilon_data as ed
from epsilon_data._internal import _uri

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cat = ed.catalog(apply_exclusions=False)
cat = cat[(cat.n_l1_events > 0)]
samp = pd.concat([cat[cat.universe == u].sample(N // 2, random_state=SEED) for u in ("politics_negrisk", "esports")])
c = duckdb.connect()


def raw(kind, aid, uni):
    g = _uri(kind, f"universe={uni}", "*", "*.parquet")
    return c.execute(f"SELECT * FROM read_parquet('{g}') WHERE CAST(asset_id AS VARCHAR)='{aid}' ORDER BY timestamp_ms, received_ns").df()


def h(df, cols):
    return hashlib.md5(pd.util.hash_pandas_object(df[cols].reset_index(drop=True), index=False).values.tobytes()).hexdigest()


res = {"l1": {"tokens": 0, "rowcount_ok": 0, "all_columns_checksum_ok": 0, "bad": []},
       "trades": {"tokens": 0, "rowcount_ok": 0, "all_columns_checksum_ok": 0, "bad": []}}
for r in samp.itertuples():
    for kind, fn in (("l1", ed.load_l1), ("trades", ed.load_trades)):
        rw = raw(kind, r.asset_id, r.universe); got = fn(r.asset_id)
        cols = [x for x in rw.columns]                       # every raw column (loader adds ts; compare raw set)
        got2 = got[cols].copy(); got2["asset_id"] = got2["asset_id"].astype(str); rw["asset_id"] = rw["asset_id"].astype(str)
        res[kind]["tokens"] += 1
        if len(got2) == len(rw): res[kind]["rowcount_ok"] += 1
        if len(got2) == len(rw) and h(got2, cols) == h(rw, cols): res[kind]["all_columns_checksum_ok"] += 1
        else: res[kind]["bad"].append({"asset_id": r.asset_id[:16], "universe": r.universe, "rows_loader": len(got2), "rows_raw": len(rw)})
print(json.dumps(res, indent=2))
