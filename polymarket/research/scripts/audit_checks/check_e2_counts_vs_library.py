#!/usr/bin/env python
"""Does e2_counts.csv (the per-(universe,month) raw->kept ledger shipped with the data) agree with the
library's ACTUAL per-partition row counts for l1 and trades? Closes the ledger's library side, leaving only its
raw-side counts (l1_raw) unverified without the 71 GB archive.

    python scripts/audit_checks/check_e2_counts_vs_library.py
"""
import sys, csv, json
sys.path.insert(0, ".")
import duckdb
from epsilon_data._internal import _uri, _root

c = duckdb.connect()
lib = {(r[0], r[1]): r[2] for r in c.execute(f"SELECT universe, month, COUNT(*) FROM read_parquet('{_uri('l1','*','*','*.parquet')}', hive_partitioning=true) GROUP BY 1,2").fetchall()}
trd = {(r[0], r[1]): r[2] for r in c.execute(f"SELECT universe, month, COUNT(*) FROM read_parquet('{_uri('trades','*','*','*.parquet')}', hive_partitioning=true) GROUP BY 1,2").fetchall()}
ok = True; rows = []
for r in csv.DictReader(open(f"{_root()}/e2_counts.csv")):
    k = (r["universe"], r["month"])
    rec = {"universe": k[0], "month": k[1], "e2_l1_kept": int(r["l1_kept"]), "lib_l1": lib[k], "e2_trades": int(r["trades"]),
           "lib_trades": trd[k], "e2_l1_raw": int(r["l1_raw"]), "keep_ratio": round(int(r["l1_kept"]) / int(r["l1_raw"]), 6)}
    rec["match"] = rec["e2_l1_kept"] == rec["lib_l1"] and rec["e2_trades"] == rec["lib_trades"]; ok &= rec["match"]
    rows.append(rec); print(json.dumps(rec))
print(json.dumps({"ALL_MATCH": ok}))
