#!/usr/bin/env python
"""Library-wide test of the documented l1 rule (epsilon_data/README.md:128-130):
   "A row exists only where best_bid/best_ask changed."  => rows whose touch equals the asset's previous
row's touch must be ZERO. Direct DuckDB scan of the library parquet per (universe, month); no loader.
Also reports how many such rows sit at a part-file boundary (the dedup-state-reset signature).

    python scripts/audit_checks/check_l1_dedup_integrity.py
"""
import sys, os, time, json
sys.path.insert(0, ".")
import duckdb
from epsilon_data._internal import _uri

c = duckdb.connect(); c.execute("SET preserve_insertion_order=false")
out = []; tot = rows = 0
for u in ("politics_negrisk", "esports"):
    for m in ("2026-06", "2026-07", "2026-08"):
        t0 = time.time()
        r = c.execute(f"""
        WITH s AS (SELECT asset_id, best_bid, best_ask, filename,
                          LAG(best_bid) OVER w AS pb, LAG(best_ask) OVER w AS pa, LAG(filename) OVER w AS pf
                   FROM read_parquet('{_uri('l1', f'universe={u}', f'month={m}', '*.parquet')}', filename=true)
                   WINDOW w AS (PARTITION BY asset_id ORDER BY timestamp_ms, received_ns))
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (WHERE pb IS NOT NULL AND best_bid = pb AND best_ask = pa) AS non_touch_move,
               COUNT(*) FILTER (WHERE pb IS NOT NULL AND best_bid = pb AND best_ask = pa AND filename <> pf) AS at_part_boundary,
               COUNT(DISTINCT filename) AS n_parts FROM s""").fetchdf().iloc[0]
        rec = {"universe": u, "month": m, "rows": int(r.rows), "non_touch_move_rows": int(r.non_touch_move),
               "share": round(float(r.non_touch_move / r.rows), 6), "at_part_boundary": int(r.at_part_boundary),
               "n_parts": int(r.n_parts), "seconds": round(time.time() - t0, 1)}
        out.append(rec); tot += rec["non_touch_move_rows"]; rows += rec["rows"]
        print(json.dumps(rec))
print(json.dumps({"TOTAL_non_touch_move_rows": tot, "of_rows": rows, "share": round(tot / rows, 6),
                  "expected_under_documented_rule": 0, "PASS": tot == 0}, indent=1))
