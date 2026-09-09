#!/usr/bin/env python
"""Independent raw -> library reconciliation for ONE (date, universe, hour) shard.

READ-ONLY against R2 (GetObject only; shard cached in scratchpad). Does NOT use the loader:
reads the library parquet directly with DuckDB so the check is independent of epsilon_data's code path.

What it tests
  * l1:      raw price_change rows -> rows where (best_bid, best_ask) changed per asset  ==  library l1 rows
             for the same assets in the same window.  (Reproduces the "touch-moving dedup" claim.)
  * trades:  raw trades rows for the hour  ==  library trades rows; and the SET of transaction_hash
             is identical (the strongest "nothing silently dropped" test available).
  * ratio:   kept/raw on this shard vs the 2.64% archive-wide figure (e2_counts.csv).

    python scripts/audit_checks/check_reconcile_raw_hour.py 2026-08-10 politics_negrisk 03
"""
import sys, os, json, time, tempfile
sys.path.insert(0, ".")
import duckdb
from epsilon_data._internal import _r2_creds, _uri

DATE, UNI, HH = sys.argv[1], sys.argv[2], sys.argv[3]
SP = os.environ.get("AUDIT_RAW_CACHE") or os.path.join(tempfile.gettempdir(), "epsilon_audit_raw")
os.makedirs(SP, exist_ok=True)
B = "epsilon-polymarket-data"
MONTH = DATE[:7]


def fetch(table):
    key = f"parquet/{DATE}/{UNI}/{table}_{UNI}_{HH}.parquet"
    dst = f"{SP}/{DATE}_{UNI}_{table}_{HH}.parquet"
    if not os.path.exists(dst):
        import boto3
        from botocore.config import Config
        kid, sec, ep = _r2_creds()
        s3 = boto3.client("s3", endpoint_url=f"https://{ep}", aws_access_key_id=kid, aws_secret_access_key=sec,
                          region_name="auto", config=Config(s3={"addressing_style": "path"},
                          signature_version="s3v4", connect_timeout=15, read_timeout=180))
        t0 = time.time(); s3.download_file(B, key, dst)   # GetObject only
        print(f"  get_object {key} -> {os.path.getsize(dst)/1e6:.1f} MB in {time.time()-t0:.1f}s", file=sys.stderr)
    return dst


pc, bba, tr = fetch("price_change"), fetch("bba"), fetch("trades")
L1 = _uri("l1", f"universe={UNI}", f"month={MONTH}", "*.parquet")
TR = _uri("trades", f"universe={UNI}", f"month={MONTH}", "*.parquet")

c = duckdb.connect(); c.execute("SET TimeZone='UTC'")
q = lambda s: c.execute(s).fetchdf()

# window = the shard's own timestamp span (sharding is by received_at; exchange ts can straddle the hour)
w = q(f"SELECT MIN(timestamp_ms) AS lo, MAX(timestamp_ms) AS hi, COUNT(*) AS n, COUNT(DISTINCT asset_id) AS n_assets FROM read_parquet('{pc}')").iloc[0]
lo, hi = int(w.lo), int(w.hi)

# --- rule A: keep a row iff (best_bid, best_ask) differs from the previous row of the same asset ---
dedup = q(f"""
WITH s AS (
  SELECT asset_id, timestamp_ms, received_ns, best_bid, best_ask,
         LAG(best_bid) OVER (PARTITION BY asset_id ORDER BY timestamp_ms, received_ns) AS pb,
         LAG(best_ask) OVER (PARTITION BY asset_id ORDER BY timestamp_ms, received_ns) AS pa
  FROM read_parquet('{pc}'))
SELECT asset_id, COUNT(*) AS raw_rows,
       COUNT(*) FILTER (WHERE pb IS NULL OR best_bid IS DISTINCT FROM pb OR best_ask IS DISTINCT FROM pa) AS kept_ruleA,
       COUNT(*) FILTER (WHERE pb IS NOT NULL AND (best_bid IS DISTINCT FROM pb OR best_ask IS DISTINCT FROM pa)) AS kept_ruleA_nofirst
FROM s GROUP BY 1""")

lib = q(f"""SELECT CAST(asset_id AS VARCHAR) AS asset_id, COUNT(*) AS lib_rows
            FROM read_parquet('{L1}') WHERE timestamp_ms BETWEEN {lo} AND {hi} GROUP BY 1""")
m = dedup.merge(lib, on="asset_id", how="outer").fillna(0)
m["diff_A"] = m.kept_ruleA - m.lib_rows
m["diff_A_nofirst"] = m.kept_ruleA_nofirst - m.lib_rows

n_bba = int(q(f"SELECT COUNT(*) FROM read_parquet('{bba}')").iloc[0, 0])

# --- trades: row count and exact tx-hash set equality in the window ---
tw = q(f"SELECT MIN(timestamp_ms) AS lo, MAX(timestamp_ms) AS hi, COUNT(*) AS n FROM read_parquet('{tr}')").iloc[0]
raw_tx = set(q(f"SELECT transaction_hash||'|'||CAST(asset_id AS VARCHAR)||'|'||CAST(size AS VARCHAR) AS k FROM read_parquet('{tr}')").k)
lib_tx = set(q(f"""SELECT transaction_hash||'|'||CAST(asset_id AS VARCHAR)||'|'||CAST(size AS VARCHAR) AS k
                   FROM read_parquet('{TR}') WHERE timestamp_ms BETWEEN {int(tw.lo)} AND {int(tw.hi)}""").k)

out = {
    "shard": {"date": DATE, "universe": UNI, "hour": HH, "ts_window_utc": [lo, hi]},
    "l1": {
        "raw_price_change_rows": int(w.n), "raw_assets": int(w.n_assets), "bba_rows_same_hour": n_bba,
        "kept_ruleA_touch_changed": int(m.kept_ruleA.sum()),
        "kept_ruleA_excluding_first_row_per_asset": int(m.kept_ruleA_nofirst.sum()),
        "library_l1_rows_in_window": int(m.lib_rows.sum()),
        "assets_exact_match_ruleA": int((m.diff_A == 0).sum()),
        "assets_exact_match_ruleA_nofirst": int((m.diff_A_nofirst == 0).sum()),
        "assets_within_1_ruleA": int((m.diff_A.abs() <= 1).sum()),
        "assets_total": int(len(m)),
        "sum_abs_diff_ruleA": int(m.diff_A.abs().sum()),
        "keep_ratio_this_shard": round(float(m.kept_ruleA.sum()) / float(w.n), 5),
        "archive_keep_ratio_documented": 0.02637,
    },
    "trades": {
        "raw_rows": int(tw.n), "library_rows_in_window": len(lib_tx),
        "raw_keys_not_in_library": len(raw_tx - lib_tx), "library_keys_not_in_raw": len(lib_tx - raw_tx),
        "tx_key_sets_identical": raw_tx == lib_tx,
    },
}
print(json.dumps(out, indent=2))
worst = m.reindex(m.diff_A.abs().sort_values(ascending=False).index).head(6)
print("\nworst per-asset deviations (rule A):")
print(worst[["asset_id", "raw_rows", "kept_ruleA", "kept_ruleA_nofirst", "lib_rows", "diff_A"]].to_string(index=False, max_colwidth=22))
