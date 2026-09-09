"""C1 · Target list. Promote the Step-B condition scan to targets.parquet.

One row per (condition_id, asset_id) — the statement of what we expect Gamma to return.
Plus a per-condition roll-up. Reads the Step-B CSV; writes to the gitignored data dir.
No network.
"""
import os, duckdb

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma")
os.makedirs(DATA, exist_ok=True)
SRC = r"C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\stepB\condition_target_list.csv"

con = duckdb.connect()
# CSV columns: condition_id, universe, n_assets, asset_ids (pipe-joined)
con.execute(f"""CREATE TEMP TABLE roll AS
  SELECT condition_id, universe, CAST(n_assets AS INT) n_assets, asset_ids
  FROM read_csv_auto('{SRC.replace(chr(92), '/')}', header=true, all_varchar=true)""")
# long form: one row per (condition_id, asset_id)
con.execute("""CREATE TEMP TABLE long AS
  SELECT condition_id, universe, UNNEST(string_split(asset_ids, '|')) AS asset_id FROM roll""")

DQ = DATA.replace(chr(92), "/")
con.execute(f"COPY (SELECT * FROM long ORDER BY condition_id, asset_id) TO '{DQ}/targets.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT condition_id, universe, n_assets FROM roll ORDER BY condition_id) TO '{DQ}/targets_rollup.parquet' (FORMAT parquet)")

nc = con.execute("SELECT COUNT(*) FROM roll").fetchone()[0]
nt = con.execute("SELECT COUNT(*) FROM long").fetchone()[0]
per = dict(con.execute("SELECT universe, COUNT(*) FROM roll GROUP BY universe").fetchall())
dist = dict(con.execute("SELECT n_assets, COUNT(*) FROM roll GROUP BY n_assets ORDER BY n_assets").fetchall())
print(f"targets.parquet: {nt} rows (condition,asset) | conditions={nc} | per universe {per}")
print(f"n_assets distribution: {dist}")
assert nt == 2 * nc, f"expected 2 assets per condition; got {nt} tokens for {nc} conditions"
print("OK C1")
