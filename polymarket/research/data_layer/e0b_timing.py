"""E0b · Diagnose the esports price_change-vs-bba gap: timing artifact or structural?
Re-align pc onto bba with a time tolerance (nearest within N ms). If agreement jumps to
~100% with a little slack, the ~40% gap is fast-book clock skew, not bad data.
Esports only, same bounded sample. httpfs, no bulk download."""
import os, configparser
import duckdb, pandas as pd, numpy as np

cfg = configparser.ConfigParser(); cfg.read(r"C:\Users\alvar\AppData\Roaming\rclone\rclone.conf")
r = cfg["r2"]; ep = r["endpoint"].replace("https://", "")
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"""CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{r['access_key_id']}',
  SECRET '{r['secret_access_key']}', ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);""")
BASE = "s3://epsilon-polymarket-data/parquet"; DATE = "2026-08-20"; TOL = 0.01
def g(tbl): return [f"{BASE}/{DATE}/esports/{tbl}_esports_1[0-5].parquet"]

pc_cnt = con.execute(f"SELECT asset_id, COUNT(*) n FROM read_parquet({g('price_change')!r}, union_by_name=true) GROUP BY asset_id").df()
bba_a = set(con.execute(f"SELECT DISTINCT asset_id FROM read_parquet({g('bba')!r}, union_by_name=true)").df().asset_id)
pc_cnt = pc_cnt[pc_cnt.asset_id.isin(bba_a)].sort_values("n", ascending=False)
sample = list(dict.fromkeys(pc_cnt.head(10).asset_id.tolist() + pc_cnt.tail(10).asset_id.tolist()))
idlist = "','".join(sample)
pc = con.execute(f"""SELECT asset_id,timestamp_ms,best_bid,best_ask FROM read_parquet({g('price_change')!r}, union_by_name=true)
     WHERE asset_id IN ('{idlist}') AND best_bid IS NOT NULL ORDER BY asset_id,timestamp_ms""").df()
bba = con.execute(f"""SELECT asset_id,timestamp_ms,best_bid,best_ask FROM read_parquet({g('bba')!r}, union_by_name=true)
     WHERE asset_id IN ('{idlist}') ORDER BY asset_id,timestamp_ms""").df()
pc["asset_id"] = pc.asset_id.astype(str); bba["asset_id"] = bba.asset_id.astype(str)

print("esports pc-vs-bba agreement (within 1c) as a function of alignment slack:")
for tol_ms in [0, 50, 200, 1000, 5000]:
    n = a = 0
    for aid in sample:
        b = bba[bba.asset_id == aid].sort_values("timestamp_ms").reset_index(drop=True)
        p = pc[pc.asset_id == aid].sort_values("timestamp_ms").reset_index(drop=True)
        if b.empty or p.empty: continue
        kw = dict(on="timestamp_ms", direction="nearest")
        if tol_ms > 0: kw["tolerance"] = tol_ms
        m = pd.merge_asof(b, p.rename(columns={"best_bid":"p_bid","best_ask":"p_ask"}), **kw).dropna(subset=["p_bid"])
        if len(m):
            a += ((abs(m.best_bid-m.p_bid)<=TOL)&(abs(m.best_ask-m.p_ask)<=TOL)).sum(); n += len(m)
    print(f"  nearest within {tol_ms:>5} ms: {a/n:.1%}  (n={n:,})" if n else f"  {tol_ms} ms: n/a")
print("OK E0b")
