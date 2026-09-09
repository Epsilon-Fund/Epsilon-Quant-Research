"""E0 · Which source is authoritative for the L1 touch? bba is the tiebreaker.

Bounded sample: 20 assets/universe (liquid + thin) across well-covered hours where
price_change, bba and book all exist. Align on timestamp; report three pairwise agreements:
  (1) price_change best_bid/ask vs bba   (2) reconstructed book touch vs bba   (3) pc vs book
Then apply the decision rule from NEXT.md without checking in. Also: void cross-tab from
tokens_raw (cheap, no re-read). Sample reads via DuckDB httpfs (predicate pushdown) — no bulk
download. R2 creds read from rclone.conf, never printed.
"""
import os, json, configparser, time
import duckdb, pandas as pd, numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma"); DQ = DATA.replace(chr(92), "/")
REP = r"C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\stepE"
os.makedirs(REP, exist_ok=True)
DATE = "2026-08-20"; HOURS = ["10", "11", "12", "13", "14", "15"]
TOL = 0.01   # 1 cent

cfg = configparser.ConfigParser(); cfg.read(r"C:\Users\alvar\AppData\Roaming\rclone\rclone.conf")
r = cfg["r2"]; ep = r["endpoint"].replace("https://", "")
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"""CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{r['access_key_id']}',
  SECRET '{r['secret_access_key']}', ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);""")
BASE = "s3://epsilon-polymarket-data/parquet"

def globs(uni, tbl):
    # single glob string with an hour character-class -> matches only hours that exist
    # (esports has no price_change/trades for some quiet hours; explicit paths would 404)
    return [f"{BASE}/{DATE}/{uni}/{tbl}_{uni}_1[0-5].parquet"]

UNI = {"politics_negrisk": "politics_negrisk", "esports": "esports"}
out = []
def log(s): out.append(s); print(s, flush=True)

for uni in UNI:
    log(f"\n===== {uni} =====")
    # counts per asset across the sample hours (from price_change), and which assets have bba+book
    try:
        pc_cnt = con.execute(f"""SELECT asset_id, COUNT(*) n FROM read_parquet({globs(uni,'price_change')!r}, union_by_name=true)
                                 GROUP BY asset_id""").df()
    except Exception as e:
        log(f"  price_change read failed: {e}"); continue
    bba_assets = set(con.execute(f"SELECT DISTINCT asset_id FROM read_parquet({globs(uni,'bba')!r}, union_by_name=true)").df().asset_id)
    book_assets = set(con.execute(f"SELECT DISTINCT asset_id FROM read_parquet({globs(uni,'book')!r}, union_by_name=true)").df().asset_id)
    pc_cnt = pc_cnt[pc_cnt.asset_id.isin(bba_assets) & pc_cnt.asset_id.isin(book_assets)].sort_values("n", ascending=False)
    liquid = pc_cnt.head(10).asset_id.tolist()
    thin = pc_cnt.tail(10).asset_id.tolist()
    sample = list(dict.fromkeys(liquid + thin))
    log(f"  sample assets: {len(sample)} (10 liquid n_pc {pc_cnt.head(10).n.min()}-{pc_cnt.head(10).n.max()}, 10 thin n_pc {pc_cnt.tail(10).n.min()}-{pc_cnt.tail(10).n.max()})")
    idlist = "','".join(sample)

    pc = con.execute(f"""SELECT asset_id, timestamp_ms, received_ns, side, price, size, best_bid, best_ask
        FROM read_parquet({globs(uni,'price_change')!r}, union_by_name=true)
        WHERE asset_id IN ('{idlist}') ORDER BY asset_id, timestamp_ms, received_ns""").df()
    bba = con.execute(f"""SELECT asset_id, timestamp_ms, best_bid, best_ask
        FROM read_parquet({globs(uni,'bba')!r}, union_by_name=true)
        WHERE asset_id IN ('{idlist}') ORDER BY asset_id, timestamp_ms""").df()
    book = con.execute(f"""SELECT asset_id, timestamp_ms, bids, asks
        FROM read_parquet({globs(uni,'book')!r}, union_by_name=true)
        WHERE asset_id IN ('{idlist}') ORDER BY asset_id, timestamp_ms""").df()
    for d in (pc, bba, book): d["asset_id"] = d["asset_id"].astype(str)

    # pre-group by asset once (avoid repeated boolean filtering of big frames)
    pc_by = {a: g for a, g in pc.groupby("asset_id")}
    bba_by = {a: g for a, g in bba.groupby("asset_id")}
    book_by = {a: g for a, g in book.groupby("asset_id")}

    # ---- reconstruct book touch, probing ONLY at the timestamps we compare against ----
    # events: 0=snapshot(reset), 1=delta(apply O(1)), 2=probe_bba, 3=probe_pc.
    # touch (max/min over the level dict) is computed only on probe events.
    def reconstruct_probe(aid, pc_probe_cap=3000):
        bk = book_by.get(aid); pcs = pc_by.get(aid); bb_ = bba_by.get(aid)
        ev = []
        if bk is not None:
            for ts, bids_j, asks_j in zip(bk.timestamp_ms.values, bk.bids.values, bk.asks.values):
                ev.append((int(ts), 0, bids_j, asks_j))
        if pcs is not None:
            for ts, side, price, size in zip(pcs["timestamp_ms"].values, pcs["side"].values, pcs["price"].values, pcs["size"].values):
                ev.append((int(ts), 1, side, (price, size)))
        bba_ts = bb_.timestamp_ms.values if bb_ is not None else np.array([])
        for ts in bba_ts: ev.append((int(ts), 2, None, None))
        pc_ts = pcs.timestamp_ms.values if pcs is not None else np.array([])
        if len(pc_ts) > pc_probe_cap:
            pc_ts = pc_ts[np.linspace(0, len(pc_ts) - 1, pc_probe_cap).astype(int)]
        for ts in pc_ts: ev.append((int(ts), 3, None, None))
        ev.sort(key=lambda x: (x[0], x[1]))   # probes(2,3) come after snap(0)/delta(1) at same ts
        bids = {}; asks = {}; out_bba = []; out_pc = []
        for ts, kind, a, b in ev:
            if kind == 0:
                bids, asks = {}, {}
                for lvl in (json.loads(a) if isinstance(a, str) else a) or []:
                    bids[float(lvl["price"])] = float(lvl["size"])
                for lvl in (json.loads(b) if isinstance(b, str) else b) or []:
                    asks[float(lvl["price"])] = float(lvl["size"])
            elif kind == 1:
                price, size = float(b[0]), float(b[1])
                d = bids if str(a).upper() in ("BUY", "BID") else asks
                if size == 0: d.pop(price, None)
                else: d[price] = size
            else:
                bb = max((p for p, s in bids.items() if s > 0), default=np.nan)
                ba = min((p for p, s in asks.items() if s > 0), default=np.nan)
                (out_bba if kind == 2 else out_pc).append((ts, bb, ba))
        return (pd.DataFrame(out_bba, columns=["timestamp_ms", "rb_bid", "rb_ask"]),
                pd.DataFrame(out_pc, columns=["timestamp_ms", "rb_bid", "rb_ask"]))

    # aggregate agreements over the sample
    n1 = a1 = a1c = 0     # pc vs bba
    n2 = a2 = a2c = 0     # reconstructed book vs bba
    n3 = a3 = a3c = 0     # pc vs reconstructed book
    per_asset = []
    for aid in sample:
        b = bba_by.get(aid); p = pc_by.get(aid)
        if b is None or b.empty: continue
        b = b.sort_values("timestamp_ms").reset_index(drop=True)
        p = (p.dropna(subset=["best_bid", "best_ask"]).sort_values("timestamp_ms").reset_index(drop=True)) if p is not None else None
        rb_bba, rb_pc = reconstruct_probe(aid)
        rb_bba = rb_bba.dropna(subset=["rb_bid", "rb_ask"]); rb_pc = rb_pc.dropna(subset=["rb_bid", "rb_ask"])
        # (1) pc vs bba : asof pc onto bba timestamps
        pa1 = np.nan
        if p is not None and not p.empty:
            m = pd.merge_asof(b, p[["timestamp_ms","best_bid","best_ask"]].rename(columns={"best_bid":"p_bid","best_ask":"p_ask"}),
                              on="timestamp_ms", direction="backward").dropna(subset=["p_bid"])
            if len(m):
                ex = ((m.best_bid==m.p_bid)&(m.best_ask==m.p_ask)).sum()
                cc = ((abs(m.best_bid-m.p_bid)<=TOL)&(abs(m.best_ask-m.p_ask)<=TOL)).sum()
                n1 += len(m); a1 += ex; a1c += cc; pa1 = cc/len(m)
        # (2) reconstructed book vs bba : rb_bba already aligned to bba timestamps
        pa2 = np.nan
        if not rb_bba.empty:
            mb = b.merge(rb_bba, on="timestamp_ms", how="inner")
            if len(mb):
                cc2 = ((abs(mb.best_bid-mb.rb_bid)<=TOL)&(abs(mb.best_ask-mb.rb_ask)<=TOL)).sum()
                n2 += len(mb); a2c += cc2; pa2 = cc2/len(mb)
        # (3) pc vs reconstructed book : rb_pc aligned to sampled pc timestamps
        pa3 = np.nan
        if p is not None and not p.empty and not rb_pc.empty:
            mp = p[["timestamp_ms","best_bid","best_ask"]].merge(rb_pc, on="timestamp_ms", how="inner").drop_duplicates("timestamp_ms")
            if len(mp):
                cc3 = ((abs(mp.best_bid-mp.rb_bid)<=TOL)&(abs(mp.best_ask-mp.rb_ask)<=TOL)).sum()
                n3 += len(mp); a3c += cc3; pa3 = cc3/len(mp)
        per_asset.append((aid[:12], pa1, pa2, pa3))
    pct = lambda a, n: f"{a/n:.1%}" if n else "n/a"
    log(f"  (1) price_change vs bba : exact {pct(a1,n1)} | within 1c {pct(a1c,n1)}  (n={n1:,} aligned)")
    log(f"  (2) reconstructed book vs bba : within 1c {pct(a2c,n2)}  (n={n2:,})")
    log(f"  (3) price_change vs reconstructed book : within 1c {pct(a3c,n3)}  (n={n3:,})")
    log(f"  per-asset within-1c (asset, pc~bba, book~bba, pc~book):")
    for aid, x1, x2, x3 in per_asset[:6]:
        log(f"     {aid}…  {x1:.2f}  {x2:.2f}  {x3:.2f}")

# ---- void cross-tab from tokens_raw (no re-read) ----
log("\n===== esports near-0.5 cross-tab (from tokens_raw) =====")
tok = pd.read_parquet(os.path.join(DATA, "tokens_raw.parquet"))
e = tok[(tok.universe == "esports") & (tok.identity_status == "resolved") & (tok.closed == True)].copy()
# per condition: both last_mid near 0.5?
near = e.groupby("condition_id").filter(lambda g: len(g) == 2 and g.last_mid.notna().all()
                                        and g.last_mid.between(0.35, 0.65).all())
ncond = near.condition_id.nunique()
voided = near[near.outcome_price == "0.5"].condition_id.nunique()
settled = near[near.outcome_price.isin(["1", "0"])].condition_id.nunique()
log(f"  esports resolved conditions ending near 0.5 (both tokens): {ncond}")
log(f"    of these, outcomePrices==0.5 (ACTUALLY VOIDED): {voided}")
log(f"    outcomePrices in 1/0 (settled but never traded to extreme): {settled}")

open(os.path.join(REP, "e0_source_check.txt"), "w", encoding="utf-8").write("\n".join(out))
print("\nOK E0")
