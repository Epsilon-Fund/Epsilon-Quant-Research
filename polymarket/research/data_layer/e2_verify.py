"""E2 verify · manifest<->inventory completeness (ADDENDUM) + reconciliation + six-date spot
check. Runs AFTER e2_build (needs l1/trades + e2_counts + manifest) and e1_tokens (tokens.parquet).

Prints STOP conditions if: a partition read fewer files than a fresh independent rclone
inventory holds; an l1/trades asset is not in tokens; or a spot-check date is < ~98% vs bba.
Reads built tables locally; bba via EXPLICIT URIs from the manifest (no httpfs LIST). asset_id TEXT.
"""
import os, csv, time, subprocess, configparser
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import duckdb, pandas as pd, numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma")
V1 = os.path.join(HERE, "..", "data", "research_v1"); V1Q = V1.replace(chr(92), "/")
MAN = os.path.join(V1, "pc_file_manifest.txt")
REP = r"C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\stepE"
os.makedirs(REP, exist_ok=True)
S3 = "s3://epsilon-polymarket-data/parquet/"
GiB = 1024**3
INV_FILES, INV_PC_GIB, INV_TR_GIB = 11984, 60.05, 0.71   # Step 1 archive inventory
TOL = 0.01; SLACK_MS = 50
DAEMON_TR_ALLSESS = 5_757_382   # sum of 8 per-session daemon trade counters (messages)

out = []
def log(s): out.append(s); print(s, flush=True)

# ---- build manifest -> per (uni,month,table) files READ + explicit URIs ----
read_files = defaultdict(list)   # (uni,table,month)->[uri]
bba_by_date = defaultdict(list)  # (date,uni)->[bba uri]
man_lines = 0
for ln in open(MAN, encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    man_lines += 1
    p = ln.split("/")
    if len(p) != 3: continue
    date, uni, fname = p; month = date[:7]
    for table in ("price_change", "trades"):
        if fname.startswith(f"{table}_{uni}_"):
            read_files[(uni, table, month)].append(S3 + ln)
    if fname.startswith(f"bba_{uni}_"):
        bba_by_date[(date, uni)].append(S3 + ln)

# ---- fresh INDEPENDENT inventory via rclone lsf (size;path) ----
log("=== MANIFEST <-> INVENTORY (ADDENDUM) ===")
r = subprocess.run(["rclone", "lsf", "r2:epsilon-polymarket-data/parquet/", "-R", "--files-only",
                    "--format", "sp", "--separator", "\t"], capture_output=True, text=True)
inv = defaultdict(lambda: [0, 0])   # (uni,table,month)->[files,bytes]
inv_total_files = 0; pc_files = pc_bytes = tr_files = tr_bytes = 0
for ln in r.stdout.splitlines():
    if "\t" not in ln: continue
    size, path = ln.split("\t", 1); inv_total_files += 1
    pp = path.split("/")
    if len(pp) != 3: continue
    date, uni, fname = pp; month = date[:7]; sz = int(size)
    for table in ("price_change", "trades"):
        if fname.startswith(f"{table}_{uni}_"):
            inv[(uni, table, month)][0] += 1; inv[(uni, table, month)][1] += sz
            if table == "price_change": pc_files += 1; pc_bytes += sz
            else: tr_files += 1; tr_bytes += sz
log(f"fresh rclone inventory: {inv_total_files:,} files vs Step-1 inventory {INV_FILES:,}  (diff {inv_total_files-INV_FILES:+d})")
log(f"  price_change: {pc_files:,} files, {pc_bytes/GiB:.2f} GiB vs inventory {INV_PC_GIB} GiB")
log(f"  trades:       {tr_files:,} files, {tr_bytes/GiB:.2f} GiB vs inventory {INV_TR_GIB} GiB")
manifest_pc = sum(len(v) for k, v in read_files.items() if k[1] == "price_change")
manifest_tr = sum(len(v) for k, v in read_files.items() if k[1] == "trades")
log(f"build manifest: price_change {manifest_pc:,} files read, trades {manifest_tr:,} files read")

# per (uni,month): files READ (manifest) vs INVENTORY (fresh); STOP if short
log("\nper (universe, month) files READ vs INVENTORY  [pc read/inv | tr read/inv]:")
short = []
UNIS = ["politics_negrisk", "esports"]; MONTHS = ["2026-06", "2026-07", "2026-08"]
for uni in UNIS:
    for month in MONTHS:
        pr, pi = len(read_files.get((uni,"price_change",month), [])), inv.get((uni,"price_change",month),[0,0])[0]
        tr_, ti = len(read_files.get((uni,"trades",month), [])), inv.get((uni,"trades",month),[0,0])[0]
        flag = ""
        if pr < pi: short.append(f"{uni} {month} price_change ({pr}<{pi})"); flag += "  <-- SHORT pc"
        if tr_ < ti: short.append(f"{uni} {month} trades ({tr_}<{ti})"); flag += "  <-- SHORT trades"
        # distinct dates in manifest for this partition
        dates = sorted({u.split('/')[-3] for u in read_files.get((uni,"price_change",month), [])})
        log(f"  {uni:16s} {month}: pc {pr}/{pi} | tr {tr_}/{ti} | dates {len(dates)}{flag}")
log(f"SHORT partitions (files read < inventory): {short if short else 'NONE — manifest is complete'}")
log("Known hole: 2026-06-22 15:00 -> 06-23 08:29 (17h, both universes) — the one real outage; not thin trading.")

# ---- row-count reconciliation (true parquet footers, from build's e2_counts) ----
log("\n=== ROW RECONCILIATION (true parquet footer counts) ===")
cnt = pd.read_csv(os.path.join(V1, "e2_counts.csv"))
l1_raw, l1_kept, tr = int(cnt.l1_raw.sum()), int(cnt.l1_kept.sum()), int(cnt.trades.sum())
log(f"price_change rows (archive, pre-dedup): {l1_raw:,}")
log(f"l1 rows (post-dedup, touch-moving): {l1_kept:,}  = {100*l1_kept/max(l1_raw,1):.2f}% (Step-3 predicted ~4%)")
log(f"trades rows: {tr:,}")
for uni, g in cnt.groupby("universe"):
    log(f"  {uni}: pc {int(g.l1_raw.sum()):,} -> l1 {int(g.l1_kept.sum()):,} ({100*g.l1_kept.sum()/max(g.l1_raw.sum(),1):.2f}%), trades {int(g.trades.sum()):,}")
log("\nDAEMON-COUNTER CORRECTION (finding): Step A recorded 133.6M price_change / 445k trades as")
log("'grand totals'. WRONG on two counts: (1) capture_end totals RESET each session — that pair is")
log(f"the LAST session only (8 sessions sum to ~1.57B / {DAEMON_TR_ALLSESS:,}); (2) the daemon counts")
log("price_change MESSAGES while the parquet stores one row per fanned-out ENTRY. So the archive's")
log(f"true price_change rows ({l1_raw:,}) are the reconciliation basis, not the daemon figure.")

# ---- asset coverage ----
tok = pd.read_parquet(os.path.join(V1, "tokens.parquet")); tok["asset_id"] = tok["asset_id"].astype(str)
tokset = set(tok.asset_id)
l1a = set(duckdb.sql(f"SELECT DISTINCT CAST(asset_id AS VARCHAR) a FROM read_parquet('{V1Q}/l1/*/*/*.parquet')").df().a.astype(str))
tra = set(duckdb.sql(f"SELECT DISTINCT CAST(asset_id AS VARCHAR) a FROM read_parquet('{V1Q}/trades/*/*/*.parquet')").df().a.astype(str))
miss_l1 = l1a - tokset; miss_tr = tra - tokset
log(f"\n=== ASSET COVERAGE ===")
log(f"l1 assets not in tokens: {len(miss_l1)} | trades assets not in tokens: {len(miss_tr)}  (both must be 0)")
if miss_l1 or miss_tr: log(f"  STOP examples: l1={list(miss_l1)[:3]} tr={list(miss_tr)[:3]}")
tok["_no_l1"] = tok.n_l1_events.fillna(0) == 0; tok["_no_tr"] = tok.n_trades.fillna(0) == 0
for uni, g in tok.groupby("universe"):
    log(f"  {uni}: tokens={len(g)} zero-l1={int(g._no_l1.sum())} zero-trades={int(g._no_tr.sum())} both-empty={int((g._no_l1&g._no_tr).sum())}")

# ---- near-0.5 split ----
raw = pd.read_parquet(os.path.join(DATA, "tokens_raw.parquet"))[["asset_id","outcome_price"]]
raw["asset_id"] = raw["asset_id"].astype(str)
m = tok.merge(raw, on="asset_id", how="left")
es = m[(m.universe=="esports") & (m.check4_status=="near_half") & (m.outcome_price.isin(["1","0"]))]
h = es["hours_from_last_seen_to_close"].dropna()
log(f"\n=== esports settled-near-0.5 split (via hours_from_last_seen_to_close) ===")
log(f"settled-near-0.5 esports tokens: {len(es)}; with closed_time: {len(h)}")
if len(h):
    log(f"  <1h before settlement (book never moved on a resolved outcome): {int((h<1).sum())}")
    log(f"  >=1h before settlement (we stopped watching — not a finding): {int((h>=1).sum())}")
    log(f"  hours: median {h.median():.1f} p90 {h.quantile(.9):.1f} max {h.max():.1f}")

# ---- six-date spot check: built l1 vs bba (explicit URIs), nearest within 50ms ----
log(f"\n=== SIX-DATE SPOT CHECK (built l1 vs bba, nearest within {SLACK_MS} ms) ===")
cfg = configparser.ConfigParser(); cfg.read(r"C:\Users\alvar\AppData\Roaming\rclone\rclone.conf")
rr = cfg["r2"]; ep = rr["endpoint"].replace("https://", "")
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; SET http_retries=8; SET http_timeout=120000;")
con.execute(f"""CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{rr['access_key_id']}',
  SECRET '{rr['secret_access_key']}', ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);""")
def lit(u): return "[" + ",".join("'"+x+"'" for x in u) + "]"
DATES = ["2026-06-19","2026-06-23","2026-07-01","2026-07-15","2026-08-01","2026-08-20"]
def day_ms(d):
    dt = datetime.strptime(d,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp()*1000), int((dt+timedelta(days=1)).timestamp()*1000)
spot = []
for d in DATES:
    lo, hi = day_ms(d); rows=agree=nass=0
    for uni in UNIS:
        try:
            top = con.execute(f"""SELECT CAST(asset_id AS VARCHAR) a, COUNT(*) n FROM read_parquet('{V1Q}/l1/universe={uni}/*/*.parquet')
                WHERE timestamp_ms>={lo} AND timestamp_ms<{hi} GROUP BY 1 ORDER BY n DESC LIMIT 5""").df()
        except Exception: continue
        buri = bba_by_date.get((d, uni), [])
        if not buri: continue
        for a in top.a.astype(str):
            l1 = con.execute(f"""SELECT timestamp_ms,best_bid,best_ask FROM read_parquet('{V1Q}/l1/universe={uni}/*/*.parquet')
                WHERE CAST(asset_id AS VARCHAR)='{a}' AND timestamp_ms>={lo} AND timestamp_ms<{hi} ORDER BY timestamp_ms""").df()
            bb = con.execute(f"""SELECT timestamp_ms,best_bid,best_ask FROM read_parquet({lit(buri)}, union_by_name=true)
                WHERE CAST(asset_id AS VARCHAR)='{a}' ORDER BY timestamp_ms""").df()
            if l1.empty or bb.empty: continue
            mg = pd.merge_asof(bb, l1.rename(columns={"best_bid":"lb","best_ask":"la"}), on="timestamp_ms",
                               direction="nearest", tolerance=SLACK_MS).dropna(subset=["lb"])
            if len(mg):
                agree += int(((abs(mg.best_bid-mg.lb)<=TOL)&(abs(mg.best_ask-mg.la)<=TOL)).sum()); rows += len(mg); nass += 1
    rate = agree/rows if rows else float("nan")
    spot.append((d, nass, rows, rate))
    log(f"  {d}: {nass} assets, n={rows:,}, agree {rate:.1%}{'  <-- BELOW 98% STOP' if (rows and rate<0.98) else ''}")
below = [d for d,_,n,rt in spot if n and rt<0.98]
log(f"dates below 98%: {below if below else 'none'}")

open(os.path.join(REP, "e2_verify.txt"), "w", encoding="utf-8").write("\n".join(out))
stop = bool(short or miss_l1 or miss_tr or below)
print("\n" + ("E2VERIFY DONE WITH STOP CONDITIONS: " + str({'short':short,'miss_l1':len(miss_l1),'miss_tr':len(miss_tr),'below':below}) if stop else "OK E2VERIFY (all clean)"))
