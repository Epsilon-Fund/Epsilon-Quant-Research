"""E2 · Build l1 (deduped to touch-moving rows) and trades from the archive.

Source: price_change (E0). File selection from a pre-listed rclone manifest -> explicit URIs
(no flaky S3 glob-LIST). l1 is deduped on (best_bid,best_ask) change in received_ns order.

l1 is built PER DAY (part-<date>.parquet inside each month partition) so each window-sort is
small — machine has ~2.6 GB free, and a month-level sort of 300M+ rows spilled for hours. Per
day the sort is bounded and a sleep/reboot costs one day, not a whole month. Dedup is per-asset
within a day (re-anchors at the day boundary; over-keeps ~1 row/asset/day — negligible). A month
partition already written as a single part-0.parquet (politics, built earlier) is kept as-is.
trades are built per month (small). RESUMABLE + ROBUST (retry transient timeouts, raise loudly).
Final pass: e2_counts.csv + obs_stats.parquet. asset_id stays TEXT.
"""
import os, csv, time, configparser
from collections import defaultdict
import duckdb

HERE = os.path.dirname(__file__)
V1 = os.path.join(HERE, "..", "data", "research_v1"); V1Q = V1.replace(chr(92), "/")
MAN = os.path.join(V1, "pc_file_manifest.txt")
SCRATCH = r"C:\Users\alvar\AppData\Local\Temp\claude\c--Users-alvar-Projects-Epsilon-Fund-Epsilon-Quant-Research\72db32b9-68e9-4f75-b9ab-149d292064e9\scratchpad\duckdb_tmp"
os.makedirs(V1, exist_ok=True); os.makedirs(SCRATCH, exist_ok=True)
MONTHS = ["2026-06", "2026-07", "2026-08"]; UNIS = ["politics_negrisk", "esports"]
S3 = "s3://epsilon-polymarket-data/parquet/"

# manifest -> explicit URIs, grouped by (uni,table,month) and (uni,table,date)
by_month = defaultdict(list); by_date = defaultdict(list); dates_of = defaultdict(set)
for ln in open(MAN, encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    p = ln.split("/")
    if len(p) != 3: continue
    date, uni, fname = p; month = date[:7]
    for table in ("price_change", "trades"):
        if fname.startswith(f"{table}_{uni}_"):
            by_month[(uni, table, month)].append(S3 + ln)
            by_date[(uni, table, date)].append(S3 + ln)
            dates_of[(uni, table, month)].add(date)

cfg = configparser.ConfigParser(); cfg.read(r"C:\Users\alvar\AppData\Roaming\rclone\rclone.conf")
r = cfg["r2"]; ep = r["endpoint"].replace("https://", "")
con = duckdb.connect(config={"threads": 2})
con.execute("INSTALL httpfs; LOAD httpfs;")
# machine has volatile free RAM (seen as low as 0.5 GB). Cap well below that and spill the
# rest to the 272 GB SSD temp dir, rather than fighting the OS for RAM (which wedged the build).
con.execute(f"SET memory_limit='1GB'; SET temp_directory='{SCRATCH.replace(chr(92),'/')}'; SET max_temp_directory_size='200GB';")
con.execute("SET preserve_insertion_order=false;")   # lets the sort/scan stream, less memory
con.execute("SET http_retries=10; SET http_retry_wait_ms=800; SET http_timeout=180000;")
con.execute(f"""CREATE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID '{r['access_key_id']}',
  SECRET '{r['secret_access_key']}', ENDPOINT '{ep}', REGION 'auto', URL_STYLE 'path', USE_SSL true);""")

def transient(e):
    s = str(e).lower()
    return any(w in s for w in ("timeout","http","connection","curl","503","reset","temporarily"))
def lit(u): return "[" + ",".join("'"+x+"'" for x in u) + "]"
def count(uris, retries=6):
    if not uris: return 0
    for i in range(retries):
        try: return con.execute(f"SELECT COUNT(*) FROM read_parquet({lit(uris)}, union_by_name=true)").fetchone()[0]
        except Exception as e:
            if transient(e) and i < retries-1: print(f"    count retry {i+1}", flush=True); time.sleep(4*(i+1)); continue
            raise
def run(sql, outp, retries=6):
    for i in range(retries):
        try: con.execute(sql); return
        except Exception as e:
            if os.path.exists(outp):
                try: os.remove(outp)
                except OSError: pass
            if transient(e) and i < retries-1: print(f"    copy retry {i+1}: {str(e)[:60]}", flush=True); time.sleep(6*(i+1)); continue
            raise
def valid(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return False
    try: con.execute(f"SELECT COUNT(*) FROM read_parquet('{path.replace(chr(92),'/')}')").fetchone(); return True
    except Exception: return False

DEDUP = """SELECT asset_id, timestamp_ms, received_ns, best_bid, best_ask,
    (best_bid+best_ask)/2.0 AS mid, (best_ask-best_bid)*100 AS spread_c
  FROM (SELECT CAST(asset_id AS VARCHAR) asset_id, timestamp_ms, received_ns, best_bid, best_ask,
               lag(best_bid) OVER w pb, lag(best_ask) OVER w pa
        FROM read_parquet({FILES}, union_by_name=true)
        WINDOW w AS (PARTITION BY asset_id ORDER BY received_ns))
  WHERE pb IS NULL OR best_bid IS DISTINCT FROM pb OR best_ask IS DISTINCT FROM pa
  ORDER BY asset_id, timestamp_ms, received_ns"""

t0 = time.time()
for uni in UNIS:
    for month in MONTHS:
        outdir = f"{V1Q}/l1/universe={uni}/month={month}"; os.makedirs(outdir, exist_ok=True)
        month_part = f"{outdir}/part-0.parquet"
        pc_dates = sorted(dates_of.get((uni, "price_change", month), set()))
        if valid(month_part):
            print(f"[{uni} {month}] l1: month-level part-0 already built ({count([month_part]):,}) — skip", flush=True)
        elif not pc_dates:
            print(f"[{uni} {month}] l1: genuinely no price_change files — skip", flush=True)
        else:
            # PER-HOUR (per file): each hourly price_change file fits in memory with no spill,
            # robust to this machine's volatile free-RAM. Existing valid day-parts are kept
            # (a whole date already covered -> skip it). Uncovered dates build one part per hour:
            # part-<date>-<hh>.parquet. Dedup re-anchors per hour (over-keeps ~1 row/asset/hour,
            # negligible). Cross-day/hour ordering is imposed by readers; each file is self-sorted.
            for date in pc_dates:
                daypart = f"{outdir}/part-{date}.parquet"
                if valid(daypart):
                    continue   # already built as a day-part earlier
                for uri in sorted(by_date[(uni, "price_change", date)]):
                    hh = uri.rsplit("_", 1)[1].split(".")[0]      # ..._<HH>.parquet
                    hourpart = f"{outdir}/part-{date}-{hh}.parquet"
                    if valid(hourpart):
                        continue
                    ts = time.time()
                    run("COPY (" + DEDUP.replace("{FILES}", lit([uri])) + f") TO '{hourpart}' (FORMAT parquet, COMPRESSION zstd)", hourpart)
                    print(f"  [{uni} {month}] {date} h{hh}: l1 {count([hourpart]):,} {time.time()-ts:.0f}s", flush=True)
            print(f"[{uni} {month}] l1 per-hour done: {len(pc_dates)} dates", flush=True)

        # trades — per month (small)
        toutdir = f"{V1Q}/trades/universe={uni}/month={month}"; os.makedirs(toutdir, exist_ok=True)
        toutp = f"{toutdir}/part-0.parquet"; trf = by_month.get((uni, "trades", month), [])
        if valid(toutp):
            print(f"[{uni} {month}] trades: already built — skip", flush=True)
        elif not trf:
            print(f"[{uni} {month}] trades: genuinely none", flush=True)
        else:
            run(f"""COPY (SELECT CAST(asset_id AS VARCHAR) asset_id, timestamp_ms, received_ns,
                     price, size, side, fee_rate_bps, CAST(transaction_hash AS VARCHAR) transaction_hash
                     FROM read_parquet({lit(trf)}, union_by_name=true)
                     ORDER BY asset_id, timestamp_ms, received_ns) TO '{toutp}' (FORMAT parquet, COMPRESSION zstd)""", toutp)
            print(f"[{uni} {month}] trades built: {count([toutp]):,} ({len(trf)}f)", flush=True)

# reconciliation counts
print("computing reconciliation counts…", flush=True)
counts = []
for uni in UNIS:
    for month in MONTHS:
        outdir = f"{V1Q}/l1/universe={uni}/month={month}"
        parts = [f"{outdir}/{f}".replace(chr(92),'/') for f in os.listdir(outdir) if f.endswith(".parquet")] if os.path.isdir(outdir) else []
        parts = [p for p in parts if valid(p)]
        raw = count(by_month.get((uni,"price_change",month), []))
        kept = count(parts) if parts else 0
        toutp = f"{V1Q}/trades/universe={uni}/month={month}/part-0.parquet"
        traw = count([toutp]) if valid(toutp) else 0
        counts.append(dict(universe=uni, month=month, l1_raw=raw, l1_kept=kept, trades=traw, l1_parts=len(parts)))
with open(os.path.join(V1, "e2_counts.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["universe","month","l1_raw","l1_kept","trades","l1_parts"]); w.writeheader(); w.writerows(counts)
tot = {k: sum(c[k] for c in counts) for k in ("l1_raw","l1_kept","trades")}
print(f"TOTALS: l1_raw={tot['l1_raw']:,} l1_kept={tot['l1_kept']:,} ({100*tot['l1_kept']/max(tot['l1_raw'],1):.2f}%) trades={tot['trades']:,}", flush=True)
missing = [f"{c['universe']} {c['month']}" for c in counts if c["l1_raw"] > 0 and c["l1_kept"] == 0]
if missing:
    raise SystemExit(f"INCOMPLETE BUILD — l1 missing for partitions with data: {missing}. Re-run (resumable).")

print("computing obs_stats…", flush=True)
con.execute(f"""CREATE TEMP TABLE l1s AS
  SELECT asset_id, MIN(timestamp_ms) first_seen, MAX(timestamp_ms) last_seen,
         COUNT(DISTINCT timestamp_ms // 86400000) n_days, COUNT(*) n_l1_events,
         median(mid) median_mid, median(spread_c) median_spread, arg_max(mid, timestamp_ms) last_mid
  FROM read_parquet('{V1Q}/l1/*/*/*.parquet') GROUP BY asset_id""")
con.execute(f"""CREATE TEMP TABLE trs AS
  SELECT asset_id, COUNT(*) n_trades FROM read_parquet('{V1Q}/trades/*/*/*.parquet') GROUP BY asset_id""")
con.execute(f"""COPY (
  SELECT l.asset_id, l.first_seen, l.last_seen, l.n_days, l.n_l1_events,
         l.median_mid, l.median_spread, l.last_mid, COALESCE(t.n_trades,0) n_trades
  FROM l1s l LEFT JOIN trs t USING(asset_id)) TO '{V1Q}/obs_stats.parquet' (FORMAT parquet)""")
nstats = con.execute(f"SELECT COUNT(*) FROM read_parquet('{V1Q}/obs_stats.parquet')").fetchone()[0]
print(f"obs_stats.parquet: {nstats} assets. E2 build done in {time.time()-t0:.0f}s. OK E2BUILD", flush=True)
