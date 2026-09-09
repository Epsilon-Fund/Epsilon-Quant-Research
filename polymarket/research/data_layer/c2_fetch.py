"""C2 · Fetch Gamma. One verbatim file per condition, resumable, silent-failure-guarded.

Corrected design approved after Step B:
  - two-pass union per batch: default (open) + closed=true (resolved)
  - repeated `condition_ids` param, <=100 ids/call, limit = batch size
  - mandatory User-Agent
  - `.notfound` ONLY when a condition is absent from BOTH passes
  - assertion: every returned market's conditionId was in the batch (catches an ignored param)
  - reconcile at end: len(targets) == cached + notfound, exactly

Network only here. No secrets used (Gamma is public). Reads targets.parquet, writes
gamma_cache/<condition_id>.json (the exact Gamma market object) or <condition_id>.notfound.
"""
import os, sys, json, time, csv, urllib.request, urllib.parse, urllib.error
import duckdb

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "gamma")
CACHE = os.path.join(DATA, "gamma_cache")
os.makedirs(CACHE, exist_ok=True)
LOG = os.path.join(DATA, "fetch_log.csv")
BATCHLOG = os.path.join(DATA, "fetch_batchlog.csv")
BASE = "https://gamma-api.polymarket.com/markets"
UA = {"User-Agent": "epsilon-research/1.0 (data-layer build; contact operator)"}
BATCH = 100
PAUSE = 0.25          # politeness between calls
RETRIES = 3

con = duckdb.connect()
targets = [r[0] for r in con.execute(
    f"SELECT DISTINCT condition_id FROM read_parquet('{DATA.replace(chr(92),'/')}/targets.parquet')").fetchall()]

def done(cid):
    return os.path.exists(os.path.join(CACHE, cid + ".json")) or os.path.exists(os.path.join(CACHE, cid + ".notfound"))

pending = [c for c in targets if not done(c)]
print(f"targets={len(targets)} already_done={len(targets)-len(pending)} pending={len(pending)}", flush=True)

def call(cids, closed):
    params = [("condition_ids", c) for c in cids] + [("limit", str(len(cids)))]
    if closed:
        params.append(("closed", "true"))
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read().decode()
                return json.loads(body), r.status, len(body)
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504) and attempt < RETRIES:
                time.sleep(1.5 * attempt); continue
            return None, e.code, 0
        except (urllib.error.URLError, TimeoutError):
            if attempt < RETRIES:
                time.sleep(1.5 * attempt); continue
            return None, -1, 0

logf = open(LOG, "a", newline="", encoding="utf-8"); logw = csv.writer(logf)
if os.path.getsize(LOG) == 0 if os.path.exists(LOG) else True:
    pass
blogf = open(BATCHLOG, "a", newline="", encoding="utf-8"); blogw = csv.writer(blogf)

t0 = time.time(); ncached = nnf = 0
for i in range(0, len(pending), BATCH):
    batch = pending[i:i+BATCH]
    reqset = set(batch)
    found = {}   # cid -> market object
    for closed in (False, True):
        data, status, nbytes = call(batch, closed)
        if data is None:
            blogw.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), len(batch), "PASS_"+("closed" if closed else "open"), "ERR", status])
            print(f"  batch {i//BATCH}: {'closed' if closed else 'open'} pass HTTP {status} after retries -> STOP", flush=True)
            raise SystemExit(f"fetch failed on batch starting {batch[0]} status={status}")
        # assertion: no market outside the requested set (ignored-param guard)
        for m in data:
            cid = m.get("conditionId")
            if cid not in reqset:
                raise SystemExit(f"ASSERTION FAILED: response carried conditionId {cid} not in batch — param ignored?")
            found.setdefault(cid, m)
        trunc = "TRUNC?" if len(data) == len(batch) else ""
        blogw.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), len(batch), "PASS_"+("closed" if closed else "open"), len(data), status, trunc])
        time.sleep(PAUSE)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for cid in batch:
        if cid in found:
            obj = found[cid]; body = json.dumps(obj)
            open(os.path.join(CACHE, cid + ".json"), "w", encoding="utf-8").write(body)
            logw.writerow([cid, 200, len(body), ts]); ncached += 1
        else:
            open(os.path.join(CACHE, cid + ".notfound"), "w", encoding="utf-8").write("")
            logw.writerow([cid, 404, 0, ts]); nnf += 1
    if (i//BATCH) % 10 == 0:
        logf.flush(); blogf.flush()
        print(f"  {i+len(batch)}/{len(pending)}  cached+={ncached} notfound+={nnf}  {time.time()-t0:.0f}s", flush=True)

logf.close(); blogf.close()

# reconcile
allj = len([f for f in os.listdir(CACHE) if f.endswith(".json")])
allnf = len([f for f in os.listdir(CACHE) if f.endswith(".notfound")])
print(f"\nRECONCILE: targets={len(targets)}  cached={allj}  notfound={allnf}  sum={allj+allnf}", flush=True)
if allj + allnf != len(targets):
    raise SystemExit(f"RECONCILE FAILED: {allj}+{allnf} != {len(targets)}")
print(f"fetch done in {time.time()-t0:.0f}s. OK C2", flush=True)
