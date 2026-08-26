---
title: "Handoff — L2 ingestion pipeline recovery: one truncated shard, a three-day silent outage, and the durable fix"
created: 2026-08-25
status: closed
owner: alvaro
project: mm
para: archive
hubs:
  - strat_market_making
  - COWORK
  - TODO
tags:
  - handoff
  - data
  - data-quality
  - infrastructure
  - capture
  - market-making
---

# Handoff 2026-08-25 — L2 ingestion pipeline recovery

> Hubs: [[strat_market_making]] · [[COWORK]] · [[TODO]] · Pipeline docs: [[mm_vps_capture_setup]] · Data plan this feeds: [[DATA_LAYER_PLAN]] · Companion session: [[2026-08-25_cowork_data_layer_session]]

## Plain-English Summary

- **What happened.** The Hetzner VPS that runs our 24/7 Polymarket L2 order-book capture hard-rebooted on **2026-08-21 21:13:15 UTC**. The reboot left three capture shards truncated. The hourly compression job hit the first bad file and **crashed the entire run** — every hour, for three days. No new Parquet was produced, so the cloud sync had nothing to push and the local disk was never pruned. R2's Parquet archive froze at 2026-08-21 and the VPS disk climbed to **93%**.
- **What we did.** Backed up all raw to cloud first, then salvaged the three corrupt shards, fixed the real bug (one bad shard no longer aborts the run), resolved a pile of committed merge conflicts in the deploy scripts, and — on a deliberate call to finish fast and safely — **stopped the collector**, freed the disk, and left the box inert. The missing Parquet was intentionally **not** rebuilt on the VPS; it will be rebuilt from raw off-server.
- **The surprising find.** The corruption was **not** a simple truncation. After the reboot, capture restarted and *appended a second, complete gzip member* to each file. The naive "keep everything before the break" salvage would have silently discarded ~90% of each shard. A member-aware rebuild recovered essentially the whole hour. **Real data loss is a single ~9-minute window per universe**, not the ~47 minutes the naive method implied.
- **State now.** Capture and all timers are **stopped and disabled** (reversible with one command). R2 raw covers **2026-08-21 → 2026-08-26**; R2 Parquet covers **2026-06-19 → 2026-08-21**; the Parquet gap after 08-21 is intentional and rebuildable from the R2 raw window. VPS disk: **21 GB free / 44%**. The durable fix is deployed and committed with a passing test.

---

## 1. What broke, and the chain that made one file a three-day outage

The pipeline is four cooperating pieces on the VPS (`/opt/epsilon/l2_ingestion`): **capture** (always-on WS recorder → hourly `*.jsonl.gz` raw shards), **compression** (hourly: raw → typed Parquet), **sync** (6-hourly: upload to R2, prune local), **expiry** (trim R2 raw once Parquet is confirmed). Full map: [[mm_vps_capture_setup]].

The failure chain, link by link:

1. **Hard reboot at 2026-08-21 21:13:15 UTC.** Every systemd unit shows that start time. The three shards open at that moment were truncated: `esports_21`, `politics_negrisk_21`, `unknown_21` (all under `data/raw/2026-08-21/`).
2. **Compression crashed on the first bad shard.** `parse_shard()` opened `esports_21.jsonl.gz`, hit the truncated gzip member, and raised `zlib.error: Error -3 while decompressing data: invalid stored block lengths`. That exception propagated out of `process_files()` and killed the whole run — **before any of the ~261 healthy pending shards were processed.** Because the bad shard sorts near the front, every hourly run died in the same place, ~1 second in.
3. **No Parquet → no sync progress.** `sync_cloud.sh` only has new Parquet to push if compression produced some. It didn't, so R2's `parquet/` prefix stayed frozen at 2026-08-21.
4. **No confirmed Parquet → no pruning.** Local raw is pruned only once its Parquet is confirmed in R2 (per-file verified). With compression dead, nothing was ever confirmed, so **local raw was never pruned** and grew ~3 GB/day.
5. **Disk climbed to 93%** (from a normal ~50%). Nothing alerted, because a *crashing* hourly job looks the same to a casual glance as a *quiet* one.

**The single link that would have prevented the outage entirely: step 2.** If one unreadable shard were skipped instead of aborting the run, the other ~261 shards would have compressed, sync would have pushed them, pruning would have run, and the disk would never have filled. The corrupt file was never the problem — **the fatal design was that one corrupt file could stop everything.** (A secondary safety net that would have shortened the outage from days to hours: an alert on `compress.service` entering a failed state. That is out of scope here but noted as an open item.)

---

## 2. Recovery, in the order it was done (upload before delete, always)

### 2.1 Verify (read-only)
Confirmed the diagnosis held: capture **active** and untouched; a `gzip -t` sweep over all raw found **exactly** the three corrupt files and no others; free disk above the 2 GB hard floor.

### 2.2 Raw to R2 first
Pushed all raw to `r2:epsilon-polymarket-data/raw` with `rclone copy` (never `sync` — copy never deletes on the remote), excluding only the currently-open hour. Verified with `rclone check --size-only --one-way`: every closed shard present in R2 at matching byte size. **Nothing local was deleted until this passed.**

### 2.3 Salvage — the multi-member discovery
The prescribed salvage was `zcat file | head -n -1 | gzip` — "keep everything up to the break, drop the last partial line." Inspecting the bytes showed why that is wrong here. Each corrupt file is actually **two gzip members concatenated**:

```
[ member 0 : pre-reboot capture, TRUNCATED mid-write by the crash ]
[ member 1 : post-reboot capture, COMPLETE, written after capture restarted and re-opened the same hourly file in append mode ]
```

`zcat` decodes member 0, hits its break, and **stops** — it never reaches member 1, which holds the bulk of the hour. The pipeline's own `gzip.open()` fails the same way. So the naive salvage (and the naive read) would keep only the small pre-reboot fragment.

We wrote a **member-aware rebuild** (`compression/salvage_truncated_shard.py`): decode member 0 up to its break (dropping only the final partial line), decode member 1 in full, concatenate, re-gzip, and re-validate as line-delimited JSON. Results:

| shard | naive salvage (member 0 only) | **member-aware rebuild** | healthy neighbour hour |
|---|---:|---:|---:|
| `esports_21` | 22,880 lines | **395,404 lines** | 357,722 (`esports_22`) |
| `politics_negrisk_21` | 27,914 lines | **386,807 lines** | 394,549 (`politics_negrisk_22`) |

`unknown_21` is the WebSocket's global new-market firehose — it carries no order-book content and never produces Parquet — so it was moved aside (`.corrupt`) rather than salvaged.

**Unit of the numbers above:** "lines" = one capture-envelope JSON record (one WS message). A healthy neighbour hour is shown so the recovery is sanity-checked against normal volume; the rebuilt hour is in the same ballpark, confirming we recovered essentially the whole hour, not a fragment.

### 2.4 Validate the rebuilt shards as JSONL (not just as gzip)
`gzip -t` only proves the container is intact; it says nothing about whether the bytes near the member boundary form valid JSON. So each rebuilt file was parsed **line by line**:

- **esports_21:** 395,404 lines, **0 dropped** as bad JSON.
- **politics_negrisk_21:** 386,807 lines, **0 dropped** as bad JSON.

The fine-grained decode stopped cleanly at member 0's break, so no garbage lines survived into the output. (Had any been present, the tool drops and counts them; the production parser also skips unparseable lines via its `bad_lines` path, so a stray bad line could never crash compression regardless.)

**Timestamp ordering across the member boundary.** The ordering key is `timestamp_ms` (the Polymarket server timestamp). Per-member ranges (ms epoch → UTC):

| shard | member 0 range | member 1 range | boundary monotonic? |
|---|---|---|---|
| esports_21 | 21:00:00.0 → 21:04:17.9 | 21:00:03.7 → 22:00:00.0 | **No** |
| politics_negrisk_21 | 21:00:00.0 → 21:04:17.9 | 21:13:06.9 → 22:00:00.0 | Yes |

The esports "No" is **benign and expected**, not time-travel: after the reboot, capture re-subscribed and Polymarket re-sent a fresh `book` **snapshot** for each asset. A book snapshot carries the market's *last-update* timestamp, which for a quiet market can predate the reboot — so member 1 opens with events timestamped as early as 21:00:03 even though they were *received* at 21:13:17. Note also that `timestamp_ms` is **not globally sorted within a shard anyway** (6,651 / 8,438 local regressions across the two files — normal for a multi-asset stream). The replay engine sorts by `timestamp_ms` and re-anchors on `book` events, so this is handled correctly; the fact that member 1 *starts* with book snapshots is exactly what gap-recovery wants. No note needed for the backtest beyond "there is a gap here" (below).

### 2.5 How much was actually lost
The boundary timestamps pin it precisely. Member 0's last flushed record is **21:04:17.9**; member 1's first *received* record is **21:13:17.3**. The reboot was 21:13:15 and capture resumed ~2 seconds later — so **the loss is not downtime.** It is the **~9-minute gzip write buffer** (21:04:18 → 21:13:15) that was never flushed to disk before the hard reboot. Both universes lost the same window (both last-flushed at 21:04:17, both resumed at 21:13:17), consistent with a shared buffering cadence.

**Net: ~9 minutes of capture lost per universe on 2026-08-21, in the 21:00 hour.** Everything before 21:04:18 and after 21:13:17 in that hour was recovered.

---

## 3. The durable fix (the real deliverable)

Changed `compression/pipeline.py` so `process_files()` **quarantines-and-continues** instead of aborting:

- The per-shard `parse_shard()` call is wrapped to catch `zlib.error`, `EOFError`, `OSError`, `UnicodeDecodeError` (never `KeyboardInterrupt`/`SystemExit`). The write path stays fatal on purpose — a schema/row-count mismatch or a disk-full `OSError` during write is a real bug or a systemic condition, not a per-shard skip.
- On catch: log at **ERROR** with the path and exception, append the path to `data/parquet/_quarantine.txt` (append-immediately, same crash-safe style as `_processed.txt`), and continue to the next shard. A quarantined shard is **never** written to `_processed.txt`.
- The run summary reports the quarantine count, and `main()` **exits non-zero** if anything was quarantined — but only *after* processing everything it could, so systemd marks the run degraded without losing the good work.
- Quarantined shards are **retried every run** (not skipped like processed ones), so a salvaged shard is picked up automatically; the path is appended to the quarantine file only once (deduped) to keep it bounded.

**`_quarantine.txt` is a TO-SALVAGE list, not a discard list.** This is the crucial framing, because quarantine-and-continue and multi-member salvage are **complementary, not substitutes**: the quarantine fix keeps the pipeline *alive* past a bad shard, but on its own it would have set these three shards aside and — without salvage — left the same data unrecovered. So a quarantined shard is **data pending recovery**. The ERROR log points the operator at the recovery tool, and the recovery procedure is:

```bash
cd /opt/epsilon/l2_ingestion
# dry-run report (members, recoverable lines, bad-json count, boundary timestamps):
venv/bin/python compression/salvage_truncated_shard.py data/raw/<date>/<shard>.jsonl.gz
# repair in place (backs the original up to *.corrupt, writes a clean single-member file):
venv/bin/python compression/salvage_truncated_shard.py --apply data/raw/<date>/<shard>.jsonl.gz
# the next compression run then parses it cleanly and removes it from the to-salvage set.
```

We deliberately did **not** auto-rebuild inside the compression run: the pipeline's contract is that it never edits its source shards (that invariant is what makes sync's size-based prune safe), so recovery stays an explicit, logged, operator-run step.

**Test:** `compression/tests/test_quarantine.py` writes one valid + one truncated shard, runs `process_files()`, and asserts the valid shard produced Parquet and landed in `_processed.txt` while the truncated one landed in `_quarantine.txt` (and *not* in `_processed.txt`), and that a second run keeps the quarantine entry deduped. Passes locally.

**Deploy:** `pipeline.py` and `salvage_truncated_shard.py` were copied to the VPS and confirmed **byte-identical** to the repo (`cmp`), and both `py_compile` on the box. A smoke test with the *deployed* code (temp state files, so production `_quarantine.txt` is untouched) confirmed a corrupt throwaway shard is quarantined while a valid one is processed — **SMOKE_PASS**. The fix is correct and in place for whenever capture is restarted.

---

## 4. Repo hygiene — resolved committed merge conflicts

Three deploy files were committed with unresolved conflict markers, so the repo could not be redeployed and nobody could read the actual deployed behaviour.

- **`sync/sync_cloud.sh`** (33 markers, nested `HEAD` / `7703de6c` / `4db6a716`). The **server copy is the authority** — it is clean, `bash -n`-valid, and is what has actually been running. Copied it down and replaced the repo version verbatim; the repo copy is now **byte-identical** to the server (`cmp` clean). The winning content is the **`4db6a716`** side: six-hourly sync, `rclone copy` (never `sync`) for both raw and parquet, parse-confirm raw pruning *before* re-upload (`prune_raw_if_parsed`), per-file size-verified pruning, `RAW_RETENTION_DAYS=3` / `PARQUET_RETENTION_DAYS=7`, and a `DISK_ALERT_PCT=90` backstop. The older `HEAD` side (parquet via `rclone sync`, age-based raw retention, no disk guard) lost.
- **`deploy/DEPLOY.md`** (3 conflicts): kept the frontmatter block; kept the more emphatic "⚠️ DO NOT TOUCH" warning; and kept `bash sync/sync_cloud.sh` over `./venv/bin/bash …` (the venv has no `bash` binary — that variant was simply wrong). Also corrected the stale "Disk safety" paragraph, which still described the old age-based 7-day raw retention, to describe the current parse-confirm per-file-verified pruning.
- **`deploy/R2_HANDOVER.md`** (1 conflict): kept the frontmatter block.

`grep -rn '^<<<<<<<' infrastructure/` returns nothing; `bash -n sync/sync_cloud.sh` passes.

---

## 5. Wrap-up decision — stop the collector, don't rebuild Parquet on the box

Late in the session the call was made (Cowork-directed) to **finish fast and safely rather than exhaustively**, on the reasoning that raw is the only irreplaceable artifact and it is already safe in R2, while Parquet is a *derived* artifact that [[DATA_LAYER_PLAN]] (Phase B) rebuilds from raw **off-server, on a proper machine.** Rebuilding it on a 2-vCPU box at 93% disk would be an hour of risk for no durable gain. So:

1. **Collector stopped and disabled** — `capture.service`, `discovery.timer`, `compress.timer`, `sync.timer`, `expire-raw.timer` all `disable --now`. The box is inert; disk stops growing. **Reversible** with `systemctl enable --now …`.
2. **Final raw push** with capture stopped (nothing open to exclude): `rclone check` → **0 differences, 216 files**. This also overwrote R2's corrupt `esports_21`/`politics_negrisk_21` with the rebuilt versions (desirable — see below).
3. **Compression backlog SKIPPED — deliberate.** The ~261-shard backlog was not compressed on the VPS. The missing Parquet (2026-08-21 hour 21 → 2026-08-26) is rebuildable from the R2 raw window and will be produced off-server per [[DATA_LAYER_PLAN]].
4. **Disk freed** with verified deletes only (a local raw shard deleted only when R2 holds it at the same byte size): 213 shards, **18.75 GB freed**, disk now **21 GB free / 44%**. The three `.corrupt` forensic originals were kept (they are *not* the objects now in R2 under those base names, so the "never delete unless verified in R2" rule correctly protects them).

**R2 object-change note (so nobody is later confused):** `raw/2026-08-21/esports_21.jsonl.gz` and `…/politics_negrisk_21.jsonl.gz` in R2 **changed size** during this session — Phase 1 pushed the corrupt originals under their real names, and the final push replaced them with the rebuilt (clean, smaller) versions. The original corrupt bytes are still preserved in R2 alongside them as `*.jsonl.gz.corrupt` (uploaded by the final `rclone copy`, which does not filter by extension). Both the rebuilt file and its forensic original therefore live in R2; the pipeline and sync only ever glob `*.jsonl.gz`, so the `.corrupt` objects are inert.

---

## 6. Coverage and current state (verified from R2 and the box)

| thing | value |
|---|---|
| R2 **raw** coverage | **2026-08-21 → 2026-08-26** (raw is expired after Parquet confirmation, so only the recent unparsed window survives) |
| R2 **Parquet** coverage | **2026-06-19 → 2026-08-21** (the permanent keeper; frozen at the outage) |
| Parquet gap | 2026-08-21 (from hour 21) → 2026-08-26 — **intentional, rebuildable from the R2 raw window** |
| VPS disk | 21 GB free / 44% (was 93%) |
| Services | capture + discovery/compress/sync/expire-raw timers **stopped and disabled** |
| Durable fix | deployed byte-identical, smoke-tested on box, committed with passing test |
| Corrupt shards | 0 remaining under `data/raw/*.jsonl.gz`; 3 `.corrupt` forensic originals retained locally + in R2 |

> ⚠️ Runbook correction: the wrap-up instruction assumed "R2 raw covers 2026-06-19 → 2026-08-25." That is not how retention works — raw is deleted once its Parquet is confirmed, so R2 raw only holds **08-21 → 08-26**. The June→August history is preserved as **Parquet** (06-19 → 08-21), which is the permanent keeper. This does not change any decision (the rebuildable window is exactly the raw that survives), but the coverage numbers above are the accurate ones.

---

## 7. The reasoning, for a reader who trades but isn't a systems engineer

**Q1 — Why did one corrupt file take the whole pipeline down for three days without anyone noticing?** Because the compression job processed shards in a single loop with no per-file guard: the first unreadable file raised an exception that escaped the loop and killed the run, so nothing behind it was processed. That starved every downstream step — no Parquet, so nothing to sync; nothing confirmed in the cloud, so no local pruning; so the disk filled. It was silent because a job that *crashes* one second in and a job that has *nothing to do* both just... end, leaving no obvious signal without watching the service's exit state. The one link that would have prevented it: making a single bad shard a skip, not a full-run abort (now fixed).

**Q2 — Why upload to R2 before deleting anything locally, even though it's slower?** Because deletion is irreversible and upload is not yet proven until it's verified. If we had reclaimed disk first — deleted raw to make room — and *then* discovered the upload had silently missed a file (R2 intermittently 501s; the current-hour shard can't be copied while it's being written), that file would be gone with no second copy. Upload-then-verify-then-delete means at every instant either the local copy exists or a byte-verified remote copy does. The cost is a slower sequence; the benefit is that no single failure loses data. (Concretely: the earlier `rclone` attempt that exited code 6 had *not* copied the open shard — deleting first would have trusted an upload that hadn't happened.)

**Q3 — Why quarantine-and-continue, rather than skip-silently or fail-fast?** *Skip-silently* (swallow the error, keep going, exit 0) would have kept the pipeline alive but hidden a real problem — you'd only find the missing data much later, in an analysis, with no breadcrumb. *Fail-fast* (the old behaviour: abort the whole run) is loud but catastrophic — one bad file blocks all the good ones, which is exactly the outage we just had. *Quarantine-and-continue* is the synthesis: process every shard you can (so a single bad file costs you only that file), record the bad one durably in a to-salvage list, and **exit non-zero after finishing** so systemd still flags the run degraded. You get fail-fast's visibility without its collateral damage — the good work lands *and* the alarm rings.

**Q4 — How much data was lost, and would the capture gate catch it?** ~9 minutes per universe on 2026-08-21 (21:04:18 → 21:13:17 UTC), i.e. the unflushed write buffer, not downtime. For a market-making backtest that replays 2026-08-21, that hour has a 9-minute hole; the first post-hole events are fresh `book` snapshots, so a well-behaved replay re-anchors cleanly rather than quoting off stale state. **Would `polymarket/research/mm_eval/capture_gate.py` catch the day as degraded?** Yes. The VPS archive ships no `capture_gaps` sidecar (the hard crash prevented capture from ever writing one), so the gate falls back to **heartbeat inference**: a universe-wide receive-clock silence longer than `gap_infer_ms` (default **30 s**) is flagged as an inferred gap and marks books suspect until their next snapshot. Our silence is ~540 s — 18× the threshold — so the gate would raise `gaps_inferred` and drive up the stale percentage for that slice. It would **not** pass unnoticed. Caveat, in the spirit of honest gating: this is *labelled inference, not ground truth* — the authoritative `capture_gaps.jsonl` was lost with the reboot, so the gate is inferring the gap from the data, not reading a recorded marker.

**Q5 — What's the next most likely way this pipeline fails, now that this one is fixed?** The pruning safety model trusts **byte-size equality** as a proxy for content equality (sound only because every producer path is write-once-per-path). The most likely next failure is that invariant being violated — e.g. a future change that rewrites or recompacts a Parquet file in place, or a raw shard that gets re-opened and appended (which is *exactly* what the reboot did to these three shards). If a path's content changes while its size happens to match a stale R2 object, the verified-prune would delete a local file that R2 does not actually match. **Cheapest early-warning signal:** periodically (or in the prune itself, on a sample) compare a content hash — `rclone check --checksum` on a small random sample of already-pruned paths, or `rclone lsf --format ph --hash md5` — and alarm on any hash mismatch at equal size. That catches an invariant break before it can delete good data. (Second-place, lower-severity: the disk `DISK_ALERT_PCT=90` guard only fires *inside* a sync run, so a stuck/disabled sync — like the one we just had — never trips it; an external "has sync succeeded in the last N hours / is `compress.service` in a failed state" check would have turned this three-day outage into a three-hour one.)

---

## 8. Open items (not done, by decision or scope)

- **Rebuild the missing Parquet off-server** for 2026-08-21 (hour 21) → 2026-08-26 from the R2 raw window — [[DATA_LAYER_PLAN]] Phase B. This is the deliberate deferral, not an oversight.
- **Restart the collector when wanted** — `systemctl enable --now capture.service discovery.timer compress.timer sync.timer expire-raw.timer`. The deployed pipeline is now quarantine-safe, so a future unclean reboot will degrade, not silently stall.
- **Alerting gap (recommended, out of scope here):** nothing pages when `compress.service` fails or when sync hasn't succeeded in hours. Adding that is the highest-leverage follow-up; it is what turns "three-day silent outage" into "someone gets pinged in an hour."
- **Content-hash spot-check in the prune** (Q5) — cheap insurance against the size-proxy invariant ever breaking.
- Untouched by design (all pre-known): capture daemon memory growth, the 418-vs-700 asset question, the missing `capture_gaps.jsonl` backup, the `crypto_control` universe. Out of scope per the task.

## 9. Where the artifacts live

- Durable fix: [pipeline.py](../../infrastructure/data/l2_ingestion/compression/pipeline.py) · salvage tool: [salvage_truncated_shard.py](../../infrastructure/data/l2_ingestion/compression/salvage_truncated_shard.py) · test: [test_quarantine.py](../../infrastructure/data/l2_ingestion/compression/tests/test_quarantine.py)
- Resolved deploy files: `sync/sync_cloud.sh`, `deploy/DEPLOY.md`, `deploy/R2_HANDOVER.md`
- Pipeline reference + corrections log: [[mm_vps_capture_setup]]
