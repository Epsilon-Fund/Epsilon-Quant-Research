# STEP K — make getting the data a solved problem, and give the repo a front door   [DONE]

Written by Claude Code, 2026-08-31. Two onboarding blockers, both found by re-reading the repo as a
person who has never seen it: (K1) the only documented way to get a local copy of the 1.09 GB library
was `rclone copy`, which needs rclone installed *and* an `[r2]` remote already in `rclone.conf` —
Justin has neither; (K2) the repo-root README predates everything Step E–J built and had no way in.
Both are fixed. K3 was run end-to-end against a scratch copy (not the built library), and the run
surfaced a real bug that is now fixed.

## Plain-English Summary

- **What shipped:** a pure-Python R2 downloader `polymarket/research/scripts/fetch_data.py` (no
  rclone), a front-door section on the repo-root `README.md`, and doc edits that make
  `python scripts/fetch_data.py` the documented way to get the data (rclone demoted to a one-liner,
  copy-only warning kept).
- **Why:** a newcomer's first hour must not require setting up a *second, different* credential path
  (rclone) for the one step that should be easiest. The loader already reads R2 with three env vars;
  the fetch script now uses the exact same three.
- **Proof:** the script was actually run — dry-run, full fetch, a delete-a-few resume test, and a
  `check_setup` + dashboard load — all against a **scratch** dir, which was then deleted. All four
  verification checks pass; a full fetch took **83 s**.
- **One real bug caught by running it:** the post-fetch verification counted the script's own
  `_fetch_receipt.json` as a 793rd file, so any *re-run* failed the file-count/byte checks. Fixed
  (verification now excludes local-only artifacts). This is exactly what "actually run it" is for.
- **Status:** Step K complete. Committed locally on `alvaro` (code only, nothing under `data/`).
  The `git push origin alvaro` from Step J is still the operator's pending action.

---

## K1 — one command that fetches the library, no rclone

### What `fetch_data.py` is

A pure-Python (boto3), S3-compatible downloader for the `research/v1` prefix of the R2 bucket. A
newcomer runs one command and needs only the same three env vars the loader already uses:

```
python scripts/fetch_data.py                 # → ./data/research_v1  (the loader default)
python scripts/fetch_data.py --dest <dir>    # fetch elsewhere
python scripts/fetch_data.py --dry-run       # list what it would fetch, transfer nothing
```

**How it meets each requirement in the brief:**

- **Same credentials as the loader.** It imports `epsilon_data._internal._r2_creds` — it does *not*
  contain a second credential loader. Env vars `EPSILON_R2_KEY_ID` / `_SECRET` / `_ENDPOINT` first,
  `rclone.conf [r2]` fallback second (so anyone already set up keeps working). No secret is ever
  printed or written.
- **boto3 added, loader stays boto3-free.** `boto3>=1.35.0` is in both `requirements.txt` and
  `pyproject.toml`, each under a comment saying it is for `fetch_data.py` only — the loader reads R2
  through DuckDB httpfs, not boto3.
- **Byte copy, not a re-encode.** `download_file` (GetObject) → write bytes → `os.replace`. No DuckDB
  round-trip, so encoding and row order are untouched and the anti-drift guarantee holds.
- **Resumable.** It skips any file already present at the same size and reports the skip count. A
  killed download is fixed by re-running the same command. (It writes to a `.part` temp file and
  renames on completion, so an interrupted file is never mistaken for a finished one.)
- **Parallel, modestly.** 8 workers (`--workers` to change).
- **Progress.** `files done/total · bytes done/total · running MB/s`, every 20 files.
- **Read-only against R2.** The script contains only `list_objects_v2` and `download_file`. There is
  no `put_object`, `delete_object`, or `copy_object` anywhere in it — by construction, not by
  discipline.

### The R2 connectivity fix (worth recording)

boto3 defaults to **virtual-hosted** addressing, which for R2 builds an unresolvable
`bucket.<account>.r2.cloudflarestorage.com` host — the first `list_objects_v2` hung until timeout.
The fix is to force **path-style** addressing plus explicit timeouts:

```python
Config(s3={"addressing_style": "path"}, signature_version="s3v4",
       connect_timeout=15, read_timeout=60, max_pool_connections=16)
```

Anyone writing another boto3-against-R2 tool in this repo needs the same config.

### Self-verification + receipt

After the copy the script proves it is complete and prints a verdict, then writes
`_fetch_receipt.json` next to `_manifest.json`. The four checks:

| # | Check | Expected | Got (scratch run) |
|---|---|---|---|
| 1 | file count on disk == objects under prefix | 792 | 792 ✓ |
| 2 | total bytes on disk == bytes in listing | 1,167,809,048 | 1,167,809,048 ✓ |
| 3 | `tokens.parquet` opens and row count | 30,772 | 30,772 ✓ |
| 4 | one `l1` + one `trades` partition non-empty | >0 / >0 | 101,051,502 / 7,227,528 ✓ |

On any failure it exits non-zero and names the failing check — a partial download that reports
success is the worst outcome, so the default is a loud, specific failure. On success it prints the
literal next commands (`set EPSILON_DATA_ROOT=<dest>` → `check_setup.py` → `streamlit run`).

### Docs updated to point at the script, not rclone

- **`epsilon_data/README.md`** — "Getting the data" now leads with **fetch it locally**
  (`fetch_data.py`, recommended), then read-straight-from-R2; rclone is demoted to a one-line
  "if you already have rclone configured" alternative. The **⚠️ copy-only warning is unchanged in
  strength** and now also notes the script is read-only by construction. A rewritten **Credentials**
  paragraph states plainly: the same three env vars serve both the R2-read path and the fetch; ask
  the operator; put them in a gitignored `.env`; they are read/write today. The Quickstart block and
  the Troubleshooting table both name `fetch_data.py`.
- **`HANDOVER.md`** — same reordering (fetch-locally first, rclone as the copy-only alternative),
  same three-env-var Credentials note.
- **`scripts/check_setup.py`** — when `tokens.parquet` is missing under the configured local root,
  the fix message now names the exact command: `python scripts/fetch_data.py --dest "<dir>"`.

---

## K2 — a front door on the repo-root README

The root `README.md` (dated June 22) had **zero** mentions of `epsilon_data`, `research_v1`,
`dashboard`, or `streamlit` in the context of the research library. Added a routing section
**near the top, above the branch-by-branch tour** (because this is the live work):

> ### Polymarket research library + dashboard (v1, Aug 2026)
> 30,772 tokens · 101 M L1 rows · 7.2 M trades · 1.09 GB — dataset, loader, Streamlit terminal.
> - Start here → `polymarket/research/HANDOVER.md`
> - Loader API + data dictionary → `polymarket/research/epsilon_data/README.md`
> - Adding a panel or a tool → `polymarket/research/CONTRIBUTING.md`
> - Get the data (no rclone needed) → `cd polymarket/research && python scripts/fetch_data.py`

All four link targets were confirmed to exist and resolve from the repo root. The section routes,
it does not explain — kept to ~8 lines as asked.

---

## K3 — tested the only way that counts (actually run)

Everything below ran against a **scratch** destination
(`C:\Users\alvar\AppData\Local\Temp\epsilon_fetch_scratch`) — the built library under
`data/research_v1` was never touched. Credentials were supplied as the three env vars only
(simulating a newcomer), read from the local `rclone.conf [r2]` and exported into the env.

**1. Dry-run** — `fetch_data.py --dry-run --dest <scratch>`:
> `R2 holds 792 files, 1.17 GB under research/v1/` · `would fetch 792, skip 0. No bytes transferred.`
Lists 792 objects, transfers nothing. ✓

**2. Full fetch** — `fetch_data.py --dest <scratch>`:
> `792 to fetch (1.17 GB)` … `fetch done in 78s (0 skipped, 792 fetched)` — all four checks `OK`,
> `_fetch_receipt.json` written. **Wall-clock 83 s** (13–16 MB/s on this connection). ✓

**3. Resume test** — deleted 3 files from the scratch copy (`tokens.parquet`, one `l1` partition, one
`trades` partition), then re-ran:
> `789 already present (same size, skipped); 3 to fetch (0.05 GB)` … `fetch done in 4s (789 skipped,
> 3 fetched)`.
Fetched **exactly** the 3 deleted, skipped the other 789. ✓ Resume works.

**4. `check_setup` + dashboard against scratch** — `EPSILON_DATA_ROOT=<scratch>`:
> `check_setup.py` all green: `loader read 30,772 tokens; reconciliation match=True`.
> Dashboard rendered headless (Streamlit `AppTest`) with **0 exceptions**. ✓

**5. Cleanup** — the scratch dir (1.1 GB) was deleted. (`rm -rf` succeeds in this Git-Bash
environment; nothing was left behind. R2 was never a deletion target at any point.)

### The bug that running it caught

On the **resume** re-run, checks 1 and 2 failed: `file_count 793 (expect 792)`,
`total_bytes` off by exactly **577 bytes**. Cause: the post-fetch verification globbed the whole dest
and counted the script's own `_fetch_receipt.json` (written by the *first* run, 577 bytes) as a
793rd data file. The receipt is local-only and never appears in R2's object listing, so it must be
excluded. **Fix:** verification now excludes `_fetch_receipt.json` (and `.part` temp files). After
the fix, a fully-present dir re-verifies clean — `0 to fetch`, all four checks `OK`, idempotent. A
first-run-only test would never have found this; K3 step 3 is precisely why.

### Could the progress read as a hang?

**Partly — and the docs now warn about it.** The *files* counter climbs steadily
(20, 40, 60 … 792), so on that axis it clearly moves. But the *bytes* counter can appear to stall:
several large L1 monthly parquets download in parallel and only complete near the end, so the GB
figure sat around 0.6 GB for much of the run and then jumped to 1.17 GB in the final line. Someone
watching only the GB number could think it stalled. It does not hang — the file counter is the honest
signal. Both `README.md` and `HANDOVER.md` now say a full fetch takes ~1–2 min and that the byte
counter can appear to sit still while large files finish in parallel.

---

## Files changed (code only — nothing under `data/`)

- `polymarket/research/scripts/fetch_data.py` — **new**, the downloader.
- `polymarket/research/scripts/check_setup.py` — missing-data message names `fetch_data.py`.
- `polymarket/research/requirements.txt`, `polymarket/research/pyproject.toml` — `boto3` (fetch-only).
- `polymarket/research/epsilon_data/README.md`, `polymarket/research/HANDOVER.md` — fetch-first docs.
- `README.md` (repo root) — front-door section.

## Out of scope (untouched, per the brief)

Step F (`book`), the backtester feed, `polymarket/execution/`, the capture pipeline,
`universes.yaml`. Nothing under `data/` committed. R2 remained a source — copy/read only.

## Open action (operator's)

Step J is already on the remote (`origin/alvaro` = `4268744`, the Step J commit — the earlier push
block was resolved). The Step K commit `9810c2b` is committed locally and is the only unpushed change
(local `alvaro` is 1 ahead of `origin/alvaro`, 16 ahead of `origin/main`). To publish Step K, the
operator runs `git push origin alvaro` — code only, `data/` gitignored.
