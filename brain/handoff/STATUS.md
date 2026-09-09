# STATUS

**Updated:** 2026-08-31 — end of Step K. Getting the data is now a one-command, no-rclone step, and the repo has a front door.

## Where we are
**Step K done.** Report: `reports/stepK.md`. A newcomer (Justin, Gonzalo) can now clone, run `python scripts/fetch_data.py`, and have the 1.09 GB library verified on disk — with only the same three R2 env vars the loader uses, no rclone. The root README now routes to the library/handover/fetch. Step J remains accepted; `book` (Step F) still intentionally not started.

## The one open action (needs the operator)
**Step J is already on the remote** — `origin/alvaro` is at `4268744` (the Step J commit; the earlier push block was resolved). The **Step K** commit `9810c2b` is committed locally and is the only thing not yet pushed (local `alvaro` is exactly 1 ahead of `origin/alvaro`, and 16 ahead of `origin/main`). **To publish Step K, run `git push origin alvaro`.** Code only — `data/` is gitignored. (Based on the local remote-tracking ref; `git fetch` to confirm.)

## What shipped in Step K
- **`scripts/fetch_data.py`** — pure-Python (boto3) R2 downloader, **no rclone**. Reuses the loader's credential path (`_r2_creds`), byte-copies 792 files in 8 parallel workers, **resumable** (skips same-size files), **read-only** against R2 (list+get only — no put/delete/copy), self-verifies (file count / bytes / tokens rows / L1+trades non-empty), writes `_fetch_receipt.json`, prints the next commands. `--dest`, `--dry-run`, `--workers`.
- **Root `README.md` front door** — a ~8-line section near the top routing to HANDOVER, the loader README, CONTRIBUTING, and the fetch command. All links resolve from repo root.
- **Docs demote rclone** — `epsilon_data/README.md` + `HANDOVER.md` now lead with `fetch_data.py`; rclone is a one-line "if you already have it" alternative with the ⚠️ copy-only warning intact. Shared three-env-var Credentials paragraph. `check_setup.py` names `fetch_data.py` when data is missing.

## K3 — run end-to-end (against a scratch dir, then deleted)
- **Full fetch:** 792 files / 1,167,809,048 bytes / 30,772 tokens / 101,051,502 L1 / 7,227,528 trades — all 4 checks OK, receipt written, **83 s** wall-clock (~13–16 MB/s).
- **Resume:** deleted 3 files, re-ran → fetched exactly 3, skipped 789 (4 s). Idempotent after fix.
- **check_setup + dashboard** against scratch: green, reconciliation match=True, dashboard 0 exceptions.
- **Bug caught by running it (now fixed):** verification counted the script's own `_fetch_receipt.json` as a 793rd file, failing re-run checks (+577 bytes). Verification now excludes local-only artifacts.
- **Progress-as-hang note:** the *files* counter climbs steadily; the *bytes* counter can look stalled while large L1 parquets finish in parallel. Docs now warn (~1–2 min; not a hang).

## Not committed (deliberate)
`data_layer/` + `brain/DATA_LAYER_BRIEF.md` (machine-specific build scripts) and `brain/handoff/` (this channel). Nothing under `data/` is committed. R2 is a source — copy/read only.

## Dashboard
Prior local instance may still be at http://localhost:8501 (Step J). Stop it whenever.
