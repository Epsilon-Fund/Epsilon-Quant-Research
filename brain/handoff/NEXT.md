# NEXT — Step K: make getting the data a solved problem, and give the repo a front door.

Written by Cowork, 2026-08-30. Supersedes the Step J NEXT (kept as `NEXT_step_J_done.md`).
**Step J is accepted.** Two gaps remain, both found by re-reading the repo as a person who has
never seen it. Both are onboarding blockers, not analysis work. Small step. Do it fully.

Context you need: Justin (partner, auditing) and Gonzalo (joining, building the backtester) will
both clone this repo cold and need the 1.09 GB library on their disk. If that first hour has
errors in it, everything downstream slows down.

---

## K1 — One command that fetches the library. No rclone.

### The problem

`epsilon_data/README.md` and `HANDOVER.md` both tell the newcomer to run:

```
rclone copy r2:epsilon-polymarket-data/research/v1  <yourdir>/research_v1  -P
```

That command works **only** for someone who has already installed rclone and already has an
`[r2]` remote in their `rclone.conf`. Justin has neither. He will get
`didn't find section in config file`, and nothing in the repo tells him how to fix it. Meanwhile
the loader itself needs no rclone at all — it reads R2 through DuckDB httpfs using
`EPSILON_R2_KEY_ID` / `EPSILON_R2_SECRET` / `EPSILON_R2_ENDPOINT`.

So we are asking newcomers to set up a second, different credential path for the one step that
should be the easiest. Remove rclone from the newcomer's critical path entirely.

### What to build

`polymarket/research/scripts/fetch_data.py` — pure Python, S3-compatible, no rclone.

```
python scripts/fetch_data.py                       # → ./data/research_v1  (the loader default)
python scripts/fetch_data.py --dest D:/epsilon/research_v1
python scripts/fetch_data.py --dry-run             # list what it would fetch, transfer nothing
```

Requirements:

- **Credentials: exactly the same three env vars the loader already uses.** Read them the same
  way `epsilon_data` does — reuse that code path, do not write a second credential loader. Keep
  the existing `rclone.conf [r2]` fallback so anyone already set up keeps working. Never print a
  secret, never put one in a command string, never write one to a file.
- **`boto3` added to `requirements.txt` and `pyproject.toml`.** Add it under a comment saying
  it is only for `fetch_data.py`, not for the loader — the loader must stay boto3-free.
- **Byte copy, not a re-encode.** `GetObject` → write bytes. Do not round-trip parquet through
  DuckDB: that would change encoding and row order and break the anti-drift guarantee.
- **Resumable.** Skip a file that already exists locally with the same size. Say how many it
  skipped. A killed download must be fixable by re-running the same command.
- **Parallel, modestly.** ~8 workers. 792 files, most of them small.
- **Progress.** Files done / total, bytes done / total, a running rate. This takes minutes; a
  silent terminal reads as a hang.
- **Read-only against R2.** The script must contain no `put_object`, `delete_object`,
  `copy_object` or any other mutating S3 call. This is not a style note — the key can delete and
  the 71 GB raw archive has no second copy.

### Verify, then say so

When the transfer finishes the script must prove the copy is complete, and print a verdict:

1. file count on disk == object count under the prefix in R2 (expect **792**);
2. total bytes on disk == total bytes in the listing (expect **~1.09 GB**);
3. `tokens.parquet` opens and has **30,772** rows;
4. one `l1` partition and one `trades` partition open and are non-empty.

Write `_fetch_receipt.json` next to `_manifest.json`: timestamp, source prefix, file count,
total bytes, and the four check results. Then print the literal next command, with the dest
path filled in:

```
set EPSILON_DATA_ROOT=<dest>
python scripts/check_setup.py
streamlit run dashboard/app.py
```

If any check fails, exit non-zero and name the failing check in one sentence. A partial download
that reports success is the worst outcome here — worse than a loud failure.

### Then fix the docs that point at rclone

In `epsilon_data/README.md` ("Getting the data") and `HANDOVER.md`, make **`python
scripts/fetch_data.py`** the documented way to get a local copy. Demote the rclone command to a
one-line "if you already have rclone configured" alternative, and keep the ⚠️ copy-only warning
attached to it — that warning stays exactly as strong as it is now. Also add a short **Credentials**
paragraph that says plainly: the three env vars are all you need for both reading straight from R2
and fetching a local copy; ask the operator for them; put them in a gitignored `.env`; they are
currently read/write, so treat them accordingly.

Have `scripts/check_setup.py` name `fetch_data.py` in the message it prints when `tokens.parquet`
is missing under the configured root.

---

## K2 — The repo root README has no way in.

`README.md` at the repo root is dated June 22 and contains **zero** occurrences of
`epsilon_data`, `research_v1`, `dashboard` or `streamlit`. Everything Step E through Step J built
is invisible to someone who clones this repo and reads the front page. That is the single most
likely place for Justin's first hour to go wrong.

Add a section — near the top, above the branch-by-branch tour, because this is the live work —
along these lines:

> ### Polymarket research library + dashboard (v1, Aug 2026)
> A cleaned, navigable L1 + trades dataset over 64 days of captured Polymarket order-book data
> (30,772 tokens · 101 M L1 rows · 7.2 M trades · 1.09 GB), a documented Python loader, and a
> Streamlit research terminal for auditing it and finding strategy ideas.
>
> - **Start here:** [`polymarket/research/HANDOVER.md`](polymarket/research/HANDOVER.md)
> - **Loader API + data dictionary:** [`polymarket/research/epsilon_data/README.md`](...)
> - **Adding a panel or a tool:** [`polymarket/research/CONTRIBUTING.md`](...)
> - **Get the data:** `cd polymarket/research && python scripts/fetch_data.py`

Keep it to roughly that size. The root README's job is to route, not to explain. Check every link
resolves from the repo root.

---

## K3 — Test it the only way that counts

Not "the code looks right." Actually run it:

1. `--dry-run` from a clean shell with only the three env vars set. Confirm it lists 792 objects
   and transfers nothing.
2. A real fetch into a **scratch destination** (not the existing `data/research_v1` — do not
   touch the built library). Let it finish. Confirm all four verification checks pass and the
   receipt is written.
3. Delete a handful of files from the scratch copy, re-run, and confirm it fetches exactly those
   and skips the rest.
4. Point `EPSILON_DATA_ROOT` at the scratch copy, run `check_setup.py`, and load the dashboard
   against it.
5. Delete the scratch copy when done. `device_bash` cannot delete — if you cannot remove it,
   leave it and say where it is.

Report: the exact commands you ran, the verification output, wall-clock time for the full fetch,
and anything that surprised you. If the fetch is slow enough that a newcomer would think it hung,
say so and say what the progress output looks like.

---

## Out of scope for Step K

Do not start Step F (the `book` tier), do not touch the backtester feed, do not change anything
under `polymarket/execution/`, do not restart the capture pipeline, do not edit `universes.yaml`,
do not commit anything under `data/`.

Standing rules unchanged: R2 is a source, never a deletion target. No `sync`/`delete`/`purge`/
`move` with `r2:` as target, in a script or at a prompt. Never delete a local file unless it is
byte-verified in R2. Never print or copy `rclone.conf`. Secrets in env vars, never in a command
string.
