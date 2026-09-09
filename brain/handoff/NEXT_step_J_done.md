# NEXT — Step J: handover. Make the next person's life easy, then publish.

Written by Cowork, 2026-08-29. Supersedes the previous NEXT.md. **Step I is accepted** once its panels land. This is the closing step: after it, the job is done.

Three parts, in order: **make the repo extensible → build the audit button → publish and document.**

---

## J1 — Make it easy to add things

Right now `dashboard/app.py` is one 300-line file. Adding a panel means editing it. That does not scale to a team, and the next person's first act will be to add a panel.

### The panel registry

Split panels into modules with a registry, so **adding a visualisation is adding a file**, not editing the app:

```
dashboard/
  app.py              shell: sidebar, search, header, routing. Thin.
  panels/
    __init__.py       the registry
    _base.py          the contract every panel implements
    market.py         one panel per file
    volume.py
    flow.py
    markout.py
    negrisk.py
    coverage.py
    ...
```

**The contract:** each panel declares a name, which section it belongs to (Explore / Audit), what it needs (a market? an event? nothing?), and a `render(ctx)` that draws it. The registry discovers them. `app.py` never names a panel explicitly.

**The test:** adding a new panel must be *one new file and zero edits elsewhere*. If it isn't, the contract is wrong.

### Install from a clean clone

Right now streamlit was pip-installed by hand into a uv-managed venv. That does not survive a fresh clone.

- Declare **every** dependency in `pyproject.toml` — streamlit, plotly, pandas, duckdb, pyarrow, the notebook kernel.
- `README.md` opens with a **quickstart that works from `git clone`**: install, set `EPSILON_DATA_ROOT`, run. Three commands, no folklore.
- Verify it by following your own instructions in a clean checkout. If a step is missing you will find it there and nowhere else.

### `CONTRIBUTING.md`

Short and concrete, aimed at someone who has never seen this:

- How to add a dashboard panel — with a **complete worked example**, a real panel written start to finish.
- How to add a loader function, and when you should (a panel needs data the API doesn't expose) versus shouldn't (you can compose it from existing calls).
- The two rules: **everything through `epsilon_data`**; **nothing is ever excluded automatically.**
- How to run the tests, and that the anti-drift test must keep passing.

### A cookbook

Beyond the five worked examples, a `notebooks/cookbook.ipynb` of short recipes for the questions people will actually ask: filter markets by liquidity; compare two markets; pull every market in an event; build a returns series; find the busiest hour; slice a token's tape by time window. Ten or so, a few lines each. This is what stops the next person reading the loader source to guess.

---

## J2 — The audit button

**Human-initiated, human-decided.** The operator's words: *"my colleague is researching a market, there is one price series that looks off, then he can run a full audit button on that slug and gets a concise report and a recommendation between sending that slug to the trash or keeping it."*

**One implementation, three surfaces:**

`epsilon_data.audit_market(ref)` does the work and returns a structured result. The dashboard renders it behind a button on the market header. A Claude Code command wraps the same function. No duplicated logic.

### What the audit checks

Everything cheap that could indicate the series is wrong:

- **Identity** — all five `check*` columns, `identity_status`, and whether the complement is present and paired.
- **Internal consistency** — do the two sides sum to ~1 across the tape? Where and how badly do they not?
- **Continuity** — gaps in the tape against the coverage calendar. Is a hole a known outage, a quiet hour, or unexplained?
- **Sanity of values** — mid outside (0,1), crossed book (bid > ask), spread of 0 or 100¢, prices that never move, prices that move impossibly fast.
- **Trades against quotes** — trades printing outside the touch at the time, or trades with no quote data at all.
- **Volume shape** — a single print that is a large share of the token's volume; duplicate transaction hashes.
- **Resolution** — for settled markets, did the winner converge? Is this one of the 14 inverted?

### What it returns

- A **verdict**: `looks fine` · `worth a look` · `recommend excluding` — with the reason in one sentence.
- The **evidence** behind it: which checks failed, with numbers. Never a bare verdict.
- Its **recommended scope**, and the trade-off stated plainly: excluding a single token leaves its complement orphaned, which will break pair views and NegRisk sums; excluding the whole market keeps the dataset internally consistent. **Say which you recommend and why — the human chooses.**
- The **exact `exclusions.csv` line** to add, ready to use.

### Writing to `exclusions.csv`

- Only ever on an explicit human action — a button click, a confirmed command. **Never as a side effect of running the audit.**
- **Append-only**, with columns: what was excluded, scope (token/market), reason, date, who.
- The dashboard keeps showing excluded items, marked as excluded. Nothing disappears.
- Reversible: removing the line restores it. Nothing is ever deleted from disk.

---

## J3 — Documentation, including the thing that is currently missing

### The backtest gap — write this down

**The research library does not feed the existing backtester, and nothing says so.** The next phase is backtesting a market-making algo; someone will assume it does and lose a day.

`mm_engine/feeds/replay_parquet.py` expects `parquet/{date}/{universe}/{table}_{shard}.parquet` with the four raw tables (`book`, `trades`, `price_change`, `bba`), plus `market` and `received_at` columns and a `capture_gaps.parquet` sidecar.

The library is `l1/universe=/month=/` with `asset_id, timestamp_ms, received_ns, best_bid, best_ask, mid, spread_c`. **No `market`, no `received_at`, no `book`, and `l1` is deduped to touch-moving rows** — it is neither `price_change` nor `bba`.

This is a legitimate difference, not a defect: the library is for *looking at data*; the backtester replays *raw events* and needs depth for queue position. But it must be stated in the README as its own section — what the library is for, what it is not for, and that **Step F (`book`) is the bridge** to backtesting.

### The two-book finding

Document what we established, because it is counter-intuitive and will otherwise be rediscovered:

- We have **both** tokens' quotes and trades. Nothing is missing.
- NO quotes are the **exact complement** of YES quotes — `NO_bid = 1 − YES_ask`, at the 1st and 99th percentile across 1.29 M observations. They carry no independent information.
- This is Polymarket's design, **not our capture**: the daemon subscribes to both tokens and stores `best_bid`/`best_ask` verbatim; nothing in the pipeline computes a complement. The matching engine can mint a complete set from a YES buyer and a NO buyer, so a bid on NO *is* an ask on YES — two order ledgers over one pool of executable liquidity.
- **Consequence:** plotting both mids is redundant by construction. **Trades are not** — those are two genuine streams of intent, and someone bearish must buy NO because they cannot short YES.
- **Limit to state honestly:** because we have no independent NO quotes, this dataset *cannot answer* whether cross-book arbitrage ever existed. That is a limit, not a finding.

### Also in the README

- The known gaps: the 2026-06-22 15:00 → 06-23 08:29 outage, the esports quiet hours, and the missing-hour sweep results.
- That `median_mid − median_spread/2` goes negative for 71 tokens — you cannot rebuild a book from two independently-taken medians.

---

## J4 — Publish

**Code to GitHub. All data to R2.** Settled by the numbers: the library is **1.09 GB across 790 files**, and two files (esports trades July at **129.1 MB** and August at **102.6 MB**) exceed GitHub's 100 MB hard limit. Git LFS's free tier is 1 GB storage and 1 GB bandwidth per month — over on both. And git history is permanent: a gigabyte in the repo is paid for by every clone, forever.

- Upload `research_v1/` to R2 with `rclone copy`. **Copy only — never `sync`, `delete`, `purge` or `move` with `r2:` as the target.**
- Write **`_manifest.json`** alongside it: what each table is, row counts, the 64-date span, coverage, the declared outage and the individual missing hours, the build provenance (which commit built it), units for every column, and the daemon-counter correction so nobody trusts the Step A figures.
- **Verify after upload** — file count and byte sizes against local. Report both.
- Confirm the loader reads from R2 with only `EPSILON_DATA_ROOT` changed. If it needs a code change, the config layer is wrong — fix it.
- Push the code branch. Never commit anything under `data/`.

---

## J5 — The handover document

One document for Gonzalo, in the repo. Not a log — an orientation.

What this dataset is and how it was built, in a page. How to get it and run the dashboard. The tree, the tables, the units, the traps. What is known to be wrong or missing. What the open questions are — the NegRisk sums, the stale-book cohort, the esports quiet hours. What Step F would add and why it matters for backtesting. And where the detailed record lives, so the reports are findable but not required reading.

Assume he is capable and knows nothing about this project. Do not make him read six step reports to start work.

---

---

## J6 — Onboarding must actually work. This is a first-class requirement, not documentation polish.

Two other people — **Justin** and **Gonzalo** — have to go from nothing to a running dashboard without help. The operator: *"if they can't load the dashboard and they can't see it and there's errors, then that's going to slow a lot the process."*

Treat a failed first-run as a build failure.

### Two ways to get the data. Support and document both.

1. **Read straight from R2** — set `EPSILON_DATA_ROOT` to the bucket. DuckDB httpfs with predicate pushdown means single-token reads only fetch the bytes they need. Nothing to download, slower on first touch. Best for a quick look.
2. **Sync locally** — one command pulls the 1.09 GB to disk. Fast browsing. Best for real work.

Both must work with **no code change** — only the env var. If either needs an edit, the config layer is wrong.

### Credentials — decided: reuse the existing key for now

The operator has decided the existing read/write R2 key is fine for onboarding Justin and Gonzalo. He will have the colleague who controls the bucket swap it for a read-only key later. **Do not treat this as a blocker.**

But because that key **can delete the archive**, the safety rule moves out of our private handoff protocol and into the documentation everyone reads. In the README and the handover doc, prominently:

> **The R2 key can write and delete. Only ever `rclone copy`. Never `sync`, `delete`, `purge` or `move` with `r2:` as the target.** The 71 GB raw archive is not backed up anywhere else — a mistyped `sync` destroys it permanently.

Give the exact safe command for fetching data, so nobody improvises one. And note in the handover doc, as an open item for the operator, that swapping to a read-only token scoped to the research prefix is the eventual fix.

Credentials never go in the repo, never in a committed config, never in a command string. Document `.env` or the rclone config file as the place, and gitignore it.

### One setup path, and you must actually run it

- A single documented sequence: clone → install → configure credentials → fetch or point at data → run.
- **Test it from a genuinely clean checkout in a fresh directory.** Following your own instructions is the only way you find the missing step, and there is always a missing step.
- A `scripts/check_setup.py` that verifies the environment and says *specifically* what is wrong: no credentials, wrong data root, missing package, unreachable bucket. Not a stack trace — a sentence telling them what to fix.
- A **Troubleshooting** section covering what will actually go wrong: credentials not set, `EPSILON_DATA_ROOT` unset or pointing at nothing, wrong Python version, streamlit not on PATH, the uv/pip venv situation.

---

## J7 — Frame the backtest work as Gonzalo's first task, not as a caveat

Reframing an earlier instruction. The library not feeding the backtester is **not a defect to warn about** — it is **the first piece of work the next person will do**, and it should be handed over as a specification rather than a disclaimer.

The operator: *"part of what Gonzalo will do is add the backtesting tools to that library so it is using this dataset instead of the raw data, which would be more confusing."*

So the README and handover doc must set that work up properly:

**Describe how the backtester feeds today.** `mm_engine/feeds/replay_parquet.py` reads the raw capture layout — `parquet/{date}/{universe}/{table}_{shard}.parquet`, the four tables `book`/`trades`/`price_change`/`bba`, with `market` and `received_at` columns and a `capture_gaps.parquet` sidecar. Explain it clearly enough that someone can follow it without reading the source.

**State exactly what a library-fed replay would need**, so the job is scoped rather than discovered:

- `market` and `received_at` — not currently in `l1`/`trades`, and say whether they are recoverable from `tokens`/the archive or need a rebuild
- `book` — does not exist yet. **This is Step F and it is the real dependency**: queue position cannot be modelled from L1 alone.
- `l1` is deduped to touch-moving rows, so it is neither `price_change` nor `bba` — say which engine events it *can* legitimately produce and which it cannot
- the gap sidecar — where that information lives now (the coverage data and the known-outage record)

**Say why it is worth doing:** replaying from the library means one clean, verified, documented dataset instead of 71 GB of raw JSON-derived parquet with no identity attached. That is the whole reason the library exists.

Do not build the adapter. **Scope it** so the next person starts with a plan instead of an archaeology project.

## Then STOP

`reports/stepJ.md`, `STATUS.md`, `LOG.md`. Report what the panel contract ended up being, that a clean-clone install actually worked, the R2 verification numbers, and anything the audit tool found while you were testing it on real markets.

Announce anything over five minutes.
