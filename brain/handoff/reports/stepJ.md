# STEP J — handover: extensible repo, audit button, publish, onboarding   [DONE]

Written by Claude Code, 2026-08-30. The closing step. Ran the three corrections → J1 (extensibility) → J2 (audit) → J3 (docs) → J4 (publish) → J5 (handover) → J6 (onboarding) → J7 (backtest spec). Both acceptance tests pass. Code committed to `alvaro`; data published to R2; nothing under `data/` committed.

## Three corrections
1. **NegRisk panel now exists** (was claimed in Step H, absent) — built in I4/J1.
2. **Stale-book figure corrected** in `stepH.md` and the cohort view: esports `near_half` = **21,082** (politics 440; 21,522 was both universes); 20,764 within ±1h of settlement.
3. **NegRisk summed at a common timestamp**, never a sum of medians (I4 + `negrisk_sum`).

## J1 — extensibility
- **Panel registry.** `dashboard/app.py` is now a thin shell; panels live in `dashboard/panels/` with a `_base` contract (`@panel(name, section, needs, order)` + `Ctx`) and `__init__` auto-discovery. **The panel contract:** a panel is a file that declares its name/section/needs and a `render(ctx)`; the shell names none of them. **Acceptance test 1 — PASS:** dropping one file (`zzprobe.py`) made a new panel appear in the registry with zero edits elsewhere; removing it made it vanish. 3 explore + 6 audit panels.
- **Config layer made root-agnostic.** `EPSILON_DATA_ROOT` reads a LOCAL dir **or** an `s3://` bucket — every read goes through DuckDB; R2 creds come from env (`EPSILON_R2_*`) or `rclone.conf`. (Fixed `data_root()` returning `Path`, which mangled `s3://`→`s3:\` on Windows.)
- **Clean-clone install.** Added streamlit/plotly/matplotlib to `pyproject.toml` and a pinned `requirements.txt`; `CONTRIBUTING.md` with a complete worked panel example + when to add a loader function; `notebooks/cookbook.ipynb` (11 recipes, executed, 0 errors).

## J2 — the audit button
`epsilon_data.audit_market(ref)` — one implementation, three surfaces (dashboard header button, direct call, CLI-callable). Checks identity (5 checks + complement), pair-sum≈1, continuity vs the outage, value sanity (crossed/out-of-range/frozen/jumps), trades-vs-quotes, volume shape (dominant print, dup hashes), resolution/inversion. Returns a **verdict** (`looks fine` / `worth a look` / `recommend excluding`) + **evidence with numbers** + a **recommended scope with the trade-off stated** (excluding a token orphans its complement and breaks pair/NegRisk views; excluding the market keeps consistency — recommends **market** scope) + the **ready exclusions.csv line(s)**. It **never writes**; `write_exclusion()` is the only writer — explicit, append-only, reversible. Tests: `test_audit_never_writes`, `test_write_exclusion_roundtrip`.

**What testing the audit found (and I fixed):** on real markets the first cut condemned the *busiest* politics market over 3 crossed ticks (0.01%) and rated 37/40 random markets "worth a look" on normal quiet-market gaps. That tool would stop people looking. Recalibrated to fractions/materiality: crossed/out-of-range are `bad` only above 0.5% (a few → `note`); spread=0 (locked) and =100¢ (one-sided) are common and now informational; continuity flags only **unexplained gaps >12h**. After the fix: busiest politics → *worth a look* (0.01% crossed, a note), an inverted market → *recommend excluding*, busiest esports → *looks fine*; 40-market sweep 8 *looks fine* / 32 *worth a look* (conservative by design — the evidence is what the human reads).

## J3/J5/J7 — documentation
- `epsilon_data/README.md` gained: a git-clone quickstart; the two data paths + credentials + the **R2 delete-safety warning**; a **Troubleshooting** table; **"What this library is for and NOT for"** (it does not feed the backtester); the **scoped backtest-adapter spec** (J7 — `market` recoverable from `tokens`, `received_at` needs a rebuild, **`book` is Step F and the real dependency**, `l1` can produce `best_bid_ask` events but not depth); the **two-book finding** (NO_bid = 1 − YES_ask exactly; both mids redundant, trades are not; cannot answer cross-book arbitrage); known gaps; the negative-`median_mid − median_spread/2` caveat.
- `HANDOVER.md` — one orientation page for Gonzalo: what it is, get it running, tree/tables/units/traps, what's wrong/missing, the open questions, and his first task (the backtest adapter), with the reports as findable-not-required.

## J4 — publish (verified)
`rclone copy research_v1 → r2:epsilon-polymarket-data/research/v1/` (copy only). **Verification: local 792 files / 1,167,809,048 bytes == R2 792 objects / 1,167,809,048 bytes — exact.** `_manifest.json` published alongside (version, build commit, source range, per-table row counts, units, coverage/outage, the daemon-counter correction, reconciliation). **Acceptance test 2 — PASS:** with only `EPSILON_DATA_ROOT=s3://…` changed (no code edit), `catalog()` (30,772), `load_l1()` (pulled a token from R2), and `coverage()` all read from R2. GitHub was ruled out by the numbers (1.09 GB; two files >100 MB hard limit).

## J6 — onboarding actually works
`scripts/check_setup.py` verifies Python, packages, `EPSILON_DATA_ROOT`, R2 creds, and a real loader read — reporting the problem in a sentence. **Clean-clone test — PASS (from a genuinely fresh dir):** `git archive HEAD` → fresh dir → new venv (**Python 3.13.5**, not the 3.14 dev venv) → `pip install -r requirements.txt` → creds via env → `check_setup.py` all green (read 30,772 tokens from R2, reconciliation match) → dashboard landing + explore render, 0 exceptions. So clone→install→configure→run works unaided, on a second Python version.

## Two acceptance tests — both PASS
1. **New panel = one file, zero edits** — verified with a throwaway panel.
2. **Both data paths differ by only `EPSILON_DATA_ROOT`** — verified (local tests + R2-direct read, no code change).

## What the audit tool found on real data (for the operator)
- The **busiest politics market has 3 crossed (bid>ask) ticks** — the intra-ms ordering artefact, 0.01% of its rows; a note, not a defect.
- Esports markets legitimately have many >1h tape gaps (quiet periods) — not holes; the audit no longer flags them.
- The **14 `inverted` markets** are the only ones the audit recommends excluding on resolution grounds — and they are upsets (correct mapping), so the operator would likely *keep* them; the tool states the evidence and lets the human decide.

## Then STOP
`reports/stepJ.md`, `STATUS.md`, `LOG.md` written; `stepH.md` corrected. Code committed to `alvaro` (`4268744`); **the `git push` was blocked by the Claude Code auto-mode classifier** (outward action to the shared GitHub remote) — it needs the operator to run it or grant a Bash push permission. Data on R2, verified. `book` (F) not started.

**Operator: to publish the code (J4's push), run:**
```
git push origin alvaro
```
(local `alvaro` is 75 commits ahead of `origin/alvaro`; data under `data/` is gitignored and won't be pushed.)

The library, its manual, the dashboard, the audit tool, and the onboarding are done and verified; Gonzalo's first task (the backtest adapter) is scoped.
