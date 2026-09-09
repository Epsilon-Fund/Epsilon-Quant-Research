---
title: "Handoff — consolidating everything that lived outside the repo, and the exact steps to converge the three diverging branches"
created: 2026-09-09
status: import done on `justin`; branch convergence NOT done — the § 3 steps are for Justin to run and sequence
owner: justin
project: infra
para: area
hubs:
  - COWORK
  - VAULT_MAP
  - MERGE_PROTOCOL
  - TODO
tags:
  - handoff
  - brain
  - collaboration
  - git
  - archive
---

# Handoff 2026-09-09 — out-of-repo consolidation + branch convergence plan

> Hubs: [[COWORK]] · [[VAULT_MAP]] · [[MERGE_PROTOCOL]] · [[TODO]]
> Companions: [[pre-migration-vault-2026-05]] (what was imported), [[2026-08-25_mm_canon_tiering_decisions]] (the rule it was imported under), [[2026-09-09_mm_gonzalo_meeting_prep]] (why convergence is urgent).

## Plain-English Summary

- **What this was.** A sweep of every place Epsilon material was living *outside* this repo — the standalone Cowork vault, the Claude project's docs, project memory, and published artifacts — and an import of everything that existed only there. The goal: nothing load-bearing survives only in a chat.
- **What came in.** The Gonzalo meeting-prep handoff (it existed nowhere else); the 43 MB of `research_v1` audit evidence that had been left out of the repo on size grounds; the May-2026 pre-migration vault, archived whole under `brain/archive/pre-migration-vault-2026-05/`; and two decision records that lived only in project memory.
- **What did *not* need importing.** Fourteen vault files had already been migrated under repo-native names — verified by matching 8–10 distinctive sentences each, not by filename. The audit summary's verdicts were already fully reflected in [[2026-09-02-research-v1-audit-justin]].
- **What is still open.** The three branches still diverge. § 3 has the exact convergence steps; the merge order is Justin's call.

---

## 1. What was imported, and from where

| Source | Landed as | Tier |
|---|---|---|
| Cowork vault `polymarket-mm-handoff/2026-09-09_gonzalo_meeting_prep.md` — the only copy anywhere | [[2026-09-09_mm_gonzalo_meeting_prep]] | ACTIVE (MM) |
| Cowork vault `research_v1_audit/` binaries — `03_dataset_overview.html` (8 MB), `research_v1_audit.html` (17 MB), `screenshots/` (18 MB, 39 files) | `polymarket/research/docs/audit_2026-09/` alongside the markdown already there | ACTIVE (data layer) |
| The whole May-2026 Cowork vault — journal, inbox, per-project README/TODO/progress, ADRs, two methodology-lessons ledgers | `brain/archive/pre-migration-vault-2026-05/`, structure preserved, indexed by [[pre-migration-vault-2026-05]] | ARCHIVED + per-thread banner |
| Project memory — the MM tiering/de-jargon rule and its acceptance test | [[2026-08-25_mm_canon_tiering_decisions]] | ACTIVE (rule) |
| Project memory — the device-mount git constraint and the `alvaro` branch fact | § 2 below | ACTIVE (ops) |

**Checked and found already present, so not re-imported:** `_shared/glossary.md`, the six `dali/research/**` notes, the three `polymarket-copytrade/research/notes/**` mirrors, and the four `research_v1_audit/*.md`/`.json` files (md5-identical to the committed copies). Full list in [[pre-migration-vault-2026-05]] § Already migrated. Eleven empty `2026-05-*.md` stubs were dropped.

**Not imported, deliberately:** the vault's `_to_delete/_mm_hub_from_main.md` is byte-identical to `main`'s [[strat_market_making]] — a scratch comparison copy, already in the repo.

## 2. Two operating facts that lived only in memory

### 2.1 Cowork on the connected mount can fetch and inspect, but never checkout, merge or pull

The `epsilon-quant-research` folder connected to a Cowork session is mounted read/write for **content** but cannot **unlink**. Git's atomic replace needs unlink, so `git checkout`, `git merge` and `git pull` abort partway — leaving a stale `.git/index.lock` and half-written files. This was hit on 2026-09-01 fast-forwarding `alvaro` `4268744` → `9810c2b`.

Two further consequences on that mount:

- `git status` needs `-c filter.nbstripout.clean=cat -c filter.nbstripout.required=false`, because the repo's nbstripout filter points at a Python path that does not exist in the session VM.
- A Cowork session should therefore only **create new files** in that working tree. Modifying a tracked file there will block Justin's next `git checkout` on his own machine.

**Rule:** from Cowork, `git fetch` plus read-only inspection (`git show <branch>:<path>`, `git log`, `git diff-tree`, `git archive <branch> | tar -x` into scratch) only. Every checkout, merge, pull, commit and push belongs in the Claude Code prompt Justin runs natively on his Mac.

### 2.2 `alvaro` is a live collaborator branch on the remote

`alvaro` exists on `origin`, it is where Alvaro works, and he pushes to it. It is not a stale local branch and must never be treated as disposable, rebased, or force-updated. Convergence merges **into** a line that keeps it — see [[MERGE_PROTOCOL]] § 1.

## 3. Branch convergence — the exact steps

Measured 2026-09-09. Merge base of `main` and `alvaro` is **`b86efe2`** (daily sync, 2026-08-23).

| Branch | Tip | Position | Carries |
|---|---|---|---|
| `main` | `eaaaa55` | — | the MM handoff cleanup (`67529bd`) + a news-agent wip commit |
| `justin` | `fdeafcd` | **2 ahead of `main`, 0 behind** → `justin` ⊇ `main` | news-agent dashboard v3.5 + a daily sync |
| `alvaro` | `9810c2b` | 15 ahead of `main`, **2 behind** | L2 ingestion, `epsilon_data`, the dashboard, `fetch_data.py`, `HANDOVER.md` |
| `justin-research-v1-audit` | `3275be5` | `alvaro` + 2 | the audit reports, the doc/env fixes, and (now) this consolidation's audit binaries |

**This is a genuine two-way merge, not a fast-forward in either direction.** Each lineage holds brain content the other lacks:

- Only on the `justin` line: the whole August news-agent arc — `newsagent_observatory_v33_findings`, `newsagent_simulated_live_findings`, the data-channel v2/v3 findings, `newsagent_agentreach_scoping`, `newsagent_divergence_pnl_exploratory` — plus the de-jargoned canon surface and the tier banners.
- Only on the `alvaro` line: `brain/handoffs/2026-08-25_cowork_data_layer_session.md`, `2026-08-25_l2_pipeline_recovery.md`, `2026-09-02-research-v1-audit-justin.md`, and everything under `polymarket/research/docs/audit_2026-09/`.

**Exactly four paths were touched on both lineages since `b86efe2`** — these are the only real conflict candidates, out of 165 files changed on the `justin` side and 80 on the `jrva` side:

```
brain/COWORK.md
brain/TODO.md
polymarket/research/notes/market_making/strat_market_making.md
polymarket/research/pyproject.toml
```

The first three are Markdown and resolve under [[MERGE_PROTOCOL]] § 3 (union TODO items; prefer the timeless version in law files; `strat_market_making.md` — **keep the de-jargoned `main` version**, it is the canon surface and the alvaro-line copy is the pre-cleanup one). `pyproject.toml` is normal engineering judgment: take the union of dependencies.

**Line endings are already clean** — zero CRLF Markdown files on all four tips, so the § 6 renormalisation dance is not needed this time.

### Recommended sequence

```bash
# 0. from a native terminal on the Mac — never from a Cowork mount
git fetch origin --prune

# 1. bring alvaro up to main (it is 2 behind; this is the cheap half)
git checkout alvaro && git merge main            # or: git merge justin, to skip a step
git push origin alvaro

# 2. bring the audit branch up
git checkout justin-research-v1-audit && git merge alvaro

# 3. the real merge: fold the justin line in. Four conflict candidates, see above.
git merge justin
#    resolve with the SMART-MERGE-AGENT PROMPT in MERGE_PROTOCOL § 3
#    strat_market_making.md → keep main's de-jargoned version

# 4. hygiene before anything is shared
python3 tools/brain_hygiene.py && cat brain/generated/hygiene_report.md

# 5. promote to main, so Gonzalo pulls ONE branch
git checkout main && git merge justin-research-v1-audit && git push origin main

# 6. everyone catches up
git checkout justin && git merge main && git push origin justin
git checkout alvaro && git merge main && git push origin alvaro   # tell Alvaro
```

After step 5, `main` has all three lines and is the single branch Gonzalo clones.

### One follow-up the merge creates

Alvaro's docs (`polymarket/research/README.md`, `HANDOVER.md`, `epsilon_data/README.md`) arrive **without** tier banners. They are active-project material, so they should not get a parked banner — instead [[strat_market_making]] § "Where things live" needs a short **Data layer** paragraph pointing at them. Flagged in [[2026-09-09_mm_gonzalo_meeting_prep]] § 6; do it once the merge lands.

## 4. Left alone on purpose

- **Repo-wide archiving of parked notes.** Moving every `PARKED` / `HISTORICAL EVIDENCE` note into an archive tree would break hundreds of `[[wikilinks]]` that resolve by basename, and the banners already do the job of stopping a fresh agent from building on them. Not attempted; raise it as its own pass if the banners prove insufficient.
- **`polymarket/research/scripts/newsagent_divergence_pnl.py`** is untracked on the audit branch — code, so out of scope here, but it is the script behind [[newsagent_divergence_pnl_exploratory]] and should be committed with it.
- **`Claude outputs/`** at the repo root is untracked Fishpond koi-game material (`koi_reference_tiers.md`, six `*_goal.txt`). Wrong repo — it belongs with the Fishpond project.
