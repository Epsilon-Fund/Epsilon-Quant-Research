---
title: "Cowork Migration & Onboarding — Team Account"
tags: [cowork, migration, onboarding, infra]
created: 2026-06-08
owner: justin
purpose: >
  Hand the new (team) Cowork account everything the personal Cowork account
  currently does, so the desktop environment can be rebuilt on first login.
  Most of this does NOT auto-migrate when you switch the logged-in account.
---

# Cowork Migration & Onboarding

You are the **new team-account Cowork**, opening this repo for the first time after
Justin switched the desktop app's logged-in account. This doc is the rebuild manifest:
it inventories what the **personal Cowork** account did and tells you what to
re-establish. Almost nothing below transfers automatically with an account switch —
treat every item as "recreate / reconnect / re-grant."

> Day-to-day research operating model is **not** in this file. For how work actually
> flows (Cowork↔Codex split, where notes go, prompt discipline), read [[COWORK]],
> [[ONBOARDING]], [[VAULT_MAP]], [[CODEX]], [[TODO]], [[OPERATING_RHYTHMS]], [[SKILL_MAP]].
> This file is only the **environment/config** layer.

---

## 0. First-login checklist (do these in order)

1. **Re-grant folder access** to the two locations in §1 (folder picker).
2. **Reconnect connectors** in §2 (Gmail, Google Drive) — OAuth, account-bound.
3. **Reinstall plugins** in §3 so the skills reappear.
4. **Recreate the 3 active scheduled tasks** in §4 (copy each prompt verbatim; same cron, all enabled). The 3 that were disabled are intentionally dropped.
5. **Recreate the artifact** in §5 if you still want it (the task that rebuilt it has been dropped).
6. Confirm the vault path in §6 is reachable before enabling tasks.

---

## 1. Folder access (re-grant)

Files live on disk and are untouched by the account switch; only the *grant* is lost.

| What | Path | Used by |
|---|---|---|
| **Research vault** (the "brain") — connected/mounted folder | `/Users/justiniturregui/Desktop/github/epsilon-quant-research` | all chat work + the `brain-*` scheduled tasks |
| **Cowork projects tree** (journal, inbox, per-subproject TODOs) | `/Users/justiniturregui/Documents/Claude/Projects/Epsilon/` | manual journal/inbox work only — the 3 tasks that used it were dropped (see §4) |

⚠ **Two namespaces.** The vault (an Obsidian + Git repo) and the `Documents/Claude/Projects/Epsilon/`
tree are *different roots*. The active `brain-*` tasks only need the **vault**. Grant the Projects
tree only if you still work the journal/inbox manually.

Scheduled-task definitions themselves live at `/Users/justiniturregui/Documents/Claude/Scheduled/<task>/SKILL.md`.

---

## 2. Connectors (reconnect — OAuth, account-bound)

| Connector | Purpose |
|---|---|
| **Gmail** | draft replies, summarize threads, search inbox |
| **Google Drive** | search, read, upload files |

No others were connected. Reconnect both on the team account.

---

## 3. Plugins & skills (reinstall)

**Installed plugin marketplaces** (reinstall these; their skills come with them):

- **daloopa** — equity/financial-modeling skills (build-model, dcf, comps, earnings, tearsheet, etc.). Requires Daloopa MCP auth.
- **data** — SQL / dashboards / viz / warehouse skills (analyze, build-dashboard, create-viz, sql-queries, explore-data, etc.). Connectors: Amplitude, Atlassian, BigQuery, Definite, Hex.
- **finance** — close/accounting skills (financial-statements, journal-entry, reconciliation, sox-testing, variance-analysis). Connectors: MS365, Slack.
- **anthropic-skills** (core) — docx, pptx, pdf, xlsx, schedule, skill-creator, setup-cowork.

**Custom / non-marketplace skills to re-add:**

- `cost-mode` — from GitHub `Sagargupta16/claude-cost-optimizer` (`skills/cost-mode/SKILL.md`). Recorded in repo `skills-lock.json`.
- **Repo-local skills** under `.agents/skills/` in the vault (e.g. `brain-janitor`, and the workflow skills referenced by the brain tasks: brain-chronicler, brain-rock-tumbler, brain-cartographer). These ship *with the repo*, so they survive the account switch as long as folder access is granted — but confirm they still trigger.

---

## 4. Scheduled tasks (3 active — recreate from the local SKILL.md files)

Only the three `brain-*` tasks were enabled and are worth keeping. Recreate each with the
**exact prompt body** from its `SKILL.md` (those files live on this machine and are unchanged),
the cron shown, all **enabled**.

| Task | Cron | Operates on | Prompt source (on this machine) |
|---|---|---|---|
| `brain-hygiene-weekly` | `30 10 * * 1` (Mon ~10:30) | vault | `~/Documents/Claude/Scheduled/brain-hygiene-weekly/SKILL.md` |
| `brain-eod-brief` | `30 10 * * *` (daily ~10:30) | vault | `~/Documents/Claude/Scheduled/brain-eod-brief/SKILL.md` |
| `brain-commit-push` | `0 22 * * *` (daily ~22:00) | vault (git) | `~/Documents/Claude/Scheduled/brain-commit-push/SKILL.md` |

> Display times in the app are a few minutes later than the cron because each task adds a small
> random **jitter** (e.g. eod-brief showed 10:40, commit-push 22:04). The cron above is canonical;
> jitter is regenerated per task and isn't meant to match bit-for-bit.

**What each does (one line each):**

- **brain-hygiene-weekly** — runs `tools/brain_hygiene.py` + `tools/brain_graph_audit.py`, reports counts, and (if dirty per `SKILL_MAP.md` thresholds) emits a ready-to-paste `brain-janitor` prompt. Scan-only; never edits notes.
- **brain-eod-brief** — refreshes the hygiene/graph reports, gathers today's commits + recently-changed notes + TODO threads, writes `brain/generated/daily_brief.md`, and recommends (not runs) brain-* workflow passes. Read-only on canonical notes.
- **brain-commit-push** — safe daily commit + push of Markdown/brain only (`brain/`, `polymarket/research/notes/`, `docs/`, `README.md`, `tools/`) to the **operator's personal branch** (refuses to run on main; never force; stop on conflict per `brain/MERGE_PROTOCOL.md`). Never stages data/notebooks/secrets/code.

**Dropped (were disabled on the personal account — not recreated):** `daily-journal-seed`,
`weekly-todo-digest-refresh`, `nightly-inbox-ingest`. Their `SKILL.md` files still exist locally
if you ever want to revive them.

---

## 5. Artifacts (recreate)

| Artifact id | Name | Path | Rebuilt by |
|---|---|---|---|
| `epsilon-monday-digest` | Epsilon Monday Digest | `/Users/justiniturregui/Documents/Claude/Artifacts/epsilon-monday-digest/index.html` | `weekly-todo-digest-refresh` |

A cross-project Monday exec view over the three subprojects' `## now` items. Light mode,
inline CSS, no external scripts. The HTML already exists at the path above on this machine.
**Note:** the task that used to rebuild it (`weekly-todo-digest-refresh`) was dropped, so it
won't auto-refresh — re-add that task if you want the weekly rebuild back.

---

## 6. Sanity checks before going live

- Both roots in §1 are reachable (open one file from each).
- `python3 tools/brain_hygiene.py` runs clean from the vault root (the brain tasks depend on it).
- Git remote is configured and `brain-commit-push` can pull/push the vault branch.
- Connectors in §2 show connected.
- The three `brain-*` tasks exist and are ON; no other scheduled tasks present.

---

## 7. How the personal Cowork actually operates (pointers, not duplication)

Once the environment is rebuilt, the *working* contract lives in the vault:

- [[COWORK]] — Cowork's orientation: active threads, the Cowork↔Codex split, prompt discipline, where to write things.
- [[ONBOARDING]] — workspace structure, the brain layers, the git branch-per-person collaboration model.
- [[MERGE_PROTOCOL]] — branch model + merge-to-main procedure + smart-merge-agent prompt.
- [[VAULT_MAP]] — the start-here map and the "where to write things" table.
- [[CODEX]] — implementation-agent rules + Markdown quality standard + realism calibration.
- [[TODO]] — authoritative live task list. Read before suggesting next actions.
- [[OPERATING_RHYTHMS]] — cadences (incl. the weekly hygiene scan).
- [[SKILL_MAP]] — the brain-* workflow skills and their event triggers.

**One-line model:** Cowork frames questions, drafts pre-registered Codex prompts, interprets
outputs, and maintains the brain. Codex implements and runs analyses. Cowork does not do
code-like research itself.
