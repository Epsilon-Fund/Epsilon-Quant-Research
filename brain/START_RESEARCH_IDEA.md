---
title: Start a New Research Idea
created: 2026-06-08
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - CODEX
  - COWORK
tags:
  - onboarding
  - research-process
  - brain
---

# Start a New Research Idea

> Use this when a human or fresh agent wants to discuss, frame, or launch a new research idea without getting lost.

Hub links: [[VAULT_MAP]] | [[ONBOARDING]] | [[CODEX]] | [[COWORK]] | [[TODO]] | [[POLYMARKET_BRAIN]] | [[SKILL_MAP]]

## Short Answer

The guide is **self-contained in `brain/`**, but the research brain is **not only `brain/`**.

- `brain/` is the control plane: maps, task list, agent lanes, workflow rules, handoffs, and onboarding.
- The broader Obsidian brain is the repo's linked Markdown universe: `brain/`, `polymarket/research/notes/`, `topics/`, `docs/`, `live_trading/CLAUDE.md`, and relevant README / manifest files.
- A fresh agent should **run Agent Bootstrap first**, then follow [[VAULT_MAP]]. It should not blindly scan the whole repo, and it should not only read the `brain/` folder.
- Collaboration runs on personal git branches ([[MERGE_PROTOCOL]]); shared notes are tracked in git, while per-person overlays in top-level `local_agents/` and scratch in top-level `scratch/` are git-ignored and stay local to each machine.

## Agent Bootstrap (do this before anything else)

Run the Agent Bootstrap — canonical copy in [[VAULT_MAP]] § Agent Bootstrap.

## Fresh-Agent Read Order

For any new research idea, then read in this order:

1. Your shared role convention:
   - Implementation (Codex / Claude Code): `brain/agents/codex/codex_lane.md`
   - Orchestration (Cowork): `brain/agents/cowork/cowork_lane.md`
2. Project hub:
   - Polymarket: `brain/POLYMARKET_BRAIN.md`
   - Crypto: `docs/STRATEGY_REFERENCE.md` and `docs/CRYPTO_DATA_MANIFEST.md` when data matters.
   - Cross-project / infrastructure: `brain/VAULT_MAP.md`, `brain/SKILL_MAP.md`, and the relevant handoff or doc.
3. The closest existing strategy hub or prior findings note before proposing a new branch.
4. Relevant data/artifact manifests before scanning raw folders.

## New-Idea Workflow

1. **Classify the idea.** Decide whether it is Polymarket, crypto, cross-project, or brain/infrastructure.
2. **Check whether it already exists.** Search the current hubs and findings notes for similar terms, market names, strategy names, wallet names, instruments, or data sources.
3. **Draft in the right scratch surface.** Use `scratch/<agent>/YYYY-MM-DD.md` (top-level, local-only) or chat while the idea is still fuzzy. Do not use canonical hubs as scratchpads.
4. **Make an idea card before running work.** A good idea card has: plain-English thesis, why now, target data, expected edge mechanism, cheapest falsifier, success/failure gate, and where the final findings note would live.
5. **Pre-register the test.** State the primary metric, sample, leakage guard, CI or uncertainty check, cost assumptions, and what result would close the idea.
6. **Overfitting check (required before live).** Any strategy whose parameters came from a hyperparameter search must clear the overfitting harness (`infrastructure/validation/overfitting_audit.py` — Deflated Sharpe Ratio at the trial count, PBO via CSCV, White's Reality Check) before it graduates to live capital, and its verdict block must be pasted into the strategy's findings note. Run searches with `collect_trials=True` so the audit has trial-level data. Pre-registered gate: deflated Sharpe > 0 AND PBO < 0.5 AND post-haircut lower-CI Sharpe materially > 0 — otherwise the strategy is flagged for review, not deployed. See `docs/STRATEGY_REFERENCE.md` § H and [[momentum_overfitting_audit_findings]] for the worked first application.
7. **Run or assign the work.** Codex handles code, notebooks, data queries, CSVs, charts, and findings notes. Cowork frames the question, writes the prompt, and interprets the result.
8. **Canonical edits happen on your personal branch.** All durable-Markdown edits ride the operator's personal branch and reach main via [[MERGE_PROTOCOL]] — no special ceremony needed beyond the branch rule in the Agent Bootstrap.
9. **Promote durable output.** Finished results go to the matching project note folder, get hub backlinks, frontmatter, and a `## Summary`.
10. **Update the map.** If the idea becomes an active branch, update `brain/TODO.md` and the relevant hub. If it becomes a durable cross-thread decision, write a `brain/handoffs/<YYYY-MM-DD>_<topic>.md`.

## Where the First Durable Note Goes

| Idea type | First durable location | Hub backlink |
|---|---|---|
| Polymarket new market map / trader screen | `polymarket/research/notes/overview/market_maps/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket literature / external context | `polymarket/research/notes/overview/foundations/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket cross-branch synthesis | `polymarket/research/notes/overview/synthesis/<topic>.md` | [[POLYMARKET_BRAIN]] |
| Polymarket MM finding | `polymarket/research/notes/market_making/<topic>_findings.md` | [[strat_market_making]] and [[COWORK]] |
| Polymarket OD finding | `polymarket/research/notes/options_delta/<topic>_findings.md` | [[strat_options_delta]] and [[COWORK]] |
| Polymarket copytrade finding | `polymarket/research/notes/copytrade/<topic>_findings.md` | [[COWORK]] |
| Polymarket dali / lineage finding | `polymarket/research/notes/dali/<topic>_findings.md` | [[COWORK]] |
| Crypto strategy research | `topics/<strategy>/research/` or `docs/` | `docs/STRATEGY_REFERENCE.md` |
| Live-trading architecture | `live_trading/CLAUDE.md` | `docs/STRATEGY_REFERENCE.md` |
| Cross-thread snapshot / decision | `brain/handoffs/<YYYY-MM-DD>_<topic>.md` | relevant hub |
| Brain infrastructure idea | `brain/OBSIDIAN_INFRA_ROADMAP.md`, `brain/SKILL_MAP.md`, or a handoff | [[VAULT_MAP]] |

## What Not To Do

- Do not tell a fresh agent to "read the brain folder" and stop there. That misses project notes in `polymarket/research/notes/`, `topics/`, and `docs/`.
- Do not tell a fresh agent to scan the entire repo first. Start with [[VAULT_MAP]] and follow the map.
- Do not create a new canonical note until the idea has a clear home, hub backlink, and unique basename.
- Do not put code, scripts, notebooks, generated reports, or raw data in `brain/`.
- Do not save Codex prompts as repo files. Prompts live in chat; outputs live as linked findings notes.
- Do not draft inside canonical hubs. Draft in agent scratch, then promote intentionally.
- Do not commit or push to `main` directly — work on your personal branch and merge via [[MERGE_PROTOCOL]].
- Do not put personal state in shared files: shared lane docs and templates live under `brain/agents/`; top-level `local_agents/<agent>.md` and `scratch/<agent>/` are git-ignored and stay private to each machine.

## Copyable Prompt

```markdown
We are starting a new research idea. Before proposing structure or next steps, read:
1. Run the Agent Bootstrap: determine role, seed/read `local_agents/<role>.md` from `brain/agents/templates/<role>.local.template.md` if missing, then read the shared law and `brain/VAULT_MAP.md` / `brain/TODO.md`.
2. Your shared role convention: `brain/agents/codex/codex_lane.md` or `brain/agents/cowork/cowork_lane.md`
3. The relevant project hub:
   - Polymarket: `brain/POLYMARKET_BRAIN.md`
   - Crypto: `docs/STRATEGY_REFERENCE.md`
4. The closest existing strategy hub, findings note, or data manifest for this idea.

Then produce an idea card with:
- Plain-English thesis
- Existing related notes / branches
- Where the durable note should live
- Cheapest falsifier
- Data needed
- Primary metric and leakage guard
- Overfitting check plan if a parameter search is involved (run with `collect_trials=True`; DSR/PBO/Reality-Check gate via `infrastructure/validation/overfitting_audit.py` before live)
- What result would close / continue / graduate the idea

Do not write to a canonical hub as scratch. If drafting is needed, use your scratch (`scratch/<agent>/`, local-only) or chat first.
All durable edits happen on the operator's personal branch — never main; merges follow `brain/MERGE_PROTOCOL.md`.
```

## When To Call Skills

- **Rock Tumbler:** when messy scratch / notebooks / exploration should become one clean findings note.
- **Chronicler:** when a session produces a decision, closure, handoff, or active-branch update.
- **Cartographer:** when a new durable branch/folder appears or a map row becomes stale.
- **Janitor:** when hygiene reports show broken links, duplicate basenames, missing summaries, or missing frontmatter.
- **Librarian:** when a reusable term, metric, column, bucket, or acronym needs a durable definition.
