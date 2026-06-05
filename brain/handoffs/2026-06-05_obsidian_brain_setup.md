---
title: "Obsidian brain setup for a returning teammate"
tags: [handoff, onboarding, obsidian, brain, codex, cowork]
created: 2026-06-05
audience: "a teammate opening the repo as an Obsidian vault for the first time"
status: shared setup guide; read with [[2026-06-05_polymarket_research_onboarding]]
---

# Obsidian brain setup

This repo is also an Obsidian vault. Open the repo root as the vault, not only the `brain/` folder. That lets Obsidian connect the hub notes in `brain/` to research notes under `polymarket/research/notes/`, docs under `docs/`, and project files referenced from the handoffs.

## First setup

1. Pull the latest repo.
2. Open Obsidian.
3. Choose "Open folder as vault" and select the repo root.
4. If Obsidian asks about community plugins, review and enable the checked-in plugins/settings when you are comfortable. The repo already includes the shared plugin/theme config.
5. Start with [[2026-06-05_polymarket_research_onboarding]], then read [[COWORK]], [[TODO]], [[POLYMARKET_BRAIN]], and [[CODEX]].

Do not worry if some links point into generated data or local-only artifacts. Data is intentionally not committed. The brain is the map; the heavy files can be regenerated or passed separately when a specific test needs them.

## What the brain gives you

- A graph of what we know, what we closed, what survived, and what is live-only.
- Backlinks between strategy hubs, findings notes, and handoffs, so you can see why a conclusion exists instead of re-running dead branches.
- A shared task source of truth in [[TODO]].
- A prompt discipline for future Codex runs, so implementation sessions start with the right context.
- A place for strategic synthesis that is separate from generated data and code.

## Navigation

Use the hubs first:

- [[COWORK]]: strategic orientation and how Cowork/Codex should collaborate.
- [[CODEX]]: implementation-agent rules, repo layout, realism calibration, and startup checklist.
- [[TODO]]: authoritative live task list.
- [[POLYMARKET_BRAIN]]: Obsidian map of Polymarket strategy clusters.
- [[glossary]]: terms, metrics, wallet labels, and execution vocabulary.

Useful Obsidian moves:

- Click wikilinks like `[[TODO]]` or `[[strat_market_making]]` to jump through the graph.
- Use Backlinks to see which notes depend on the current note.
- Use Graph View to spot hubs, orphans, and broken links.
- Use search for wallet addresses, block names, strategy labels, and result phrases like `CLOSED`, `MERITS-LIVE-MEASUREMENT-LOOP`, or `mark-to-mid`.

## Cowork and Codex

Cowork is for strategic discussion, interpretation, and prompt drafting. Codex is for repo inspection, implementation, tests, long analyses, and writing concrete findings docs.

Any serious Codex prompt should tell Codex to read these in order:

1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `brain/POLYMARKET_BRAIN.md`
5. The relevant strategy hub or findings note

That read order is important. It prevents a new session from optimizing a stale idea or missing a branch that was already falsified.

## Personalizing your own brain

This commit gives you the shared seed brain. After that, personalize carefully:

- Edits to already-tracked brain files show up in git and can be shared normally.
- New files under `brain/` are ignored by default because `.gitignore` still has `brain/`.
- To share a new brain note, intentionally force-add it with `git add -f brain/path/to/note.md`.
- For private scratch notes, keep them under `brain/personal/` or another clearly local path and do not force-add them.
- Do not commit `.env` files, credentials, generated data, venvs, `.tmp/`, or `.obsidian/workspace.json`.

Recommended pattern:

- Shared strategic handoff: `brain/handoffs/<YYYY-MM-DD>_<topic>.md`
- Shared Polymarket findings: `polymarket/research/notes/<cluster>/<topic>_findings.md`
- Task state: edit [[TODO]]
- Terminology: edit [[glossary]]
- Private thinking: local ignored note unless it becomes useful to the team

## Operating principle

The brain is not a diary dump. It is a memory system for decisions. A good note should say what we tested, what survived, what closed, why, and where to go next. If a note changes the strategy map, link it from the relevant hub and update [[TODO]].
