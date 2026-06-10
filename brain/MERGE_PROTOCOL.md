---
title: Merge Protocol — branch-per-person collaboration
created: 2026-06-10
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - ONBOARDING
tags:
  - brain
  - collaboration
  - git
  - merge
---

# Merge Protocol — branch-per-person collaboration

> How this repo's collaborators (and their agents) share work through git. This is the canonical reference for the branch model and for resolving merge conflicts in the Markdown brain. The collaborator-facing setup guide is [[ONBOARDING]] § Collaboration model; this file is the procedure.

Hub links: [[VAULT_MAP]] | [[ONBOARDING]] | [[TODO]] | [[CODEX]] | [[COWORK]]

## Summary

- Each collaborator works on a **personal branch named after their GitHub handle** (e.g. `justin`, `alvaro`). `main` is the shared integration branch; **nobody commits to main directly**.
- Daily commit/push goes to your own branch (the per-person `brain-commit-push` scheduled task enforces this — it refuses to run on main and never pulls or merges main).
- Merging a personal branch into main is a **deliberate, agent-assisted step** described below. Conflicted Markdown is resolved by a smart agent using the copy-pasteable prompt in this file — never by blind `--ours`/`--theirs`.
- Personal state (`local_agents/`, `scratch/`) is git-ignored and never participates in any branch or merge, so it can never conflict.

## 1. The branch model

| Surface | Branch | Who writes |
|---|---|---|
| `main` | shared integration baseline | nobody directly — only merges via this protocol |
| `<handle>` (e.g. `justin`, `alvaro`) | one per collaborator, created once off main | that collaborator + their agents |
| `local_agents/<handle>.md`, `scratch/<handle>/` | not in git at all (ignored) | that person only |

Rules:

- Create your branch **once**: `git checkout main && git pull && git checkout -b <handle>`.
- **EOD**: commit + push to YOUR branch only (the `brain-commit-push` scheduled task does this; it must be configured per person and refuses to run on main).
- **Session start, and after anyone merges to main**: catch up with `git checkout <handle> && git merge main`. Resolve any conflicts using § 3 below. Branches that don't pull main in regularly drift and make merges painful.
- Merge to main in **small, frequent units** (a finished note, a completed task) — not big infrequent dumps.
- Recommended: a GitHub branch-protection rule on `main` blocking direct pushes, if the plan supports it.

## 2. Merging a personal branch into main

Run when a unit of work on a personal branch is ready to become shared baseline:

```bash
git checkout main
git pull origin main
git merge <person-branch>        # e.g. git merge alvaro
```

- **No conflicts:** review `git diff origin/main --stat` sanity, then `git push origin main`. Done — tell the other collaborator(s) so they merge main into their branch.
- **Conflicts:** for each conflicted `.md` file, hand the file to a smart agent (Claude Code / Codex) with the SMART-MERGE-AGENT PROMPT in § 3. The agent resolves the file (or flags it for human decision), you `git add` it, then:

```bash
git commit                        # completes the merge
git push origin main
```

- **After every merge to main, everyone catches up:** each collaborator runs `git checkout <handle> && git merge main` on their machine at next session start.

Non-Markdown conflicts (code, configs) follow normal engineering judgment — the smart-merge prompt below is for the Markdown brain.

## 3. SMART-MERGE-AGENT PROMPT (copy-paste, one conflicted file per run or batched)

```markdown
You are resolving git merge conflicts in the Markdown "brain" of the epsilon-quant-research repo, merging a personal branch into main. Read brain/MERGE_PROTOCOL.md and brain/VAULT_MAP.md first for context. For each conflicted .md file, produce a single resolved version that obeys these rules, in priority order:

1. NEVER drop an open `[ ]` task. If either side has an open task the other lacks, the resolution keeps it.
2. TODO/task sections are resolved by UNION: keep both sides' new items, dedupe items that are the same task by content (not by exact string), and preserve checkbox state — if either side checked an item `[x]`, it stays checked; an item open on one side and absent on the other stays open.
3. Law/map files (CODEX.md, COWORK.md, VAULT_MAP.md, ONBOARDING.md, MERGE_PROTOCOL.md, SKILL_MAP.md, OPERATING_RHYTHMS.md, lane docs): prefer the timeless, person-agnostic version of a passage. Dated status prose does not belong in law files — if one side added it, move it to brain/TODO.md or a dated brain/handoffs/ note instead of keeping it in the law file.
4. New findings notes with distinct basenames are KEPT on both sides — never delete one to "simplify." (Distinct basenames cannot conflict in git; this rule is about not pruning during cleanup.)
5. Genuine semantic conflict — the same claim, number, verdict, or decision edited two different ways — do NOT guess and do NOT average. Emit BOTH versions inline, clearly labeled:

   > ⚠️ MERGE CONFLICT — HUMAN DECISION NEEDED
   > Version A (main): ...
   > Version B (<branch>): ...

   and list every such flag in your final report.
6. Preserve frontmatter; if both sides edited frontmatter, union tags/hubs and keep the most recent date fields.
7. After resolving, reread the file cold per CODEX.md § Markdown quality standard: no orphaned conflict markers, no broken wikilinks introduced.

Report per file: what was unioned, what was deduped, and any human-decision flags. Do not `git add` or commit unless asked.
```

## 4. What never conflicts

- `local_agents/<handle>.md` (personal agent overlays) and `scratch/<handle>/` (personal WIP) are **git-ignored** — they exist only on each person's machine, are never on any branch, and therefore can never appear in a merge. Keep per-person "current focus / waiting-on" state there (the templates in `brain/agents/templates/` have a dedicated section for it).
- `brain/generated/` reports are git-ignored and regenerable — never resolve a conflict in them; regenerate with `python tools/brain_hygiene.py`.

## 5. Per-person scheduled task

The daily `brain-commit-push` scheduled task is **per-person** and must target its owner's branch, never main. Justin's is configured (definition lives outside the repo at `~/Documents/Claude/Scheduled/brain-commit-push/SKILL.md`); each new collaborator configures their own equivalent on their machine — see [[ONBOARDING]] § Onboarding a new collaborator. The task: stages brain + research-notes Markdown only, commits, pushes to the current personal branch; refuses to run on main; aborts and reports on push/pull failure; never force-pushes; never pulls or merges main (catching up from main is a deliberate session-start step, not part of the EOD push).
