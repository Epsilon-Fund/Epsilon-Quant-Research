---
title: Obsidian Infrastructure Roadmap
created: 2026-06-07
status: draft
tags:
  - obsidian
  - brain
  - collaboration
  - agent-infra
hubs:
  - CODEX
  - COWORK
  - POLYMARKET_BRAIN
---

# Obsidian Infrastructure Roadmap

> Brain = the collection of Markdown files in this repo/vault.
> Audience = Justin, coworker, Codex/Cowork agents, and future automation systems.

Hub links: [[CODEX]] | [[COWORK]] | [[POLYMARKET_BRAIN]] | [[TODO]] | [[glossary]]

> **Implementation update (2026-06-10, current):** the live-sync layer this roadmap originally recommended (Obsidian Relay, with `tools/brain_edit_guard.py` cooperative edit locks) was used 2026-06-07→10 and then **retired**. The collaboration layer of record is now the **git branch-per-person model**: each collaborator works on a personal branch named by GitHub handle, `main` is the integration branch, and merges into main are deliberate and agent-assisted. See [[MERGE_PROTOCOL]] and [[ONBOARDING]] § Collaboration model; decision record in [[2026-06-10_relay_retirement_branch_model]]. The navigation/hygiene layers below (maps, skills, generated reports, graph audits) remain current. Superseded sync-layer rationale was removed from this file and is preserved in git history.

## Executive Recommendation

Use **git (branch-per-person) for the shared Markdown layer**, and build an **AI-readable navigation layer** on top of the existing repo layout.

The current best setup is:

1. **Branch-per-person git collaboration:** one personal branch per collaborator; `main` as integration branch; deliberate agent-assisted merges ([[MERGE_PROTOCOL]]).
2. **Separate agent lanes:** keep Codex, Cowork/Claude Code, and any future agent prompts/scratch logs separate so each agent can have its own formatting, startup instructions, and response style.
3. **GitHub history layer:** commits, rollback, branch history, and code/data work all live in the same repo and model.
4. **Vault map:** add a compact master route map so agents do not scan the whole repo every time.
5. **Skill map:** define repeatable agent/human workflows such as daily brief, chronicler, janitor, sherpa, and rock tumbler.
6. **Query and graph hygiene:** use QMD/Markdown queries plus Graphify-style graph audits to find stale branches, orphan notes, missing backlinks, and duplicate topic islands.

This is better than forcing the whole repo into a new folder taxonomy immediately. The current repo already has useful domain structure. The missing layer is not "more folders"; it is **maps, conventions, and scheduled hygiene**.

## Current Setup Facts

| Area | Current read |
|---|---|
| Vault/repo root | `/Users/justiniturregui/Desktop/github/epsilon-quant-research` |
| Obsidian attachment default | `Attachments/` |
| Markdown brain size | Small: 233 non-venv/non-Git `.md` files, about 2.9 MB total, measured 2026-06-07 |
| All Markdown | 309 `.md` files including hidden/vendor-ish areas, still tiny relative to the repo |
| Full repo size | Huge: about 160 GB, mostly data/caches and Git history |
| Largest local areas | `.git` about 30 GB, `polymarket/research/data` about 119 GB, `topics/momentum` about 8.7 GB |
| Attachments folder | Currently empty locally; attachment policy can stay simple for now |
| GitHub caveat | Git is history-strong; same-note concurrency is handled by the branch model + merge protocol rather than live sync |
| Current Git backup gap | resolved 2026-06-07 — `.gitignore` inverted so `brain/**/*.md` is tracked |

## Collaboration Operating Model

Git is both the **working collaboration layer** and the **audit/history layer**: one personal branch per collaborator, `main` as integration branch, deliberate agent-assisted merges. The procedure of record is [[MERGE_PROTOCOL]]; collaborator setup is [[ONBOARDING]] § Collaboration model. (The 2026-06 evaluation of live-sync alternatives that this section previously contained is preserved in git history; it was superseded when the live-sync layer was retired on 2026-06-10.)

Rules that remain current regardless of sync layer:

1. Keep generated data and large research artifacts out of the note collaboration layer.
2. Keep agent-specific prompting and scratch work in agent-specific lanes.
3. Treat canonical shared hubs as summary surfaces, not live scratchpads.
4. When two agents are active, each agent writes to its own scratch/daily note first, then one chronicler pass updates canonical notes.

## Recommended Folder Philosophy

Keep the existing domain folders and add a thin operating layer.

### Existing domain folders stay meaningful

| Domain | Keep writing here |
|---|---|
| Polymarket findings | `polymarket/research/notes/` |
| Polymarket maps/synthesis | `polymarket/research/notes/overview/` |
| Crypto strategy research | `topics/<strategy>/research/` and `docs/` |
| Cross-project handoffs | `brain/handoffs/` |
| Shared canonical orientation | `brain/POLYMARKET_BRAIN.md`, `brain/TODO.md`, `brain/VAULT_MAP.md` |
| Codex operating prompt/context | `brain/CODEX.md` or future `brain/agents/codex/` |
| Cowork/Claude Code operating prompt/context | `brain/COWORK.md` or future `brain/agents/cowork/` |
| Attachments | `Attachments/` |

### New operating layer

Add these when we build out the infra:

```text
brain/
|-- OBSIDIAN_INFRA_ROADMAP.md      # this roadmap
|-- VAULT_MAP.md                   # master table of contents for humans and agents
|-- SKILL_MAP.md                   # repeatable agent workflows and when to run them
|-- OPERATING_RHYTHMS.md           # daily/weekly/monthly hygiene cadence
|-- GENERATED_INDEX.md             # generated index, refreshed by scripts
|-- agents/
|   |-- codex/                     # Codex-specific prompts, scratch logs, style rules
|   `-- cowork/                    # Cowork/Claude Code-specific prompts, scratch logs, style rules
|-- generated/
|   |-- graph_audit.md             # orphan, hub, duplicate, and stale-branch audit
|   |-- daily_brief.md             # generated current-state brief
|   `-- stale_notes.md             # old or unlinked note report
`-- daily/
    `-- YYYY-MM-DD.md              # daily log if we choose brain-native daily notes
```

Important Git note: resolved 2026-06-07 — `.gitignore` was inverted so new `brain/**/*.md` is tracked automatically; only `brain/generated/`, `local_agents/`, and `scratch/` stay ignored.

### Agent Separation Model

Keep the **knowledge brain shared** and the **agent operating surfaces separate**.

This avoids the worst collision case: two agents using the same canonical hub as a scratchpad. Codex and Cowork/Claude Code can have different model behavior, response formatting, startup checklists, and note-writing style without forking the actual research memory.

Recommended layers:

| Layer | Purpose | Who edits | Example files |
|---|---|---|---|
| Canonical shared brain | Durable research truth, roadmaps, decisions, status | one owner at a time, usually via chronicler pass | `brain/VAULT_MAP.md`, `brain/TODO.md`, strategy hubs |
| Codex lane | Shared Codex-type startup, prompting, and formatting rules | Codex-style agents, intentionally | `brain/CODEX.md`, `brain/agents/codex/` |
| Cowork lane | Shared Cowork/Claude Code-type startup, prompting, and formatting rules | Cowork-style agents, intentionally | `brain/COWORK.md`, `brain/agents/cowork/` |
| Scratch/session notes | Fast-moving work-in-progress | the active agent only | `scratch/<agent>/YYYY-MM-DD.md` |
| Chronicled outputs | Clean summaries promoted from scratch | assigned chronicler | `brain/handoffs/`, findings notes, branch summaries |

Rules:

1. Agents do not use canonical hubs as scratchpads.
2. Agents can update their own lane freely.
3. Shared canonical files get updated through short, intentional passes.
4. If both agents need the same canonical file, one agent owns the edit and the other writes a linked scratch note.
5. Concurrent same-note edits resolve at merge time per [[MERGE_PROTOCOL]]; agents should still avoid making them the default workflow.

## What To Learn From The Reference Systems

### Karpathy LLM Wiki

Reference: [Karpathy LLM wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

Useful lessons:

- Treat Markdown as a durable local wiki, not as disposable chat logs.
- Prefer compact canonical pages over scattered transcripts.
- Build a clear "start here" surface for humans and agents.
- Make the wiki easy for tools to read, search, query, and regenerate.
- Let the graph emerge from links, but do not depend on the visual graph alone.

Implementation idea:

- `[[VAULT_MAP]]` becomes the agent start page.
- Each research branch gets one canonical hub note.
- Messy research gets tumbled into polished findings notes rather than accumulating as raw chat fragments.

### ar9av/obsidian-wiki

Reference: [ar9av/obsidian-wiki](https://github.com/ar9av/obsidian-wiki)

Useful lessons:

- Design the vault as an agent-operable knowledge base.
- Give agents repeatable skills for reading, writing, querying, and linking notes.
- Treat Obsidian not just as a note app, but as the interface for structured collaboration.

Implementation idea:

- Create a repo-specific `[[SKILL_MAP]]`.
- Encode what each agent should do when starting, ending, summarizing, grooming, or auditing work.
- Keep local command/query recipes near the notes they operate on.

### kepano/obsidian-skills

Reference: [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills)

Useful lessons:

- Agents need Obsidian literacy: wikilinks, aliases, embeds, callouts, properties, tags, headings, canvases, and backlinks.
- Notes should be written for graph navigation, not just as flat Markdown.
- Style conventions matter because they make future automation easier.

Implementation idea:

- Add an Obsidian writing standard to `[[SKILL_MAP]]`.
- Define required frontmatter for durable notes.
- Require unique basenames, hub backlinks, short summaries, and status tags.

### Graphify

Reference: [safishamsi/graphify](https://github.com/safishamsi/graphify)

Useful lessons:

- Graph structure is a hygiene signal.
- Orphans, dead ends, duplicate clusters, missing hubs, and overgrown hubs should be visible.
- A graph audit should produce actionable cleanup tasks, not just a pretty visualization.

Implementation idea:

- Weekly graph audit produces `brain/generated/graph_audit.md`.
- The janitor system converts graph issues into concrete fixes.
- The vault map links to graph audit outputs.

### PARA

PARA = Projects, Areas, Resources, Archives.

Useful lessons:

- Every note should have a life-cycle state.
- Active work should be easy to find.
- Archived or falsified branches should not keep looking active.

Implementation idea:

- Use PARA as a **metadata overlay**, not a forced folder migration.
- Add `para: project|area|resource|archive` to frontmatter when useful.
- Keep domain folders intact, but make branch state visible.

### QMD

Reference: [QMD](https://github.com/sambecker/qmd)

Useful lessons:

- Markdown can be queried like a lightweight knowledge database.
- Queries are ideal for indexes, stale-note audits, tag audits, and daily briefs.
- Query outputs can become generated Markdown files that agents read first.

Implementation idea:

- Use QMD or similar local Markdown queries to generate `generated/GENERATED_INDEX.md`.
- Query for missing frontmatter, stale tasks, unlinked notes, duplicate basenames, and notes without hub backlinks.

### Cron Jobs And Automations

Useful lessons:

- Hygiene should not depend on memory or motivation.
- Run small, boring checks on a schedule.
- Generated reports should be Markdown notes inside the vault so humans and agents can read them.

Implementation idea:

- Daily: brief, changed-note digest, active task extraction.
- Weekly: graph audit, stale branch report, orphan link cleanup.
- Monthly: archive review, strategy map refresh, skill map refresh.

## Infrastructure Pieces To Build

### 1. Vault Map

Purpose: let agents and humans navigate the repo without scanning every folder.

File: `brain/VAULT_MAP.md`

Suggested contents:

- one-line repo purpose
- top-level folder map
- core hubs and what each owns
- where to write each type of note
- active research branches
- closed/falsified branches
- generated reports
- commands or queries for deeper inspection

Good vault map shape:

```text
# Vault Map

## Start Here
Read [[CODEX]], [[TODO]], [[COWORK]], then the relevant project hub.

## Core Hubs
| Hub | Owns | Read when |

## Where To Write
| Content type | Path | Required backlink |

## Active Branches
| Branch | Owner | Folder | Status | Next gate |

## Generated Reports
| Report | Source | Refresh cadence |
```

### 2. Indeaverse Roadmap

Purpose: make the repo's idea-space navigable, especially when multiple research lines are alive.

Working definition: the Indeaverse is the map of repo ideas, not just the file tree. It should show how research branches, strategy concepts, findings, closures, and implementation paths relate.

Possible outputs:

- `brain/IDEA_GRAPH.md`: human-readable map of concepts and active branches.
- `brain/ROADMAP_INDEX.md`: project roadmaps and their links.
- `brain/generated/branch_registry.md`: generated list of branch notes and status.
- Obsidian Canvas for visual branch layout if we want a more spatial view.

Recommendation: start with Markdown maps before Canvas. Markdown is easier for agents to update reliably.

### 3. Skill Map

Purpose: define repeatable agent workflows so brain hygiene becomes an operating system.

File: `brain/SKILL_MAP.md`

Candidate skills:

| Skill | Purpose | Output |
|---|---|---|
| Daily Brief | Summarize what changed, what matters, and what is next | `brain/generated/daily_brief.md` |
| Agent Scratch Log | Let each agent write fast without colliding with another agent | `scratch/<agent>/YYYY-MM-DD.md` |
| Daily Log | Record what happened today in a durable shared note | `brain/daily/YYYY-MM-DD.md` |
| Sherpa System | Route a human/agent to the right context for a task | context pack and suggested reads |
| Rock Tumbler System | Turn messy findings into polished canonical notes | cleaned findings note with links and decisions |
| Chronicler System | Capture decisions, closures, handoffs, and why things changed | dated handoff or decision note |
| Janitor System | Fix stale links, orphan notes, duplicate basenames, and missing metadata | cleanup report plus small edits |
| Cartographer System | Maintain vault map, idea graph, and generated indexes | refreshed maps |
| Librarian System | Maintain glossary, table dictionaries, manifests, and definitions | definition updates |

### 4. Generated Index

Purpose: reduce scan cost for agents.

File: `brain/GENERATED_INDEX.md`

This should be generated, not hand-maintained forever.

Include:

- all Markdown files by folder
- recently changed notes
- orphan notes
- notes missing hub backlinks
- notes missing summary/frontmatter
- duplicate basenames
- stale TODOs
- active research branches

### 5. Operating Rhythms

Purpose: turn hygiene into a schedule.

File: `brain/OPERATING_RHYTHMS.md`

Suggested cadence:

| Cadence | System | What it does |
|---|---|---|
| Daily morning | Daily Brief | Reads changed notes, TODOs, and recent Git state |
| During active work | Agent Scratch Log | Keeps Codex/Cowork WIP separate until promoted |
| Daily end | Chronicler | Writes what changed and what decisions were made |
| Every research branch close | Rock Tumbler | Converts messy branch work into a canonical findings note |
| Weekly | Janitor | Link, metadata, duplicate basename, stale task cleanup |
| Weekly | Graphify/Cartographer | Refresh graph audit and vault map |
| Monthly | Archive Review | Move closed branches to archive state and refresh roadmaps |

## Note Standards

Every durable note should be easy to read cold.

Minimum frontmatter for new durable notes:

```yaml
---
title: Human Readable Title
created: YYYY-MM-DD
status: active
owner: justin
project: polymarket|crypto|infra|cross-project
para: project|area|resource|archive
hubs:
  - CODEX
tags:
  - research
---
```

Minimum body:

1. Hub backlinks near the top.
2. Plain-English summary.
3. Why this note exists.
4. Findings, decision, or open question.
5. Next action or closure status.

Recommended status vocabulary:

| Status | Meaning |
|---|---|
| `active` | Work is live and may change decisions |
| `candidate` | Interesting but not yet validated |
| `watching` | Waiting on data, market state, or another branch |
| `closed` | Branch completed or falsified |
| `archived` | Useful history, not active |
| `generated` | Produced by script/automation |

## Research Branch Rules

A new research branch should have:

- a unique note/folder name
- one hub backlink
- one owner
- a clear status
- a clear next gate
- a final chronicler/rock-tumbler pass before being called done

Recommended branch note shape:

```text
# Branch Name

Hub links: [[COWORK]] | [[POLYMARKET_BRAIN]] | [[TODO]]

## Summary
What this branch tests and why it matters.

## Current Status
active/candidate/closed, with one-line reason.

## Evidence
Links to notes, scripts, results, charts, and data manifests.

## Decision
Continue, pause, archive, or promote.

## Next Gate
The next falsifiable action.
```

## Attachments

Current policy:

- default Obsidian attachments folder: `Attachments/`
- keep images/PDFs there unless a local strategy folder already owns a figure/gallery convention
- Markdown notes should link or embed attachments with enough caption context to be useful later
- attachments sync through git like any other tracked file; keep them small and descriptive

Ranked attachment options:

| Rank | Option | Pros | Cons | Verdict |
|---:|---|---|---|---|
| 1 | `Attachments/` at vault root | Simple, matches current setting, easy to audit | Can become messy without naming rules | Recommended now |
| 2 | `Attachments/<topic>/` subfolders | Cleaner for big topic clusters | Requires people to choose folders consistently | Use once attachment volume grows |
| 3 | Per-note attachment folders | Best local organization | More complex and noisy in repo | Only if attachment-heavy work starts |
| 4 | Store figures beside generated outputs | Best for code-owned plots | Less Obsidian-native; can scatter media | Use for research scripts and galleries |

Naming rule: prefer descriptive filenames like `2026-06-07_kpeg_oos_surface.png` over screenshots with generic names.

## Roadmap

### Phase 0: Sync Model And Source-Of-Truth Rules

Status: done (final form 2026-06-10).

Outcome: the sync layer is the **git branch-per-person model** ([[MERGE_PROTOCOL]], [[ONBOARDING]] § Collaboration model). `.gitignore` was inverted so `brain/**/*.md` is tracked; Obsidian attachment default is `Attachments/`; coworker onboarding is [[ONBOARDING]] § Onboarding a new collaborator. An interim live-sync layer with cooperative edit locks ran 2026-06-07→10 and was retired — see [[2026-06-10_relay_retirement_branch_model]].

### Phase 1: Core Maps

Build:

- `brain/VAULT_MAP.md`
- `brain/SKILL_MAP.md`
- `brain/OPERATING_RHYTHMS.md`
- `brain/agents/codex/`
- `brain/agents/cowork/`

Outcome:

- agents can start from a small number of files
- coworker has one place to understand where notes go
- new research branches have a predictable path
- Codex and Cowork can work without same-note scratch collisions

### Phase 2: Hygiene Scripts

Build scripts or QMD queries for:

- duplicate basenames
- broken wikilinks
- orphan notes
- notes without hub backlinks
- notes without summary/frontmatter
- stale TODOs
- recently changed Markdown digest

Outcome:

- `brain/GENERATED_INDEX.md`
- `brain/generated/stale_notes.md`
- `brain/generated/daily_brief.md`

### Phase 3: Graph Layer

Build:

- Graphify-style graph audit
- hub centrality report
- orphan/dead-end report
- stale branch map

Outcome:

- `brain/generated/graph_audit.md`
- optional Obsidian Canvas or visual graph artifact

### Phase 4: Agent Workflow Layer

Build:

- Daily Brief system
- Daily Log system
- Sherpa system
- Rock Tumbler system
- Chronicler system
- Janitor system
- Cartographer/Librarian systems

Outcome:

- agents become more consistent
- research gets captured as durable knowledge
- cleanup is scheduled rather than ad hoc

### Phase 5: Indeaverse

Build:

- idea graph
- strategy branch registry
- roadmap index
- cross-project link map

Outcome:

- the brain becomes navigable by idea, not just by folder
- repo ideas, active threads, closed branches, and implementation paths are linked cleanly

## Open Decisions

| Decision | Options | Current lean |
|---|---|---|
| Sync layer | live-sync plugin vs git branch-per-person | **Decided 2026-06-10:** git branch-per-person ([[MERGE_PROTOCOL]]) |
| Should the brain be split per agent? | separate whole brains vs shared canonical brain plus agent lanes | Shared canonical brain plus separate agent lanes |
| Where should generated reports live? | `brain/generated/` vs `docs/obsidian/generated/` | `brain/generated/` (git-ignored, regenerable) |
| Where should daily logs live? | `brain/daily/`, existing `journal_logs/`, or `scratch/<agent>/` | Agent scratch in `scratch/<agent>/`; shared daily log only after chronicler pass |
| Should PARA be folder-based? | full migration vs metadata overlay | Metadata overlay first |

## Immediate Next Build Steps

1. Create `[[VAULT_MAP]]`.
2. Create `[[SKILL_MAP]]`.
3. Create `[[OPERATING_RHYTHMS]]`.
4. Create `brain/agents/codex/` and `brain/agents/cowork/` operating lanes; keep scratch in top-level `scratch/<agent>/`.
5. Add a small hygiene script or query that reports duplicate basenames, orphan notes, and notes missing hub backlinks.
6. Decide which core infra files should be tracked in Git even though `brain/` ignores new files.
7. Run one janitor pass and one chronicler pass after a normal workday to prove the cadence.

## Working Principle

The goal is not to make the vault pretty. The goal is to make the brain **fast to enter, hard to lose, easy to audit, and reliable for multiple agents and humans working in parallel**.
