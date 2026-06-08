---
title: Obsidian Infrastructure Roadmap
created: 2026-06-07
status: draft
tags:
  - obsidian
  - brain
  - collaboration
  - relay
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

> **Implementation update (2026-06-08, current):** the Relay model below originally recommended a **root share**, and an interim build used a lane-separated/git-only-control-plane scheme. The **current, simplified setup** is: Relay shares the research folders + **all of `brain/`** (so [[VAULT_MAP]], [[TODO]], the hubs, handoffs, generated reports, shared agent templates/role conventions, and edit locks are live for both collaborators). The surfaces kept off Relay are each person's local agent overlays in **top-level `local_agents/<agent>.md`** and scratch in **top-level `scratch/<agent>/`** — never added to Relay and never committed, so per-person instructions and WIP can't collide. Concurrent edits to shared canonical notes are coordinated by `tools/brain_edit_guard.py` (cooperative locks that sync live via Relay). The live boundary of record is [[ONBOARDING]] § Sync model and [[VAULT_MAP]]; treat the "root share" language below as original rationale, not current config.

## Executive Recommendation

Use **Relay Free for the live Markdown layer**, keep **GitHub as backup/version history**, and build an **AI-readable navigation layer** on top of the existing repo layout.

The current best setup is:

1. **Relay root share:** share the repo/vault root so new Markdown research folders appear without manually adding each folder.
2. **Separate agent lanes:** keep Codex, Cowork/Claude Code, and any future agent prompts/scratch logs separate so each agent can have its own formatting, startup instructions, and response style.
3. **GitHub snapshot layer:** keep GitHub for commits, rollback, branch history, and code/data work, not for day-to-day collaborative note editing.
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
| Relay fit | Good for Markdown collaboration because the text layer is tiny |
| Relay Free caveat | Relay Free is mainly useful here because we care about `.md`; do not rely on it for attachment-heavy workflows |
| GitHub caveat | Git is great for history, bad for real-time same-note collaboration |
| Current Git backup gap | `.gitignore` currently ignores new files under `brain/`, so new brain docs may need force-add or a tracked docs location if Git backup matters |

## Relay Operating Model

Relay should be the **working collaboration layer** for Markdown notes. GitHub should be the **audit/history layer**.

### Why root sharing makes sense here

Root sharing solves the "new research branch" problem. If a new folder appears under the repo root and contains Markdown notes, it can show up through the shared vault without needing to manually add that folder to Relay.

This matters because research does not always start in a clean folder:

- a new strategy branch may start under `polymarket/research/notes/<cluster>/`
- a crypto branch may start under `topics/<strategy>/research/`
- a cross-project idea may start under `brain/handoffs/`
- a repo-level operating note may start under `brain/`
- a doc may start in `docs/`

Root share is acceptable because the Markdown layer is small. The heavy repo parts are mostly data, notebooks, and caches. Relay Free should not be treated as a storage layer for those.

### Rules

1. Edit shared Markdown in Obsidian/Relay first.
2. Use GitHub for snapshots, code, notebooks, and durable audit history.
3. Avoid having two people edit the same note through GitHub while others edit it through Relay.
4. Keep generated data and large research artifacts out of the note collaboration layer.
5. If attachments become important, either upgrade Relay or keep attachment workflows explicitly separate.
6. Do not paste invite keys into public docs or issues. Send keys directly to collaborators.
7. Keep agent-specific prompting and scratch work in agent-specific lanes.
8. Treat canonical shared hubs as summary surfaces, not live scratchpads.
9. When two agents are active, each agent writes to its own scratch/daily note first, then one chronicler pass updates canonical notes.
10. If a canonical note is likely to be edited by both agents, assign a temporary owner before editing it.

### Coworker Quick Start

1. Clone or open the repo locally.
2. Open the repo root as an Obsidian vault.
3. Install and enable the Relay plugin.
4. Join the Relay server using the invite key Justin sends separately.
5. Confirm the shared folder is the vault root.
6. Make a tiny test edit to an agreed note or create a temporary test note, then confirm it appears on Justin's machine.
7. Use GitHub for code and periodic backups, not as the live note-editing surface.

## Ranked Collaboration Layout Options

| Rank | Option | Pros | Cons | Verdict |
|---:|---|---|---|---|
| 1 | **Root Relay + shared canonical brain + separate agent lanes** | Captures new Markdown folders automatically; avoids same-note agent collisions; lets Codex/Cowork keep different prompting and formatting; preserves one shared research truth | Needs discipline around what is scratch vs canonical; requires a chronicler/janitor pass | Recommended |
| 2 | **Root Relay + existing repo folders + AI maps only** | Minimal migration; best fit for whole-repo discovery; keeps code/research/docs together | Agents may collide on hub files if they use the same notes as scratchpads | Good but less safe than separate agent lanes |
| 3 | **Selective Relay folders + existing repo folders** | Cleaner sharing boundary; less noise; safer if private/local notes exist outside shared folders | New research folders can be missed; requires manual maintenance; attachments outside shared folders can break | Good fallback if root share becomes noisy |
| 4 | **Full PARA folder migration** | Very clean human mental model; easy to explain Projects/Areas/Resources/Archives | Disruptive to existing code/research layout; risks breaking established paths and agent instructions | Use PARA as metadata/status overlay first |
| 5 | **Separate Obsidian brain repo** | Clean vault; easy permissions; low sync noise | Splits notes from code, scripts, data, and results; agents need cross-repo context stitching | Useful only if current repo becomes too noisy |
| 6 | **GitHub/Obsidian Git only** | Strong audit trail; free; familiar to technical users | Slow for live collaboration; merge conflicts; awkward for non-technical note editing | Backup layer, not primary collaboration layer |

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

Important Git note: because new files under `brain/` are currently ignored by `.gitignore`, either force-add core infra docs to GitHub or put generated/tracked operating docs in a tracked folder such as `docs/obsidian/`. Relay will still see local Markdown either way.

### Agent Separation Model

Keep the **knowledge brain shared** and the **agent operating surfaces separate**.

This avoids the worst Relay collision case: two agents using the same canonical hub as a scratchpad. Codex and Cowork/Claude Code can have different model behavior, response formatting, startup checklists, and note-writing style without forking the actual research memory.

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
5. Same-note Relay edits are acceptable for human collaboration, but they should not be the default for autonomous agents.

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
- do not assume Relay Free will sync attachments reliably

Ranked attachment options:

| Rank | Option | Pros | Cons | Verdict |
|---:|---|---|---|---|
| 1 | `Attachments/` at vault root | Simple, matches current setting, easy to audit | Can become messy without naming rules | Recommended now |
| 2 | `Attachments/<topic>/` subfolders | Cleaner for big topic clusters | Requires people to choose folders consistently | Use once attachment volume grows |
| 3 | Per-note attachment folders | Best local organization | More complex and noisy in repo | Only if attachment-heavy work starts |
| 4 | Store figures beside generated outputs | Best for code-owned plots | Less Obsidian-native; can scatter media | Use for research scripts and galleries |

Naming rule: prefer descriptive filenames like `2026-06-07_kpeg_oos_surface.png` over screenshots with generic names.

## Roadmap

### Phase 0: Relay And Source-Of-Truth Rules

Status: in progress.

Tasks:

- root shared folder in Relay
- coworker quick-start
- Obsidian attachment default set to `Attachments/`
- decide whether root Relay creates unacceptable noise
- decide which brain infra files should be force-added to Git despite `.gitignore`

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
| Should root Relay remain the default? | root share vs selective folders | Root share unless noisy Markdown becomes a real problem |
| Should the brain be split per agent? | separate whole brains vs shared canonical brain plus agent lanes | Shared canonical brain plus separate agent lanes |
| Where should generated reports live? | `brain/generated/` vs `docs/obsidian/generated/` | `brain/generated/` for Obsidian, force-add important files if Git backup matters |
| Where should daily logs live? | `brain/daily/`, existing `journal_logs/`, or `scratch/<agent>/` | Agent scratch in `scratch/<agent>/`; shared daily log only after chronicler pass |
| Should PARA be folder-based? | full migration vs metadata overlay | Metadata overlay first |
| Should invite keys live in notes? | yes vs no | No; send directly |
| Should attachments sync through Relay? | Free `.md` workflow vs paid attachment workflow | Free `.md` now, upgrade only if attachments become central |

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
