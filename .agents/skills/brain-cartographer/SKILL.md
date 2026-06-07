---
name: brain-cartographer
description: >
  Use for epsilon-quant-research vault navigation and map maintenance:
  updating `brain/VAULT_MAP.md`, active research branch tables, where-to-write
  rules, generated indexes, graph audit interpretation, branch registry
  accuracy, or stale map cleanup. Trigger when the user says Cartographer,
  update the vault map, new branch/folder appeared, folder moved, map is
  stale, graph audit has structural issues, or agents are getting lost.
---

# Brain Cartographer

Use this skill to keep the brain navigable without restructuring the repo unnecessarily.

## Report-First Workflow

1. Read `brain/VAULT_MAP.md`, `brain/SKILL_MAP.md`, `brain/OPERATING_RHYTHMS.md`, `brain/TODO.md`, and relevant hubs.
2. Inspect actual repo state with targeted `find`, `rg`, and `git status` commands.
3. Run `python3 tools/brain_hygiene.py` and, when graph shape matters, `python3 tools/brain_graph_audit.py`.
4. Compare map claims to reality:
   - Active research branches.
   - Where-to-write paths.
   - Core hubs.
   - Generated reports and data manifests.
   - Agent lanes.
5. Fix inaccuracies. Avoid broad taxonomy changes unless explicitly requested.
6. Re-run the relevant scanner and summarize map deltas.

## Guardrails

- Do not invent hub notes just because a map references one. Flag missing hubs.
- Do not move files as part of Cartographer unless the user asked for restructuring.
- Keep `VAULT_MAP.md` concise; detailed plans belong in roadmap or handoff notes.

## Good Output

- The map sends humans and agents to the right place on the first try.
- New branches and closed branches have accurate status and next navigation point.
