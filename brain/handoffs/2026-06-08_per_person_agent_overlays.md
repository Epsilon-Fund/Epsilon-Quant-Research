---
title: Per-Person Agent Overlays
created: 2026-06-08
status: active
owner: justin
project: infra
para: area
hubs:
  - VAULT_MAP
  - ONBOARDING
  - TODO
tags:
  - handoff
  - obsidian
  - collaboration
---
# Per-Person Agent Overlays

> Hub: [[VAULT_MAP]] · [[ONBOARDING]] · [[TODO]]

## Summary

- The shared brain stays shared: `brain/CODEX.md`, `brain/COWORK.md`, maps, hubs, handoffs, shared role-convention docs, templates, and edit locks remain under Relay/Git.
- Each person now has private local agent overlays in top-level `local_agents/`: `local_agents/codex.md` and `local_agents/cowork.md`.
- `local_agents/` is git-ignored, excluded from hygiene/graph scanners, and must never be added to Relay's shared-folder list.
- Any fresh agent should seed its missing local overlay from the shared template, read the overlay first, then obey the shared law.

## Model

Shared law remains shared and canonical:

- [[CODEX]] — Codex/repo implementation law.
- [[COWORK]] — Cowork strategic law and prompt discipline.
- [[VAULT_MAP]], [[ONBOARDING]], [[START_RESEARCH_IDEA]], [[TODO]] — shared orientation and workflow surfaces.
- [[codex_lane]] and [[cowork_lane]] — shared role conventions, not private per-person instruction files.
- [[codex.local.template.md|codex.local.template]] and [[cowork.local.template.md|cowork.local.template]] — shared templates for seeding private overlays.

Private local overlays stay outside Relay and Git:

- `local_agents/codex.md` — this machine's Codex personal style/preferences/workflow notes.
- `local_agents/cowork.md` — this machine's Cowork / Claude Code personal style/preferences/workflow notes.
- `scratch/<agent>/` — local WIP only.

## Bootstrap

Every fresh agent should do this first:

1. Determine role: Codex -> `local_agents/codex.md`; Cowork/Claude Code -> `local_agents/cowork.md`.
2. If the overlay is missing, create `local_agents/` and copy the matching template from `brain/agents/templates/<role>.local.template.md`, then tell the user: "seeded your local <role> overlay - edit it to set your personal preferences."
3. Read the local overlay first, then the shared law (`brain/CODEX.md` or `brain/COWORK.md`), then [[VAULT_MAP]], then [[TODO]].

Precedence: local overlay controls voice/preferences; shared law and repo invariants always win.

## Implementation Notes

- Added shared templates under `brain/agents/templates/`.
- Seeded this machine's ignored overlays under `local_agents/`.
- Added `/local_agents/` to `.gitignore`.
- Added `local_agents/` to `tools/brain_hygiene.py` and `tools/brain_graph_audit.py` exclusions.
- Reframed shared lane docs as role conventions rather than private lanes.
- Updated shared start-here docs with the bootstrap protocol.

## Remaining Action

Coworker's machine should use the same Relay shared-folder set, but keep its own top-level `local_agents/` and `scratch/` folders local. On first session, their agent should seed overlays from the shared templates.
