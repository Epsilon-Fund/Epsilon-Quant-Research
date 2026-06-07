---
name: brain-rock-tumbler
description: >
  Use for epsilon-quant-research when messy research, scratch notes,
  notebooks, scripts, or exploratory branches need to become one clean
  canonical Markdown findings note. Trigger when the user says Rock Tumbler,
  tumble this branch, make this durable, consolidate findings, close a
  research branch, promote scratch to findings, or summarize a mature
  experiment.
---

# Brain Rock Tumbler

Use this skill to turn messy work into a canonical note that a cold reader can trust.

## Report-First Workflow

1. Read `brain/SKILL_MAP.md`, `brain/VAULT_MAP.md`, `brain/CODEX.md`, `brain/COWORK.md`, and the relevant strategy hub.
2. Gather source material: scratch notes, notebooks, result CSVs, scripts, existing notes, recent commits, and TODO entries.
3. Identify the canonical destination from `brain/VAULT_MAP.md` "Where to write things".
4. Write or update one durable `*_findings.md` or strategy note with:
   - Hub backlinks near the top.
   - YAML frontmatter following `brain/CODEX.md`.
   - `## Summary` in plain English.
   - Research question, data/experiment, evidence, decision, and next gate.
5. Link source artifacts without copying large raw outputs into the note.
6. Update the relevant hub or `brain/TODO.md` only if status/next gate changed.

## Guardrails

- Do not fabricate results. Pull every claim from existing artifacts.
- Preserve exact numbers, costs, dates, CIs, and falsification decisions.
- Mark weak or incomplete evidence clearly.
- Keep failed branches durable if they prevent rediscovery.

## Good Output

- One canonical note replaces a pile of scratch context.
- The note explains what was tested, why, what happened, decision, and next gate.
- A future agent can continue without reading the whole branch history.
