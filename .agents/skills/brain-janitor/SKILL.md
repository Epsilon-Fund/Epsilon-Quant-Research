---
name: brain-janitor
description: >
  Use for epsilon-quant-research Obsidian brain hygiene cleanup: fixing broken
  wikilinks, duplicate basenames, missing YAML frontmatter, missing Summary
  sections, missing hub backlinks, stale TODOs, orphan notes, or issues
  reported by `tools/brain_hygiene.py` / `brain-hygiene-weekly`. Trigger
  when the user says Janitor, hygiene pass, cleanup the brain, fix scanner
  issues, or asks to act on `brain/generated/hygiene_report.md`.
---

# Brain Janitor

Use this skill to fix issues found by the brain hygiene scanners. The scanner finds issues; the Janitor makes deliberate, reviewable edits.

## Report-First Workflow

1. Read `brain/SKILL_MAP.md`, `brain/OPERATING_RHYTHMS.md`, `brain/VAULT_MAP.md`, and the relevant hub.
2. Run `python3 tools/brain_hygiene.py`.
3. Read `brain/generated/hygiene_report.md`, `brain/generated/stale_notes.md`, and `brain/GENERATED_INDEX.md` if present.
4. Group fixes by type: broken links, duplicate basenames, missing frontmatter, missing summaries, missing hub backlinks, stale TODOs.
5. Fix in small batches. Keep wikilinks intact. If renaming a note, update inbound wikilinks in the same batch.
6. Re-run `python3 tools/brain_hygiene.py` and report before/after counts.
7. Write or update a short handoff in `brain/handoffs/` if the pass changes durable state.

## Guardrails

- Do not alter research conclusions, numbers, confidence intervals, or decisions.
- Do not edit `brain/generated/`; it is regenerable.
- Do not put code or scripts in `brain/`.
- Do not silently delete note content. Reconcile or rename instead.
- Prefer reviewable batches over one large mixed edit.

## Good Output

- Broken links, duplicate basenames, orphans, and missing hub backlinks trend toward zero.
- Frontmatter and Summary backlog drops substantially.
- The final response includes before/after scanner counts and any flagged-but-unfixed items.
