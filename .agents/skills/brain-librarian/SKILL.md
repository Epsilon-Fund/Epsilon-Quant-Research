---
name: brain-librarian
description: >
  Use for epsilon-quant-research glossary, table dictionary, schema,
  shorthand, acronym, bucket-label, column-name, and terminology maintenance.
  Trigger when a reusable term appears, a CSV/parquet column needs a
  definition, a strategy nickname becomes shared vocabulary, or the user says
  Librarian, define this, add to glossary, or update table dictionary.
---

# Brain Librarian

Use this skill to define reusable language once and link to it, so agents do not repeatedly rediscover meanings.

## Report-First Workflow

1. Read `brain/glossary.md`, the relevant data manifest, and any local table dictionary such as `polymarket/research/notes/overview/data_quality/polymarket_table_dictionary.md`.
2. Search the term with `rg` to see how it is used in context.
3. Decide whether the term belongs in:
   - `brain/glossary.md` for cross-project concepts and acronyms.
   - A table dictionary for columns, bucket labels, or schema terms.
   - A strategy hub for branch-local vocabulary.
4. Add a compact definition, aliases if useful, and one or two links to canonical context.
5. Update nearby notes only when a backlink clearly improves navigation.

## Guardrails

- Do not define one-off terms that will not recur.
- Do not rewrite research content just to add vocabulary.
- Do not guess column meaning from a name alone. Inspect code, manifests, or data examples.
- Prefer precise short definitions over encyclopedia entries.

## Good Output

- A future reader can understand the term without scanning old sessions.
- Reused columns, buckets, and acronyms have one canonical definition.
