---
title: "Inbox — EOD ingest conventions (May 2026)"
created: 2026-05-09
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: infra
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - infra
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** Working practice from the standalone-vault era. The live equivalents are [[COWORK]], [[VAULT_MAP]] and [[MERGE_PROTOCOL]] — do not follow this note's process.

# Epsilon inbox

Drop EOD summaries from each Claude.ai chat / Claude Code session here. The `nightly-inbox-ingest` scheduled task fires at 10pm UK every day, routes each file's sections into the right cowork files, and archives the processed file to `_archive/YYYY-MM/`.

## filename convention

`YYYY-MM-DD_<chat-slug>.md`

Use these slugs for known chats so routing works automatically:

| Slug | Source | Owns |
|---|---|---|
| `pm-data` | Claude.ai-1 | polymarket data / cohort research (narrative + decisions) |
| `pm-exec` | Claude.ai-2 | polymarket exec / live bot (narrative + decisions) |
| `cc-pm-data` | Claude Code on polymarket data | code changes, audits, METRICS_REFERENCE re-runs |
| `cc-pm-exec` | Claude Code on polymarket exec | code changes, fixes, kernel/HTTP-client work |
| `cc-crypto` | Claude Code on crypto dashboard / adhoc | dashboard fixes, deploy work, anything ad-hoc |

Examples:
- `2026-05-12_pm-exec.md`
- `2026-05-12_cc-pm-data.md`
- `2026-05-14_cc-crypto.md`

For a one-off chat that doesn't fit, use a descriptive slug (e.g. `2026-05-12_macro-newsletter.md`). The ingest will route it via the `## meta` section of today's journal and flag for manual sorting.

## section → cowork-file routing

The EOD prompt produces six standard sections. Each routes as follows:

| Section | Where it lands |
|---|---|
| **1. Shipped today** | today's journal entry (sectioned by project) + weekly `progress.md` rollup |
| **2. Decisions taken** | new ADR in `decisions/` if material, else today's journal |
| **3. Bugs / surprising lessons** | `methodology-lessons.md` (append on top), cross-referenced from journal |
| **4. Open questions / blockers** | relevant `TODO.md` "now" section + Monday digest callout if strategic |
| **5. Next 1-3 todos** | relevant `TODO.md` "now" section |
| **6. Discrepancies with other chats** | today's journal with `⚠` flag, surfaced for next session |

Routing by slug:

| Slug | Cowork project |
|---|---|
| `pm-data` / `cc-pm-data` | `polymarket-copytrade/` |
| `pm-exec` / `cc-pm-exec` | `polymarket-copytrade/` |
| `cc-crypto` (and crypto-flavoured adhoc) | `crypto-momentum/` |

Claude Code outputs (`cc-*`) take precedence over Claude.ai chats on code-shape claims — if they conflict, the audit wins, and the chat's claim is logged as a discrepancy.

## universal EOD prompt

Drop this into each chat. Say "EOD" when you're closing for the day to trigger it.

```
EOD summary. Be brief. Skip sections that have nothing.

Context:
- Date: <YYYY-MM-DD>
- Scope: <one-line: what this chat owns>

1. Shipped today.
   Bullets, not prose. Include file paths, function names, commit hashes if any.

2. Decisions taken.
   What we resolved and the why. Anything that should become an ADR.

3. Bugs / surprising lessons.
   Only if real. Skip if nothing surprising. Include: what was wrong,
   how it manifested, the fix, the generalisable lesson.

4. Open questions / blockers.
   What's blocking forward motion. Who/what needs to resolve it
   (cowork-claude, claude-code, the user, a collaborator).

5. Next 1-3 todos.
   Smallest concrete actions.

6. Discrepancies with other chats (if any).
   Anything you said today that might contradict another chat or
   cowork-claude. Flag for reconciliation.

If nothing changed today, just respond "no activity" and stop.
Do not produce filler.
```

For Claude Code, replace section 1 with:

```
1. Files touched. Full paths + nature of change (created / edited / deleted).
   New tests + status. Existing tests run + pass/fail. New deps. Anything weird.
```

## skipping

If a chat had no activity, don't drop a file. The ingest will skip an empty inbox silently.

## archive

Processed files move to `_archive/YYYY-MM/<original-filename>`. Don't delete the archive — it's the audit trail for any future "wait, what did exec say on the 14th?".
