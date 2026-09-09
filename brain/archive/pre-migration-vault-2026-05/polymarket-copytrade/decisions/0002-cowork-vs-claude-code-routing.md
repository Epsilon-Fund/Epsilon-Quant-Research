---
title: "ADR 0002 — Cowork vs Claude Code routing for Phase 5"
created: 2026-05-11
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

# 0002 — Cowork vs Claude Code routing for Phase 5

- **date**: 2026-05-11
- **status**: proposed

## Context

Phase 5 (walk-forward backtesting) involves both build work (writing the WF driver, running 72+ backtests, materialising outputs to parquet) and synthesis work (interpreting Stage 1 / Stage 1.5 results, deciding what to tweak, recapping state across sessions). Without an explicit routing rule, both kinds of work could land in either tool, producing duplicated state and unclear authoritativeness.

## Decision

**Phase 5 *build* is Claude Code work** (file access, execution, code-shape changes are authoritative). **Cowork is for synthesis / discussion / state-recap** — narrative around what was built, decisions taken, lessons learned, multi-session catch-ups.

This formalises the convention already implicit in the chat-slug routing map (`pm-*` = Claude.ai chat / narrative; `cc-*` = Claude Code / authoritative code-shape claims).

## Alternatives considered

- **No formal rule** — let each session decide. Risk: duplicated state, ambiguous authoritativeness when claims conflict.
- **All work in Claude Code** — loses the synthesis / cross-session catch-up affordance that cowork provides.
- **All work in cowork** — loses execution + file-write affordances that Claude Code provides for the build.

## Consequences

- The EOD ingest's authoritative-precedence rule (`cc-*` wins over same-day `pm-*` on code-shape claims) is the operational consequence of this convention.
- Cowork files become the durable narrative layer; the codebase + Claude Code session transcripts are the durable build layer.
- When the same fact appears in both tools, the cowork side defers to the Claude Code side on code-shape claims.

## Source

EOD summary `2026-05-11_pm-data.md`
