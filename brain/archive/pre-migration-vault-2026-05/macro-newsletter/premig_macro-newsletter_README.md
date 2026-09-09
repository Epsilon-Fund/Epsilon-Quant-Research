---
title: "Macro newsletter — n8n pipeline (May 2026)"
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

# macro newsletter

Automated weekly newsletter pipeline. n8n (Docker, self-hosted, localhost) fetches Bloomberg (Gmail), Goldman Sachs, JPMorgan, Bank of America research, AI-summarises per source, then commits Jekyll-formatted markdown to `epsilon-fund.github.io` under `_research/`.

## Status

Operational. Lower priority. Notes go here when the pipeline is touched.

## Sources

- Bloomberg (via Gmail)
- Goldman Sachs
- JPMorgan
- Bank of America
- Plus: ING Think, Capital Economics, IMF, BIS, ECB

## Stack

- n8n (Docker, self-hosted)
- Gemini API (Basic LLM Chain node) for per-source summarisation
- GitHub API for auto-commit
- Jekyll-formatted markdown output → GitHub Pages
