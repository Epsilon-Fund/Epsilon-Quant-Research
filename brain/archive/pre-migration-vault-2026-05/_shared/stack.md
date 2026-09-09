---
title: "Stack inventory (May 2026)"
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

# stack inventory

## languages / libs

- Python: pandas, numpy, Optuna (TPE), Plotly, `python-binance`, DuckDB
- Jupyter notebooks for crypto research
- `uv` for Python packaging (polymarket research stack)
- `py-clob-client` (via vendored kernel) for Polymarket order submission
- `polymarket_sdk_signer` (vendored from midas) for order signing

## ide / tooling

- VS Code (Mac)
- Claude Code for engine / config edits
- Cowork (this folder) for journal + research notes

## data sources

- **Crypto** — Binance API (`get_historical_klines`)
- **Macro** — FRED, ECB Data Portal, Eurostat, DBnomics
- **Crypto analytics** — Aperiodic
- **Polymarket — historical trades** — Goldsky GraphQL (Polymarket subgraph) for delta; warproxxx CSV (Dec 2022 → Oct 2025) for bulk seed
- **Polymarket — markets metadata** — Gamma API
- **Polymarket — live trades** — RTDS WebSocket (`wss://ws-live-data.polymarket.com`)
- **Polymarket — order venue** — CLOB via `py-clob-client` (kernel)

## storage / processing

- **DuckDB over Parquet glob** — primary analytical store for polymarket research. No Postgres.
- **Append-only Parquet shards**, schema-uniform across shards.
- **Append-only JSONL journal** — execution bot's source of truth (positions, dedup, in-flight orders all journal-replayed at startup; no DB).
- Gitignored: `data/`, `*.parquet`, `.env`.

## automation

- n8n (Docker, self-hosted, localhost) — newsletter pipeline
- Gemini API (Basic LLM Chain node)
- GitHub API — auto-commit to `epsilon-fund.github.io`

## deployment targets

- **Hyperliquid vault** — crypto momentum strategies (spot: BTC/ETH/SOL only)
- **Polymarket** — copytrade bot. UK-blocked for order submission (RTDS reads work fine). VPN routes (US East / Frankfurt / Tokyo) bypass; long-term VPS in non-blocked region needed.

## repos

- `Epsilon-Fund/epsilon-quant-research` — private; Justin (Mac) + Dimitris (Windows)
  - `polymarket/research/` — cohort research stack (DuckDB + Parquet)
  - `polymarket/execution/` — live copy-trading bot (single-leader PoC)
- `Epsilon-Fund/epsilon-fund.github.io` — public Jekyll newsletter
- vendored: `_kernel/` from `midas/executor/` (treated as frozen)

## conventions

- Cross-platform paths: `os.path.join(...)` with commented-out root alternatives for Windows/Mac.
- Polymarket: lowercase 0x-prefixed addresses; `(transaction_hash, log_index)` as unique fill identifier.
- All research metrics lookahead-free (filter by timestamp before aggregating).
- Execution: idempotent submission via `client_order_id`; ACK / REJECT / AMBIGUOUS handled distinctly; atomic file writes (`.tmp` + `mv`) for any cross-process file (e.g. `leader_rankings.parquet`).
- Reference: `warproxxx/poly_data` (GPL-3.0, **patterns only — don't copy code**).
