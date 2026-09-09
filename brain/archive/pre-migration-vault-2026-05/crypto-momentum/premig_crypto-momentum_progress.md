---
title: "Crypto momentum — progress log (May 2026)"
created: 2026-05-11
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: crypto
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - crypto
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** The crypto momentum book is live, but this note is a May-2026 snapshot of it, not its current state. The live record is `docs/STRATEGY_REFERENCE.md` and [[TODO]] § Crypto — do not build on this note or quote its numbers as current.

# crypto momentum — progress log

Append-only weekly rollup. Newest at top.

---

## week of 2026-05-11

- **STRATEGY_REFERENCE.md landed** — Claude Code audit of the entire crypto-momentum stack. Engine internals (`wf_engine`, `cpcv_engine`, `cpcv_portfolio`, `xs_strategy`), strategy specs (`momentum_swing`, `momentum_no_vol`, `BBBreakout`, `make_xs_strategy`), metric formulas with trustworthiness ratings, conventions. 1100-line ref doc.
- Mirrored to cowork as `research/notes/strategy-reference.md`.
- Captured **XS momentum recap** in `notes/xs-momentum-recap.md` — likely-scrapped status flagged.
- Extracted **methodology lessons** in `notes/methodology-lessons.md` — durable lessons that survive strategy decisions. Append-only ledger.
- **Major status correction (2026-05-11):** crypto-momentum is **NOT parked**. Live trading on VPS, 4 open momentum positions, real money in flight. Earlier cowork framing was wrong. Captured operational state in `notes/dashboard-status.md` from user's dashboard chat (flagged ~80% accurate, code-shape claims cross-checked against STRATEGY_REFERENCE).
- **5 new methodology lessons** appended from the dashboard chat: disk+session_state atomic sync, historical-P&L snapshotted at event, schema migrations need explicit deploy script, minimise running processes, drawdown calcs on zero-start curves need absolute dollar.
- **Open production issues** documented in README: stop loss state staleness fix-pending-VPS-verification, BB Breakout no production params, VPS security (raw port no HTTPS), execution-hour P&L unverified.
- Diagnostic still open: BNB in `momentum_swing` registry but absent from `ACTIVE_ASSETS`.
- **Evening 2026-05-11 (cc-crypto, uncommitted):** live-tendency fragment + BB manual-arm picks up WS live close + WS resilience pass (`force_reset_shared_ws()`, `↻ Reconnect WS` button, per-symbol try/except in subscription loop) + missing ENTRY backfills (ADA + BTC_0510 → 5 closed pairs) + `build_trade_pairs` matches by `position_id` with orphan-EXIT warnings + 3-4dp display tightening.

## week of 2026-05-04

- Subproject parked while polymarket is in flight.
- Catch-up summary pass scheduled after polymarket structure is seeded.

## historical (pre-cowork)

- WF engine, CPCV engine, portfolio modules built and stabilised before 2026-05-11. Detail in `notes/strategy-reference.md`.
- Bugs fixed during build-out: `risk_per_trade` typo (max-leverage positioning), Calmar using raw vs annualised return, dead params causing -999 Optuna scores, `plateau_summary` display bug.
- Cross-platform path conflict with Dimitris resolved via `os.path.join`.
- XS momentum thread started, three signal variants tested through WF + perturbation (raw / rolling Sharpe / residual alpha). Residual alpha showed strongest perturbation result; composite signal designed but never built. Empirical results overall not promising → likely scrap. Full recap in `notes/xs-momentum-recap.md`.
