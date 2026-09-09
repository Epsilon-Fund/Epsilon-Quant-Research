---
title: "Polymarket Brain Map"
created: 2026-06-05
updated: 2026-08-25
status: active
owner: justin
project: infra
para: area
hubs:
  - COWORK
  - CODEX
tags:
  - obsidian
  - brain
  - infra
---
# Polymarket Brain Map

> Start here for anything Polymarket. As of 2026-08-25 this map has one active project and a set of parked/deprioritised areas. If you are here to work, you almost certainly want the market-making project.

## The one active project: market-making

**Read, in order: [[strat_market_making]] (the hub — current state of both lanes, honest framing, reliability ledger) → [[mm_model]] (the model — fundamentals vs additions).** Those two notes are self-contained and de-jargoned; they are the canon surface. Everything else under `polymarket/research/notes/` carries a status banner (HISTORICAL EVIDENCE / PARKED / DEPRIORITISED) — evidence notes back the canon with details and numbers, parked notes are archaeology.

Operational references for the active project:

- [[mm_vps_capture_setup]] — the live 24/7 order-book capture (rented server → Cloudflare R2 bucket): cloud layout, formats, pull commands, and the two known capture gaps.
- [[mm_engine_build_log]] — the replay engine's build history and component map (historical evidence; the engine lives in `polymarket/research/mm_engine/` + `mm_eval/`).
- [[MM_JOIN2_RUNBOOK]] (`polymarket/execution/maker/`) — operator runbook for the live measurement machinery.
- [[polymarket_data_manifest]] — where all Polymarket data artifacts live; [[polymarket_table_dictionary]] and [[polymarket_csv_output_audit]] — shared table/CSV conventions; [[METRICS_REFERENCE]] — metric formulas.
- [[mm_clob_capture_semantics]] — what the public order-book feed can and cannot prove (it is anonymous: no wallets, no order IDs, no own-queue position).

## Deprioritised (not active, not archived)

- **Copy-trading** (`notes/copytrade/`) — identify skilled wallets and test whether copying them survives execution costs. Engineering was completed to the brink of a first tiny live trade and paused. Its execution/signing infrastructure (`polymarket/execution/` mirror + signer) is **live and shared with market-making**. Pick up only with Justin.
- **News-agent / calibration observatory** (`notes/news_agent/`, `polymarket/research/newsagent/`) — a public forecasting-calibration showcase, not a trading strategy. Shipped; has its own open items owned by Justin.

## Parked (historical record — do not build on)

Every note in these areas carries a parked banner; concepts the active project needs are already explained inline in the canon surface.

- **Earlier market-making eras** — single-venue quoting variants (tested, closed) and wallet-level studies of profitable makers (which motivated the politics focus). In `notes/market_making/` with PARKED banners.
- **Valuation / fair-value overlay** (`notes/options_delta/`) — pricing Polymarket binaries against external fair values; closed standalone. One era's bridge ideas were folded into the old maker work; nothing here is input to the active project.
- **Microstructure signal lineage** (`notes/dali/`) — order-flow research that was falsified for direct trading; its salvageable concepts (e.g. book-imbalance as a state/gate variable) are described where needed in the canon surface. Notes carry both an older audit banner (which graded them as *history*) and the parked banner.
- **Archived strays** — `polymarket/archive/` (a retired pipeline, sports-arb stray, an execution-stack audit snapshot).

## Rules that outlive any project

- Never mark inventory or unrealized P&L at the midpoint; use the executable exit price.
- Read spread/depth from a real captured book, never estimate them.
- Test-data splits keep a market's whole life on one side (see [[mm_model]] for why).
- Every Cowork-authored implementation prompt starts by telling the agent to read [[CODEX]], then [[TODO]], [[COWORK]], this map, and [[strat_market_making]].
