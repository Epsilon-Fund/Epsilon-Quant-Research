---
title: "Epsilon — Master TODO"
created: 2026-06-05
updated: 2026-08-25
status: active
owner: justin
project: infra
para: area
hubs:
  - CODEX
  - COWORK
  - VAULT_MAP
tags:
  - obsidian
  - brain
  - infra
---
# Epsilon — Master TODO

> Single live task list. Rewritten 2026-08-25 for the market-making handoff: **one active thread**; everything else is deprioritised or parked. The full pre-handoff task history (all threads, legacy codenames) is preserved verbatim in [[TODO_ARCHIVE]] — reference only, never work from it.

## ACTIVE — Market-Making on Polymarket

> Canon surface: [[strat_market_making]] (state of both lanes + reliability ledger) → [[mm_model]] (fundamentals vs additions). Read those before touching anything below. Honest framing applies to all of it: the evaluation numbers are preliminary; the settled assets are the data, the engine, the accounting/split methodology, and the measured 160 ms live latency.

**Handoff (in progress, owner Justin + Cowork):**
- [x] Repo synced: all work committed, pushed, merged to main (2026-08-23).
- [x] Brain/vault cleanup: de-jargoned canon surface written; status banners on all historical notes; bootstrap path cleaned (2026-08-25).
- [ ] **Market-making onboarding doc for the new collaborator** — written hand-in-hand with Justin: where we got to, what is built, what is reliable vs not, the paths forward. Concise.
- [ ] Handover email + repo access for the new collaborator.

**Research (the main open path — prime territory for the new collaborator):**
- [ ] **P&L spike anatomy.** Understand when and why account P&L jumps sharply up or down. Current state of knowledge: one dissected episode (violent price move; undefended quoter stacks a large wrong-way position; flow-imbalance rises ~30 min before, post-fill drift confirms during) plus the held-out market map (calm markets earn in the approach weeks, give it back near resolution). Wanted: more episodes dissected, how early the warning is detectable, and whether the approach/endgame boundary can be timed per market. Data: the R2 capture (politics + esports since 2026-06-19).

**Live measurement loop (blocked on instrument redesign, owner Justin first):**
- [ ] **(Justin) Pair-quoting design.** Two-sided quoting without inventory = an order to buy YES + an order to buy NO (buying NO at p ≡ selling YES at 1−p), plus event-market merge/split-to-$1 routes. Rework inventory/cap/costing conventions around the pair. Justin is taking a first pass himself before handing off.
- [ ] **Redesign the measurement instrument:** one persistent resting order (no re-quote churn), on a market chosen for fillability (thin queue beats wide spread), tiny size, hard caps, operator-confirm.
- [ ] **First real fills → calibrate.** Use real fills to fit the fill/queue model and replace the instant-order assumption with the measured 160 ms. This is the gate for trusting any backtest number.
- [ ] Later, once fills exist: re-run the evaluation with calibrated fill + latency models; only then revisit the defended quoter's ¢-numbers.

**Data upkeep (cheap, unowned):**
- [ ] Fix capture universe config: two configured universes (culture, crypto-as-control) never landed in the cloud bucket — only politics + esports exist.
- [ ] Sync the capture-health/gap log into the permanent layout before raw-file expiry deletes it (gap ground-truth is currently being lost).

## DEPRIORITISED — not active, not archived

- **Copy-trading** — paused at the brink of a first $10 live trade; its execution/signing infra is live and shared with market-making. Open items preserved in [[TODO_ARCHIVE]] § copytrade. Resume only with Justin.
- **News-agent / calibration observatory** — shipped; Justin-owned open items (sign-offs, API keys, settlements) in [[TODO_ARCHIVE]] § News-Agent.
- **Crypto live momentum book** — live; three standing items (re-baseline quoted Sharpe, plateau-centre re-optimisation rule, overfitting gate on new searches) in [[TODO_ARCHIVE]] § Crypto.
- **Brain infrastructure / skills lifecycle** — operational, low-touch; remaining items in [[TODO_ARCHIVE]].

## PARKED — historical record only

Earlier market-making eras, the valuation/fair-value overlay thread, and the microstructure signal lineage. All notes carry PARKED banners; their task sections live in [[TODO_ARCHIVE]]. Do not reopen without an explicit decision by Justin.

---

## Data layer — research_v1 (carried over from `alvaro` at the 2026-09-09 merge)

> Status: **built and published** (Steps A–K, 26 Aug → 1 Sep 2026). Live state is in `brain/handoff/STATUS.md`; the plan below is preserved for its still-open items (Step F `book` depth tier, NegRisk population recompute, read-only R2 token). Checkbox state is as of 25 Aug and has not been re-marked — trust STATUS.md over it.

> Full plan: [[DATA_LAYER_PLAN]]. Session context: [[2026-08-25_cowork_data_layer_session]].
> Premise: we hold **67 days** of capture (R2 parquet = 11,985 objects / 71.08 GiB, 2026-06-19 → 2026-08-21) and every published MM result was computed on 11-18 days. Making this data easy to research with is the precondition for re-running the ladder at full sample.

**Storage decision (taken 2026-08-25):** R2 stays, as a distribution channel not a query engine. Queries run against a local mirror; DuckDB over Parquet on disk; no database server. Derived datasets are versioned and immutable (`research/v1/`, then `v2`); Layer-0 raw is never touched.

**Phase A — know what we have**
- [ ] A1 `rclone lsl` manifests of `parquet/`, `raw/`, `research-live-clob/` -> day-by-day coverage calendar per universe, missing hours, volume trend. 30 min, no download.
- [ ] A2 Pull one hour, verify the four tables against `mm_engine/feeds/replay_parquet.py` column by column (incl. the known `bba` drift: ships `spread`, not sizes).
- [ ] A3 Build `catalog.parquet` — one row per (asset_id, date): identity, coverage, activity, microstructure, `capture_gate` verdict, lifecycle. **Key output: how many event groups have a full lifecycle inside the window** — the number that decides whether 67 days can certify what 18 could not.

**Phase B — research dataset**
- [ ] B1 Layout: Hive partitions `universe=/date=`, sorted by `(asset_id, timestamp_ms)`, book exploded to `bid_px_1..10`/`bid_sz_1..10`/`ask_px_1..10`/`ask_sz_1..10`, convenience columns (`mid`, `microprice`, `spread_c`, `imbalance`), zstd, 64-128 MB files.
- [ ] B2 Builder script — idempotent, resumable, one day at a time.
- [ ] B3 Validate: row counts match source; one market spot-checked against raw JSONL; `capture_gate` verdicts unchanged; convenience columns match recomputation.
- [ ] B4 Publish to `r2:epsilon-polymarket-data/research/v1/` with `_manifest.json`. Built once by one person; everyone else downloads the product.
- Rule: convenience columns are for viewing/screening only. The engine always recomputes from raw levels.

**Phase C — loader (`get_data`)**
- [ ] C1 `catalog()` / `load()` / `events()`, config via env var, lazy local cache that fetches missing slices from R2 on demand, `sync()` for bulk, DuckDB-over-S3 as explicit fallback.
- [ ] C2 **Anti-drift test:** `events()` must produce a stream identical to `replay_parquet` on a shared day. It is the ONLY files -> `MarketEvent` conversion in the codebase; engine and research both use it.

**Phase D — viewer**
- [ ] D1 Pure plotting functions (data in, figure out), eight panels on one time axis: price+spread band, trade prints, spread, depth at touch and to 5 levels, volume, book imbalance, post-trade markout, capture-health strip.
- [ ] D2 Small multiples + the screen scatter (spread vs depth / vs adverse selection).
- [ ] D3 **Optional strategy-overlay hook** designed in from the start (quote ladder, fills, inventory, PnL, gate state as background shading) so Gonzalo supplies a data source rather than a second plotting stack.
- [ ] D4 Thin local app over the functions — pick market, pick window, render.

**Phase E — onboarding**
- [ ] E1 README: clone -> one env var -> `catalog()` -> first plot, under an hour.
- [ ] E2 Gonzalo's first work: strategy-debugging views on the D3 overlay — where we quoted, where we filled, inventory, PnL, gate state; "when are we losing on this quoting and does it coincide with price running?"

**Still open**
- [ ] Decide whether to keep capturing — **after A3**, not before. Costs a few euros/month and the stream is irreplaceable.
- [ ] Re-enable the `crypto_control` universe (11 commented-out lines in `universes.yaml`). It is the falsification instrument and it is currently off. `culture_other` was never written into the config at all.
- [ ] Where the loader module lives, and matching the existing crypto `get_data` conventions.
- [ ] Research-tooling API beyond the viewer (screens, sweeps, markout tooling) — its own design session.

---

