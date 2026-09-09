---
title: "Dali — roadmap checklist (May 2026)"
created: 2026-05-27
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: polymarket
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - dali
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** This research lineage is closed/parked. Any concept from it that the active market-making project needs is explained inline in [[strat_market_making]] / [[mm_model]] — do not build on this note.

# Dali — Roadmap Checklist

> Strategy: short-horizon OFI/microstructure ML on Polymarket. Two parallel thesis lines: **taker** (directional, consume liquidity) and **maker** (liquidity provision, Avellaneda-Stoikov framework). Validate signal in Phase 1 before building any execution or ML infrastructure.
>
> Last updated from repo state: 2026-05-27

---

## NOW — Block A0: Live OFI Capture (starts 2026-05-28)

The smoke run is done. 12-market shortlist is locked. Full 24h capture starts tomorrow morning.

**Start command (from `polymarket/research/`):**
```bash
PYTHONPATH=. uv run python scripts/dali_block_a0_prepare.py
PYTHONPATH=. uv run python scripts/dali_block_a0_capture.py \
  --config configs/block_a0_capture.generated.yaml \
  --run-id block_a0_20260528_morning \
  --duration-hours 24
```
Add `caffeinate -dimsu` prefix if running on laptop. Use `tmux` on VPS.

- [ ] **Run 24h Block A0 capture** (2026-05-28 morning start)
- [ ] **Audit capture after 24h** — run `dali_block_a0_capture_audit.py`, write `notes/block_a0_capture_status.md`, check event counts per market
  - Minimum to proceed: enough `last_trade_price` events for sign audit progress + useful counts across ≥ several markets + no catastrophic gaps
  - If thin: extend to 48h with the same `--run-id`
- [ ] **Sign convention — live** — once capture has 50+ classifiable `last_trade_price` trades, run `dali_sign_convention_audit.py` to establish `live_to_aggressor()` normalization
- [ ] **Replay OFI features** — run `dali_clob_replay_features.py` on the capture; compute rolling OFI windows, z-scores, book staleness
- [ ] **Taker analysis** — CKS-style R² and hit-rate-by-magnitude vs. fill-only baseline
- [ ] **Maker analysis** — counterfactual fill quality, adverse selection rate per market
- [ ] **Classify each market**: tradeable / ambiguous / not-tradeable per strategy class
- [ ] Document results in `research/notes/block_a0_results.md`

**12-market shortlist (locked):**
- 4 geopolitics/world-event markets (fee-free)
- 4 AI/tech markets
- 2 sports markets
- 1 finance/equity-index market
- 1 crypto/finance-style market

---

## NOW — Block B follow-on (OUTCOME 3 delivered)

Block B is complete (2026-05-27). Outcome: **Mixed Results Requiring Live Validation**. Use as Block A target prioritisation input, not tradability conclusion.

Key findings to carry forward into Block A design:
- AI/product: inverse-maker-side top-decile has ~50.7% hit rate, positive EV after 1 tick at 300s. Operator-filtered result climbs to 53.1%
- Equity-index: inverse-maker-side after operator removal → 58.8% hit rate at 300s — most promising signal in historical fills
- Crypto: walk-forward test set collapses (sub-50%, negative return) after operator removal
- Sports: no family clears 55% top-decile hit rate
- **Conclusion:** No result establishes live tradability. Block A is the decisive test

- [ ] (low priority) Sports pre-game vs in-game segmentation — local metadata lacks game-start field; would require external data join
- [ ] (low priority) AI/product per-market walk-forward at market level — current per-market results from 4 markets only, too small

---

## NOW — Block C: Sign Convention (historical ✅ resolved; live 🔲 pending)

- [x] **Historical sign convention** — resolved 2026-05-27. `historical_to_aggressor()` confirmed correct across all 4 families. `maker_side=BUY` → aggressor `SELL`; `maker_side=SELL` → aggressor `BUY`. No rerun needed.
- [ ] **Live sign convention** — unresolved. Only 1 `last_trade_price` event captured (unclassifiable). Needs 50+ classifiable trades from Block A0. `live_to_aggressor()` returns `UNKNOWN` by default until threshold met.

---

## DEFERRED — Block D: Backtest Engine Extensions
> **Trigger:** ≥1 strategy class shows tradeable signal in Block A

- [ ] Multi-strategy parallel evaluation
- [ ] Realistic order rejection modeling
- [ ] Per-category fee model integration (`fee = C × 0.03 × p × (1-p)`, 0% geopolitics → 1.8% crypto)
- [ ] Maker/taker classification at execution time
- [ ] Walk-forward validation (purged k-fold + embargo — Lopez de Prado)

---

## DEFERRED — Block E: Wallet / Competition Analysis (parallel research)

- [ ] Cluster historical wallets by behavior (systematic vs discretionary, retail vs sophisticated)
- [ ] Identify dominant wallets per target market
- [ ] Filter Block A shortlist by competition intensity (prefer retail-dominated)
- [ ] Cross-reference copytrade `operator_denylist` — already used in Block B, also relevant here

---

## DEFERRED — Block F: Parameter Search
> **Trigger conditions (all must be true):**
> - 3+ market families captured
> - 24h+ capture per family
> - 200+ combined `last_trade_price` events
> - 50+ classifiable live trades for sign convention (or rule avoids trade-side normalization)
> Current state: far below threshold (1 `last_trade_price` event from smoke)

- [ ] Set up Optuna TPE sampler over rule-based strategies (200-500 trials)
- [ ] Define parameter space (OFI window, threshold, horizon Δ)
- [ ] Train/validation/test split: 60% / 20% / 20% chronological
- [ ] Objective: net-of-cost Sharpe, require ≥100 test-set trades before trusting

---

## PHASE 2 — After Signal Validation (post Block A)

### Block G — Live Deployment Infrastructure
- [ ] Production execution stack with risk gates and circuit breakers
- [ ] Monitoring + alerts (fill rate, P&L, position limits)
- [ ] Tiered architecture: Tier 1 (tick/rules) → Tier 2 (30s model) → Tier 3 (5min LLM) → Tier 4 (hourly retrain)
- [ ] Integration with Midas kernel (shared with polymarket-copytrade)

### Block H — ML Model Training
> **Trigger:** Rule-based baseline shows positive edge first

- [ ] LightGBM regression on OFI feature vector (~30 features)
- [ ] Normalized target: `(price_{t+Δ} - price_t) / realized_vol_t`
- [ ] Huber loss; early stopping on held-out validation
- [ ] Regime conditioning: momentum signal gated by price level (bounded variable)
- [ ] Calibration layer post-training

### Block I — Cross-Platform Arbitrage
- [ ] Lead-lag analysis: Polymarket vs Kalshi on overlapping questions
- [ ] Identify persistent gaps (>3%) above cost threshold
- [ ] Target illiquid Polymarket markets where retail-driven mispricings persist longer

### Block J — Resolution-Criteria Edge Scanning
- [ ] LLM pipeline to read market resolution fine print at scale
- [ ] Flag markets where literal resolution rule diverges from consensus pricing
- [ ] Run as background process (~$5-20/day at Haiku pricing)

---

## Research Gaps (deep-dive queue)

- [ ] **Avellaneda-Stoikov adaptation for bounded prices** — paper assumes Gaussian / unbounded. What adaptations for [0,1] range?
- [ ] **Maker rebate optimization** — 20-25% of taker fees (50% Finance). Optimal quoting under rebate.
- [ ] **Adverse selection in prediction markets** — empirical studies on retail-heavy prediction market flow specifically
- [ ] **Cross-platform price discovery** — Polymarket vs Kalshi vs Manifold lead-lag
- [ ] **LLM-based forecasters SOTA** — Halawi et al. (2024); Schoenegger & Park (2024)
- [ ] **Polymarket manipulation literature** — growing but sparse
- [ ] **Minimum useful OFI sample size** — CKS used months; what's the statistical floor?
- [ ] **Bias-corrected calibration metrics** — beyond Brier score for hold-to-resolution strategies

---

## done

- [x] Production research infrastructure: market universe screening, historical fill TFI baseline, live CLOB capture pipeline
- [x] Replay parser computing CKS-style OFI with maintained book state (`lib/clob_book.py`)
- [x] Sign normalization library (`lib/trade_sign_normalization.py`) — historical convention confirmed, live placeholder in place
- [x] Executable-price backtest engine (`lib/backtest_engine.py`) — no spread double-counting
- [x] **Block B: Historical TFI Deep-Dive** (2026-05-27) — OUTCOME 3 Mixed Results; operator-filtered equity-index is most promising family for Block A targeting
- [x] **Block C: Historical sign convention audit** (2026-05-27) — `historical_to_aggressor()` confirmed correct, all 4 families
- [x] Block A0 smoke capture (2026-05-27) — 12-market shortlist locked, configs generated, capture scripts ready
- [x] Literature synthesis (`research/01-literature-synthesis.md`)
- [x] Factor construction + model architecture (`research/02-factor-construction.md`)
