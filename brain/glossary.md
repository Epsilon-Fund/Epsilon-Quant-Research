---
title: "glossary"
created: 2026-06-05
status: closed
owner: justin
project: infra
para: area
hubs:
  - CODEX
  - COWORK
tags:
  - obsidian
  - brain
  - infra
---
# glossary

Terms and acronyms across Epsilon projects.

## general

- **Epsilon Fund Group** — GitHub org, parent.
- **Hyperliquid** — perp DEX, deployment target for crypto strategies.
- **Polymarket** — prediction market, target for the copytrade bot.

## quant infra

- **WF** — walk-forward optimisation.
- **CPCV** — combinatorial purged cross-validation.
- **OOS** — out of sample.
- **TPE** — tree-structured Parzen estimator (Optuna sampler).
- **ADX** — average directional index.
- **Calmar** — annualised return / max drawdown.
- **Plateau analysis** — robustness check on Optuna parameter neighbourhood.
- **DSR** — Deflated Sharpe Ratio: the observed Sharpe minus the Sharpe a search of N trials would find by luck on zero-edge data (the "selection haircut"). See [[OVERFITTING_VALIDATION]].
- **PBO** — Probability of Backtest Overfitting: how often the in-sample-best config lands in the bottom half out-of-sample (~0.5 = selection carries no info; > 0.5 = anti-informative). See [[OVERFITTING_VALIDATION]].
- **CSCV** — Combinatorially Symmetric Cross-Validation: the all-symmetric-block-partitions procedure that produces PBO.
- **Reality Check** — White's (2000) bootstrap test of whether the best-of-search config beats zero after accounting for the search; studentised = Hansen (2005) SPA refinement. See [[OVERFITTING_VALIDATION]].
- **SPA** — Superior Predictive Ability (Hansen 2005): studentised, less conservative successor to White's Reality Check.
- **Selection haircut (SR\*)** — expected maximum Sharpe of an N-trial search on zero-edge data; subtracted from the observed Sharpe to deflate it.
- **Synthetic-null MC** — empirical DSR cross-check: run the whole candidate-config set on block-bootstrapped synthetic markets (drift/fat-tails/correlation preserved, trend structure destroyed) and ask whether the search can manufacture the real max Sharpe. Gate: real > null 95th percentile. See [[OVERFITTING_VALIDATION]].

## copytrade — markets & infra

- **NegRisk** — Polymarket's multi-outcome / categorical market type. Outcomes mutually exclusive; introduces on-chain merge/split events that aren't captured in raw trades, so position-level PnL is conflated for active NegRisk traders. Market-level metrics are robust.
- **CLOB** — central limit order book (Polymarket's order matching layer).
- **Condition ID** — Polymarket identifier for a market resolution condition. One-to-many with `asset_id`s on multi-outcome (NegRisk) markets.
- **Asset ID** — token-level identifier for one outcome of a market. The bot uses `(condition_id, asset_id)` as the position key.
- **Gamma API** — Polymarket markets metadata API (`clob_token_ids`, outcome prices, condition ID, neg_risk flag).
- **Goldsky** — subgraph indexing service used to pull Polymarket trades via GraphQL.
- **RTDS** — Polymarket's real-time data stream WebSocket (`wss://ws-live-data.polymarket.com`); the live trade firehose the bot subscribes to.
- **Proxy wallet** — the smart-contract wallet that actually holds positions; the leader's `proxyWallet` from RTDS is the canonical leader ID. Distinct from the EOA that signs.
- **EOA** — externally owned account; the keypair that signs orders. Different from the proxy wallet that owns positions.
- **Operator addresses** — relayers, MM bots, HFT — 12 deny-listed addresses accounting for ~38% of total fills on Polymarket.
- **Phantom position score** — heuristic flag for NegRisk-arb traders (>>1 → arb-like behaviour, position-level PnL untrustworthy).
- **leader_rankings.parquet** — interface contract from research → execution. Schema in `decisions/0001-leader-rankings-schema.md`.

## copytrade — execution

- **py-clob-client** — official Polymarket CLOB Python client; used inside the vendored kernel.
- **midas / midas kernel** — internal trading framework; bot vendored `_kernel/` from `midas/executor/` (treated as frozen).
- **PolymarketVenueAdapter** — kernel module implementing the venue Protocol (order lifecycle, idempotency, ambiguous-submit detection).
- **FOK / IOC** — fill-or-kill / immediate-or-cancel order types. Bot requests FOK; kernel exposes IOC, which is functionally equivalent with immediate-expiry on Polymarket.
- **Leader fill / current book** — two pricing modes the bot supports. `leader_fill` mirrors the leader's exact fill price; `current_book` uses best ask/bid + slippage.
- **Journal-replay** — bot's universal recovery pattern. State (positions, dedup, in-flight orders) is rebuilt by replaying the JSONL journal on every startup.
- **Synthetic transaction_hash** — bot synthesises `<client_order_id>:fill:<ts_ns>` because the kernel's `VenueFillEvent` doesn't expose the on-chain hash. Cross-ref to PolygonScan needs manual `venue_order_id` lookup.

## copytrade — methodology

- **Cohort copy-trading** — strategy of mirroring a *group* of leaders rather than a single trader (Tatv thesis). Survives leader churn, diversifies skill source.
- **Position-level PnL** — one row per `(address, market_id, outcome_index)`. Suspect on NegRisk-active traders.
- **Market-level PnL** — collapse outcomes within a market. NegRisk-robust; preferred for ranking.
- **Point-in-time bankroll** — bankroll as it was at a given historical date (for honest backtesting and live sizing). Distinct from `est_bankroll_usd_30d_max_approx` which is lifetime peak (descriptive only).
- **Self-consistency check** — sum of realised PnL across all closed positions should equal $0 (winners' payouts = losers' losses, modulo open positions and fees). Holds in v1 data.
- **Sizing math** — `leader_fraction = leader_trade_usd / leader_bankroll`, then `bot_bet = leader_fraction × strategy_capital`, capped by per-trade and total-deployed limits.

## copytrade — cohort pools (data side)

Six pools materialised at `polymarket-copy/data/cohorts/*.parquet`. Detail in `polymarket-copytrade/research/04-cohort-pools.md`.

- **Pool A (`high_sharpe_directional`)** — 3,304. Sharpe-driven, NegRisk-light. Exec v1: usable.
- **Pool B (`high_profit_factor_with_size`)** — 556. PF + size. Exec v1: usable.
- **Pool C (`negrisk_specialists`)** — 113 (smallest). 70%+ NegRisk volume. **Exec v1: off-limits** until multi-outcome position keying.
- **Pool D (`sports_directional_fast`)** — 2,152. Holding < 48hrs, binary markets. Exec v1: latency-sensitive.
- **Pool E (`patient_accumulators`)** — 225. Maker-conviction, 1+ week holds. Exec v1: needs resolution-handler path.
- **Pool F (`high_kelly_edge`)** — 7,703 (largest, too permissive standalone). Use only as multi-pool tiebreaker.

**Cohort overlap structure**: 9,788 unique in 1+ pools; 3,278 in 2+; 947 in 3+; 6 in 4+; 0 in 5+.

## copytrade — known leaders / addresses

- **Domah** (`0x9d84ce…`) — patient-accumulator + high-Kelly-edge cohort member; phantom 8.45 (pure NegRisk arb shape); lifetime peak bankroll $59.07M. Sample profile output at `polymarket/research/notes/copytrade/profile_domah.md`. **Exec: do not mirror in v1** (NegRisk arb).
- **Smoke target** (`0x6a72f61820b2…`) — top-of-leaderboard ($14.95M PnL); A + F pools; phantom 2.18; NegRisk 0.05; role_balance 0.84. Passes all 6 exec-readiness filters. Recommended target for first real-money smoke.
- **Operator `0x4bfb41…`** — pure relayer; 305M fills, $17.34B notional, 1.99M counterparties, 0 maker fills. Carries $61M of "PnL" on our books from accumulated matching flow — not real edge. In the operator deny-list.

## dali — short-horizon ML

- **Dali** — Epsilon's short-horizon OFI/microstructure ML project for Polymarket. Separate from the copytrade bot. Targets taker (directional) and maker (liquidity provision) strategies.
- **OFI (Order Flow Imbalance)** — signal capturing net directional pressure from limit order placements, cancellations, and fills. Core signal from Cont, Kukanov & Stoikov (2014). Explains ~65% of short-term price variance on equity CLOBs; TFI explains only ~32%.
- **TFI (Trade Flow Imbalance)** — fill-only subset of OFI. Weaker signal; TFI is a degraded subset of OFI. Dali TFI baseline: OUTCOME 3 Mixed Results, Block B complete 2026-05-27.
- **CKS** — shorthand for Cont, Kukanov & Stoikov (2014) OFI paper and their feature construction methodology. Dali uses CKS-style OFI computed via maintained book state replay.
- **AS-MM / Avellaneda-Stoikov** — canonical closed-form optimal market-making framework (2008). Starting point for Dali maker strategy. Needs adaptation for Polymarket's bounded [0,1] price range.
- **Block A / Block B / ... / Block J** — Dali's phase 1 research blocks. See `dali/TODO.md` for full structure. Block B (TFI deep-dive) and Block C (historical sign convention) complete as of 2026-05-27. Block A0 (24h live OFI capture) starts 2026-05-28.
- **maker_side** — Polymarket historical fill field. Is the passive maker's token side (`BUY` = maker paid USDC, received outcome token). Token-side aggressor = inverse. Confirmed by historical audit (`historical_to_aggressor()` correct).
- **inverse_maker_side / historical_to_aggressor()** — the confirmed historical token-side aggressor convention: `maker_side=BUY` → aggressor `SELL`; `maker_side=SELL` → aggressor `BUY`.
- **live_to_aggressor()** — live sign normalization helper. Returns `UNKNOWN` by default until 50+ classifiable `last_trade_price` events are captured.
- **Block A0 shortlist** — 12 live-capture markets: 4 geopolitics (fee-free), 4 AI/tech, 2 sports, 1 finance/equity-index, 1 crypto. Locked 2026-05-27. Full 24h capture 2026-05-28.
- **LOBFrame** — open-source LOB forecasting framework (Briola et al. 2024). Reference implementation for OFI feature pipelines.
- **Task 5 / Block F** — Optuna parameter search over rule-based strategies. Deferred: requires 3+ families, 24h+ per family, 200+ `last_trade_price` events, and sign convention resolved.

## tooling

- **DuckDB** — embedded analytical SQL engine; used over Parquet glob, no Postgres needed.
- **uv** — Python packaging / venv manager (Astral).
- **Parquet shard** — partitioned columnar dataset; we use schema-uniform append-only shards.

## macro

- **n8n** — workflow automation tool, hosts the newsletter pipeline.
- **Jekyll** — static site generator behind `epsilon-fund.github.io`.
