---
title: "Polymarket research onboarding — everything we've found (for the returning Midas teammate)"
tags: [handoff, onboarding, polymarket, copytrade, market-making, options-delta, dali, summary]
created: 2026-06-05
audience: "a teammate who built Midas (the copytrade execution bot) and is rejoining cold; assumes strong engineering, light on the latest research state"
status: orientation doc — a single readable pass over everything Polymarket so you don't have to audit the whole Obsidian vault. Source notes remain authoritative for exact numbers; wikilinks point to them.
relationship: >
  Reads across the whole PM brain: [[COWORK]], [[CODEX]], [[TODO]], [[POLYMARKET_BRAIN]], the two strat hubs
  [[strat_market_making]] / [[strat_options_delta]], the Block-K synthesis [[block_k_plain_english_synthesis]],
  the copytrade/data log [[polymarket/research/RESEARCH_FINDINGS|RESEARCH_FINDINGS]], and the latest state map
  [[2026-06-04_state_of_the_arc_and_novelty_frontier]]. Forward-looking (hypothetical) ideas live separately in
  [[2026-06-05_novelty_frontier_map]] — this doc is about what we ACTUALLY found, not what we might try.
---

# Polymarket research onboarding

> **How to read this.** Section 0 is the whole thing in plain English — read it and you're oriented. Section 1 is your domain (copytrade + Midas execution) in depth. Sections 2-4 are the three research strats (market-making, options-delta, dali microstructure). Section 5 is the clean "closed vs survives vs live-now" ledger. Section 6 is a navigation map for the Obsidian vault. Everything below Section 0 is drill-down — skim the headers, dive where you have questions.

---

## 0. Plain English — the whole state in one page

We've been hunting for a durable trading edge on **Polymarket** (the prediction market where shares pay $1 if an event happens, $0 if not — so the price *is* a probability). We ran four parallel threads. Here's where each landed:

1. **copytrade (your thread — Midas).** Find and mirror wallets that are *already* profitable. This is the **standing live track** — the infrastructure (Midas) is built and tested, the smoke target is chosen, and the next step is operational, not research. Most live-ready of everything.
2. **MM — market-making.** Quote around fair value, earn the spread + maker rebate instead of betting on direction. Our own single-venue maker is **closed (it loses)** for a well-understood reason. But real maker wallets *are* profitable, and one specific cell — **politics NegRisk structured makers (+2,290 bps)** — survived every check and is the anchor for a live measurement loop.
3. **OD — options-delta.** A "Will BTC be up over the next 4h?" market literally *is* a digital option, and Polymarket **overprices the longshot side and volatility**. But after a dozen tests, **PM pricing is efficient net-of-cost** across crypto *and* equities — so any pure "we price it better" strategy is **closed**. OD survives only as a sizing/selection layer inside the maker lifecycle.
4. **dali — short-horizon microstructure.** Order-flow / top-of-book signals. The signal is **real but it's mean-reversion to fair value on a wide-spread venue** — not tradeable as a taker or maker. **Closed** across all tiers (rules, ML, deep learning).

**The one-line takeaway:** *Offline Polymarket research is essentially exhausted. PM financial-binary pricing is efficient net-of-cost. The durable edges are **execution and copy**, and the remaining questions can only be answered **live**.*

**Three ideas that explain 90% of our findings, in plain words:**
- **Mark-to-mid is not money.** A position can look profitable "at the mid price," but on a venue with 15-30¢ spreads, actually *closing* it gives most of the gain back. Almost every false positive we hit was this illusion.
- **Adverse selection is structural, not a latency race.** When you quote and get filled, you often get filled *because* the price is about to move against you. On crypto markets we proved you can't dodge this by being faster — it's baked into the spread and the resolution jump.
- **Our method is blind to novelty.** We mine *historical* wallet behavior. That's great at *falsifying* ideas and at *finding copyable winners*, but an edge nobody runs yet leaves no historical trace — so the method reads "no edge" exactly where the best opportunities hide. That's why the frontier is now live + deliberately novel (separate doc: [[2026-06-05_novelty_frontier_map]]).

**Where we are right now:** copytrade smoke deployment is the default next action; the politics-NegRisk maker live loop is the second live track; everything else is either closed or parked behind a live measurement loop.

---

## 1. copytrade — your domain (data, cohorts, Midas, current state)

This is the repo's nominal primary thread and the one closest to live money.

### 1.1 The data foundation (what we own)
From [[polymarket/research/RESEARCH_FINDINGS|RESEARCH_FINDINGS]]:
- **~1.06B fills** in `raw_trades` (913.4M Goldsky + 151.1M warproxxx seed), covering 2022-11-21 → 2026-04-24. **2.62M distinct addresses**, **797k markets** with ≥1 closed position.
- **`closed_positions.parquet`** is the load-bearing artifact: **269,974,929 closed positions**, self-consistency confirmed (Σ realized PnL = $0.00 — every winner's redemption matches a loser's lost stake). **96.4% of positions are held to resolution** (most never close via on-book sells), which is why the redemption-synthesis design exists.
- **22% of fills are on NegRisk markets** — this shapes everything (see phantom score).
- **Goldsky lag ~9 days** + sync gap → any walk-forward must use cutoffs ≥9 days behind run date. We chose Goldsky over running our own Polygon node (cheaper, but no mint/merge/redeem events — a known v1 limitation).

### 1.2 The two metrics that matter for attribution
- **`phantom_position_score`** — the NegRisk-arb detector. `1.0 = clean directional`, `>1.0 = arb-shaped` (holds both YES and NO; cancels at market level). **~20% of traders have score >2**, meaning their *per-position* PnL is unreliable — for them, rank by **market-level (`mkt_*`) metrics only**, never position-level.
- **Operator denylist** — **12 hardcoded addresses** carry ~38% of all fills and must be excluded: 2 pure relayers (e.g. `0x4bfb41…`, 305M fills, $17.34B notional, 0 maker fills, carries $61M of *fake* PnL), 7 MM bots, 3 HFT. Heuristics: maker:taker ratio >50 or <0.02; >500k counterparties; >95% sub-second AND >1M fills.

### 1.3 The relayer/attribution finding (read this — it's pure execution)
[[copytrade_relayer_implications]] — we chased a hypothesized "invisible taker wallet" bug and **closed it: no PnL/position rebuild needed.** The reason matters for Midas:
- Polymarket's CTF Exchange `_matchOrders` emits the active taker-order wallet as **`maker = takerOrder.maker`, `taker = address(this)`** — so **the active wallet is already in the `maker` column.** 414.7M / 416.1M v1 internal-leg rows join the trader panel cleanly.
- Sampled 50k internal rows → `tx.from` is a relay/submission layer (2,051 distinct, 0 join to traders) → **not** trader identity. Confirmed.
- **v1→v2 cutover happened 2026-04-28**: new exchange contracts `0xe111…996b` (standard) and `0xe2222d…0f59` (neg-risk) emit the same pattern (`taker = address(this)`); both added to a now version-aware `EXCHANGE_INTERNAL_LEG_V1/_V2` denylist.
- **Net for you:** maker-filtered copytrade already sees these fills correctly. The only real artifact is **style labels** (a wallet's maker:taker ratio shifts after reclassifying internal legs — e.g. Domah 7.89→5.67, still maker-heavy) — affects style framing, *not* PnL. Add an `active_order_leg` flag before any style-based claim.

### 1.4 Cohorts and the smoke target
Six stratified cohort pools at `data/cohorts/*.parquet` (union: **9,788 unique addresses**, 947 in ≥3 pools, only 6 in 4 pools, 0 in 5+). Per glossary exec-readiness:
- **Smoke target = `0x6a72f61820b2…`** — top of leaderboard ($14.95M lifetime PnL, 4,359 positions, phantom 2.18 = non-arb, NegRisk share 0.05, role_balance 0.84). **Passes all 6 exec-readiness filters → recommended first real-money smoke.** (NOTE: the TODO also flags a re-ranking exercise excluding `split_position_signature >60%`; the leaderboard-top remains the clean default.)
- **Domah `0x9d84ce…`** — patient accumulator, $4.0M `mkt_total_pnl`, **phantom 8.45 (heavy NegRisk arb) → do NOT mirror in v1.** His macro/maker/18-24h cell is intact for a *paper-trade*, but NegRisk position-keying isn't ready in exec v1.
- **Cohort C (negrisk_specialists, 113 wallets) is OFF-LIMITS for exec v1** until multi-outcome `(condition_id, asset_id)` keying + merge/split handling lands.

### 1.5 The Midas execution stack (your code, current state)
From [[block_e_audit]] + glossary (last exec commit 2026-05-18):
- `watcher/leader_watcher.py` — subscribes the **RTDS firehose** (`wss://ws-live-data.polymarket.com`, `activity/trades`), filters on `proxyWallet` == leader, emits `LeaderFillObserved`. RTDS carries `proxyWallet, conditionId, asset, side, size, price, ts, txHash`.
- `signal/classifier.py` — entry/exit classification via journal-rebuilt ledgers → `MirrorSignalEmitted`.
- `mirror/mirror_engine.py` — risk gates + order submission, tracks positions/daily PnL, emits `FillRecorded` / risk halts.
- **Kernel:** vendored `_kernel/` from `midas/executor/`, treated as **frozen**. `PolymarketVenueAdapter` implements the venue Protocol (order lifecycle, idempotency, ambiguous-submit detection). Bot requests **FOK**, kernel exposes **IOC** (equivalent on PM). Two pricing modes: `leader_fill` (mirror exact fill price) vs `current_book` (best ask/bid + slippage).
- **Recovery = journal-replay:** positions, dedup, in-flight orders rebuilt by replaying the JSONL journal on every startup.
- **Caveat:** kernel's `VenueFillEvent` doesn't expose the on-chain hash, so the bot synthesizes `<client_order_id>:fill:<ts_ns>` — PolygonScan cross-ref needs a manual `venue_order_id` lookup.
- This code consumes **one configured leader** and does no wallet clustering.

### 1.6 The path to first real money (current checklist, from TODO § copytrade)
1. PLAN.md sync + snapshot commit + tag (~10 min, engineering-complete marker).
2. **Slack you** about the kernel encoding bug workaround before the executor goes live. *(This is literally a TODO item with your name on it.)*
3. Polymarket creds into `.env` (private key → `derive_api_keys.py` → `.env`; read-only auth check first).
4. Pre-flight smoke target — profile current open positions via Gamma + RTDS pre-subscription; confirm no NegRisk markets.
5. **First real-money smoke** per `polymarket/execution/scripts/SMOKE_REAL.md`: `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, `SIZING_USD=10`.
- Post-smoke hardening: resolution-handler path, tear down the `MAX_REAL_ORDERS`/operator-confirm harness once stable, VPS in a non-blocked region (US East / Frankfurt / Tokyo), `POLYMARKET_LEADER_RANKINGS_PATH` env + read-on-refresh (gates multi-leader).

### 1.7 The Phase-5 backtest design (locked, not yet implemented)
[[phase5_design]] — monthly walk-forward cohort backtest, expanding IS window, OOS = next month, 2024-01 → 2026-05.
- **Three cohorts:** `B_high_pf_with_size`, `BC_directional_negrisk`, `E_patient_accumulators` (all share guards `n_pos>200, active_days>90, mkt_std_pnl>1.0, NOT operator`).
- **Maker-only copy is the default** (taker fills excluded as latency-sensitive); `--include_takers` is a sensitivity mode.
- **Slippage = next-fill model:** assume you fill at the *next other-trader print* in the same market/outcome/direction within `[15s, 300s]`, else a `3¢` fallback. This *is* the live execution policy in simulation form.
- **Latency → fallback mapping (important for Midas):** Goldsky polling 5-15min → fallback ≥5¢; CLOB WebSocket 5-30s → 2-3¢; Polygon event sub 1-5s → 1-2¢; mempool intercept → 0¢. If >40% of fills hit fallback, the cohort is on quiet markets → flag.
- Success bar (default slippage): annualized return >20% on ≥2 of 3 windows, OOS Sharpe >1.0, max DD <30%, >100 OOS signals. *Robust* if it survives a 5¢ fallback with Sharpe >0.7.

### 1.8 copytrade research verdicts you should know
- **Structural directional carriers** ([[copytrade_structural_directional_carriers_findings]], 2026-06-04, the most recent copytrade note): we asked whether the non-politics structural-maker edge (sports/residual/equities) becomes copyable as a *direction* (ignore the maker wrapper). **Verdict: NOT copyable as a taker.** Sports taker-copy −61.8 bps [CI −702, +497], residual −224.2 bps, equities thin/negative. The maker counterfactuals are point-positive but CI-crossing and aren't a copy-execution model. **Keep the audited Domah-style smoke path; don't promote these to smoke targets.**
- **Block B (historical TFI)** ([[block_b_findings]]): fill-only trade-flow-imbalance is a mixed/weak signal — *but* removing operators (esp. relayers) lifts hit rates a lot (crypto 34%→52%, equity-index 47%→59% at 300s). The catch ([[block_e_lite_findings]]): **relayer-category operators can't be filtered live** from RTDS `proxyWallet` alone (only MM-bot/HFT map directly; relayers need post-hoc raw maker/taker checks). So the historical operator-removal lift may not be reproducible in a live CLOB/RTDS pipeline — a real gap to keep in mind.
- **Weather FTC take-profit** ([[weather_ftc_state]]): a separate copytrade-adjacent strat (fade weather markets after a temperature "tail cross"). After fixing bookkeeping bugs (a fictional 2.27M% return → a realistic 628%), the honest read is: **NOT deployable on analysis alone.** Whether it makes money depends entirely on **execution behavior**: "sticky" quoting (post and chase up) loses −400% to −900% ROI/yr in every cell; "track-down" quoting (continuously cancel/repost at best bid) is positive in every cell. The real stick-at-price fill rate is ~11.8%, not the 26% the optimistic model assumed. **Next step is an execution-calibration live test, not a profit test** — and it's the most direct test of Midas's execution quality: measure sticky-vs-track-down fill behavior at `(p_in=0.85, p_out=0.90)`, $10-50/trade, ms-resolution logging, success = ≥50 filled trades, key readout `cancel_replace_latency_p90 < 2s`.

---

## 2. MM — market-making (Block K maker side)

Full plain-English version: [[block_k_plain_english_synthesis]]. Hub: [[strat_market_making]].

**The core equation everything fights over:**
`maker profit = spread captured + rebate − adverse selection − inventory/resolution risk`

**Polymarket fee facts (real, confirmed):** maker fee = **0**; taker fee = `C·feeRate·p·(1−p)` (max at 50¢, →0 at the edges); feeRate by category (Crypto 0.07, Sports 0.03, **Geopolitics 0**). Maker rebate is funded *out of* taker fees: 20% crypto / 25% other fee-enabled / **0% geopolitics** (no taker fee → nothing to rebate). This is why "earn the rebate" is real in crypto and a myth in geopolitics.

### 2.1 What's CLOSED (do not re-run)
Our own single-venue maker is dead across every anchor we tried, all for the same structural-adverse-selection reason:
- **K1** generous economics gate "passes" — but fee-free geopolitics also passes, proving it's just a mark-to-mid spread-capture gate, not a real edge.
- **K2** proper Avellaneda-Stoikov in logit space, optimized, *with a real exit*: **−1,126 bps, CI<0.**
- **K-PEG** chase-maker looked great at **+759 bps** — but our + Codex's audit proved it's **mark-to-mid**; force a realistic exit and it flips to **−753 bps** (the exit half-spread ~1,635 bps is bigger than the whole edge). Passive exit doesn't rescue it (~12% fill, need ~40%). *The entry signal is real and broad-based (79% win, survives dropping the best 5%) — it's the exit that loses.*
- **K2v3** Binance/digital-anchored quoting: 0/681 buckets clear; the anchor *increased* adverse selection (325 vs 145 bps).
- **K2v2** defensive (pull/widen on Binance move): **−4,316 bps; the defense fired on <0.1% of fills** → the toxic fills aren't preceded by a visible Binance move → **adverse selection is structural, not a dodgeable latency race.**

### 2.2 What SURVIVED — K5 and the politics anchor
- **K5 (model-free, real wallets):** real maker-heavy wallets *are* profitable on crypto-4h: **+171 bps, CI [34, 327]** (256 wallets) — the first robust positive in the block. Winners' playbook: **64% two-sided, 78.8% carry-to-resolution, 0.8% in the late near-50¢ spike zone.** **Capacity warning: top-3 wallets capture ~95% of positive profit per market** → winner-take-most. Geopolitics negative (no rebate).
- **K5b:** the moat is **capital/scale + structure, NOT speed** → build in Python/Midas, **defer Rust.**
- **Deployability (K9):** honest median + capacity → only **~$78/active day** standalone, ~90% in one grab-bag cell; crypto cells ≈ $0/day. MM standalone doesn't justify a dedicated bot.
- **politics_negrisk is the proven anchor:** after full NegRisk merge/split/redeem accounting (125,937/125,937 receipts decoded, $0 missing), the structured-non-top3 politics maker cell is **+2,290 bps, CI [1,020, 3,621]**, median wallet 14.5 bps, non-rebate-dependent. Verdict **MERITS-LIVE-MEASUREMENT-LOOP.** Caveat: deployable EV at *honest median* sizing is only **$9-189/day** under the capacity ladder (the big mean numbers are a fat-right-tail artifact). 2026 non-top3 politics-NegRisk flow is $381.1M, active every observed day.
- **2026-06-04 directional decomposition (important reframe):** across sports, residual-misc, equities AND politics, the clean *neutral* (`arb_like`) structured-maker subset is empty/negative — **the historical edge sits in `two_sided_directional` wallets.** Two consequences: (1) the directional pick is the copyable alpha (→ feed to copytrade, but §1.8 showed taker-copy doesn't clear); (2) **nobody runs disciplined neutral MM in slow politics/sports markets**, so whether that works is an *open novelty question, testable only live* — the politics live loop is best understood as a *novel neutral-MM test*, not a reproduce-the-winners test.

### 2.3 MM state now
Shared maker infra is **built and tested** (`polymarket/execution/maker/`: event calendar, NegRisk inventory tracker, resolution/redemption handler, one-condition maker engine, CLI; 256 execution tests green) — measurement-grade, not production-grade (missing true queue position, fill-share, book-depth-at-quote telemetry). Phase-2 deployment is lane-owned: one market, 1 contract, telemetry-heavy, ≥30 settled markets before any scale decision. First-mover liquidity in newly-created markets is scoped as a separate forward-test-only branch (MERITS-LIVE-STAGE-1-CAPTURE).

---

## 3. OD — options-delta (Block K options side)

Hub: [[strat_options_delta]]. Foundation: [[block_k_maker_options_research]].

**The framing:** a `btc-updown-4h-*` contract literally *is* a cash-or-nothing digital option (strike = window-open price, expiry = 4h later). Polymarket **overprices the longshot side and short-dated vol** (K6: +3.7 vol pts avg, **+24 vol pts in the far-from-strike/late-window bucket**, CI clears; K7: cross-category longshot premium). The OD thesis was to *sell the overpriced side and carry to resolution.*

### 3.1 What's CLOSED
- **The pricing thesis is efficient net-of-cost — robustly, across crypto AND equities.** 5+ independent tests: crypto terminal calibration, same-day touch+terminal, SPX close N(z), conditional-probability calibration, pricing-model-form (Merton/Kou jump-diffusion). PM ≈ fair everywhere. **Do not propose a pricing/forecasting strategy.**
- **OD-as-standalone taker is closed three ways:** source-vs-valuation (−40.47c/episode), passive-only survival, pure-taker not clearing the spread. The far/late zero-threshold squeaks to +3.01c [0.02, 6.50] but the median selected edge is only 0.36c — economically hair-thin.
- **Continuous/banded gamma-scalp (K6) is dead on turnover:** the vol *sign* is right (+24 pts) but the best bucket nets **−9.39c, of which 9.56c is hedge turnover** vs a 0.72c unhedged edge. *Turnover is the killer, not the vol estimate.*
- **K3** (cross-venue basis): 4h has no anti-arb fee, Binance leads ~10s, but raw post-fee basis is thin/IS-only. **K4** (intra-PM arb): essentially zero on the owned universe.

### 3.2 What SURVIVES (barely, live-only)
- **Strategy A** = passive maker entry (rich, far/late zone) → **hold to resolution** (never pay PM's exit spread) → **static** external hedge on Binance (continuous hedging is what killed K6). Reframed: OD is the *valuation/signal layer* (what's mispriced, how much directional risk), MM is the *execution/lifecycle layer* (how you get filled). The hedge is the *least* important lever — Phase-2 proved a full hedge cuts variance only ~10% in the far-|z| gold mine (delta is tiny far from strike).
- Verdict after the realism re-audit: **MERITS-LIVE-MEASUREMENT-LOOP at $10-100 scale** for the source-clean/rich-short passive harvest, with **fair-value-scaled sizing the only stress-surviving sleeve (~$0.79/day stress run-rate)**; flat one-contract crosses zero under adverse-regime stress. The bare global gate stays **CONFIRM-CLOSE**.
- **Kronos** (forward-vol foundation model) stays **gated off** until a static-hedge unhedged lifecycle clears OOS — and it must beat dumb baselines (EWMA, HAR-RV) on *trade PnL*, tail-calibrated in the far/late bucket, or it's A1.7 round two. See [[2026-05-31_kronos_hermes_eval]].

---

## 4. dali — short-horizon microstructure lineage (the A blocks)

Hub: [[COWORK]] § dali. The original direct local-microstructure continuation branch is **falsified across all three tiers and both execution modes**, but "dali" is the broader research lineage that fed Block K/MM/OD — not globally closed.

- We captured live order books (A0/A0b/A0c) and replayed OFI/TOB signals. **The strongest signal — `tob_imbalance_level` — hits 73.7% at 5s top decile (real).** But:
  - **Tier 1 taker:** 0/12 cells positive after entering at ask + exiting at bid (−1,300 to −2,150 bps). The mid-mid alpha is real as a *pattern* but un-tradeable as a taker.
  - **Tier 1 maker-at-mid:** the one +554 bps cell (A14c) was an **overlap-math artifact** — non-overlap flips it to −451 bps, fill rate 9.0%→0.2% (A14h).
  - **Tier 2 LightGBM:** no edge; **calibration breaks at high confidence** (model is *less* accurate when most confident, P≥0.70 gap up to −16pp).
  - **Tier 3 (DeepLOB):** foreclosed by the calibration diagnosis — the gap is structural, not architecture-shaped.
- **Diagnosis:** the TOB/OFI signal is **structural mean-reversion to fair value on a wide-spread venue** — a market-structure fact, not a model problem. Passive reversion framing (A18) and cross-market lead-lag (Block I, Binance→PM) also closed negative.
- **Salvageable lessons (preserved):** live sign-convention infra, capture+replay pipeline, TOB as a *state variable* (could feed copytrade leader screens), PM microstructure facts (exchange-internal-leg attribution, depth≠flow, calibration-breaks-at-extremes), and the methodology spine (**non-overlap math by default**, the Briola caveat is real, overlap math is treacherous).

---

## 5. The ledger — closed vs survives vs live-now

**CLOSED (do not relitigate; died on robust grounds):**
- PM financial-binary *pricing* (crypto + equities) — efficient net-of-cost, 5+ tests.
- Single-venue neutral crypto MM (K2/K2v2/K2v3) — structural adverse selection.
- OD-as-standalone taker/valuation — source-vs-valuation, passive-only, pure-taker all fail.
- Continuous/banded gamma-scalp (K6) — hedge turnover.
- Intra-PM arb on owned universe (K4); cross-venue lead-lag taker (Block I).
- dali direct microstructure — all tiers (taker, maker, ML, deep learning).
- Taker-copy of structural directional carriers — doesn't clear net-of-cost.

**SURVIVES / merits a live measurement loop:**
- **copytrade** — the standing live track (Midas + safety harness + smoke target). *Your thread.*
- **politics NegRisk structured maker** (+2,290 bps) — strongest historical cell; novel-neutral-MM live loop.
- **OD Strategy A** — tiny, tail-fragile, fair-value-scaled sizing only, $10-100 scale.
- **First-mover liquidity** in new markets — forward-test-only, Stage-1 capture.

**LIVE / NEXT ACTIONS:**
- copytrade smoke ($10, operator-confirm) — operationally hours away.
- politics-NegRisk maker measurement loop (separate handoff [[2026-06-03_politics_negrisk_live_loop]]).

**The governing discipline (every result is held to this):** non-overlap math, net-of-cost, confidence intervals on every headline, OOS confirmation before any IS positive is called deployable, **never confuse mark-to-mid with realizable PnL**, and **forecast accuracy ≠ net-of-cost profit** (the Briola caveat). Full rules: [[CODEX]] § Realism calibration.

---

## 6. Navigation map (so you can drill in Obsidian)

**Start here:** [[block_k_plain_english_synthesis]] (the no-background-needed Block K explainer + glossary), then [[2026-06-04_state_of_the_arc_and_novelty_frontier]] (the current high-level map).

**Hubs:** [[COWORK]] (orientation + active threads), [[CODEX]] (implementation rules + realism calibration), [[TODO]] (authoritative live task list), [[POLYMARKET_BRAIN]] (Obsidian map), [[glossary]] (every term incl. Midas semantics).

**Your thread (copytrade):** [[polymarket/research/RESEARCH_FINDINGS|RESEARCH_FINDINGS]] (data + trader population), [[phase5_design]] (backtest design), [[copytrade_relayer_implications]] (attribution), [[profile_domah]], [[copytrade_structural_directional_carriers_findings]], [[weather_ftc_state]], [[block_e_audit]] / [[block_e_lite_findings]] (operator detection + exec inventory). Execution code: `polymarket/execution/` (`PLAN.md`, `watcher/`, `signal/`, `mirror/`).

**MM:** hub [[strat_market_making]]; [[block_k5_findings]], [[block_k5b_findings]], [[mm_deployable_cells_findings]], [[mm_politics_negrisk_accounting_findings]], [[mm_structural_maker_directional_decomposition_findings]].

**OD:** hub [[strat_options_delta]]; [[block_k6_vol_findings]], [[od_strategy_a_v2_lifecycle_findings]], [[od_strategy_a_realism_reaudit_findings]], [[od_equities_index_pricing_scope_findings]], [[2026-05-31_kronos_hermes_eval]].

**dali:** [[block_a13_tob_imbalance_findings]] (the 73.7% signal), [[block_a17_lightgbm_findings]] (ML closure), [[block_a14h_maker_non_overlap_findings]] (overlap-artifact autopsy), [[block_i_leadlag_feasibility_findings]].

**Forward-looking ideas (explicitly NOT findings — hypotheses to discuss):** [[2026-06-05_novelty_frontier_map]] + [[2026-06-05_novelty_deep_research]].

---

## 7. Open questions worth discussing (real, not hypothetical novelty)

These are decisions/unknowns the existing research surfaced — good discussion fodder:
1. **Smoke target lock-in.** Mirror `0x6a72f6…` (passes all 6 filters) vs paper-trade Domah's macro/maker/18-24h cell first? The structural-carriers result says *don't* widen to directional carriers yet.
2. **Multi-leader timing.** Exec v1 is single-leader; cohort work + `leader_rankings.parquet` contract exist. When do we wire `POLYMARKET_LEADER_RANKINGS_PATH` + read-on-refresh?
3. **The live relayer-filter gap.** Block B's big operator-removal lift may not survive live because relayers can't be filtered from RTDS `proxyWallet` alone — does this change how much we trust any copy signal?
4. **NegRisk in exec.** Cohort C and Domah are blocked on `(condition_id, asset_id)` keying + merge/split handling. Is that worth building now, given the politics-NegRisk maker edge is the strongest cell we have?
5. **Weather FTC as an execution benchmark.** Run the sticky-vs-track-down calibration test as a way to *measure Midas's execution quality* (cancel-replace latency, queue fills) even if the strategy itself never deploys?
6. **MM vs copy re-merge.** The politics-NegRisk maker loop and copytrade share plumbing; do we run them as one live track or two?

---

> This doc is a snapshot of *what we found*. For *what we might try next* (foundation-model forward-vol, Kalshi macro→crypto-vol, NegRisk-basket consistency, LLM resolution-criteria scanning, etc.), see the sibling notes [[2026-06-05_novelty_frontier_map]] and [[2026-06-05_novelty_deep_research]] — kept separate on purpose so research and speculation don't blur.
