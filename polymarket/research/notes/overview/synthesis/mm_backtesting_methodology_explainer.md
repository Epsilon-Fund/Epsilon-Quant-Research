---
title: "How Institutional Market-Making Backtesting Works — and What It Means for Us (Anonymous L2, Mixed Market Speeds, Gappy Data)"
created: 2026-06-15
updated: 2026-06-23
status: active
owner: justin
project: polymarket
para: resource
audience: "Justin + Alvaro — to understand MM backtesting fully before we divide the work"
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - market-making
  - backtesting
  - methodology
  - queue-position
  - latency
  - explainer
---

# How Institutional Market-Making Backtesting Works — and What It Means for Us

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · concepts: [[mm_concepts_and_strategy_buildup]] · our data limits: [[mm_clob_capture_semantics]]
> An **understanding** document (deep-research + internal mapping) that now also walks **the engine we actually built** (§6 — written to be audited and defended). Build plan + handoff: [[2026-06-23_mm_engine_phase01_buildplan|mm_engine build plan]].

## Plain-English Summary

- **The institutional MM backtest stands on two legs: queue position and latency.** Get either wrong and the backtest *flatters* — it fabricates fills you'd never get. Frameworks, strategy, and architecture all hang off those two.
- **Queue position is the binding constraint in every market we touch, and we can only ever *model* it.** Our public Polymarket L2 is **anonymous** (aggregated price levels, no order IDs, no maker identity, no queue). On-chain `OrderFilled` data (via Goldsky) is **ground truth for fills** but does **not** recover queue position or our own fill rate — and there is **no public L3/order-level feed** (§1b). So fill rate is a live-only unknown.
- **Latency is NOT one-size-fits-all — it scales with market speed, and our targets span both regimes.** Negligible in **slow** markets (politics, pre-game/outright sports — resolution in days, books move in seconds). **Material** in **fast** markets we also examine: **crypto 4h/daily** (Binance leads the PM; speed arbitrage is documented) and **in-play sports** (prices jump on goals; exchanges engineer delays to protect makers). A latency model is cheap and worth building; its *importance* is market-dependent (§2).
- **The punchline that should drive the plan:** the dominant unknown — our true passive fill rate net of adverse selection — **cannot be backtested from public data.** It can only be calibrated from *our own* live fills. So the 1-contract live loop is **not downstream of the backtest; it is the instrument that makes the backtest trustworthy.** Build the bot and backtester as **one event-driven code path**, run replay and 1-contract-live in parallel, calibrate, then size.
- **What we've built so far (§6).** A strategy-agnostic event-driven engine — *same code path* for replay and live-shadow — with **honest economics** (maker rebate; PnL split into realized / mark-to-mid-unrealized / settled) and a **0%-gap** same-code-path reconciliation. The realism of the *fill rate itself* still rides on the stub queue model until Alvaro's models + Join-2 calibration land.

---

## 0. The one mental model (read this first)

A replay backtest **cannot move the market** — it replays recorded book + trade events and asks "would my resting order have filled?" The answer depends on two things the recording doesn't directly tell you:

1. **Were you near the front of the queue at your price when a trade printed?** (queue position)
2. **Did your quote even exist at that moment, given the delay between seeing data and your order landing?** (latency)

hftbacktest — the reference open-source HFT backtester — says it plainly: *"Without queue modeling, backtests optimistically assume instant fills when your price is reached. This drastically overstates HFT profitability."* ([hftbacktest queue docs](https://www.mintlify.com/nkaz001/hftbacktest/concepts/queue-position)). The two legs interact: latency decides *when* your quote is live relative to the trade tape; the queue model decides *whether* it would have been hit. **Queue position binds in every market; latency binds only in fast ones — and we target both.** Get either wrong and you over-fill.

---

## 1. Fill simulation & queue position — the heart of it

### Why queue position is decisive
A passive maker at the best bid fills only after the size **ahead** of it trades through. If 25 contracts rest ahead and 20 trade, you don't fill. So fill probability is governed by *size-ahead*, not "did price touch my level." Ignoring it is the single largest source of backtest over-optimism — measured: a naive "100% fill against the first market order" assumption produced an **85% fill rate vs a 60%** realistic baseline (DeLise 2024, [arXiv:2407.16527](https://arxiv.org/pdf/2407.16527)); a touch-defined fill of **~50% is <2% realized** once cancellations count (Lokin-Yu 2025, [arXiv:2502.18625](https://arxiv.org/pdf/2502.18625)).

### The models (pessimistic → calibrated)
- **RiskAverse / "trade-through" (pessimistic):** queue advances *only* on trades at your price, never on cancels. hftbacktest's `RiskAverseQueueModel` ([order-fill docs](https://hftbacktest.readthedocs.io/en/latest/order_fill.html)).
- **Probabilistic (`ProbQueueModel`):** each size-decrease is attributed ahead-vs-behind via `f(back)/(f(front)+f(back))`, `f` ∈ {power, log}; higher exponent ⇒ more conservative ([queue models](https://hftbacktest.readthedocs.io/en/v1.8.4/reference/queue_models.html)).
- **L3 FIFO (exact):** simulate the real per-order queue — but needs order-level data **we do not have and cannot get** (§1b).

### Estimating queue from *aggregate* L2 (our situation)
With only price-level sizes, the field standard is the **Rigtorp (2013) algorithm** ([rigtorp.se](https://rigtorp.se/2013/06/08/estimating-order-queue-position.html)) that `ProbQueueModel` implements: assume **back of queue** on entry; advance fully on identifiable trades; attribute ambiguous size-decreases *probabilistically*. The calibrating `f` is *"ideally estimated using data from our own fills"* — **you cannot calibrate the queue model without your own live fills.**

### Theory backing it
Cont-Stoikov-Talreja (2010) Poisson birth-death book → closed-form fill probabilities ([PDF](http://rama.cont.perso.math.cnrs.fr/pdf/CST2010.pdf)); the queue-reactive model (Huang-Lehalle-Rosenbaum 2015) makes intensities state-dependent ([arXiv:1312.0563](https://arxiv.org/pdf/1312.0563)); Moallemi-Yuan (2016) show adverse-selection cost **rises with queue depth** ([paper](https://moallemi.com/ciamac/papers/queue-value-2016.pdf)).

### The realism kicker — adverse selection / "negative drift"
Fills aren't random: a resting bid tends to fill exactly when price is about to fall. DeLise measures **−0.45 to −0.48 tick** post-fill drift on Treasury futures — roughly the half-tick a taker pays anyway, so the passive "edge" is largely illusory unless you select fills well ([arXiv:2407.16527](https://arxiv.org/pdf/2407.16527)). This is Glosten-Milgrom made empirical.

### 1b. Can on-chain fills (Goldsky) or L3 rescue the queue problem? — **No.**
This directly answers "can we cross-reference our L2 with Goldsky `OrderFilled` data, or is that just another approximation?"

**Polymarket is a hybrid CLOB: off-chain matching (operator-run, price-time/FIFO), on-chain settlement** via the CTFExchange / NegRiskCtfExchange on Polygon ([Polymarket docs](https://docs.polymarket.com/trading/overview)). The consequence is decisive for us:

- **Only matched FILLS settle on-chain.** The `OrderFilled` event carries `orderHash, maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled, fee` — and *nothing about the resting book*: no limit price, no resting size, no arrival time, no queue rank ([ITrading.sol](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/interfaces/ITrading.sol)). **Resting orders and ordinary cancellations are off-chain book mutations that never touch the chain** ([cancel docs](https://docs.polymarket.com/trading/orders/cancel)). So posted-and-pulled liquidity — the dominant behavior of real makers — is *invisible* on-chain.
- **Goldsky just indexes those same on-chain logs**, so it inherits every limit ([Goldsky Polymarket dataset](https://goldsky.com/blog/polymarket-dataset)).
- **No public L3/MBO.** The public `market` channel is explicitly "level 2 price data" (aggregated levels, a per-update checksum `hash`, no order IDs) ([market channel](https://docs.polymarket.com/market-data/websocket/market-channel)); order-level PLACEMENT/UPDATE/CANCELLATION exists only on the authenticated `user` channel **for your own orders** ([user channel](https://docs.polymarket.com/market-data/websocket/user-channel)). Matching *is* FIFO, but the per-order arrival times needed to compute your rank are private to the operator ("no L3" = "no *public* L3").

**So what cross-referencing genuinely buys us (it's not nothing):**
- **Ground truth for fills:** disambiguate an L2 depletion as *fill* (matches an on-chain `OrderFilled`) vs *cancel* (no match) — tightening the "clean vs ambiguous" reconstruction the capture-semantics note already requires (alignment is approximate due to WS-vs-chain clock skew).
- **Wallet attribution + others' realized markouts:** we know *who* filled at *what* price and *when*, so we can measure the adverse selection of makers who actually traded. This is exactly what K5 / the relayer attribution already use.

**What it cannot do (and never will, from public data):** give our *own* forward queue position or fill rate; reveal the resting book; or measure competitors' true cancel/fill ratio (on-chain has a survivorship bias — only liquidity that *traded*).

> **MAP TO US.** Cross-referencing Goldsky is **optional, not required** — for **queue + latency** backtesting the aggregate L2 is enough (the queue models are *designed* to infer fill-vs-cancel from size changes), so we are **not** using on-chain fills for now. It could later sharpen fill/cancel labeling, but it does **not** solve the queue problem; queue position and our fill rate remain **live-only**. Today: run **optimistic + pessimistic queue bounds** and report the range, calibrating `f` only once we have our own fills. Our existing tools: `scripts/od_v4_queue_replay.py` (queue-aware replay), `scripts/dali_clob_replay_features.py` (L2 state-builder).

---

## 2. Latency — market-dependent, not "barely matters" (correcting the earlier framing)

### Taxonomy & measurement
Firms decompose **tick-to-trade** into **feed**, **order-entry**, and **order-response** latency, measured with hardware NIC/switch timestamping and taps, not software clocks ([Databento](https://databento.com/microstructure/tick-to-trade)). Colocation is the physical lever (≈287 µs one-way fiber floor to CME's engine; ~4.9 µs/km — [Databento CME](https://databento.com/blog/cme-colocation)).

### How backtests model it
hftbacktest models feed + order-entry + order-response separately, with a ladder: `ConstantLatency` → feed-derived → `IntpOrderLatency` (interpolate *measured* round-trips, collected by "submitting unexecutable orders regularly") ([latency models](https://hftbacktest.readthedocs.io/en/v1.8.4/latency_models.html)). The same strategy on the same data swung **Sharpe −0.20 → +1.54 → −0.38** purely by changing the latency model ([tutorial](https://hftbacktest.readthedocs.io/en/latest/tutorials/Impact%20of%20Order%20Latency.html)). Mechanism: a replay can't move the market, so mis-set latency fabricates fills you couldn't have reached.

### How much it matters depends on market speed — and our targets span both
Latency/staleness bite in proportion to **information-arrival rate** (how fast quotes go stale), which decides who earns the spread via adverse selection.

| our market | speed | does latency/queue bite? | evidence |
|---|---|---|---|
| **Politics NegRisk** | slow (days–weeks; seconds-scale book) | **Latency ~no; queue yes.** A coarse constant round-trip is fine. | wide persistent spreads, edge from pricing not speed (Dubach 2026, [arXiv:2604.24366](https://arxiv.org/html/2604.24366v1), *single-author preprint*) |
| **Sports — pre-game / outright / futures** | slow (updates minutes–days) | **Latency ~no.** Pickoff is model/repricing lag, not wire latency. | futures among least efficient; stale prices repriced hours later ([Twenty First Group](https://www.twentyfirstgroup.com/the-future-of-futures-solving-the-outright-market-in-sports-betting/)) |
| **Sports — IN-PLAY / live** | **fast** (jumps on goals/scores) | **Latency YES.** Stale-quote window seconds-to-~5min after surprise goals; exchanges engineer a 2–8s bet delay to protect makers; courtsiding is physical-latency arb. | Croxson-Reade [EJ 2014](https://academic.oup.com/ej/article/124/575/62/5076978); Angelini et al. [2022](https://centaur.reading.ac.uk/98329/1/information_efficiency_angelini_de_angelis_singleton.pdf); [Betfair in-play delay](https://caanberry.com/betfair-exchange-in-play-delay-explained/); [courtsiding](https://en.wikipedia.org/wiki/Courtsiding) |
| **Crypto 4h / daily up-down** | **fast** (a CEX leads) | **Latency YES, strongly.** Binance leads the PM; your resting quote is stale relative to Binance and gets sniped. | ~$40M extracted from Polymarket via *speed not forecasting* (Saguillo et al., AFT 2025 [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)); our own K3 (Binance leads PM ~10s) and K5-STRESS crypto_4h **−1,886 bps** adverse selection |

The unifying mechanism: continuous order books mechanically let public news snipe resting quotes — a built-in feature, with the liquidity cost scaling to event frequency (Budish-Cramton-Shim, [QJE 2015](https://academic.oup.com/qje/article/130/4/1547/1916146)).

> **MAP TO US (corrected).** My earlier "latency barely matters for us" was true **only for the slowest target (politics/pre-game)** and wrong as a blanket claim — we also examine crypto 4h/daily and (in-play) sports, where latency-driven adverse selection is real and is *exactly* what killed single-venue crypto MM (K2 family). So:
> - **Build a latency model — it is cheap and easy** (a constant round-trip now; measure our *own* round-trip live by submitting a few unexecutable orders — the Phase-2 protocol is specced in [[mm_latency_measurement_spec]], feeding `ConstantLatency`/`SampledLatency` in `mm_engine/latency_models.py`). It is part of the shared engine and pays off as we touch faster markets. I was too dismissive: constructing it is low-cost and worth it.
> - **Its importance is market-stratified:** a minor refinement for politics; **essential** for crypto/in-play sports.
> - **But latency modeling does not solve the bottleneck.** Even a perfect latency model still leaves queue position / fill rate unknown (§1b). Queue is the binding unknown everywhere; latency is an *additional* binding unknown in the fast markets.

---

## 3. Frameworks & the realism pitfalls

### hftbacktest (the repo you saw)
Rust core + Numba/Python; **full L2/L3 reconstruction from feed + trade events**, tick-by-tick, with pluggable **queue** and **latency** models; the same code can run live ([repo](https://github.com/nkaz001/hftbacktest)). It assumes **zero market impact** and its docs warn you **must reconcile against live** ([order fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html)). It's built for **CEX crypto (Binance/Bybit) and lit venues** — so for us it's a **methodology + component source to port, not a drop-in** (no Polymarket adapter; and it expects richer feeds than Polymarket's anonymous L2).

**Does it apply to our crypto 4h/daily markets?** Its *methods* apply **most** there — crypto is its native fast/latency-sensitive domain, and the queue+latency models matter precisely because adverse selection is severe. **But two caveats keep it from being a green light:** (1) the **data limit is identical** — Polymarket crypto markets still give us anonymous L2, not the order-level feeds hftbacktest ideally wants; and (2) we **already closed single-venue crypto neutral MM** (K2/K2v2/K2v3, −1,126 to −4,316 bps, structural adverse selection from the Binance lead). So hftbacktest-grade crypto backtesting would more rigorously **confirm that close** (or serve an OD/hedged variant that hedges the Binance leg) — not reopen neutral crypto MM. Use it where latency/queue realism is the question; don't use it to relitigate a robust close.

### Event-driven L2 replay vs bar/OHLC
Bar/OHLC can't adjudicate a passive fill (intrabar path unknown — [QuantStart](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II/)); you need event-driven tick/L2 replay, which we already do (`dali_clob_replay_features.py`).

### The pitfall checklist

| pitfall | what it is | our exposure |
|---|---|---|
| **Fill optimism** | assuming a resting order fills on touch | high — anonymous L2 forces a queue *assumption* |
| **Ignoring queue position** | not modeling size-ahead | high — the central risk, all markets |
| **Ignoring latency** | filling at prices you couldn't reach in time | **low for politics; high for crypto/in-play sports** |
| **Own-order market impact** | replay can't see your size moving others | medium — thin politics depth; our quote may be the touch |
| **Adverse selection / negative drift** | fills coincide with adverse moves | high — but we measure it (markout) |
| **Lookahead** | using future info | event-driven replay prevents it structurally |
| **Overfitting** | too many configs on too little data | we own the harness (Deflated Sharpe/CPCV; Min Backtest Length 5yr ⇒ ≤45 configs — [Bailey-LdP](https://www.ams.org/notices/201405/rnoti-p458.pdf)) |

A subtle one: simulating price path and order flow **independently** inflates short-term strategies; the fix fills your order only when a *real trade* executes behind your modeled queue ([arXiv:2409.12721](https://arxiv.org/pdf/2409.12721)). Our replay aligns to real trade prints — keep it that way. Two extra pitfalls specific to us: **data gaps** (laptop sleeps → book stale after a gap; only clean, gap-free intervals are valid for fill sim — enforce the ≤5s-staleness rule) and **anonymous L2** (no queue truth).

### Validation
Reconcile the backtest against **shadow/live on the same dates**, tuning the two error sources (latency, queue) until fills/position/equity align ([hftbacktest debugging](https://hftbacktest.readthedocs.io/en/latest/debugging_backtesting_and_live_discrepancies.html)); benchmark vs a **naive symmetric quoter** (the original A-S A/B — [manuscript](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)); start small.

---

## 4. Strategy design as a result (why our slow-market thesis is the right shape)

- **Avellaneda-Stoikov (2008):** reservation price `r = s − qγσ²(T−t)`; optimal spread `= γσ²(T−t) + (2/γ)ln(1+γ/k)`. **Models inventory risk, explicitly NOT adverse selection** ([manuscript](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)). *(Liquidity term is `(2/γ)`, not the `(2/k)` some blogs misquote — verified.)*
- **Guéant-Lehalle-Fernandez-Tapia (2011):** closed-form A-S with inventory limits, "used in practice by major banks for illiquid quote-driven markets" ([arXiv:1105.3115](https://arxiv.org/abs/1105.3115)).
- **Glosten-Milgrom (1985):** a spread exists from **adverse selection alone**, widening with informed flow — and if informed flow is high enough, **the market breaks** (you need ample uninformed flow) ([JFE 1985](https://www.sciencedirect.com/science/article/pii/0304405X85900443)).
- **The speed fork:** continuous books are a **winner-take-all speed race** a non-colocated player loses by construction (Budish-Cramton-Shim, [QJE 2015](https://academic.oup.com/qje/article/130/4/1547/1916146)); slow makers respond by **widening quotes** to avoid being sniped (CFA, [link](https://blogs.cfainstitute.org/marketintegrity/2014/12/18/hft-price-improvement-adverse-selection-an-expensive-way-to-get-tighter-spreads/)). The small-player remedy: **quote the slower/wider/less-contested venue and control inventory** (Hummingbot XEMM — [docs](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)).

> **MAP TO US.** Top-tier-sourced validation of [[mm_concepts_and_strategy_buildup]] Layers 1–6. A-S is what K2 tested, and A-S ignores adverse selection — which is *why* K2 died in **crypto/fast** markets where the Binance lead makes adverse selection (−1,886 bps) dominate. The correct venue for a non-colocated player is the **slow, wide, low-contention** one — politics — quoting wide, skewing on inventory, keeping ample uninformed (retail) flow, and **avoiding news/event windows** (where informed flow and latency-sniping spike). For crypto/in-play sports the same models say: don't quote naked; if you go there at all, it's the hedged/OD variant, not neutral MM.

---

## 5. Build the bot now, or after the backtest? (the "same code path" answer)

**Institutional consensus, two parts:**
1. **Same strategy code runs in backtest and live** — swap only the data feed and execution adapter. Asserted independently by hftbacktest ("deploy a live bot using the same algorithm code"), NautilusTrader ("backtest-to-live parity… no code changes," deterministic clock), Hummingbot (strategy/connector split via a central `Clock`), and QuantStart ([event-driven](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/), [Nautilus](https://github.com/nautechsystems/nautilus_trader)). The event loop kills lookahead structurally.
2. **Sequencing is contested — but MM is the exception.** Low-frequency = edge-first (validate alpha before heavy infra); **HFT/MM = build execution-realism infra (latency + queue) up front / alongside research**, because the backtest is meaningless without it ([exegy](https://www.exegy.com/alpha-decay/), [hftbacktest](https://hftbacktest.readthedocs.io/en/latest/)).

**Building blocks of a MM bot:** market-data handler → **order-book builder** → **quoting/signal logic** → **order manager** (place/cancel/replace, *idempotent*, rate-limit throttled — exchange APIs are unreliable, so track orders *before* submit and *query before resubmit*) → **inventory/position tracker** → **risk limits + kill-switch** (daily-loss pause, drawdown flatten, *reconciliation-mismatch halt*) → **journal/telemetry** → **startup + continuous reconciliation** ([Hummingbot arch](https://hummingbot.org/blog/hummingbot-architecture---part-1/), [Nautilus live](https://nautilustrader.io/docs/latest/concepts/live/)).

> **MAP TO US — the decisive point.** Because the dominant unknown (queue/fill rate) is calibratable *only* from our own live fills (§1, §1b), **the bot is not downstream of the backtest — it is the calibration instrument.** So: **one event-driven engine** with shared quoting code; run **replay (optimistic+pessimistic queue bounds) and 1-contract live in parallel**; use live fills to **calibrate the queue model and the latency constant**; then trust the backtest enough to size. **We built exactly this — see §6.** The shared backtest↔live engine, raw telemetry, and the 0%-gap reconciliation now exist in `polymarket/research/mm_engine/`; the live execution path bridges to the existing `maker_engine.py` / `negrisk_inventory.py` / `resolution_handler.py` safety+signing (256 tests, [[mm_maker_infra_audit_findings]]) at Join 2. What remains is **Alvaro's real queue/latency models** (the stub is optimistic) and **Join-2 live calibration**.

---

## 6. What we actually built — the `mm_engine`, end to end (built to be audited and defended)

This is the concrete realization of §5: **one strategy-agnostic, event-driven engine** in `polymarket/research/mm_engine/`. The *same* `run_engine` function drives both replay and live-shadow — only the feed differs — which is the same-code-path that makes a backtest trustworthy. Strategy, queue model, and latency model are pluggable behind the frozen `interfaces.py`; today they're a placeholder **symmetric quoter** plus stub models (Alvaro's real ones drop in at Join 1).

### 6.1 The loop — four steps per event

`engine.py:run_engine` walks the event stream and, for each event, does four things:

1. **Event in.** `feeds/replay.py` (captured data) or `feeds/live_shadow.py` (live WS) emit *identical* `MarketEvent`s. A `GapMarker` (capture gap / disconnect) marks the book **stale** and **cancels all resting orders** — what a real disconnect does to your quotes.
2. **Book updates** (`book.py:BookTracker`). A `book` event is a full snapshot (re-anchor); **every `price_change` mutates the running book** (level size changed / added / removed); `best_bid_ask` mutates nothing — it's used only as a **checksum** against our reconstructed top. The tracker maintains top-N, flags `stale`, and cross-checks reconstructed L1 vs native `best_bid_ask`. *Reconstruction is exact only on complete, unambiguous intervals* — Alvaro's audit came back **≈100% clean on decisive checkpoints** (CI lower ~99%); the only "mismatches" are sub-millisecond same-timestamp bursts (flagged *ambiguous*, not errors), and a capture gap marks the book stale so we never trust reconstruction across one. Periodic snapshots re-sync any drift; the native L1 catches divergence.
3. **Fills first** (`fills.py:FillSimulator`). On a *real* `last_trade`, only orders placed on **earlier** events can fill (no lookahead). Two gates: **latency** (your quote must have had time to land) then **queue** (`QueueModel.fill → FillResult(qty, queue_ahead)`). Backtest realizes fills; live-shadow logs only.
4. **Mark equity, then re-quote.** The strategy proposes quotes; `orders.py:OrderManager` reconciles them against what's resting — place / cancel / replace / throttle / idempotent no-op, with **deterministic client IDs** (reproducible replay); then a `get_queue_ahead` snapshot is logged per resting order.

Why this is *correct*, not just functional: fills-before-requote = **lookahead-safe**; deterministic IDs = **reproducible**; gap→cancel-all = **realistic disconnect**; reconstructed-vs-native L1 = **self-auditing book**. Everything is written raw by `telemetry.py` (fill log, order log, per-quote queue snapshots) so the analysis layer computes metrics downstream.

### 6.2 The economics layer — what makes the PnL *honest*

**How PnL is actually computed (read this first — it's not a "spread" formula).** PnL is **derived from the simulated fills**: the queue × latency model decides which of our resting quotes fill against the real trade tape, and PnL falls out as **realized** (offsetting fills) + **settled** (resolution) + **rebate**. The engine does *not* compute "spread × round-trips." "Spread capture" is only the *label* for the realized PnL you get *when a round trip actually completes* (you filled the bid **and** later the ask) — and that isn't guaranteed: one-sided fills are common, and then there's no spread captured, just inventory that settles later. The maker equation `profit = spread + rebate − adverse selection − inventory/resolution risk` is a **practitioner PnL-attribution heuristic** — gross capture (spread + rebate) minus two cost channels — *not* the engine's calculation, and its "adverse selection / inventory" terms are **emergent from the fills, and only as accurate as the (currently optimistic) queue + latency model.** *Attribution (be precise — it is **not** "the GM/AS decomposition," which a microstructure-literate reader will catch):* the additive spread decomposition traces to the empirical microstructure literature — Stoll (1978), Ho-Stoll (1981), Glosten-Harris (1988), Madhavan-Smidt (1991), **Huang-Stoll (1997)**; Glosten-Milgrom is the microfoundation for the *adverse-selection* term and Avellaneda-Stoikov for the *inventory* term; the rebate is a maker-taker institutional add-on, in neither. (Our foundational note [[block_k_plain_english_synthesis]] correctly attributes only adverse selection to GM.) This layer makes that fill-derived PnL **honest**:

- **Fee + maker rebate (`fees.py`).** We quote passively → **0 maker fee**, and *earn* `rebate = rebate_rate · fee_rate · qty · p·(1−p)` (taker fee peaks at 50¢; crypto `0.07` → 1.75¢, exactly what K3 measured). The schedule resolves **per-market `fee` field → canonical `FEE_BY_CATEGORY`** (one source of truth, lazy-imported — no duplicated numbers) **→ `fee_free`** bound. The taker-fee path is wired for a future crossing leg but unused (passive-only).
- **Three-way PnL.** `gross` / `net_ex_rebate` / `net_with_rebate` — the K5-STRESS `net_without_rebate` discipline: because the rebate is *policy-fragile* (PM can change it; historically charged 0 on many markets), a rebate-only "edge" must be visible. `net_ex_rebate` is the conservative read.
- **Realized / unrealized / settled — the mark-to-mid lesson, made measurable.** An average-cost ledger (`_Pos`/`_apply_fill`) splits PnL: offsetting round-trips → **realized** (spread actually banked); open inventory marked-to-mid → **unrealized**, flagged as *paper*; `EngineResult.settle(resolution_map)` carries open inventory to the **$1/$0 payoff → settled** (matching our prior research's `position·(payoff − entry) + rebate`, *not* mark-to-mid). The worked example says it all: **±$53 / −$47 settled vs +$1 mark-to-mid** — the K-PEG inflation trap, now a number instead of an assumption. This is how we *measure* whether resolution matters rather than guess it.
- **Record → replay — the same-code-path proof.** The replayer is a deterministic *"fake venue"*: it re-emits recorded events as if they were happening live. `feeds/live_shadow.py:record_to` has a live session record its own event frames; `reconcile.py:reconcile_against_recording` then replays that exact shard and checks the engine made the *identical* decisions → **0% gap**. *Why it matters:* it guarantees the backtest runs the **exact logic that runs live** — there's no "the live bot does something the backtest didn't." (It proves adapter-parity + determinism, **not** real-fill realism — that's Join 2.)

### 6.3 Honest now vs still-on-the-stub (how to defend it)

- **Honest now:** rebate modeled from the canonical schedule; realizable money separated from mark-to-mid paper; settlement matches our own methodology; same-code-path proven at 0% gap; book reconstruction 100% clean.
- **Still on the stub (by design):** fills use the **optimistic** stub queue model, so the *fill rate* — and therefore how much PnL lands in realized vs settled — is optimistic until Alvaro's `RiskAverse`/`ProbQueue` models and **Join-2 live calibration**; `settle()` needs the resolution map wired for a real run; the taker-fee path is unexercised.
- **The defense, in one line:** we never present a single flattering number — every result is **bracketed** (optimistic vs pessimistic queue), reported **with and without rebate**, and **split realized / unrealized / settled**, with the explicit statement that the true fill rate is calibrated *live* (§8). That is exactly what an honest MM backtest looks like.

### 6.4 Module map (file → role)

| file | role |
|---|---|
| `interfaces.py` | the frozen contract: `MarketEvent`, `BookState`, `Order`, `FillResult` + `QueueModel`/`LatencyModel`/`Strategy` protocols |
| `events.py` | event + `GapMarker` types; envelope→events normalization (shared by both feeds) |
| `feeds/replay.py` · `feeds/replay_parquet.py` · `feeds/live_shadow.py` | the feed adapters — JSONL replay, **Parquet replay** (equivalence-tested byte-identical to JSONL), and live-shadow (+ `record_to` for record→replay) |
| `book.py` | `BookTracker` — top-N reconstruction, staleness/gaps, L1 cross-check |
| `strategies.py` | `SymmetricQuoter` placeholder (the A/B baseline) |
| `orders.py` | `OrderManager` — place/cancel/replace, idempotent, throttle, deterministic IDs |
| `queue_models.py` · `latency_models.py` | `OptimisticQueue` / `ConstantLatency` stubs → **Alvaro's lane** |
| `fills.py` | `FillSimulator` — latency gate × queue gate against the real trade tape |
| `fees.py` | `FeeModel` / `FeeSchedule` — per-market → category → fee_free; rebate + taker-fee |
| `engine.py` | `run_engine` + `EngineResult` (3-way PnL, realized/unrealized) + `settle()` |
| `reconcile.py` | the Join-1 artifact: `reconcile_against_recording`, gap vs tolerance |
| `telemetry.py` | raw append-only fill / order / quote logs |

---

## 7. The mapping table (institutional practice → us)

| institutional practice | transfers to our setting? | what we do |
|---|---|---|
| Queue-position fill model | **Yes, central — but we can only estimate** (no order IDs, no public L3) | Rigtorp/ProbQueue with **optimistic+pessimistic bounds**; calibrate `f` from live fills |
| On-chain fills (Goldsky) cross-ref | **Optional — not needed for queue+latency** | anonymous L2 suffices; skip for now, revisit only to sharpen fill-vs-cancel labeling |
| L3 / order-by-order | **Not available** (no public L3; book is off-chain operator-only) | model the queue; never assume order-level truth |
| Latency model | **Market-dependent** — minor for politics, **essential for crypto/in-play sports** | build a cheap constant model now; measure our own round-trip live; don't dismiss it |
| Full L2 event replay | **Yes** | already have it; enforce **gap/staleness ≤5s** — clean intervals only |
| Zero-market-impact assumption | **Watch it** — politics depth thin; our quote may be the touch | flag thin-book markets; verify live |
| Adverse selection / negative drift | **Yes, load-bearing** | measure markout (full-pop now; *our* fills live) |
| Overfitting controls (DSR/CPCV) | **Yes** | reuse the momentum-audit harness; few configs |
| Same-code-path backtest↔live | **Yes** | one event-driven engine, shared quoting code |
| hftbacktest as a tool | **Methodology/component source, not drop-in** | port its queue+latency models; applies *most* to crypto, but crypto MM is already closed |

---

## 8. The hard constraint to internalize *before* we plan

**Queue position and our true fill rate cannot be backtested from public data** — not from anonymous L2, not from Goldsky on-chain fills, and there is no public L3. In the **fast** markets (crypto 4h/daily, in-play sports) **latency is a second live-only unknown** on top. So the deliverable of our backtesting effort is **not** "is the strategy profitable?" It is:

> *"Under what queue/fill-rate (and, in fast markets, latency) assumptions does it break even, and what is the breakeven fill rate?"* — and the **1-contract live loop measures whether reality clears that bar.**

Every backtest number is conditional on a queue (and latency) assumption; the live loop is what makes it real. That means **backtest research and the live measurement loop are one feedback loop, not two phases** — which is the single most important thing to agree on before splitting the work.

---

## Sources

**Polymarket data architecture (Goldsky/OrderFilled/L3):** [Polymarket trading overview](https://docs.polymarket.com/trading/overview) · [market channel (L2)](https://docs.polymarket.com/market-data/websocket/market-channel) · [user channel (own orders)](https://docs.polymarket.com/market-data/websocket/user-channel) · [cancel docs](https://docs.polymarket.com/trading/orders/cancel) · [ITrading.sol (OrderFilled)](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/interfaces/ITrading.sol) · [OrderStructs.sol](https://github.com/Polymarket/ctf-exchange/blob/main/src/exchange/libraries/OrderStructs.sol) · [Goldsky Polymarket dataset](https://goldsky.com/blog/polymarket-dataset) · [polymarket-subgraph](https://github.com/Polymarket/polymarket-subgraph)

**Queue / fill simulation:** [hftbacktest queue concepts](https://www.mintlify.com/nkaz001/hftbacktest/concepts/queue-position) · [order fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html) · [queue models](https://hftbacktest.readthedocs.io/en/v1.8.4/reference/queue_models.html) · [Rigtorp](https://rigtorp.se/2013/06/08/estimating-order-queue-position.html) · [Cont-Stoikov-Talreja](http://rama.cont.perso.math.cnrs.fr/pdf/CST2010.pdf) · [Huang-Lehalle-Rosenbaum](https://arxiv.org/pdf/1312.0563) · [Moallemi-Yuan](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) · [DeLise negative drift](https://arxiv.org/pdf/2407.16527) · [Lokin-Yu](https://arxiv.org/pdf/2502.18625)

**Latency (general + sports + crypto):** [Databento tick-to-trade](https://databento.com/microstructure/tick-to-trade) · [CME colocation](https://databento.com/blog/cme-colocation) · [hftbacktest latency models](https://hftbacktest.readthedocs.io/en/v1.8.4/latency_models.html) · [Impact of Order Latency](https://hftbacktest.readthedocs.io/en/latest/tutorials/Impact%20of%20Order%20Latency.html) · [Aquilina-Budish-O'Neill QJE 2022](https://academic.oup.com/qje/article/137/1/493/6368348) · [Budish-Cramton-Shim QJE 2015](https://academic.oup.com/qje/article/130/4/1547/1916146) · Sports: [Croxson-Reade EJ 2014](https://academic.oup.com/ej/article/124/575/62/5076978) · [Angelini et al. 2022](https://centaur.reading.ac.uk/98329/1/information_efficiency_angelini_de_angelis_singleton.pdf) · [Betfair in-play delay](https://caanberry.com/betfair-exchange-in-play-delay-explained/) · [courtsiding](https://en.wikipedia.org/wiki/Courtsiding) · [futures inefficiency](https://www.twentyfirstgroup.com/the-future-of-futures-solving-the-outright-market-in-sports-betting/) · Crypto: [Saguillo et al. AFT 2025](https://arxiv.org/abs/2508.03474) · [Dubach Polymarket microstructure](https://arxiv.org/html/2604.24366v1)

**Frameworks / methodology / architecture:** [hftbacktest repo](https://github.com/nkaz001/hftbacktest) · [debugging discrepancies](https://hftbacktest.readthedocs.io/en/latest/debugging_backtesting_and_live_discrepancies.html) · [QuantStart LOB backtests](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II/) · [Market Simulation under Adverse Selection](https://arxiv.org/pdf/2409.12721) · [Bailey-Borwein-LdP-Zhu AMS 2014](https://www.ams.org/notices/201405/rnoti-p458.pdf) · [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) · [Hummingbot architecture](https://hummingbot.org/blog/hummingbot-architecture---part-1/)

**Strategy:** [Avellaneda-Stoikov](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf) · [Guéant-Lehalle-Fernandez-Tapia](https://arxiv.org/abs/1105.3115) · [Glosten-Milgrom](https://www.sciencedirect.com/science/article/pii/0304405X85900443) · [Hummingbot XEMM](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)

**Verification caveats:** A-S liquidity term is `(2/γ)` (a `(2/k)` variant circulates — wrong); Glosten-Milgrom market-breakdown is qualitative, not a crisp cutoff; latency-race "tax" has two conflated figures (~0.5 bp UK vs ~1.5 bp US-extrapolated, same paper); *which* CEX leads crypto price discovery is unsettled (state it as "a fast CEX leads the slower PM"); Betfair-delay/courtsiding and the crypto "15–20% resolve in final 10s" figures are practitioner/anecdotal; the Polymarket microstructure preprint is single-author/recent; some hftbacktest code-level details come from an AI-generated docs mirror (concepts corroborated by official docs); on-chain "no L3" means no *public* L3. The maker-PnL equation is a **practitioner attribution heuristic**, *not* "the Glosten-Milgrom / Avellaneda-Stoikov decomposition" — GM/AS are microfoundations for two terms only; the additive spread decomposition descends from Stoll (1978) / Ho-Stoll (1981) / Glosten-Harris (1988) / Madhavan-Smidt (1991) / Huang-Stoll (1997), and the rebate is a maker-taker institutional feature in neither.

## Cross-links

Strat shape this validates: [[mm_concepts_and_strategy_buildup]] (Layers 1–6) · [[strat_market_making]]. Our data limits: [[mm_clob_capture_semantics]]. Existing tooling to build on: `scripts/dali_clob_replay_features.py` (L2 replay), `scripts/od_v4_queue_replay.py` (queue-aware fill replay), `scripts/dali_block_k2_quoting_sim.py` (maker sim), `polymarket/execution/maker/` (the live engine, [[mm_maker_infra_audit_findings]]). Live loop the backtest feeds: [[mm_politics_negrisk_live_loop_design]].
