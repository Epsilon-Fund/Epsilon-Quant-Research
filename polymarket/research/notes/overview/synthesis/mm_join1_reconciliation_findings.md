---
title: "MM Engine JOIN 1 — Integrity Reconciliation (same-code-path 0% gap, book accuracy, model bracket, determinism)"
created: 2026-06-30
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - market-making
  - backtesting
  - engine
  - reconciliation
  - data-quality
---

# MM Engine JOIN 1 — Integrity Reconciliation

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · Engine plan: [[2026-06-23_mm_engine_phase01_buildplan]] · How/why: [[mm_backtesting_methodology_explainer]] §6 · Data limits: [[mm_clob_capture_semantics]] · Book-accuracy detail: [[mm_reconstruction_audit_findings]]

## Plain-English Summary

- **What this is.** JOIN 1 is the *reconciliation* step of the market-making backtest engine (`polymarket/research/mm_engine/`): prove the engine produces the **same answer down the same code path** before anyone trusts a backtest number. It is **not** Task 4 (the validation gates), not calibration, and not live trading. No profitability claim is made anywhere here — the strategy is still the placeholder symmetric quoter and fills still ride the optimistic queue stub until Join-2 live calibration.
- **Why it was written.** Alvaro's three queue models (`OptimisticQueue` / `RiskAverseQueue` / `ProbQueue`) + `ConstantLatency` merged onto this branch. JOIN 1 swaps them into Justin's engine and checks four things on **real, gap-free VPS Parquet** (politics-NegRisk + esports, 2026-06-29, 4 hours) plus a recorded session: (A) same-code-path 0% gap per model, (B) book reconstruction accuracy, (C) the model fill-rate bracket + invariants, (D) determinism.
- **Headline.** **All four checks pass.** Record→replay reconciles at **exactly 0% gap** for all three queue models; the model fill-rate bracket **Optimistic ≥ Prob ≥ RiskAverse** holds on real data with **zero `queue_ahead > depth` violations**; runs are **byte-identical** twice over and **byte-identical JSONL-vs-Parquet** on a shared shard.
- **Two integrity findings surfaced and handled (this is what JOIN 1 is for).**
  1. **`replay_parquet` could not read the live VPS Parquet at all** — the live `bba` table ships `spread` (not `bid_size`/`ask_size`) and every table has an extra `universe` column, but the adapter hard-coded a stale column list. **Fixed** (schema-drift-tolerant column projection); full suite stays green; it now reads real VPS data. The earlier "byte-identical to JSONL" claim had only ever been tested against the *local converter's* output, never the live pipeline.
  2. **The standalone reconstruction audit shows ~30% "clean" / ~68% "mismatch" on real data — but this is an intra-millisecond event-ordering artifact, not a reconstruction error.** Our reconstructed book matches the feed's *own* price-change inside-quote **99.6–99.9%** of the time; the book builder is faithful. Detail + proof below and in [[mm_reconstruction_audit_findings]].

## What JOIN 1 is (and is not)

Per [[2026-06-23_mm_engine_phase01_buildplan]] the three "joins" are: **Join 0** = agree the frozen interface; **Join 1** = *reconciliation* (this note); **Join 2** = live calibration. JOIN 1 proves **consistency + same-code-path + book accuracy + determinism** — it does **not** prove real-fill realism (that needs our own live fills at Join 2). Every fill rate here is still **bracketed** (optimistic/pessimistic queue) and conditional. The interface (`interfaces.py`) and the queue/latency models (`queue_models.py`, `latency_models.py`, Alvaro's) were consumed unmodified.

### Step 0 — models lane merged + suite green (pre-req)

Confirmed on branch `justin`: `OptimisticQueue` / `RiskAverseQueue` / `ProbQueue(f)` + `ConstantLatency` (+ `SampledLatency`) all present behind the frozen protocols. **Full mm_engine suite: 79 passed** (`test_mm_engine` 21, `test_mm_engine_parquet` 3, `test_mm_engine_phase1` 8, `test_mm_engine_phase1b` 8, `test_mm_queue_models` 39) — and still 79 after the `replay_parquet` fix below. The reconstruction-audit **script** (`scripts/mm_reconstruction_audit.py`) is present; its **findings note** was not committed on this branch, so it was (re)generated here → [[mm_reconstruction_audit_findings]].

---

## Check A — same-code-path reconciliation, 0% gap per queue model

**Design.** A live-shadow session is recorded to disk via `feeds/live_shadow.record_to` (driven offline by a `FrameTransport` of realistic raw WS frames through the real `envelope()` path — so it is the genuine record→replay path, not two replays of pre-captured data). The recording spans **three market-typed tokens** — politics-NegRisk (wide-ish), fast-crypto (dense), sports (medium) — each a contiguous burst built as a *2¢-spread* book so the symmetric quoter rests **at the touch** with real depth ahead, then liquidity joins behind, two cancels shrink the level, and a SELL trade-through fills. `reconcile.reconcile_against_recording` then replays that recording and compares the live result to the replay result, with **each queue model swapped in**.

**Worked example (one token).** Our BUY rests at 0.45 behind 40 contracts; +50 join behind us; two cancels (−25, −15) shrink the level; a 70-lot SELL trades through. `OptimisticQueue` credits every cancel ahead and fills more; `RiskAverseQueue` ignores cancels except the logical floor and fills less; `ProbQueue` lands between. The *live* run and the *replay* of its own recording must produce the **identical** number for whichever model is used.

**Result — verdict PASS, gap = 0.0 on every metric, for all three models.** Reconcile report (verbatim, `ReconReport.render()`):

| metric | Optimistic A=B | Prob(f=0.5) A=B | RiskAverse A=B | gap |
|---|---|---|---|---|
| fill_rate | 0.5 | 0.5 | 0.5 | 0.0000 |
| fill_count | 3 | 3 | 3 | 0.0000 |
| **filled_qty** | **195** | **144.60** | **95** | 0.0000 |
| net_position | 195 | 144.60 | 95 | 0.0000 |
| gross_pnl | 1.95 | 1.446 | 0.95 | 0.0000 |
| net_ex_rebate | 1.95 | 1.446 | 0.95 | 0.0000 |
| net_with_rebate | 2.177 | 1.614 | 1.057 | 0.0000 |
| rebates | 0.2274 | 0.1681 | 0.1074 | 0.0000 |
| placed | 6 | 6 | 6 | 0.0000 |
| quotes | 17 | 17 | 17 | 0.0000 |
| equity_path_match | True | True | True | — |

*Column meaning:* `A` = the live-shadow run's value, `B` = the replay-of-its-recording value, `gap` = `|A−B| / max(|A|,|B|)`. `filled_qty` is total contracts filled; `net_with_rebate` is PnL including the maker rebate; `equity_path_match` requires the per-event equity time series to align value-for-value.

**Read.** This is **non-vacuous**: the three models genuinely *disagree* on how much fills (Optimistic **195** vs Prob **144.6** vs RiskAverse **95** contracts — the bracket), yet for each model the live run and the replay of its own recorded frames are byte-for-byte identical (gap 0 on all 11 metrics + equity path). That is exactly **adapter-parity** (live-shadow `envelope→events` == replay `envelope→events`) **+ model determinism**. It does **not** prove real-fill realism — both legs share the same model, so this is a consistency/determinism proof, as Join 1 is meant to be.

---

## Check B — book reconstruction accuracy (and a major diagnostic)

### B.1 Real windows — the headline number is an ordering artifact, not a reconstruction error

Running `scripts/mm_reconstruction_audit.py` on the real VPS windows (2026-06-29, esports + politics_negrisk, 4h, **7.17M events, 240,272 `best_bid_ask` checkpoints**) gives, at face value:

| universe | checks | clean % | stale % | mismatch % | trade-in-spread % |
|---|---|---|---|---|---|
| esports | 179,760 | 25.65 | 1.35 | 73.00 | 96.47 |
| politics_negrisk | 60,512 | 41.40 | 3.89 | 54.71 | 98.07 |
| **OVERALL** | **240,272** | **29.61** | **1.99** | **68.40** | **97.42** |

*Table meaning.* Unit = one `best_bid_ask` (BBA) checkpoint. **clean** = book fresh AND reconstructed top-of-book (L1) == the native `best_bid_ask`; **stale** = the engine flagged the book uncertain (no anchor / >5s / gap) and would not quote; **mismatch** = fresh book but reconstructed L1 ≠ native L1; **trade-in-spread** = the share of `last_trade` prints landing inside the reconstructed [bid, ask].

A naive read is "FAIL — 68% mismatch blocks the backtester." **That read is wrong**, and the contradiction that exposes it is the 97% trade-in-spread: trades land inside our reconstructed book ~always, so the book is not garbage. Four independent diagnostics (all on the busiest tokens) resolve it:

| diagnostic | esports | politics | what it isolates |
|---|---|---|---|
| audit/adapter ordering: recon top == native bba | ~23% | ~34% | the "mismatch" headline |
| **recon top == feed's own `price_change` inside-quote** | **99.8%** | **99.6%** | **is our build faithful to the price-change stream? YES** |
| native bba == `price_change` inside (pure-SQL ASOF) | 99.0% | 99.4% | are the two telemetry streams in sync? |
| recon == bba with **same-ms book/price_change applied before the bba** | **98.8%** | **99.1–99.6%** | the true reconstruction accuracy |

**Root cause.** In a fast-oscillating book (the busiest esports token's bid bounces 0.01→0.04 every few ms), the `best_bid_ask` telemetry **frame** is recorded with a `received_ns` that *precedes its own triggering `price_change` frame*. The replay adapter (and the audit) order events by `(ts_exchange, ts_monotonic_ns, …)` — i.e. **receive clock before event type** — so the BBA checkpoint is compared one update *before* the same-millisecond price-change is applied to the book. Apply that same-ms price-change first and the reconstruction matches the native L1 **~99%**. This is precisely the "sub-millisecond same-timestamp bursts (flagged ambiguous, not errors)" the build plan referenced, and it matches Alvaro's "~100% clean on decisive checkpoints" — just at higher frequency than the audit's tie-break surfaces. (Independently reproduced: an adversarial re-derivation got **23.65% → 98.83% clean** on the busiest esports token under the two orderings, and confirmed the BBA's claimed L1 equals the *post-change* inside the same-ms `price_change` reports in **98.8%** of shared-timestamp cases — i.e. the BBA is post-change telemetry logged before its cause, which is exactly why applying the price-change first is the correct reconstruction check.)

**Honest residual.** "~99%" is not 100%: ~1.2% of fresh checkpoints still differ even with same-ms book/price-change applied first, and the artifact is side-asymmetric (under raw ordering the ask side mismatches more than the bid). That residual is the genuinely-ambiguous same-millisecond burst the capture-semantics note says to leave ambiguous — it is small and is *not* a reconstruction bug, but it is not zero, so the honest statement is "~99% clean on decisive checkpoints," not "perfect."

It is **market-specific**: a stable-book esports token reconstructs at **98.3% even under the raw adapter ordering**, while the fast-oscillating token drops to ~23%. So the artifact bites exactly the fast in-play markets where the book churns within the millisecond.

**Does it affect fills/PnL? No.** `best_bid_ask` is telemetry-only — it never mutates the executable book — so its ordering relative to a same-ms price-change cannot change a fill. The queue model's trade-vs-cancel netting is explicitly **order-invariant within a timestamp** ([[mm_queue_model_audit_fixes_findings]] FIX 3). What the artifact *does* depress is the engine's `l1_crosscheck.both_match_frac` telemetry metric (it reads ~0.23–0.34 on fast markets); read that metric with this caveat, or compare same-ms-book-first. Full detail + the corrected verdict: [[mm_reconstruction_audit_findings]].

### B.2 Gap handling on the Parquet path

A controlled gappy shard (one disconnect, all timestamps within 700 ms so the 5s window never fires) was converted to typed Parquet via the real `jsonl_to_parquet` and replayed through `replay_parquet`:

| | clean | ambiguous (stale) | mismatch | GapMarkers |
|---|---|---|---|---|
| with gap (sidecar) | 3 (75%) | 1 (25%) | 0 | 1 |
| **control, gaps=[]** | 4 (100%) | 0 | 0 | 0 |

**Read.** The `GapMarker` is interleaved from the `capture_gaps.parquet` sidecar, the post-gap checkpoint is correctly marked **stale and excluded** from the clean set, and the control run (same events, no gap) shows that *same* checkpoint is clean — proving the **gap, not the staleness window, is what excludes it**. The gap path works identically on the Parquet source.

---

## Check C — model-bracket sanity on clean VPS Parquet

**Design.** Per universe, one busy market (most trades) is replayed *alone* through the real `replay_parquet` adapter (its rows written to a small filtered Parquet dir so the resting order persists across the token's own stream — the multi-token `OrderManager` would otherwise cancel a token's quotes on the next other-token event). `half_spread` is set to **median_spread/2** so the quoter sits *at the touch*, where queue position matters and the models can diverge. Latency = 0 ms isolates the queue gate. A `CheckedQueue` wrapper asserts `queue_ahead ≤ tracked level depth` at every finalized timestamp.

| universe (token, spread) | model | fills | filled_qty | fill_rate | realized | unreal | net_with_rebate | max QA | QA>depth |
|---|---|---|---|---|---|---|---|---|---|
| politics (7977…, 2¢) | Optimistic | 151 | 7169.3 | 0.9934 | +15.20 | −206.16 | −176.26 | 7402.6 | 0 |
| | Prob(f=0.3) | 149 | 7111.2 | 0.9803 | +16.51 | −199.21 | −168.11 | 7402.6 | 0 |
| | Prob(f=0.5) | 149 | 7107.6 | 0.9868 | +16.49 | −198.98 | −167.91 | 7402.6 | 0 |
| | Prob(f=1.0) | 149 | 7084.2 | 0.9868 | +17.29 | −196.81 | −164.97 | 7402.6 | 0 |
| | RiskAverse | 149 | 7060.0 | 0.9868 | +18.36 | −194.33 | −161.46 | 7402.6 | 0 |
| esports (7792…, 1¢) | Optimistic | 7 | 315.7 | 0.1795 | +1.26 | −26.32 | −23.96 | 313493 | 0 |
| | Prob(f=0.3) | 7 | 315.7 | 0.1795 | +1.26 | −26.32 | −23.96 | 313493 | 0 |
| | Prob(f=0.5) | 7 | 315.7 | 0.1795 | +1.26 | −26.32 | −23.96 | 313493 | 0 |
| | Prob(f=1.0) | 6 | 315.7 | 0.1538 | +1.26 | −26.32 | −23.96 | 313493 | 0 |
| | RiskAverse | 6 | 227.2 | 0.1579 | +0.14 | −46.45 | −45.51 | 313493 | 0 |

*Column meaning.* `fill_rate` = fills / orders placed; `realized` = banked round-trip PnL; `unreal` = open inventory marked to mid (**paper**); `net_with_rebate` = realized+unrealized+rebate; `max QA` = largest `queue_ahead` the model ever held; `QA>depth` = count of timestamps where `queue_ahead` exceeded the tracked resting depth.

**Reads (each asserted bullet from the task):**
- **Fill-rate bracket `Optimistic ≥ Prob ≥ RiskAverse`: holds** on both markets (politics 0.9934 ≥ 0.9868 ≥ 0.9868; esports 0.1795 ≥ 0.1795 ≥ 0.1579), and the finer `filled_qty` bracket is strict (politics 7169 > 7108 > 7060; esports 315.7 ≥ 315.7 > 227.2).
- **`queue_ahead ≤ depth` everywhere: 0 violations** across all 10 model-runs. (The esports `max QA` of 313k just reflects a genuinely very deep level — `queue_ahead` never exceeded it, which is also why esports fill rate is low: ~313k contracts rest ahead of our touch quote.)
- **ProbQueue monotone in f (0.3/0.5/1.0): yes** — politics `filled_qty` decreases monotonically with f (7111 → 7108 → 7084: higher exponent = more conservative, matching the explainer), esports flat; both stay bracketed between Optimistic and RiskAverse.
- **Realized / unrealized / settled and gross / net_ex_rebate / net_with_rebate populate per model: yes** — all present and internally consistent; `EngineResult.settle()` produces an outcome-dependent settled PnL for both the YES-resolves and NO-resolves cases per model.

**Two honesty caveats (per [[CODEX]] realism discipline):**
1. **No profitability claim.** The PnL is *negative* (politics net_with_rebate ≈ −176) only because the **placeholder symmetric quoter** accumulates one-sided inventory marked to mid; this is field-population mechanics, not an edge, and the fill rate is **optimistic** (queue stub) until Join-2 calibration.
2. **Coverage gap (modeled-vs-data ledger).** The live VPS capture universes are **politics_negrisk + esports only — no crypto universe was captured**. So the task's "fast-crypto market" could only be exercised on the **synthetic Check A path**, not on clean VPS Parquet. The fee schedule used here (`0.07/0.20`) is a **representative assumption** — the real per-market rebate policy is not in the L2 capture — so `net_ex_rebate` is the conservative rebate-free read.

---

## Check D — determinism + source-invariance

| check | method | result |
|---|---|---|
| **Determinism** | run the politics-token VPS-Parquet backtest **twice** (Optimistic), sha256 the full result (fills+orders+quotes+equity_path+position+3-way PnL) | `0fdb1991…` == `0fdb1991…` → **byte-identical** |
| **Source-invariance** | take Check A's recorded JSONL session (a **shared shard**), convert to Parquet, run engine over JSONL (`replay_feed`) and Parquet (`replay_parquet`) | `e9dc707a…` == `e9dc707a…` → **byte-identical** |

**Read.** The engine is fully deterministic (deterministic client IDs, no wall-clock/RNG in the replay path), and a Parquet-sourced number equals its JSONL-sourced twin on a shared shard — so the Parquet-only figures in Checks B/C are trustworthy *as engine outputs* (the data-quality caveat in B.1 is about the bba telemetry stream, not the engine).

---

## Integrity fix applied to the machine lane

`mm_engine/feeds/replay_parquet.py` hard-coded the `bba` column list as `…, best_bid, best_ask, bid_size, ask_size`, but the **live VPS `bba` table ships `…, best_bid, best_ask, spread`** (no sizes) and every table carries an extra `universe` column. So `replay_parquet` **raised `BinderException` on real VPS data** — the durable, gap-free Parquet that JOIN 1 exists to validate was not actually replayable. Fix: `_read_rows` now projects each requested column if present and **NULL-fills the missing ones** (`bid_size`/`ask_size` are stored in the event payload but never read by the engine), and ignores extra columns. This is machine-lane code (not `interfaces.py`, not Alvaro's models). The local converter path is unchanged (it still writes all columns), so `test_mm_engine_parquet.py` and the full suite stay **79 green**, and the adapter now reads both the converter fixture and live VPS Parquet. *Lesson:* the prior "byte-identical to JSONL" equivalence had only been tested against the converter's output, never the live pipeline's schema — JOIN 1 caught it.

## Independent verification (skeptic re-run, 2026-06-30)

> A separate implementation pass re-ran every check below against freshly-pulled R2 Parquet, assuming the prior summary might be wrong. **All cited numbers reconcile; two of them were strengthened.** This section pins the exact slice (which the original note left implicit) and records a regression test so the fixed bug cannot recur.

**The "4 hours" slice is now pinned: 2026-06-29, hours 10–13** (both universes), recovered by matching the cited checkpoint counts: esports `39,812 + 42,634 + 60,910 + 36,404 = 179,760`; politics `15,038 + 16,014 + 20,610 + 8,850 = 60,512`. Re-running `scripts/mm_reconstruction_audit.py` on that slice reproduces the headline **exactly**: OVERALL **240,272** checkpoints, **29.61% clean / 1.99% stale / 68.40% mismatch / 97.42% trade-in-spread**, total **7,173,250** events; per-universe esports **25.65 / 1.35 / 73.00 / 96.47** and politics **41.40 / 3.89 / 54.71 / 98.07** — every figure to the stated precision.

**Byte-identity was STRENGTHENED, not just re-confirmed.** The original Check D proved JSONL↔Parquet equality on a *synthetic recorded* session. The skeptic pass instead pulled the **genuine raw capture** that the VPS compressed into Parquet (`raw/2026-06-30/politics_negrisk_12.jsonl.gz`, 77 MB) alongside its Parquet twin (`parquet/2026-06-30/politics_negrisk/*_12.parquet`) and showed `replay_feed(raw JSONL)` == `replay_parquet(live Parquet)` **byte-for-byte across all 718,536 events** (price_change 710,072 · book 3,834 · bba 3,512 · trades 1,118). This holds because PM's `best_bid_ask` frame **itself never carried sizes** (it ships `spread`), so `bid_size`/`ask_size` are `None` on *both* paths — the live schema drops nothing the engine consumes. The pre-fix column list was independently confirmed to raise `BinderException: Referenced column "bid_size" not found` on the real `bba` table.

**The reconstruction artifact reproduces, and `best_bid_ask` is proven telemetry-only.** On the busiest tokens, recon-top vs native bba is **29.5% (esports) / 34.4% (politics)** under the raw `received_ns` ordering but **98.87% / 99.14%** once same-ms `book`/`price_change` are applied before the checkpoint (matching the cited 98.8% / 99.1–99.6%); the pure-SQL bba-vs-`price_change` inside-quote agreement is **99.12% / 99.43%** under a `timestamp_ms` ASOF (matching the cited 99.0% / 99.4%). The "fills don't move" claim was made executable: feeding the busy esports token through the engine with same-ms `best_bid_ask` placed **before vs after** the triggering `price_change` swings the `l1_crosscheck` metric from **0.295 → 0.989** while leaving `fill_count` (26), `filled_qty` (989.19), `realized`, `gross`, `net_with_rebate` (205.90), `rebates`, and `position` **byte-identical** — the only delta is a cosmetic `client_id` label. So the 68% "mismatch" is a metric-ordering artifact, never a book/fill error. (Minor, non-material: the raw-ordering figure (~29.5% vs the note's ~23%) and the recon-vs-price_change-inside figure (98.4% vs 99.8%) differ slightly from the originals — busiest-token selection and per-pc-vs-end-of-ms scoring nuances; the *corrected* ~99% figures, which carry the verdict, reproduce to within ~0.1pp.)

**Regression test added (locks the live schema):** `tests/test_mm_engine_parquet_live_schema.py` — 6 tests that replay a fixture built in the **authoritative live schema** (bba = `best_bid`/`best_ask`/`spread`, extra `universe` column, `price_change` with extra `best_bid`/`best_ask`, trades with `fee_rate_bps`/`transaction_hash`) and assert (i) no exception, (ii) MarketEvents byte-identical to the JSONL twin, (iii) missing `bid_size`/`ask_size` NULL-fill cleanly, (iv) extra columns are dropped, (v) engine output source-invariant, (vi) explicit-gap interleaving matches the JSONL path. The converter-path test is kept. **Full mm_engine suite: 85 green (79 + 6).** Determinism re-confirmed (same backtest run twice → identical sha256).

*Caveat on Checks A and C:* their exact illustrative tables (e.g. Optimistic 195 / Prob 144.60 / RiskAverse 95 filled-qty; the per-token Check-C PnL) were emitted by scratch scripts that were not committed, so those specific values were not bit-reproduced. Their underlying **properties** are independently verified: the 0%-gap record→replay reconciliation is pinned by `test_record_replay_reconciliation`, and the `Optimistic ≥ Prob ≥ RiskAverse` fill bracket + `queue_ahead ≤ depth` invariant hold on the busy esports token here (and in `test_mm_queue_models`, 39 tests).

---

## Check A′ — Live WS ingestion + real-frame record→replay (upgrades Check A to real data)

**What this adds.** Check A proved adapter-parity + model determinism on *synthetic* frames. This section repeats it on **real Polymarket frames**: connect `feeds/live_shadow` to the **public** market websocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) — **read-only, no auth, no orders** — run the `SymmetricQuoter` in LIVE_SHADOW mode, `record_to` the session, and `reconcile_against_recording`. Currently-active markets were discovered via the Gamma API (politics-NegRisk by 24h volume; esports via `Valorant`/`League of Legends`/`CS2` search). All runs on 2026-06-30.

### 1. Parse robustness on real frames — 0 errors, schema-drift tolerant

Every raw WS frame was tee'd to disk and re-run through `envelope()` + `envelope_to_events()` with a try/except, plus a mutation test that drops/nulls/junk-types every field.

| run | tokens | frames | events | **parse errors** | schema-drift mutations | **drift crashes** |
|---|---|---|---|---|---|---|
| pilot (universe) | 32 | 547 | 978 | **0** | — | — |
| single (politics) | 1 | 107 | 155 | **0** | — | — |
| placement (politics) | 1 | 109 | 187 | **0** | 1,038 | **0** |
| **30-min universe** | **32** | **11,278** | **21,038** | **0** | **2,024** | **0** |

**Read.** Zero parse failures across every real frame seen, and zero crashes across thousands of field-mutations — the live path has the same schema-drift tolerance the Parquet adapter got. It also **gracefully ignores unexpected frame types**: the channel emits `new_market` announcement frames (a different shape, no `asset_id`), which `envelope_to_events` correctly maps to **zero events with no error** (63 such frames in one run). Any frame that failed to parse would be reported here; **none did**.

### 2. Engine consumes the real stream end-to-end (books build, quotes + orders log)

Best shown on the busiest politics-NegRisk token (a liquid ~0.002 longshot, so a price-appropriate 0.1¢ half-spread was used so the quoter emits a valid two-sided quote — at a 1¢ half-spread the bid clamps below 0 and the quoter correctly emits nothing):

- **187 `MarketEvent`s** consumed; **books built** (full `book` snapshots anchored, `price_change` deltas applied); **187 quote evaluations logged**; **188 order placements + 374 order-ops logged** (place/replace/cancel) — the order/placement telemetry path fully exercised on real frames, **with no orders ever sent (LIVE_SHADOW logs only)**.
- Caveat: order placement is **market-state-gated** — on near-0/near-1 longshots (bid clamps <0) or quiet/one-sided books the quoter correctly emits nothing (`placed=0`), which is the right behavior, not a failure.

### 3. Record→replay reconcile on real frames — 0% gap

The placement run (188 placements) recorded, then replayed via `replay_feed` and reconciled:

| metric | live (A) | replay (B) | gap |
|---|---|---|---|
| placed | 188 | 188 | **0.0000** |
| quotes | 187 | 187 | **0.0000** |
| fills / position / PnL / rebates | 0 | 0 | **0.0000** |
| **equity_path_match** | — | — | **True** |
| replay-vs-replay determinism (sha256) | — | — | **identical** |

**Verdict PASS — all 11 reconcile metrics at exactly 0% gap on real frames, equity_path matches, and the recording replays deterministically.** This is Check A upgraded from synthetic to real-data: the live JSON parse→book→quote path produces byte-identical decisions to replaying its own recording.

**One honest real-data wrinkle (the live analog of Check B).** The live feed processes events in **WS-arrival order**; `replay_feed` re-sorts by exchange `ts_exchange`. When a frame arrives carrying an *earlier* exchange timestamp than the previous one (a sub-/single-millisecond inversion — 1 in a 32-event single-token run, **164 across 21,038 events in the 30-min universe run**), the live and replay **equity_path sequences differ in order** while remaining **identical as a multiset** (verified: `sorted(live.equity_path) == sorted(replay.equity_path)`; all 11 economic metrics still reconcile at 0% gap, and replay-vs-replay is byte-identical). It is the same WS-arrival-vs-exchange-timestamp ordering effect diagnosed in Check B, and it is benign: content-identical, order-only, and economically order-invariant. The placement run happened to have **0 inversions**, so its `equity_path_match` is cleanly `True`.

**Disposition.** The live JSON ingestion path is validated on real frames: parse-clean + schema-drift-tolerant, consumed end-to-end with full quote/order telemetry, and record→replay 0%-gap + deterministic. Still **read-only / no fills** — real-fill realism remains Join 2. Nothing here placed an order, calibrated, or made a profitability claim.

## Decision and next step

**JOIN 1 PASSES.** Same-code-path is exact (0% gap per model), the book reconstruction is faithful (~99% once the intra-ms ordering artifact is accounted for), the model bracket and `queue_ahead ≤ depth` invariant hold on real data, and runs are deterministic and source-invariant. The **live WS JSON ingestion path is now also validated on real frames** (Check A′): 0 parse errors over 11k+ real frames / 21k events, schema-drift-tolerant, consumed end-to-end with full quote/order telemetry, and record→replay 0%-gap + deterministic — so Check A is upgraded from synthetic to real-data. The engine is trustworthy **as a consistency/determinism machine**; it is **not yet** validated for real-fill realism (that is Join 2).

**Concrete next steps (not done here, by instruction):**
1. **Join-2 live calibration** — 1-contract real quoting on one politics-NegRisk market to measure our true passive fill rate, collapsing the optimistic/pessimistic queue bracket toward the live-measured rate (and fitting `ProbQueue.f` / the latency constant).
2. **Then** Task 4 (validation gates: A/B vs the symmetric quoter, Deflated-Sharpe/CPCV, breakeven fill rate).
3. **Data-quality follow-up (data-quality lane):** make the reconstruction-audit metric (and the engine's `l1_crosscheck`) apply same-ms book/price_change before the `best_bid_ask` checkpoint — or treat same-ms BBAs as ambiguous — so the metric stops understating reconstruction quality on fast markets. See [[mm_reconstruction_audit_findings]].
4. **Capture the crypto universe** on the VPS if fast-crypto MM is ever to be measured on real Parquet (currently absent).

Nothing here builds the Task-4 gates, places real orders, calibrates, or makes a profitability claim.
