---
title: "MM Engine — Phase 0 & 1 Build Plan (Justin = machine) + Handoff (Alvaro = models)"
created: 2026-06-23
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - COWORK
tags:
  - market-making
  - backtesting
  - engine
  - build-plan
  - handoff
---

# MM Engine — Phase 0 & 1 Build Plan + Alvaro Handoff

> Plan: [[mm_backtest_infra_plan|2026-06-16_mm_backtest_infra_plan]] · Why: [[mm_backtesting_methodology_explainer]] · Data limits: [[mm_clob_capture_semantics]].
> **Naming:** "Phase 0/1" here = the **backtest-engine** effort (NOT the politics live-loop phases). The three joins from the plan: **Join 0** = agree interface (Phase 0); **Join 1** = reconciliation; **Join 2** = calibration.

## Context (state as of 2026-06-23)

VPS→cloud **live L2 capture is up and verified** (continuous, gap-free going forward; the local ~1-week is the gappy test fixture). Goal: **one strategy-agnostic engine** that runs the same quoting code in replay and live, made *reliable* by reconciling backtest vs live. Split: **Justin builds the machine; Alvaro builds the models.** The strategy is a stub (symmetric quoter) for now — we are building the machine, not a strat.

**Reuse, don't rebuild:** `polymarket/research/lib/clob_book.py` (`ClobBook` — the book builder/OFI), `scripts/dali_clob_replay_features.py` (replay/state logic; note its convention: `best_bid_ask` is telemetry-only, never mutates the executable book), the capture format from `scripts/dali_live_clob_capture.py`, and `polymarket/execution/maker/` (safety + signing — only bridged later, at Join 2's real 1-contract step).

**Engine home:** new package `polymarket/research/mm_engine/` (research venv; reuses `lib/clob_book`). Live-**shadow** (read-only WS) lives in the same package. Real 1-contract execution bridges to `execution/maker/` at Join 2, not before — that keeps Phase 0/1 inside one venv.

---

## The shared interface — the ONLY thing both halves touch (agree in Phase 0)

Put this in `mm_engine/interfaces.py`. It is the whole "contract": one event shape + three function shapes. Once agreed, the halves are independent.

```python
@dataclass(frozen=True)
class MarketEvent:                 # emitted IDENTICALLY by replay and live adapters
    type: str                      # "book" | "price_change" | "last_trade" | "best_bid_ask"
    token_id: str
    ts_exchange: int               # PM message.timestamp (ms epoch) — event-time ordering
    ts_local_iso: str              # received_at (UTC ISO)
    ts_monotonic_ns: int           # received_monotonic_ns — gaps/latency only
    payload: dict                  # raw fields (levels+sizes / price+side / etc.)

@dataclass
class BookState:
    token_id: str
    bids: list[tuple[float, float]]  # (price, size), top-N
    asks: list[tuple[float, float]]
    ts_exchange: int
    stale: bool                      # True if beyond staleness window or across a capture gap

class QueueModel(Protocol):          # ALVARO
    def on_event(self, ev: MarketEvent, book: BookState) -> None: ...
    def fill(self, our_order, book: BookState, trade: MarketEvent | None) -> float: ...  # filled qty (0..order size)
    def calibrate(self, live_fills) -> None: ...

class LatencyModel(Protocol):        # ALVARO
    def round_trip_ms(self, ts_exchange: int) -> float: ...

class Strategy(Protocol):            # JUSTIN stubs (symmetric quoter); real one later
    def quote(self, book: BookState, inventory: float, params: dict) -> list["Order"]: ...
```

Plus one agreement: **the reconciliation metric set + tolerance** — fill rate, position path, equity; target **<5%** backtest-vs-live fill-rate gap.

**Repo invariants for all code:** `PYTHONPATH=. uv run …` from `polymarket/research/`; DuckDB over Parquet; lowercase `0x` addresses; **lookahead-free** (filter by `ts_exchange` before aggregating); **non-overlap** accounting; CI (not point estimates) before any "positive." Deterministic/seeded replay.

---

## Phase 0 — agree + scaffold (both; ~an afternoon)

**Joint (Join 0):** finalize `interfaces.py` above and the reconciliation metrics. This is the only co-decision before parallel work — freeze it.

**Justin:**
1. Create `mm_engine/` with `interfaces.py` + stub implementations: `SymmetricQuoter` (Strategy), `OptimisticQueue` (QueueModel: back-of-queue, fill on trade-through), `ConstantLatency` (LatencyModel).
2. **Replay feed adapter** — read the captured JSONL shards → emit `MarketEvent` stream in `ts_exchange` order, honoring `capture_gaps.jsonl` (mark `stale=True` after a gap until the next full `book`).
3. **Live-shadow feed adapter** — connect to the public market WS → emit *identical* `MarketEvent`s. (Read-only; no orders.)
4. **Day-one smoke:** run `SymmetricQuoter` through the replay adapter on one market, then the *same code* against the live-shadow adapter. This smoke is the proof the same-code-path works.
5. **Document the VPS capture** (no summary note exists yet): write `mm_clob_capture_semantics`-linked note or section — cloud path, shard layout, format, how to pull/sync. Grounds the replay adapter and unblocks Alvaro.

**Acceptance (Phase 0):** `interfaces.py` agreed; `mm_engine` imports; the symmetric quoter runs through both adapters and logs quotes; VPS capture documented.

---

## Phase 1 — Justin: build the machine (against stub models)

1. **Book builder** — wrap/`lib.clob_book.ClobBook` behind `BookState`; reconstruct top-N from `book`+`price_change`; cross-check L1 vs `best_bid_ask`; enforce ≤5s staleness + gap handling.
2. **Order manager** — place/cancel/replace; idempotent; cancel-replace throttle. In backtest it routes orders to the fill simulator; in live-shadow it logs only.
3. **Fill-simulator slot** — on each trade event, ask `QueueModel.fill(our_order, book, trade)` (× `LatencyModel` for staleness of our quote vs the trade) → realize fills against the **real trade tape** (never independent simulation).
4. **Telemetry/journal** — per quote/cancel/fill: queue rank (from the model), fill share, post-fill markout, own round-trip, `news_proximate`. Append-only JSONL (match `journal/` pattern).
5. **Reconciliation harness** — run replay + live-shadow over the same dates; emit a report comparing fill rate / position path / equity vs the tolerance.
6. Keep the **symmetric quoter** as the running strategy; expose the `Strategy`/`QueueModel`/`LatencyModel` swap points so Alvaro's real models drop in.

**Acceptance (Justin, Phase 1):** engine runs end-to-end in replay **and** live-shadow with the placeholder quoter; reconstructed L1 matches `best_bid_ask` on a high reported fraction; telemetry logs all fields; reconciliation harness produces a report. → ready for **Join 1**.

---

## Phase 1 — Alvaro: build the models  ⟵ HANDOFF

> Alvaro (or his Claude Code) owns the realism models that plug into Justin's engine via `interfaces.py`. You can start **immediately on the raw VPS L2** — you don't need Justin's finished engine to prototype, only the agreed interface.

**Front-load this (it's the foundational go/no-go):**
1. **Reconstruction audit** — on the ~1 week of L2 (and ongoing VPS data): rebuild top-N from `book`+`price_change`, compare L1 to native `best_bid_ask`, and report **clean vs ambiguous interval %** per market/category, plus gap impact. If our book doesn't reconstruct cleanly, the whole backtest is shaky — surface this first. (Method spine: [[mm_clob_capture_semantics]] § Required Reconstruction Audit.)

Then:
2. **Queue model** — implement behind `QueueModel`: `OptimisticQueue` (already stubbed), `RiskAverseQueue` (advance only on trades), and `ProbQueue` (Rigtorp/probabilistic attribution of size-decreases; tunable `f`), each with the `calibrate(live_fills)` hook. Report results as **optimistic/pessimistic bounds** until live fills exist. (Detail: [[mm_backtesting_methodology_explainer]] §1.)
3. **Latency model** — `ConstantLatency` round-trip; spec the "measure our own round-trip via unexecutable orders" step for later calibration. (Market-stratified: minor for politics; matters for crypto/in-play — §2 of the explainer.)
4. **Validation/gates** — A/B vs the symmetric quoter; wire `infrastructure/validation/overfitting_audit.py` (Deflated Sharpe / CPCV — already built for momentum). Output the **breakeven-fill-rate** per candidate market.

**What you need from Justin:** the agreed `interfaces.py` (Phase 0) and the replay harness to run models in-situ (Phase 1) — but the audit + model math run standalone on raw L2 now.
**Acceptance (Alvaro, Phase 1):** reconstruction audit reported; queue + latency models implemented behind the interface (runnable, not just math); gates wired. → ready for **Join 1**.

---

## The joins (recap — what each needs)

- **Join 1 — Reconciliation:** swap stubs → Alvaro's real models; run replay + live-shadow over the same dates; verify same-code-path + book accuracy + event/fill-logic consistency. *(Shadow has no real orders, so this proves consistency + same-code-path, not yet real fills.)*
- **Join 2 — Calibration:** 1-contract real quoting on one market (bridge to `execution/maker/` safety+signing) → real fills calibrate the queue `f` + latency constant → re-run; bounds should collapse toward the live-measured rate. This is when the backtest earns "reliable."

---

## Pointers

[[2026-06-16_mm_backtest_infra_plan]] (the plan) · [[mm_backtesting_methodology_explainer]] (queue/latency/realism) · [[mm_concepts_and_strategy_buildup]] (the strat, for the eventual real `quote()`) · [[mm_clob_capture_semantics]] (data + audit) · [[mm_maker_infra_audit_findings]] (the live maker safety/signing to bridge at Join 2). Consolidates: `dali_clob_replay_features.py`, `od_v4_queue_replay.py`, `dali_block_k2_quoting_sim.py`, `dali_paper_backtest.py`.
