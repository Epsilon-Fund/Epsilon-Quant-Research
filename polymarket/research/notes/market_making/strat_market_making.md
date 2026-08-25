---
title: "Market-Making on Polymarket — project hub (start here)"
created: 2026-06-03
updated: 2026-08-25
status: active
owner: justin
project: polymarket-mm
hubs:
  - COWORK
tags:
  - market_making
  - hub
---

# Market-Making on Polymarket — project hub (start here)

> **This is the only active research thread in this repo.** If you were asked to work on "the market-making project", this note is the whole map. It is self-contained: every concept the project needs is explained here or in [[mm_model]]. Older research notes are historical evidence with status banners — cite them as history if you must, **never build on them**.

## The idea

Rest passive buy and sell orders (quotes) on a small set of slow, deliberately chosen Polymarket event markets, and earn the gap between the buy and sell price. Two forces decide a market maker's profit:

- **Spread capture** — what you earn when ordinary, uninformed traders cross your quotes.
- **Adverse selection** — what you lose when someone hits your quote *because they know the price is about to move*. On Polymarket this is dominated by resolution: prices ultimately jump to 0 or 1, and the final days before resolution are where quoters get run over.

Speed is not the edge and was never the plan. The candidate edge is **market selection plus defense**: quote calm markets during the weeks approaching resolution, and detect and step aside from the violent episodes. Whether that candidate edge is real money is **not yet established** — see the honest framing below.

## Honest framing — read before anything else

We moved into backtesting faster than we should have. The correct way to hold this project's results is as **two understood lessons, one open path, and one low-trust start** — not as a validated strategy:

1. **Lesson (settled): how to test on this data.** A naive train-on-week-1 / test-on-week-2 split silently puts each market's calm mid-life in training and its violent endgame in testing, which makes any strategy look broken for the wrong reason. The fix that works: split by **whole market** — train on some markets' entire lifetimes, test on entirely different markets' lifetimes. Under that split, tuned settings transfer from training to test almost one-for-one. This lesson is trustworthy and must shape all future evaluation.
2. **The main open path: understanding P&L spikes.** The account-level P&L occasionally jumps sharply up or down. What we know so far: the losses trace to violent single-market price moves (the studied episode: 18¢ → 75¢ in minutes) during which an undefended quoter keeps getting filled on one side and stacks a large wrong-way position; a flow-imbalance measure tends to rise tens of minutes *before* such moves and post-fill price drift confirms *during* them; and on held-out data, calm markets earn in the approach weeks and give it back in the final days. **This anatomy is the most important thing to keep investigating** — it is the difference between a defensible quoting operation and slow bleed. It is understood at the level of one dissected episode and aggregate maps, not at the level of a reliable causal model.
3. **Started but raw — low trust: robustness of the positive result.** A first stress-testing pass asked whether the small measured edge survives different fill assumptions and accounting conventions. It reported "yes, directionally" — but this work is early, partly analytical rather than computed (the underlying raw book data had been cleaned off the laptop), and rests on assumptions we have not validated against reality, above all **latency** (everything was computed as if orders land instantly) and the **fill model** (we cannot see our own queue position in anonymous public data, so fills are simulated). Treat every ¢-per-contract number in the notes as **preliminary**. The only way to make this trustworthy is real fills from a live measurement loop.

**Keep-in-mind list (intuitions confirmed by testing):** never value leftover inventory at the midpoint — value it at the price you could actually exit at (mid-marking manufactured phantom profits in parked work); an undefended symmetric quoter is not a neutral baseline, it is a hidden directional bet; fewer simulated fills is not automatically "conservative" for profit, because for a quoter being picked off, missing fills helps; markets that trade simultaneously share news, so statistical confidence computed as if they were independent overstates certainty.

## The two lanes

### Lane 1 — backtesting / evaluation (last worked: July 2026)

What exists and works mechanically:

- A **replay engine**: feed it captured order-book data and a quoting strategy, and it deterministically simulates resting orders, queue position, fills, inventory, and honest P&L. Replaying a recorded run reproduces it exactly. Fills are always reported as a **bracket** between an optimistic and a pessimistic queue assumption, because true queue position is unobservable from public data.
- An **evaluation harness** implementing the whole-market split, honest exit-priced accounting, and standard overfitting checks.
- A **defended quoter**: centers quotes on the size-weighted fair price, skews against accumulated inventory, hard-caps position, watches the two warning signals (flow imbalance leading, post-fill drift confirming), and responds gradually — suspend the exposed side, shrink size, then withdraw and passively unwind. Its measured politics performance is *positive but preliminary* (see framing above).
- The **market map**: quote calm markets in the approach window; defend or stand aside in the endgame.

The model itself — what is fundamental versus what is an addition — is written up in [[mm_model]]. Read that second, after this note.

### Lane 2 — live machinery (last worked: 12 July 2026; stopped deliberately, mid-redesign)

The same strategy code wired to place real orders through the signing/safety layer, with hard caps, an operator-confirm requirement, and a no-real-order default proven by tests. Status:

- Real order path **works against the live venue** (it had to be rebuilt mid-session when Polymarket deprecated its old client library; orders now go through the successor SDK in an isolated subprocess).
- **Measured:** order round-trip latency ≈ **160 ms** (30 clean probes). This replaces the backtests' instant-order assumption when the loop resumes.
- **Achieved:** the first genuine resting order — placed, rested, cleanly cancelled, $0 spent, account left flat.
- **Why it stopped:** the session surfaced design findings that make the then-current instrument the wrong shape. (a) You cannot rest a *sell* on a token you don't own — but this is **conceptually solved**: in a binary market, an order to buy YES plus an order to buy NO *is* a bid and an ask (buying NO at p ≡ selling YES at 1−p), and event markets' merge/split-to-$1 mechanics give further routes. Justin is working this design himself; treat it as direction, not a blocker. (b) The quoter cancelled its own order within seconds on every price tick — the measurement instrument needs a *place-one-order-and-hold-it* mode. (c) On deep books a small order sits tens of thousands of shares back in the queue and will never fill — the measurement market must be chosen for **fillability first**.
- **Next concrete step when resumed:** the redesigned single persistent order on a fillability-chosen market → first real fills → calibrate the fill model against reality. Real fills are the gate for everything downstream.

### The data (live, shared with the machinery)

A 24/7 capture of Polymarket order books runs on a rented server (Alvaro's) and backs up to a cloud bucket (Justin's Cloudflare R2 account), continuously since 19 June 2026. Raw event logs are converted to typed columnar files (kept forever); the replay engine reads either form. Operational map and pull commands: [[mm_vps_capture_setup]]. Known gaps, both cheap and unfixed: only the politics and esports universes were ever actually captured (two further configured universes never landed), and the capture-health log is deleted when raw files expire, permanently losing gap ground-truth.

## Reliability ledger

| Trust it | Treat as preliminary / do not trust yet |
|---|---|
| The capture pipeline and its data | Every ¢/contract edge number |
| Replay determinism (record → replay, zero gap) | The fill/queue model (never validated against a real fill of ours) |
| Honest exit-priced accounting convention | The instant-order latency assumption inside all backtests |
| The whole-market split design | The robustness/stress-testing pass (early, partly uncomputed) |
| The spike anatomy at one-episode level | Any extrapolation of the market map beyond politics/esports |
| Measured 160 ms live latency | |

## Where things live

- **Notes (this folder):** `polymarket/research/notes/market_making/` — this hub, [[mm_model]], and the historical findings notes (each carries a status banner).
- **Replay engine + evaluation:** `polymarket/research/mm_engine/` and `polymarket/research/mm_eval/`.
- **Live machinery:** `polymarket/execution/` (`maker/` for the quoting bridge, runbook and live guide; `mirror/` for the venue client, signing, and the SDK gateway). The signing/venue layer is shared, live infrastructure — it is also used by the (currently deprioritised) copy-trading thread.
- **Data:** cloud bucket per [[mm_vps_capture_setup]]; local pulls land under `polymarket/research/data/`.

## Out of scope (parked — do not build on)

Earlier research eras live in this folder and elsewhere with their own notes: single-venue quoting variants that were tested and closed, wallet-level studies of profitable makers that motivated the politics focus, a valuation/fair-value overlay thread, and a market-microstructure signal lineage. Where a concept from that era matters to this project, it is already explained inline above or in [[mm_model]]. If a note's banner says historical or parked, it is evidence for the archaeologist, not input for the builder.
