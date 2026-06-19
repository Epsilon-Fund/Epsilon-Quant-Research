---
title: "Handoff — Kronos (forward-vol for the digital-option model) + Hermes (copytrade ops) evaluation"
tags: [handoff, block-k, kronos, vol, neural-net, hermes, copytrade, ops]
created: 2026-05-31
status: evaluation / scoping — no code run yet
purpose: >
  Assess two external repos against the live threads. Kronos (financial-candlestick foundation model)
  is scoped as a candidate FORWARD-VOL estimator for the K6 digital-option vol model — gated, not adopted.
  Hermes (Nous self-improving agent) is scoped as copytrade OPS infra only — zero alpha.
relationship: >
  Builds on [[block_k6_vol_findings]] (the vol model this would feed), [[block_k_plain_english_synthesis]]
  (Block K arc), [[block_a17_lightgbm_findings]] (the ML closure whose caveats govern any NN use), and
  [[2026-05-30_maker_options_delta_pivot]] (Strategy A = passive entry → hold-to-resolution → STATIC
  external hedge, the only place Kronos can matter).
---

# Handoff — Kronos + Hermes evaluation

## TL;DR

- **Kronos** = open-source decoder-only foundation model for OHLCV K-lines (probabilistic, finetunable;
  Kronos-base 102M params, 512-bar context; BTC/USDT demo). It forecasts the **external underlying**, not
  the Polymarket book. That makes it the *right side of the wall* — the direct dali book-to-book
  reversion branch was falsified; Kronos has nothing to say there and shouldn't.
- **The one additive idea:** in the K6 vol model, replace the **backward-looking EWMA-of-realized-vol** term
  with a **forward, path-implied remaining-window vol** derived from Kronos's sampled price paths — and read
  the digital fair value `P(close>K)` off the *same* paths for internal consistency.
- **Yes it is "using neural networks," but it is the ALLOWED kind.** The A1.7 ML closure was about predicting
  *Polymarket's own next move from its own book*. Forecasting Binance vol is a different target in a domain
  where NN time-series models have real published skill. The closure's discipline still binds (see gates).
- **Hermes** = Nous self-improving agent harness. Purely copytrade **ops** (serverless $5 VPS, operator-confirm
  messaging gateway, cron audits). No research value. Selective adoption only; redundant with midas + brain.
- **It does NOT solve the actual blockers:** Polymarket exit cost, maker capacity (K5: top-3 wallets ≈ 95% of
  crypto-4h maker profit), or K6's hedge-turnover problem. Both repos are edge-*enablers*, not new alpha.

## How the K6 vol model is built today (what Kronos would touch)

From [[block_k6_vol_findings]]:

- Invert the European digital `P_up = N( ln(S/K) / (σ·√τ) )` (K from the Binance window-open ref) to get
  Polymarket's **implied vol**. 95.97% of rows yield a valid positive IV.
- Compare PM IV to a **causal EWMA of Binance realized vol** — that EWMA is the forecast-of-vol-available-at-t.
- Signal = `PM_IV − EWMA`. Clean-source gap ≈ **+3.7 vol pts** average; the **far/late** bucket
  (`far_absz_ge1 | late_lt30m`) is the only one that clears: **+24.1 vol pts, CI [0.139, 0.342]**.
- **The trade still lost:** best bucket net **−9.39c**, of which **9.56c was Binance banded-hedge turnover**
  against a **0.72c unhedged** edge. The vol *sign* is right; **continuous/banded delta-hedging is the killer.**
  Static hedge is explicitly **UNTESTED**.

So the vol *forecast* was never the thing that broke K6 — turnover was. Keep that ordering sacred (Gate 1).

## The additive thesis — Kronos as a forward-vol estimator

Kronos doesn't output vol directly; it outputs a **distribution of forward OHLCV paths** to window close.
From `sample_count` paths over the remaining horizon τ you get:

1. **A causal forecast of remaining-window realized vol** = dispersion of the sampled paths. This is the
   quantity the trade actually cares about. The EWMA is a backward-looking vol *level* that doesn't even know
   τ; Kronos conditions on the recent K-line pattern and naturally respects time-to-close — exactly the axis
   that matters in the far/late bucket (the only one that clears).
2. **The digital fair value `P(close>K)`** by counting paths that finish in-the-money — from the *same* model
   that produced the vol, instead of plugging a separate σ into N(d₂). One internally-consistent object.

Why this is plausibly better than EWMA: EWMA misses regime shifts, vol seasonality, and τ-shrinkage; a
path model trained on crypto K-lines captures all three. Why it's still only *modest*: the signal sign is
already correct, so Kronos buys **magnitude/timing precision and fewer false entries**, not a sign flip.

## Gates (in order) — or this is A1.7 round two

1. **Static hedge FIRST. Kronos cannot rescue the banded-hedge loser.** Turnover (9.56c) swamps the edge
   (0.72c); a better vol number does nothing about turnover. Build the static-hedge Strategy-A variant
   (already an open Codex task — it needs no Kronos), confirm a surviving margin, *then* ask whether a
   sharper vol input widens it. If static hedge doesn't clear, Kronos is moot.
2. **Beat the dumb baselines OOS on net-of-cost PnL, non-overlap.** EWMA is the **negative control**. Insert a
   cheap middle rung — **HAR-RV / GARCH**, the standard causal forward-vol forecasters, interpretable and
   ~free. The real idea is "EWMA → forward vol"; Kronos is one heavyweight way, HAR-RV another. Kronos must
   beat **HAR-RV too**, on *trade PnL* (not RMSE/accuracy), to justify a 100M-param transformer. Briola
   caveat: a great vol forecast can still lose net-of-cost.
3. **Tail-calibration check in far/late, |z|≥1 specifically.** The signal lives entirely in the tails, and
   A1.7's exact failure was NN calibration breaking in the tails ([[block_a17_lightgbm_findings]]). Average
   calibration is not enough — verify Kronos's path dispersion is calibrated in the one regime that clears.
   This is the single highest-risk failure mode and it is the same one that killed the last ML attempt.

## Concrete test to run (Codex job)

Bake-off on the existing K6 panel, no new capture:

- **Inputs:** `data/analysis/k6_vol_gap_panel.parquet` + the K3 Binance path panel for path construction.
- **Vol estimators (3 arms):** (A) causal EWMA = control; (B) HAR-RV (and/or GARCH); (C) Kronos forward-vol
  (finetuned on the K3 Binance crypto K-lines, 1–5m bars, 512-bar context covers a 4h window; `sample_count`
  high enough for stable tail quantiles). All strictly causal — only data available at `t`.
- **Trade:** Strategy-A digital vol harvest with a **STATIC** hedge (no banded/continuous rebalancing).
  Hold to resolution. PM taker fee at entry only.
- **Eval:** net-of-cost PnL, **non-overlap**, CI bars on every headline. Bucket by |z| × τ; report far/late
  separately. Calibration diagram per arm, restricted to far/late |z|≥1.
- **Decision gate:** arm C only "wins" if it beats both A and B on net-of-cost lower-CI in far/late AND is
  tail-calibrated there. Otherwise document and keep EWMA/HAR-RV.

## Hermes — copytrade ops only (appendix)

Nous self-improving agent: serverless backends (Modal/Daytona hibernate-when-idle, $5 VPS), messaging
gateway (Telegram/Slack/…), cron scheduler, subagents, MCP. Maps to the **copytrade exec TODO**, not research:

- Serverless/VPS in a non-blocked region → the "unattended operation" line item.
- Messaging gateway → the operator-confirm smoke loop (`MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`)
  — approve each real-money fill from Telegram.
- Cron + delivery → recurring leader audits, scheduled captures, daily PnL/markout reports.
- Subagents/RPC → multi-leader cohort orchestration; MCP → wrap Gamma/RTDS/midas as tools.

**Caveat:** redundant with midas (execution skeleton) + brain/handoff/skills + Codex/Cowork. Borrow the
*patterns* (serverless deploy, operator-confirm-via-Telegram, cron) — do not adopt wholesale, and do not
mistake its "learning loop" for anything beyond your existing handoff discipline. Zero alpha.

## Routing (who gets this)

- **This note → commit it; it's for the other Cowork chats** (orientation/strategy). Linked into the Block K
  cluster via the wikilinks above.
- **The actual Kronos bake-off → Codex.** It's long-running computation + transformer finetuning + backtests
  = a Codex job by the COWORK.md split. Cowork drafts the prompt; Codex runs it and writes
  `notes/options_delta/block_k6_kronos_vol_bakeoff_findings.md`. A ready-to-paste Codex prompt accompanies this note in chat.
- **Sequencing reminder:** Kronos is gated behind the static-hedge Strategy-A backtest (already queued for
  Codex). Don't start the bake-off until static hedge clears — otherwise you're tuning the vol input on a
  strategy that loses on turnover regardless.
