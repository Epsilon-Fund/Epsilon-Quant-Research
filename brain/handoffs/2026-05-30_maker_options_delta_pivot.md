---
title: "Handoff — Maker / Options-Delta Pivot (new research thread)"
tags: [handoff, dali, maker, avellaneda-stoikov, options, delta, block-k]
created: 2026-05-30
status: kickoff — backed by deep-research report (notes/overview/synthesis/block_k_maker_options_research.md)
purpose: Start a fresh Cowork thread on treating Polymarket binaries as digital options and pivoting microstructure work into maker/liquidity-provision, where signals skew quotes instead of triggering takes.
relationship: The local directional-continuation branch is falsified (P3' OOS + A0c holdout retest, both negative — see Findings Update below). This thread is the redesign path into Block K. Do NOT re-run directional microstructure backtests here.
---

# Handoff — Maker / Options-Delta Pivot (Block K)

## Read first (orientation, ~5 min)

- **`notes/overview/synthesis/block_k_maker_options_research.md` — the deep-research report. START HERE.** Academic foundations
  (A-S/GLFT, Glosten-Milgrom, digital-option Greeks, Hasbrouck), the literature for each strategy class, a
  synthesis table mapping every class to our closed findings, and four pre-scoped first validation tests on
  data we already own. This handoff is the *why/orientation*; that report is the *substance*.
- `brain/COWORK.md` — repo conventions, where things live, Codex-vs-Cowork split. Obey it.
- `brain/TODO.md` — live task list. The direct dali local microstructure branch is falsified/redesigned; copytrade is primary by default.
- This doc — the *why* and *first steps* for the new thread. The repo state is authoritative for what's done.

Do not relitigate the directional-continuation branch. It was falsified honestly across all tiers and both
execution modes (see `notes/block_a14*` … `block_a17_*`, and `notes/dali/block_a1x_external_note_reconciliation.md`).
**This thread takes that rejection as a given and redesigns forward from it.**

## FINDINGS UPDATE — 2026-05-30 (post P3' OOS + A0c holdout retest) — READ THIS

The last two retests rejected the direct local signal at the deepest possible level. See
`notes/dali/block_p3prime_oos_findings.md` and `notes/dali/block_a0c_holdout_retest_findings.md`.

- **P3' OOS reversion replication:** fails. 0/grid cells clear CI>0 / n≥30 / fill≥2%.
- **A0c Retest A (binary-bet / 4h-boundary, 6 BTC windows):** failed OOS, decisively (pooled fixed-300s
  −2,341 to −2,831 bps; boundary-hold n=6, CIs span ±10,000s of bps = noise). The binary-bet thesis is dead
  *with adequate windows*, not merely underpowered.
- **A0c Retest B (deep-book fade):** artifact-confirmed.
- **A0c Retest C (the key one):** the A1.3 **73.7% TOB hit rate does NOT replicate OOS — it falls to 36.0%
  pooled** (−37.7pp), 25–46% per window. The directional accuracy itself was in-sample. **"The signal is
  real" is no longer a safe premise.**

**What this means for THIS thread — two adjustments, not a cancellation:**

1. **The directional signal's "second life as a quote-skew input" must be re-cast as CONTRARIAN.** 36% hit
   on the *extreme* decile is well below coin-flip, which is the mean-reversion signature: extreme OFI/TOB
   predicts price moving the *opposite* way. So if a signal-skew term is used at all, lean quotes *against*
   the extreme (toward reversion), and treat even that as a hypothesis to be OOS-validated here, not a given.
   Do not assume any momentum lean. Realistically, **start the maker model with NO directional skew** and
   prove the rebate+spread baseline first; only add a (contrarian) skew if it earns incremental edge OOS.
2. **The maker thesis must now stand on liquidity-provision economics alone** (rebate + spread − adverse
   selection − inventory/resolution risk) plus, for crypto, the **external delta/option mispricing** — NOT
   on any local Polymarket directional alpha, which is now fully foreclosed. The adverse-selection
   measurement in step 1 below is therefore the whole ballgame, and the external-reference (Binance) edge is
   the only directional input with any surviving credibility (and even it is unproven — that's prompt P6).

## BLOCK K RESULTS — 2026-05-31 (READ THIS FIRST; supersedes the "first steps" below)

The validation tests below were run (K1–K6 + K-PEG + audits). Full plain-English synthesis with glossary:
[[block_k_plain_english_synthesis]] (Obsidian hub linking every Block K note). Per-block notes are
authoritative for numbers — key ones: [[block_k5_findings]], [[block_k5b_findings]], [[block_k6_vol_findings]],
[[block_kpeg_robustness_findings]], [[block_k2v3_findings]], [[block_k2v2_findings]]. Summary:

**Single-venue Polymarket maker is CLOSED across every anchor** — mid (K2: −1,126 bps), Binance/digital
(K2v3: 0/681 buckets clear, anchor *raised* adverse selection 325 vs 145 bps), and defensive (K2v2: −4,316
bps; the pull/widen defense fired <0.1% → **adverse selection is structural, not a dodgeable latency race**).
K-PEG's +759 bps was a **mark-to-mid artifact** (no lookahead/bug — we + an independent Codex audit
confirmed and reproduced it; realizable round-trip is −753, maker-exit −569 at ~12% fill). BUT the shape
analysis proved the **entry reversion alpha is real and broad-based** (79% win, survives dropping the best
5% of fills) — the entire loss is *exit cost*, not a bad signal. K3: 4h has no anti-arb fee, Binance leads
~10s, raw basis thin. K4: intra-PM arb ~zero. K6: Polymarket **overprices vol** (+3.7 pts avg, +24 far/late)
but continuous-delta-hedged gamma scalp loses on hedge cost — **static hedge untested.**

**K5 is the breakthrough (model-free, closed-position realized PnL):** real maker-heavy wallets ARE
profitable — crypto-4h **+171 bps, CI [34, 327]** (clears), 256 wallets; pooled top-maker +145 bps CI
[85, 210]. The winners' playbook: **64% two-sided, 78.8% carry-to-resolution, 0.8% in the late near-50¢
spike zone** — exactly the lifecycle our single-leg sims couldn't represent (hence the sims were too harsh).
**Capacity warning: top-3 wallets capture ~95% of positive crypto-4h profit per market** (winner-take-most).
Geopolitics negative (no rebate). Survivorship caveat: closed positions only.

**Where this leaves the two tracks of the original thesis below:** Track A is *validated in structure* by
K5 and re-specified — passive entry + hold-to-resolution + **static** external hedge (continuous hedging is
dead, K6). Track B (no-underlying pure rebate) is effectively closed (geopolitics negative). The new
Strategy B is "copy/learn the profitable makers" (re-merges with copytrade). The original "first steps"
below are historical — done. Live next steps: (1) decompose why top-3 makers dominate (speed vs capital vs
carry-risk → tells us if the moat is speed/Rust or structure); (2) backtest the winners' playbook as
Strategy A, OOS-gated. Do not spawn new single-venue quoting/anchor/continuous-hedge variants.

## The one-sentence thesis

Stop trading the signal; **quote around it.** Polymarket pays makers and charges takers, so liquidity
provision can be net-positive *before any directional edge*, and the microstructure signals that failed as
triggers (rolling-rank OFI, TOB imbalance) get a second life as **quote-skew inputs**. For crypto up/down
markets specifically, the contract literally *is* a digital option on BTC/ETH/SOL, so you can also price its
**delta** against the external spot/perp venue and hedge direction out — earning the maker rebate + spread
from Polymarket's uninformed flow while neutralizing the adverse-selection/direction risk that killed the
taker thesis.

Two framings that merge into one strategy on crypto markets:

1. **Maker / liquidity provision (Avellaneda-Stoikov).** Closed-form optimal quoting around a reservation
   price that skews with inventory. Microstructure signal → which side to lean. Midas is already a passive
   quoter, so infra is mostly built.
2. **Polymarket-as-option / delta.** `btc-updown-4h-*` is a 4-hour cash-or-nothing binary struck at the
   window-open price. Given external spot + vol + time-to-close you can compute a model fair value and a
   delta (∂price/∂BTC). Polymarket price − model price = a relative-value basis; delta lets you hedge on
   Binance/OKX. This is the cross-platform (Block I) lead-lag work sharpened into an options-pricing lens.

The merged strategy: **market-make the Polymarket binary while delta-hedging the external underlying.**
Positive baseline economics from rebates + spread; direction hedged out; signal skews the quotes. That is
the thing this whole project has been missing — an edge that doesn't require predicting Polymarket's own
next tick.

## Why this survives when the scalp didn't

The taker/maker-at-mid program died because the local OFI/TOB signal predicts **mean-reversion to
micro-price (fair value)**, and that target sits *inside* the spread you must cross. Two structural exits
from that trap:

- **Maker, not taker:** you don't cross the spread, you *earn* it (plus rebate). The signal only needs to
  beat a coin flip on which side to skew — a 55% signal is useful here, where it was worthless as a trigger.
- **External reference, not self-reference:** delta vs Binance is an off-book edge ("Polymarket lags the
  underlying"), immune to the local-reversion diagnosis.

## The hard parts — center these, don't hand-wave them

This is not free money. The make-or-break questions, in order:

1. **Adverse selection is the killer, not inventory.** `notes/block_a14c/a14h` already found: maker fill
   rate is *flow-capped* (collapsed 9.0% → 0.2% under non-overlap), and the one filled cell ate ~248 bps of
   5s adverse selection. The entire thesis reduces to: **is (rebate + spread capture) > (adverse selection +
   inventory/resolution risk)?** Measure adverse selection FIRST, on data we already own, before building a
   quoter. If fills are mostly toxic, the rebate doesn't save you (Glosten-Milgrom / Kyle).
2. **A-S assumes unbounded Gaussian mid dynamics; Polymarket prices live in [0,1]** with absorbing behavior
   near 0/1 and a 0/1 jump at resolution. The vanilla A-S reservation-price/quote formulas need a
   bounded-price adaptation (logit-space, or a model respecting the resolution barrier). This is the open
   research gap already logged in `brain/TODO.md` ("Avellaneda-Stoikov adaptation for [0,1] bounded prices").
3. **Resolution risk is a discontinuity.** Holding inventory into a binary resolution is a 0/1 jump. Either
   flatten before close or hedge (complement leg, or external delta hedge for crypto). Quoting policy must be
   time-to-resolution aware.
4. **Fill rate / market selection.** Polymarket has slow takers despite deep books. Concentrate on the
   trade-rich markets (A0c crypto-4h roll, daily crypto, Hormuz-class geopolitics); avoid quote-noise markets
   (e.g. JD Vance: 1.35M price_change / 138 trades — useless).

## Two tracks (they have different risk and shouldn't be conflated)

- **Track A — crypto-4h maker + external delta hedge.** Has a real hedgeable underlying → can neutralize
  direction. Cleanest test of the merged thesis. Start here.
- **Track B — pure maker rebate on no-underlying markets** (geopolitics/sports). No external hedge; you're
  purely betting that flow is uninformed enough that rebate + spread > adverse selection. Higher risk,
  simpler infra. Parallel, lower priority.

## Concrete first steps (validation before infrastructure)

**Canonical version of these tests lives in `notes/overview/synthesis/block_k_maker_options_research.md` → "Recommended first
validation tests."** They run on data we already own (A0/A0b/A0c); no new capture. Summary + the refinements
the deep research added:

1. **Maker-economics decomposition — THE GATE.** Per market/category, reconstruct passive fills (A1.4h fill
   proxy) and compute `rebate + spread capture − adverse selection(5/30/60s) − inventory/resolution risk`.
   Use the *confirmed* fee formula: maker fee 0; rebate = {20% crypto, 25% other fee-enabled, **0%
   geopolitics**} of taker fee `C·feeRate·p·(1−p)`. **Refinement:** geopolitics has no rebate cushion, so
   split the table fee-enabled vs fee-free and expect fee-free to be a negative control. Output: is baseline
   maker PnL positive *before any skew*? Nothing gets built until this is positive somewhere.
2. **Logit-space A-S sim** (formulas from arXiv:2510.15205 — the bounded-[0,1] adaptation; cited in the
   report). Run with **zero directional skew first**; add a τ-and-distance-to-50¢ widening term for the
   crypto gamma/delta-spike zone; then add a *small contrarian* skew (signal is 36% OOS = reversion) and
   measure incremental PnL over baseline, OOS on A0c only.
3. **Crypto-4h digital-basis + fee check.** (a) Confirm whether 4h crypto markets carry the dynamic anti-arb
   taker fee that already killed the 15-min lag-arb. (b) N(d₂) fair value from Binance + window realized vol
   vs Polymarket mid; measure lead-lag and post-fee survival. Folds in prompt P6.
4. **Combinatorial-arb scan (parallel, high-credibility).** Scan owned captures for YES+NO≠$1 and
   logically-linked-market violations (Saguillo 2025 documents ~$40M of this, model-free). Quantify frequency
   × size × whether our latency could capture it. Sidesteps adverse selection entirely.

**Sequencing:** Test 1 first (gates 2); run 3 and 4 in parallel. **Decision gate:** net maker PnL > 0 on ≥1
category before skew → proceed to logit-space sim, then paper-quoting smoke on Midas. If 1 is negative
everywhere → document and fall back to copytrade.

## What to reuse (don't rebuild)

- **Midas** passive quoter (already a maker) — the execution skeleton.
- OFI / TOB / rolling-rank signal infra (P1 sidecar `scripts/dali_block_p1_rollingrank.py`) — now an input,
  not a trigger.
- Capture + replay + non-overlap backtest pipeline.
- The A1.4h fill proxy + adverse-selection windows.
- Cross-market lead-lag scoping (prompt P6) folds into step 2.

## Guardrails (from the brain — enforce)

- Net-of-cost, non-overlap, CI bars on every headline. Overlap math is treacherous (it manufactured every
  prior false positive).
- No ML before a rule-based maker edge exists (Briola caveat: 70%+ accuracy still lost money net of cost).
- Validate signal/economics before building infrastructure. Fastest path to first dollar.
- Forecasting accuracy ≠ net-of-cost trading profit. The maker version restates this as: high fill volume ≠
  profit if fills are adversely selected.

## Key references

- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book* — the quoting framework.
- Stoikov (2018), *The Micro-Price* — reservation-price anchor where there's no external underlying.
- Glosten & Milgrom (1985); Kyle (1985) — why making for uninformed flow pays and making for informed flow
  doesn't. The adverse-selection foundation.
- Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading* — textbook A-S with inventory +
  adverse selection + practical extensions.
- Digital / cash-or-nothing option pricing + barrier intuition for the resolution jump.
- Repo: `notes/dali/block_a14c_maker_at_mid_findings.md`, `notes/dali/block_a14h_maker_non_overlap_findings.md`
  (the maker headwinds), `notes/dali/block_a1x_external_note_reconciliation.md` (signal-as-input framing),
  `brain/handoffs/2026-05-27_cowork_transition.md` (maker fee asymmetry + A-S gap).

## Suggested identifier

Call this **Block K — Maker / Options-Delta**. It absorbs the deferred maker thesis and the Block I
lead-lag thread under one roof. Independent of the A0c retest + Optuna falsifier currently running; if
either of those reopens a *directional* edge, it becomes an even better quote-skew input here.
