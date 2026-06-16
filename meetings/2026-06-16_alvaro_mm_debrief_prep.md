---
title: "Meeting Prep — Alvaro MM Debrief (2026-06-16)"
created: 2026-06-15
updated: 2026-06-15
status: prep
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - COWORK
tags:
  - meeting
  - market-making
  - prep
  - presentation
---

# Meeting Prep — Alvaro MM Debrief

> Hubs: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · deep-dive teaching doc: [[mm_concepts_and_strategy_buildup]]
> Purpose: a present-from document for the debrief — what the research found, the paths forward, and the work left before we deploy.

## Plain-English Summary

- **Goal of the meeting:** get on the same page about the market-making (MM) research, then decide the path forward and split the work.
- **Present it in two halves.** Part 1 = *the research that's been done* (walk Alvaro through it, then discuss). Part 2 = *steps forward* — the paths we have and the work left to test **before deployment**.
- **The headline:** building our *own* single-venue maker is robustly closed; real maker wallets *are* profitable (K5) but winner-take-most; the one live monetization candidate is **politics NegRisk neutral MM** (+2,290 bps historically), and it's a **measurement loop we have not switched on yet**. We are only collecting data right now, and that capture lapsed Jun 12.
- **The one honest caveat to lead with:** the number that decides whether our own bot works — the passive two-sided fill rate net of adverse selection — is a **live unknown we have not measured.** That's an argument for running a small live loop, not for assuming the answer.

---

# PART 1 — The research that's been done (walk through, then discuss)

**Open with the one-liner.** "We pivoted from betting direction to *providing liquidity* — being the market maker. We tested it ~10 ways. Building our own single-venue maker is dead for a clear structural reason, but real makers *are* profitable and we now know their playbook. The catch is capacity, and the one thing that decides whether *we* can do it is only measurable live."

**The arc in five beats (this is the story):**

1. **The trade idea.** Quote both sides, earn `spread + rebate − adverse selection − inventory/resolution risk`. Polymarket *pays* makers and *charges* takers, so liquidity provision can be net-positive before any view on direction.
2. **Our own maker died — robustly.** Classic Avellaneda-Stoikov inventory quoting (K2) lost **−1,126 bps**; the defensive version (K2v2) **−4,316 bps** with the defense firing <0.1% of the time; the optimized chaser (K-PEG) looked like +759 bps but that was a **mark-to-mid mirage** (−753 on a real exit). Two lessons: **mark-to-mid is not money** (the exit eats it), and **adverse selection here is structural, not a latency race you can dodge.**
3. **But real makers profit (K5, model-free).** We measured the *realized* PnL of real maker wallets (defined structurally as ≥70% maker-share, ≥1,000 fills — not cherry-picked). Crypto-4h **+171 bps, CI [34, 327]**; profitable across most categories (politics +443, other +147). Their **playbook**: 64% two-sided, 78.8% carry-to-resolution, 0.8% in the late spike zone.
4. **The capacity reality.** Top-3 wallets capture **~95%** of per-market profit; the moat is **capital/scale + structure, not speed** (so Python is fine, Rust deferred). Stress-tested, the *typical* maker is ≈ breakeven; only **structured non-top3** flow in a few categories clears. Standalone, the deployable run-rate is small: **~$78/active day**, ~90% in one grab-bag cell.
5. **So MM consolidates into an execution layer + one live bet.** The durable value is the *lifecycle* (slow-market selection + two-sided + carry + spike-avoidance + non-incumbent cells). The one place with both a positive setup and real size is **politics NegRisk** (+2,290 bps, CI [1,020, 3,621]; flow $381.1M, active 146/146 days).

**What's robustly closed — say "let's not relitigate these":** single-venue crypto neutral MM (structural adverse selection, directly tested); K-PEG chase (mark-to-mid); the NegRisk **basket-consistency arb** (real but a ~4-second latency game, net-negative after fees on ~98% of episodes); continuous-hedge gamma scalp; and PM *terminal pricing* (efficient net-of-cost across crypto and equities).

**The two honest caveats to put on the table (don't hide these):**
- **The one-sided-fill assumption is unmeasured.** Our losing sims were *single-leg* (enter, then cross to exit) — they baked in an exit cost a continuous two-sided maker never pays. The skilled K5 cohort's realized fills are actually maker-*favorable* (+616 bps at 60s). So "you'll be forced to carry / get picked off" is an assumption; inventory control is exactly the response to one-sided flow, and our true fill rate is a **live unknown**.
- **The size is structurally small and right-tailed.** Politics capacity: mean EV/day ≈ $1.5k / $6k / $29.9k at 0.25% / 1% / 5% capture, but **median-wallet EV/day is only $9 / $38 / $189**. The +2,290 bps is what *historical* wallets earned — not yet proof we get the fills.

**Pointer:** the full teaching doc — every concept (queue, adverse selection, fees, mark-to-mid vs realizable, A-S, NegRisk redemption), a one-line glossary of K1–K7, the K5 methodology, and the realism bottleneck — is [[mm_concepts_and_strategy_buildup]]. Send Alvaro this before the meeting if he wants depth.

---

# PART 2 — Steps forward: the paths, and the work left before deployment

## 2.1 The strategic fork (the real decision)

**Two of these are market-making; one is not — keep them separate.**

| path | what it is | MM? | what the evidence says |
|---|---|---|---|
| **A — our own neutral-MM bot** (politics-first) | run a two-sided passive maker in slow NegRisk markets | **yes** | **your preferred direction.** Legitimate but *unproven* — lives in the method's novelty blind spot (history can't validate an edge nobody runs yet); settled only live |
| **C — MM as execution layer for OD** | MM supplies the lifecycle, OD the (crypto) signal | **yes** | "two layers of one strategy"; OD standalone is closed, so this is a crypto-side variant of A |
| **B — copy the directional wallets** | mirror profitable wallets' *picks* via the Midas copytrade pipeline | **no — copytrade** | a *spinoff* of the MM research, not MM (see 2.1b); you've already deprioritized copytrade |

### 2.1b Why "copy the winners" keeps coming up — and why it isn't MM

When we decomposed the profitable "maker" wallets (K5), their edge turned out to be **directional, not neutral spread-capture**: the clean neutral subset is empty/negative across sports, residual-misc, equities, and politics; the profit sits in `two_sided_directional`/`mixed` wallets that take a view and happen to use maker orders ([[mm_structural_maker_directional_decomposition_findings]]). That forks into two *different* strategies with two *different* homes:

- **Copy their directional picks** → the pick is the copyable alpha, the maker wrapper is incidental → mirror via the existing Midas watcher→signal→mirror pipeline. **This is copytrade, not MM.**
- **Be the neutral maker nobody is** → capture the spread the directional players leave → this **is** MM, and it's exactly path A / the live loop.

So "copy directional wallets" only appears in the MM thread because that's where we *discovered* those wallets. It also means part of K5's "+443 bps politics" is **directional, not neutral** — so the neutral-MM edge is unproven, which is why the live loop is a **novelty test, not a reproduce-the-winners test.** For this debrief: treat the MM decision as **A vs C**, and hand B to copytrade (or set it aside, since you've deprioritized it).

**The tension to name:** the repo's standing lean was B/C; you lean A (the genuine MM bet). The cheap, decisive move is the same regardless — **run the Phase-2 measurement loop** (2.2): it validates path A or tells us the neutral-maker fills aren't there.

## 2.2 The one experiment that adjudicates everything

**Politics NegRisk Phase-2 live measurement loop** — already designed and pre-registered ([[mm_politics_negrisk_live_loop_design]]):
- One market at a time, **1 contract, ~$30 notional**, observation-grade telemetry.
- **Pre-registered gates (decide scale only after these):** fill share > 0% in ≥5 markets; post-fill 60s drift lower-CI > −500 bps (vs crypto's −1,886); news-proximate adverse fills < 50%; net-of-cost PnL lower-CI > 0; resolution drag < 10%.
- **Sample floor:** ≥ 30 settled markets or 90 days.
- This *is* the direct live test of classic neutral inventory MM in a slow market — i.e., path A's whole thesis.

## 2.3 Work left before deployment (the checklist)

| # | task | status / blocker | owner? |
|---|---|---|---|
| 1 | **Restart the live capture** (it's lapsed since Jun 12) | one command; runs ~16h on an awake Mac — fragile, candidate to automate (VPS/cron) | |
| 2 | **Analyze the first-mover capture we already have** (~9 days, 3 lanes, unanalyzed) | compute only; report book shape, spread/depth, adverse-selection reconstruction (public L2 is anonymous → report clean-vs-ambiguous rates) | |
| 3 | **Add the missing maker telemetry** before any scale | infra is "measurement-grade," missing: true queue position, side-adjusted drift bps, volume-weighted fill share, book depth at quote/fill, quote age, cancel latency | |
| 4 | **Wire + run the Phase-2 1-contract smoke** on a screened politics market | needs a market that passes the 5-screen filter; infra (256 tests) is no longer the blocker | |
| 5 | **Read out the gates over ≥30 settled markets**, then a scale (Phase-3) decision | blocked on 1–4 | |

**Cheap parallel introspection win (no new data):** resolve the brain's internal contradiction on the dali reversion signal — one note says "dead across all framings / ~0.1% passive fill," another flags three *never-run* framings (continuous rolling-rank sizing, ask-depletion fade side, fill-probability-gated passive). A few hours of compute; either closes it cleanly or surfaces the one framing that isn't just a fill-rate question. Introspection, not a new bet.

**Do NOT build (stay-closed):** single-venue crypto neutral MM; K-PEG-style chasing; the basket-consistency arb; continuous/banded gamma scalp; any PM terminal-pricing strategy. **And for the politics bot specifically: OD is not needed (no vol surface for "will X be confirmed"), and you don't hedge — the balanced NegRisk basket neutralizes structurally.**

## 2.4 How to split the work (pick one main seam)

- **By layer (recommended):** one owns research/gates/analysis (which market passes, adverse-selection measurement, reading the loop); the other owns execution/ops (maker engine, telemetry, daily capture, the 1-contract smoke). Clean interface: research hands execution a market + frozen params; execution hands back telemetry.
- **By thread:** one owns the politics NegRisk loop (path A's test), the other owns first-mover Stage-1 capture + analysis; shared infra.
- **Regardless:** assign **one owner to "keep the capture alive + automate it."** It's the most fragile, most ownable piece, and it's currently lapsed.

## 2.5 Decisions to make in the room

1. **Path A vs B/C** — or formally, "we run the Phase-2 loop next and let it decide." Agree the framing.
2. **Deploy real $30 / 1 contract, or keep it paper** for the first loop?
3. **Automate the capture** (VPS/cron) so it stops lapsing? Who owns ops?
4. **Telemetry before or after** the first smoke?
5. **Is MM worth two people** vs the live crypto momentum book (2.24 Sharpe, under overfitting review) and copytrade? Honest portfolio question given the small median EV.

---

# Reference (have open during the meeting)

## Data audit — what we actually have

**Historical base (the 9-month model-free dataset):** `data/trades` 43 GB, `closed_positions.parquet` 28 GB, `bankroll_timeseries.parquet` 12 GB, `traders.parquet` 570 MB, `markets/` 197 MB. This is the K5 / accounting base.

**Live CLOB capture (`data/live_clob`, ~71 GB total). The three MM lanes:**

| lane (market scope) | size | shards | days |
|---|---|---|---|
| `mm_stage1_first_mover` (new sports/other/geopolitics) | 17 GB | 199 | Jun 4–13 |
| `mm_stage1_broad_live` (crypto-fast + equities + sports) | 24 GB | 195 | Jun 5–13 |
| `mm_stage1_slow_crypto_finance` (crypto-4h/daily + finance) | 23 GB | 195 | Jun 5–13 |

≈ **64 GB / ~590 shards over ~9–10 days**, **observation-only (no orders).** **Lapsed since Jun 12** — nothing capturing now; Jun 13/14/15 missing. This feeds the first-mover branch + diagnostics, *not* the politics loop.

## Restart the capture (the manual command)

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/mm_stage1_live_control.py start --duration-hours 16
```
`status` for health, `stop` to end. It launches the 3 lanes in `screen` + `caffeinate` (Mac must stay awake — hence the lapses and the case for automating it).

## What's already built (execution infra)

`polymarket/execution/maker/`: `maker_engine.py`, `negrisk_inventory.py`, `resolution_handler.py`, `event_calendar.py`, NegRisk-aware signing. **Measurement-grade; 67 maker + 256 full execution tests pass.** Missing for production telemetry: queue position, side-adjusted drift, fill share, book depth at quote/fill, quote age, cancel latency. ([[mm_maker_infra_audit_findings]])

## Reading list (priority order)

1. [[mm_concepts_and_strategy_buildup]] — the full teaching doc (concepts + block glossary + K5 method + build-up + audit). **Start here.**
2. [[strat_market_making]] — MM hub / live status.
3. `brain/TODO.md` § MM — authoritative open tasks.
4. [[mm_politics_negrisk_live_loop_design]] — the Phase-2 plan + gates.
5. [[mm_politics_negrisk_accounting_findings]] — the +2,290 bps headline + capacity ledger.
6. [[2026-06-04_state_of_the_arc_and_novelty_frontier]] — the one-page high-level map.

## Source notes

Evidence: [[block_k5_findings]] · [[block_k5b_findings]] · [[block_k5_stress_findings]] · [[mm_deployable_cells_findings]] · [[mm_negrisk_consistency_scanner_findings]] · [[mm_first_mover_liquidity_scope_findings]] · [[mm_maker_infra_audit_findings]] · [[mm_clob_capture_semantics]]. Sibling/optional: [[strat_options_delta]].
