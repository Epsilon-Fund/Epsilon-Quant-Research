---
title: "Handoff — Politics NegRisk live measurement loop (design + next-chat kickoff)"
tags: [handoff, mm, politics, negrisk, live-loop, market-making, adverse-selection, kickoff]
created: 2026-06-03
status: kickoff — design agreed in Cowork; next chat discusses politics NegRisk markets and drafts the build/measurement path. No live code shipped yet.
purpose: >
  Hand a fresh chat the full design for the Polymarket politics NegRisk live MEASUREMENT loop:
  a structured non-top3 passive maker, measurement-first, anchored on the accounting-confirmed
  politics_negrisk cell. This is the one proven, standing, EXECUTION-side edge left after the
  whole OD/MM/dali arc converged — OD-as-pricing is closed across crypto and equities; the
  durable edge is structural liquidity provision, not forecasting.
relationship: >
  Sibling/anchor doc: [[mm_politics_negrisk_accounting_findings]] (the 2,290 bps cell + persistence).
  Hub: [[strat_market_making]]. The OD arc that converged into MM: [[strat_options_delta]] and
  [[od_strategy_a_realism_reaudit_findings]]. Methodology spine: [[CODEX]] § Realism calibration and
  [[COWORK]] § The research loop.
---

# Handoff — Politics NegRisk live measurement loop

## You are a fresh Cowork chat. Read these first, in order (Obsidian wikilinks)

Orientation (cheap, do these):
1. [[COWORK]] — repo conventions, Cowork-vs-Codex split, the research loop + its live terminus. Obey it.
2. [[CODEX]] — what any Codex prompt you draft must tell Codex to read/obey; **especially § Realism calibration** (statistical vs economic, borrowed baselines, capacity-as-assumption, assumption-vs-live ledger, the reopen filter).
3. [[TODO]] § MM and § copytrade — live state. The MM standalone line is sub-scale; politics_negrisk is the one cell that cleared to a live-measurement verdict. copytrade is the shared live plumbing.
4. [[strat_market_making]] — the MM hub.

The proven edge + why this loop exists:
5. [[mm_politics_negrisk_accounting_findings]] — **START HERE for the edge.** The structured non-top3 politics maker cell survived full NegRisk merge/split/redeem accounting (125,937/125,937 receipts, 0 missing) and *strengthened* to **2,290 bps, CI [1,020, 3,621]**, median wallet **14.5 bps**, non-rebate. Persistence cut: flow is recurring (146/146 days), edge survives in settled NON-election buckets (Non-US elections +3,673, Trump personnel +1,018, Other politics +1,046). Honest deployable: **$0.4k–$1.6k/month at 1% capture** median; the $40–170k/mo mean is fat-right-tail to be earned live.
6. [[block_k5_findings]], [[block_k5_stress_findings]], [[block_k5b_findings]], [[mm_deployable_cells_findings]] — the structured-non-top3 playbook, the de-biased gate, and the "moat is capital/structure not speed" finding.
7. [[block_kpeg_findings]] — the load-bearing lesson: the maker ENTRY alpha is real (79% win), the *entire* loss was exit cost → carry-to-resolution dodges the exit-spread tax.

Why the rest of the arc converged here (so you do NOT relitigate pricing):
8. [[od_strategy_a_realism_reaudit_findings]] — OD-as-standalone is closed three independent ways (source-vs-valuation −40.47c, passive-only survival, pure-taker not clearing). OD lives on only as sizing/selection inside the MM lifecycle. The crypto longshot harvest + touch-fade are tiny secondary sleeves, not the anchor.
9. [[od_same_day_crypto_pricing_gate_findings]], [[od_equities_index_pricing_scope_findings]] — PM financial-binary **pricing is efficient net-of-cost across crypto AND equities**. Do not propose a pricing/forecasting strategy here. The edge is execution.

## What this loop is (one paragraph)

A **structured non-top3 passive maker on politics NegRisk markets, measurement-first**: two-sided passive quoting, carry-to-resolution, spike-zone avoidance, non-incumbent cell selection, correct NegRisk basket accounting. It rides the copytrade plumbing (Midas exec, Polymarket creds, the `MAX_REAL_ORDERS`/operator-confirm safety harness). It is a **measurement loop to test one hypothesis, not a trading-system build**, and it must be able to kill cheaply.

## The central hypothesis the loop tests

Single-venue crypto MM died because adverse selection was **structural and faster than we could react** (K2v2 defensive pull fired <0.1%). The politics cell shows a real structured-non-top3 edge in the data, but it is only deployable if **politics adverse selection is *dodgeable* in a way crypto's wasn't** — and it plausibly is, because politics moves on **discrete news on a minutes timescale**, not crypto's continuous sub-second flow. So the whole loop is one experiment: *can a non-incumbent dodge politics adverse selection well enough to keep the 2,290 bps?* If yes, it's real. If you get picked off on news as badly as crypto, the historical edge was a selection/survivorship artifact and it dies live. Everything below serves running that test cleanly.

## Design across four dimensions

### 1. Edge & cell selection
- Quote where the edge is **proven settled**, not where flow is biggest: **Non-US elections (+3,673 bps), Trump personnel/policy (+1,018), Other politics (+1,046)**. **Deprioritize 2028 US presidential outrights** — 36% of flow but zero settled PnL (forward-only) and multi-year capital lockup.
- Non-incumbent filter from the [[block_k5_stress_findings]] wallet-market cache: quote only markets with real non-top3 headroom (top-3 take ~95% where present).
- Uninformed-flow identification: reuse Block E wallet clustering + `traders_directionality.parquet` + the copytrade leader audits to find retail-directional markets (good — provide liquidity) vs specialist-dominated (bad — you become the adverse-selection sink). The politics top-3 makers likely overlap the audited copytrade leaders.

### 2. NegRisk machinery & exec (the hard prerequisite)
- A step up from the copytrade smoke, which explicitly AVOIDS NegRisk markets. New build on top of existing Midas/CLOB/safety-harness:
  - **Real-time composite-NegRisk inventory** — merge/split/redeem/convert change basket position WITHOUT trades; the executor must track effective exposure live. The offline decoder exists (CTF ERC1155 + USDC ERC20 from Polygon receipts, reconciled in [[mm_politics_negrisk_accounting_findings]]); the live version runs it near-real-time. This is a **moat, not just plumbing** — the $34.6M the conversion layer moved is value mis-accounting entrants give away.
  - **Resolution-handler path** — TODO flags it as needed before scaling; carry-to-resolution makes it mandatory.
- This accelerates the copytrade "v2 NegRisk handling" item — not orphaned effort.

### 3. Risk, sizing & resolution
- Measurement-scale first (1-contract / ~$10). **Median-not-mean sizing** (§ Realism calibration): size to the $0.4–1.6k/mo median, never the fat-tail mean. Event-aware: bigger on proven settled buckets, ~zero on forward-only outrights until live fills validate.
- Per-event and per-basket dollar caps for the left tail (a market resolving against carried inventory on news).
- **Carry vs resolution risk tension:** carry is the whole edge (dodges the K-PEG exit tax) but means eating UMA-dispute/ambiguous/delayed resolution. Mitigate by cell selection (prefer objective clean-resolution markets — connects to Block J resolution-criteria scanning and the weather_ftc resolution work), not by exit-before-resolution (which re-imports the exit tax). Log realized resolution outcomes vs expectation to QUANTIFY the resolution drag.

### 4. Adverse selection & news (the gap, and the crux)
- **Log the live-only unknowns** per fill/market/bucket: fill share by top-maker rank, queue position, cancels, missed fills, and **post-fill price drift tagged with whether a news event preceded it** (the key new field — it directly measures whether losses cluster on news = adverse selection vs random = clean provision).
- **Scheduled events are dodgeable** — a lightweight politics event calendar (debates, court dates, election days, policy announcements) lets you pull/widen quotes ahead of known events.
- **Unscheduled news is the real test.** Options for a news feed, cheapest first: (a) a scheduled-event calendar (covers the predictable majority); (b) a news/X (Twitter) API scrape or LLM news scanner ([[dali_literature_synthesis|Block J]]-style) for breaking events — note the X API is now paid/rate-limited, so treat a full firehose as heavy and gate it on whether the basic telemetry shows news-driven adverse selection is the binding loss. Do NOT build the heavy news pipeline before Phase 2 telemetry proves it's needed (validate-before-infra).

## Phased plan (don't over-build before the test)
- **Phase 0** — copytrade smoke (non-NegRisk) proves basic plumbing (already the live default in [[TODO]] § copytrade).
- **Phase 1** — build composite-NegRisk exec layer + resolution handler.
- **Phase 2** — measurement-only deployment on settled-bucket markets at 1-contract, telemetry-heavy, to answer "is politics adverse selection dodgeable + is our fill share real."
- **Phase 3** — scale within proven buckets ONLY if Phase 2 says adverse selection is dodgeable and fill share survives; otherwise CLOSE cleanly as a selection artifact.

## What the next chat should decide / do
1. Confirm the proven-bucket cell list and the non-incumbent + uninformed-flow selection rule (which exact markets to quote in Phase 2).
2. Scope the Phase-1 exec build (composite-NegRisk inventory + resolution handler) against what Midas already has — draft the Codex/exec prompt.
3. Pre-register the Phase-2 measurement success criteria: what fill share, post-fill-drift/news-tagged adverse-selection profile, and net-of-cost-per-resolved-market would say "dodgeable → scale" vs "picked-off → close."
4. Decide the news-feed approach (scheduled calendar first; X/LLM only if telemetry demands it).

## Codex prompt discipline (per [[COWORK]])
Any Codex prompt this chat drafts MUST open with the read-order preamble:
```markdown
Before doing anything else, read:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `brain/POLYMARKET_BRAIN.md`
5. `polymarket/research/notes/market_making/strat_market_making.md` (+ this loop's anchor note, mm_politics_negrisk_accounting_findings.md)
```
Prompts stay in chat (not committed). Outputs are `*_findings.md` under `polymarket/research/notes/market_making/` (research) or exec artifacts under `polymarket/execution/` (Midas), with wikilinks back to [[strat_market_making]] and a pointer added here. Update [[TODO]] § MM with verdicts. Enforce: lookahead-free, market-cluster CI, net-of-cost, no mark-to-mid, no infra before the measurement validates the dodgeability hypothesis.

## Guardrails carried from the arc
- This is EXECUTION, not pricing. Do not reopen a politics pricing/forecasting model — PM-binary pricing is efficient ([[od_same_day_crypto_pricing_gate_findings]], [[od_equities_index_pricing_scope_findings]]).
- It is a MEASUREMENT loop, not a trading-system build. Phase 2 must be able to close cheaply.
- Capacity is a live-only unknown (Rule 3): the $0.4–1.6k/mo is a median proxy from one historical capture, not a promise. Real fill share/queue is what Phase 2 measures.
- Honesty over momentum: if Phase 2 shows politics picks you off like crypto did, close it and say so.

## Parked / not this loop
- Crypto longshot harvest + same-day touch-fade: tiny secondary sleeves only; not the anchor.
- Equities SPX close-style (CONFIRM-CLOSE) and SPX-opens (MERITS-LIVE-COLLECTOR, low priority, wide/thin pre-open book) — do not let these pull focus from politics. See [[od_equities_index_pricing_scope_findings]].
- Touch-risk skip filter: LOG-FEATURE-ONLY telemetry, not a gate. See [[od_touch_risk_filter_findings]].
