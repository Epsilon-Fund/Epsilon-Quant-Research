---
title: "Handoff — OD neg-risk first-passage / barrier branch (kickoff for a fresh Cowork chat)"
tags: [handoff, od, options-delta, neg-risk, barrier, first-passage, ofi, kickoff]
created: 2026-06-02
status: kickoff — fresh branch. No code run yet. This chat's job is to read the brain, form an opinion, and DRAFT the Codex scoping prompt.
purpose: >
  Start a new OD sub-branch: price Polymarket crypto "will-hit-X" / price-band neg-risk markets as
  FIRST-PASSAGE / BARRIER claims using Binance path + jump intensity + OFI. This is the one high-volume
  corner left after Gate 0, and it is a genuinely different instrument from the closed terminal-digital work.
relationship: >
  Builds on the OD arc (hub [[strat_options_delta]]) and the Gate 0 universe map
  [[od_cross_asset_gate0_universe_map_findings]]. Sibling execution layer is [[strat_market_making]].
  "Block K" is the historical joint arc; this is a new OD sub-branch, NOT a reopen of closed work.
---

# Handoff — OD neg-risk first-passage / barrier branch

## You are a fresh Cowork chat. Read these first, in this order (Obsidian wikilinks)

Orientation (cheap, do these):
1. [[COWORK]] — repo conventions, Cowork-vs-Codex split, where things live. Obey it.
2. [[CODEX]] — what any Codex prompt you draft must tell Codex to read/obey.
3. [[TODO]] § OD and § MM — live state. The OD standalone-pricing line is CLOSED (see below); this is a new branch.
4. [[strat_options_delta]] — the OD hub + full "Current state" history. START HERE for the arc.

Why-it's-closed context (so you do NOT relitigate it):
5. [[od_cross_asset_gate0_universe_map_findings]] — the map that pointed here. Only `crypto-daily` cleared
   on scale; **close-above/price-band is the LARGEST corner ($6.5M/day, mostly crypto), rejected ONLY on the
   external-pricing gate (path-dependent barriers / "clean ref = no") — not on capacity.** That rejection is
   the opening for this branch.
6. [[od_pricing_model_form_findings]], [[od_conditional_prob_calibration_findings]], [[od_v4_calibration_gate_findings]],
   [[od_v4_queue_replay_findings]], [[od_strategy_a_v3_pnl_risk_findings]] — the TERMINAL-digital pricing line,
   closed three independent ways (jump models, conditional P(ITM), calibration). Key fact: near-strike 4h
   resolution is ~unpredictable, so "out-price the terminal digital" is dead. DO NOT propose another terminal
   up/down or terminal close-above pricing model — that's the closed branch.
7. [[block_k4_arb_scan_findings]] — intra-PM arb ≈ zero on its (narrow) universe. This is the SKEPTICAL PRIOR:
   PM internal consistency is often tight, and pure consistency-arb is a latency game we'd lose (speed is not
   our moat — [[block_k5b_findings]]). So this branch is NOT internal-consistency arb.
8. [[mm_deployable_cells_findings]], [[block_k5_findings]], [[block_k5_stress_findings]] — the capacity wall:
   real maker edge exists but is top-3-concentrated and mostly sub-scale; longshot premium NOT independently
   confirmed. Carry this caution into any capacity claim.

## The thesis for THIS branch (what's new and why it isn't relitigating)

Polymarket crypto neg-risk / price-band markets split into two sub-families:

- **Terminal "close above X" ladders** — each bucket is a European digital. Pricing these = the CLOSED branch
  on more strikes. **Out of scope** (low priority; expect the same null).
- **Barrier "will BTC *hit* X" markets** — these are **first-passage / touch** claims (does the path reach X
  *anytime* before expiry). This is a DIFFERENT instrument, never priced in our work, and it's where the
  high volume is. The thesis:
  - First-passage probability is path-dependent → **Binance high-freq realized path** genuinely informs it
    (unlike terminal digitals, where the path was irrelevant).
  - A touch is often a **jump** → jump-intensity modeling matters.
  - **OFI is legitimately useful HERE** (it was wrong-horizon for terminal fair value): near-horizon order-flow
    pressure toward the barrier is directly relevant to touch probability. This is the one place OFI earns a
    pricing role.
  - **Why an edge could persist:** retail is bad at estimating "probability of touching X anytime in a month"
    (first-passage is unintuitive), so behavioral mispricing is plausibly larger and stickier than on up/down —
    and this corner is NOT capacity-capped ($6.5M/day family).

## Your task (this Cowork chat)

1. Read the files above; form a real opinion on whether the first-passage/barrier thesis is sound or whether
   it inherits the same closure (be willing to say "this is also closed" — honesty over momentum; we are ~12
   runs deep and every pricing line has closed).
2. Then DRAFT a Codex scoping prompt (Cowork drafts, Codex runs — per [[CODEX]]). The prompt must:
   - **Classify first** per market: is each "will-hit-X" market truly touch-anytime (American/barrier) vs
     terminal close-above (European)? The math and resolution semantics differ entirely. Pull PM Gamma
     resolution text to decide. Only the barrier subset is in scope.
   - **Cheap gate before any build:** do PM touch-prices systematically differ from a Binance-history
     first-passage estimate (empirical touch frequency conditioned on distance-to-barrier z and time-left t,
     plus a jump-aware first-passage model), **net of PM fees and executable spread**? Lower-CI > 0, cluster
     by market.
   - Use OFI as a near-horizon touch-intensity feature, not standalone alpha.
   - Keep the [[CODEX]] discipline: brain-read header, uv/DuckDB/Parquet, lookahead-free, CI on headlines,
     validate signal before infra, no mark-to-mid, net-of-cost. Output a `*_findings.md` under
     `notes/options_delta/` and update [[strat_options_delta]] + [[TODO]] § OD.
   - Capacity caveat: measure top-3 concentration on the barrier markets specifically (Gate 0 used a family
     proxy), since "high family volume" ≠ "non-incumbent headroom."

## Guardrails carried from the arc (enforce)

- This is NOT the closed terminal-digital branch and NOT internal-consistency arb. If it collapses into either,
  stop and say so.
- Forecasting accuracy ≠ net-of-cost profit (Briola). Spread/fees have killed every taker variant; barrier
  entries must be net-of-cost from the start.
- No new data capture required for the scoping gate — it runs on Binance history + a PM neg-risk universe
  scrape. (Deribit and the 08:00-aligned RV collector are PARKED until a VPS exists, which is itself parked
  until a strategy justifies it — see [[od_rv_deribit_daily_capture_findings]].)
- Be willing to close it cheaply. The win condition for THIS chat is a well-aimed scoping prompt + an honest
  opinion, not a strategy.

## Parked (do not action here)
- Deribit 08:00-aligned 4h/hourly RV collector — built and smoke-tested, parked pending VPS.
- Equities / single-stock / index families — Gate 0 thin; recheck only if volume grows.
- Live-paper steal-share instrumentation (prompt C) — gated on Midas/colleague; not this branch.
