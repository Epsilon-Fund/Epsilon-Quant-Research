---
title: "Handoff — News-Agent Showcase: v0+v0b gates failed honestly; Calibration Observatory shipped"
tags: [handoff, news-agent, showcase, block-j, gate-result, pre-registration, observatory]
created: 2026-07-05
status: shipped — fair-value framing closed by two gates; live measurement loop running; three items on Justin
purpose: >
  Chronicler snapshot of the 2026-07-04/05 implementation session that opened the
  News-Agent Fair-Value + Calibration Showcase thread (the never-run Block-J /
  LLM-forecaster frontier framed as a PUBLIC showcase, not a trading strategy),
  ran BOTH pre-registered gates (v0 point-forecast; v0b 5-perspective ensemble),
  closed the fair-value framing when both fired the same early-stop, and shipped
  the honest re-scope: the Epsilon Calibration Observatory measurement loop.
---

> **SUPERSEDED SECTIONS BELOW (kept for the record):** the original version of this
> handoff ended at "v0 STOP; v0b awaiting sign-off." The session continued under the
> standing goal mandate: v0b was pre-registered (Amendment 2) and run — **early-stop
> fired again** (Brier +0.171 vs mid, tracking 40%) → fair-value framing CLOSED (no
> v0c). The showcase then shipped as the **Calibration Observatory**: agent % + band
> vs PM mid on 5 live politics markets, daily append-only ledger snapshots
> (sf-2026-001…005 live), `calibrate` scoring on settlement, retrospective gate
> scoreboard displayed with the market winning. Package `polymarket/research/newsagent/`;
> dashboard regenerates to `data/newsagent/showcase/` (self-contained HTML+JSON,
> IP-scrub verified). Remaining Justin items moved to [[TODO]] § News-Agent Showcase:
> Scheme-A weighting sign-off, GUARDIAN/ANTHROPIC keys for unattended runs, website
> handoff of the showcase artifacts. Current state: [[strat_news_agent_showcase]].

# Handoff — News-Agent Showcase v0: STOP

> Hub: [[strat_news_agent_showcase]] · [[POLYMARKET_BRAIN]] · [[TODO]] § News-Agent Showcase

## What happened (one paragraph)

The showcase idea — a public page showing Epsilon's news-agent fair-value % next to the Polymarket mid on high-liquidity politics markets, hooked on a Brier/calibration track record — was taken through the full no-infra-before-signal loop in one session: pre-registration locked first ([[newsagent_v0_gate_preregistration]]), external radar run in parallel ([[newsagent_repo_data_radar_findings]]), a 10-market / 6-family resolved June-2026 universe built lookahead-free, contamination canaries verified CLEAN, and the pre-registered cheapest falsifier executed. **The early-stop fired** (Brier ours 0.547 vs mid 0.354, diff +0.194 > +0.15; news-tracking 40% < 50%) → remaining forecasts aborted, **no dashboard built**. Full result + mechanism diagnosis: [[newsagent_v0_gate_findings]].

## Decisions made

1. **STOP enforced as pre-registered** — no gate-shopping, no post-hoc metric relaxation. The v1 dashboard, ledger wiring, and site integration were all gated out.
2. **Amendment 1 (documented before any scoring):** GDELT DOC API entered a persistent 429 state (reproduced from this machine AND the Hetzner VPS) → packet source switched to Guardian Open Platform + Wikipedia Current Events daily pages (timestamps stricter than GDELT's leaky end-bound).
3. **Falsifier trio composition deviation (documented):** Fed-June markets failed the pre-registered information filter (mid pinned >95c) → trio = Iran-deal / Starmer / Cepeda.
4. **Diagnosis recorded as fixable-prompt-failure, not concept-closure:** the status-quo-weighted single-sample forecaster under-reacts to probability-shifting news even when it is in-packet (held 3% on the Iran deal with "deal could be signed by Sunday" in evidence), while it *did* react to settled facts (Cepeda post-runoff) and beat the mid 6/6 on the quiet NO-market.

## Waiting on Justin (two decisions)

- **Run v0b or close?** The redesigned gate (N=5 trimmed-mean ensemble, evidence-weighting prompt, multi-source packets + relevance filter; same bars, same trio; ~300 Sonnet samples, zero new infra) is fully specified in [[newsagent_v0_gate_findings]] § Redesigned gate proposal. It was deliberately NOT run without sign-off.
- **Source-weighting scheme** (only relevant if v0b passes): recommended Scheme A = Wikipedia perennial-sources tiers (CC BY-SA) + Iffy Index blocklist (CC BY); AllSides is NC-grey; Ad Fontes/MBFC data are paid/contract-gated; unofficial MBFC scrapes ruled out. Table in [[newsagent_repo_data_radar_findings]].

## Reusable assets left behind

- Pipeline scripts (`polymarket/research/scripts/newsagent_v0_*.py`): universe selection with family caps + info filter, chunked CLOB mid fetch, two packet fetchers (GDELT + Guardian/WP), leak-audited prompt renderer, pre-registered scorer with family-clustered bootstrap.
- Raw artifacts (git-ignored, local): 56 lookahead-free news packets, 15 forecasts, canary results, price caches — enough to rescore or extend without refetching.
- Radar verdicts: forecasting-tools (MIT) = Adopt-grade harness for any future forecasting work; FinceptTerminal = AGPL+commercial, design-borrow only; ForecastBench scoring patterns (incl. interim scoring vs prior-day mid) = Borrow.
- Operational lessons: GDELT DOC hard-throttles (1 req/5s, sticky multi-hour 429s, ~24h end-bound leak → client-side seendate filter mandatory); Gamma slug lookups on closed markets need `closed=true`.

## Guardrails honored

Pre-registration before computation; amendment documented before scoring; canary contamination checks; forecasters never saw mid/outcome/web; append-only raw caches; no cross-import; no commits outside the `justin` branch; ledger untouched (no post-hoc entries — live ledger logging was gated out with v1).
