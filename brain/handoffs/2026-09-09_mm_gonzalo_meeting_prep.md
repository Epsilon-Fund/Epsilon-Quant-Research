---
title: "Handoff — market-making meeting prep for Gonzalo: the project in one page, the two lanes, what is reliable vs preliminary"
created: 2026-09-09
status: active — pre-meeting document; § 8 checklist open, branch convergence outstanding
owner: justin
project: polymarket-mm
para: area
hubs:
  - COWORK
  - POLYMARKET_BRAIN
  - TODO
  - strat_market_making
tags:
  - handoff
  - market_making
  - onboarding
  - gonzalo
---

# Handoff 2026-09-09 — market-making meeting prep for Gonzalo

> Hubs: [[COWORK]] · [[POLYMARKET_BRAIN]] · [[TODO]] · Canon surface: [[strat_market_making]] → [[mm_model]]
> Data-layer companions: [[2026-09-02-research-v1-audit-justin]], [[2026-08-25_cowork_data_layer_session]], [[2026-08-25_l2_pipeline_recovery]]
> Imported 2026-09-09 from the standalone Cowork vault (`Epsilon/polymarket-mm-handoff/`), where it was the only copy. Content unchanged below the frontmatter.
> § 6's branch convergence is worked out step by step in [[2026-09-09_vault_consolidation]] § 3; the tiering rule behind § 2's framing is [[2026-08-25_mm_canon_tiering_decisions]].

## Plain-English Summary

- **What this is.** The talk-through document for handing the Polymarket market-making project to Gonzalo: the idea in three sentences, the two research lanes and how far each honestly got, Alvaro's data library and dashboard, an explicit reliable-vs-preliminary ledger, the repo state that must converge before he pulls, a ~75-minute meeting plan, and a pre-meeting checklist.
- **Why it was written.** The project's own notes read more confident than the work is. This document sets the honest framing first — *we moved into backtesting faster than we should have; nothing is a validated strategy yet* — so a new collaborator is not misled by a ¢-per-contract number.
- **The one thing to carry out of it.** § 5's reliability table. Method and data are the assets; the strategy is unproven. Gonzalo's first research task is P&L spike anatomy, going from one dissected episode to many.
- **Open at time of writing.** Branch convergence (§ 6), the read-only R2 token, Alvaro's four outstanding asks, and Justin's own pair-quoting design pass.

---

## 1. The project in three sentences

Rest passive buy and sell orders on a small set of slow, deliberately chosen Polymarket event markets and earn the spread. Profit is spread captured from ordinary traders minus losses to informed traders who hit your quote right before the price moves — and on Polymarket that second term is dominated by resolution, where prices jump to 0 or 1. Speed is not the edge; the candidate edge is **choosing the right markets at the right point in their life, and stepping aside from the violent episodes.**

**Honest framing (say this first):** we moved into backtesting faster than we should have. Nothing is a validated strategy yet. What Justin and Alvaro have is: a settled evaluation *method*, one open research question that matters more than everything else (why P&L spikes), a live order path that works, and two months of real order-book data. Every profit number in the notes is preliminary.

---

## 2. Lane 1 — backtesting / evaluation

**What is built.** A replay engine that takes captured order-book data plus a quoting strategy and deterministically simulates resting orders, queue position, fills, inventory and P&L (replaying a recorded run reproduces it exactly). An evaluation harness. A "defended" quoter that centres on the size-weighted fair price, skews against inventory, caps position, and watches two warning signals with a graduated response.

**Where it got to — three things, held at three different levels of trust:**

1. **Settled lesson — how to test.** The first evaluation split data by calendar date (train week 1, test week 2). That silently put every market's calm mid-life in training and its violent endgame in testing, so everything looked broken for the wrong reason. The fix: hold out **whole markets** — train on some markets' entire lifetimes, test on other markets' entire lifetimes. Under that split, tuned settings transfer almost one-for-one. Trust this; it must shape all future evaluation.
2. **The main open question — P&L spike anatomy.** Account P&L occasionally jumps sharply. What is known: losses trace to violent single-market moves (the dissected episode: 18¢ → 75¢ in minutes) during which an undefended quoter keeps getting filled on one side and stacks a large wrong-way position (~1,800 contracts); an order-flow-imbalance measure rose ~30 minutes *before* that move and post-fill price drift confirmed *during* it; and on held-out data, calm politics markets earn in the approach weeks and give it back in the final days. This is one dissected episode plus aggregate maps — **not a causal model. This is Gonzalo's territory.**
3. **Started, raw, low trust — robustness.** A first pass asked whether the small positive number survives different fill assumptions and accounting conventions. It said "directionally yes", but it was partly reasoned rather than computed (the raw book data had been cleaned off the laptop), and the two assumptions it rests on — instant orders, and a simulated fill model never checked against a real fill — are exactly the ones most likely to be wrong. Keep as intuition, not result.

**Additions tested and dropped (so nobody re-imports them):** a textbook academic quoting model (assumes smooth prices and uninformed flow — wrong for markets that jump to 0/1) and a basket-carry variant (safe, no edge). The *pair-quoting arithmetic* under basket-carry is live (see lane 2); the strategy variant is parked.

---

## 3. Lane 2 — live machinery

**What is built.** The same strategy code wired to place real orders through the venue client and signing layer, with hard inventory caps, an operator-confirm step, and a no-real-order default proven by tests. The order path was rebuilt mid-session when Polymarket deprecated its old client library; orders now go through the successor SDK in an isolated subprocess.

**Measured / achieved (12 July):** round-trip order latency ≈ **160 ms** (30 probes) — this replaces the backtests' instant-order assumption when the loop resumes. First genuine resting order placed, rested, cleanly cancelled, **$0 spent**, account flat.

**Why it stopped (deliberately):** three design findings.
- You cannot rest a *sell* on a token you don't own. **Conceptually solved:** in a binary market, an order to buy YES plus an order to buy NO *is* a bid and an ask (buying NO at p ≡ selling YES at 1−p); event markets' merge/split-to-$1 mechanics give further routes. Justin is taking a first pass at this design himself before handing it over.
- The quoter cancelled its own order within seconds on every price tick. The measurement instrument needs a *place-one-order-and-hold-it* mode.
- On deep books a small order sits tens of thousands of shares back in the queue and never fills. The measurement market must be chosen for **fillability first**.

**Next concrete step when resumed:** one persistent order on a fillability-chosen market → first real fills → calibrate the fill model against reality. Real fills are the gate for trusting any backtest number.

---

## 4. The data layer — Alvaro's research library and dashboard

**What it is.** The 24/7 order-book capture (rented server → Justin's Cloudflare R2 bucket, running since 19 June) has been turned by Alvaro into a curated research dataset, **`research_v1`**: 64 days (19 June → 21 August), politics + esports, 30,772 tokens, ~101M top-of-book change rows, 7.2M trades. On top of it: a loader library (`epsilon_data`, with a manual and worked notebook), a no-rclone fetch script, and a **Streamlit dashboard** (`polymarket/research/dashboard/`).

**What the dashboard does.** Two modes. *Explore* — pick one market and see five linked panels: price with the bid–ask band and trade prints, a markout view (what the price did after each trade — i.e. adverse selection made visible), and a NegRisk panel for multi-outcome events. *Audit* — dataset-wide health: reconciliation against raw shards, coverage calendar, distributions, outliers, activity by time of day, stale-book cohorts. Adding a panel is adding a file. Alvaro is preparing a simple walk-through of it for the meeting.

**How it ties in.** This is **lane 1's foundation, and the first tool for the spike-anatomy question.** The replay engine and the evaluation harness consume this capture; the dashboard's market + markout panels are exactly the "market moved, here is what fills would have suffered" view that the spike work needs, per market, per episode. It is not a strategy and not lane 2; it is the shared ground both lanes stand on.

**What Justin's independent audit found (2–3 Sept, done with Claude Code; full reports in the repo under `polymarket/research/docs/audit_2026-09/`):**
- *Sound:* the token-to-market identity mapping (90/90 random + adversarial tokens match the exchange and metadata oracles on all fields; 20/20 resolved markets end on the correct winner). Raw → library reconciliation drops nothing.
- *Weak:* the library's *own* evidence of its quality — the build code for the token table and its check columns is not in the repo; several checks are identities or cover a fraction of a percent.
- *Two real defects:* (1) the top-of-book de-duplication resets at each part-file boundary, leaving ~196k non-change rows (0.19%) all at the top of each hour — fixable in the build; (2) the NegRisk composite-sum helper forward-fills without bound, giving ~20-hour-stale sums — and the instantaneous multi-outcome sum is *not measurable* from this capture at all (fewer than 2 of 25 legs quote in any given second). Doc/setup defects (env loading, outage window, the broken `fed july` search example) are fixed in Justin's PR onto Alvaro's branch.
- *Open asks to Alvaro (sent 2 Sept):* the build code; the definition of one check column; whether the part-boundary reset was known; permission to run two full sweeps as build gates.

**Known capture gaps (cheap, unowned):** two configured universes (culture, crypto-as-control) never landed — only politics + esports exist; the capture-health/gap log is deleted on raw-file expiry.

---

## 5. What is reliable vs preliminary

| Trust it | Preliminary / do not trust yet |
|---|---|
| The capture pipeline and the `research_v1` dataset (identity mapping verified, nothing dropped) | Every ¢-per-contract edge number |
| Replay determinism (record → replay, zero gap) | The simulated fill model (never validated against a real fill of ours) |
| Honest exit-priced accounting convention | The instant-order latency assumption inside all backtests |
| The whole-market evaluation split | The robustness/stress-testing pass |
| The spike anatomy at one-episode level | Any market-map claim beyond politics/esports |
| Measured 160 ms live latency | The library's self-reported quality checks (use Justin's audit instead) |
| The dashboard as a *viewing* tool | The NegRisk composite-sum panel (staleness defect) |

---

## 6. Repo state — must converge before Gonzalo pulls

Three lines of work currently diverge:

| line | contains | state |
|---|---|---|
| `main` (= `justin` minus 2 daily-sync commits) | the de-jargoned canon surface + tier banners + lean TODO (cleanup commit `67529bd`) | pushed |
| `alvaro` | the L2 ingestion system, `epsilon_data` library, dashboard, fetch script, `HANDOVER.md` (17 commits not in `main`) | pushed, **never merged to main** |
| `justin-research-v1-audit` (off `alvaro`) | the audit reports + doc/env fixes (2 commits) | local; PR onto `alvaro` pending Alvaro's answers |

Gonzalo must pull a single branch that has *all three*. Note the merge will bring Alvaro's docs (`polymarket/research/README.md`, `HANDOVER.md`, `epsilon_data/README.md`) in **without** the tier banners — they are active-project material and belong in the hub's "Where things live" section, which needs a short *Data layer* paragraph added once merged.

---

## 7. Proposed meeting plan (~75 min)

1. **The idea and the honest framing** (Justin, 10 min) — §1 above, verbatim. Set the expectation: method and data are the assets; the strategy is unproven.
2. **Lane 1 walk-through** (Justin, 15 min) — the three levels of trust. Open the hub and `mm_model.md` on screen; show the fundamentals-vs-additions split and why the additions are last.
3. **The data layer** (Alvaro, 20 min) — his simple explanation of the dataset and dashboard. Steer it toward Explore mode on one politics market with the markout panel, because that is what Gonzalo will live in. Flag the two real defects and the open asks openly — they are Gonzalo's first contact with the data's edges.
4. **Lane 2 and pair-quoting** (Justin, 10 min) — what works live, why it stopped, the pair-quoting resolution, and that Justin is finishing that design himself.
5. **Reliability ledger** (5 min) — §5 on screen. This is the single most important slide for not misleading a new person.
6. **Gonzalo's first work and setup** (15 min) —
   - *Setup:* pull the converged branch; the agent-orientation message (below); fetch `research_v1` with Alvaro's script; run the dashboard with `EPSILON_DATA_ROOT` exported.
   - *First task (research):* **P&L spike anatomy** — go from one dissected episode to many. Use the dashboard's market + markout panels to find and characterise violent episodes across the 64 days; measure how early the imbalance warning shows up; test whether the approach/endgame boundary can be timed per market. Deliverable: a findings note in the market-making folder following the canon's plain-language rule.
   - *Second task (engineering, per the audit):* the ~5 GB full-depth book tier and wiring the replay engine to the library — the audit summary already assigns this to Gonzalo.
7. **Decisions to close** — who owns capture upkeep (the two gaps); the R2 **read-only** token for Gonzalo (the current key is read/write); Alvaro's four open asks and their timing; when Justin's pair-quoting design lands.

---

## 8. Pre-meeting checklist (Justin)

- [ ] Converge the branches (§6) so Gonzalo pulls one thing. Add the *Data layer* paragraph to the hub after the merge.
- [ ] Confirm the dashboard starts cleanly on your machine with `EPSILON_DATA_ROOT` exported (a stale instance was found on port 8501 during the audit).
- [ ] Mint the read-only R2 token.
- [ ] Run the fresh-chat orientation test once more on the converged branch.
- [ ] Ask Alvaro to keep his walk-through to the Explore panels + the two defects.

**Agent-orientation message for Gonzalo** (paste into a fresh Claude Code chat in the repo):

> I'm picking up the market-making project in this repo. Orient yourself before doing anything: read `CLAUDE.md` at the root and follow its bootstrap (`brain/CODEX.md`, then `brain/VAULT_MAP.md`, `brain/TODO.md`, `brain/POLYMARKET_BRAIN.md`). Those point to the canon surface — read both in order: `polymarket/research/notes/market_making/strat_market_making.md`, then `mm_model.md`. Market-making is the only active thread; every other research note carries a status banner (HISTORICAL EVIDENCE / PARKED / DEPRIORITISED) — never build on one, never quote its numbers as current. `brain/TODO.md` is the live task list; `TODO_ARCHIVE.md` is reference only. Then tell me: where is the work up to, how are the two lanes doing, what is reliable vs preliminary, and what is my first piece of work?
