---
title: "Handoff — the MM canon cleanup: tiering rule, de-jargon rule, and the acceptance test that produced every status banner in this repo"
created: 2026-08-25
status: applied — the banners, the de-jargoned hub and the lean TODO all exist; this note records the rule that produced them
owner: justin
project: polymarket-mm
para: area
hubs:
  - COWORK
  - VAULT_MAP
  - TODO
  - strat_market_making
tags:
  - handoff
  - market_making
  - brain
  - onboarding
  - canon
---

# Handoff 2026-08-25 — MM canon cleanup: the tiering rule behind the banners

> Hubs: [[COWORK]] · [[VAULT_MAP]] · [[TODO]] · Canon surface: [[strat_market_making]] → [[mm_model]]
> Written up 2026-09-09 from the decisions settled with Justin on 2026-08-25. Those decisions were applied to the repo at the time (cleanup commit on `main`, `67529bd`) but the **rule itself** was never written down here — only the results were. Consumers: [[2026-09-09_mm_gonzalo_meeting_prep]], [[pre-migration-vault-2026-05]], [[2026-09-09_vault_consolidation]].

## Plain-English Summary

- **What this is.** The standing rule for how this repo's research notes are classified and written, settled when the market-making project was being prepared for a third collaborator. It explains *why* almost every note in `polymarket/research/notes/` opens with an `ACTIVE` / `HISTORICAL EVIDENCE` / `PARKED` / `DEPRIORITISED` banner, and what the banners are protecting against.
- **The goal it serves.** A fresh agent chat, pointed only at the brain's bootstrap path, must be able to answer "where are we on the market-making work, how are the two lanes doing, what is reliable vs preliminary" **correctly, using no legacy codenames and inventing nothing.**
- **Why it matters after the fact.** Anyone doing a cleanup, an import, or a merge needs this rule, or they will re-import material at the wrong tier and quietly undo the work. It is the reason imported archive material gets a banner rather than a folder move alone.

---

## 1. The acceptance test

The cleanup was not "tidy the notes." It had a single pass/fail test:

> Open a **fresh chat** pointed at this repo. Ask: *"Where are we on the MM work, how are the two lanes doing?"* The answer must be **correct**, must use **zero legacy codenames** — no K / K5 / K-PEG, no OD, no dali, no Task 5 / 5.1, no Join-2, no NSQ — and must contain **no hallucination**.

Two consequences follow directly:

- **Concepts are explained inline** in the canon surface, not resolved by link-crawling. [[strat_market_making]] and [[mm_model]] must be self-contained; a reader who never opens a third note must still understand the project.
- **Parked work is referenced minimally**, as "touches parked work, out of scope" — never summarised, never quoted.

## 2. The three tiers

| Tier | What is in it | How it is marked |
|---|---|---|
| **ACTIVE** | The MM-with-Alvaro work: the backtesting/evaluation lane, the live-machinery lane (order path on the new SDK, 160 ms measured latency, first real resting order, $0 spent), and the VPS→R2 24/7 L2 capture running since 2026-06-19. | no banner — the canon surface |
| **CONTEXT** | Concepts the project needs but whose source notes are not live: the wallet studies that motivated the politics focus, the imbalance-as-quoter-gate finding, the textbook quoting model and basket-carry as "additions tested, and why they fail on PM". | explained **inline** in the canon; source notes carry `HISTORICAL EVIDENCE` |
| **PARKED** | Everything K-era, OD (one line maximum in any MM doc), the dali lineage, and superseded Task-4/5 verdicts. | `PARKED` banner; keeps its old codenames |
| **Copytrade — middle tier** | Not priority A, but **not parked either**: its execution/signing infrastructure (the Midas mirror) is live infrastructure that market-making depends on. | `DEPRIORITISED` banner |

Exact banner wording is in the notes themselves — copy an existing one rather than writing a new phrasing.

## 3. Justin's epistemic reframing — this overrides the notes' own confidence

**"We jumped too quick into backtesting."** The backtesting lane's outputs are **lessons, not settled results**, held at three distinct levels of trust:

1. **Settled lesson — how to split test data.** Calendar walk-forward broke on a regime confound (every market's calm mid-life landed in train, its violent endgame in test). Train on some whole markets, test on other whole markets, each for its full life. Carry this forward; it must shape all future evaluation.
2. **The main open path — how P&L spikes occur.** Violent price moves, one-sided fill stacking, a flow-imbalance warning roughly 30 minutes ahead, post-fill drift confirming during, approach-window earns and endgame bleeds. One dissected episode plus aggregate maps — **not a causal model.** This is where the next person digs deepest, and we are still at the stage of understanding items 1 and 2.
3. **Started but raw, LOW TRUST — whether the positive result is real.** The fill/queue bracket and latency assumptions. Vital, but partly reasoned rather than computed. File under intuitions to keep in mind, **not results.**

→ **Every ¢-per-contract number in this repo is PRELIMINARY.** Never present one as near-certified.

**The sell-side blocker is conceptually solved.** The old finding was that you cannot rest an ask on a token you do not own. In a binary market an order to **buy YES plus an order to buy NO is a bid and an ask** (buying NO at *p* ≡ selling YES at 1−*p*), and NegRisk merge/split-to-$1 gives further routes. Not a blocker — direction. Justin is taking the first design pass himself.

## 4. Structure rules that came out of it

- **Fundamentals vs additions, explicitly separated.** The model is: naive symmetric quoting + honest exit-priced costing + a fill bracket + the whole-market split + the market map. Inventory skew, position caps, toxicity gates, asymmetric repricing and size dampening are **ADDITIONS** — each individually justified, and each added only once the fundamentals are down.
- **Full de-jargon of the canon docs.** Subagent delegation is fine for this. Parked notes keep their old names and their parked banners; only the canon surface is de-jargoned.
- **The bootstrap path is what gets cleaned**: `CLAUDE.md` → law files → [[VAULT_MAP]] → [[TODO]] → [[POLYMARKET_BRAIN]] → the MM hub. That chain is the acceptance test's whole input.
- **The onboarding doc is written *with* Justin**, at chat-summary conciseness: "Justin and Alvaro got here, verified X (reliable), Y is less reliable, look into Z before the goal." The worked instance is [[2026-09-09_mm_gonzalo_meeting_prep]].
- **Sequencing: git sync first, cleanup edits second**, on the personal branch — never on `main`. See [[MERGE_PROTOCOL]].

## 5. Where this leaves an importer

Any note arriving from outside this repo — a chat, an old vault, another branch — is untiered until someone tiers it. The default is **not** ACTIVE. Import it, banner it at the tier its thread sits in today, and link it from an index so it is reachable without being stumbled into. That is the rule [[pre-migration-vault-2026-05]] was built under.
