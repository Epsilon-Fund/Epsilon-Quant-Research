---
title: "Epsilon — Master TODO"
created: 2026-06-05
updated: 2026-08-26
status: active
owner: justin
project: infra
para: area
hubs:
  - CODEX
  - COWORK
  - VAULT_MAP
tags:
  - obsidian
  - brain
  - infra
---
# Epsilon — Master TODO

> Single live task list. Rewritten 2026-08-25 for the market-making handoff: **one active thread**; everything else is deprioritised or parked. The full pre-handoff task history (all threads, legacy codenames) is preserved verbatim in [[TODO_ARCHIVE]] — reference only, never work from it.

## ACTIVE — Market-Making on Polymarket

> Canon surface: [[strat_market_making]] (state of both lanes + reliability ledger) → [[mm_model]] (fundamentals vs additions). Read those before touching anything below. Honest framing applies to all of it: the evaluation numbers are preliminary; the settled assets are the data, the engine, the accounting/split methodology, and the measured 160 ms live latency.

**Handoff (in progress, owner Justin + Cowork):**
- [x] Repo synced: all work committed, pushed, merged to main (2026-08-23).
- [x] Brain/vault cleanup: de-jargoned canon surface written; status banners on all historical notes; bootstrap path cleaned (2026-08-25).
- [ ] **Market-making onboarding doc for the new collaborator** — written hand-in-hand with Justin: where we got to, what is built, what is reliable vs not, the paths forward. Concise.
- [ ] Handover email + repo access for the new collaborator.

**Research (the main open path — prime territory for the new collaborator):**
- [ ] **P&L spike anatomy.** Understand when and why account P&L jumps sharply up or down. Current state of knowledge: one dissected episode (violent price move; undefended quoter stacks a large wrong-way position; flow-imbalance rises ~30 min before, post-fill drift confirms during) plus the held-out market map (calm markets earn in the approach weeks, give it back near resolution). Wanted: more episodes dissected, how early the warning is detectable, and whether the approach/endgame boundary can be timed per market. Data: the R2 capture (politics + esports since 2026-06-19).

**Live measurement loop (blocked on instrument redesign, owner Justin first):**
- [ ] **(Justin) Pair-quoting design.** Two-sided quoting without inventory = an order to buy YES + an order to buy NO (buying NO at p ≡ selling YES at 1−p), plus event-market merge/split-to-$1 routes. Rework inventory/cap/costing conventions around the pair. Justin is taking a first pass himself before handing off.
- [ ] **Redesign the measurement instrument:** one persistent resting order (no re-quote churn), on a market chosen for fillability (thin queue beats wide spread), tiny size, hard caps, operator-confirm.
- [ ] **First real fills → calibrate.** Use real fills to fit the fill/queue model and replace the instant-order assumption with the measured 160 ms. This is the gate for trusting any backtest number.
- [ ] Later, once fills exist: re-run the evaluation with calibrated fill + latency models; only then revisit the defended quoter's ¢-numbers.

**Data upkeep (cheap, unowned):**
- [ ] Fix capture universe config: two configured universes (culture, crypto-as-control) never landed in the cloud bucket — only politics + esports exist.
- [ ] Sync the capture-health/gap log into the permanent layout before raw-file expiry deletes it (gap ground-truth is currently being lost).

## DEPRIORITISED — not active, not archived

- **Copy-trading** — paused at the brink of a first $10 live trade; its execution/signing infra is live and shared with market-making. Open items preserved in [[TODO_ARCHIVE]] § copytrade. Resume only with Justin.
- **News-agent / calibration observatory** — shipped; Justin-owned open items (sign-offs, API keys, settlements) in [[TODO_ARCHIVE]] § News-Agent. **Dashboard v3.5 built 2026-08-26** (presentation only — the grid is grouped by the evidence that actually feeds each market, and the 10 of 24 cards whose published number is an untouched onboarding prior now say so; no number, parameter or ledger entry changed; 738 tests green). Two things it leaves for Justin, neither of them work anyone should start unasked:
  - **Deploy.** v3.4 and v3.5 are both built and un-deployed; the push to Vercel is Justin's.
  - **The question the grouping surfaces, which display cannot answer.** 10 of 24 published numbers have zero relevant evidence behind them, and the page now says so on each of them — but it still publishes them, and they still enter the append-only ledger and will be scored. Whether a market with no evidence should publish a number at all (or publish it un-ledgered, or be retired from the slate) is a **method decision, pre-registered before anything is computed**, exactly as the v3.4 pass was. Not started, not recommended either way here.
- **Crypto live momentum book** — live; three standing items (re-baseline quoted Sharpe, plateau-centre re-optimisation rule, overfitting gate on new searches) in [[TODO_ARCHIVE]] § Crypto.
- **Brain infrastructure / skills lifecycle** — operational, low-touch; remaining items in [[TODO_ARCHIVE]].

## PARKED — historical record only

Earlier market-making eras, the valuation/fair-value overlay thread, and the microstructure signal lineage. All notes carry PARKED banners; their task sections live in [[TODO_ARCHIVE]]. Do not reopen without an explicit decision by Justin.
