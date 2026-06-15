---
title: "Meeting Prep — Alvaro MM Debrief (2026-06-16)"
created: 2026-06-15
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
---

# Meeting Prep — Alvaro MM Debrief

> Hub: [[strat_market_making]] · [[POLYMARKET_BRAIN]] · [[TODO]] § MM
> Purpose: understand the market-making strat head-to-toe before debriefing with Alvaro and deciding how to split the work going forward.

## Plain-English Summary

- **What MM is:** the pivot from betting *direction* to *providing liquidity* (quoting both sides, earning `spread + rebate − adverse selection − inventory/resolution risk`) on Polymarket.
- **Where it landed:** building our *own* single-venue maker is **closed/dead** (proven ~10 ways). But real maker wallets *are* profitable (K5), and one live monetization candidate survives: **politics NegRisk neutral market-making**, treated as a measurement loop, not yet a money-printer.
- **What we actually do today:** run an **observation-only live CLOB capture** (3 lanes) to measure the live-only unknowns (fill share, queue, adverse selection) that 9 months of history can't answer. **We are not placing any live orders yet.**
- **The honest catch (bring this to the meeting):** the validated edge is *structurally small* — median-wallet EV is ~$9–$189/day; the headline $1.5k–$30k/day figures are right-tail/mean scenarios that depend on capturing fill share we have not yet proven live.
- **Data status:** ~64 GB of live capture over ~Jun 4→13 across 3 lanes, plus the 9-month historical base. **The daily capture lapsed — nothing has run since Jun 12 (so Jun 13/14/15 are missing).** Restart command is in §5.

---

## 1. The strategy, head to toe (the "Block K" arc)

The whole maker program lived under the historical name **Block K**, now split into two hubs: **MM** ([[strat_market_making]], quote for spread + rebate) and **OD** ([[strat_options_delta]], digital-option vol/basis + delta hedge). The single best read is [[block_k_plain_english_synthesis]] — written for someone with zero quant background.

**The core equation everything fights over:**

```
maker profit per trade = spread captured + rebate − adverse selection − inventory/resolution risk
```

The first two terms are what you earn; the last two kill you. **Adverse selection** (getting filled by someone who knows the price is about to move) is the killer.

**Our own single-venue maker is CLOSED across every anchor we tried:**

| test | what it tried | result |
|---|---|---|
| K1 | baseline maker economics | "passes" only by marking-to-mid (fee-free geopolitics also "passed" → not a real edge) |
| K2 | Avellaneda-Stoikov logit quoting, real exit | **−1,126 bps**, CI < 0 |
| K-PEG | optimized "chase" maker | **+759 bps** — but it was a **mark-to-mid artifact**; force a real exit → **−753 bps** |
| K2v3 | anchor quotes to Binance fair value | **0/681 buckets clear**; anchor *raised* adverse selection (325 vs 145 bps) |
| K2v2 | defensive pull/widen on Binance move | **−4,316 bps**; defense fired <0.1% → **adverse selection is structural, not a dodgeable latency race** |

**Two lessons that everything converged on:**
1. **Mark-to-mid is not money.** On a 15–30¢-spread venue, "the price reverted in my favor" means nothing if collecting it means crossing the spread back. The problem is always the **exit**, never the entry.
2. **Adverse selection here is structural**, not a latency race you can out-react. You can only avoid being there.

**The breakthrough — K5 (model-free, real wallets):** instead of simulating, we measured the realized closed-position PnL of real maker wallets over 9 months. Real maker-heavy wallets **are profitable**: crypto-4h **+171 bps, CI [34, 327]**; pooled top-maker **+145 bps, CI [85, 210]**. The winners' playbook is now known:
- **64% two-sided** (offset a buy with a later sell, never cross the spread to exit)
- **78.8% carry-to-resolution** (hold to $1/$0, don't pay to exit)
- **0.8% in the late near-50¢ spike zone** (avoid the gamma-spike danger area)
- **Capacity warning: top-3 wallets capture ~95% of positive profit per market** (winner-take-most).

**Follow-ups:** K5b → the moat is **capital/scale + structure, NOT speed** (so build in Python/Midas; **defer Rust**). K5-STRESS → survivorship wasn't the problem; **selection** was (the median/typical maker ≈ breakeven), but the **structured-non-top3 cut clears in 4 categories**. The longshot/vol-overpricing premium is **not** independently confirmed, so the surviving edge is **structural liquidity provision**, not mispricing.

**Bottom line:** MM standalone doesn't justify a dedicated bot (~$78/active day, ~90% of it in one grab-bag cell). Its durable value is the **execution/lifecycle layer** (two-sided passive entry + carry-to-resolution + spike-avoidance + non-incumbent cell selection) that *any* Polymarket edge needs to dodge the exit tax. The repo's standing recommendation: **consolidate to OD-primary, MM = the execution layer.**

**The two strategies still alive** (from [[block_k_plain_english_synthesis]] §6):
- **Strategy A — "capture and carry":** passive maker entry → hold to resolution → static hedge on Binance. Never pays the PM exit spread. This is the OD-integrated build.
- **Strategy B — "don't compete, copy":** find/mirror the wallets already profitably making these markets (re-merges with the copytrade thread). Cheaper, model-free, uses data we already own.

---

## 2. How it monetizes (be honest about this in the meeting)

There is **no live revenue today** — we are in measurement, not trading. The monetization *paths*, ranked by how real they are:

1. **Politics NegRisk neutral market-making — the one live candidate.** Corrected-carry structured-non-top3 edge is **+2,290 bps, CI [1,020, 3,621]** ([[mm_politics_negrisk_accounting_findings]]). It is the biggest, most persistent cell and it survives proper accounting. **Verdict: merits a live *measurement* loop, not yet a trading system.**
   - **Capacity is the catch:** at 0.25% / 1% / 5% capture of non-top3 flow, **mean** EV/day ≈ **$1.5k / $6.0k / $29.9k**, but **median-wallet** EV/day is only **$9 / $38 / $189**. The big numbers are a right-tail scenario.
   - Flow is real and persistent: **$381.1M** 2026 non-top3 politics flow through May 26, active **146/146** days, every day ≥ $250k.

2. **MM as the execution layer for OD Strategy A.** OD supplies the edge (cross-sectional longshot/vol overpricing); MM supplies the lifecycle that avoids the exit tax. This is the "two layers of one strategy" framing.

3. **Strategy B / copy-learn** the profitable makers — folds back into copytrade (the repo's nominal primary thread).

**Explicitly CLOSED (do not pitch as monetization):**
- The **NegRisk basket-consistency arb** ("YES legs sum to 1"). Real and offline-persistent, but **net-negative after fees on ~98% of episodes**; liquid-basket violations close in **~4 s at 0.10c**. It's a **latency game** for sophisticated bots — our value is *measuring* the ~$34.6M gap, not racing for it ([[mm_negrisk_consistency_scanner_findings]]).
- Single-venue quoting / continuous-hedge gamma scalp — exhausted, do not re-run.

---

## 3. Why we need the dataset, and where we look

History (the 9-month base) is model-free but **can't see the live-only unknowns**: real fill share, queue position, missed fills, adverse selection after *our* fills, quote fade. Public Polymarket L2 is also **anonymous** — no maker wallet, order ID, or queue position ([[mm_clob_capture_semantics]]) — so we **capture the live book forward** to measure quoteability and adverse selection ourselves.

The live capture runs in **three lanes**, each targeting a different open question (`scripts/mm_stage1_live_control.py`):

| lane | market scope (per cycle) | question it answers |
|---|---|---|
| **A — first-mover** | sports_recurring ×14, other_residual ×2, geopolitics_diagnostic ×2 | do newly-created markets leave non-incumbent liquidity before top-3 consolidate? ([[mm_first_mover_liquidity_scope_findings]]) |
| **B — broad diagnostics** | crypto_fast 5m ×12 / 15m ×6, equity_index open ×1 / close ×2, sports ×8, other ×4, geopolitics ×3 | broad book-shape / spread / adverse-selection baseline across categories |
| **C — slow crypto + finance** | crypto_4h ×8, crypto_daily ×8, equity_index open ×1 / close ×2, crypto_fast_5m ×8, sports ×4 | slow-market neutral-MM + the OD/equities up-down scope |

**Important framing for the meeting:** this capture feeds the **first-mover novelty branch + broad/slow diagnostics**. It is **observation-only (no orders, no auth).** The **politics NegRisk live loop is a separate, not-yet-deployed** maker-engine deployment — the highest-EV candidate, built but never switched on.

---

## 4. Data audit — exactly what we have

**Historical base (the 9-month model-free dataset, `polymarket/research/data/`):**

| artifact | size | what it is |
|---|---|---|
| `trades/` | 43 GB | historical fills (the K5 etc. base) |
| `closed_positions.parquet` | 28 GB | resolved positions → realized PnL |
| `bankroll_timeseries.parquet` | 12 GB | per-wallet bankroll over time |
| `traders.parquet` | 570 MB | wallet/trader index |
| `markets/` | 197 MB | market metadata |

**Live CLOB capture (`data/live_clob/`, ~71 GB total). The three MM lanes:**

| lane | size | shards | days covered |
|---|---|---|---|
| `mm_stage1_first_mover` | 17 GB | 199 | Jun 4–13 |
| `mm_stage1_broad_live` | 24 GB | 195 | Jun 5–13 |
| `mm_stage1_slow_crypto_finance` | 23 GB | 195 | Jun 5–13 |

So ≈ **64 GB / ~590 shards over ~9–10 calendar days.** Plus the older `block_a0*` OFI captures and a 3-hour `mm_negrisk_consistency_scan`.

**The gap you flagged is real.** The capture is **gap-tolerant but the laptop must be awake** (it uses `caffeinate` + detached `screen` sessions). The last run started **2026-06-12 08:12Z** and ended ~Jun 13 00:12. The state file `data/live_clob/mm_stage1_current.json` is **empty → nothing is capturing right now**, so **Jun 13, 14, and 15 are missing**. Each manual kick runs ~16 h by default, so even on active days it's not 24/7.

---

## 5. Restart the capture (the manual command)

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/mm_stage1_live_control.py start --duration-hours 16
```

- `status` (health / what's running) and `stop` (graceful) are the other subcommands.
- `--takeover` stops any existing `mm_stage1` screens first.
- It generates fresh Gamma/CLOB configs, launches the 3 lanes in `screen`, and starts `caffeinate` so the Mac won't sleep.

**This being manual + awake-only is the single most fragile part of the operation** — a strong candidate to automate (VPS/cron) and a clean thing for one person to own (see §8).

---

## 6. What's already built (execution infra)

`polymarket/execution/maker/` — **measurement-grade, not production-grade** ([[mm_maker_infra_audit_findings]]); **67 maker/NegRisk tests + 256 full execution tests pass**:

- `maker_engine.py` (31 KB) — one-market two-sided passive quoting scaffold (1 contract, join best bid/ask, cancel-replace, telemetry)
- `negrisk_inventory.py` — composite basket inventory poller (SPLIT/MERGE/REDEEM/CONVERSION)
- `resolution_handler.py` — resolution detection + self-redemption (web3)
- `event_calendar.py` + `politics_events.yaml` — scheduled-event proximity
- `cli.py`; plus NegRisk-aware signing in the shared venue adapter
- Docs: `polymarket/execution/PLAN.md` (45 KB), `CLAUDE.md`, `maker/README.md`

**Telemetry gaps before this is production-grade:** true queue position, side-adjusted drift bps, volume-weighted fill share, book depth at quote/fill, quote age, cancel latency.

---

## 7. Reading list (in priority order)

1. **[[block_k_plain_english_synthesis]]** — the whole arc, no background needed. *Read this first.*
2. **[[strat_market_making]]** — MM hub + current state (this is the live status line).
3. **`brain/TODO.md` § MM** (lines ~130–180) — authoritative open tasks + the de-biased K5→K9 arc.
4. **[[mm_politics_negrisk_live_loop_design]]** — the live-loop design, 5-screen filter, and pre-registered Phase-2 gates (the monetization plan).
5. **[[mm_politics_negrisk_accounting_findings]]** — the +2,290 bps headline + capacity ledger.
6. **[[mm_negrisk_consistency_scanner_findings]]** — why the basket arb is closed.
7. **[[block_k5_findings]]** / **[[block_k5b_findings]]** / **[[mm_deployable_cells_findings]]** — the profitable-maker reality check, the moat (scale not speed), and the ~$78/day deployability.
8. **[[mm_first_mover_liquidity_scope_findings]]** — what the current Stage-1 capture is actually testing.
9. **`polymarket/execution/maker/README.md`** + **`PLAN.md`** — what's built.
10. **[[2026-06-04_state_of_the_arc_and_novelty_frontier]]** — the one-page high-level map.

---

## 8. Current plan of action (phase map)

| phase | what | status |
|---|---|---|
| Phase 0 | copytrade smoke (plumbing) | live default |
| Phase 1 | exec build: NegRisk signing → resolution handler → inventory tracker → maker engine + calendar | **DONE** (256 tests green) |
| Phase 2 | measurement-only deployment, 1 contract, ≥30 settled markets, full telemetry | **NOT STARTED** — blocker below |
| Phase 3 | scale within proven buckets *if* Phase-2 gates pass | blocked on Phase 2 |
| (parallel) | Stage-1 live CLOB capture (first-mover / broad / slow), observation-only | **running, but lapsed since Jun 12** |

**Phase-2 blocker:** it's lane-owned — someone has to (a) run the 5-screen filter to pick a market that passes its gate, (b) wire the 1-market / 1-contract / ~$30 telemetry smoke, and (c) optionally add the missing telemetry first. Pre-registered success gates (must hit before any scale): fill share > 0% in ≥5 markets; post-fill 60s drift lower-CI > −500 bps; news-proximate adverse fills < 50%; net-of-cost lower-CI > 0 over ≥30 settled markets; resolution drag < 10%.

---

## 9. Ways to separate the work (for discussion with Alvaro)

There are three clean seams. Pick one main split, and regardless of which, **assign a single owner to "keep the capture alive + automate it"** — it's the most fragile, most ownable workstream.

**Option A — split by layer (recommended).**
- **Research/analysis owner:** runs the gates, measures adverse selection on captured data (`mm_stage1_analyze_capture.py`), decides which lane/market passes, owns the OD valuation overlay and the findings notes.
- **Execution/ops owner:** owns the maker engine + telemetry upgrades (queue position, drift bps, fill share), resolution/redemption, daily capture ops, and the eventual Phase-2 1-contract smoke.
- *Clean interface:* research hands execution a market + frozen quoting params; execution hands back telemetry. Matches the repo's existing implementation-vs-orchestration split.

**Option B — split by open question / thread.**
- **Owner 1:** Politics NegRisk live loop (Phase 2) — the highest-EV monetization candidate (5-screen selection + gates + deployment).
- **Owner 2:** First-mover liquidity Stage-1 capture (the forward-test novelty branch) — the 3 lanes + quoteability/adverse-selection analysis.
- *Shared:* the execution infra + the data pipeline.

**Option C — split by Strategy A vs B.**
- **Owner 1:** Strategy B (copy/learn profitable makers) — model-free, re-merges with copytrade, uses the 9-month data.
- **Owner 2:** Strategy A (capture + carry + static hedge) — the OD-integrated execution build.

---

## 10. Open questions to settle in the meeting

1. **Deploy or not?** Do we switch on Politics NegRisk Phase-2 with real money (~$30, 1 contract), or keep paper? Who owns it?
2. **Automate the capture?** Move to a VPS/cron so it stops lapsing and goes 24/7 instead of 16h-awake-only. Who owns ops?
3. **Telemetry first?** Add queue position / drift / fill-share telemetry *before* the Phase-2 smoke, or deploy thin and iterate?
4. **OD-primary vs MM-primary** — agree the consolidation framing (MM as execution layer for OD).
5. **Is MM worth two people?** The validated edge is structurally small (median-wallet ~$9–$189/day; standalone ~$78/day), while the crypto momentum book is already live (headline 2.24 Sharpe, under overfitting review). Honest portfolio question: how much of our combined time should MM get vs copytrade / crypto?
