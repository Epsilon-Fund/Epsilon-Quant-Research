---
title: "NegRisk basket price-consistency scanner — does the deterministic 'complementary prices sum to 1' arb survive fees, depth, and latency for a small player?"
tags: [market-making, negrisk, arbitrage, consistency, live-scan, results]
created: 2026-06-10
status: close-for-small-player
owner: justin
project: polymarket
para: resource
hubs:
  - strat_market_making
  - COWORK
---

# NegRisk basket price-consistency scanner — is the deterministic complementary-price arb capturable?

> Hub: [[strat_market_making]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]
> Builds on: [[mm_politics_negrisk_accounting_findings]] (our own $34.6M accounting gap) · [[block_k4_arb_scan_findings]] (owned-universe rebalancing/combinatorial scan) · [[2026-06-05_novelty_deep_research]] § B.1 (the ~$40M academic study). Methodology spine: [[CODEX]] § Realism calibration.

## Plain-English Summary

- **The idea.** Inside a Polymarket NegRisk event exactly one of N mutually-exclusive YES outcomes resolves to $1, so the YES prices should sum to ~1. Two deterministic arbitrages exist: **buy-all** (if the best asks sum to *less* than $1, buy every YES leg, hold to resolution, redeem exactly $1) and **sell-all** (if the best bids sum to *more* than $1, mint a full YES set for net $1 via the NegRiskAdapter and sell every leg). Persistent deviations would be free money. This is the edge that produced our own **$34.6M** accounting gap ([[mm_politics_negrisk_accounting_findings]]) and the **~$40M** academic extraction figure.
- **What we built.** A read-only scanner that sweeps the **entire active NegRisk universe (2,359 events / 32,436 legs)** every ~2 minutes via batched CLOB book fetches, recording per-basket sum-of-best-asks and sum-of-best-bids, top-of-book depth, and fee-netted edge; plus a **millisecond-resolution replay** over locally captured book streams for three baskets (Fed June rate decision, World-Cup continent, SPX year-end close) to see how violations behave *between* poll snapshots. No orders, no auth.
- **Headline.** The deterministic arithmetic edge is **real and offline-persistent** — violations exist on ~1.3% of events (buy-all) and ~5.4% (sell-all), gross edge ~0.8–1.0c per basket, and at 2-minute poll cadence they survive for **minutes to hours**. But it is **not capturable by a small player**: (1) net-of-fee edge is **negative on essentially every episode** under both fee schedules we modeled; (2) the handful of net-positive episodes are dust-depth weather tails *at* resolution or IPO baskets with capital locked for **568 days**; (3) the millisecond replay shows that on a *liquid* basket (Fed) the real sell-side violations are only **0.10c** and close in a **median ~4 seconds** — below any plausible 5-leg-basket fill latency.
- **Verdict: CLOSE for a small player as a standalone trading system.** The offline arithmetic edge exists but does not survive the live-capture crux (fee + depth + latency). This matches the academic finding that the ~$40M is a **latency game** captured by sophisticated bots; our durable value is *measuring* the gap (already done in [[mm_politics_negrisk_accounting_findings]]), not racing for it. A live-measurement loop is **not** warranted because the offline economics are already net-negative before any live unknown is resolved.

## Why this note exists

[[2026-06-05_novelty_deep_research]] § B.1 flagged NegRisk-basket consistency as "the strongest structural edge — deterministic, not predictive." It is independently corroborated twice: our own merge/split/redeem accounting gap of **$34.6M** ([[mm_politics_negrisk_accounting_findings]]) and the AFT-2025 study "Unravelling the Probabilistic Forest" ([arXiv 2508.03474](https://arxiv.org/abs/2508.03474)): ~$40M extracted Apr-2024→Apr-2025 across 17,218 conditions, 73% of profit from market-rebalancing arb. The open question this note answers is the **live-vs-offline crux**: a deterministic edge that exists in a snapshot is worthless if it closes in 200ms at $5 of depth. So we measure not just whether violations exist, but **how long they persist and how much depth backs the violating quotes** — the two things that decide capturability.

This is distinct from [[block_k4_arb_scan_findings]] (K4), which scanned only our *owned 24h capture universe* for the same two arb types and found 1 rebalancing interval, 0 capturable. This note scans the **entire live NegRisk universe** and adds the persistence/depth dimension K4 lacked.

## The two arbs, with a worked example

**Buy-all (dutch book).** Take the live "Fed Decision in June?" event, 5 legs. If the best asks were: no-change $0.991, +25bps $0.005, −25bps $0.004, +50bps $0.002, −50bps $0.002 → ask-sum = $1.004 > $1, **no violation**. But if a stale quote left the no-change ask at $0.985, ask-sum = $0.998 < $1: buy one of each YES leg for $0.998 total, hold to the June meeting, exactly one leg pays $1 → **+0.2c per basket, risk-free**.

**Sell-all (mint + sell).** If instead the best *bids* sum to more than $1 — e.g. no-change bid $0.991 plus the four tail bids summing to $0.012 → bid-sum = $1.003 — you mint a full YES set (deposit $1 of USDC into the NegRiskAdapter, receive one of every YES) and sell each leg into its bid for $1.003 → **+0.3c per basket**. The mint guarantees you can deliver every leg, so this is also deterministic.

Both are exact arithmetic given executable quotes and enough depth; the entire question is whether the quotes are *real, deep, and slow enough* to hit before they move.

## Method and data

### Live universe poller (`scripts/mm_negrisk_consistency_poll.py`)

- **Universe.** Every active `negRisk` event from the Gamma API, paginated: **2,359 events, 32,436 YES legs** (median 8 legs/event, max 139). Refreshed every ~45 min.
- **Per cycle (~2 min cadence).** Batch-fetch the YES-side CLOB book for every leg via `POST /books` (100 tokens/request). Per event compute `ask_sum` (sum of best asks; `None` unless every leg has a live ask) and `bid_sum` (sum of best bids; missing legs contribute 0, so `bid_sum > 1` is a **conservative confirmed** sell violation), plus the binding (minimum-across-legs) top-of-book depth in both $ and shares. Full per-leg detail is stored whenever a basket is within ±3c of either boundary.
- **This run.** 90 cycles ≈ **3 hours**, 0 batch errors. ~43.6% of event-cycles had a complete ask side (the rest have at least one leg with an empty ask book — typical for deep-tail legs quoted only on one side).

### Millisecond-resolution replay (`scripts/mm_negrisk_basket_replay.py`)

The poller samples every 2 minutes, so its "persistence" is censored at that cadence. To see the true intra-violation dynamics we replayed the **locally captured public WS book streams** (the `mm_stage1_*` lanes, see [[polymarket_data_manifest]]) for three baskets whose legs we happened to capture, using the canonical state builder (`scripts/dali_clob_replay_features.replay`) under K4 discipline (a leg counts as live only if its book state is complete and `book_staleness_seconds <= 5`). We sample every leg's last state on the union of exchange timestamps and collapse violations into exchange-time intervals.

### Fee schedules (assumption ledger — this is the critical knob)

Detection is **gross**; we net fees per episode under two schedules and report both, because the applied Polymarket fee on these markets is itself a live unknown:

| schedule | formula per share per leg | source | label |
|---|---|---|---|
| `repo` | `0.05 · p · (1−p)` | repo-canonical K1/K5-STRESS model | middle estimate |
| `gamma_harsh` | `0.10 · min(p, 1−p)` | Gamma's declared `takerBaseFee = 1000 bps` on these events | harsh ceiling |
| (fee-free) | `0` | Polymarket historically charged 0 taker fee on most CLOB markets | optimistic bound |

Both modeled schedules are **harshest exactly where violations live** — flat many-leg baskets have many legs near the price extremes, and both fee curves are summed across all legs. We therefore lead with the **fee-free optimistic bound** in the verdict and show it still fails on depth + latency + lockup, so the conclusion does not rest on a borrowed/over-harsh fee assumption (per [[CODEX]] § Realism calibration rule 2).

## Results

### 1. Prevalence — violations are real and not rare

Unit of observation: one (event, cycle) snapshot over the live universe; "ever violating" counts distinct events that violated in ≥1 of the 90 cycles.

| direction | events scanned | events ever violating | violating-event share | violating event-cycle share |
|---|---:|---:|---:|---:|
| buy-all (ask-sum < 1) | 2,364 | 30 | 1.27% | 0.39% |
| sell-all (bid-sum > 1) | 2,364 | 127 | 5.37% | 1.79% |

**Read.** Deviations from the $1 coherence condition genuinely occur — sell-side more often than buy-side, on roughly 1-in-20 events at some point over three hours. This confirms the *existence* half of the thesis. Sell-side being more common is expected: tail legs often have a few resting bids above their fair ~0 value, nudging the bid-sum over 1.

### 2. Edge and net-of-fee — the deterministic edge is annihilated by fees

One row per violation **episode** (a maximal run of consecutive violating cycles for one event+direction). CI is a cluster bootstrap over events. `net_repo`/`net_harsh` subtract the two modeled fee schedules.

| direction | episodes | events | gross edge (c) | gross CI (c) | net repo (c) | net harsh (c) | episodes net-repo > 0 |
|---|---:|---:|---:|---|---:|---:|---:|
| buy-all | 61 | 30 | 1.01 | [0.80, 1.25] | **−2.37** | **−7.74** | 3 / 61 |
| sell-all | 291 | 127 | 0.76 | [0.67, 0.85] | **−2.29** | **−7.72** | 4 / 291 |

**Read.** Gross edge is real and CI-positive (~0.8–1.0c/basket). But under *either* modeled fee schedule the mean net edge is **−2 to −8c** and **7 of 352 episodes** survive fees at all. The fee curve dominates because a NegRisk basket trades every leg, and summing `0.05·p(1−p)` (or the harsh `0.10·min(p,1−p)`) across 5–139 legs routinely exceeds a sub-1c gross gap. The edge only matters at all in the **fee-free** world — see §4.

### 3. Persistence and depth — the live-vs-offline crux

| direction | episode duration p50 | p90 | max | still-open at run end (right-censored) | binding depth p50 (baskets at top-of-book) | depth p90 | depth max |
|---|---:|---:|---:|---:|---:|---:|---:|
| buy-all | 359 s | 3,478 s | 5,802 s | 10 / 61 | 6.0 | 17.3 | 589 |
| sell-all | 242 s | 5,802 s | 5,802 s | 45 / 291 | 8.0 | 66.8 | 40,524 |

"Binding depth" = the minimum, across all legs, of the shares quoted at that leg's best price — i.e. how many full baskets you could assemble *without walking past the top of book*.

**Read.** At 2-minute poll cadence, violations **persist for minutes to hours** (median ~5 min, p90 ~1 hour, many still open when the run ended). That looks tradeable — but the persistence is precisely *because the violations are uneconomic*: nobody bothers to arb a 0.5c gap on a basket nobody is trading. The **binding depth is dust**: a median of 6–8 baskets at top-of-book (≈$6–8 of edge-bearing notional). The deep exceptions (the 40,524-basket sell-side max) are 60-leg sports/world-cup baskets where (a) the gross edge is net-negative after fees and (b) you must mint and sell all 60 legs atomically — exactly the convert path the academic study says participants mismodel, and not a small-player operation.

### 4. The millisecond replay — what poll cadence hides

The poller can't see *inside* a 2-minute window. The ms replay can. Coverage of the three captured baskets:

| basket | total legs | legs captured | max concurrent live legs | states | max bid-sum | min ask-sum | confirmed sell violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fed June rate decision | 5 | 5 | 4 | 80,860 | 1.001 | 0.998 | 327 |
| World-Cup continent | 7 | 4 | 3 | 41,617 | 0.761 | 0.774 | 0 |
| SPX year-end close | 6 | 3 | 3 | 17,843 | 0.480 | 0.540 | 0 |

Only the Fed basket reached a near-complete concurrent book (4 of 5 legs; the conservative bid-sum already exceeds 1 with one leg missing, so its sell violations are confirmed). Its violation dynamics:

| metric | value |
|---|---:|
| confirmed sell-violation intervals | 327 |
| edge | 0.10c (p50 = max) |
| duration p50 | **4,044 ms (~4 s)** |
| duration p90 | 10,416 ms (~10 s) |
| duration max | 68,735 ms (~69 s) |
| binding depth p50 | $107 |

**Read — this is the decisive panel.** On a *liquid* near-term basket, the real sell-side violation is only **0.10c**, and even though it is backed by decent depth (~$107), it **closes in a median of ~4 seconds**. A small player submitting a 5-leg basket (5 marketable orders + a mint) cannot reliably fill all legs inside 4 seconds, and 0.10c is below even the fee-free transaction friction. The "minutes-to-hours persistence" from the poller (§3) and the "~4-second persistence" here are **not contradictory** — they describe two different populations: the poll-persistent ones are stale, deep-tail, *uneconomic* mispricings (tiny edge, dust depth, or capital locked for years), while the economically interesting liquid-basket violations close in seconds. Both point the same way for a small player.

### 5. The only net-positive episodes — and why they're not real money

Across all 352 episodes, **7 were net-positive under the repo fee schedule.** Every one fails a different practical gate:

| event | direction | net repo (c) | binding depth (baskets) | days to resolution |
|---|---|---:|---:|---:|
| anthropic-ipo-closing-market-cap | buy-all | 0.89 | 17.3 | **568** |
| highest-temperature-in-munich (Jun 10) | sell-all | 0.85 | **0.12** | −0.05 (resolving) |
| highest-temperature-in-paris (Jun 10) | sell-all | 0.26 | **0.07** | −0.09 (resolving) |
| spacex-ipo-closing-market-cap | buy-all | 0.10 | 589 | **568** |
| xrp-price (Jun 10) | sell-all | 0.09 | 15 | 0.03 |

**Read.** The depth-bearing positives (Anthropic/SpaceX IPO market-cap baskets, ~17–589 baskets) lock capital for **568 days** to earn ~1c — an annualized return below cash. The near-term positives (Munich/Paris weather tails) are **0.07–0.12 baskets** of depth (cents of notional) and are mid-resolution. None is a deployable edge.

## Charts

![Sum-of-best-asks and sum-of-best-bids across the NegRisk universe](../../data/analysis/plots/market_making/mm_negrisk_consistency_sum_histograms.png)

*Distribution of basket price-sums at the last cycle. X-axis: sum of best asks ($ to buy every leg, left) and sum of best bids ($ to sell every leg, right); red line at $1.00 is the coherence point. Most baskets sit just above $1 on the ask side (the spread tax) and just below on the bid side; the mass that crosses $1 (asks < 1, bids > 1) is the violating tail. The tail is thin and close to the line — i.e. small gross deviations, consistent with §2.*

![How long violations survive at poll cadence](../../data/analysis/plots/market_making/mm_negrisk_consistency_persistence.png)

*Episode duration at 2-minute poll cadence, split into episodes that closed within the run vs those still open at run end (right-censored). Many violations persist for the full window — but, per §3–4, this reflects uneconomic deep-tail mispricings, not capturable depth. Durations are a lower bound at 120s resolution.*

![Violation size vs executable depth](../../data/analysis/plots/market_making/mm_negrisk_consistency_edge_vs_depth.png)

*Each point is an episode: gross edge (cents/basket, y) vs baskets assemblable at top-of-book (log x). The cloud sits at low depth and low edge; nothing is simultaneously high-edge and deep. The few deep points are large multi-leg baskets whose edge is net-negative after fees.*

![Gross vs net-of-fee edge by family](../../data/analysis/plots/market_making/mm_negrisk_consistency_gross_vs_net.png)

*Gross edge vs net under the repo (5%·p(1−p)) and harsh (10%·min(p,1−p)) fee schedules, by direction and event family (weather, sports, politics, economics, crypto, other). Gross bars are positive; both net bars go negative across every family — the fee, summed over all legs, exceeds the sub-1c gross gap everywhere.*

## Realism calibration — assumption ledger and verdict

Following [[CODEX]] § Realism calibration, separating what is modeled from what only live trading can resolve:

**(a) Modeled assumptions (offline).**
- Fee schedules `repo` and `gamma_harsh` (and a fee-free optimistic bound). The applied fee is itself uncertain; we lead with fee-free so the verdict does not rest on an over-harsh fee.
- Top-of-book depth = min shares across legs (conservative — ignores deeper levels, but you can't fill a basket faster than its thinnest leg).
- Detection is YES-side only (the two real arbs are buy-all-YES and mint+sell-all-YES); NO-side merge variants would give the same coherence condition.
- `bid_sum` with missing legs contributing 0 is a conservative lower bound, so confirmed sell violations are genuine.
- Empirical base rates from a 3-hour full-universe sweep + ms replay of three captured baskets.

**(b) Live-only unknowns (not resolved here, and not worth resolving given (a)).**
- Real fill latency for an atomic N-leg basket (the ms replay says the window is ~4s on liquid baskets — likely shorter than our fill time).
- Atomic mint/merge via the NegRiskAdapter, and its gas cost (not modeled; non-trivial on a 1c gross edge).
- Slippage walking the book beyond best level on the deep multi-leg baskets.
- Competition from the latency bots that the academic study credits with 73% of the $40M.
- Whether displayed quotes are real or vanish on a marketable order (adverse selection / quote fade).

**Statistical survival ≠ economic materiality (rule 4).** The gross edge is CI-positive (statistically real). The deployable per-basket edge after fees is **negative** under both modeled schedules, and even fee-free it is bounded by dust depth (median ~7 baskets ≈ cents) on near-term markets and by 568-day capital lockup on the deep ones. This is "deterministic arithmetic edge exists offline, economically ~zero and uncapturable live for a small player."

**Verdict: CLOSE for a small player as a standalone trading system.**
- The deterministic offline edge **exists and persists** — the existence half of the thesis (and our $34.6M accounting gap) is confirmed.
- It is **not capturable live by us**: net-negative after fees on ~98% of episodes; the fee-free positives are dust-depth or 568-day-lockup; and the ms replay shows liquid-basket violations close in ~4s at 0.10c.
- This is consistent with the academic framing: the ~$40M is a **latency game** won by sophisticated players with atomic execution and ms reaction; our edge belongs to **measuring** the gap (done — [[mm_politics_negrisk_accounting_findings]]), not racing for it.
- A **live-measurement loop is NOT warranted** here, unlike the politics structured-maker cell: there, the offline economics were positive and only live fill/queue unknowns remained; here the offline economics are already net-negative before any live unknown, so instrumenting mint/merge fill latency would only confirm a closed door (per [[CODEX]] § Realism calibration rule 3, reopening requires real positive evidence, which we do not have).

## Decision and next step

- **Decision:** CLOSE the NegRisk basket-consistency arb as a deployable edge for this book. It is a deterministic-but-uncapturable latency game for a small player; do not build a basket-arb executor.
- **What stays open (unchanged):** the NegRisk *accounting* work ([[mm_politics_negrisk_accounting_findings]]) and the politics structured-maker **live loop** are separate and not affected — those harvest the maker spread / structured-flow edge, not the basket-consistency arb, and their offline economics are positive.
- **Concrete next action:** none for this branch. Fold the scanner (`scripts/mm_negrisk_consistency_poll.py`) into the toolbox as a cheap monitor that could be re-run if Polymarket ever (a) confirms genuinely fee-free NegRisk trading *and* (b) we gain atomic basket execution — both of which would be required before reopening. Otherwise this is a confirmed close.

## Artifacts

- Scanner: `scripts/mm_negrisk_consistency_poll.py` · analyzer: `scripts/mm_negrisk_consistency_analyze.py` · ms replay: `scripts/mm_negrisk_basket_replay.py` · plots: `scripts/mm_negrisk_consistency_plots.py`
- Result CSVs (under `data/analysis/csv_outputs/market_making/`): `mm_negrisk_consistency_prevalence.csv`, `mm_negrisk_consistency_episodes.csv`, `mm_negrisk_consistency_event_summary.csv`, `mm_negrisk_basket_replay_summary.csv`, `mm_negrisk_basket_replay_intervals.csv`
- Raw scan: `data/live_clob/mm_negrisk_consistency_scan/<run_id>/` (universe snapshot + `cycles.jsonl` + `cycle_meta.jsonl`); filtered replay captures: `data/analysis/negrisk_replay/`
- Plots (under `data/analysis/plots/market_making/`): `mm_negrisk_consistency_sum_histograms.png`, `mm_negrisk_consistency_persistence.png`, `mm_negrisk_consistency_edge_vs_depth.png`, `mm_negrisk_consistency_gross_vs_net.png`
