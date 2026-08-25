---
title: "MM Task 5 / 5.1 — Methodology Build-Up, the Ladder, and What We're Actually Testing"
created: 2026-07-10
status: active
owner: justin
project: polymarket
hubs:
  - strat_market_making
  - mm_backtesting_methodology_explainer
tags:
  - market-making
  - methodology
  - explainer
  - avellaneda-stoikov
  - vpin
  - adverse-selection
  - cpcv
  - queue-model
---

> **HISTORICAL EVIDENCE (2026-08-25).** Detailed findings behind the active market-making project, kept for verification. Read the de-jargoned canon surface first — [[strat_market_making]] + [[mm_model]] — and treat every number here as preliminary per the hub reliability ledger.


# MM Task 5 / 5.1 — Methodology Build-Up, the Ladder, and What We're Actually Testing

> Hubs: [[strat_market_making]] · sibling of [[mm_backtesting_methodology_explainer]] (that note covers the *engine*; this note covers the *Task-5/5.1 evaluation logic, the ladder, and the concepts*). Findings of record: [[mm_task5_inventory_quoter_findings]] (consolidated) · [[mm_task5_1_neutral_quoter_cpcv_findings]] (full 5.1). Code: `mm_engine/strategies.py`, `mm_engine/queue_models.py`, `mm_eval/cpcv.py`, `mm_eval/tape.py`.
>
> **Purpose.** A from-first-principles explainer built during Justin's manual audit, because the ladder, the rungs, the tests, and "which markets we action on" had become hard to hold in one head. **Task 5.1 is canonical**; Task 5 appears only to explain what was fixed and why. No profitability claim — measurement only until Join 2.

## How to read this note

Two numbered spines run through it:

- **M1–M8 — the methodology build-up.** A concept ladder: each step exists because the previous one raised a question it couldn't answer. Read top to bottom once and the whole evaluation should click.
- **R1–R9 — the results.** What the machine actually found, one row each, traceable to a chart or table in the findings of record.

Part 1 defines the vocabulary. Part 2 is M1–M8. Part 3 is the controllers and the ladder. Part 4 is R1–R9. Part 5 is the live status ("what markets, what tests, what ships"). Part 6 is the standing verify-list. The appendix carries the formulas.

---

## Part 1 — Vocabulary (the concepts everything else is built from)

**Maker vs taker.** A *taker* crosses the spread to trade now (pays the spread). A *maker* posts resting limit orders and waits to be hit (earns the spread, but only when someone chooses to trade against it). Everything here is about *making*.

**The touch.** The best bid and best ask — the top of the book. "Joining the touch" means resting your bid at the current best bid and your ask at the current best ask. "Off the touch" means resting deeper (wider), where you fill less often.

**Mid vs microprice.** Mid = (best bid + best ask) / 2. **Microprice** is the size-weighted mid: it leans toward the side with *less* size, because a thin ask / heavy bid says the next move is likely up. It is the 0-parameter fair value the Task-5 quoters center on (formula in appendix).

**Spread capture vs adverse selection — the two things that decide a maker's PnL.**
- *Spread capture* = the edge you earn buying at the bid and selling at the ask around an unchanged fair value.
- *Adverse selection* = the loss from being filled right before the price moves against you, because the person hitting you knew something. A maker's net PnL is spread capture **minus** adverse selection. On Polymarket the adverse-selection term is dominated by **resolution** — the price jumping to 0 or 1 — not by ordinary diffusion.

**Inventory and carry.** *Inventory* (`q`) is your signed net position (long +, short −). A maker who keeps getting hit on one side *accumulates* inventory. *Carry-to-resolution* = holding inventory until the market settles and collecting the payoff, instead of trading out. Carry is only safe if the inventory is *balanced* (e.g. a complementary NegRisk basket where one leg always pays).

**Toxicity.** Shorthand for "flow that is about to move the price against the resting side." We measure it two ways (the "two lenses"):
- **VPIN (Lens 1)** — volume-clock order-flow imbalance. Chop trades into equal-*volume* buckets; a bucket that is nearly all buys (or all sells) is toxic. It *leads* the move and names which side is exposed. Sweep-distance weighting up-weights trades that punch through multiple book levels (a LOTECH idea).
- **AS-z (Lens 2)** — post-fill adverse-selection z-score. After a fill, did the mid drift against us more than a calm baseline would predict? `z < −2` flags it. It *confirms* toxicity and is the most volume-efficient flag we measured.

**Cohort.** A market's *type*, tagged from its **lead-in window only** (before the period we score, so it can never leak): *aggressive vs benign* on **sweep share** (fraction of early volume that sweeps multiple levels), *thick vs thin* on flow. **Benign = low sweep share = calm/uninformed early flow.** This is a property we can know in advance and use to *select* markets.

**τ-regime (time-to-resolution).** A bucket of how far a market is from settling: mid-life / approach / endgame (politics), pre / in-play (esports). It is a **reporting** axis — we slice results by it — but **never a split axis** (see M4 for why that distinction is the whole Task-5 story).

---

## Part 2 — M1–M8: the methodology, built up

Each step is here because the previous one left a hole.

**M1 — What are we even measuring?** Task 4 showed a symmetric fixed-spread quoter looked profitable — but the "profit" was a **directional inventory bet** (it drifted long or short and rode the market), not spread capture. So the real question isn't "did it make money," it's "**does it capture spread net of adverse selection, without a hidden directional bet?**" That forces an inventory-aware controller and an honest PnL.

**M2 — Why naive PnL lies (costing).** If you mark leftover inventory at the *mid*, you book profit you can't realize — a resting position is only worth what you can *exit* at (the bid if long, the ask if short). Mark-to-mid is exactly what produced the phantom K-PEG "+759 bps." So all PnL here is **costed**: realized round-trips **plus** leftover inventory marked at the executable touch. This is what "honest ¢" means — cents/contract you'd actually keep. *(This raised M3: it assumes you actually get the fills.)*

**M3 — Do we even get filled? (the queue model).** A backtest can't see our own queue position from anonymous public L2, and *when* you fill decides whether you're picked off. So fills are **bracketed** by three models that differ only in how a cancel is attributed ahead-vs-behind us: **Optimistic** (cancels ahead → most fills → upper bound), **RiskAverse/pessimistic** (cancels behind → fewest fills → lower bound), **ProbQueue(f)** (probabilistic split, default f=0.5 → the middle, *read as a bracket midpoint, not a point estimate*). `f` is calibrated from our own live fills — which is a first-order reason to run Join 2. *(This raised M4: even honest, filled PnL is meaningless if measured on the wrong data.)*

**M4 — On what data do we judge generalisation? (the split — the crux of the whole arc).** Task 5 used a **single calendar cut through the same markets**: in-sample = 06-19→23, out-of-sample = 06-24→30. But a market's *calm mid-life* fell in-sample and its *toxic pre-resolution endgame* fell out-of-sample — so the IS→OOS collapse wasn't overfitting, it was a **regime confound**. The tell: the drop was near-uniform across every config and *scaled with the inventory cap* (overfitting punishes the selected config hardest; this punished everything together). **The fix (5.1, canonical): whole-market nested CPCV** — the split unit is the whole NegRisk event / match, so a market's entire lifecycle stays on one side; 6 cohort-balanced folds, every C(6,2)=15 held-out pair a path, inner-fold selects knobs / outer-fold estimates. Now configs transfer ~1:1 IS→OOS. *(This raised M5: how do you describe a market's type without leaking?)*

**M5 — Describing markets without leaking (cohorts + τ as a report axis).** Cohorts are tagged from the **lead-in window only**, so the tag is knowable before the scored period — leakage-safe, and usable as a *selection* rule. τ-regime is reported (we slice the surface by mid-life/approach/endgame) but is **never** a split axis — splitting on τ is exactly the Task-5 mistake. *(This raised M6: given honest, filled, well-split, well-described data — how do you decide a knob earns its place?)*

**M6 — Deciding a knob earns its place (the ladder + keep-gate).** Configs are ordered into a **ladder**: each rung must beat the previous *kept* rung. A rung is **KEPT** only if its per-group paired OOS delta vs the previous kept config has a group-cluster bootstrap **lower CI > 0** — evaluated under the **pessimistic queue**. Point-up but CI-through-zero = **FRAGILE** (not kept). This is deliberately a pessimistic queue (worst-case fills) so we never certify an edge that only exists under favorable fill assumptions we can't justify. *(Caveat flagged in Part 6: "fewest fills" is not automatically "worst PnL," so this conservatism deserves per-category scrutiny.)*

**M7 — Did we just get lucky selecting? (the overfitting audit).** Even a clean ladder can select a lucky config. Three standard gates (shared `infrastructure/validation/overfitting_audit.py`): **PBO** (Probability of Backtest Overfitting — how often the IS-best config is below-median OOS; 0.5 = luck), **DSR** (Deflated Sharpe — is the Sharpe real after penalising the number of configs tried), **White's Reality Check** (are the best configs unlikely under pure snooping). *(This raised M8: what does "it survives" actually buy us?)*

**M8 — Statistical survival vs economic materiality (and concurrency).** A rung can be CI-certified and still worthless: esports A-S rung 2 is certified but its *level* ≈ 0, so the certified quantity is **avoided bleed, not profit**. Separately, **concurrency**: politics markets all trade at once (overlap 1.00), so their PnLs are news-correlated → the group bootstrap's *effective* N is below its nominal K=11 → **the OOS CIs are somewhat overconfident (too narrow)**. Consequence: at this sample size the honest weight belongs on **IS monotonicity + mechanism + PBO 0.00**, with the OOS CI as a directional check — and "nothing certifies" means "**not enough independent data to certify**," not "no signal." Concurrency is also an *edge lever* (cross-market inventory netting), not just a caveat. This is why the next step is a **live measurement loop**, not more folds.

---

## Part 3 — The controllers and the ladder

### The controllers (what each quoter *is*)

All conform to the frozen `Strategy` protocol `quote(book, inventory, params)` — every knob is injected via `params`, the interface is untouched, and all state is derived from the event stream (deterministic replay).

| controller | class | one-line description |
|---|---|---|
| **Symmetric baseline** | `SymmetricQuoter` | mid ± fixed half-spread, **inventory-blind**. The A/B floor. Closest backtest analog to the "join-the-touch" live quoter. |
| **NSQ (Task 5.1, canonical)** | `NeutralSpikeQuoter` | microprice skew + tight cap + two-lens gate + graduated response + asymmetric repricing + OFI dampening. **No calendar flatten** — carries balanced inventory, refuses one-sided stacking. |
| **A-S ladder** | `ASQuoter` | Avellaneda-Stoikov: replaces the hand-tuned skew with the derived `γσ²τ`, and (rung 2) the derived optimal half-spread. Adverse-selection-**blind** by construction. |
| **Basket-carry** | `BasketCarryQuoter` | joint quoting across a NegRisk event's legs, skewing each leg toward a balanced basket, carrying to expiry instead of flattening. |

> **NSQ vs "join-the-touch."** NSQ is an *active* controller — its quotes move with inventory and toxicity. "Join-the-touch" (the naive quoter in the June live-loop design) just sits at the current best bid/ask, refreshes on a timer, no skew, no gate. The built Join-2 machinery would run the naive quoter (tests carry/dodgeability); 5.1's shipping candidate is NSQ (tests the controller). **Deciding which one Join 2 actually runs is the open scope question.**

### The ladder (each rung must beat the previous *kept* rung)

The rungs are cumulative — you add one mechanism at a time and re-gate, so any surviving edge is attributable to a specific mechanism rather than the whole bundle.

*politics ladder (defensive rungs gated ON):*

| rung | what it adds vs the rung below |
|---|---|
| baseline | — (Symmetric floor) |
| NSQ core | microprice reservation `r = micro − k·q` + tight `inv_cap` (reduce-only at cap) |
| + two-lens | VPIN (Lens 1) + AS-z (Lens 2) gate with graduated response |
| + asymmetry | asymmetric repricing — chase a moving market slowly, withdraw instantly |
| **+ size-dampen (full stack)** | continuous OFI size-dampening on the pressured side |
| A-S rung 1 | replace hand-tuned `k` with derived skew `γσ²τ` |
| A-S rung 2 | + A-S optimal half-spread `γσ²τ/2 + (1/γ)·ln(1+γ/k_arr)` |
| A-S rung 3 | rung 2 + the toxicity overlay (the piece A-S omits) |
| basket-carry | joint NegRisk-leg quoting, balance-skew, carry to expiry |

*esports ladder (defensive rungs diagnostic, per the pre-registered v0 gate):* baseline → NSQ core → **A-S rung 2** → basket-carry.

**The graduated response (the heart of NSQ's gate).** Lens 1 only (directional flow) → *suspend new quotes on the exposed side* (stop one-sided stacking; the other side keeps quoting and passively reduces). Lens 2 only (adverse drift) → *shrink opening size* to a fraction. **Both** (or at cap) → *pull opening quotes, rest a reduce-only quote at a widened offset* (passive unwind — never dump at the touch, never cross).

**Why A-S is adverse-selection-blind and why that matters (the Q1 insight).** A-S's optimal half-spread is, mechanically, a **fill-suppression knob** (a wider quote fills less). Fill suppression only pays where fills are net-toxic. Its τ-term *shrinks* toward expiry — backwards for a resolving market, where adverse selection *grows* toward expiry — so the flatten/gate must stay a separate overlay with the opposite sign. Result: A-S widening is coincidentally right for esports (in-play fills are toxic → sitting out avoids the bleed) and wrong for politics (approach-window fills are net-positive → sitting out forfeits the edge). See R6.

### Assumptions & institutional origin (why each is *believed* to work — and where the belief breaks on PM)

The user's audit question in one place: every mechanism carries a theory that says "this works *if* the world looks like X." Listing X makes the failures legible.

| mechanism | origin | assumes (to work) | holds on PM? |
|---|---|---|---|
| **Microprice fair value** | Stoikov (2018), market-microstructure standard | the size-weighted mid is a better one-step predictor of the next mid than the raw mid | **Yes** — LOTECH validation: MAE **0.75 ticks vs mid's 8.82** (~12× closer), holding through spike and post-sweep. This is the strongest-supported assumption in the stack. |
| **Linear inventory skew** (`r = micro − k·q`) | Ho-Stoll (1981) inventory model; the reservation-price idea | shading quotes against inventory mean-reverts your position at a tolerable cost in spread | **Partly** — it does bound inventory (R1), but the *right* `k` is regime-dependent; a fixed `k` under-reacts in spikes. NSQ adds the gate precisely because linear skew alone is too slow. |
| **Tight position cap** | risk-limit convention | ruin comes from unbounded one-sided inventory; a hard cap truncates the tail | **Yes** — mechanically true; the cap is what stops the −1,800 stack (R1). Cost: caps also cap upside on genuine two-sided flow. |
| **VPIN (Lens 1)** | Easley, López de Prado & O'Hara (2012), *"Flow Toxicity and Liquidity in a High-Frequency World"* — **an institutional flow-toxicity metric** | order-flow imbalance on a **volume clock** leads price, so toxic flow is visible *before* the adverse move | **Directionally** — it does lead (R8: VPIN jumps ~30 min before the spike). But alone it's a weak *skip* rule (v0: +0.07¢ politics, −0.05¢ esports) — it names the exposed side, it doesn't decide to stop. |
| **AS-z (Lens 2)** | adverse-selection / markout monitoring (execution-desk standard) | if your *own* fills systematically drift against you vs a calm baseline, you're being adversely selected | **Yes, best-supported gate** — most efficient flag measured (+0.22¢/+0.59¢ rescue on ~6–7% of volume). Assumes a stable calm baseline; NSQ freezes the baseline during flags so a spike can't normalize itself. |
| **OFI dampening** | Cont, Kukanov & Stoikov (2014), order-flow-imbalance price-impact model | short-horizon price moves are ~linear in signed order-flow imbalance, so fade size as OFI builds | **Plausible, unproven here** — a diagnostic rung; helps the point estimate but never independently certified. |
| **Avellaneda-Stoikov** (rungs 1–3) | **Avellaneda & Stoikov (2008)** — the canonical academic/institutional optimal-MM model (HJB solution) | (1) mid = **driftless arithmetic Brownian motion, constant σ**; (2) fills arrive **Poisson**, intensity `λ = A·e^{−k·δ}` decaying with distance from mid, from **uninformed** counterparties (**no adverse selection**); (3) a finite **liquidation horizon** τ with a terminal inventory penalty | **Mostly no** — PM prices *jump* to 0/1 at resolution (not diffusion), σ is regime-switching, and adverse selection is the dominant risk (exactly what A-S omits). Its τ-term shrinks toward expiry, backwards for a resolving market. It "works" for esports only because its wide spread coincidentally suppresses toxic fills — right answer, wrong mechanism (R6). |
| **Basket-carry** | NegRisk structural arithmetic (complementary outcomes sum to $1) | a balanced complementary basket has a **floored** resolution payoff (one leg always pays), so carrying it is low-variance | **Structurally yes, empirically no edge** — it carries safely but loses to A-S rung 2 head-to-head (R6); partial-partition carry is still a slow inventory bet. |

**The one-sentence version:** the assumptions that *hold* on PM (microprice, cap, AS-z) are microstructure/execution facts; the one that *breaks* (A-S's no-adverse-selection diffusion world) is the classical model built for equities/FX, which is why importing it wholesale fails except where its side-effect happens to help.

---

## Part 4 — The markets: what we tested, how we classified them, with examples

### The universe

Both categories are the full 17.9-day R2 L2 capture (2026-06-19 → 07-07), narrowed to the **top-32 most-traded *quotable* tokens** per category (≥150 prints, average price 5–95¢ — the Task-4 quotability rule), which roll up into the whole-market split units:

| universe | days | tokens | markets | trades | split-unit groups |
|---|---|---|---|---|---|
| politics_negrisk | 17.9 | 1,261 | 663 | 541,673 | **11 NegRisk event groups** |
| esports | 17.9 | 4,942 | 2,640 | 1,132,590 | **16 match groups** |

fee = 0, rebate = 0 (captured truth); latency 0 ms (fair for politics, an optimistic upper bound for esports). One politics token (the **Netanyahu outright**, 23.2M L2 events) is excluded by an ex-ante replay-feasibility cap (>8M events ≈ 10 GB RAM / 10 CPU-h per sweep); its *group* keeps its other legs, so the split-unit count is unchanged.

### Two ways we classify a market

**(1) Cohort — the market's *type*, from its lead-in window only.** Structural features (flow rate, book depth, **sweep share**, spread, 1-min mid-vol, average price, queue turnover) are computed only over the first `min(24 h, 25% of observed span)`, so the tag is knowable in advance and can never leak the outcome. Two axes:

- **`cohort_aggr` — aggressive vs benign, split at the category-median *sweep share*** (fraction of trade volume that executes *beyond the touch*, i.e. sweeps multiple book levels). High sweep share = aggressive/informed participants; low = benign/uninformed (retail-like). **This is the primary surface axis.**
- **`cohort_liq` — thick vs thin, by flow.**

**(2) τ-regime — how close to resolution**, a *reporting* slice only (never a split axis): politics mid-life (>48 h) / approach (6–48 h) / endgame (<6 h); esports pre (>6 h) / in-play (<6 h).

### Concrete examples (so the abstractions have faces)

| group (example) | category | cohort read | why it matters here |
|---|---|---|---|
| **Musk-tweet-count weekly** | politics | **benign** (quiet, wide two-sided flow) | The archetype of the certified-toxic case: calm most of its life, then picked off in the final hours as the count resolves. It's where the **−3.13¢ benign-endgame bleed** lives — and the single-market illustration of the Task-5 confound (calm week trained the knobs, resolution week judged them). |
| **Fed-July** (rate decision) | politics | **aggressive / thick** (long-dated outright) | Deep, professionally-made book; mildly negative mid-life (−1.11¢), endgame mostly beyond the capture window. Represents the aggressive cohort where A-S widening just forfeits spread. |
| **Iran-meeting, Lula-2026** | politics | mixed | Used in the worked CPCV split example (held-out folds) — illustrate that a fold mixes early/late-starting, aggressive/benign groups. |
| **esports matches** (per match = one group) | esports | mostly aggressive in-play | In-play flow is genuinely toxic (a viewer knows the game state before the book) — the baseline bleeds −3.45¢ filling into it; A-S rung 2's wide, selective quote *sits out* the toxic window. |

The **benign politics approach window** (+2.66¢) and the **benign politics endgame** (−3.13¢) are the two cells that decide the whole politics story — same cohort, opposite τ. That is the market structure a live loop must exploit (quote the approach) and defend (gate the endgame).

---

## Part 5 — R1–R9: what the machine found, and *why* each worked or failed

Each result ties to a chart or table (Part 6). "Why" is the evaluation the user asked for — not just the number, but the mechanism behind it.

**R1 — Inventory control works mechanically.** Baseline ran a 14,053-contract one-sided book (ended 12,948 short); through the biggest spike it stacked to −1,800, while NSQ held ±270 and crossed ~flat. *Why it works:* the cap truncates the tail and the gate suspends the exposed side, so one-directional flow can't accumulate. *Chart:* `inventory_path`.

**R2 — Costing kills the baseline.** Charging exits, the symmetric quoter loses OOS everywhere (Task 5: politics −0.96¢, esports −36.4¢ — one token carried 33.7k contracts to resolution for −$17.3k; 5.1 whole-market: −0.13¢ / −3.45¢). *Why it fails:* mark-to-mid booked unrealizable carry; at the executable touch the "profit" was exit cost all along. *Lesson:* never quote the uncontrolled baseline, never cite naive PnL.

**R3 — The diagnosis: Task 5's "no edge" was the *test*, not the strategy.** In `mm_task5_is_selection.csv` the IS→OOS drop was near-uniform across configs and *scaled with the cap*. *Why that's diagnostic:* overfitting punishes the *selected* config hardest; a whole-surface, cap-scaled drop instead means the OOS window was categorically more toxic (the endgame). Regime confound, not overfit. *Chart:* `mm_task5_ladder_oos` (the Task-5 collapse).

**R4 — The fix works: ~1:1 transfer.** Under whole-market nested CPCV every config sits on the training≈held-out diagonal; the cap-scaled collapse is gone; the baseline re-attributes to −0.13¢. *Why:* holding whole lifecycles out means IS and OOS both contain calm+endgame — no regime is quarantined to one side. *Chart:* `inner_outer_by_cap`.

**R5 — v0 toxicity gate: where defences are *allowed*.** Pre-registered rule (before the numbers): wire the defensive knobs for a category iff a (cohort × τ) cell is point-negative AND leave-one-market-out-negative ≥ 50%. Result: politics **benign × endgame** certified toxic (−0.64¢, CI [−1.02, −0.05], LOMO 100%); esports has no net-negative cell → withhold. *Why it matters:* it stops us bolting defences onto markets that don't need them (esports in-play flow is positive), and it makes the AS-z flag the empirically-best rescuer. *Charts:* `v0_surface`, `mm_task5_v0_ttr_regime`.

**R6 — The ladder (the core result).** *Politics (11 groups):* the full NSQ stack is the only config positive on every path (+0.29¢ honest, +0.65¢ vs baseline), improving monotonically in component order — but every delta's CI spans zero at K=11 → all **FRAGILE**, nothing ships. *Why it fails to certify:* power, not signal — too few independent groups, and concurrency makes even those CIs overconfident. *Esports (16 groups):* A-S rung 2 is **CERTIFIED** (+6.66¢ vs baseline, [+1.77, +11.81], uniform across all 16 groups). *Why it works:* its derived wide spread is selective → it sits out toxic in-play fills; uniformity (not magnitude) is what tightened the CI where the raw skew quoter's whale-skewed +4.70¢ failed. *But* its level ≈ +0.03¢ → certified quantity is **bleed-avoidance, not profit**. This flips Task-5's "A-S certified worse" (a calendar artifact).

**R7 — The (cohort × τ) surface (held-out).** Politics benign markets earn **+2.66¢ in approach** and give back **−3.13¢ in endgame**; aggressive outrights mildly negative mid-life. *Why it's the most useful output:* it's a *market-selection map* drawn from held-out data — quote benign approach windows, gate/avoid benign endgames — independent of any controller. *Chart:* `surface_heatmap`.

**R8 — The spike and the leading signal.** VPIN jumped ~0→0.57 about 30 min *before* the main move; AS-z confirmed during it (z→−13); the conjunction fired the full defensive response through the toxic window. *Why it matters:* it's the mechanistic proof the two-lens design is timed right (lead + confirm), replicating the LOTECH post-event finding on PM data. One episode — illustration, not evidence. *Chart:* `toxicity_trace`.

**R9 — Overfitting audit.** PBO **0.00** both (selection transfers — vs Task 5's 0.5); DSR **fails** both (19 daily obs vs a ~43-effective-trial haircut — sample length, not signal); White's RC **p≈0.03** both (best configs unlikely pure snooping). *Why the split:* PBO asks "did selection generalize" (yes); DSR asks "is the Sharpe real given how many configs we tried" (can't tell on 18 days); RC asks "is the best config's mean > 0 after snooping correction" (yes, supporting). Together: config-level skill is real, economic level uncertified → live measurement.

---

## Part 6 — Every chart and table, explained

Eight charts exist. Six are canonical (Task 5.1); two are Task 5, kept only to *see* what was fixed. Each entry: what it plots, the intuition, and what to notice.

### The two market-move-with-fills charts (the ones you asked about)

These are the closest we have to "market moves alongside our strategies and fills." Both are the *same* politics spike (~22:50 UTC, the sample's largest), viewed two ways.

![Toxicity trace through the spike](../../data/analysis/plots/market_making/mm_task5_1_toxicity_trace.png)

**`toxicity_trace` — do the signals fire in time?** Three stacked panels on one timeline: top = mid price (0.18 → 0.75 in minutes); middle = Lens 1 volume-clock VPIN (red dots = `directional_flag` firing); bottom = Lens 2 post-fill drift z-score (red dashed = the −2 threshold). *Intuition:* a good toxicity gate must **lead** (warn before) and **confirm** (verify during). *What to notice:* VPIN jumps ~0→0.57 **~30 min before** the move (the lead), AS-z plunges to −13 **during** it (the confirm). The two firing together is what triggers NSQ's full defensive response. One episode — illustration of the mechanism, not statistical evidence.

![Inventory path: baseline vs NeutralSpikeQuoter](../../data/analysis/plots/market_making/mm_task5_1_inventory_path.png)

**`inventory_path` — what the two strategies *do* in that spike.** Left axis = net position (contracts), right axis (grey) = mid; red = symmetric baseline, blue = NSQ. *Intuition:* net position is the running sum of fills, so this *is* the fills picture — a falling line means you're being hit on your asks (accumulating short). *What to notice:* both start net short from pre-spike buy pressure ("negative to begin with"); at the spike the baseline sells into the entire rally, stacking to **−1,800 short as price runs to 0.75** (maximally wrong-way — picked off), while NSQ suspends its exposed SELL side and rides through at **±270**, crossing ~flat. This is R1 made visual.

> *If you want a deeper fills view* — every individual fill plotted as a dot on the price path, colored by side, sized by adverse markout — that's a new chart I can generate from the per-fill telemetry (`mm_eval` logs fills per token×config). Say the word and I'll add a per-market "tape + our fills" panel.

### The four evaluation charts (Task 5.1, canonical)

![Split diagnosis](../../data/analysis/plots/market_making/mm_task5_1_split_diagnosis.png)

**`split_diagnosis` — why the old split was broken.** Each horizontal bar = one event group's lifecycle in calendar time; black head = its leakage-safe lead-in; colors = CPCV folds; red dashed line = Task 5's 06-24 calendar cut. *What to notice:* most politics bars *straddle* the red line — the old cut put the same market's calm phase left (IS) and its endgame right (OOS). The new folds keep every bar whole. This is M4 in one picture.

![Inner→outer transfer by cap](../../data/analysis/plots/market_making/mm_task5_1_inner_outer_by_cap.png)

**`inner_outer_by_cap` — did the fix work?** Each dot = one config; x = its ¢/contract on CPCV *training* groups, y = on *held-out* groups; color = inventory cap. *Intuition:* if configs generalize, dots sit on the 45° diagonal. *What to notice:* they do, in both categories — the cap-scaled IS→OOS collapse that indicted Task 5 has vanished. The baseline (yellow) sits *below* the diagonal — its directional inventory bet makes even honest transfer noisy. This is R4.

![v0 toxicity surface](../../data/analysis/plots/market_making/mm_task5_1_v0_surface.png)

**`v0_surface` — where is toxicity net-negative?** The (cohort × τ-regime) markout surface on baseline fills, with CIs. *What to notice:* exactly one cell is certified net-negative — politics **benign × endgame** (−0.64¢) — which is what earns politics its defensive-knob wiring; every esports cell is positive (→ withhold). This is R5.

![Kept-rung performance surface](../../data/analysis/plots/market_making/mm_task5_1_surface_heatmap.png)

**`surface_heatmap` — where the edge actually lives.** Costed ¢/contract of the kept rung per (cohort × τ) cell, pooled over groups. *What to notice (politics, kept = baseline, so this is the market's own structure):* benign **+2.66¢ in approach**, **−3.13¢ in endgame**; aggressive outrights mildly negative mid-life. This is the market-selection map — R7.

### The two Task-5 charts (superseded — kept to show the fix)

![Task-5 ladder OOS](../../data/analysis/plots/market_making/mm_task5_ladder_oos.png)

**`mm_task5_ladder_oos` — the collapse itself.** Per rung: open square = in-sample (selection-window) ¢; filled markers = the OOS queue bracket (▲ Optimistic / ● Prob / ▼ RiskAverse). *What to notice:* the IS squares sit well to the *right* of the OOS markers for the promising rungs (e.g. politics v1: IS ~+0.35 vs OOS ~+0.03) — the in-sample-to-held-out drop that Task 5 read as "no edge." Also note the three OOS markers are tightly clustered — the *queue* assumption was never the problem, the *split* was. Under 5.1 (`inner_outer_by_cap`) that IS↔OOS gap closes.

![Task-5 v0 TTR regime](../../data/analysis/plots/market_making/mm_task5_v0_ttr_regime.png)

**`mm_task5_v0_ttr_regime` — the near-expiry toxicity precursor.** Net markout(30s) (blue) and adverse selection (red dashed) vs time-to-resolution buckets. *What to notice:* politics markout is deeply negative in the **<6 h** bucket (−1.0¢) and positive through mid-life (peaking ~+0.9¢ at 7–30 d) — the "edge in mid-life, eaten near expiry" finding that 5.1's `v0_surface` later sharpened to *benign-cohort* endgames. Esports markout is positive across all buckets — the evidence for withholding esports defences. This is the Task-5 seed of R5/R7.

### How to read the ladder table (the one you found unclear)

The full ladder tables live in [[mm_task5_1_neutral_quoter_cpcv_findings]] §6; the compact version is R6. Column-by-column:

- **rung / modal config** — the mechanism added at this step, and the specific knob values most often inner-selected (what would actually ship).
- **honest pooled ¢** — volume-pooled costed OOS ¢/contract (the "honest ¢" of Part M2), pessimistic queue.
- **path mean [p10, p90]** — the distribution of that number across the 15 CPCV paths. If p10 > 0, the config is positive on essentially every path (the politics full stack is: p10 +0.13).
- **Δ vs prev kept [95% CI]** — *the keep-gate.* Paired per-group delta against the last kept rung, bootstrapped over groups. **Lower CI > 0 ⇒ KEPT; CI spans 0 ⇒ FRAGILE (dropped).**
- **keep** — KEPT / FRAGILE / DIAGNOSTIC (ran outside the gate because v0 didn't support its defensive knob).

Reading it aloud for esports A-S rung 2: "modal config `rung2[γ=1e-4,cap=200]`, honest level +0.03¢ (≈ economically zero), but its delta over the baseline is +6.66¢ with CI [+1.77, +11.81] entirely above zero and consistent across all 16 groups → **KEPT/certified**, and what's certified is the +6.66¢ of *avoided bleed*, not the +0.03¢ level."

---

## Part 7 — Status: what markets, what tests, what ships

| | politics_negrisk | esports |
|---|---|---|
| **Groups** | 11 NegRisk events | 16 matches |
| **Tests run** | costed ladder · v0 toxicity surface · nested CPCV · PBO/DSR/RC | same |
| **v0 defensive wiring** | ON (benign×endgame certified toxic) | withheld (no net-negative cell) |
| **Certified rung?** | none — all FRAGILE at K=11 | A-S rung 2 (bleed-avoidance, level ≈0) |
| **Best honest ¢** | full NSQ stack +0.29¢ (uncertified) | rung 2 +0.03¢ (certified vs baseline) |
| **Ships to Join 2 as** | full NSQ stack `damp[k=5e-6,cap=500,w=20,d=0.6]`, quote benign mid-life/approach, gate the endgame | A-S rung 2 `[γ=1e-4,cap=200]`, live A/B vs rung 2 + two-lens overlay |
| **Deployable edge today** | nil (measurement loop, not a trading system) | nil (risk control, not revenue) |

**Plain reading of "which markets we action on":** neither category ships as a *trading system*. Both ship as *measurement configs* for the Join-2 one-contract loop. Politics is where the interesting (uncertified) signal is — quote the benign approach window where the held-out +2.66¢ lives, let the two-lens gate defend the certified-toxic benign endgame. Esports ships a certified *defensive* config whose job is to not bleed.

---

## Part 8 — Standing verify-list (open items an audit should pressure-test)

Ordered by how much they'd move the verdict. Items 1–2 are exactly the models Justin flagged for the queue/costing verification pass.

1. **Is "pessimistic queue" actually the conservative direction for PnL?** The gate uses the fewest-fills (RiskAverse) model on the theory that it's conservative. But for a maker whose marginal fills are net-toxic, *fewer* fills can look *better* — so "pessimistic-fills = pessimistic-PnL" is a claim to check per category, not an axiom. Combined with the pessimistic *CI lower bound*, this double-conservatism is a large part of why politics reads FRAGILE. **Verify:** re-run the keep-gate under Prob(0.5) and Optimistic and see whether any politics rung flips (the bracket table says the *level* is queue-insensitive; check whether the *delta CI* is too).
2. **Does the fill model match reality?** The whole verdict rests on the queue model's fill attribution (`ProbQueue.f`) and the costed-touch exit convention, neither validated against our own fills (anonymous L2 can't). **Verify:** re-read `mm_join1_reconciliation_findings` for the last engine-vs-tape reconciliation, and treat Join-2 fills as the ground-truth calibration of `f`.
3. **Costing convention.** Leftover inventory is marked at the last observed touch, depth-of-exit unmodeled — this *flatters large terminal inventories* (i.e. the baseline), so it's conservative for the *controlled* configs but check the large-inventory tail.
4. **CI overconfidence from concurrency.** Politics overlap = 1.00 → effective N < 11 → OOS CIs too narrow. Strengthens the don't-ship verdicts, but would also erode a future politics "keep." Not purgeable; only more independent calendar span fixes it.
5. **`k_arr` fragility (A-S rung 2).** The certified esports spread depends on an arrival-decay estimate from thin pre-match lead-ins → live recalibration mandatory before trusting the derived spread.
6. **v0 reflexivity.** Toxicity was mapped on *baseline* fills; NSQ changes which fills happen, so the withheld esports defences might be supported under its own fills (rung 3's +0.69¢ hint). Only the Join-2 A/B resolves it — don't un-withhold from the backtest.
7. **Latency 0 ms flatters esports most** — the certified rung is a latency-naive upper bound; a snipe faster than our cancel could erase it.
8. **Lens-1 warm-up hole** — the session VPIN band needs ~30 buckets; a first-hour spike is unprotected. Live mitigation (seed the band from the lead-in) is designed, untested.

---

## Appendix — Formulas

**Microprice** (size-weighted mid; leans toward the thin side):

```
microprice = (bid_price·ask_size + ask_price·bid_size) / (bid_size + ask_size)
```

**NSQ reservation (inventory skew):** `r = microprice − k·q`   (`q` = signed inventory, `k` = skew knob)

**A-S reservation (rung 1):** `r = microprice − q·γ·σ²·τ`
where σ² = causal EWMA of microprice-change variance (price²/hour), τ = time-to-resolution (hours, capped at 168), γ = risk-aversion knob. Note the skew *shrinks* as τ→0.

**A-S optimal half-spread (rung 2):**

```
half_spread = γ·σ²·τ/2  +  (1/γ)·ln(1 + γ/k_arr)
```

The first term →0 at expiry; the second is a floor ≈ `1/k_arr` (arrival-decay). `k_arr` is IS-calibrated per token from the lead-in. Clamped to [tick, 0.05].

**VPIN (Lens 1):** volume-clock buckets of size `EWMA(typical trade size)·25`; per closed bucket imbalance `|B−S|/(B+S)`; VPIN = mean over last `w` buckets. Sweep weight per trade `w_sweep = exp(min(dist_ticks, 6)·0.5)`. Elevated when VPIN ≥ max(session 0.90-quantile, 0.30 floor).

**AS-z (Lens 2):** on each own fill, `signed_drift = side·(mid_{t+30s} − mid_at_fill)` (side = +1 buy / −1 sell; negative = adverse); `z = (signed_drift − μ)/σ` vs the rolling mean/std of the last 50 *calm-period* drifts; flag when `z < −2` (persists 120 s).

**OFI (Cont et al.) dampening:** signed pressure `norm = tanh(OFI_ewma / (3·scale))` ∈ (−1,1); pressured-side opening size `×= max(1 − d·|norm|, 0.2)`.

**Queue cancel attribution** (fraction of a cancel taken from *ahead* of us): `frontᶠ/(frontᶠ + backᶠ)`, then `advance = min(max(raw, floor), Δ)` with `floor = max(0, Δ − back)`. Optimistic → attribution = Δ; RiskAverse → 0 (floor only); Prob → the power form (default f=0.5).

**Costed PnL:** `realized round-trip cash + leftover_inventory · exit_touch` where `exit_touch` = best bid if long, best ask if short (never mid). Per-contract "honest ¢" = costed PnL / contracts traded, on held-out groups.

## Cross-links

Findings of record: [[mm_task5_inventory_quoter_findings]] (consolidated) · [[mm_task5_1_neutral_quoter_cpcv_findings]] (full 5.1). Engine + queue: [[mm_backtesting_methodology_explainer]] · [[mm_engine_queue_models]] · [[mm_join1_reconciliation_findings]]. Design inputs: [[mm_market_screen_and_ttr_regime_findings]]. Live loop this feeds: [[mm_politics_negrisk_live_loop_design]]. Hub: [[strat_market_making]] · [[COWORK]].
