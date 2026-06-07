---
title: "Epsilon — Master TODO"
created: 2026-06-05
status: closed
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

> Single consolidated checklist. Updated from handoffs. Active threads: **copytrade** (Midas bot, per-leader audit), **MM** (market-making) + **OD** (options-delta) — the two strats split out of the former "Block K" — and **dali / Polymarket research lineage**.
> Last updated: 2026-06-06 (PM CLOB capture semantics documented — see [[mm_clob_capture_semantics]]; shared MM directional decomposition + non-politics target screen + maker-infra audit + first-mover liquidity scope landed — see [[mm_structural_maker_directional_decomposition_findings]], [[mm_nonpolitics_target_screen_findings]], [[mm_maker_infra_audit_findings]], [[mm_first_mover_liquidity_scope_findings]]).

> **Naming:** "Block K" is the historical joint arc; it is now two strats. **MM** = market-making (hub [[strat_market_making]]); **OD** = options-delta (hub [[strat_options_delta]]). Shared origin docs: [[block_k_plain_english_synthesis]] · [[block_k_maker_options_research]]. Prompts should name MM or OD.

---

## Brain Infrastructure (Obsidian) — state 2026-06-07

> Implements [[OBSIDIAN_INFRA_ROADMAP]]. Goal: clean 2-person workspace + dynamically-evolving brain with minimal manual upkeep. Maps live in [[VAULT_MAP]] · [[SKILL_MAP]] · [[OPERATING_RHYTHMS]]. Hygiene tooling: `tools/brain_hygiene.py`.

**Phase 0 — Relay + source-of-truth rules**

- [x] `.gitignore` inverted 2026-06-07: `brain/**/*.md` now tracked; only `brain/generated/` + `brain/agents/*/scratch/` ignored (no more force-add).
- [ ] Relay root share + coworker quick-start confirmed working on both machines (invite key sent directly, not committed).

**Phase 1 — Core maps** (structure built 2026-06-07)

- [x] [[VAULT_MAP]] — start-here surface.
- [x] [[SKILL_MAP]] — repeatable workflows catalog.
- [x] [[OPERATING_RHYTHMS]] — hygiene cadence.
- [x] `brain/agents/codex/` + `brain/agents/cowork/` lanes ([[codex_lane]], [[cowork_lane]]).
- [x] [[ONBOARDING]] — agent-agnostic collaborator brief.
- [x] Verify VAULT_MAP active-branches + where-to-write tables match reality (Codex pass).

**Phase 2 — Hygiene scripts** (tooling built 2026-06-07)

- [x] `tools/brain_hygiene.py`: dupes, broken links, orphans, missing hub backlinks, missing frontmatter/summary, stale TODOs, recent-change digest.
- [x] Outputs moved to ignored `brain/generated/`: `GENERATED_INDEX.md`, `hygiene_report.md`, `stale_notes.md` (not committed; one command to regenerate; durable map is [[VAULT_MAP]]).
- [x] Weekly scheduled scan wired up (`brain-hygiene-weekly`, Mondays) — read-only, refreshes reports only.
- [x] Codex Janitor pass — quick wins done: **0 duplicate basenames, 0 broken links, 0 orphans, 0 missing hub backlinks** (commits `fa8dd20`, `8fccd47`, `7523b89`, `e763803`, `151ee04`; handoff [[2026-06-07_brain_hygiene_cleanup]]).
- [x] **Codex Janitor pass — finish frontmatter/summary backfill.** Follow-up pass reduced scanner counts from **119 → 12** missing frontmatter and **68 → 0** findings without Summary. The remaining 12 are intentional skips: empty daily shell, root GitHub README, and generic CLAUDE/README/PLAN convention docs. See [[2026-06-07_brain_hygiene_cleanup]].

### Phase 3 — Graph layer (tooling built 2026-06-07)

> Graphify-style structural audit. Built as a script by Cowork (it's tooling, not note-cleanup).

- [x] `tools/brain_graph_audit.py` → `brain/generated/graph_audit.md`: hub authorities + index hubs, orphans, dead-ends, weakly-connected components / topic islands, over-connected hubs.
- [x] Review first audit; graph remains one connected component with **0 orphans / 0 topic islands**. Only dead-end is the intentionally empty [[2026-06-07]] root daily shell; no link-repair pass warranted unless we decide to delete or fill it.
- [ ] Optional: Obsidian Canvas for a spatial branch view.

### Phase 4 — Agent workflow layer (scheduled agents)

> These are the "systems" from [[SKILL_MAP]] made autonomous. Goal: agents (and humans on offset hours) never get lost in the note pile. **Caveat:** anything that auto-commits brain Markdown must respect the Relay-vs-Git rule — Relay owns live note collaboration; a Git agent must `pull` first and surface conflicts, never force-push. Scope auto-commit to brain/notes; leave code/data to manual Git discipline.

- [x] **EOD brief** — scheduled task `brain-eod-brief` (daily 21:30 local): refreshes hygiene + graph audit, writes `brain/generated/daily_brief.md` (what changed / decisions / next gates / hygiene). Read-only on canonical notes.
- [x] **Commit/push agent** — scheduled task `brain-commit-push` (daily 22:00 local): stages brain + research-notes markdown only, commits, `pull --no-rebase`, pushes; aborts + reports on conflict, never force-pushes. Solves offset-hours sync.
- [ ] **Intraday brief** — optional lighter midday digest. Hold until the EOD brief proves useful.
- [ ] Each collaborator sets up their own `brain-commit-push` on their machine (symmetric, pull-first → safe).
- [ ] Decide brief ownership when both agents are active (one chronicler pass, not two).

> Note: both scheduled tasks run only while the Cowork app is open; if closed at the scheduled time they run on next launch. The commit/push agent's `nbstripout` git filter must be working locally — if a `git pull` ever errors on notebooks, the agent stops and reports rather than forcing (safe by design).

### Phase 5 — Indeaverse (deferred)

- [ ] Idea graph + strategy branch registry + roadmap index, navigable by concept not folder. See [[OBSIDIAN_INFRA_ROADMAP]] § Indeaverse. Build last.

---

## MM — Market-Making (state 2026-06-04)

> **NEXT: lane-owned live measurement only after a sleeve passes its research gate.** Shared Phase-1 maker infra is already built and tested (256 execution tests green; see [[mm_maker_infra_audit_findings]]). The non-politics K5 historical sleeves have **no clean neutral-MM live candidates** after the directional-vs-neutral gate; sports-like, residual misc, and equities up/down should not be screened or captured as reproduce-the-winners neutral MM sleeves. Separate novelty scope: first-mover liquidity in newly created markets **merits live Stage-1 capture** as a forward-test-only measurement loop, not a production edge. Politics deployment remains owned by the separate politics lane and its handoff.

Hub: [[strat_market_making]]. Key notes: [[block_k5_findings]] · [[block_k5b_findings]] · [[block_k5_stress_findings]] · [[mm_deployable_cells_findings]] · [[block_k2v3_findings]] · [[block_kpeg_robustness_findings]].

- 2026-06-04 MM structural-maker directional decomposition: **NON-POLITICS NEUTRAL GATE FAILS**. Joined structured non-top3 wallet-market rows to `traders_directionality.parquet`. `sports_like` is historically positive (**+237 bps**, CI **[+104, +366]**) but has **0 neutral `arb_like` wallets**; the positive denominator is `two_sided_directional`/`mixed`. `residual_misc` is historically positive (**+111 bps**, CI **[+41, +179]**) but the neutral slice is tiny and negative (**2 wallets, $69k gross, -298 bps**). Equity up/down has 0 neutral wallets and remains thin. Politics appears only as a read-only comparison row. See [[mm_structural_maker_directional_decomposition_findings]].
- 2026-06-04 MM `other:misc_other` retag + non-politics target screen: **NO LIVE NON-POLITICS CANDIDATES**. Retagged the fallback cell and requalified structure by sleeve: `sports_like` **$314.5M gross / +237 bps**, `residual_misc` **$494.8M / +111 bps**, `culture_like` **$12.7M / +555 bps**, `business_tech_like` **$5.7M / +235 bps**, `politics_news_like` **$20.0M / CI-crossing and out-of-lane**. Because the requested screen sleeves fail the neutral gate, the live Gamma/CLOB five-screen candidate list is intentionally empty. See [[mm_nonpolitics_target_screen_findings]].
- 2026-06-04 MM maker infra audit: **MEASUREMENT-GRADE, NOT PRODUCTION-GRADE**. `polymarket/execution/maker/` already contains event calendar, NegRisk inventory tracker, resolution/redemption handler, one-condition maker engine, maker CLI, and NegRisk-aware signing/metadata plumbing. Maker + NegRisk dependency slice: **67 tests passed**; full execution suite: **256 tests passed**. Missing for stronger telemetry: true queue position, side-adjusted drift bps, volume-weighted fill share, book depth at quote/fill, quote age, and cancel latency. See [[mm_maker_infra_audit_findings]].
- 2026-06-04 MM first-mover liquidity scope: **MERITS-LIVE-STAGE-1-CAPTURE / NO EDGE VERDICT**. In `$10k+` markets created after 2025-08-01 with first-week flow, only **12.9%** reached persistent 70% final-top3 maker share by day 7. Lower-bound pre-consolidation non-top3 flow was **$6.62B**, or **59.7%** of first-week maker dollars. Largest practical pools: daily crypto (**$2.77B**, fast-market control), sports (**$1.67B**), other/residual (**$1.63B**), geopolitics (**$0.36B**). This is a scope/capacity result only; live unknowns are spread/depth, queue, fillability, and adverse selection. See [[mm_first_mover_liquidity_scope_findings]].
- 2026-06-06 PM CLOB capture semantics: **PUBLIC L2 IS ANONYMOUS / RECONSTRUCT WITH AUDIT**. The market websocket gives `book`, `price_change`, `best_bid_ask`, and `last_trade_price`, with raw PM timestamps plus local `received_at` and monotonic receive clocks. It does **not** expose maker wallets, order IDs, or queue position. Trade-vs-cancel labels must be inferred by aligning trade prints with nearby L2 depletion and validated with a reconstruction audit; ambiguous same-ms/burst intervals should stay ambiguous. See [[mm_clob_capture_semantics]].
- 2026-06-03 MM politics NegRisk accounting: **MERITS-LIVE-MEASUREMENT-LOOP**. Indexed **2.10M** NegRisk activity rows across **284** structured wallets and decoded **125,937/125,937** relevant politics tx-wallet receipts. The corrected-carry structured-non-top3 row is **+2,290 bps**, CI **[1,020, 3,621]**, median wallet **14.5 bps**, net ex rebate **+2,276 bps**. Capacity remains Rule-3-as-assumption: 0.25% / 1% / 5% non-top3 flow implies mean EV/day about **$1.5k / $6.0k / $29.9k**, median-wallet EV/day about **$9 / $38 / $189**. Live unknowns: fill share, queue, missed fills, and operational merge/split/redeem accounting. See [[mm_politics_negrisk_accounting_findings]].
- 2026-06-03 MM equities index up/down structural-maker scope: **THIN-DATA / live collector needed**. This is not an OD pricing reopen; [[od_equities_index_pricing_scope_findings]] stays closed on fair-value pricing. Preferred close-style SPX/NDX structured-non-top3 settled cut is point-positive (**+246 bps**, median wallet **+269 bps**, ex rebate **+206 bps**) but CI-crossing (**[-403, +838] bps**) and only **$84.8k gross**. NDX is negative, SPX alone is underpowered, and QQQ's tiny positive diagnostic has only 8 markets. No ES/MES adverse-selection conclusion is possible locally because fill-aligned futures states and PM quote/queue history are missing. See [[mm_equities_updown_structural_scope_findings]].
- 2026-06-03 MM politics NegRisk persistence/deployable-capacity check: **standing live-loop anchor, event-aware sizing**. 2026 non-top3 politics-NegRisk maker flow is **$381.1M** through 2026-05-26 over **146/146 observed active days**; every observed 2026 day has at least **$250k** flow. Flow mix is not just the 2024 election: 2028 US presidential outrights **36%**, Trump personnel/policy **24%**, non-US elections **19%**, other politics **16%**, 2026 US races/midterms **5%**. Settled-only no-mark-to-mid edge holds outside 2024: 2025 close-year **+1,402 bps**, CI **[387, 2,919]**; 2026 settled **+1,356 bps**, CI **[744, 2,507]**. But median-bps monthly EV at 1% capture is only **$0.4k-$1.6k** in observed 2026 months; mean-bps **$66k-$254k/month** is a right-tail scenario, not the default. See [[mm_politics_negrisk_accounting_findings]].
- 2026-06-02 MM/OD realism re-audit: MM standalone remains sub-scale under the base median/cell-specific capture model (**$78/active day**), but this is a modeled capacity assumption, not a hard cap. Sensitivity: fixed 1% capture of non-top3 2026 flow is about **$304/day** median EV across the 8 deployable cells; fixed 5% is about **$1,521/day**. Treat structured non-top3 MM as a live capacity/moat measurement loop and OD execution layer. The prior `politics_negrisk` accounting block is superseded by the 2026-06-03 audit above. See [[od_strategy_a_realism_reaudit_findings]].

**Single-venue Polymarket maker is CLOSED across every anchor tried:**
- K1 generous economics gate "passes" but only because it marks to mid (fee-free geopolitics also passes → not a real edge).
- K2 logit Avellaneda-Stoikov, optimized, with real exit: −1,126 bps, CI<0.
- K2v3 Binance/digital-anchored + delta-widened: 0/681 buckets clear; the anchor *increased* adverse selection (325 vs 145 bps).
- K2v2 defensive (pull/widen when Binance moves): −4,316 bps; defense fired <0.1% → **adverse selection here is structural, not a dodgeable latency race.**
- K-PEG (+759 bps) = a **mark-to-mid artifact** (we + Codex audit reproduced exactly); realizable round-trip −753; maker-exit −569 (~12% exit fill, need ~40%). Shape analysis: the entry alpha is **real and broad-based** (79% win, survives dropping top 5% of fills); the entire loss is exit cost. (This entry signal is the bridge into OD's Strategy A.)

**The de-biased arc (K5 → K5b → K5-STRESS → deployability):**
- **K5** (closed-position, model-free): real makers profit; crypto-4h **+171 bps, CI [34, 327]**; playbook = **64% two-sided, 78.8% carry-to-resolution, 0.8% spike-zone**. Capacity warning: top-3 take ~95%/market. See [[block_k5_findings]].
- **K5b:** moat is **capital/scale + structure, NOT speed** → build in Python/Midas, **defer Rust**. Below-top3 field not robustly positive; only well-structured books profit. See [[block_k5b_findings]].
- **K5-STRESS** (de-biased gate): survivorship was **not** the problem (−1.4 bps); **selection was** — the typical/median maker ≈ breakeven. But the **structured-non-top3 cut clears in 4 categories** (crypto_4h, culture, other, sports), non-rebate-dependent, stable. Longshot premium **not** independently confirmed (calibration 0/cats) → edge is structural liquidity provision, not mispricing. politics_negrisk was the biggest cell and now survives proper NegRisk accounting (**+2,290 bps**, CI **[1,020, 3,621]** on corrected-carry re-cut). See [[block_k5_stress_findings]] and [[mm_politics_negrisk_accounting_findings]].
- **Deployability (K9):** honest median + capacity + 2026-flow run-rate = **~$78/active day (~$2.3k/30d), ~90% in one grab-bag cell `other:misc_other`; crypto cells ≈ $0/day.** The 2026-06-04 directional decomposition adds a stricter caveat: the non-politics positives are historical structured-maker patterns, but not clean neutral-liquidity MM sleeves. See [[mm_deployable_cells_findings]], [[mm_structural_maker_directional_decomposition_findings]], and [[mm_nonpolitics_target_screen_findings]].

**Conclusion / consolidation:** MM standalone doesn't justify a dedicated bot. Its durable value is the **execution/lifecycle layer** (two-sided passive entry + carry-to-resolution + spike-avoidance + non-incumbent cell selection) that any Polymarket edge needs to dodge the exit-spread tax. This folds into **OD Strategy A**, where OD supplies the actual edge (the cross-sectional longshot/vol overpricing). **Recommend consolidating to OD-primary; MM = execution layer.** Hubs: [[strat_market_making]] · [[strat_options_delta]].

Open MM tasks:
- [x] Why top-3 dominate → scale/structure, not speed (K5b). Rust deferred.
- [x] `politics_negrisk` NegRisk merge/split/redeem accounting — unblocked; verdict **MERITS-LIVE-MEASUREMENT-LOOP**. See [[mm_politics_negrisk_accounting_findings]].
- [x] `politics_negrisk` persistence/deployable translation — standing live-loop anchor with event-aware sizing; median monthly EV remains small until live capture/fill share is proven.
- [x] **Equities index up/down as structural MM scope** — completed; verdict **THIN-DATA**. Close-style SPX/NDX structured-non-top3 is point-positive but not CI-positive and too small after filters; do not reopen OD pricing. See [[mm_equities_updown_structural_scope_findings]].
- [x] **Cell list + market-selection rule** — decided 2026-06-03. Proven buckets: Non-US elections (+3,673 bps), Trump personnel/policy (+1,018 bps), Other politics (+1,046 bps). Exclude 2028 outrights (forward-only). Five-screen filter: negRisk flag + bucket check + ≥5% non-top3 headroom + uninformed-flow preference + resolution-clarity gate. See [[mm_politics_negrisk_live_loop_design]].
- [x] **Phase-2 success criteria pre-registered** — decided 2026-06-03. Fill share > 0% in ≥5 markets; post-fill 60s drift lower CI > −500 bps (not approaching crypto's −1,886); news-proximate adverse fills < 50%; net-of-cost lower CI > 0 over ≥30 settled markets; resolution drag < 10%. Sample floor: 30 settled markets or 90 days. See [[mm_politics_negrisk_live_loop_design]].
- [x] **News-feed approach** — decided 2026-06-03. Scheduled-event calendar only (static YAML, manual refresh). Gate breaking-news feed on Phase-2 telemetry showing news-proximate adverse fills > 50%. See [[mm_politics_negrisk_live_loop_design]].
- [x] **Phase-1 exec build / shared maker infra audit** — built and tested. `polymarket/execution/maker/` is measurement-grade, not production-grade; 67 maker/NegRisk dependency tests and 256 full execution tests pass. See [[mm_maker_infra_audit_findings]].
- [x] **SPX close-style up/down live MM collector decision** — do not run from the non-politics lane now. The equity up/down neutral subset is empty and the all-structured row remains thin/CI-crossing. See [[mm_structural_maker_directional_decomposition_findings]] and [[mm_nonpolitics_target_screen_findings]].
- [ ] **Phase-2 deployment** — lane-owned. Shared infra is no longer the blocker; only a lane that passes its own market-selection gate should wire a one-market, 1-contract, telemetry-heavy smoke. Minimum 30 settled markets before any scale decision.
- [ ] **Phase-3 scale decision** — blocked on Phase 2 passing the pre-registered gates.
- [x] Characterize `other:misc_other` — retagged into sports-like, residual misc, politics-news-like, culture-like, and business/tech-like. Historical positives do not survive the neutral gate for the requested non-politics sleeves. See [[mm_nonpolitics_target_screen_findings]].
- [x] **First-mover liquidity scope** — completed; cheap historical falsifier did not fire. Newly created markets show large first-week flow before persistent final-top3 dominance, but this is a forward-test-only novelty branch. See [[mm_first_mover_liquidity_scope_findings]].
- [ ] **First-mover Stage-1 live capture:** watch newly created sports, retagged other/residual, and geopolitics/politics-news-like markets for first-week book shape, spread/depth, trade arrivals, and concentration; include daily crypto only as a fast-market control if cheap. No trading/scale until live quoteability and adverse-selection telemetry pass. Public L2 is anonymous; any trade-vs-cancel attribution must follow [[mm_clob_capture_semantics]] and report clean vs ambiguous reconstruction rates.
- [ ] **Merged test (OD task):** does an OD-skewed passive maker beat the MM baseline AND OD's taker form?
- [ ] **Capacity/speed caveat for OD expansion:** live-paper instrumentation needed. See [[od_cross_asset_updown_scoping]].
- [ ] Do NOT spawn new single-venue quoting/anchor/continuous-hedge variants — exhausted.

---

## OD — Options-Delta (state 2026-06-06) — REFRAMED as the valuation/signal layer
- 2026-06-01: OD v3 PnL/risk deep-dive wrote `notes/options_delta/od_strategy_a_v3_pnl_risk_findings.md`. Verdict FAIL: per-asset concurrent capital is official; tail stress and incremental-over-MM decide whether OD remains separate from MM.

- 2026-06-07 OD replacement-fair sensitivity: **CLOSE remains**. Existing 23-fill v4 far-|z| strict-rich short artifact was repriced with RV physical probability, Arm B empirical conditional, Merton, Kou, and Edgeworth fairs. Best non-RV borrowed-baseline lower CI was -1.61c on `arm_b_empirical_conditional` with 15/23 fills surviving the 1c gate. No dataset remake; this is a selection/sizing sensitivity only. See [[od_replacement_fair_sensitivity_findings]].

- 2026-06-02 OD conditional-probability calibration: **CLOSE**. Binance-only Arm B empirical `P(resolve|z,t)` applied to the v4 far-|z| strict-rich short set gives mean predicted ITM 52.22%, observed ITM 39.13%, model edge 3.83c, and realized net EV CI [-1.84c, 26.49c]. OD remains closed standalone unless explicitly reopened with a stronger, pre-registered data source. See [[od_conditional_prob_calibration_findings]].


- 2026-06-06 OD methodology realism audit: **RV-model fair is physical-probability fair, not option-IV fair**. Strategy A's `token_model_fair`/`claim_model_fair` path is mostly causal EWMA realized-vol `P(resolve)`; K6 `pm_iv_annualized` is PM midpoint inverted into vol units and is circular if used as fair for the same PM market. Before any reopen, separate `physical_prob_fair`, `external_iv_fair`, and `pm_mid_implied_vol`; standalone OD remains closed and OD valuation is only a cautious sizing/selection overlay for source-clean passive MM/carry. See [[od_methodology_realism_audit_findings]].
- 2026-06-03 OD SPX opens up/down current-data scope: **MERITS-LIVE-COLLECTOR / NO PRICING VERDICT**. The close-market SPX analysis did not include the separate `spx-open-daily-up-or-down` series. Current Gamma/local data show a daily recurring, small-capacity-clearing template: 63 recent Gamma rows, 54 recent event dates, 31 local Dali candidates, 62,131 local fills, $6.27M local candidate volume, median event volume $179k, weighted non-top3 share 59.3%, and current-live-flow non-top3 headroom about $37.8k/day. Current book snapshot is not a pricing gate: June 3 was post-open/near-resolved; June 4 pre-open was 51/64 on Up with small top-depth and a 13c spread. Local CLOB scan found 0 replayable historical book lines, so the next step is a live after-close-to-open PM-book + ES/MES collector, not OPRA/ML. See [[od_equities_index_pricing_scope_findings]].
- 2026-06-03 OD SPX implied-vol N(z) last swing: **STILL-BLOCKED for strict VIX+ES+best-ask replay; no ML/options build**. Cboe VIX/VIX9D direct data covers the local sample, but historical PM SPX daily best asks and resolution-time overlapping CME ES prices are missing. Pricing branch remains CONFIRM-CLOSE from the realized-vol gate. See [[od_equities_index_pricing_scope_findings]].
- 2026-06-03 OD Binance touch-risk skip pre-test: **DROP as hard skip; log telemetry only**. A causal 5m Binance score using momentum, taker-flow-imbalance proxy, and Lee-Mykland jumps over the grown Binance tail base does not produce strong held-out touch/jump separation and does not reliably lift PM stress EV on the tiny Strategy-A fill overlay. True Binance L2 OFI is missing from the saved cache. See [[od_touch_risk_filter_findings]].
- 2026-06-03 OD pure-taker attribution on captured crypto-4h LOB: **CONFIRM-CLOSE for standalone taker OD**. Recomputed v3 `N(z)` against contemporaneous one-second Binance and crossed displayed PM best ask plus fee, first qualifying signal per strict-source market, carry to resolution. Broad strict-source taker row is **+1.95c/contract CI [-26.04c, 26.73c]**; Strategy-A far row is **+0.80c CI [-27.00c, 27.92c]**. Far/late zero-threshold is **+3.01c CI [0.02c, 6.50c]**, but median selected edge is only **0.36c** and the `>=1c` far/late row crosses zero **[-0.97c, 7.98c]**. Stale-vs-contemporaneous control rejected **0 rows** because K6 `p_model` was already contemporaneous. Branch fired: OD valuation does not prove it beats spread alone; surviving Strategy A is passive source-clean MM/carry with OD as sizing/selection. See [[od_strategy_a_realism_reaudit_findings]].
- 2026-06-03 OD equities SPX daily up/down pricing gate: **CONFIRM-CLOSE / no OPRA build**. SPX clears small-cap capacity, but the cheap causal realized-vol `N(z)` residual on actual executable fills does not show a clean net-of-fee edge with market-date CI; empirical SPX calibration does not rescue it. Pricing thesis now looks comprehensively closed across crypto terminal digitals and equities index up/down unless future fresh data clears this cheap gate first. See [[od_equities_index_pricing_scope_findings]].
- 2026-06-02 Binance daily momentum + Polymarket BTC/ETH binary hybrid: **CLOSE / DO NOT BUILD**. The [[STRATEGY_REFERENCE]] six-asset daily momentum baseline is **77.05% CAGR / 27.16% ann. vol / 2.24 Sharpe / -26.20% max drawdown**. Best hedge case improves max drawdown to **-19.02%** but cuts CAGR to **47.23%** and Sharpe to **1.76**; best alpha proxy still trails at **65.59% CAGR / 1.78 Sharpe / -31.53% max drawdown**. PM depth/capital efficiency are binding. See [[2026-06-02_binance_momentum_polymarket_hybrid]].
- 2026-06-02 OD Strategy A realism re-audit: **MERITS-LIVE-MEASUREMENT-LOOP** for the source-clean/rich-short passive longshot harvest at $10-$100 scale, while the bare one-position-global far gate remains **CONFIRM-CLOSE**. Deciding numbers: strict-rich per-asset flat sizing is **18.36c/filled contract CI [0.25c, 27.43c]** across 7 markets / 22 fills; $50 dollar-delta cap is **15.49c/weighted contract CI [0.48c, 47.54c]**. K6 static far all-tau diagnostic is **9.58c CI [0.43c, 21.39c]**. Same-day Arm T remains standalone **CLOSE**, but Tier-1 `pos_z_ge_3/1_4h` joins this cluster: raw **3.49c CI [2.29c, 5.43c]**, deployable **0.17c / 0.35c / 0.79c** at **5% / 10% / 22.68%** capacity. The borrowed 1.98c structural queue baseline is diagnostic only. See [[od_strategy_a_realism_reaudit_findings]].
- 2026-06-02 OD Strategy A tail/sizing robustness: **CONFIRM MERITS-LIVE-MEASUREMENT-LOOP**, narrowed to tail-aware sizing. PM strict-rich source-clean 4h fills do **not** grow offline beyond 7 markets / 22 fills; extended cached Binance 4h history grows the tail base to **35,466 BTC/ETH/SOL asset-windows / 1,666,845 pre-cutoff states**. Under adverse-regime stress, flat one-contract is **1.74c/contract CI [-0.39c, 5.07c]**, while **rv-edge-scaled** is the largest surviving size at **4.19c/weighted contract CI [0.62c, 6.66c]** and **$0.79/day CI [$0.09, $1.72]**. Same-day touch tail-adjusts to **1.47c/contract CI [0.78c, 1.86c]** on held-out but has zero observed touches and ~96c loss-given-touch. Source-vs-valuation remains unproven: **-40.47c/episode CI [-120.93c, 1.49c]** for strict-rich minus strict-source same markets. See [[od_strategy_a_realism_reaudit_findings]].
- 2026-06-02 OD same-day Arm T Tier-1 edge-concentration extension: **Arm T CLOSE; Arm E CLOSE**. best registered held-out Tier-1 cell pos_z_ge_3/1_4h has 5% haircut EV 0.17c with CI [0.12c, 0.27c] and BH capacity q=0.0003; mean is below the pre-registered 0.25c materiality bar. See [[od_same_day_crypto_pricing_gate_findings]].
- 2026-06-02 OD same-day crypto pricing gate confirmation: **Arm T MERITS-BUILD; Arm E CLOSE**. Arm T: best confirmed cell neg_z_ge_3/15m_1h has held-out net CI [0.15c, 0.80c] and 5% capacity-haircut CI [0.01c, 0.04c]; BH capacity q=0.0065; mechanism is fragile calibration residual: PM is close to empirical first-passage, not near terminal. Arm E remains closed. See [[od_same_day_crypto_pricing_gate_findings]].
- 2026-06-02 OD same-day Arm T **realism calibration**: confirmation survivor passed the stats but was **economically trivial (~0.02c/contract after 5% haircut)** and was a calibration residual, NOT the behavioral "touch priced like terminal" gap. Superseded by Tier-1 **CLOSE** above; no constrained live loop unless a future fresh sample clears the same materiality/OOS/BH bar. The **1.98c structural baseline is borrowed (crypto-4h MM) and is NOT a valid Arm T kill switch** — deriving a touch-specific passive baseline needs live quote/queue data. General harshness rules in [[CODEX]] § Realism calibration.
- 2026-06-02 OD-RV Deribit daily settlement check: **PARK PM daily; proceed only with 08:00 UTC aligned 4h/hourly capture**. PM daily BTC/ETH resolves at 16:00 UTC from Binance 1m candles, while active Deribit BTC/ETH option expiries are all 08:00 UTC. No clean 16:00 synthetic was found. Aligned PM BTC/ETH 4h `12:00AM-4:00AM ET` and hourly `3AM ET` windows do end at 08:00 UTC; collector implemented in `polymarket/research/scripts/od_rv_deribit_aligned_capture.py`. Need 60-100 independent asset-days before any net-of-cost CI. See [[od_rv_deribit_daily_capture_findings]].
- 2026-06-02 OD-RV Deribit daily scoping: **SUPERSEDED**. The degraded one-window daily basis in [[od_rv_deribit_daily_scoping_findings]] remains useful as historical context, but the daily product is parked by the settlement mismatch.
- 2026-06-02 OD/MM cross-asset Gate 0 universe/capacity map: **ONLY crypto-daily clears the cheap filter**. Crypto-daily clears the stated `$50k/day` volume and `$25k/day` non-top3 headroom minimum, but the parallel settlement check still parks literal PM daily BTC/ETH vs Deribit daily because of the 16:00 UTC vs 08:00 UTC mismatch. Index up/down and single-stock up/down are clean but thin; close-above/price-band has large flow but is mostly path-dependent crypto hit/barrier markets and fails the clean-reference gate; true financial neg-risk baskets are thin one-offs. See [[od_cross_asset_gate0_universe_map_findings]] and [[od_cross_asset_updown_scoping]].
- 2026-06-02 OD pricing-model-form diagnostic: **CLOSE remains**. Captured-window/live 1s Merton/Kou jump-aware repricing reduced the original far-|z| strict-rich model edge from Gaussian +3.53c to Merton +1.03c CI [-2.84c, 2.59c] and Kou +0.85c CI [-3.39c, 2.53c]. A cheap Arm-C higher-moment/Edgeworth extension left the original-set edge at +3.47c CI [2.40c, 4.31c], but the best MM-integrated lower CI after the K5 top-maker haircut and v4 structural queue baseline was still negative (-1.07c). Deribit DVOL was BTC/ETH-only and illustrative, and also failed the structural-incremental lens. See [[od_pricing_model_form_findings]].
- Reopen guardrail: the 2026-06-02 close used Binance **5-minute** history as a broad base-rate calibration. Any pricing-model-form reopen should validate on captured-window/live **1s Binance + Polymarket LOB/WS** where available, including OFI/depth/jump/source-basis features from `block_a0c_roll_features`, `block_a0c_features`, and the K2v2 daily 1s cache. Multi-year 5m history is the prior/control, not a substitute for captured live-window evidence.

- 2026-06-01 OD v4 exploratory queue replay: ran by user override after Phase 0 failed. Best OOS queue-adjusted-after-top3 row: 11 markets, mean 1.98c, CI [0.66c, 3.43c], 1.55 expected contracts. Not an official gate pass because Phase 0 calibration failed. See [[od_v4_queue_replay_findings]].

- 2026-06-01 OD v4 calibration gate: **FAIL**. Full-panel far-|z| strict-rich shorts are only 23 fills / 8 markets; mean gross short EV 16.92c, market-cluster CI [-1.43c, 26.38c], realized ITM 39.13%. Queue-aware replay remains blocked unless the calibration/EV gate is reopened with a corrected forward-vol model. See [[od_v4_calibration_gate_findings]].


Hub: [[strat_options_delta]]. Key notes: [[od_methodology_realism_audit_findings]] · [[od_strategy_a_v2_lifecycle_findings]] · [[block_k6_vol_findings]] · [[block_k7_findings]] · [[block_k3_leadlag_findings]]. Handoffs: [[2026-05-30_maker_options_delta_pivot]] · [[2026-05-31_kronos_hermes_eval]].

**Reframing (2026-06-01):** OD and MM are two LAYERS of one strategy, not competitors. **MM = execution/lifecycle** (passive fills, two-sided, carry, rebate). **OD = valuation/signal** (longshot+vol overpricing harvest → sell the rich side, risk-capped; fair-value entry filter; regime selection; dollar-delta caps). The Binance hedge is a **minor overlay**, not OD's edge.

**OD Strategy A v3:** OD Strategy A v3 bare baseline failed the Phase-1 global gate: OOS far-|z| n=6 markets / 68 fills, mean 118.08c, CI [-17.14c, 323.46c]. Best OD filter was `strict_rich_short_ge_005m` with mean 105.20c, CI [16.19c, 258.69c], lower-CI lift 33.33c. The global-embargo power bottleneck remains explicit; per-asset concurrency is only a diagnostic unless the capital assumption is reopened. See [[od_strategy_a_v3_findings]].

**OD Strategy A v2 lifecycle gate FAILED on power, not edge:**
- OOS far-|z| family n=**6 markets** / 68 fills, mean **+118.08c**, CI **[-17.14c, 323.46c]** (missed lower-CI by 17.14c; fat-tail driven — one +618c episode, drop it → ~+18c).
- **Strict-source far-|z| CLEARS** [4.13c, 319.24c] same markets → source-basis filter may be a real design ingredient. Diagnosis: underpowered, not no-edge.
- **Hedge near-irrelevant in the gold mine:** Phase 2 full hedge cut far-|z| variance only 9.76%, mean unchanged (delta tiny far from strike). mid/near hedge cuts variance 77–97% but pays away 50–80% of premium. Minimal hedge in gold mine ≈ zero.
- Real Phase-1 edge = **shorting the overpriced longshot side and carrying** (cross-sectional bias, not spread capture).
- Phase 0 token mapping: 5/370 mismatches (1.35%) — prior cells fine; use `asset_id/outcome_index`.
- Kronos/HAR forward-vol bake-off remains gated off until the unhedged lifecycle clears OOS.

Historical context:
- K3: 4h crypto has **no anti-arb fee**; Binance leads ~10s; raw post-fee basis thin. K3v2/v3h hedged dynamic-basis don't clear.
- K4: intra-Polymarket arb ~zero on owned universe.
- K6 vol: Polymarket **overprices vol** (+3.7 vol pts avg, +24 far/late, CI clears) but continuous/banded-hedged gamma scalp is −9.39c far/late, of which 9.56c is hedge turnover. Turnover is the killer, not the vol estimate.
- K6 **Strategy A (STATIC hedge)** ran: **fixed the turnover** (far/late hedge cost 9.56c → 2.18c; mean −9.39c → **+1.07c**). Gate (far/late lower-CI>0) **fails on power** — CI [−1.04, 3.99] at n=11 — but **far/mid clears** (n=8, +11.15c, CI [0.96, 21.41]) and the **far |z|≥1 family is positive across τ**. Underpowered near-miss, not a closed door. See [[block_k6_strategy_a_static_hedge_findings]].
- K7: cross-category longshot premium (additional mispricing lens).

Open OD tasks:
- [x] **OD methodology realism audit** (2026-06-06): completed; wrote the RV-vs-IV correction and broader unrealistic-methods ledger. Treat EWMA `N(z)` fair as physical forecast probability, PM IV as diagnostic only, and external option-IV fair as a separate object that still needs settlement-aligned data. See [[od_methodology_realism_audit_findings]].
- [x] **Binance daily momentum + Polymarket BTC/ETH binary hybrid** (2026-06-02): closed. HEDGE overlay improves drawdown only by paying away too much CAGR/Sharpe; ALPHA sleeve does not beat the daily momentum baseline even with generous proxy edge assumptions. See [[2026-06-02_binance_momentum_polymarket_hybrid]].
- [x] **Strategy A v2** (2026-06-01): lifecycle + Phase 2 hedge frontier ran; Gate 1 failed on power; hedge near-irrelevant in the gold mine. See [[od_strategy_a_v2_lifecycle_findings]].
- [x] **Strategy A v3 — power + OD entry filter** (2026-06-01): completed; see [[od_strategy_a_v3_findings]]. Original task: (a) **Power:** pool BTC+ETH+SOL, widen far-|z|/longshot family, pull in more captured 4h windows → n from 6 to dozens; (b) **OD entry filter (the headline test):** only take fills when PM is rich vs Binance RV physical-probability fair / PM midpoint-implied vol > causal forecast vol — does the longshot/vol filter beat the bare lifecycle's lower-CI?; (c) **dollar-delta inventory caps** to tame the fat tail; (d) **make source-basis filter official** (strict already clears); (e) hedge demoted to footnote overlay — keep static h-sweep + the **24h portfolio-rollover** variant (cheap, low downside) but it is NOT the edge. Pre-register far-|z| all-τ OOS lower-CI>0 as the gate.
- [x] **OD pure-taker attribution** (2026-06-03): completed; broad strict-source taker and Strategy-A far taker rows do not clear lower CI, and the far/late zero-threshold survivor is economically hair-thin. No active pure-taker OD sleeve unless future fresh/live evidence clears a materiality-aware market-cluster gate. See [[od_strategy_a_realism_reaudit_findings]].
- [x] **Equities index up/down small-capacity scope** (2026-06-03): completed; SPX daily up/down clears capacity/persistence at `$10-$100` scale, while NDX, single stocks, and close-above ladders remain deferred. The follow-on cheap N(z) pricing gate now closes the SPX pricing branch; no OPRA build. See [[od_equities_index_pricing_scope_findings]].
- [x] **SPX daily up/down N(z)/realized-vol pricing gate** (2026-06-03): completed; CONFIRM-CLOSE. Follow-on VIX+ES last-swing audit is strict-data-blocked, so do not build Cboe/OPRA/ML unless a future live best-ask collector first produces a lower-CI-positive net executable edge. See [[od_equities_index_pricing_scope_findings]].
- [x] **SPX implied-vol N(z) last swing audit** (2026-06-03): completed; strict VIX+ES+best-ask replay is data-blocked. Cboe VIX/VIX9D direct data covers the local sample, but historical PM SPX daily best asks and overlapping historical CME ES prices are missing; no ML/options-chain build. See [[od_equities_index_pricing_scope_findings]].
- [x] **SPX opens up/down current-data scope** (2026-06-03): completed; opens was a separate series missing from the close-style SPX analysis and clears current small-capacity/persistence, but has no pricing verdict. Local CLOB scan found 0 replayable pre-open book lines. See [[od_equities_index_pricing_scope_findings]].
- [ ] **SPX opens pre-open live collector:** for 30-50 sessions, log PM best bid/ask/depth/queue from after prior official close until official open, plus ES/MES front-month state and official WSJ open/close settlement. Only after that run a rule-based ES/MES `N(z)` residual with market-date CI; no OPRA/options-chain/ML build before a lower-CI-positive net executable residual appears.
- [x] **Pricing-model-form diagnostic** (2026-06-02): captured-window/live 1s first. Merton/Kou jump diffusion ran; cheap Arm-C higher-moment/Edgeworth extension ran; full Bates/Variance-Gamma/Kronos remain blocked by package/sample discipline; Deribit BTC/ETH DVOL ran as an illustrative anchor and was reported with the same MM-incremental columns. See [[od_pricing_model_form_findings]].
- [x] **OD-RV Deribit settlement-alignment desk check + collector** (2026-06-02): daily PM BTC/ETH vs Deribit is parked due to a structural 8h tail; 08:00 UTC aligned PM BTC/ETH 4h/hourly capture is implemented. See [[od_rv_deribit_daily_capture_findings]].
- [x] **OD same-day crypto pricing gate confirmation** (2026-06-02): completed; Arm T MERITS-BUILD, Arm E CLOSE. See [[od_same_day_crypto_pricing_gate_findings]].
- [x] **OD same-day Arm T Tier-1 edge-concentration extension** (2026-06-02): completed; Arm T CLOSE. best registered held-out Tier-1 cell pos_z_ge_3/1_4h has 5% haircut EV 0.17c with CI [0.12c, 0.27c] and BH capacity q=0.0003; mean is below the pre-registered 0.25c materiality bar. See [[od_same_day_crypto_pricing_gate_findings]].
- [x] **No same-day Arm T live measurement loop** (2026-06-02): Tier-1 did not lift edge enough, so the conditional live-loop task is closed. Fold same-day first-passage/HAR-Kou IV gap into MM as a caution feature only.
- [x] **Binance OFI/jump touch-risk skip pre-test** (2026-06-03): completed; no hard skip. Log adverse momentum, taker-flow imbalance proxy, and jump flags as telemetry only. See [[od_touch_risk_filter_findings]].
- [ ] **OD Strategy A live measurement loop spec:** instrument source-clean/rich-short passive quotes on BTC/ETH/SOL 4h using **rv-edge-scaled sizing as the only stress-surviving active sleeve**; log flat one-contract, $50 cap, 10/25/50% Kelly, and pure-taker OD far/late triggers as shadow policies only. Include same-day Arm T Tier-1 `pos_z_ge_3/1_4h` touch-fade cluster members at tiny risk budget with explicit barrier/touch acceleration logs. Capital stays $10-$100 until live queue/fill-share data justifies more. Required logs: quote/cancel/queue, missed fills, top-maker rank/share, source-basis events, barrier-touch events, adverse high-vol/large-move regime tags, executable spread by `(z,tau)`, stale-vs-contemporaneous Binance deltas, and realized carry-to-resolution PnL. This is a measurement loop only, not a trading-system build.
- [ ] **Run OD-RV 08:00 aligned capture for 30-50 calendar days** (BTC+ETH, one independent asset-day per expiry timestamp; do not double-count nested 4h and hourly products). Evaluate both PM-rich and PM-cheap directions with PM/Deribit bid-ask, fees, call-spread replication error, leg latency, and settlement-reference mismatch PnL.
- [x] **Cross-asset PM financial-binary Gate 0:** mapped PM daily crypto, index up/down, single-stock up/down, close-above/price-band, and neg-risk markets against external references and K5-style incumbent concentration. Original result: only crypto-daily cleared the `$50k/day` operation-scale capacity filter; literal PM daily BTC/ETH vs Deribit daily remains parked by settlement mismatch. Superseded for SPX small-cap scope by the 2026-06-03 equities pass: SPX daily up/down clears `$10-$100` capacity, and the follow-on N(z) gate confirms close/no OPRA build. See [[od_cross_asset_gate0_universe_map_findings]] and [[od_equities_index_pricing_scope_findings]].
- [ ] **Only after v3 clears OOS:** EWMA vs HAR-RV vs Kronos forward-vol bake-off, net-of-cost, tail-calibrated. Kronos scoped in [[2026-05-31_kronos_hermes_eval]].
- [ ] Do NOT re-run continuous/banded-hedge gamma-scalp variants — closed (turnover).

---

---

## copytrade

### strategic decision
- [ ] **Smoke target re-selection.** Re-rank candidates by recency × hold-to-resolution × audit-deployable-cell-count, excluding `split_position_signature > 60%`. Likely new target: Domah's `macro / maker / 18-24` cell (highest n_deployable_cells = 3) or any leader from next 5 audits landing ≥3 cells. Original `0x6a72f61820b2…` excluded (lifetime-PnL ↔ copyability ρ = −0.21).
- [x] **Structural directional-carrier copyability audit** (2026-06-04): completed. Sports/residual/equities `two_sided_directional` carrier wallets pass the historical carrier screen, but pure taker-copy does **not** clear net-of-cost. Vetted sports positive-K5 top 3: leader replay +183 bps, taker-copy **-62 bps**, CI **[-702, +497]**; residual taker-copy **-224 bps**, CI **[-495, +56]**; equities thin and negative. Do not promote structural carriers into smoke targets. See [[copytrade_structural_directional_carriers_findings]].
- [x] **Exchange-internal-leg artifact quantified.** `notes/copytrade/copytrade_relayer_implications.md` finds tx.from is relay-layer, active wallet is already emitted in `maker`, and Domah smoke cell remains intact.
- [ ] **Paper-trade vs $10-smoke decision.** Recommendation: paper-trade Domah's deployable cell while audit framework matures to ≥12 leaders.

### exec — path to first real-money fill
1. [ ] **PLAN.md sync + snapshot commit + tag.** Engineering-complete marker. ~10 min. 2. [ ] **Slack colleague** about the kernel encoding bug workaround before midas executor goes live. 3. [ ] **Polymarket credentials into `.env`** — private key → `derive_api_keys.py` → `.env`. Read-only auth check first. 4. [ ] **Pre-flight smoke target** — profile current open positions via Gamma + RTDS pre-subscription. Confirm no NegRisk markets. 5. [ ] **First real-money smoke** per `midas/scripts/SMOKE_REAL.md`. `MAX_REAL_ORDERS=1`, `REQUIRE_OPERATOR_CONFIRM=true`, `SIZING_USD=10`.

### data / research
- [ ] **Audit 5 more leaders** via `scripts/domah_copy_audit.py`. Sort top-50 by `active_days_last_90d` desc + `hold_to_resolution_share` desc, exclude `split_position_signature > 60%`. Goal: 12 audits before re-checking cross-leader intersection.
- [ ] **Re-check cross-leader intersection at 12 audits.** Currently 2 narrow shared cells across 7 audits — too thin.
- [ ] **If `other / taker / afternoons` stays deployable for ≥3 leaders**, design cohort-level signal. Currently 2.
- [ ] **Weather FTC TP live-look diagnostic** — 5 min manual observation per active weather market: track best_bid/best_ask after a cross, classify as track-down-feasible / wide-spread-sticky / real-crash. Need 3-4 markets.
- [ ] **Small-live weather deployment: WS-passive at (p_in=0.50, p_out=0.90)** for 2-4 weeks. Focus: Seoul, Shanghai, Tokyo, Wellington, London. Expectation: ~6 fills/day, +6% ROI optimistic exit.
- [ ] **Document `traders_directionality.parquet`** — new canonical arb-detection source replacing `phantom_position_score < 2.0`. Add to `polymarket/research/docs/METRICS_REFERENCE.md`.
- [ ] **Document `split_position_signature`** — catches `splitPosition`-based directional construction (`0xd38b71f3` failure mode). 10 of top-50 by lifetime PnL fail this filter.

### exec hardening (post-smoke)
- [ ] **Resolution-handler path** in exec — needed before scaling, skippable for smoke.
- [ ] Tear down safety harness (`MAX_REAL_ORDERS`, operator-confirm) once stable.
- [ ] Provision VPS in non-blocked region (US East / Frankfurt / Tokyo) for unattended operation.
- [ ] Add `POLYMARKET_LEADER_RANKINGS_PATH` env var + read-on-refresh logic (gates multi-leader).

### later / v2
- [ ] Index NegRisk merge/split events from Polygon (true position-level PnL) for copytrade leaders/executor. MM politics audit has a working research implementation in [[mm_politics_negrisk_accounting_findings]], but generic copytrade position rebuild remains open.
- [ ] Mark-to-market on open positions.
- [ ] Multi-leader cohort orchestration (RTDS multi-subscribe, per-leader weights).
- [ ] Composite NegRisk handling on exec side.

---

## dali
- Block I Binance-return lead-lag feasibility gate: CONFIRM-CLOSE; train-selected OOS `L=60s H=5s latency=1s threshold=20bp` = -21.580c CI [-31.731c, -10.857c]; best OOS 1s diagnostic `L=2s H=1s latency=1s threshold=0bp` = -9.478c CI [-12.571c, -7.232c]. Timestamp alignment is clean for 1s+ but not sub-second; saved artifacts lack Binance/OKX OFI, and OKX klines alone would not supply it. See [[block_i_leadlag_feasibility_findings]].
- A18 passive reversion-to-microprice gate: 0 pooled market-cluster CI-positive rows; best `rolling_rank_sizing`/`passive_maker` W=5 H=30 = -1.232c CI [-1.631c, -0.924c], exec fill 0.09%. Verdict in [[block_a18_passive_reversion_findings]].

### Block A1: Live OFI/Maker sniff test ✅ (2026-05-28)
A0/A0b replayed into `data/analysis/block_a1_features.parquet`; cost-QA diagnostics written to `notes/dali/block_a1_results.md`.

- [x] Live sign convention resolved on A0 + A0b
- [x] Capture audits confirmed clean
- [x] Batch replay wrapper added
- [x] Depth-normalized OFI diagnostics written
- [x] Maker proxy and corrected cost overlay written
- [x] A1.1 segment/L2-proxy diagnostics written to `notes/dali/block_a11_plan_and_diagnostics.md`
- [x] **A1.2 MLOFI** (2026-05-28). Per-level OFI for L1..L10 with integrated / depth-weighted / exp-decay variants. **Result: L1 CKS wins at 1s/5s/30s by 15-25pp; only 300s depth-weighted shows +1.27pp delta. Keep L1 as A2 baseline; optional MLOFI sidecar logging.** See `notes/dali/block_a12_mlofi_findings.md`.
- [x] **A1.3 TOB imbalance level deep-dive** (2026-05-28). **Headline: `tob_imbalance_level` hits 73.7% at 5s top decile (CI [67.6%, 77.7%]), beats L1 OFI's 64.1%.** Persistent, market-dependent flip times (sub-second crypto vs minutes geopolitics). OFI adds no incremental info above TOB. Conditional TFI lights up to 62.6% in high-TOB regimes at 5s. **Promote `tob_imbalance_level` to a primary A2 candidate.** See `notes/dali/block_a13_tob_imbalance_findings.md`.
- [x] **A1.4 executable taker QA** (2026-05-28). **DECISIVE NEGATIVE: 0/12 market-horizon cells positive after entering at ask + exiting at bid.** Mean PnL -1300 to -2150 bps. The mid-mid alpha is real as descriptive pattern but is NOT tradeable as a taker on this universe at 5s/10s horizons. Mean gap vs A1's mid-return overlay: -560.8 bps (overlay was over-optimistic by that much). See `notes/dali/block_a14_executable_taker_findings.md`.
- [x] **A1.4b refined-exit taker on TOB signal** (2026-05-28). 0/36 cells positive. Best config: `cfg_signal_reversal` (gains +325 bps mean vs A1.4) but still -1060 bps average. Take-profit configs UNDERPERFORM fixed-5s (right-tail clipping artifact). Confirms refined exit alone is necessary but not sufficient. See `notes/dali/block_a14b_refined_exit_findings.md`.
- [x] **A1.4c maker-at-mid simulation** (2026-05-28). **1/16 markets "maker thesis lives": btc-updown-4h-1779912000 with W=10s, H=30s, exit_symmetric_maker → +554.9 bps mean PnL at 9.0% fill rate (5s adverse selection 248 bps).** 5 markets fills-too-rare, 10 markets adverse-selection-wipes-rebate. **Decisive next test is queue+latency model.** See `notes/dali/block_a14c_maker_at_mid_findings.md`.
- [x] **A1.4d tight-spread-conditional taker entry** (2026-05-28). 0/198 cells positive at 5s/30s; 1 positive cell at 300s on same BTC-4h-1779912000 market (S=500 bps, +843 bps mean) but CI [-799, 1770] crosses zero. Spread filter alone doesn't rescue taker. Universe selection is necessary but not sufficient. See `notes/dali/block_a14d_tight_spread_findings.md`.
- [x] **A1.5 TOB extensions** (2026-05-28). Multi-level imbalance: L1 wins at 1s/5s/30s; `exp_decay_alpha_0p5` wins at 300s by +4.92pp. Micro-price-as-target: 97.3% hit at 1s — but contaminated by autocorrelation (signal and target share imbalance term). Micro-price-change-as-signal: underperforms TOB by ~20pp. Keep L1 TOB primary; exp-decay at 300s sidecar. See `notes/dali/block_a15_tob_extensions_findings.md`.
- [x] **A1.4f combined refined-exit + tight-spread** (2026-05-28). 5/660 positive cells, ALL on BTC-4h-1779912000 at 300s fixed-horizon, CI [-611, 1741] (crosses zero). Combination didn't surface new winners — just confirmed the same single-market result. See `notes/dali/block_a14f_combined_findings.md`.
- [x] **A1.4g exit-family exploration up to 300s** (2026-05-28). 0/165 cells crossed zero. **BTC-4h-1779912000's best smart exit (asymmetric TP3000/SL300) is -492 bps mean.** Smart exits KILL the BTC-4h fixed-horizon winner. Reveals that the BTC-4h "edge" only exists at 300s fixed hold, not with intelligent exits. See `notes/dali/block_a14g_exit_family_findings.md`.
- [x] **A1.5b decoupled micro-price target** (2026-05-28). OFI vs micro-price target: +5.45pp at 1s, +0.09pp at 5s, -2.14pp at 30s, -1.04pp at 300s. TFI degrades vs micro-target everywhere. **Diagnosis: original OFI/TFI/TOB signal is mostly mean-reversion to micro-price (fair value), not multi-second alpha.** See `notes/dali/block_a15b_decoupled_findings.md`.
- [x] **A1.6 binary-bet hypothesis under non-overlap** (2026-05-28). **0/225 cells cleared CI robustness bar.** Critically, the A14f BTC-4h winner does NOT replicate: `btc-updown-4h-1779912000` at fixed_300s top_decile goes from +844 bps overlap (A14f) to -1968 bps non-overlap (A1.6) — same market, same signal, same horizon. **A14f was an overlap-math artifact.** Regime-filter (Lipton-style) didn't help. See `notes/dali/block_a16_binary_bet_findings.md`.
- [x] **A1.4h maker-at-mid non-overlap retest** (2026-05-28). **Maker thesis dead.** Same BTC-4h cell that was +554.9 bps in A14c overlap math is -451.3 bps non-overlap, fill rate collapses 9.0% → 0.2%. 15 of 16 markets verdict "fills too rare even in best case." 0/192 robust positive cells. The A14c maker positive was the same overlap-math artifact as A14f's taker positive. See `notes/dali/block_a14h_maker_non_overlap_findings.md`.
- [x] **A1.7 LightGBM Tier 2 minimal pass** (2026-05-28). **No Tier 2 edge found.** 0/10 markets with positive ML mean PnL after non-overlap backtest with executable cost. **Diagnostic: probability calibration breaks at high confidence.** Model is well-calibrated through P=0.65-0.70 then systematically underperforms at P≥0.70 (gap up to -16pp). Consistent with A15b's mean-reversion diagnosis — extreme OFI events mean-revert most strongly at 5s. **This forecloses Tier 3 (DeepLOB) too** because the signal-to-PnL gap is structural, not architecture-shaped. See `notes/dali/block_a17_lightgbm_findings.md`.

### Strategic state (post-A14h + A1.7 — original direct local microstructure branch falsified)

**The original direct local microstructure continuation branch is falsified across all three tiers and both execution modes.** Dali itself is not globally closed; it is the broader Polymarket research lineage and redesign trail that later fed Block K/MM/OD.

| tier / mode | result |
|---|---|
| Tier 1 taker (8 angles) | Dead under non-overlap math. A14, A14b, A14d, A14f, A14g, A1.6 all converge. |
| Tier 1 maker | Dead under non-overlap math. A1.4h confirmed +554 bps A14c result was overlap artifact (collapsed to -451 bps + fill rate 9.0% → 0.2%). |
| Tier 2 (LightGBM) | No edge. 0/10 markets positive. Calibration breaks at high confidence (-16pp gap at P≥0.70). |
| Tier 3 (LSTM/DeepLOB) | Foreclosed by A1.7 calibration diagnosis. Signal-to-PnL gap is structural mean-reversion to fair value, not model-architecture-shaped. |

**Diagnostic synthesis across the negative results:**
- A15b: signal mostly predicts mean-reversion to micro-price (fair value)
- A1.6: overlap math systematically over-states deployable PnL; non-overlap kills positives
- A1.4h: maker fill rate is flow-capped (Polymarket has slow takers despite deep books), not competition-capped
- A1.7 calibration: high-confidence ML predictions UNDERPERFORM because extreme OFI is the most mean-reverting

**The TOB/OFI signal is REAL but it's structural mean-reversion to fair value on Polymarket's wide-spread universe. That's not a model problem; it's a market-structure problem.** No additional capture (A2), execution refinement (A14e queue+latency), or model complexity (Tier 3 DeepLOB) will change this conclusion.

**Salvageable lessons from dali (worth preserving):**
- Live sign convention infrastructure
- Capture + replay + analysis pipeline
- TOB imbalance as a STATE variable (not for direct trading) — could feed copytrade leader screens, fair-value calculations
- Polymarket-specific microstructure facts: exchange-internal-leg artifact, depth ≠ flow, calibration-breaks-at-extremes
- Methodology lessons: non-overlap math by default; Briola caveat is real and observed; overlap math is treacherous

**Pivot to copytrade primary. Three candidate directions:**

1. **Copytrade scaling + smoke deployment (RECOMMENDED).** Most shovel-ready. Infrastructure mostly built; smoke target identified; first $10 fill is operationally hours of work. Leverages 9 months of prior work. 2. **Block I — Cross-platform arb (Polymarket vs Kalshi vs options).** Different cost structure means signals that died on Polymarket might survive on Kalshi. Different domain class. 3. **Block J — Resolution-criteria LLM signal scanning.** LLM reads news + resolution criteria. Heaviest setup but potentially high-value if it works.

**A0c capture runs as scheduled** for data archival, but no further analysis budget on dali microstructure.

### NEXT — Block A0c: targeted-universe 24h capture (runs 2026-05-29 morning)

Validate universe-selection criteria from A1.3 + A1.4 before committing to A2's VPS budget. Selection criteria: tight spread, deep relative depth, live-trade-rate (not just quote noise), explicit neg-risk inclusion.

- [ ] **Build A0c shortlist** — 15-20 markets meeting criteria. Crypto 4h up/down (multiple consecutive windows), daily crypto up/down (BTC/ETH/SOL), 2-3 NBA/Champions League in-game, 2-3 neg-risk outright, 2-3 tight-spread geopolitics. Skip equity_index / single_stock.
- [ ] **Launch 24h capture** as `block_a0c_targeted_<date>_morning`. Caffeinate locally; VPS for A2.
- [ ] **Capture audit** post-run: `notes/dali/block_a0c_capture_status_final.md`.
- [ ] **A1.x extended analysis** on A0+A0b+A0c combined: signal stability across captures with different universe composition.

### IMMEDIATE — Current default priority

The direct dali continuation branch is not the default next spend. Current priority stays copytrade smoke deployment unless Justin explicitly reopens a new Polymarket research branch.

- [ ] **PLAN.md sync + snapshot commit + tag** (copytrade § exec). Engineering-complete marker. ~10 min.
- [ ] **Slack colleague** about kernel encoding bug workaround before midas executor goes live.
- [ ] **Polymarket credentials into `.env`** — private key → derive_api_keys.py → .env. Read-only auth check first.
- [ ] **Pre-flight smoke target** — profile current open positions via Gamma + RTDS pre-subscription. Confirm no NegRisk markets.
- [ ] **First real-money smoke** — MAX_REAL_ORDERS=1, REQUIRE_OPERATOR_CONFIRM=true, SIZING_USD=10.

### A0c capture — lineage data, no default deep-analysis budget

- [ ] **Launch A0c 24h capture** as previously planned (2026-05-29 morning). Data stays in the dali lineage. No A1.x analysis pipeline run on it unless a specific redesign branch is reopened.

### Conditional / deferred dali follow-ups

Only worth doing if copytrade hits a clear ceiling and a fresh research bet is needed.

- [ ] **A14e queue + latency model** — would be execution autopsy, not signal rescue. Skip unless formal falsification of maker thesis is wanted.
- [x] **Block I first probe: Binance-return lead-lag into PM crypto direction** — completed 2026-06-02. Timestamp alignment is clean for 1s+ measurement, but the saved-artifact executable taker gate is negative after spread+fee: train-selected OOS `L=60s H=5s latency=1s threshold=20bp` = -21.580c CI [-31.731c, -10.857c]; best OOS 1s diagnostic = -9.478c CI [-12.571c, -7.232c]. Saved artifacts lack Binance/OKX OFI, so true OFI remains untested rather than a capture-spend priority. See [[block_i_leadlag_feasibility_findings]].
- [ ] **Block J resolution-criteria LLM scanning** — completely different domain class. Heaviest setup but potentially highest-value if successful.
- [ ] **Use TOB imbalance as a state variable in copytrade** — could feed leader screens or smart-order-routing logic without requiring direct microstructure trading.

### Block A2: design depends on A14e + A1.6 outcomes

**Strategic question for A2 design — now three branches:**

- If A14e maker queue+latency shows positive cells with reasonable CI → **A2 maker-focused**. Capture priorities: queue telemetry, RTDS proxyWallet stream, cancel latency measurement, post-fill adverse-selection windows.
- If A1.6 binary-bet hypothesis shows multiple 4h crypto markets surviving fixed-300s hold → **A2 binary-bet focused**. Capture priorities: longer-horizon windows aligned to 4h binary boundaries, more crypto 4h variants, TOB persistence across the full pre-resolution window. Different strategy entirely from microstructure HFT.
- If neither survives → **A2 deferred**; pivot to one of: copytrade scaling (already in progress), Block I (cross-platform Polymarket-vs-Kalshi), Block J (LLM-driven resolution-criteria signal), or new microstructure thesis on non-binary markets.

- [ ] Provision VPS in non-blocked region (US East / Frankfurt / Tokyo) — needed regardless of which thesis survives
- [ ] **Decide A2 thesis** post-A14e + A1.6
- [ ] Build A2 capture config from chosen thesis
- [ ] Run 1-2 week capture per chosen thesis

### Block C: sign convention (historical ✅ live 🔲)
- [x] Historical `maker_side` semantics confirmed: `BUY` → aggressor `SELL`, `SELL` → aggressor `BUY`. `historical_to_aggressor()` correct. No rerun needed. (2026-05-27)
- [x] Live `last_trade_price.side` semantics confirmed as token-side aggressor on A0 + A0b. (2026-05-28)

### Block B: historical TFI deep-dive ✅ (2026-05-27)
OUTCOME 3: Mixed Results. Most promising: equity-index operator-filtered (58.8% hit rate, 300s). Use as Block A target input only.
- [ ] (low priority) Sports pre-game vs in-game segmentation — needs external game-start data
- [ ] (low priority) AI/product per-market walk-forward at individual market level

### Reinterpretation (post-copytrade-relayer-implications, 2026-05-28)

The "relayer" addresses are Polymarket CTF Exchange v1 contracts. Active taker-order wallet is already emitted in the event's `maker` field per `_matchOrders` (`Trading.sol`). 99.7% maker-side trader-attribution confirmed. **No PnL/position rebuild needed for copytrade.** Style labels biased toward maker — add an `active_order_leg` flag before any style-based claim.

- [x] **v1 vs v2 emit pattern verified** — both ongoing. v2 contracts (`0xE111…996B`, `0xe2222…0F59`) emit OrderFilled with `taker = address(this)` the same way v1 did. (Done in copytrade-relayer-implications analysis.)
- [x] **`data_infra/operator_denylist.py` updated** — renamed `PURE_RELAYERS` to `EXCHANGE_INTERNAL_LEG_V1` / `_V2`, added v2 addresses, updated `block_e_lite.py` accordingly. (Done.)
- [ ] **Re-run Block B on event-source-classified data** (low priority, follow-up). Partition historical fills by emit-path and recompute TFI hit rates per partition. The "operator-filtered" hit rate of 58% is now interpretable as "TFI on the non-`_matchOrders` subset (likely single-sided `_fillOrder` fills)" — could be a real cleaner-attribution effect or a composition artifact. Block B's other findings (raw TFI, walk-forward) are unaffected.
- [ ] **Add `active_order_leg` flag** to `views.sql` and style-ratio computation before any style-based cohort claim. Domah ratio 7.89 → 5.67 after reclassification (still maker-heavy). Some traders flip category. Affects style framing only, not PnL.

### DEFERRED — Blocks D–F
> **Block D** (backtest extensions): trigger = ≥1 tradeable signal in Block A
> **Block F** (Optuna search): trigger = 3+ families, 24h+ each, 200+ `last_trade_price` events, sign resolved

- [ ] Block D: multi-strategy eval, realistic order rejection, per-category fees, walk-forward validation
- [ ] Block E: wallet/competition clustering — identify retail vs sophisticated per market
- [ ] Block F: Optuna TPE over rule-based strategies (200-500 trials, 60/20/20 split, net-of-cost Sharpe objective)

### PHASE 2 (post-signal validation)
- [ ] Block G: production execution stack (risk gates, circuit breakers, monitoring)
- [ ] Block H: LightGBM regression on OFI features — only after rule-based baseline shows edge
- [ ] Block I: cross-platform arb (Polymarket vs Kalshi vs options)
- [ ] Block J: resolution-criteria edge scanning via LLM at scale

### Research gaps
- [ ] Avellaneda-Stoikov adaptation for [0,1] bounded prices
- [ ] Maker rebate optimization (20-25% taker fee rebate, 50% Finance)
- [ ] Adverse selection empirics in retail-heavy prediction market flow
- [ ] Cross-platform lead-lag: Polymarket vs Kalshi vs Manifold
- [ ] LLM forecaster SOTA: Halawi et al. (2024), Schoenegger & Park (2024)
- [ ] Minimum useful OFI sample size (CKS used months; what's the floor?)

---

## done (recent)

- [x] **External strategy-note triage** (2026-05-29, v2). Uploaded a 24-strategy OFI/TOB/L2 mid-frequency research note (external origin). Archived as `notes/overview/foundations/external_ofi_tob_l2_midfreq_strategy_research.md`; triaged in `notes/dali/block_a1x_external_note_reconciliation.md`. **Corrected framing (v2 supersedes a too-aggressive v1):** A1.4–A1.7 closed the *directional-continuation* use of the local signal (taker + maker-at-mid), NOT the signal — the 73.7% TOB hit (A1.3) is real. Three genuinely-untested items remain: **(a) continuous rolling-rank sizing vs decile gating** (no new data), **(b) explicit mean-reversion-to-microprice framing incl. a passive/maker reversion route** (no new data — the diagnosis is a strategy instruction we never followed; key open cell = can a passive route capture the reversion at a fill rate that survives non-overlap?), **(c) true-L2 features** (#5/#6/#7/#15, needs A2 capture). Plus **#21 cross-market lead-lag** (off-book, Block I). Do-first = (a)+(b) on existing A0/A0b/A0c replay; do NOT re-run continuation taker/maker-at-mid. Copytrade stays primary by default but (a)/(b) are a few hours of replay on data we own.
- [x] **A1.4 executable taker QA** (2026-05-28). 0/12 cells positive; mid-mid alpha is real but wiped by spread crossing. Strategic implication: maker thesis pivot or tighter-spread universe before A2. See `polymarket/research/notes/dali/block_a14_executable_taker_findings.md`.
- [x] **A1.3 TOB imbalance level deep-dive** (2026-05-28). `tob_imbalance_level` is the strongest A1.x signal: 73.7% hit at 5s top decile. Beats L1 OFI; OFI adds nothing incremental. Persistence is market-specific. Promote to A2 primary candidate. See `polymarket/research/notes/dali/block_a13_tob_imbalance_findings.md`.
- [x] **A1.2 MLOFI** (2026-05-28). L1 wins at <300s. Don't redesign A2 around per-level OFI. See `polymarket/research/notes/dali/block_a12_mlofi_findings.md`.
- [x] Copytrade-relayer implications (2026-05-28). Active wallet is in `maker`, not missing. No PnL/position rebuild. Domah smoke cell intact. Style labels biased toward maker. `operator_denylist.py` updated v1/v2-aware. See [`polymarket/research/notes/copytrade/copytrade_relayer_implications.md`](../polymarket/research/notes/copytrade/copytrade_relayer_implications.md).
- [x] Relayer-identity dig (2026-05-28). The two "relayer" addresses are Polymarket's legacy CTF Exchange v1 contracts (standard + neg-risk). See [`polymarket/research/notes/copytrade/relayer_dig_findings.md`](../polymarket/research/notes/copytrade/relayer_dig_findings.md) and [`polymarket/research/notes/copytrade/block_b_reinterpretation.md`](../polymarket/research/notes/copytrade/block_b_reinterpretation.md).
- [x] Block A0 (24h) + A0b (12h) live capture complete (2026-05-27/28). A0: 2,095 trades / 12 markets. A0b: 6,063 trades / 9 markets. Crypto 4h up/down + NBA in-game = volume-rich families. AI/product family still too thin to validate Block B finding.
- [x] Block B: historical TFI deep-dive complete (2026-05-27) — **reinterpretation pending; see dali / Reinterpretation below**
- [x] Historical sign convention audit complete (2026-05-27)
- [x] Block A0 smoke capture + 12-market shortlist locked (2026-05-27)
- [x] Dali infrastructure: market screen, TFI baseline, live CLOB capture, OFI replay, backtest engine
- [x] Obsidian vault consolidated into repo root (2026-05-27)
