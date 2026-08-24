---
title: "External Repo Audit — 5 quant GitHubs triaged against our live threads (MM-first)"
tags: [handoff, external-research, repo-audit, market-making, options-delta, crypto, skills, tooling]
created: 2026-06-28
status: assessment complete — adoption gated, concrete next steps proposed (no code changes yet)
owner: justin
hubs:
  - strat_market_making
  - COWORK
  - TODO
  - POLYMARKET_BRAIN
relationship: >
  First systematic multi-repo external audit. Triages microsoft/qlib, wilsonfreitas/awesome-quant,
  goldmansachs/gs-quant, stefan-jansen/machine-learning-for-trading, and
  cantaro86/Financial-Models-Numerical-Methods against the live threads, MM-first. Companion to the
  prior one-repo evals [[2026-05-31_kronos_hermes_eval]] (Kronos/Hermes) and the imported library
  [[external_ofi_tob_l2_midfreq_strategy_research]]. Feeds [[strat_market_making]] MM Path B
  ([[mm_backtest_research_roadmap]] / [[mm_backtesting_methodology_explainer]]).
---

> Hub: [[strat_market_making]] · [[COWORK]] · [[TODO]] · [[POLYMARKET_BRAIN]]

## Summary

- **What this is:** an audit of five external GitHub repos Justin flagged, scored for what we can actually
  *take* — as skills, research, execution code, or reference — and ranked **MM-first** because the live
  priority is the Polymarket market-making backtester ([[mm_backtest_research_roadmap]]).
- **"When did we last do a repo audit?"** There is **no recurring repo/skills-audit skill or cadence** in
  this vault. The closest prior passes were one-repo evals: **[[2026-05-31_kronos_hermes_eval]]** (~4 weeks
  ago) and the **2026-05-29** imported [[external_ofi_tob_l2_midfreq_strategy_research|OFI/TOB/L2 library]].
  An informal "find the right MM-backtest repo" scan around **2026-06-20** is how **hftbacktest** already
  became our queue-model reference. **None of these five has been formally audited before — this is the
  first systematic sweep.**
- **One-line takeaway:** the highest-value items are not whole platforms — they are **`hftbacktest`** (the
  MM/queue-model backtester we're already modelling our interface on) and **`awesome-quant` itself** (the
  index that surfaces hftbacktest, PyLOB, nautilus_trader, backtester-mcp). qlib/gs-quant/ml4t/FMNM are
  **reference-and-borrow**, not adopt — and one (FMNM) is **AGPL-3.0**, so reimplement, never vendor.
- **Discipline lens:** every verdict is filtered through our own law — *no infra before signal*, *no ML
  before a rule-based baseline*, *no cross-import between the polymarket and crypto projects* ([[CODEX]]).
  "Useful" here means "borrow a pattern / reference an implementation," not "bolt on a framework."

## Why this audit, and the lens I used

Justin asked for a proper audit of five repos and "anything useful we can take." The trap with big quant
repos is adopting a platform before we have a validated signal — exactly what [[CODEX]] § Anti-patterns
forbids. So each repo is scored on **what is liftable now** against a specific live thread, with the licence
and the integration cost stated, and the verdict capped at one of: **Adopt / Borrow-pattern / Reference /
Skip**. MM is weighted first because [[strat_market_making]] MM Path B is the active build.

**Practical example of the lens:** qlib ships a slick RL "order-execution" module. It *sounds* like MM, but
it optimises how to liquidate a parent order vs TWAP (taker-side impact) — it has nothing to say about
passive-maker queue position or fill probability, which is the entire MM backtester problem
([[mm_backtesting_methodology_explainer]]). So qlib scores **Reference** for a possible future OD/execution
reopen, **not** Adopt for MM. That distinction is the whole point of the audit.

## Per-repo verdicts

### 1. wilsonfreitas/awesome-quant — the meta-resource (curated index)

**Facts:** curated list (the canonical "awesome" index of quant libraries), actively maintained, MIT-style
list licence. Not code — a map.

This is the **highest-leverage answer to the literal request** ("find useful repos for our work"), because
it *is* that engine. Mining it for our threads surfaced, in priority order:

| Repo (from the list) | Lang | Why it matters to us | Thread |
|---|---|---|---|
| **hftbacktest** (`nkaz001`) | Python/Rust | "HFT & **market-making** backtesting tool — accounts for **limit orders, queue positions, and latencies**, full tick data for trades and order books." This is the exact object the MM roadmap is hand-rolling (queue model + latency model + fill sim). Already our reference for ProbQueueModel1/2/3. | **MM (top)** |
| **PyLOB** (`DrAshBooth`) | Python | Fully-functioning fast limit-order-book matching engine. Reference + unit-test oracle for our Phase-1 "reconstruct book state from L2 events" builder, and for empirically checking Polymarket's FIFO price-time priority. | **MM** |
| **nautilus_trader** (`nautechsystems`) | Python/Rust | High-performance event-driven backtester **with the same code path for backtest and live** — literally our Phase-1 Join-2 goal ("replay historical L2 + live VPS stream, same code"). Heavy to adopt; gold-standard architecture to copy. | **MM / execution** |
| **backtester-mcp** (`bcosm`) | Python | Local-first backtester with **built-in overfitting checks (PBO, deflated Sharpe, bootstrap CI, walk-forward)** *and a native MCP server*. Maps onto both our CPCV/overfitting-audit culture and our MCP/skills setup. | **crypto / skills** |
| **vectorbt** (`polakowo`) | Python | Vectorised backtesting/research toolkit — fast parameter sweeps. Pattern reference for crypto WF/CPCV research. | crypto |
| **fypy**, **Pyderivatives**, **vollib** | Python | Option pricing: Fourier/characteristic-function models, Heston/Kou/Bates, risk-neutral densities, IV/greeks. | **OD** |

**Read:** awesome-quant earns a standing place in the brain. It found four MM-relevant tools in one pass and
the option-pricing libs that map onto OD. Treat it like we treated the OFI note: triage a shortlist into the
vault, don't import wholesale.

**Verdict: ADOPT as a standing resource.** It is the cheapest possible "repo radar."

### 2. (from awesome-quant) hftbacktest — the single most useful repo for MM

**Facts:** `nkaz001/hftbacktest`, Python + Rust, active, focused specifically on HFT/market-making backtests
with realistic fill mechanics. (Already cited across [[mm_backtesting_methodology_explainer]],
[[mm_queue_model_research]], [[mm_backtest_research_roadmap]].)

It implements, as a maintained library, the three things Phase 0 is trying to specify from scratch: the
**event-driven queue-model interface** (`on_trade` / `on_depth` handlers, not a monolithic
`queue_model(book,order)`), the **probabilistic L2 queue models** (uniform / power-law / ML — our
optimistic↔pessimistic bracketing), and a **distributional latency model**. Our Decisions D2 (queue
interface) and D3 (latency) are essentially "re-derive hftbacktest's API." 

**Caveat (be honest):** it's built for centralised-exchange tick tapes; Polymarket is a slow, hybrid
off-chain-match/on-chain-settle venue with anonymous L2 ([[mm_clob_capture_semantics]]). So this is
**adopt-the-design, and probably the engine for the fill-sim/queue layer** — not a drop-in for the whole
pipeline. It does not solve adverse selection (D6) or the Polymarket-specific matching verification (Phase 2).

**Verdict: ADOPT (design, likely engine) for the MM fill-sim/queue layer.** Top concrete action below.

### 3. microsoft/qlib — AI quant *platform*

**Facts:** **MIT**, **42.9k★**, 6.7k forks, 2,065 commits, very active (now bundled with RD-Agent LLM
auto-factor mining). Full ML pipeline (data→train→backtest), point-in-time DB, Alpha158/Alpha360 factor
sets, nested decision/execution framework, **RL for order execution** (TWAP/PPO/OPDS/OPD), high-freq
examples. Equities / China-A-share centric, daily-bar default.

- **Crypto (`live_trading`/`topics`/`infrastructure`):** MODERATE conceptually, LOW direct. Its
  point-in-time discipline and nested execution echo our lookahead-free + walk-forward culture, and Alpha158
  is a nice feature catalogue to mine. But it's an opinionated platform — adopting it wholesale is precisely
  "infra before signal." Borrow ideas, not the framework.
- **MM:** LOW. The RL order-execution module is taker-side parent-order liquidation, **not** maker queue/fill
  modelling (see the practical example above). Do not confuse it with hftbacktest.
- **OD / execution:** the `examples/rl_order_execution` baselines are a decent reference *if* OD execution
  sizing/liquidation modelling ever reopens.
- **Skills:** RD-Agent's auto-factor loop is interesting but collides with our "no ML before a rule-based
  baseline" law ([[CODEX]], Briola caveat). Watch-item, not adopt.

**Verdict: REFERENCE (borrow ideas).** Don't adopt the platform. Biggest legitimate pull is the
RL-order-execution examples for a future OD/execution branch, plus factor inspiration for crypto.

### 4. stefan-jansen/machine-learning-for-trading (ML4T)

**Facts:** **MIT**, **19.3k★**, 5.3k forks. Companion code to the *Machine Learning for Trading* book
(3rd ed), "from data sourcing to live execution." Educational notebooks across the full ML-for-trading
workflow.

- **Crypto (most relevant):** MODERATE–HIGH as reference. Strong, liftable chapters on **GARCH / time-series**,
  **cointegration & pairs trading** (directly the dormant `statarb` stub in `live_trading`), gradient-boosting
  (LightGBM/XGBoost) workflows, and — importantly — **backtest overfitting / deflated Sharpe / multiple
  testing**, which reinforces our CPCV + overfitting-audit discipline. Also a deep-RL-for-trading chapter.
- **MM / OD:** LOW direct.
- **Caveat:** it's *book* code, not production libraries — borrow patterns, not infra; some notebooks carry
  dated deps.

**Verdict: REFERENCE / borrow (MIT-safe).** Concrete pull when the statarb stub is built: the
cointegration/pairs + GARCH notebooks, and the overfitting chapter as an independent cross-check on
our CPCV engine (`infrastructure/walkforward/cpcv_engine.py`).

### 5. cantaro86/Financial-Models-Numerical-Methods (FMNM)

**Facts:** **AGPL-3.0** (strong copyleft — this matters), widely-cited (~6k★) notebook collection on
quant-finance numerical methods. Covers Black-Scholes, SDE simulation & parameter estimation,
**Fourier/COS pricing**, **Merton & Kou jump-diffusion**, Lévy (Variance-Gamma / NIG), **Heston**
stochastic-vol, PDE/finite-difference, American options (LSMC), and **Kalman filter + Ornstein-Uhlenbeck
mean-reversion / pairs trading**.

- **OD (most relevant):** HIGH as a reference implementation. Our OD work already prices **Merton, Kou,
  Edgeworth** fairs and inverts digitals ([[od_methodology_realism_audit_findings]]); FMNM is the canonical,
  readable source for exactly those characteristic-function / Fourier-inversion / implied-vol routines. When
  OD reopens, this is the textbook code to check our math against.
- **Crypto statarb:** MODERATE — the Kalman + OU + pairs-trading notebooks map onto the statarb stub.
- **MM:** LOW.
- **Licence caveat (decision-relevant):** **AGPL-3.0**. Fine to read and learn from; **do not copy code into
  this repo** — reimplement clean from the math. Flagging explicitly so no one pastes a notebook in.

**Verdict: REFERENCE-ONLY (AGPL → reimplement, never vendor).** High value as the OD/statarb math spec.

### 6. goldmansachs/gs-quant

**Facts:** **Apache-2.0**, **10.6k★**, 159 watchers, very active (release 2.0.3, Jun 9 2026). GS's Python
toolkit for derivatives structuring, pricing, and risk. **Most APIs require a GS Marquee institutional
client id/secret** ("available to institutional clients of Goldman Sachs").

- The genuinely valuable parts (the Marquee-backed risk-transfer pricing/risk) are **credential-gated** —
  we don't have institutional access, so they're inert for us.
- **OD:** LOW–MODERATE and wrong-instrument: it prices vanilla/exotic derivatives on real underlyings, not
  binary [0,1] prediction-market digitals. FMNM/fypy serve OD better.
- **MM:** none.
- **Skills (one genuine nugget):** the repo now ships a `.claude/skills/` directory — a real example of a
  quant desk packaging Claude skills. Worth a look purely as a **skills-authoring pattern** for our own
  [[SKILL_MAP]] work.

**Verdict: SKIP for now** (credential-gated, equities/derivatives-centric, wrong instrument). Keep only as a
risk-API design reference and a skills-packaging example.

## Priority stack (MM-first)

| Rank | Item | Verdict | Thread | Licence |
|---|---|---|---|---|
| 1 | **hftbacktest** | Adopt (design + likely engine) | MM fill-sim/queue layer | OSS |
| 2 | **awesome-quant** | Adopt (standing radar) | all | list |
| 3 | **PyLOB / nautilus_trader / backtester-mcp** | Borrow-pattern | MM book-builder · backtest=live · overfitting/MCP | OSS |
| 4 | **FMNM** | Reference-only (reimplement) | OD math · statarb (Kalman/OU) | **AGPL-3.0** |
| 5 | **ml4t** | Reference / borrow | crypto statarb (pairs/GARCH) · overfitting | MIT |
| 6 | **qlib** | Reference (ideas) | OD/execution RL · crypto factors | MIT |
| 7 | **gs-quant** | Skip | (skills-packaging example only) | Apache-2.0 |

**Read of the table:** the value concentrates at the top and is squarely on the most-pressing thread. Ranks
1–3 are all MM-backtester pieces; 4–6 are reference libraries for OD and crypto that cost nothing to consult
and shouldn't be adopted as frameworks; 7 is a skip with a small skills footnote.

## Ready-to-run next steps (no code changes yet)

1. **MM (do first) — pre-registered Codex evaluation of hftbacktest against Phase-0/1.** Prompt Codex (read
   order per [[POLYMARKET_BRAIN]] § Prompt Rule) to: (a) check whether hftbacktest's queue-model + latency-
   model API satisfies Decisions **D2/D3** in [[mm_backtest_research_roadmap]]; (b) confirm it can ingest our
   per-shard L2 parquet ([[polymarket_l2_ingestion]]); (c) verify the FIFO/price-time assumption is
   configurable for Polymarket; (d) decide *adopt-engine vs adopt-design-only*. Output →
   `polymarket/research/notes/market_making/data_ingestion/hftbacktest_eval_findings.md`. **Gate before any
   code:** it must not pull us into "infra before signal" — it's allowed because the L2 capture is already
   live and Phase-0 is explicitly a tooling decision.
2. **Standing radar — triage shortlist note.** A short Librarian/foundations note pinning the awesome-quant
   picks (hftbacktest, PyLOB, nautilus_trader, backtester-mcp, fypy/Pyderivatives) with one line each, mirror
   of [[external_ofi_tob_l2_midfreq_strategy_research]]. Optional recurring quarterly diff pass.
3. **Cross-link** this audit from [[strat_market_making]] (MM Path B) and add a one-line pointer in
   [[TODO]] § MM Path B so the hftbacktest decision is tracked.
4. **Defer** FMNM/ml4t/qlib to the moment their thread is active (OD reopen; statarb stub build), with the
   AGPL caveat on FMNM recorded.

## Decision

Adopt **hftbacktest** (design, likely engine) into the MM backtester plan and **awesome-quant** as a standing
resource; treat **PyLOB / nautilus_trader / backtester-mcp** as pattern references for the same build. Hold
**FMNM** (AGPL → reimplement) and **ml4t** (MIT) as on-tap references for OD and crypto-statarb respectively;
keep **qlib** as an ideas reference; **skip gs-quant** (credential-gated, wrong instrument). No code until the
Step-1 Codex eval returns — this stays inside "no infra before signal" because Phase-0 is itself the
sanctioned tooling-decision step.

## Cross-links

- MM build: [[strat_market_making]] · [[mm_backtest_research_roadmap]] · [[mm_backtesting_methodology_explainer]] · [[mm_queue_model_research]] · [[polymarket_l2_ingestion]] · [[mm_clob_capture_semantics]]
- Prior external evals: [[2026-05-31_kronos_hermes_eval]] · [[external_ofi_tob_l2_midfreq_strategy_research]] · [[block_a1x_external_note_reconciliation]]
- OD reference target: [[od_methodology_realism_audit_findings]] · [[strat_options_delta]]
- Law / discipline: [[CODEX]] · [[COWORK]] · [[TODO]] · [[SKILL_MAP]]
