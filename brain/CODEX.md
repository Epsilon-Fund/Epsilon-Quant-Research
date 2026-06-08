---
title: "Codex Orientation — Epsilon Quant Research"
created: 2026-06-05
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
# Codex Orientation — Epsilon Quant Research

> Read this file at the start of every Codex session.
> It is the single source of truth for what this repo is, where things live, and how Codex should operate.
> **Also read `brain/COWORK.md`** — it shares the same brain and the active task list lives in `brain/TODO.md`.

---

## Agent Bootstrap (do this before anything else)

1. Determine your role: Codex -> `local_agents/codex.md`; Cowork/Claude Code -> `local_agents/cowork.md`.
2. If that file does not exist, create `local_agents/` and copy the matching template from `brain/agents/templates/<role>.local.template.md` into it, then tell the user "seeded your local <role> overlay - edit it to set your personal preferences."
3. Read your `local_agents/<role>.md` overlay (personal style), then the shared law `brain/CODEX.md` or `brain/COWORK.md`, then `brain/VAULT_MAP.md`, then `brain/TODO.md`.

Precedence: personal overlay = voice/preferences; shared `CODEX`/`COWORK` + repo invariants = law (always win).

## What this repo is

Two independent research projects, one repo:

| Project | Root | What it does |
|---|---|---|
| **Polymarket alpha** | `polymarket/research/` + `polymarket/execution/` (Midas) | Prediction-market trading: copytrade (Midas bot, per-leader audit), Block K (maker + options-delta), and the dali/Polymarket research lineage |
| **Crypto live trading** | `live_trading/` + `topics/` + `infrastructure/` | Momentum / stat-arb / BB-breakout strategies on Binance; walk-forward + CPCV research engine |

They share **no code**. Separate venvs, separate pyproject/requirements. Never cross-import.

---

## Brain — shared context hub

```
brain/
├── COWORK.md     ← Cowork session orientation (mirrors this file's strategic state)
├── CODEX.md      ← THIS FILE — Codex session orientation
├── TODO.md       ← AUTHORITATIVE live task list — read before suggesting next actions
├── POLYMARKET_BRAIN.md ← Obsidian map for Polymarket strategy clusters
├── glossary.md   ← term definitions across both projects
├── agents/templates/ ← shared templates for local-only agent overlays
├── handoffs/     ← dated cross-thread snapshots (Obsidian wikilinks, human-readable)
```

`brain/TODO.md` is the ground truth for what's active and what's next. Always read it before starting work.

Findings and notes live in strategy subfolders under `polymarket/research/notes/` (Polymarket) and `docs/STRATEGY_REFERENCE.md` (crypto). Do not create prompt files or new top-level brain files without a clear reason — append to an existing hub or create a findings note in the right subfolder.

For data artifacts, do not rediscover the file tree from scratch unless the task is explicitly a fresh audit. Use [[polymarket_data_manifest]] for Polymarket Parquet/CSV/JSONL/DuckDB/raw data, [[polymarket_csv_output_audit]] for result-table layout, [[polymarket_plot_gallery_index]] for generated figures, [[storage_consolidation_audit_2026_06_05]] for disk-pressure/storage-layout decisions, and [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] for crypto/live-trading Parquet, pickle, CSV, and JSONL artifacts.

---

## Obsidian wikilinks

Notes use basename-style wikilinks so the graph connects. This matters because Obsidian makes the research graph visible: hubs, orphans, stale branches, and broken links are easy to spot, and future agents can load the right context without rediscovering it from scratch. When you create a new `.md`:
1. Add wikilinks back to the relevant hub (see Hub Map below).
2. Use a unique basename — duplicates break navigation.
3. Save findings to the correct location (see "Where to write things").

**Hub Map — link new notes back to these:**

| New note type | Link back to hub |
|---|---|
| Polymarket findings | [[COWORK]] § Canonical strategy docs AND [[POLYMARKET_BRAIN]] |
| Crypto strategy findings | `docs/STRATEGY_REFERENCE.md` |
| Live trading architecture | `live_trading/CLAUDE.md` |
| Cross-project context / snapshots | `brain/handoffs/<date>_<topic>.md` (new file OK) |
| Glossary additions | `brain/glossary.md` |

---

## Where to write things

| Content type | Location |
|---|---|
| Polymarket synthesis / plain English | `polymarket/research/notes/overview/synthesis/<topic>.md` |
| Polymarket foundations / deep research | `polymarket/research/notes/overview/foundations/<topic>.md` |
| Polymarket data quality / methodology | `polymarket/research/notes/overview/data_quality/<topic>.md` |
| Polymarket market maps / screens | `polymarket/research/notes/overview/market_maps/<topic>.md` |
| MM market-making findings | `polymarket/research/notes/market_making/<topic>_findings.md` |
| OD options-delta findings | `polymarket/research/notes/options_delta/<topic>_findings.md` |
| copytrade findings | `polymarket/research/notes/copytrade/<topic>_findings.md` |
| dali / Polymarket research-lineage findings | `polymarket/research/notes/dali/<topic>_findings.md` |
| Crypto strategy / WF findings | `topics/<strategy>/research/` or `docs/` |
| Live trading architecture notes | `live_trading/CLAUDE.md` (append) |
| Strategic snapshots / cross-thread | `brain/handoffs/<YYYY-MM-DD>_<topic>.md` |
| Task list updates | Edit `brain/TODO.md` directly |
| Code / scripts | Under the relevant project — never in `brain/` |
| Codex results | Relevant `polymarket/research/notes/<cluster>/` folder as `*_results.md` or `*_findings.md` |
| Polymarket CSV report outputs | `polymarket/research/data/analysis/csv_outputs/<cluster>/<topic>.csv`; explain any non-obvious columns in the linked `.md` |
| Polymarket chart outputs | `polymarket/research/data/analysis/plots/<cluster>/<topic>.<png/svg>` for new figures, unless an existing script already owns a local plot folder |
| Polymarket data manifests | Update [[polymarket_data_manifest]] for durable data-family changes; do not link every Parquet/JSONL shard individually |
| Crypto/live-trading data manifests | Update [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] for durable cache/pickle/CSV/JSONL family changes |

---

## Markdown quality standard

Every `.md` that Codex writes must be understandable to a future human or agent without opening the script first. Do not paste raw tables or terse metrics without explaining what they mean.

Required structure for findings/results notes:

1. **Self-contained strategy headline:** title the note with what the strategy actually is in plain English. Internal labels like K5, KPEG, A14, Kronos, or Hermes may appear in parentheses, but never as the only headline. A reader who just arrived should know the trade idea, data source, and development status within the first few lines.
2. **Top plain-English summary:** immediately after frontmatter and hub/table-term backlinks, add a short section named `## Plain-English Summary` or `## Summary`. It must be 2-5 bullets or a tight paragraph saying: what this note is about, why it was written, what data/experiment it covers, and the one-line takeaway/status. This comes before results tables, charts, verdicts, or implementation details.
3. **Plain-English headline:** say what happened, why it matters, and whether the result changes the plan.
4. **Design / phase explanation:** define the strategy, experiment design, sample split, phases, gates, and any assumptions before showing results.
5. **Practical example:** include a small concrete example when a concept, phase, formula, or gate could otherwise feel abstract. Example: describe one hypothetical trade, market, fill, hedge, or row and show how the rule would treat it.
6. **Table meaning:** for every table, explain the unit of observation, filters, sample, metric definitions, and every non-obvious column. Do not dump raw CSV headers like `far_absz_ge1_all_tau`, `CI lift`, `prem retained`, or `top_mkt_share` without a local explanation or a wikilink to a definition note. If column names are compact, add a short column glossary immediately before or after the table.
7. **Definition links:** use wikilinks for referenced notes, hubs, prior findings, strategy labels, bucket glossaries, and reusable definitions. If a shorthand term is reused across notes, define it once in a master definition note such as `[[polymarket_table_dictionary]]` or `[[glossary]]`, then link to it instead of leaving readers to remember where the term came from.
8. **Charts when useful:** if a table, surface, distribution, time series, confidence interval, or bucket comparison is easier to understand visually, generate a chart with Python and embed it in the markdown. Every chart needs a caption or nearby read explaining axes, units, filters/sample, color scale, and what the reader should notice. Do not add decorative charts; use visuals to make decisions and diagnostics easier to inspect.
9. **Read / interpretation:** after each important table or chart, add a plain-English read. Say what would count as good/bad and what the actual result implies.
10. **Reread pass:** after writing any `.md`, reread it as if you just arrived cold. Fix every orphaned shorthand: CSV column names, indicator names, bucket labels, phase names, code names, chart axes, and plot legends must either be explained in the note or linked to a definition/hub. Do this before considering the note complete.
11. **Decision and next step:** state the gate outcome, caveats, and the concrete next action or "do not continue this branch" conclusion.

Clean note shape beats dense note shape. Prefer short sections with clear labels over one giant results dump. If a result needs phases, write the phases as named sections. If a design has variants, describe the variants before presenting the comparison table. Use practical examples to make rules inspectable, not as decoration. Do not assume the reader remembers project code names; write every note so it can be independently read from a cold start. Any table that cannot be explained clearly should not be included yet.

Do not manually hard-wrap prose mid-sentence or mid-list-item. Let markdown/editor wrapping handle long lines. In particular, do not write a bold label and then continue the same sentence on an indented next line, because it reads like a broken or subordinate thought. Keep examples like `**Phase 3, risk caps:** keep the same unhedged lifecycle, but skip fills that would push episode dollar-delta exposure above a cap.` on one logical line, or split into two real sentences/paragraphs.

---

## Project 1 — Polymarket alpha

### Active threads (as of 2026-06-01)

**MM — Market-Making**

Hub: `polymarket/research/notes/market_making/strat_market_making.md`
Historical Block K maker work is split out. Single-venue Polymarket maker is CLOSED; live value is real-maker moat diagnosis plus copy/learn-the-winners.

Open tasks: decompose why top-3 makers dominate (K5 wallet data) → capacity/moat diagnosis. Paper-trade only after an OOS-cleared design exists.

**OD — Options-Delta**

Hub: `polymarket/research/notes/options_delta/strat_options_delta.md`
OD Strategy A v2 lifecycle failed the primary OOS far-|z| gate under global time embargo. Phase 2 hedge frontier ran anyway as a diagnostic and did not rescue the unhedged gate; Kronos/HAR forward-vol bake-off stays gated off unless the unhedged lifecycle gate/capital assumption is explicitly reopened.

Open tasks: none on OD execution by default; keep Kronos/Hermes forward-vol work gated until an explicitly reopened OOS design clears.

**copytrade (parallel thread)**

Midas bot, per-leader audit, path to first real-money smoke.
Hub: `brain/TODO.md` § copytrade + `polymarket/research/notes/` (relayer, Domah audit files).

**dali / Polymarket research lineage**

Not globally closed. The original direct local microstructure continuation branch had multiple negative/falsifying tests, but dali is the broader Polymarket research lineage and redesign trail that fed Block K/MM/OD. Treat individual branches as falsified when the notes say so; do not label the whole line closed unless Justin explicitly says it.

### Key invariants (never violate)
- `PYTHONPATH=. uv run python …` from inside `polymarket/research/`
- DuckDB over Parquet, no Postgres, no DB server
- All metrics must be lookahead-free (filter by timestamp before aggregating)
- Parquet shards are append-only — never edit in place
- Addresses: lowercase, `0x`-prefixed
- Non-overlap math by default on all backtest analysis
- Require CI (not just point estimates) before calling a result "positive"

### Run environment
```bash
cd polymarket/research
uv run python scripts/<script>.py
```

---

## Project 2 — Crypto live trading

### Architecture
```
live_trading/
├── app.py                  ← Unified Streamlit app (all strategies)
├── pages/
│   ├── 1_Dashboards.py     ← Live signals + trade forms
│   ├── 2_Trade_Log.py      ← Trade journal
│   └── 3_Portfolio.py      ← Equity/drawdown charts
├── shared/                 ← ALL reusable logic — data_loader, charts, styles, components
└── dashboards/
    ├── momentum/           ← COMPLETE (ETH, SOL, BNB, ADA, XRP, AVAX, BTC)
    ├── statarb/            ← STUB — strategies.py, dashboard.py, optimise.py, streamlit_app.py not written
    └── bbbreakout/         ← STUB — same state as statarb
```

Full architecture reference: `live_trading/CLAUDE.md`
Strategy + engine reference: `docs/STRATEGY_REFERENCE.md`

### Active universe (momentum, live)
ADAUSDT, AVAXUSDT, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT

### Key invariants
- `data_dir` is always an absolute path to the strategy's dashboard dir
- Widget keys prefixed: `f"{prefix}_..."` — prevents DuplicateWidgetID
- `trades.json` is append-only
- `positions.json` is read fresh (uncached) every render cycle
- Python ≥ 3.10 required
- Walk-forward engine: `infrastructure/walkforward/wf_engine.py`
- CPCV engine: `infrastructure/walkforward/cpcv_engine.py`

### Research pipeline
Latest research lives in CPCV notebooks:
- Momentum: `topics/momentum/strategies/momentum_cpcv/`
- BB breakout: `topics/momentum/strategies/bb_cpcv/`
- Cross-sectional (in-progress): `topics/momentum/xs_momentum/`

---

## Cowork vs Codex split

| Tool | Role |
|---|---|
| **Cowork** | Strategic discussion, prompt drafting, interpretation of Codex outputs, updating `brain/` and `polymarket/research/notes/` docs |
| **Codex** | Implementation, long-running computation, producing CSVs / findings docs / scripts |

Codex results get saved as `*_results.md` or `*_findings.md` in the appropriate notes subfolder, then linked from the relevant hub. Cowork interprets and updates `brain/TODO.md`.

When Cowork drafts a prompt for Codex, the prompt must instruct Codex to run the Agent Bootstrap first, then read `brain/CODEX.md`, `brain/TODO.md`, `brain/COWORK.md`, `brain/POLYMARKET_BRAIN.md`, and the relevant strategy hub. Prompt text should stay in chat, not in repo files.

If the task touches generated datasets, result tables, notebooks, plots, or "orphan"/attachment cleanup, the prompt does not need a full file inventory. It should simply add: "For data artifacts, use [[polymarket_data_manifest]], [[polymarket_csv_output_audit]], [[polymarket_plot_gallery_index]], and/or [[docs/CRYPTO_DATA_MANIFEST|crypto data manifest]] as applicable; do not relink raw shards one by one."

---

## Anti-patterns — enforce these

- No infra before signal is validated
- No ML until rule-based baseline shows edge (Briola caveat)
- No CI-free "positive" results — always compute confidence intervals
- No cross-importing between polymarket/ and crypto projects
- No optimising on insufficient data — respect Task 5 triggers in Polymarket

---

## Realism calibration — be strict, but not theatrically strict

Gates must be honest in BOTH directions: don't wave through fragile results, and don't kill real ones with bars that don't apply. When you set a gate, state explicitly which knobs are *fair* and which are *borrowed or harsher than the instrument warrants*. Four rules:

1. **OOS only with enough data.** An out-of-sample / embargo split is only meaningful when the sample can support it. If the data is a single short window (e.g., one 24h capture), treat it ALL as training and say so — do not manufacture a train/test split that has no power and then call the noise a "fail." State sample size and why a split is or isn't warranted. (Conversely, when you DO have months of independent resolution dates, an OOS split is required — see the same-day Arm T pass, which had ~2 months and split legitimately.)

2. **Borrowed baselines are warnings, not kill switches.** A structural/MM baseline imported from a *different* instrument (e.g., the crypto-4h `1.98c` queue baseline) may be reported as a diagnostic, but it must NOT be the official gate for a different instrument unless it is re-derived for that instrument. Label it "borrowed" wherever it appears. Killing a touch-market result with a crypto-4h up/down baseline is pushing harder than reality supports.

3. **Separate assumptions from live-only unknowns (ship an assumption ledger).** Every gate verdict carries a short two-part ledger: (a) **modeled assumptions** — e.g., the 5% non-incumbent capacity share, taker-entry-at-ask+fee, empirical base rates from history; and (b) **things only live trading/capture can resolve** — passive/maker fill rate, real non-incumbent fill share, adverse selection, instrument-specific structural baseline, edge persistence vs latency/lead-lag. A "MERITS-BUILD" that depends on untested live quantities is a **"merits a live MEASUREMENT loop," not "merits a trading system."** Say which one it is.

4. **Statistical survival ≠ economic materiality.** A result can clear OOS + CI + FDR and still be too small to deploy after a realistic capacity haircut. Always report the deployable per-contract edge in absolute terms (cents/contract after haircut), not just the CI sign. A lower-CI of `[0.01c, 0.04c]` is "statistically positive, economically ~zero" — call it that.

Worked anchor: the 2026-06-02 same-day OD Arm T survivor passed OOS+BH (fair) but netted ~0.02c/contract after the 5% haircut (immaterial), was killed on paper only by a *borrowed* 1.98c baseline (too harsh), and its real edge driver (passive fill rate, touch-specific baseline, true capacity) is live-only. Correct disposition: live measurement loop, not a trading system. See [[od_same_day_crypto_pricing_gate_findings]].

5. **Reopen filter — the discipline cuts both ways.** A closed branch merits a realism re-audit ONLY if it died on one of the four mechanisms above (powerless OOS, borrowed baseline, single arbitrary cell, capacity-proxy-as-hard-cap). Branches that died on robust grounds — non-overlap math killing overlap artifacts, large-magnitude structural adverse selection, PM ≈ empirical efficiency, removal of a mark-to-mid inflation — **stay closed**; reopening those is motivated reasoning. Distinct and usually more valuable than re-auditing dead strats: **never-run branches** — cheap gates we scoped and skipped, e.g. a passive/reversion framing of a signal whose *continuation* framing was correctly falsified (the dali 73.7% TOB signal). When asked "what else can we check," prefer never-run cheap gates over reheating robust closures. Reopen on real evidence; hold the line otherwise.

---

## On startup — checklist

1. Read this file (`brain/CODEX.md`)
2. Read `brain/TODO.md` — check active thread, open tasks, blockers
3. Read `brain/POLYMARKET_BRAIN.md` for Polymarket work, then the relevant hub (MM, OD, copytrade, dali, or STRATEGY_REFERENCE as appropriate)
4. For data-heavy work, read the relevant data/artifact manifest before scanning raw folders
5. Only then begin implementation

When you produce output (findings, scripts, results), save to the right location and add wikilinks back to the hub.
