---
title: "Relay retired — git branch-per-person collaboration model installed"
created: 2026-06-10
status: archived
owner: justin
project: infra
para: archive
hubs:
  - VAULT_MAP
  - TODO
tags:
  - brain
  - infra
  - collaboration
  - handoff
---

# 2026-06-10 — Relay retired; branch-per-person git model installed

Hub links: [[VAULT_MAP]] | [[MERGE_PROTOCOL]] | [[ONBOARDING]] | [[TODO]]

## Summary

- On 2026-06-10 we **abandoned Obsidian Relay** (the live Markdown sync plugin) as the collaboration layer. The coworker's Relay sync was broken, and live CRDT sync added a lock mechanism (`tools/brain_edit_guard.py`) and per-machine config that git makes unnecessary.
- The replacement is a **git branch-per-person model**: each collaborator works on a personal branch named after their GitHub handle; `main` is the shared integration branch; merges into main are deliberate and agent-assisted. The full procedure, including the smart-merge-agent prompt, is [[MERGE_PROTOCOL]]; the collaborator-facing model is [[ONBOARDING]] § Collaboration model.
- The Relay-era **cooperative edit-lock mechanism is retired**: `tools/brain_edit_guard.py` moved to `archive/`, `brain/agents/locks/` reduced to a tombstone, and all "acquire an edit lock" instructions stripped from live docs. Branch isolation makes per-file locks obsolete.
- In the same pass, the law files ([[CODEX]], [[COWORK]]) were made **timeless and person-agnostic**: dated per-thread status prose was removed (preserved below) and replaced with pointers to [[TODO]] and [[VAULT_MAP]] § Active research branches. The Agent Bootstrap was deduplicated to one canonical copy in [[VAULT_MAP]], and the where-to-write table was consolidated into [[VAULT_MAP]].

## What changed where

| Surface | Change |
|---|---|
| [[ONBOARDING]] | § "Sync model: Relay + Git" replaced by § "Collaboration model (git branches)" + § "Onboarding a new collaborator" |
| [[MERGE_PROTOCOL]] | NEW — branch model, merge-to-main procedure, smart-merge-agent prompt |
| [[VAULT_MAP]] | canonical Agent Bootstrap (now with the personal-branch rule), canonical where-to-write table, lock instructions removed |
| [[CODEX]] / [[COWORK]] | dated status sections replaced with pointers; bootstrap/where-to-write reduced to pointers |
| [[SKILL_MAP]], [[OPERATING_RHYTHMS]], [[START_RESEARCH_IDEA]], [[OBSIDIAN_INFRA_ROADMAP]], lane docs, templates | Relay + lock instructions removed; branch model referenced |
| `tools/brain_edit_guard.py` | moved to `archive/brain_edit_guard.py` (DEPRECATED header added); lock registry reduced to the [[agent_edit_locks]] tombstone |
| `.gitignore` | dropped `.obsidian/plugins/system3-relay/` and `brain/agents/locks/*.lock.md`; `/local_agents/` and `/scratch/` stay ignored (per-person isolation) |
| `brain-commit-push` scheduled task | pushes to the operator's personal branch only; refuses to run on main; no pull/merge of main |

Historical handoffs that describe the Relay setup (e.g. [[2026-06-08_per_person_agent_overlays]]) are dated snapshots and were deliberately left as history.

---

## Appendix — snapshot prose moved out of the law files (verbatim, for the record)

The sections below were removed from [[CODEX]] and [[COWORK]] so those files carry no dated status. Thread status is authoritative in [[TODO]] and [[VAULT_MAP]] § Active research branches. Nothing below should be treated as current.

### From [[CODEX]] § "Active threads (as of 2026-06-01)"

**MM — Market-Making** — Hub: `polymarket/research/notes/market_making/strat_market_making.md`. Historical Block K maker work is split out. Single-venue Polymarket maker is CLOSED; live value is real-maker moat diagnosis plus copy/learn-the-winners. Open tasks: decompose why top-3 makers dominate (K5 wallet data) → capacity/moat diagnosis. Paper-trade only after an OOS-cleared design exists.

**OD — Options-Delta** — Hub: `polymarket/research/notes/options_delta/strat_options_delta.md`. OD Strategy A v2 lifecycle failed the primary OOS far-|z| gate under global time embargo. Phase 2 hedge frontier ran anyway as a diagnostic and did not rescue the unhedged gate; Kronos/HAR forward-vol bake-off stays gated off unless the unhedged lifecycle gate/capital assumption is explicitly reopened. Open tasks: none on OD execution by default; keep Kronos/Hermes forward-vol work gated until an explicitly reopened OOS design clears.

**copytrade (parallel thread)** — Midas bot, per-leader audit, path to first real-money smoke. Hub: `brain/TODO.md` § copytrade + `polymarket/research/notes/` (relayer, Domah audit files).

**dali / Polymarket research lineage** — Not globally closed. The original direct local microstructure continuation branch had multiple negative/falsifying tests, but dali is the broader Polymarket research lineage and redesign trail that fed Block K/MM/OD. Treat individual branches as falsified when the notes say so; do not label the whole line closed unless Justin explicitly says it.

### From [[COWORK]] § "Active threads" (the dated status column, 2026-06-06 vintage)

- **copytrade**: cohort-based copy trading on Polymarket; per-leader audit; first real-money smoke pending.
- **MM**: single-venue maker **closed**; real-maker playbook validated (K5) but de-biased deployable EV is **sub-scale** ([[mm_deployable_cells_findings]]) → MM reframed as the **execution layer for OD**.
- **OD**: digital-option valuation/signal layer; standalone OD is closed, and the 2026-06-06 audit says RV-model fair is physical-probability fair, not option-IV fair. Surviving use is cautious sizing/selection inside source-clean passive MM/carry.
- **dali**: older short-horizon OFI/TFI work and redesign trail that fed Block K/MM/OD; individual branches can be falsified, but the line is not globally closed.

### From [[COWORK]] § "Strategic state (snapshot, 2026-05-28 post-A14h + A1.7)"

The original direct local microstructure continuation branch: all tiers tested, all execution modes tested, all negative. The 4-quadrant decision matrix landed in the bottom-right (both A14h and A1.7 negative).

| tier / mode | result |
|---|---|
| Tier 1 taker (8 angles) | Dead under non-overlap math |
| Tier 1 maker (A14h) | Dead. A14c's +554 bps was overlap artifact (-451 bps non-overlap, fill rate 9.0% → 0.2%) |
| Tier 2 LightGBM (A1.7) | No edge. Calibration breaks at P≥0.70 (-16pp gap) — model is LESS accurate at high confidence |
| Tier 3 LSTM/DeepLOB | Foreclosed by calibration diagnosis — gap is structural, not architecture |

The TOB/OFI signal is REAL but it's structural mean-reversion to fair value on Polymarket's wide-spread universe. A15b + A1.7 calibration converge on the same diagnosis. No additional capture or model complexity will change this. Redesigned forward from dali: sign convention infrastructure, capture+replay+analysis pipeline, TOB as state variable, Polymarket microstructure facts (exchange-internal-leg, depth != flow, calibration extremes), methodology lessons (non-overlap math by default, Briola caveat is real), and the Block K/MM/OD redesign path. Pivot to copytrade primary; candidate directions ranked: (1) copytrade scaling + smoke deployment, (2) Block I cross-platform arb, (3) Block J resolution-criteria LLM scanning.

(This snapshot also lives, in fuller form, in [[TODO]] § dali "Strategic state" and [[2026-05-28_dali_falsification_archive]].)

### From [[COWORK]] § "Immediate next decisions (snapshot, 2026-05-28)"

1. Begin copytrade smoke deployment path: PLAN.md sync → Polymarket creds → pre-flight smoke target → first $10 real-money fill (runbook [[polymarket/execution/scripts/SMOKE_REAL|SMOKE_REAL]]).
2. Launch A0c 24h capture as scheduled — data goes into the dali lineage; no further A1.x direct-continuation analysis budget unless explicitly reopened.
3. Document dali redesign state as a research-note artifact.
4. Defer old direct-continuation follow-ups (A14e queue+latency, Block I, Block J) unless copytrade hits a clear ceiling.
