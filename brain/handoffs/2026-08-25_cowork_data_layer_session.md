---
title: "Cowork session — repo catch-up, L2 pipeline incident, and the data-layer plan"
created: 2026-08-25
status: active
owner: alvaro
project: polymarket
para: area
hubs:
  - COWORK
  - strat_market_making
  - POLYMARKET_BRAIN
tags:
  - market-making
  - data-ingestion
  - handoff
  - planning
  - onboarding
---
# Cowork session — repo catch-up, L2 pipeline incident, and the data-layer plan

> Hub: [[COWORK]] · [[strat_market_making]] · Plan: [[DATA_LAYER_PLAN]] · Recovery task: [[2026-08-25_l2_pipeline_recovery]]

## Plain-English Summary

- **What happened.** Alvaro returned after ~2 months away, merged Justin's 63 commits into `alvaro`, mapped the MM project and the code end to end, discovered the L2 capture pipeline had been silently broken since 2026-08-21, and set the direction for the next phase: stop building infrastructure, make the data we already have easy to research with.
- **The state of MM.** Paused since 2026-07-12, waiting on a decision that belongs to the strategy side, not the implementation side. Nothing has moved in six weeks.
- **The pivot.** We hold 67 days of capture and every published result was computed on 11–18. The highest-value action available is not more data or more infrastructure — it is re-running the existing analysis at full sample, which requires the data to be usable first.
- **What was produced.** Four reference pages, a recovery runbook for Claude Code, and an ordered build plan for the data layer ([[DATA_LAYER_PLAN]]).
- **Next actor.** Alvaro, starting on Phase A of the plan. Then Gonzalo joins on the visualisation layer.

---

## 1 · Repo catch-up

`origin/justin` (main + one brain-sync commit) merged into local `alvaro`. Two conflicts, both in Markdown, both resolved in favour of Justin's newer content: `mm_backtest_research_roadmap.md` (status line + status-update block) and `strat_market_making.md` (cross-links section, with the `[[mm_backtest_research_roadmap]]` wikilink restored where Justin's version had dropped it).

Note for future sessions: git operations on this repo **cannot** be driven from the Cowork device bridge. It cannot delete files, so git cannot clean up its own lock files, and any merge or checkout that must remove a working-tree file fails partway. Read-only inspection is fine; anything that rewrites the working tree has to be run by the operator or by Claude Code.

## 2 · The L2 pipeline incident

The VPS rebooted at **2026-08-21 21:13:15 UTC**, truncating the three gzip shards open at that moment (`esports_21`, `politics_negrisk_21`, `unknown_21` on 2026-08-21). `compression/pipeline.py` raises on an unreadable shard and the exception propagates out of `process_files()`, so **one corrupt file aborted the entire run** — 261 pending shards, hourly, for three days. Cascade: no new Parquet → nothing to sync → R2 frozen at 2026-08-21 → local raw never pruned → disk climbed to 88%.

Capture itself never stopped and remains healthy. The recovery, the durable quarantine-and-continue fix, and the repo-side cleanup are specified in [[2026-08-25_l2_pipeline_recovery]] and were handed to Claude Code.

Other findings from reading the pipeline source:

- **The repo copy of `sync/sync_cloud.sh` is a committed merge conflict** — 33 unresolved markers, nested, referencing `7703de6c` and `4db6a716`. `deploy/DEPLOY.md` and `deploy/R2_HANDOVER.md` likewise. The **deployed** copy on the VPS is clean and parses; the server is therefore the authority for that file.
- **The two missing universes were a decision, not a bug.** `crypto_control` is commented out in `universes.yaml` with a dated note ("DISABLED for now (2026-06-18) … re-enable once politics+esports capture is validated"); `culture_other` was never written into the config at all. Capture was validated in June and nobody went back. The crypto control is the falsification instrument and it is currently switched off.
- Capture daemon memory sits at ~1.1 GB against a 2 GB cap, peak 1.7 GB — a slow leak worth watching.
- 418 assets subscribed against ~694 planned; unexplained, cheap to check.
- `capture_gaps.jsonl` is never synced to R2, so gap ground truth is lost on raw expiry.

## 3 · The archive is bigger than the notes said

R2 `parquet/` holds **11,985 objects / 71.08 GiB**, spanning **2026-06-19 → 2026-08-21**. The June notes recorded 4.6 GB. Storage costs about $0.92/month; egress from R2 is free.

**The observation that reframes the project:** every published MM result — Task 4, Task 5, Task 5.1, the market screen, the fragility audit, the +0.29¢ — was computed on 11 to 18 days. Politics failed to certify at K = 11 event groups and DSR failed on 19 daily observations, and both notes state explicitly that these are power failures rather than verdicts. Four times the window plausibly means three to four times the groups. **Re-running the existing ladder at full sample is the highest-value action available in this project, and nobody has ever done it.**

## 4 · Decisions taken

- **Do not switch off capture yet.** Cost is a few euros a month; the stream is irreplaceable; decide after the catalog says how many complete market lifecycles the window contains.
- **R2 stays the storage** — as a distribution channel, not a query engine. Queries always run against a local copy.
- **No database server.** Parquet on disk plus DuckDB, which is a library rather than a service.
- **The derived dataset is versioned and immutable** — `research/v1/`, then `v2`, never a mutated `v1`. Layer-0 raw is never touched so any derived layer can be rebuilt.
- **One conversion.** Exactly one function in the codebase turns stored files into a `MarketEvent` stream; the engine and research both use it, enforced by a parity test against `replay_parquet`.
- **Convenience columns are for viewing only.** The engine always recomputes from raw levels.
- **The plotting layer takes strategy data as an optional overlay** from the start, so the strategy-debugging views are a data source rather than a second plotting stack.

## 5 · Reference pages produced

Published as Cowork artifacts (private to Alvaro's account unless shared):

- **Field Map** — where each branch of the MM project stands, what is proven vs directional vs dead, the decision that unblocks the live loop, and an ordered reading list: https://claude.ai/code/artifact/97abd556-edf2-4046-b637-d1341d0f875b
- **How the Maker Model Works** — the three models plus the judge, the chain from a captured message to cents per contract, the queue models, the quoters, the costing convention, and what each link in the chain is worth: https://claude.ai/code/artifact/9ad187f7-b8a5-43c9-8e5a-3d9a742b6ce4
- **L2 Data Atlas** — what the capture collects, the machine that runs it, the storage layout, how to get the data, and the known faults: https://claude.ai/code/artifact/99e3ea71-3557-4bb7-bfbb-4e86689b2004
- **Data Layer Plan** — the build plan, also committed here as [[DATA_LAYER_PLAN]]: https://claude.ai/code/artifact/e8b67361-bb27-459a-95e1-01a9f61eb9e9

## 6 · Where this leaves MM

Unchanged and still waiting. The 2026-07-12 handoff names Cowork as next actor for the Step-1 redesign, and that redesign remains unstarted. Two routes exist around the no-short constraint, and only one was scoped:

1. **Paired quoting** — a bid on YES and a bid on NO. This is what the literature assumes ("a bid for one contract is also recorded as an ask for the complementary contract … the problem reduces to liquidity provision in a single contract with short selling allowed", *Optimal Market Making in Prediction Markets*, arXiv 2607.17991). It forces pair-aware inventory, costing and queue accounting through the whole stack.
2. **Complete sets** — Polymarket's CTF splits 1 pUSD into 1 YES + 1 NO, merges the pair back to 1 pUSD, and redeems the winner 1:1 after resolution. A maker can therefore mint genuine two-sided inventory and rest real asks — **and the existing single-token machinery stays valid**, because you would be quoting a token you actually hold.

Route 2 was not considered in the July redesign scoping and is very likely the cheaper path to a first calibration loop. It should be put to Justin.

## 7 · Next actor and next steps

**Alvaro**, on [[DATA_LAYER_PLAN]] Phase A — the coverage manifest and the catalog. Neither is blocked by the pipeline recovery. Then Phases B–D, then Gonzalo onboards onto the strategy-overlay views.

Still open: whether to keep capturing (decide after A3), whether to re-enable the crypto control, where the loader module lives and how it should match the existing crypto `get_data` conventions, and the research-tooling API beyond the viewer.
