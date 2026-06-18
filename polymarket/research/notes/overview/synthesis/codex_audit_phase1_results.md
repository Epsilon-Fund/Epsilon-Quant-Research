---
title: Dali Phase 1 Diagnostics Results
created: 2026-05-23
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
tags:
  - dali
  - diagnostics
  - audit
  - phase-1
  - research
---

# Dali Phase 1 Diagnostics Results

> Hub: [[COWORK]]

Generated: 2026-05-23

## Summary

This note records the Dali Phase 1 diagnostic tasks and their outcomes. It covers sign-convention inference, TFI magnitude analysis, maintained book state/OFI work, and follow-up requirements. The main status is diagnostic rather than deployable: longer live captures and stronger normalization evidence were needed before downstream reruns.

## Task 1: Sign Convention Via Inference

What was done:

- Added `lib/trade_sign_normalization.py`.
- Added `scripts/dali_sign_convention_audit.py`.
- Audited both existing live CLOB capture files.
- Wrote `notes/dali/sign_convention_findings.md`.

What was found:

- Live sample has only 1 `last_trade_price` event.
- Classified live trades: 0.
- Unclassifiable live trades: 1.
- Live `last_trade_price.side` normalization is not established.
- Historical `maker_side` semantics are clear: it is the passive maker's token
  side, so token-side aggressor is the inverse.

Deviation from spec:

- No normalized TFI rerun was performed because the live side convention did not
  meet the amended 50-classifiable-trade threshold.

What is needed next:

- Longer live captures with at least 50 classifiable trades.
- Keep using `historical_to_aggressor()` for historical token-side aggressor.
- Keep `live_to_aggressor()` returning `UNKNOWN` by default until evidence is
  sufficient.

## Task 2: Hit-Rate-By-Magnitude Analysis

What was done:

- Added `scripts/dali_tfi_magnitude_analysis.py`.
- Generated `notebooks/tfi_magnitude_analysis.ipynb`.
- Generated 12 PNG charts under `notebooks/figs/`.
- Wrote summary table to `data/analysis/csv_outputs/dali/dali_tfi_magnitude_summary.csv`.

Families covered:

- `daily_crypto_up_down`
- `daily_equity_index`
- `ai_product`
- `sports_game_lines`

What was found:

- The strongest-looking positive averages are often not matched by clean rising
  hit rates.
- AI/product and equity-index inverse-maker-side results remain more
  tail/asymmetry-shaped than clean directional hit-rate-shaped.
- Crypto has stronger 300s maker-side behavior, but short-horizon results are
  mixed and still likely contaminated by resolution/external-price dynamics.
- This supports continuing diagnostics, not jumping to ML.

Deviation from spec:

- Sign-normalized rerun was not performed because Task 1 did not establish live
  normalization.

What is needed next:

- Re-run this notebook after sign normalization is established.
- Sweep last-N-seconds exclusion later, especially on crypto and sports.

## Task 3: Maintained Book State And OFI

What was done:

- Added `lib/clob_book.py` with maintained price-level book state and CKS-style
  OFI helpers.
- Updated `scripts/dali_clob_replay_features.py`.
- Added `tests/test_ofi_calculation.py`.
- Replayed the existing AI/product capture into
  `data/analysis/dali_clob_features_ai_product_20260523T160956Z.parquet`.

What was found:

- Replay still produces 189 per-asset rows from 123 raw messages across 10 token
  IDs.
- New columns include event OFI, rolling OFI windows, OFI z-score, book
  staleness, book-complete flag, telemetry-only best bid/ask, and separated
  latency fields.
- Existing short capture has 18 nonzero OFI-event rows.
- Initial `book` snapshot timestamps remain stale and must not be treated as
  live latency.

Deviation from spec:

- `best_bid_ask` does not mutate executable book state. It is stored only as
  telemetry, per the amendment.

What is needed next:

- Longer captures to make rolling 300s OFI and z-score columns meaningful.
- Compare OFI universality only after 3+ families have enough trade events.

## Task 4: Minimal Executable-Price Paper Engine

What was done:

- Added `lib/backtest_engine.py`.
- Added `scripts/dali_paper_backtest.py`.
- Added `configs/backtest_default.yaml`.
- Added `tests/test_backtest_engine.py`.
- Ran the engine on the existing AI/product capture.

What was found:

- Synthetic tests verify executable-price entry/exit and book walking.
- Existing capture produced 1 closed correctness trade, force-closed at the end.
- Net PnL on that one correctness trade was `-0.005`, which is not strategy
  evidence.

Deviation from spec:

- No parameter search, no Sharpe, and no optimization were run.
- Spread cost is recorded as diagnostic implied spread cost only; it is not
  double-counted because execution uses actual bid/ask book prices.

What is needed next:

- Run on longer captures.
- Add stricter rules only after sign convention and OFI features have enough
  live observations.

## Task 5: Parameter Search Deferred

What was done:

- Added `notes/overview/data_quality/task5_trigger_conditions.md`.

What was found:

- Current data is far below the search threshold.

Trigger conditions:

- 3+ captured families.
- 24h+ capture per family.
- 200+ combined `last_trade_price` events.
- 50+ classifiable trades for sign convention, unless the tested rule avoids
  trade-side normalization.

What is needed next:

- Keep gathering live data; do not implement Optuna/search yet.

## Block B: Historical Fill-Only TFI Deep Dive

Generated: 2026-05-27

What was done:

- Added `scripts/dali_tfi_deep_dive.py`.
- Reused cached family eval/fills outputs instead of repeatedly scanning the
  full historical trade shards.
- Generated the revised Block B notebooks:
  - `notebooks/block_b_resolution_sweep.ipynb`
  - `notebooks/block_b_per_market_heterogeneity.ipynb`
  - `notebooks/block_b_sports_analysis.ipynb`
  - `notebooks/block_b_volume_interaction.ipynb`
- Wrote consolidated and component CSVs under `data/analysis/`, including
  `dali_tfi_deep_dive_summary.csv` and `tfi_exclusion_sweep_<family>.csv`.
- Wrote synthesis to `notes/copytrade/block_b_findings.md`.

What was found:

- Decision matrix outcome: **OUTCOME 3 — Mixed Results Requiring Live
  Validation**.
- The decision uses `inverse_maker_side`, because historical `maker_side` is
  passive maker token side and `inverse_maker_side` is the confirmed historical
  aggressor convention.
- Unfiltered historical-aggressor TFI does not pass the full Outcome 1 screen:
  no clean combination of exclusion/volume/magnitude has monotone hit-rate
  improvement, top-decile hit rate above 55%, CI excluding 50%, and positive
  rough net EV.
- AI/product shows positive average EV in several inverse-maker-side buckets,
  but hit rates are near 50% and remain tail/asymmetry-shaped.
- Sports is now explicit: league-level inverse-maker-side top-decile hit rates
  do not clear 55%; local metadata has no game-start field, so pre-game vs
  in-game segmentation remains a TODO rather than an inferred split.
- Operator-filtered historical rows are interesting, especially equity-index
  inverse-maker-side after removing denylisted operator addresses, but this is
  only a Block A target-list clue because live CLOB trade messages may not
  expose the same address-level filter.

Deviation from spec / caveats:

- Exclusion sweep uses the union of the revised prompt windows
  `{300, 600, 1800, 3600, 7200}` and the handoff continuity windows
  `{0, 60, 120, 1200}`.
- Costs are conservative tick proxies only; historical fills still lack L2
  spread/depth and queue context.
- Passive-maker-side positive crypto rows remain in the outputs for diagnostics
  but are not used to classify historical aggressor TFI as salvageable.

What is needed next:

- Use Block B as a shortlist input for Block A live OFI capture, not as a
  tradability conclusion.
- Prioritize live validation around the mixed/pocket signals: AI/product
  high-volume buckets and equity-index behavior after operator-style flow is
  removed historically.
