---
title: "Polymarket Data Manifest"
created: 2026-06-05
status: active
owner: justin
project: polymarket
para: resource
hubs:
  - POLYMARKET_BRAIN
  - COWORK
tags:
  - research
  - data-quality
---
# Polymarket Data Manifest

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- This note documents the large Polymarket data artifacts that should be understood as datasets rather than standalone notes.
- Parquet, DuckDB, and raw compressed files are documented at the family level so the graph stays useful; findings-support CSVs and operational JSONL checkpoints are wikilinked directly.
- Zip archives are intentionally ignored here: treat them as raw payload containers or transport artifacts, not as Obsidian knowledge nodes.
- For schemas and formulas, use [[METRICS_REFERENCE]]; for CSV layout rules, use [[polymarket_csv_output_audit]]; for disk-pressure decisions, use [[storage_consolidation_audit_2026_06_05]].

## Data Handling Rule

Do not try to link every Parquet shard. Link findings notes, CSV result tables, and dataset-family manifests. Parquet and JSONL raw shards are usually too numerous and too large to be useful as individual graph nodes.

For storage cleanup, do not delete or rewrite data files ad hoc. Use [[storage_consolidation_audit_2026_06_05]] to separate safe temp/Git cleanup from real dataset redesign.

Current storage state as of 2026-06-05: stale DuckDB spill and Git temp-pack garbage are gone; most useful SNAPPY Parquets that got smaller under ZSTD were rewritten in place; `closed_positions.parquet` dictionary rewrite was sampled and rejected because it grew the file. Do not redo those passes unless new data is generated or the storage design changes.

## Core Parquet Families

| family | important paths | meaning | main readers |
|---|---|---|---|
| raw fill seed | [[polymarket/research/data/raw/orderFilled_complete.csv.xz|orderFilled_complete.csv.xz]], [[polymarket/research/data/trades/trades_seed.parquet|trades_seed.parquet]] | Warproxxx bulk historical `OrderFilled` seed converted to Parquet. | `raw_trades` view, [[validation_report]], [[RESEARCH_FINDINGS]] |
| Goldsky delta shards | `polymarket/research/data/trades/trades_delta_shard*.parquet` | Append-only trade shards after the seed window. Keep as shards; never edit in place. | `raw_trades` view, copytrade, MM, OD, dali |
| Gamma market snapshots | [[polymarket/research/data/markets/markets_2026-05-06.parquet|markets_2026-05-06.parquet]] | Event/market metadata snapshot used for joins and market families. | `markets_tokens` view, [[METRICS_REFERENCE]], dali universe screens |
| closed positions | [[polymarket/research/data/closed_positions.parquet|closed_positions.parquet]] | Resolved `(address, market, outcome)` positions with synthetic redemption. | [[RESEARCH_FINDINGS]], [[block_e_audit]], trader profiles, copytrade |
| trader panel | [[polymarket/research/data/traders.parquet|traders.parquet]] | Per-wallet metrics, style, bankroll, drawdown, and operator flag. | cohorts, profiles, [[api_reconciliation_v1]], copytrade |
| bankroll panel | [[polymarket/research/data/bankroll_timeseries.parquet|bankroll_timeseries.parquet]] | Date-address bankroll reconstruction used for capital-aware analysis. | copytrade sizing, bankroll diagnostics |
| cohorts | `polymarket/research/data/cohorts/*.parquet` | Six stratified trader pools from `traders_filtered`. | [[RESEARCH_FINDINGS]], cohort notebooks |
| directionality | [[polymarket/research/data/directionality_classification/traders_directionality.parquet|traders_directionality.parquet]] | Directional/arb-like wallet style sidecar. | [[block_e_audit]], [[mm_politics_negrisk_live_loop_design]] |
| copyability | [[polymarket/research/data/copyability_candidates/traders_copyability_metrics.parquet|traders_copyability_metrics.parquet]] | Copyability prefilter metrics and deployable-cell counts. | [[block_e_audit]], copytrade candidate screens |
| live CLOB captures | `polymarket/research/data/live_clob/**/*.jsonl` | Raw captured public PM CLOB book/trade events for A0/A0b/A0c/dali replay and MM Stage-1 measurement. Public L2 is anonymous; see [[mm_clob_capture_semantics]] before using it for trade-vs-cancel attribution. | [[block_a0_runbook]], [[mm_clob_capture_semantics]], A1/A11/A12/A13/A14+ replay notes |
| analysis feature panels | [[polymarket/research/data/analysis/block_a1_features.parquet|block_a1_features.parquet]], [[polymarket/research/data/analysis/block_a0c_features.parquet|block_a0c_features.parquet]], [[polymarket/research/data/analysis/block_a0c_roll_features.parquet|block_a0c_roll_features.parquet]], [[polymarket/research/data/analysis/block_a12_mlofi_features.parquet|block_a12_mlofi_features.parquet]], [[polymarket/research/data/analysis/block_a15_features.parquet|block_a15_features.parquet]], [[polymarket/research/data/analysis/block_a17_lightgbm_features.parquet|block_a17_lightgbm_features.parquet]] | Derived replay/feature panels for dali and Block K experiments. | dali A/P notes, MM, OD |
| external market data | `polymarket/research/data/external/**/*.parquet` | Binance/Deribit/external history used by OD and hybrid tests. | OD notes, [[2026-06-02_binance_momentum_polymarket_hybrid]] |
| backtest runs | `polymarket/research/data/backtests/**/*.parquet` | Generated replay/backtest outputs and old paper journals. | dali backtest scripts and paper-trading notes |

## DuckDB And Compressed Files

- [[polymarket/research/data/analysis/_copytrade_relayer_implications.duckdb|_copytrade_relayer_implications.duckdb]] - persistent scratch/work database for the relayer implications probe. Keep documented, but do not treat it as a source of truth over Parquet.
- [[polymarket/research/data/raw/orderFilled_complete.csv.xz|orderFilled_complete.csv.xz]] - compressed raw seed input; the usable query surface is `trades_seed.parquet` plus the delta shards.

## JSONL Support Files

These JSONL files support capture-status, live-loop, or execution findings. Raw hourly capture shards are documented by family above; the files below are the specific support/checkpoint artifacts worth graph-linking.

- [[polymarket/research/data/live_clob/block_a0/block_a0_20260528_morning/capture_gaps.jsonl|block_a0_20260528_morning capture gaps]]
- [[polymarket/research/data/live_clob/block_a0/block_a0_smoke_20260527/capture_gaps.jsonl|block_a0_smoke_20260527 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0b/block_a0b_fg_debug/capture_gaps.jsonl|block_a0b_fg_debug capture gaps]]
- [[polymarket/research/data/live_clob/block_a0b/block_a0b_nohup_debug/capture_gaps.jsonl|block_a0b_nohup_debug capture gaps]]
- [[polymarket/research/data/live_clob/block_a0b/block_a0b_replacements_20260527/capture_gaps.jsonl|block_a0b_replacements_20260527 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0b/block_a0b_replacements_v2_20260527/capture_gaps.jsonl|block_a0b_replacements_v2_20260527 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0b/block_a0b_smoke_check/capture_gaps.jsonl|block_a0b_smoke_check capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c/block_a0c_smoke_20260529/capture_gaps.jsonl|block_a0c_smoke_20260529 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c/block_a0c_targeted_20260529_morning/capture_gaps.jsonl|block_a0c_targeted_20260529_morning capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_000_20260529T090916Z/capture_gaps.jsonl|a0c crypto roll chunk 000 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_001_20260529T100921Z/capture_gaps.jsonl|a0c crypto roll chunk 001 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_002_20260529T110930Z/capture_gaps.jsonl|a0c crypto roll chunk 002 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_003_20260529T120937Z/capture_gaps.jsonl|a0c crypto roll chunk 003 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_004_20260529T130943Z/capture_gaps.jsonl|a0c crypto roll chunk 004 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_005_20260529T140948Z/capture_gaps.jsonl|a0c crypto roll chunk 005 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_006_20260529T150953Z/capture_gaps.jsonl|a0c crypto roll chunk 006 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_007_20260529T160959Z/capture_gaps.jsonl|a0c crypto roll chunk 007 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_008_20260529T171005Z/capture_gaps.jsonl|a0c crypto roll chunk 008 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_009_20260529T181012Z/capture_gaps.jsonl|a0c crypto roll chunk 009 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_010_20260529T191025Z/capture_gaps.jsonl|a0c crypto roll chunk 010 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_011_20260529T201035Z/capture_gaps.jsonl|a0c crypto roll chunk 011 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_012_20260529T211044Z/capture_gaps.jsonl|a0c crypto roll chunk 012 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_013_20260529T221053Z/capture_gaps.jsonl|a0c crypto roll chunk 013 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_014_20260529T231102Z/capture_gaps.jsonl|a0c crypto roll chunk 014 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_015_20260530T001108Z/capture_gaps.jsonl|a0c crypto roll chunk 015 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_016_20260530T011117Z/capture_gaps.jsonl|a0c crypto roll chunk 016 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_017_20260530T021122Z/capture_gaps.jsonl|a0c crypto roll chunk 017 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_018_20260530T031128Z/capture_gaps.jsonl|a0c crypto roll chunk 018 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_019_20260530T041133Z/capture_gaps.jsonl|a0c crypto roll chunk 019 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_020_20260530T051137Z/capture_gaps.jsonl|a0c crypto roll chunk 020 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_021_20260530T061143Z/capture_gaps.jsonl|a0c crypto roll chunk 021 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_022_20260530T071149Z/capture_gaps.jsonl|a0c crypto roll chunk 022 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_20260529_morning/block_a0c_crypto_roll_20260529_morning_chunk_023_20260530T081154Z/capture_gaps.jsonl|a0c crypto roll chunk 023 capture gaps]]
- [[polymarket/research/data/live_clob/block_a0c_crypto_roll/block_a0c_crypto_roll_smoke_20260529/block_a0c_crypto_roll_smoke_20260529_chunk_000_20260529T090841Z/capture_gaps.jsonl|a0c crypto roll smoke capture gaps]]
- [[polymarket/research/data/analysis/mm_politics_negrisk_activity_checkpoint.jsonl|MM politics NegRisk activity checkpoint]]
- [[polymarket/research/data/analysis/mm_politics_negrisk_receipt_checkpoint.jsonl|MM politics NegRisk receipt checkpoint]]
- [[polymarket/research/data/markets/gamma_token_lookup_cache.jsonl|Gamma token lookup cache]]

## CSV Findings Support Index

These are the generated CSVs that back findings notes. Keep new result tables under `data/analysis/csv_outputs/<cluster>/` per [[polymarket_csv_output_audit]].

### Auxiliary CSVs

- [[polymarket/research/data/backtests/paper_journals/dali_clob_ai_product_20260523T160956Z_paper_journal_20260523T165128Z.csv|dali CLOB AI product paper journal 20260523T165128Z]]
- [[polymarket/research/data/backtests/paper_journals/dali_clob_ai_product_20260523T160956Z_paper_journal_20260523T165230Z.csv|dali CLOB AI product paper journal 20260523T165230Z]]
- [[polymarket/research/data/external/spx_iv_last_swing/cme_es_settlements_probe.csv|CME ES settlements probe]]
- [[polymarket/research/data/external/spx_iv_last_swing/vix9d_history.csv|VIX9D history]]
- [[polymarket/research/data/external/spx_iv_last_swing/vix_history.csv|VIX history]]

### Copytrade

- [[polymarket/research/data/analysis/csv_outputs/copytrade/copytrade_bias_systematicity.csv|copytrade bias systematicity]]
- [[polymarket/research/data/analysis/csv_outputs/copytrade/copytrade_cohort_bias_table.csv|copytrade cohort bias table]]

### Dali

- [[polymarket/research/data/analysis/csv_outputs/dali/a0c_holdout_retest_surface.csv|A0c holdout retest surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a11_ofi_component_sweep.csv|A11 OFI component sweep]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a11_segment_surface.csv|A11 segment surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a12_mlofi_comparison.csv|A12 MLOFI comparison]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a12_mlofi_decile_aggregate.csv|A12 MLOFI decile aggregate]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a12_mlofi_market_panel.csv|A12 MLOFI market panel]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_a11_reconciliation.csv|A13/A11 reconciliation]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_conditional_tfi.csv|A13 TOB conditional TFI]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_control_buckets.csv|A13 TOB control buckets]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_decile_aggregate.csv|A13 TOB decile aggregate]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_ofi_joint_signal.csv|A13 TOB/OFI joint signal]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_persistence_by_market.csv|A13 TOB persistence by market]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a13_tob_persistence_runs.csv|A13 TOB persistence runs]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14_executable_taker_results.csv|A14 executable taker results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14b_refined_exit_results.csv|A14b refined exit results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14c_maker_at_mid_results.csv|A14c maker at mid results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14d_tight_spread_results.csv|A14d tight spread results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14f_combined_results.csv|A14f combined results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14g_exit_family_results.csv|A14g exit family results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14h_maker_non_overlap_results.csv|A14h maker non-overlap results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a14i_pyramiding_results.csv|A14i pyramiding results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15_baseline_check.csv|A15 baseline check]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15_imbalance_variants_decile.csv|A15 imbalance variants decile]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15_microprice_signal_decile.csv|A15 microprice signal decile]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15_microprice_target_comparison.csv|A15 microprice target comparison]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15b_baseline_check.csv|A15b baseline check]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a15b_decoupled_results.csv|A15b decoupled results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a16_binary_bet_results.csv|A16 binary bet results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a17_feature_importance.csv|A17 feature importance]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a17_lightgbm_results.csv|A17 LightGBM results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a18_passive_reversion_executed.csv|A18 passive reversion executed]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a18_passive_reversion_market_clusters.csv|A18 passive reversion market clusters]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a18_passive_reversion_surface.csv|A18 passive reversion surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a1_decile_aggregate.csv|A1 decile aggregate]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a1_horizon_surface.csv|A1 horizon surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a1_results.csv|A1 results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_a_market_competition_snapshot.csv|Block A market competition snapshot]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_i_leadlag_alignment.csv|Block I lead-lag alignment]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_i_leadlag_executable_market.csv|Block I lead-lag executable market]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_i_leadlag_executable_summary.csv|Block I lead-lag executable summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_i_leadlag_selected_trades.csv|Block I lead-lag selected trades]]
- [[polymarket/research/data/analysis/csv_outputs/dali/block_i_leadlag_signal_summary.csv|Block I lead-lag signal summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_closed_backtest_market_screen.csv|dali closed backtest market screen]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_family_screen_summary.csv|dali family screen summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_family_shortlist_examples.csv|dali family shortlist examples]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_forward_candidate_audit.csv|dali forward candidate audit]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_forward_candidate_competition_audit.csv|dali forward candidate competition audit]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_forward_viable_market_screen.csv|dali forward viable market screen]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_gamma_current_candidate_markets.csv|dali Gamma current candidate markets]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_gamma_current_future_candidate_markets.csv|dali Gamma current future candidate markets]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_open_recent_market_screen.csv|dali open recent market screen]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_ai_product_100_candidates.csv|dali TFI AI product 100 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_ai_product_100_summary.csv|dali TFI AI product 100 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_candidates.csv|dali TFI crypto 250 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_exlast300_candidates.csv|dali TFI crypto 250 exlast300 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_exlast300_summary.csv|dali TFI crypto 250 exlast300 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_exlast600_candidates.csv|dali TFI crypto 250 exlast600 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_exlast600_summary.csv|dali TFI crypto 250 exlast600 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_crypto_250_summary.csv|dali TFI crypto 250 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_deep_dive_summary.csv|dali TFI deep dive summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_equity_index_100_candidates.csv|dali TFI equity index 100 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_equity_index_100_summary.csv|dali TFI equity index 100 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_magnitude_summary.csv|dali TFI magnitude summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_operator_category_attribution.csv|dali TFI operator category attribution]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_operator_filter_comparison.csv|dali TFI operator filter comparison]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_per_market_diagnostics.csv|dali TFI per-market diagnostics]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_per_market_heterogeneity_summary.csv|dali TFI per-market heterogeneity summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_per_market_top_decile.csv|dali TFI per-market top decile]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_smoke_crypto_25_candidates.csv|dali TFI smoke crypto 25 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_smoke_crypto_25_summary.csv|dali TFI smoke crypto 25 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_sports_100_candidates.csv|dali TFI sports 100 candidates]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_sports_100_summary.csv|dali TFI sports 100 summary]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_sports_explicit.csv|dali TFI sports explicit]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_sports_league_breakdown.csv|dali TFI sports league breakdown]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_volume_interaction.csv|dali TFI volume interaction]]
- [[polymarket/research/data/analysis/csv_outputs/dali/dali_tfi_walk_forward.csv|dali TFI walk-forward]]
- [[polymarket/research/data/analysis/csv_outputs/dali/historical_sign_audit_results.csv|historical sign audit results]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p1_rollingrank_by_market.csv|P1 rolling-rank by market]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p1_rollingrank_row_count_heatmap.csv|P1 rolling-rank row-count heatmap]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p1_rollingrank_surface.csv|P1 rolling-rank surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p2_reversion_passive_fillfrontier.csv|P2 reversion passive fill frontier]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p2_reversion_surface.csv|P2 reversion surface]]
- [[polymarket/research/data/analysis/csv_outputs/dali/p3prime_oos_replication.csv|P3 prime OOS replication]]
- [[polymarket/research/data/analysis/csv_outputs/dali/tfi_exclusion_sweep_ai_product.csv|TFI exclusion sweep AI product]]
- [[polymarket/research/data/analysis/csv_outputs/dali/tfi_exclusion_sweep_crypto.csv|TFI exclusion sweep crypto]]
- [[polymarket/research/data/analysis/csv_outputs/dali/tfi_exclusion_sweep_equity_index.csv|TFI exclusion sweep equity index]]
- [[polymarket/research/data/analysis/csv_outputs/dali/tfi_exclusion_sweep_sports.csv|TFI exclusion sweep sports]]

### Market-Making

- [[polymarket/research/data/analysis/csv_outputs/market_making/k1_maker_economics.csv|K1 maker economics]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k2_quoting_sim.csv|K2 quoting sim]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k2v2_defensive_maker.csv|K2v2 defensive maker]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k2v3_digital_maker.csv|K2v3 digital maker]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k5_real_maker_pnl.csv|K5 real maker PnL]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k5_stress.csv|K5 stress]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k5_stress_gate_rows_for_realism_reaudit.csv|K5 stress gate rows for realism reaudit]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/k5b_dominance_decomp.csv|K5b dominance decomposition]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_chase_optimization.csv|K-PEG chase optimization]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_maker_exit.csv|K-PEG maker exit]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_portfolio_timeseries.csv|K-PEG portfolio timeseries]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_robustness.csv|K-PEG robustness]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_robustness_phase.csv|K-PEG robustness phase]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_robustness_review_metrics.csv|K-PEG robustness review metrics]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/kpeg_selected_fills.csv|K-PEG selected fills]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_deployable_cells.csv|MM deployable cells]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_deployable_cells_capacity_sensitivity.csv|MM deployable cells capacity sensitivity]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_adverse_selection_audit.csv|MM equities up/down adverse-selection audit]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_coverage.csv|MM equities up/down coverage]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_gate.csv|MM equities up/down gate]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_equities_updown_structural_scope_markets.csv|MM equities up/down markets]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_lob_coverage_proxy_calibration.csv|MM LOB coverage proxy calibration]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_accounting_summary.csv|MM politics NegRisk accounting summary]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_capacity_ladder.csv|MM politics NegRisk capacity ladder]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_event_audit.csv|MM politics NegRisk event audit]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_event_flow_2026.csv|MM politics NegRisk persistence event flow 2026]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_flow_by_market_day.csv|MM politics NegRisk persistence flow by market day]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_monthly.csv|MM politics NegRisk persistence monthly]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_pnl_by_close_year.csv|MM politics NegRisk persistence PnL by close year]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_pnl_by_event.csv|MM politics NegRisk persistence PnL by event]]
- [[polymarket/research/data/analysis/csv_outputs/market_making/mm_politics_negrisk_persistence_quarterly.csv|MM politics NegRisk persistence quarterly]]

### Options-Delta

- [[polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_baseline.csv|Binance momentum / Polymarket hybrid baseline]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_metadata.csv|Binance momentum / Polymarket hybrid metadata]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_results.csv|Binance momentum / Polymarket hybrid results]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/binance_momentum_polymarket_hybrid_timing.csv|Binance momentum / Polymarket hybrid timing]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3_leadlag_basis.csv|K3 lead-lag basis]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal.csv|K3v2 lead-lag causal]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v2_leadlag_causal_hedged_ext.csv|K3v2 lead-lag causal hedged ext]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v3h2_persistence_summary.csv|K3v3h2 persistence summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v3h2_persistence_trades.csv|K3v3h2 persistence trades]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades.csv|K3v3h hedged basis trades]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k3v3h_hedged_basis_trades_ext.csv|K3v3h hedged basis trades ext]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k4_combinatorial_arb.csv|K4 combinatorial arb]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k6_gamma_scalp_trades.csv|K6 gamma scalp trades]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k6_strategy_a_static_hedge.csv|K6 Strategy A static hedge]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k6_vol_gap.csv|K6 vol gap]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/k7_longshot_calibration.csv|K7 longshot calibration]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_bins.csv|OD conditional-prob calibration bins]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_conditional_prob_calibration_summary.csv|OD conditional-prob calibration summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_market_detail.csv|OD cross-asset gate0 market detail]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_reference_checks.csv|OD cross-asset gate0 reference checks]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_cross_asset_gate0_universe_map.csv|OD cross-asset gate0 universe map]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_market_detail.csv|OD equities index pricing scope market detail]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_reference_paths.csv|OD equities index pricing scope reference paths]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_index_pricing_scope_template_summary.csv|OD equities index pricing scope template summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_clob_scan.csv|OD equities SPX IV/NZ last-swing CLOB scan]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_cme_es_probe.csv|OD equities SPX IV/NZ last-swing CME ES probe]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_data_audit.csv|OD equities SPX IV/NZ last-swing data audit]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_iv_nz_last_swing_vix_coverage.csv|OD equities SPX IV/NZ last-swing VIX coverage]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_calibration.csv|OD equities SPX NZ pricing calibration]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_data_ledger.csv|OD equities SPX NZ pricing data ledger]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_date_summary.csv|OD equities SPX NZ pricing date summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_residual_sample.csv|OD equities SPX NZ pricing residual sample]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_nz_pricing_summary.csv|OD equities SPX NZ pricing summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_local_clob_scan.csv|OD equities SPX open up/down local CLOB scan]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_market_detail.csv|OD equities SPX open up/down market detail]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_equities_spx_open_updown_scope_summary.csv|OD equities SPX open up/down summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pricing_model_form_deribit.csv|OD pricing model form Deribit]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pricing_model_form_reliability.csv|OD pricing model form reliability]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pricing_model_form_shape.csv|OD pricing model form shape]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pricing_model_form_summary.csv|OD pricing model form summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pure_taker_attribution_spread_regime.csv|OD pure-taker attribution spread regime]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_pure_taker_attribution_summary.csv|OD pure-taker attribution summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_basis.csv|OD RV Deribit daily basis]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_data_gate.csv|OD RV Deribit daily data gate]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_settlement_mismatch.csv|OD RV Deribit daily settlement mismatch]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_rv_deribit_daily_summary.csv|OD RV Deribit daily summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_heldout_cells.csv|OD same-day crypto Arm T confirmation heldout cells]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_confirmation_train_cells.csv|OD same-day crypto Arm T confirmation train cells]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_heldout_cells.csv|OD same-day crypto Arm T tier1 heldout cells]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_kou_params.csv|OD same-day crypto Arm T tier1 Kou params]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_arm_t_tier1_train_cells.csv|OD same-day crypto Arm T tier1 train cells]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_arm_summary.csv|OD same-day crypto pricing arm summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_bucket_summary.csv|OD same-day crypto pricing bucket summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_calibration.csv|OD same-day crypto pricing calibration]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_capacity.csv|OD same-day crypto pricing capacity]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_classification.csv|OD same-day crypto pricing classification]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_current_quotes.csv|OD same-day crypto pricing current quotes]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_same_day_crypto_pricing_history_markets.csv|OD same-day crypto pricing history markets]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_designs.csv|OD Strategy A realism reaudit designs]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_k6_far_family.csv|OD Strategy A realism reaudit K6 far family]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_realism_reaudit_same_day_touch_capacity.csv|OD Strategy A realism reaudit same-day touch capacity]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_adverse_regimes.csv|OD Strategy A tail adverse regimes]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_same_day_touch.csv|OD Strategy A tail same-day touch]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_sample_growth.csv|OD Strategy A tail sample growth]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_sizing_robustness.csv|OD Strategy A tail sizing robustness]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_source_vs_valuation.csv|OD Strategy A tail source versus valuation]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_tail_terminal_daily_boundary.csv|OD Strategy A tail terminal daily boundary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_v2_lifecycle.csv|OD Strategy A v2 lifecycle]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_v3.csv|OD Strategy A v3]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_strategy_a_v3_pnl_risk.csv|OD Strategy A v3 PnL risk]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_touch_risk_filter_data_ledger.csv|OD touch-risk filter data ledger]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_touch_risk_filter_deciles.csv|OD touch-risk filter deciles]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_touch_risk_filter_pm_skip.csv|OD touch-risk filter PM skip]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_touch_risk_filter_separation.csv|OD touch-risk filter separation]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_touch_risk_filter_skip_sweep.csv|OD touch-risk filter skip sweep]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_bins.csv|OD v4 calibration gate bins]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_v4_calibration_gate_summary.csv|OD v4 calibration gate summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/od_v4_queue_replay_summary.csv|OD v4 queue replay summary]]
- [[polymarket/research/data/analysis/csv_outputs/options_delta/optd_ceiling_search.csv|OPTD ceiling search]]
