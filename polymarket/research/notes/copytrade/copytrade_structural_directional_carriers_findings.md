---
title: "Copytrade Structural Directional Carriers Findings"
tags: [copytrade, market-making, directionality, structural-carriers]
created: 2026-06-04
status: not-copyable-as-taker-direction
---

# Copytrade Structural Directional Carriers Findings

> Hub: [[COWORK]]
> Source MM diagnostic: [[mm_structural_maker_directional_decomposition_findings]]

## Plain-English Summary

The non-politics structural-maker edge did not become a clean copytrade lead when reframed as "copy the direction, not the maker wrapper." The historical directional-carrier wallets are real, and many pass the simple copyability screen, but the pure taker-copy replay does not clear net-of-cost in sports, residual misc, or equities up/down.

The best read is strict: these wallets are not clean neutral MM sleeves, and they are also not validated directional copy targets under a taker-copy execution model. The value appears bound up with maker execution, queue/fill selection, and wallet-specific behavior rather than a direction that can be safely copied after observing the leader fill.

## Inputs

This pass used only existing local artifacts:

- `data/analysis/k5_stress_wallet_market_full.parquet`
- `data/directionality_classification/traders_directionality.parquet`
- `data/copyability_candidates/traders_copyability_metrics.parquet`
- `data/closed_positions.parquet`
- local historical trade shards under `data/trades/`
- [[mm_structural_maker_directional_decomposition_findings]]

No new capture or live book data was used.

The carrier universe is the `two_sided_directional` structured non-top3 wallet-market set from the MM directional decomposition lane. Sports-like was reconstructed with the strict retag/structure logic from the previous MM batch and matches the saved sports row. The residual-misc wallet-level retag artifact was not saved in the MM note, so the recreated residual carrier table should be treated as a diagnostic close cousin, not a canonical restatement of that note.

## Copyability Screen

Screen used the requested rule:

- Keep `two_sided_directional` structured non-top3 carrier wallets by sleeve.
- Exclude `split_position_signature > 60%`.
- Rank by `active_days_last_90d x hold_to_resolution_share x (1 + n_deployable_cells)`.
- `n_deployable_cells` is only populated for already audited leaders, so most candidates remain `unrun` and rank mainly on recency and hold-to-resolution.

| sleeve | carrier wallets | screen-pass wallets | historical carrier gross | historical carrier bps | read |
|---|---:|---:|---:|---:|---|
| sports_like | 247 | 198 | $250.9M | +240.8 bps | primary diagnostic sleeve |
| residual_misc | 297 | 204 | $391.1M | +101.7 bps | diagnostic; recreated residual split differs slightly from saved MM note |
| equities_updown_close_spx_ndx | 19 | 19 | $57.2k | +154.6 bps | too thin |

## Taker-Copy Replay

The execution proxy reuses the Domah audit shape:

- For each leader fill, copy the entered direction on the same outcome token.
- Pure taker branch `B`: fill at the next same-side print within 300 seconds; if absent, use a 3c worse fallback.
- Pure maker counterfactual `C_real`: quote at the leader print price and require at least two qualifying follow-on prints within 300 seconds.
- `C_opt`: same but requires at least one qualifying follow-on print.
- Carry copied inventory to resolution only; positions without `resolution_price` are excluded.
- Apply sleeve-level empirical taker-fee / maker-rebate rates from K5 carrier rows.
- CI is market-cluster bootstrap over `market_id`.

This is still a historical execution proxy, not a live order-book replay. It cannot measure queue position, current spread/depth, missed fills, or event-time book dynamics.

## Replay Results

### Sports-Like

The copyability-only top 3 was a useful negative control: it included sports-negative wallets and pure taker-copy lost money. The vetted sports cohort below required screen-pass, positive sports K5 bps, and at least $100k sports sleeve gross.

| branch | wallets | markets | notional | PnL | bps | CI bps |
|---|---:|---:|---:|---:|---:|---:|
| leader replay | 3 | 3,670 | $26.4M | +$484.2k | +183.5 | n/a |
| pure taker `B` | 3 | 3,670 | $26.6M | -$164.4k | -61.8 | [-702.0, +497.0] |
| maker `C_real` | 3 | 3,670 | $16.4M | +$575.2k | +350.2 | [-343.6, +980.6] |
| maker `C_opt` | 3 | 3,670 | $18.5M | +$499.5k | +270.1 | [-340.4, +866.1] |

Verdict: **NOT-COPYABLE** as taker direction. The leader/carry path is positive, but the direction copied as taker does not clear. Maker counterfactual is point-positive but CI-crossing and not a copytrade execution model.

### Residual Misc

Positive-K5 top 3 by copyability score:

| branch | wallets | markets | notional | PnL | bps | CI bps |
|---|---:|---:|---:|---:|---:|---:|
| leader replay | 3 | 11,782 | $26.2M | +$198.3k | +75.6 | n/a |
| pure taker `B` | 3 | 11,782 | $26.5M | -$594.2k | -224.2 | [-495.0, +56.3] |
| maker `C_real` | 3 | 11,782 | $15.3M | +$59.2k | +38.6 | [-276.4, +403.7] |
| maker `C_opt` | 3 | 11,782 | $17.6M | +$105.2k | +59.9 | [-277.1, +390.7] |

Verdict: **NOT-COPYABLE** as taker direction. The taker branch is negative; maker branches do not clear.

### Equities Up/Down

All screen-pass positive-K5 equities carriers:

| branch | wallets | markets | notional | PnL | bps | CI bps |
|---|---:|---:|---:|---:|---:|---:|
| leader replay | 11 | 51 | $48.1k | +$1.6k | +331.4 | n/a |
| pure taker `B` | 11 | 51 | $49.2k | -$0.6k | -114.1 | [-972.4, +445.5] |
| maker `C_real` | 11 | 51 | $22.0k | -$0.8k | -378.6 | [-1,462.6, +709.8] |
| maker `C_opt` | 11 | 51 | $28.0k | -$0.2k | -59.7 | [-1,038.8, +682.9] |

Verdict: **NOT-COPYABLE / THIN**. Too little carrier gross and the taker branch is negative.

## Shortlist Verdict

No sleeve graduates to a vetted directional copy target from this pass.

Research-only candidates remain useful for manual review, but not as copytrade smoke targets:

| sleeve | research candidates | reason not promoted |
|---|---|---|
| sports_like | `0x2d6ac4f70307102ac46e9e6ded67f3838ddf8add`, `0x7ea571c40408f340c1c8fc8eaacebab53c1bde7b`, `0xcf7379b4b891c06d88807f6f70efa75378120215` | positive leader replay, but taker-copy fails |
| residual_misc | `0x4133bcbad1d9c41de776646696f41c34d0a65e70`, `0x68146921df11eab44296dc4e58025ca84741a9e7`, `0x2d6ac4f70307102ac46e9e6ded67f3838ddf8add` | taker-copy negative |
| equities_updown_close_spx_ndx | all positive-K5 carriers | tiny and negative taker-copy |

## Decision

The structural directional carriers should not replace the current copytrade smoke-target selection. Keep the existing copytrade path: audit more leaders with `scripts/domah_copy_audit.py`, prefer cells that clear the audited taker-copy/role-mirrored thresholds, and use `traders_directionality` / `split_position_signature` as screens rather than as a proof of copyability.

The MM implication is also consistent with [[mm_structural_maker_directional_decomposition_findings]]: non-politics sports/residual/equities historical edge should not be promoted into neutral MM or taker-copy production edge. It belongs, at most, in a live measurement loop if a future sleeve first supplies a concrete execution hypothesis.

## Artifacts

- Wallet-market carrier rows: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_wallet_market.parquet`
- Wallet screen: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_wallet_screen.csv`
- Pre-backtest shortlist: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_shortlist_pre_backtest.csv`
- Combined replay summary: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_backtest_summary.csv`
- Sports positive-K5 replay positions: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_sports_like_posk5_top3_positions.parquet`
- Residual positive-K5 replay positions: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_residual_misc_posk5_top3_positions.parquet`
- Equities replay positions: `data/analysis/csv_outputs/copytrade/copytrade_structural_directional_carriers_equities_updown_posk5_all_positions.parquet`
