---
tags: [dali, block-a14g, executable-cost, results]
---
> Hub: [[COWORK]]


## Summary

- Scope: Block A1.4g Exit-Family Findings in the Dali research lineage area.
- Existing takeaway/status: A1.4g tests longer-horizon TOB-imbalance taker exits with a 300s time-stop backstop across all 11 primary-read markets. 0 of 165 market-config rows cross zero on mean PnL, and 0 have bootstrap lower CI above zero. The best row is `a0b:1971905` / `exit_imbalance_recovery_t0.1` at -125.0 bps with CI [-174.1 bps, -61.6 bps] and mean hold 264.0s. Crypto 4h markets have 0 positive mean rows.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
> Table terms: [[polymarket_table_dictionary]]

# Block A1.4g Exit-Family Findings

## Headline

A1.4g tests longer-horizon TOB-imbalance taker exits with a 300s time-stop backstop across all 11 primary-read markets. 0 of 165 market-config rows cross zero on mean PnL, and 0 have bootstrap lower CI above zero. The best row is `a0b:1971905` / `exit_imbalance_recovery_t0.1` at -125.0 bps with CI [-174.1 bps, -61.6 bps] and mean hold 264.0s. Crypto 4h markets have 0 positive mean rows.

## Method

- Candidate universe: all `primary_read` markets in `block_a1_results.csv`, with `a0b:2364426` explicitly included.
- Signal: per-market top decile by absolute current TOB imbalance level, `tob_imbalance_level = direction_factor * tob_imbalance`.
- Entry: instantaneous taker at touch. Long token signals pay `best_ask`; short token signals receive `best_bid`.
- Backstop: every config exits no later than 300s when sufficient forward book state exists.
- Exit families: strength decay to 25/50/75% of entry magnitude, trailing stops at 30/50/70% retrace, imbalance recovery below 0.1/0.2/0.3, spread widening by 2x/3x, asymmetric TP/SL pairs, and `exit_compound_v1`.
- PnL: executable bid/ask round trip with taker fees on entry and exit using A1's `FEE_BY_CATEGORY`.
- Confidence intervals: 200-sample block bootstrap on 300s contiguous clock-time blocks.

## Exit-Family Ranking

| family | config | markets | positive | mean pnl | mean median | win | mean hold | max hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exit_imbalance_recovery | exit_imbalance_recovery_t0.3 | 11 | 0 | -732.8 bps | -586.5 bps | 2.3% | 178.1s | 300.0s |
| exit_imbalance_recovery | exit_imbalance_recovery_t0.2 | 11 | 0 | -741.5 bps | -595.6 bps | 2.5% | 184.2s | 300.0s |
| exit_strength_decay | exit_strength_decay_p25 | 11 | 0 | -741.7 bps | -598.3 bps | 2.4% | 181.6s | 300.0s |
| exit_strength_decay | exit_strength_decay_p50 | 11 | 0 | -768.9 bps | -613.6 bps | 0.4% | 167.6s | 300.0s |
| exit_strength_decay | exit_strength_decay_p75 | 11 | 0 | -782.2 bps | -628.3 bps | 0.3% | 155.3s | 300.0s |
| exit_imbalance_recovery | exit_imbalance_recovery_t0.1 | 11 | 0 | -801.3 bps | -626.7 bps | 1.5% | 189.9s | 300.0s |
| exit_compound_v1 | exit_compound_v1 | 11 | 0 | -809.8 bps | -627.4 bps | 0.4% | 155.4s | 300.0s |
| exit_asymmetric | exit_asymmetric_tp3000_sl300 | 11 | 0 | -825.1 bps | -694.6 bps | 7.8% | 239.0s | 300.0s |
| exit_trailing_stop | exit_trailing_stop_r70 | 11 | 0 | -840.3 bps | -653.6 bps | 4.5% | 218.3s | 300.0s |
| exit_asymmetric | exit_asymmetric_tp800_sl100 | 11 | 0 | -841.6 bps | -718.7 bps | 1.7% | 204.6s | 300.0s |
| exit_asymmetric | exit_asymmetric_tp1500_sl200 | 11 | 0 | -844.1 bps | -702.8 bps | 5.9% | 221.2s | 300.0s |
| exit_trailing_stop | exit_trailing_stop_r30 | 11 | 0 | -856.5 bps | -640.1 bps | 2.9% | 198.5s | 300.0s |
| exit_trailing_stop | exit_trailing_stop_r50 | 11 | 0 | -867.6 bps | -644.3 bps | 3.2% | 205.1s | 300.0s |
| exit_spread_widening | exit_spread_widening_f2 | 11 | 0 | -955.3 bps | -690.2 bps | 1.1% | 190.9s | 300.0s |
| exit_spread_widening | exit_spread_widening_f3 | 11 | 0 | -1038.8 bps | -718.4 bps | 2.9% | 231.4s | 300.0s |

## Positive Rows

_No rows._

## Per-Market Verdicts

| market | slug | family | best config | mean pnl | CI | win | mean hold | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0:558934 | will-spain-win-the-2026-fifa-world-cup-963 | sports_game_lines | exit_trailing_stop_r30 | -344.5 bps | [-353.3 bps, -337.6 bps] | 0.0% | 300.0s | still negative |
| a0:558936 | will-france-win-the-2026-fifa-world-cup-92 | sports_game_lines | exit_asymmetric_tp1500_sl200 | -336.2 bps | [-339.6 bps, -332.9 bps] | 0.0% | 300.0s | still negative |
| a0:665325 | us-iran-nuclear-deal-before-2027 | geopolitics_policy | exit_imbalance_recovery_t0.3 | -337.0 bps | [-379.6 bps, -297.6 bps] | 0.0% | 208.5s | still negative |
| a0b:1971905 | strait-of-hormuz-traffic-returns-to-normal | geopolitics_policy | exit_imbalance_recovery_t0.1 | -125.0 bps | [-174.1 bps, -61.6 bps] | 0.0% | 264.0s | still negative |
| a0b:2327929 | nba-okc-sas-2026-05-28 | sports_game_lines | exit_asymmetric_tp3000_sl300 | -512.4 bps | [-523.7 bps, -507.7 bps] | 0.0% | 300.0s | still negative |
| a0b:2362124 | bitcoin-up-or-down-on-may-28-2026 | daily_crypto_up_down | exit_asymmetric_tp3000_sl300 | -777.9 bps | [-862.9 bps, -685.3 bps] | 10.6% | 260.9s | still negative |
| a0b:2362186 | ethereum-up-or-down-on-may-28-2026 | daily_crypto_up_down | exit_strength_decay_p75 | -1322.7 bps | [-1436.0 bps, -1210.4 bps] | 0.0% | 62.8s | still negative |
| a0b:2364426 | btc-updown-4h-1779912000 | crypto_4h_up_down | exit_asymmetric_tp3000_sl300 | -492.2 bps | [-1601.7 bps, 567.1 bps] | 44.4% | 131.5s | still negative |
| a0b:2366225 | btc-updown-4h-1779926400 | crypto_4h_up_down | exit_strength_decay_p75 | -1588.3 bps | [-1901.8 bps, -1292.7 bps] | 0.0% | 11.2s | still negative |
| a0b:2367777 | btc-updown-4h-1779940800 | crypto_4h_up_down | exit_strength_decay_p25 | -1252.0 bps | [-1355.2 bps, -1103.7 bps] | 2.1% | 22.1s | still negative |
| a0b:566136 | will-psg-win-the-202526-champions-league | sports_neg_risk_outright | exit_asymmetric_tp1500_sl200 | -506.6 bps | [-508.2 bps, -505.4 bps] | 0.0% | 300.0s | still negative |

- `a0:558934` (will-spain-win-the-2026-fifa-world-cup-963): still negative; best `exit_trailing_stop_r30` at -344.5 bps, mean hold 300.0s.
- `a0:558936` (will-france-win-the-2026-fifa-world-cup-924): still negative; best `exit_asymmetric_tp1500_sl200` at -336.2 bps, mean hold 300.0s.
- `a0:665325` (us-iran-nuclear-deal-before-2027): still negative; best `exit_imbalance_recovery_t0.3` at -337.0 bps, mean hold 208.5s.
- `a0b:1971905` (strait-of-hormuz-traffic-returns-to-normal-by-end-of-june): still negative; best `exit_imbalance_recovery_t0.1` at -125.0 bps, mean hold 264.0s.
- `a0b:2327929` (nba-okc-sas-2026-05-28): still negative; best `exit_asymmetric_tp3000_sl300` at -512.4 bps, mean hold 300.0s.
- `a0b:2362124` (bitcoin-up-or-down-on-may-28-2026): still negative; best `exit_asymmetric_tp3000_sl300` at -777.9 bps, mean hold 260.9s.
- `a0b:2362186` (ethereum-up-or-down-on-may-28-2026): still negative; best `exit_strength_decay_p75` at -1322.7 bps, mean hold 62.8s.
- `a0b:2364426` (btc-updown-4h-1779912000): still negative; best `exit_asymmetric_tp3000_sl300` at -492.2 bps, mean hold 131.5s.
- `a0b:2366225` (btc-updown-4h-1779926400): still negative; best `exit_strength_decay_p75` at -1588.3 bps, mean hold 11.2s.
- `a0b:2367777` (btc-updown-4h-1779940800): still negative; best `exit_strength_decay_p25` at -1252.0 bps, mean hold 22.1s.
- `a0b:566136` (will-psg-win-the-202526-champions-league): still negative; best `exit_asymmetric_tp1500_sl200` at -506.6 bps, mean hold 300.0s.

## Hold-Time Read

The 300s backstop changed the experiment but did not rescue taker economics. Several best rows, especially sports and PSG, simply ride to the 300s time stop and remain negative. The crypto 4h rows that exit earlier still lose hundreds to thousands of bps. The closest row is Hormuz June with imbalance recovery at -125.0 bps, so the longer horizon helps reduce the damage in some slow books but does not create a positive executable taker edge in this capture.

## Interpretation

The key question is whether any config crosses zero with a reasonable CI. Rows with positive mean but CI spanning zero are exploratory only; rows with `ci_lo > 0` are the only robust positives in this pass. The table above should be read as exit-family exploration, not parameter selection, because the same A0/A0b capture is being reused.

Recommended next action for Justin: do not promote any A1.4g taker exit family into A2 as an edge; use A2 to monitor TOB signal quality with an explicit tight-spread/maker-first executable screen.
