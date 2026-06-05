# Block K5-STRESS Findings

> **Strat:** [[strat_market_making]] (Market-Making). Sibling: [[strat_options_delta]]. Arc: [[block_k_plain_english_synthesis]].
> Table terms: [[polymarket_table_dictionary]]

## Verdict

**4 category/categories pass the strict K5-STRESS gate** after typical structured-non-top3 and NegRisk-reliability filters. The looser mechanical PnL gate passes 6 categories. Treat passers as frozen paper-test candidates, not production green-lights.

The ex-ante maker population is **18,724 wallets**, defined only by corrected maker share >= 70% and >= 1,000 passive fills. It covers **529,170,802 passive fills** and **$16,361,630,265** maker notional.

Adding open/unresolved inventory changes PnL by **$10,306,951** or **-1.4 bps** versus resolved-only.

## Gate Summary

| category | proceed | full-pop bps | CI | typical wallet | structured non-top3 | structured median | non-top3 CI | ex rebate bps | positive months | calib net gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_4h | YES | 275 bps | [52.7 bps, 549 bps] | -26.3 bps | 189 bps | 2.4 bps | [21.8 bps, 373 bps] | 244 bps | 87.5% | 0.78c |
| culture | YES | 152 bps | [15.6 bps, 267 bps] | 10.1 bps | 401 bps | 11.7 bps | [134 bps, 821 bps] | 140 bps | 68.4% | 1.58c |
| other | YES | 99.6 bps | [54.6 bps, 141 bps] | 8.3 bps | 133 bps | 18.7 bps | [57.6 bps, 209 bps] | 81.1 bps | 78.0% | -0.85c |
| sports | YES | 54.8 bps | [2.3 bps, 108 bps] | -15.6 bps | 137 bps | 4.4 bps | [27.4 bps, 242 bps] | 38.9 bps | 68.1% | -3.27c |
| politics_negrisk | NO | 395 bps | [162 bps, 721 bps] | 10.2 bps | 1,177 bps | 9.8 bps | [502 bps, 2,174 bps] | 375 bps | 82.9% | -1.64c |
| tech | NO | 156 bps | [-47.6 bps, 457 bps] | 12.5 bps | 288 bps | -8.3 bps | [-423 bps, 1,189 bps] | 142 bps | 64.6% | -0.51c |
| economics | NO | 67.4 bps | [25.9 bps, 121 bps] | 19.6 bps | -69.2 bps | 31.6 bps | [-225 bps, 121 bps] | 60.7 bps | 83.7% | 0.91c |
| weather | NO | 67.2 bps | [-10.7 bps, 143 bps] | 9.6 bps | -110 bps | -16.8 bps | [-267 bps, 53.7 bps] | 56.5 bps | 82.9% | -0.28c |
| daily_crypto | NO | 59.9 bps | [52.0 bps, 66.6 bps] | -14.8 bps | 87.5 bps | -83.2 bps | [80.4 bps, 94.7 bps] | 29.2 bps | 84.6% | -0.13c |
| finance | NO | 30.0 bps | [-56.4 bps, 90.2 bps] | 10.1 bps | -82.4 bps | -143 bps | [-632 bps, 205 bps] | 15.1 bps | 73.0% | -2.74c |
| geopolitics | NO | -9.6 bps | [-83.4 bps, 54.2 bps] | 10.1 bps | 119 bps | 14.2 bps | [-17.7 bps, 266 bps] | -11.9 bps | 63.0% | -0.57c |

Gate rule used here: full-population CI lower bound above zero, structured non-top3 CI lower bound above zero, typical structured non-top3 wallet above zero, net still positive without rebate, not concentrated into fewer than two-thirds positive active months, and no unresolved NegRisk accounting caveat. Calibration is reported as an independent survivorship-immune cross-check, not as a substitute for wallet PnL; **0 categories** have tail calibration CI above zero.

## Survivorship Fix

| scope | wallets | markets | gross | net PnL | bps | CI |
| --- | --- | --- | --- | --- | --- | --- |
| closed_resolved_only | 17710 | 793096 | $22,881,024,328 | $254,016,324 | 111 bps | [69.8 bps, 151 bps] |
| full_marked_resolved_plus_open | 18433 | 833938 | $24,123,149,246 | $264,323,275 | 110 bps | [74.2 bps, 157 bps] |

Resolved inventory uses `closed_positions.parquet`, which already settles held/never-closed positions to the resolved $1/$0 outcome. Open and unresolved markets are reconstructed from raw fills and marked to the latest executed trade price because historical book mids are not in the owned fill files.

## Structure And Deployability

The structured playbook is defined ex ante as:

- two-sided maker USD share >= 60%
- carry-token share >= 50%
- crypto late near-50c spike-zone share <= 2%

The deployable number is the structured sub-population after excluding each market's global top-3 maker wallets. That is the closest historical proxy for what a non-incumbent can expect.

## Rebate / Cost Interpretation

The `net_without_rebate_bps` column removes maker rebates from the full-population result. Any category that only survives with rebate is policy-fragile. `data/analysis/csv_outputs/market_making/k5_stress.csv` also includes base marked PnL, maker rebate, and taker-fee columns for every aggregate.

## Adverse Selection

60s markouts below use a deterministic full-population sample of maker fills, not only winners.

| category | covered fills | sample share | mean markout | adverse cost | positive rate |
| --- | --- | --- | --- | --- | --- |
| tech | 798 | 0.0% | 86.9 bps | -86.9 bps | 39.3% |
| economics | 278 | 0.0% | 203 bps | -203 bps | 33.8% |
| geopolitics | 3212 | 0.0% | 230 bps | -230 bps | 43.0% |
| sports | 9941 | 0.0% | 232 bps | -232 bps | 40.6% |
| culture | 400271 | 100.0% | 312 bps | -312 bps | 40.4% |
| politics_negrisk | 6275 | 0.1% | 336 bps | -336 bps | 48.3% |
| other | 25165 | 0.0% | 403 bps | -403 bps | 41.2% |
| finance | 6736 | 0.1% | 778 bps | -778 bps | 45.8% |
| daily_crypto | 127752 | 0.1% | 1,051 bps | -1,051 bps | 50.3% |
| unknown | 20125 | n/a | 1,063 bps | -1,063 bps | 49.7% |
| weather | 2829 | 0.1% | 1,387 bps | -1,387 bps | 35.3% |
| crypto_4h | 394551 | 100.0% | 1,886 bps | -1,886 bps | 39.7% |

Positive markout means maker-favorable reversion; adverse cost is `-markout`.

## NegRisk

NegRisk merge/split/redemption mechanics are not in raw fill files; token-level trade PnL is reported, but linked-outcome settlement accounting remains unreliable for this gate.

Politics NegRisk rows are therefore shown for diagnostics but should not be used as a green-light until merge/split/redemption events are explicitly reconstructed.

## Calibration Cross-Check

The calibration rows bucket resolved binary markets by last observed traded price at fixed times-to-resolution. This is survivorship-immune to wallet selection, but it is still a trade-price proxy rather than a historical quote book. A positive tail calibration gap corroborates a real longshot premium; it does not prove that a new maker can capture it after inventory and capacity.

## My Read

K5's closed-only result was a useful reality check, but K5-STRESS is the gating version. The only acceptable build signal is a category that remains positive for the whole maker population and for structured non-incumbents, without being rebate-only or one-episode. Categories failing that standard are not maker-build candidates.
