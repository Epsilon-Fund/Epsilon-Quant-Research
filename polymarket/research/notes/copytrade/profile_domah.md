# Trader profile — `0x9d84ce0306f8551e02efef1680475fc0f1dc1344`
> Table terms: [[polymarket_table_dictionary]]

**Verdict:** single-pool candidate (patient_accumulators)
**Pools qualified (2):** patient_accumulators, high_kelly_edge

## Headline metrics
| metric | value | note |
|---|---:|---|
| n_closed_positions | 16,306 |  |
| n_distinct_markets | 8,928 |  |
| active_days | 1,228 |  |
| n_fills_total | 604,311 |  |
| **mkt_total_pnl** | $4,007,338 | **PRIMARY — trust this** |
| pos_total_pnl | $4,007,338 | inflated for NegRisk arb |
| mkt_profit_factor | 1.57 |  |
| mkt_dollar_win_rate | 0.610 |  |
| mkt_sharpe | _1.04_ | _DIAGNOSTIC_ |
| mkt_kelly_fraction | 0.226 |  |
| phantom_position_score | 8.45 | >1 ⇒ NegRisk arb |
| negrisk_volume_share | 0.58 |  |

## Style profile
| metric | value | note |
|---|---:|---|
| style_role_balance | 0.89 | 1.0 = pure maker, 0.0 = pure taker |
| style_avg_holding_hours | 1709.6 |  |
| style_median_holding_hours | 793.0 |  |
| style_pct_sub_second | 31.7% | % fills within 1s of another from same address |
| style_avg_fill_size_usd | $271.82 |  |
| style_max_fill_size_usd | $399,600 |  |
| style_buy_sell_symmetry | 0.75 | 0 = directional, 1 = MM-shaped |

## Capital footprint
- **est_bankroll_usd_30d_max_approx**: $59,065,105
  - Lifetime peak deployed capital. Descriptive only — NOT to be used for forward-looking sizing decisions.

## Monthly cumulative PnL
| month | month_pnl | cum_pnl | n_markets_resolved |
|---|---:|---:|---:|
| 2022-12-01 00:00:00 | $361 | $361 | 2 |
| 2023-01-01 00:00:00 | $356 | $717 | 3 |
| 2023-02-01 00:00:00 | $-530 | $187 | 1 |
| 2023-03-01 00:00:00 | $24,868 | $25,055 | 27 |
| 2023-04-01 00:00:00 | $968 | $26,022 | 6 |
| 2023-05-01 00:00:00 | $42,873 | $68,896 | 9 |
| 2023-06-01 00:00:00 | $27,022 | $95,918 | 22 |
| 2023-07-01 00:00:00 | $10,845 | $106,762 | 57 |
| 2023-08-01 00:00:00 | $35,132 | $141,895 | 50 |
| 2023-09-01 00:00:00 | $32,335 | $174,230 | 43 |
| 2023-10-01 00:00:00 | $10,718 | $184,948 | 74 |
| 2023-11-01 00:00:00 | $19,986 | $204,934 | 82 |
| 2023-12-01 00:00:00 | $120,384 | $325,319 | 155 |
| 2024-01-01 00:00:00 | $189,141 | $514,460 | 179 |
| 2024-02-01 00:00:00 | $59,882 | $574,342 | 185 |
| 2024-03-01 00:00:00 | $97,249 | $671,591 | 305 |
| 2024-04-01 00:00:00 | $64,936 | $736,527 | 275 |
| 2024-05-01 00:00:00 | $-47,564 | $688,963 | 395 |
| 2024-06-01 00:00:00 | $58,359 | $747,322 | 464 |
| 2024-07-01 00:00:00 | $366,789 | $1,114,111 | 303 |
| 2024-08-01 00:00:00 | $788,856 | $1,902,967 | 460 |
| 2024-09-01 00:00:00 | $606,204 | $2,509,170 | 374 |
| 2024-10-01 00:00:00 | $103,186 | $2,612,356 | 282 |
| 2024-11-01 00:00:00 | $-197,342 | $2,415,015 | 877 |
| 2024-12-01 00:00:00 | $461,809 | $2,876,824 | 416 |
| 2025-01-01 00:00:00 | $15,699 | $2,892,523 | 112 |
| 2025-02-01 00:00:00 | $6,860 | $2,899,383 | 47 |
| 2025-03-01 00:00:00 | $6,568 | $2,905,951 | 27 |
| 2025-04-01 00:00:00 | $28,995 | $2,934,947 | 33 |
| 2025-05-01 00:00:00 | $-26,199 | $2,908,747 | 18 |
| 2025-06-01 00:00:00 | $122,495 | $3,031,242 | 184 |
| 2025-07-01 00:00:00 | $3,977 | $3,035,220 | 58 |
| 2025-08-01 00:00:00 | $-29,383 | $3,005,837 | 261 |
| 2025-09-01 00:00:00 | $28,328 | $3,034,165 | 320 |
| 2025-10-01 00:00:00 | $118,567 | $3,152,732 | 338 |
| 2025-11-01 00:00:00 | $23,494 | $3,176,226 | 159 |
| 2025-12-01 00:00:00 | $340,312 | $3,516,538 | 658 |
| 2026-01-01 00:00:00 | $289,551 | $3,806,089 | 329 |
| 2026-02-01 00:00:00 | $-18,954 | $3,787,135 | 159 |
| 2026-03-01 00:00:00 | $-161,820 | $3,625,315 | 273 |
| 2026-04-01 00:00:00 | $-4,457 | $3,620,858 | 14 |
| 2026-05-01 00:00:00 | $2,176 | $3,623,034 | 3 |
| 2026-06-01 00:00:00 | $-14,515 | $3,608,519 | 31 |
| 2026-07-01 00:00:00 | $-239 | $3,608,281 | 1 |
| 2026-11-01 00:00:00 | $330 | $3,608,611 | 2 |
| 2026-12-01 00:00:00 | $345,034 | $3,953,645 | 75 |

## Market mix

### Top 10 winning markets
| market_id | question | realised_pnl | n_fills | holding_h | neg_risk |
|---|---|---:|---:|---:|---|
| 253697 | Will Joe Biden win the 2024 Democratic Presidential Nominati | $815,127 | 1742 | 5226.2 | True |
| 253597 | Will Kamala Harris win the 2024 US Presidential Election? | $546,767 | 7399 | 4102.6 | True |
| 253592 | Will Joe Biden win the 2024 US Presidential Election? | $541,707 | 2823 | 7329.9 | True |
| 253722 | Biden wins the Popular Vote? | $352,002 | 1188 | 7183.3 | True |
| 572470 | Will Trump nominate Kevin Hassett as the next Fed chair? | $180,087 | 2512 | 12220.6 | True |
| 252294 | Biden drops out of presidential race? | $150,747 | 1813 | 9574.4 | False |
| 572469 | Will Trump nominate Kevin Warsh as the next Fed chair? | $137,086 | 2444 | 12284.4 | True |
| 254112 | Will JD Vance win the 2024 Republican VP nomination? | $135,531 | 552 | 4843.3 | True |
| 255053 | Will a Democrat win Nevada Presidential Election? | $114,281 | 393 | 5056.5 | True |
| 253913 | Will weed be rescheduled in 2024? | $108,062 | 591 | 8358.9 | False |

### Top 10 losing markets
| market_id | question | realised_pnl | n_fills | holding_h | neg_risk |
|---|---|---:|---:|---:|---|
| 253591 | Will Donald Trump win the 2024 US Presidential Election? | $-1,163,941 | 17141 | 7328.4 | True |
| 253701 | Will Kamala Harris win the 2024 Democratic Presidential Nomi | $-480,191 | 459 | 5219.1 | True |
| 253706 | Will Donald Trump win the popular vote in the 2024 President | $-396,380 | 4845 | 7206.4 | True |
| 255151 | Will a Democrat win Pennsylvania Presidential Election? | $-203,649 | 849 | 5838.7 | True |
| 501585 | Will a Democrat win the popular vote and the Presidency?  | $-200,546 | 1968 | 4171.6 | True |
| 519068 | Will Nicușor Dan win the Romanian presidential election? | $-129,684 | 140 | 5629.8 | True |
| 240613 | Which party wins 2024 US Presidential Election? | $-126,160 | 3562 | 7170.8 | False |
| 253750 | Ethereum ETF approved by May 31? | $-122,942 | 1085 | 3415.0 | False |
| 255152 | Will a Republican win Pennsylvania Presidential Election? | $-115,508 | 810 | 5838.7 | True |
| 1299962 | Will the government shutdown last 5 days or more? | $-114,325 | 453 | 1410.3 | False |

### NegRisk vs regular split
| neg_risk | pnl | n_markets |
|---|---:|---:|
| False | $1,779,120 | 4550 |
| True | $2,174,525 | 3598 |

## Holding-duration distribution (hours)
- n: 14,765, p10: 46.7, p50: 793.0, p90: 4799.23, p99: 8208.41, max: 15291.03

## Activity cadence
- 1004 days with at least one fill
- Avg fills/active-day: 643.2
- Max fills on a single day: 9,190

## Cohort positioning
| pool | trader's pnl_percentile | trader's pf_percentile | pool median PnL | pool median PF |
|---|---:|---:|---:|---:|
| patient_accumulators | 100% | 45% | $222,025 | 1.63 |
| high_kelly_edge | 100% | 56% | $2,132 | 1.51 |

---

_PnL computed from on-chain trade data + market resolution. Polymarket's UI may show different numbers due to fee accounting, mark-to-market on open positions, and merge/split events not captured here. Treat differences <20% as internally normal._
