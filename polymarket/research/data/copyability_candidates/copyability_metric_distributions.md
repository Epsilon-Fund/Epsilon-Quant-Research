---
title: "Copyability candidate metrics — distributions"
created: 2026-06-07
status: generated
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
# Copyability candidate metrics — distributions
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Population: **572,205 traders** (n_closed_positions > 50, operators excluded). No scoring, no thresholds — these tables are meant to be eyeballed for picking audit candidates.

**Output parquet:** `data/copyability_candidates/traders_copyability_metrics.parquet`

## 1. Schema

| column | description |
|---|---|
| `address` | PK (lowercase 0x… hex) |
| `mkt_total_pnl` | from traders.parquet — lifetime market-level PnL |
| `mkt_profit_factor` | from traders.parquet |
| `mkt_dollar_win_rate` | from traders.parquet |
| `mkt_sharpe` | from traders.parquet (diagnostic only — naive annualisation) |
| `n_closed_positions` | from traders.parquet |
| `n_distinct_markets` | from traders.parquet |
| `active_days` | from traders.parquet (lifetime) |
| `n_fills_total` | from traders.parquet |
| `phantom_position_score` | from traders.parquet — diagnostic, do NOT use as arb filter |
| `negrisk_volume_share` | from traders.parquet |
| `style_role_balance` | from traders.parquet (1.0 = pure maker) |
| `style_pct_sub_second` | from traders.parquet — share of fills <1s after another fill |
| `style_avg_holding_hours` | from traders.parquet |
| `style_buy_sell_symmetry` | from traders.parquet (1 = MM-shaped, 0 = directional) |
| `est_bankroll_usd_30d_max_approx` | from traders.parquet (descriptive only) |
| `primary_style` | from traders_directionality.parquet |
| `pct_markets_balanced_and_offsetting_vw` | from traders_directionality.parquet |
| `pct_markets_two_sided_directional_vw` | from traders_directionality.parquet |
| `fill_concentration_p50` | from traders_directionality.parquet |
| `net_to_gross_exposure` | from traders_directionality.parquet |
| `fragmentation_index` | median fills per closed position — high ⇒ trader scales in |
| `hold_to_resolution_share` | fraction of positions whose last fill was within final 5% of market life |
| `split_position_signature` | fraction of 2-outcome markets with bought_only+sold_no_buy pair — split-construction tell |
| `market_family_concentration` | HHI on |pnl| across families (0=uniform, 1=single family) |
| `dominant_family` | family contributing highest |pnl| share |
| `active_days_last_90d` | distinct days with ≥1 fill in last 90d (snapshot = max trade ts) |
| `volume_30d_to_lifetime_ratio` | last-30d fill notional / lifetime fill notional |
| `lifetime_volume_usd` | lifetime fill notional (all fills, all markets, open and closed) |
| `median_per_position_pnl` | median realised_pnl over closed positions |
| `pct_markets_winning` | fraction of markets where net PnL > 0 |
| `win_loss_size_ratio` | avg_pnl_winning_mkts / |avg_pnl_losing_mkts| |
| `audit_status` | 'audited' | 'flagged_uncopyable' | 'unrun' |
| `n_deployable_cells` | audited only — # family rows with A_real_capture>0.30 AND adv_sel>0.85 AND leader_pnl>0 |

## 2. Per-metric distributions

| metric | p10 | p25 | p50 | p75 | p90 | p99 | n_non_null |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fragmentation_index` | 1.000 | 1.000 | 2.000 | 2.000 | 3.000 | 11.000 | 572,205 |
| `hold_to_resolution_share` | 0.000 | 0.000 | 0.010 | 0.051 | 0.119 | 0.310 | 571,707 |
| `split_position_signature` | 0.000 | 0.055 | 0.258 | 0.542 | 0.820 | 1.000 | 571,639 |
| `market_family_concentration` | 0.344 | 0.452 | 0.675 | 0.966 | 1.000 | 1.000 | 572,205 |
| `active_days_last_90d` | 0.000 | 1.000 | 6.000 | 17.000 | 37.000 | 76.000 | 572,205 |
| `volume_30d_to_lifetime_ratio` | 0.000 | 0.000 | 0.000 | 0.190 | 0.892 | 1.000 | 572,205 |
| `median_per_position_pnl` | -2.400 | -0.721 | 0.005 | 0.200 | 1.425 | 27.527 | 572,205 |
| `pct_markets_winning` | 0.267 | 0.400 | 0.526 | 0.703 | 0.888 | 1.000 | 572,205 |
| `win_loss_size_ratio` | 0.077 | 0.307 | 0.735 | 1.227 | 2.277 | 29.095 | 559,460 |
| `n_deployable_cells` | (values, n=2): 0.000, 1.000 | | | | | | | 2 |
| `pct_markets_balanced_and_offsetting_vw` | 0.000 | 0.000 | 0.012 | 0.066 | 0.206 | 0.742 | 572,205 |
| `pct_markets_two_sided_directional_vw` | 0.000 | 0.022 | 0.141 | 0.352 | 0.584 | 0.956 | 572,205 |
| `fill_concentration_p50` | 0.680 | 0.811 | 0.969 | 0.998 | 0.999 | 1.000 | 572,205 |
| `net_to_gross_exposure` | 0.000 | 0.000 | 0.823 | 1.000 | 1.000 | 1.000 | 572,166 |

## 3. Categorical breakdowns

### `primary_style`

| value | n | share |
|---|---:|---:|
| `two_sided_directional` | 234,425 | 41.0% |
| `pure_directional` | 211,796 | 37.0% |
| `mixed` | 88,079 | 15.4% |
| `arb_like` | 37,905 | 6.6% |

### `dominant_family`

| value | n | share |
|---|---:|---:|
| `crypto` | 232,341 | 40.6% |
| `sports` | 140,569 | 24.6% |
| `other` | 104,659 | 18.3% |
| `politics` | 78,007 | 13.6% |
| `weather` | 12,215 | 2.1% |
| `macro` | 4,414 | 0.8% |

### `audit_status`

| value | n | share |
|---|---:|---:|
| `unrun` | 572,202 | 100.0% |
| `audited` | 2 | 0.0% |
| `flagged_uncopyable` | 1 | 0.0% |

## 4. Cross-tab — `primary_style` × `split_position_signature` quartile

Hypothesis: high split_position_signature should be concentrated in `arb_like`. If high-split traders show up across multiple styles, this new metric is doing independent work.

| _split_quartile | arb_like | mixed | pure_directional | two_sided_directional | ALL |
|---|---|---|---|---|---|
| Q1 (low) | 789 | 10064 | 69716 | 62483 | 143052 |
| Q2 | 3615 | 21437 | 46657 | 71352 | 143061 |
| Q3 | 4628 | 25285 | 45234 | 67894 | 143041 |
| Q4 | 7896 | 19943 | 29092 | 28899 | 85830 |
| P90+ | 20977 | 11350 | 21097 | 3797 | 57221 |
| ALL | 37905 | 88079 | 211796 | 234425 | 572205 |

Row-normalised (% of each `primary_style` falling in each split quartile):

| primary_style | Q1 (low) | Q2 | Q3 | Q4 | P90+ |
|---|---|---|---|---|---|
| arb_like | 2.1 | 9.5 | 12.2 | 20.8 | 55.3 |
| mixed | 11.4 | 24.3 | 28.7 | 22.6 | 12.9 |
| pure_directional | 32.9 | 22.0 | 21.4 | 13.7 | 10.0 |
| two_sided_directional | 26.7 | 30.4 | 29.0 | 12.3 | 1.6 |

## 5. Verification — known traders

| label | primary_style | audit | n_dep_cells | mkt_pnl | n_pos | frag_idx | hold_to_res | split_sig | fam_HHI | dom_family | active_90d | vol_30d_ratio | med_pos_pnl | pct_mkts_win | wl_size | role_bal | sub_sec |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Domah (audited, directional politics specialist)** | `two_sided_directional` | `audited` | 1.000 | $4,007,338 | 16,306 | 11.0 | 27.3% | **6.2%** | 0.546 | politics | 90 | 0.7% | $6 | 62.4% | 0.94 | 0.89 | 31.7% |
| **0xee00ba (audited, lower role_balance follow-up)** | `two_sided_directional` | `audited` | 0.000 | $4,658,451 | 11,726 | 3.0 | 7.1% | **50.3%** | 0.542 | sports | 70 | 8.7% | $6 | 70.3% | 0.53 | 0.79 | 40.2% |
| **0xd38b71f3 (flagged_uncopyable — split-construction)** | `arb_like` | `flagged_uncopyable` | n/a | $5,678,261 | 1,103 | 4.0 | 4.5% | **99.6%** | 0.960 | sports | 6 | 2.0% | $1 | 49.8% | 1.52 | 0.22 | 89.8% |
| **top_leaderboard (Phase 4 DD)** | `two_sided_directional` | `unrun` | n/a | $14,952,586 | 4,359 | 17.0 | 32.9% | **26.8%** | 0.673 | sports | 85 | 8.9% | $36 | 54.3% | 1.05 | 0.84 | 59.2% |
| **negrisk_specialist (Phase 4 DD — directional NegRisk)** | `two_sided_directional` | `unrun` | n/a | $1,795,457 | 1,365 | 10.0 | 37.1% | **30.1%** | 0.611 | politics | 90 | 5.7% | $7 | 54.9% | 2.24 | 0.78 | 46.7% |

Addresses (`address` column from output):

- `0x9d84ce0306f8551e02efef1680475fc0f1dc1344` — Domah (audited, directional politics specialist)
- `0xee00ba338c59557141789b127927a55f5cc5cea1` — 0xee00ba (audited, lower role_balance follow-up)
- `0xd38b71f3e8ed1af71983e5c309eac3dfa9b35029` — 0xd38b71f3 (flagged_uncopyable — split-construction)
- `0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee` — top_leaderboard (Phase 4 DD)
- `0x629bc4a1e53e1d475beb7ea3d388791e96dd995a` — negrisk_specialist (Phase 4 DD — directional NegRisk)

## 6. Top 50 by `mkt_total_pnl`

Starting point for picking audit candidates. This is sorted lifetime PnL; eyeball it alongside the columns above and pick 5-10 traders to audit.

| addr | audit | style | pnl | n_pos | split_sig | hold_res | frag | fam_HHI | dom_fam | win_rate | wl_size | active_90d | vol_30d_pct | role_bal | negrisk_share | n_dep_cells |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0x1f2dd6d473f3… | unrun | two_sided_directional | $22,073,955 | 86 | 46.3% | 2.4% | 27.0 | 0.964 | politics | 55.6% | 12.42 | 0 | 0.0% | 0.75 | 96.4% | n/a |
| 0x6a72f61820b2… | unrun | two_sided_directional | $14,952,586 | 4,359 | 26.8% | 32.9% | 17.0 | 0.673 | sports | 54.3% | 1.05 | 85 | 8.9% | 0.84 | 5.3% | n/a |
| 0x204f72f35326… | unrun | mixed | $10,636,907 | 167,135 | 18.4% | 74.4% | 14.0 | 0.620 | other | 57.2% | 0.83 | 90 | 12.8% | 0.57 | 52.2% | n/a |
| 0x2005d16a84ce… | unrun | two_sided_directional | $8,378,137 | 101,129 | 7.1% | 30.3% | 8.0 | 0.543 | other | 55.8% | 1.05 | 90 | 30.5% | 0.90 | 30.7% | n/a |
| 0x2a2c53bd278c… | unrun | two_sided_directional | $8,124,592 | 4,576 | 35.0% | 20.3% | 5.0 | 0.514 | sports | 54.1% | 1.00 | 53 | 47.8% | 0.88 | 11.1% | n/a |
| 0xdb27bf2ac5d4… | unrun | two_sided_directional | $6,349,427 | 2,148 | 76.0% | 25.8% | 14.0 | 0.771 | sports | 48.2% | 1.23 | 89 | 7.1% | 0.96 | 21.9% | n/a |
| 0x507e52ef684c… | unrun | two_sided_directional | $6,025,904 | 95,003 | 18.7% | 23.3% | 5.0 | 0.506 | other | 53.6% | 0.98 | 90 | 21.5% | 0.82 | 21.4% | n/a |
| 0xd38b71f3e8ed… | flagged_uncopyable | arb_like | $5,678,261 | 1,103 | 99.6% | 4.5% | 4.0 | 0.960 | sports | 49.8% | 1.52 | 6 | 2.0% | 0.22 | 0.5% | n/a |
| 0x17db3fcd93ba… | unrun | two_sided_directional | $5,453,680 | 235 | 77.4% | 0.0% | 9.0 | 0.716 | other | 75.8% | 2.08 | 0 | 0.0% | 0.59 | 3.4% | n/a |
| 0x006cc834cc09… | unrun | two_sided_directional | $5,083,695 | 625 | 32.2% | 7.4% | 3.0 | 0.497 | other | 45.2% | 1.85 | 19 | 0.0% | 0.71 | 15.1% | n/a |
| 0xe90bec87d9ef… | unrun | two_sided_directional | $4,936,822 | 8,430 | 53.5% | 23.8% | 3.0 | 0.520 | sports | 52.5% | 0.96 | 38 | 0.0% | 0.78 | 19.1% | n/a |
| 0x343d4466dc32… | unrun | arb_like | $4,827,994 | 59 | 96.2% | 0.0% | 19.0 | 1.000 | sports | 63.6% | 2.96 | 0 | 0.0% | 0.58 | 0.0% | n/a |
| 0x14964aefa2cd… | unrun | two_sided_directional | $4,697,680 | 1,496 | 59.2% | 20.2% | 6.0 | 0.975 | sports | 55.9% | 1.03 | 31 | 1.0% | 0.86 | 2.2% | n/a |
| 0xee00ba338c59… | audited | two_sided_directional | $4,658,451 | 11,726 | 50.3% | 7.1% | 3.0 | 0.542 | sports | 70.3% | 0.53 | 70 | 8.7% | 0.79 | 23.4% | 0.000 |
| 0xee613b3fc183… | unrun | mixed | $4,333,151 | 73,733 | 10.8% | 33.8% | 7.0 | 0.696 | sports | 51.8% | 1.02 | 81 | 13.7% | 0.84 | 0.1% | n/a |
| 0x9d84ce0306f8… | audited | two_sided_directional | $4,007,338 | 16,306 | 6.2% | 27.3% | 11.0 | 0.546 | politics | 62.4% | 0.94 | 90 | 0.7% | 0.89 | 57.5% | 1.000 |
| 0x7fb7ad0d194d… | unrun | two_sided_directional | $3,468,586 | 4,031 | 60.0% | 8.1% | 4.0 | 0.980 | sports | 51.3% | 1.05 | 2 | 0.0% | 0.82 | 1.2% | n/a |
| 0x93abbc022ce9… | unrun | arb_like | $3,464,953 | 1,228 | 98.2% | 13.9% | 8.0 | 0.825 | sports | 50.8% | 1.30 | 89 | 26.5% | 0.87 | 1.6% | n/a |
| 0xf705fa045201… | unrun | two_sided_directional | $3,446,528 | 2,907 | 31.5% | 20.5% | 15.0 | 1.000 | crypto | 51.7% | 2.20 | 90 | 7.4% | 0.27 | 0.0% | n/a |
| 0x0b9cae2b0dfe… | unrun | two_sided_directional | $3,365,174 | 1,676 | 60.1% | 10.5% | 4.0 | 0.517 | sports | 84.5% | 0.34 | 75 | 9.5% | 0.89 | 49.5% | n/a |
| 0x5bffcf561bca… | unrun | two_sided_directional | $3,275,528 | 1,108 | 35.6% | 21.3% | 8.0 | 0.726 | politics | 70.8% | 0.55 | 62 | 1.5% | 0.90 | 77.2% | n/a |
| 0x16b29c50f243… | unrun | arb_like | $3,140,045 | 9,708 | 58.0% | 30.4% | 4.0 | 0.871 | sports | 50.5% | 1.03 | 0 | 0.0% | 0.76 | 6.7% | n/a |
| 0x63ce34216125… | unrun | mixed | $3,049,690 | 66,995 | 3.2% | 5.9% | 64.0 | 1.000 | crypto | 54.2% | 1.17 | 60 | 0.0% | 0.68 | 0.0% | n/a |
| 0xb786b8b6335e… | unrun | two_sided_directional | $2,968,787 | 14,903 | 12.0% | 14.8% | 5.0 | 0.633 | sports | 56.5% | 1.05 | 23 | 0.0% | 0.87 | 15.0% | n/a |
| 0xa9878e59934a… | unrun | two_sided_directional | $2,811,928 | 2,275 | 51.6% | 9.1% | 3.0 | 0.728 | sports | 81.0% | 0.36 | 2 | 0.0% | 0.71 | 25.8% | n/a |
| 0xb6d6e99d3bfe… | unrun | two_sided_directional | $2,801,231 | 613 | 34.8% | 9.2% | 8.0 | 0.499 | other | 44.0% | 5.16 | 25 | 74.1% | 0.35 | 0.1% | n/a |
| 0xc2e7800b5af4… | unrun | arb_like | $2,745,591 | 336 | 97.6% | 34.7% | 32.0 | 0.508 | sports | 51.2% | 0.98 | 33 | 7.0% | 0.31 | 67.4% | n/a |
| 0xa6a856a8c8a7… | unrun | mixed | $2,710,832 | 21,216 | 31.2% | 0.1% | 6.0 | 0.980 | sports | 52.4% | 1.07 | 33 | 0.0% | 0.51 | 1.0% | n/a |
| 0x3cf3e8d5427a… | unrun | two_sided_directional | $2,701,637 | 6,474 | 6.1% | 36.3% | 12.0 | 0.758 | politics | 53.2% | 1.99 | 0 | 0.0% | 0.84 | 64.4% | n/a |
| 0xd7f85d0eb0fe… | unrun | pure_directional | $2,620,029 | 82 | 51.5% | 15.0% | 4.0 | 0.870 | politics | 38.8% | 16.54 | 60 | 0.6% | 0.89 | 89.1% | n/a |
| 0x1ff26f9f8a04… | unrun | two_sided_directional | $2,576,000 | 204 | 52.1% | 20.0% | 11.5 | 0.908 | sports | 57.4% | 1.07 | 14 | 0.0% | 0.76 | 0.8% | n/a |
| 0x777d9f00c2b4… | unrun | arb_like | $2,532,591 | 764 | 96.2% | 5.7% | 6.0 | 0.684 | sports | 52.5% | 1.23 | 66 | 56.8% | 0.75 | 10.7% | n/a |
| 0x461f3e886dca… | unrun | arb_like | $2,518,686 | 231 | 74.8% | 0.0% | 28.0 | 1.000 | sports | 43.3% | 1.46 | 0 | 0.0% | 0.43 | 0.0% | n/a |
| 0x51393c00184b… | unrun | arb_like | $2,500,694 | 7,421 | 64.9% | 17.9% | 4.0 | 0.979 | sports | 51.8% | 1.08 | 0 | 0.0% | 0.44 | 0.0% | n/a |
| 0x7ee1b64be701… | unrun | arb_like | $2,355,855 | 113 | 93.8% | 0.0% | 24.0 | 1.000 | sports | 56.9% | 0.95 | 0 | 0.0% | 0.51 | 0.0% | n/a |
| 0xcc500cbcc8b7… | unrun | two_sided_directional | $2,339,924 | 30,682 | 44.4% | 5.7% | 6.0 | 0.846 | crypto | 57.6% | 0.99 | 90 | 4.9% | 0.65 | 18.1% | n/a |
| 0x65e4246d770f… | unrun | arb_like | $2,310,022 | 206 | 94.8% | 0.0% | 23.0 | 1.000 | sports | 45.0% | 1.48 | 0 | 0.0% | 0.56 | 0.0% | n/a |
| 0xd0b4c4c020ab… | unrun | arb_like | $2,305,665 | 1,577 | 97.3% | 4.9% | 16.0 | 0.500 | sports | 54.7% | 0.91 | 16 | 0.0% | 0.63 | 39.1% | n/a |
| 0x1d8a377c5020… | unrun | arb_like | $2,299,269 | 183 | 94.4% | 6.1% | 4.0 | 0.989 | sports | 63.4% | 1.59 | 11 | 0.0% | 0.32 | 0.4% | n/a |
| 0x000d257d2dc7… | unrun | two_sided_directional | $2,289,570 | 2,033 | 67.1% | 16.2% | 10.0 | 0.593 | politics | 85.8% | 0.39 | 71 | 4.1% | 0.41 | 28.4% | n/a |
| 0xdc876e687377… | unrun | two_sided_directional | $2,288,427 | 7,322 | 49.5% | 37.2% | 5.0 | 0.578 | sports | 49.8% | 1.06 | 45 | 0.0% | 0.77 | 30.8% | n/a |
| 0xed107a85a458… | unrun | two_sided_directional | $2,223,493 | 2,307 | 76.7% | 20.3% | 6.0 | 0.313 | politics | 88.7% | 0.71 | 90 | 15.4% | 0.70 | 32.4% | n/a |
| 0x8c80d213c0cb… | unrun | two_sided_directional | $2,211,509 | 101 | 78.3% | 14.0% | 28.0 | 0.448 | sports | 41.8% | 4.97 | 46 | 37.6% | 0.81 | 31.9% | n/a |
| 0xd0d6053c3c37… | unrun | mixed | $2,185,548 | 106,489 | 6.5% | 4.7% | 52.0 | 1.000 | crypto | 54.1% | 1.12 | 61 | 0.0% | 0.63 | 0.0% | n/a |
| 0xf195721ad850… | unrun | arb_like | $2,155,617 | 109 | 92.2% | 32.7% | 46.0 | 0.979 | sports | 55.2% | 1.27 | 22 | 0.0% | 0.50 | 1.6% | n/a |
| 0xd1acd3925d89… | unrun | two_sided_directional | $2,134,073 | 2,307 | 30.7% | 15.1% | 5.0 | 0.350 | politics | 61.2% | 2.02 | 38 | 1.7% | 0.65 | 56.6% | n/a |
| 0x37c1874a60d3… | unrun | two_sided_directional | $2,120,159 | 8,459 | 39.9% | 32.5% | 5.0 | 0.583 | sports | 51.5% | 1.00 | 46 | 51.3% | 0.79 | 27.0% | n/a |
| 0x03e8a544e97e… | unrun | two_sided_directional | $1,998,420 | 1,260 | 89.0% | 26.6% | 34.0 | 0.896 | sports | 51.4% | 1.01 | 90 | 37.7% | 0.84 | 2.9% | n/a |
| 0x57cd939930fd… | unrun | mixed | $1,951,057 | 958 | 93.6% | 10.4% | 5.0 | 0.599 | other | 52.8% | 1.42 | 73 | 34.8% | 0.79 | 81.9% | n/a |
| 0x145c5dad6033… | unrun | arb_like | $1,926,321 | 138 | 89.6% | 6.1% | 5.0 | 0.994 | sports | 46.5% | 2.27 | 0 | 0.0% | 0.27 | 0.0% | n/a |
