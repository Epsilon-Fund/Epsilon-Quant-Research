---
title: "Metric distributions — traders_directionality"
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
# Metric distributions — traders_directionality
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Population: 572,205 traders with n_closed_positions > 50, operators excluded.

Thresholds (from the build script):
  - FC_BALANCED      = 0.60
  - NTG_OFFSETTING   = 0.20
  - FC_SKEWED        = 0.80
  - NTG_DIRECTIONAL  = 0.60
  - ARB_LIKE_THRESH  = 0.30
  - TWO_SIDED_THRESH = 0.20
  - PURE_DIR_FC_FLOOR= 0.85

## Per-metric percentiles

| metric | p10 | p25 | p50 | p75 | p90 | p99 | n_nonnull |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fill_concentration_p10` | 0.520 | 0.560 | 0.684 | 0.968 | 0.997 | 1.000 | 572,205 |
| `fill_concentration_p50` | 0.680 | 0.811 | 0.969 | 0.998 | 0.999 | 1.000 | 572,205 |
| `net_to_gross_exposure` | 0.000 | 0.000 | 0.823 | 1.000 | 1.000 | 1.000 | 572,166 |
| `pct_markets_balanced_and_offsetting` | 0.000 | 0.000 | 0.044 | 0.139 | 0.283 | 0.677 | 572,205 |
| `pct_markets_two_sided_directional` | 0.000 | 0.026 | 0.085 | 0.196 | 0.355 | 0.606 | 572,205 |
| `pct_markets_balanced_and_offsetting_vw` | 0.000 | 0.000 | 0.012 | 0.066 | 0.206 | 0.742 | 572,205 |
| `pct_markets_two_sided_directional_vw` | 0.000 | 0.022 | 0.141 | 0.352 | 0.584 | 0.956 | 572,205 |

## primary_style breakdown

| style | n | share |
|---|---:|---:|
| `two_sided_directional` | 234,425 | 41.0% |
| `pure_directional` | 211,796 | 37.0% |
| `mixed` | 88,079 | 15.4% |
| `arb_like` | 37,905 | 6.6% |

## Threshold justification

- **FC_BALANCED=0.60 / NTG_OFFSETTING=0.20** (arb signal): a perfectly balanced 2-outcome arb position has fc=0.5; allowing up to 0.60 gives slack for slightly skewed-but-still-balanced fills. NTG<0.20 is strict — true arb leaves you holding offsetting positions, not net-long one side.
- **FC_SKEWED=0.80 / NTG_DIRECTIONAL=0.60** (two-sided directional): the Domah signature — one side dominates volume (>=80%), and exposure is net-long (>0.6) but the per-outcome PnLs flip signs at resolution. These thresholds carve out the population the phantom score mislabels as arb.
- **ARB_LIKE_THRESH=0.30**: yields **37,905 arb_like traders** (6.6% of the qualifying population). Picked because: at 0.10 (an earlier placeholder), 18.9% of the population came up as arb_like — sweeping in many casual cross-side traders. At 0.30, the rate drops to ~6.6%, a defensible 'real arbitrageur' rate for prediction markets. The threshold sits between the population p90 (0.206) and p99 (0.742) of `pct_markets_balanced_and_offsetting_vw`, so the rule is: 'at least 30% of your volume is spent on arb-shaped markets'.
- **TWO_SIDED_THRESH=0.20** is higher than the arb threshold because two-sided directional behaviour is common (any trader who switches sides once will produce some), and we want to flag only traders for whom it's a substantive share of their book. Picks up ~41% of the population, mostly traders who hedge or reposition without closing.
- **PURE_DIR_FC_FLOOR=0.85**: a trader whose vol-weighted median market has 85%+ of volume on one outcome is by construction a one-side-at-a-time directional bettor. Sits below the population p50 of 0.969, so it catches anyone whose typical market really is one-sided.