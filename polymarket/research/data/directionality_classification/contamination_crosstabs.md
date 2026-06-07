---
title: "Contamination cross-tabs"
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
# Contamination cross-tabs
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


## Cross-tab 1 — primary_style × OLD Pool C (`negrisk_specialists`)

Question: how many of the OLD Pool C members (under the pre-rewrite definition — `negrisk_volume_share > 0.7` AND the other gates, with NO arb filter) are reclassified as `two_sided_directional` (i.e., phantom + negrisk-share flagged them as NegRisk arb when they were actually directional)?

OLD Pool C is recomputed from `traders.parquet` here rather than read from `data/cohorts/negrisk_specialists.parquet`, because that file is now already filtered by the new arb gate and would make this crosstab circular.

OLD Pool C size: **113** (matches the pre-rewrite cohort size of 113)

| primary_style | False | True | ALL |
|---|---|---|---|
| arb_like | 37905 | 0 | 37905 |
| mixed | 88069 | 10 | 88079 |
| pure_directional | 211788 | 8 | 211796 |
| two_sided_directional | 234330 | 95 | 234425 |
| ALL | 572092 | 113 | 572205 |

### Pool C composition under new classification

- `arb_like`             : 0 (0.0%) — the intended population
- `two_sided_directional`: 95 (84.1%) — **contamination**
- `pure_directional`     : 8 (7.1%) — also contamination
- `mixed`                : 10 (8.8%)

**Contamination measure: 91.2% of Pool C looks directional, not arb.**

## Cross-tab 2 — phantom_position_score bucket × primary_style

Question: how much does the existing phantom score conflate the three populations? (if phantom were a clean discriminator, high-phantom rows should be ~entirely `arb_like`.)

| phantom_bucket | arb_like | mixed | pure_directional | two_sided_directional | ALL |
|---|---|---|---|---|---|
| 1.0_to_1.3 | 31267 | 30335 | 156720 | 74530 | 292852 |
| 1.3_to_2.0 | 3132 | 14585 | 28328 | 53834 | 99879 |
| 2.0_to_3.0 | 1515 | 10542 | 11365 | 35153 | 58575 |
| 3.0_to_5.0 | 1001 | 10155 | 7029 | 29593 | 47778 |
| >=5.0 | 990 | 22462 | 8343 | 41302 | 73097 |
| null | 0 | 0 | 11 | 13 | 24 |
| ALL | 37905 | 88079 | 211796 | 234425 | 572205 |

### Among traders with phantom_position_score >= 3.0:
- `two_sided_directional`: 58.7%
- `mixed`: 27.0%
- `pure_directional`: 12.7%
- `arb_like`: 1.6%

(Phantom >= 3.0 was the implicit 'NegRisk arb' threshold under the old design.)