# Polymarket Esports Latency Trader Screen
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


**Generated:** 2026-05-20  
**Purpose:** first-pass screen for traders with high win rate, strong prop-market PnL, and possible latency-arb fingerprints in esports markets  
**Primary local data:** `data/trades/*.parquet`, `data/markets/markets_2026-05-06.parquet`, `data/closed_positions.parquet`, `data/traders.parquet`  
**Companion notebook:** `notebooks/esports_latency_trader_screen.ipynb`  
**Intermediate cache:** `data/analysis/esports_latency_traders/`

This report starts from the esports market map and asks a narrower question: which wallets look unusually good in the most feasible latency markets, especially League of Legends and Dota 2 intragame props?

The answer is not "these wallets are doing latency arb." The local parquet has fills, prices, positions, and final outcomes, but it does not have the true in-game event timestamp for each kill, first blood, tower, Roshan, Baron, or threshold crossing. So the screen uses a **price-snap proxy**:

- Identify the winning outcome token for closed markets.
- Find the first time the winning token traded at or above $0.90.
- Count winner-token buys before that snap, especially buys at $0.80 or lower.
- Combine that with realised PnL, win rate, profit factor, market count, and prop-market concentration.

That makes this a triage layer. The next layer should join external match-event timestamps and ask whether these fills happened after the true event but before the market repriced.

## Repo And Handoff Context

Read before extending this screen:

- `CLAUDE.md`: research is offline-first, DuckDB over parquet, append-only data, no Postgres, and no live trading from the research module.
- `README.md`: the fill layer includes 1,064,500,317 fills through 2026-04-24 UTC, and `closed_positions.parquet` is already materialised.
- `notes/overview/data_quality/validation_report.md`: Gamma token ordering is validated; self-trades are dropped at view level; redemption synthesis matters for realised PnL.
- `notes/copytrade/phase5_design.md`: use ASOF / next-fill patterns for latency and slippage work; maker/taker distinction matters.
- `docs/METRICS_REFERENCE.md`: local formulas and caveats for `raw_trades`, `trader_actions`, `closed_positions`, and `traders.parquet`.

## Candidate Universe

I classified local Gamma markets by slug and question text, then retained:

- **Primary latency props:** LoL and Dota 2 live total kills, first blood, objective props, and multikill props.
- **Game-winner controls:** LoL and Dota 2 game winners.
- **CS2 activity controls:** CS2 moneyline/game-winner/odd-even markets.
- **Valorant controls:** Valorant moneyline and game-winner markets.

| bucket | game | market type | markets | closed markets | Gamma volume |
| --- | --- | --- | ---: | ---: | ---: |
| cs2_activity_control | Counter Strike 2 | series_moneyline_or_other | 7,480 | 7,396 | $412,202,695 |
| cs2_activity_control | Counter Strike 2 | game_winner | 9,443 | 9,332 | $171,220,272 |
| cs2_activity_control | Counter Strike 2 | odd_even_total_rounds | 6,988 | 6,757 | $597,783 |
| cs2_activity_control | Counter Strike 2 | odd_even_total_kills | 6,988 | 6,183 | $289,534 |
| game_winner_control | League of Legends | game_winner | 2,707 | 2,520 | $437,842,057 |
| game_winner_control | Dota 2 | game_winner | 2,901 | 2,875 | $142,851,631 |
| primary_latency_prop | League of Legends | live_total_kills | 16,139 | 15,907 | $7,353,275 |
| primary_latency_prop | Dota 2 | live_total_kills | 23,830 | 23,736 | $3,911,029 |
| primary_latency_prop | League of Legends | first_blood | 1,300 | 1,229 | $1,668,057 |
| primary_latency_prop | League of Legends | objective_or_multikill_prop | 11,223 | 9,708 | $1,565,698 |
| primary_latency_prop | Dota 2 | objective_or_multikill_prop | 6,984 | 6,773 | $213,335 |
| primary_latency_prop | Dota 2 | first_blood | 1,030 | 1,011 | $205,940 |
| valorant_control | Valorant | series_moneyline_or_other | 2,281 | 2,259 | $53,414,904 |
| valorant_control | Valorant | game_winner | 2,921 | 2,873 | $21,017,702 |

The screen produced:

- 102,215 candidate markets.
- 204,430 candidate outcome tokens.
- 240,392 trader x bucket x game x market-type position cells.
- 18,153 ranked trader cells after minimum activity and profitability filters.
- 210 address-level primary prop candidates.

## Metrics

Position metrics:

- `n_positions`, `n_markets`, `gross_usd_volume`, `realised_pnl`.
- `win_rate`: profitable closed positions / all closed positions in the cell.
- `profit_factor`: gross profit / absolute gross loss.
- `held_to_resolution_rate` and `avg_holding_hours` are retained in the cache for later copyability work.

Latency-shape metrics:

- `winner_buy_actions`: buy actions on the eventual winning outcome token.
- `pre_snap_winner_buys`: winner-token buys before the first trade at $0.90 or higher.
- `stale_pre_snap_winner_buys`: pre-snap winner-token buys at $0.80 or lower.
- `near300_snap_winner_buys`: winner-token buys in the 300 seconds before the first $0.90 snap.
- `avg_edge_to_90_cents`: average cents between the entry price and $0.90 for pre-snap winner buys.
- `pre_snap_capture_rate`: pre-snap winner buys / winner buys.
- `near300_capture_rate`: near-300-second winner buys / winner buys.

Scoring:

- Cell-level `latency_suspicion_score` rewards high win rate, profit factor, repeated market coverage, strong near-snap buying, cheap pre-snap winner entries, and meaningful volume.
- Address-level `aggregate_latency_score` aggregates only primary latency-prop cells.
- The score is a ranker, not a probability. It intentionally over-includes candidates for manual review.

## Main Result

The top aggregate addresses are mostly automated prop specialists with extremely high realised win rates in LoL/Dota 2 props. They often buy the winning side close to the price snap, but the strongest high-WR names do **not** have a high share of cheap pre-snap buys. That means they are interesting, but not yet direct proof of event-latency capture.

| rank | address | cells | games | positions | gross USD | PnL | win rate | profit factor | near-300 rate | pre-snap rate | stale rate | edge to 90c | score |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0xb8415ce9e03af2618cf6f6c549c26e8953613cfb` | 5 | Dota 2, LoL | 8,070 | $348,113 | $6,083 | 0.993 | 3.57 | 0.524 | 0.002 | 0.002 | 57.1c | 0.860 |
| 2 | `0x90a50913050e7320799ca2f51aed42f37684c3f1` | 4 | Dota 2, LoL | 2,661 | $211,054 | $5,055 | 0.989 | 7.68 | 0.597 | 0.001 | 0.001 | 89.0c | 0.860 |
| 3 | `0x5a05e30ef2ae848927c7a724b93292bf8cd14c5a` | 6 | Dota 2, LoL | 2,039 | $54,171 | $2,244 | 0.995 | 5.42 | 0.738 | 0.013 | 0.013 | 50.4c | 0.857 |
| 4 | `0x6fdddf25b92251ed1515703cda43bf8ff5f5d385` | 3 | Dota 2, LoL | 6,761 | $454,598 | $21,957 | 0.993 | 12.27 | 0.422 | 0.005 | 0.004 | 47.7c | 0.857 |
| 5 | `0x1a6df901e63422055a1658b0ef2b2a871507cf4a` | 2 | Dota 2, LoL | 677 | $52,562 | $1,393 | 0.950 | 24.93 | 0.342 | 0.005 | 0.005 | 88.7c | 0.856 |
| 6 | `0x033eeb3efed4a42059faace467c5c3f6eb2e1bb9` | 3 | Dota 2, LoL | 1,074 | $54,018 | $878 | 0.974 | 3.35 | 0.555 | 0.004 | 0.004 | 86.5c | 0.855 |
| 7 | `0x642c673dd63f9de86715a9566c55671eb52821a7` | 6 | Dota 2, LoL | 4,489 | $160,926 | $2,988 | 0.979 | 19.37 | 0.807 | 0.004 | 0.004 | 45.9c | 0.853 |
| 8 | `0xe303b593e9055265e9c9ba0db2ed70f98d6b401a` | 3 | Dota 2, LoL | 2,792 | $22,850 | $681 | 0.996 | 9.95 | 0.822 | 0.001 | 0.001 | 47.2c | 0.842 |
| 9 | `0xfe1efad76c5ff1e5c127d86fcc225480965d7ee3` | 2 | LoL | 81 | $3,507 | $937 | 0.815 | 4.20 | 0.449 | 0.327 | 0.306 | 28.5c | 0.838 |
| 10 | `0x5f4738cf7db322c772c4b6ce2ace6fd6d677f611` | 1 | Dota 2 | 30 | $2,059 | $420 | 0.900 | 84.91 | 0.208 | 0.667 | 0.667 | 27.9c | 0.823 |
| 11 | `0xad72ffe37df00548959c2f86f0333e7c958e4f5e` | 5 | Dota 2, LoL | 514 | $69,920 | $11,615 | 0.994 | 193.95 | 0.668 | 0.054 | 0.041 | 22.9c | 0.814 |
| 12 | `0xda325ce63bc3b797ca44e9d5ca7805eb4a82db4c` | 2 | LoL | 928 | $119,022 | $1,154 | 0.926 | 1.77 | 0.406 | 0.003 | 0.003 | 88.5c | 0.813 |

## Stronger Latency-Shape Subset

The stricter screen requires at least five stale pre-snap winner buys. This is closer to the behaviour we would expect from a latency strategy: buying the eventual winning token before the market has traded at $0.90, often at prices still far below certainty.

This subset is more useful for event-timestamp validation than the raw score leaderboard.

| rank | address | cells | games | positions | gross USD | PnL | win rate | profit factor | stale buys | pre-snap rate | avg entry | edge to 90c | score |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0xb8415ce9e03af2618cf6f6c549c26e8953613cfb` | 5 | Dota 2, LoL | 8,070 | $348,113 | $6,083 | 0.993 | 3.57 | 9 | 0.002 | $0.329 | 57.1c | 0.860 |
| 2 | `0x5a05e30ef2ae848927c7a724b93292bf8cd14c5a` | 6 | Dota 2, LoL | 2,039 | $54,171 | $2,244 | 0.995 | 5.42 | 15 | 0.013 | $0.396 | 50.4c | 0.857 |
| 3 | `0x6fdddf25b92251ed1515703cda43bf8ff5f5d385` | 3 | Dota 2, LoL | 6,761 | $454,598 | $21,957 | 0.993 | 12.27 | 26 | 0.005 | $0.423 | 47.7c | 0.857 |
| 4 | `0x642c673dd63f9de86715a9566c55671eb52821a7` | 6 | Dota 2, LoL | 4,489 | $160,926 | $2,988 | 0.979 | 19.37 | 10 | 0.004 | $0.441 | 45.9c | 0.853 |
| 5 | `0xfe1efad76c5ff1e5c127d86fcc225480965d7ee3` | 2 | LoL | 81 | $3,507 | $937 | 0.815 | 4.20 | 15 | 0.327 | $0.615 | 28.5c | 0.838 |
| 6 | `0x5f4738cf7db322c772c4b6ce2ace6fd6d677f611` | 1 | Dota 2 | 30 | $2,059 | $420 | 0.900 | 84.91 | 16 | 0.667 | $0.621 | 27.9c | 0.823 |
| 7 | `0xad72ffe37df00548959c2f86f0333e7c958e4f5e` | 5 | Dota 2, LoL | 514 | $69,920 | $11,615 | 0.994 | 193.95 | 12 | 0.054 | $0.671 | 22.9c | 0.814 |
| 8 | `0x27b3b62c862578836706909bcaeec5851b8a59b1` | 3 | LoL | 595 | $24,684 | $550 | 0.976 | 6.39 | 5 | 0.011 | $0.600 | 30.0c | 0.810 |
| 9 | `0x51f9122c7ed2ceb749d8d63617153e98d3ba19a0` | 4 | Dota 2, LoL | 5,802 | $104,258 | $889 | 0.940 | 2.62 | 19 | 0.007 | $0.613 | 28.7c | 0.806 |
| 10 | `0x14abdc6effdb42c94b17466c91dc09528c072f5a` | 3 | LoL | 203 | $5,607 | $356 | 0.990 | 120.31 | 7 | 0.052 | $0.584 | 31.6c | 0.804 |
| 11 | `0x92dc7bb5d2e4e98ea96ddd4b770a450c088f9980` | 4 | Dota 2, LoL | 1,201 | $32,596 | $891 | 0.979 | 178.43 | 8 | 0.012 | $0.650 | 25.0c | 0.803 |
| 12 | `0xbbdc27a253d163c6da200719c7eeef316faf9da2` | 2 | LoL | 64 | $9,543 | $1,723 | 0.984 | 345.62 | 5 | 0.074 | $0.529 | 37.1c | 0.802 |

Notable lower-ranked latency-shape names:

- `0x6e32312760e4604d45a8ae69cede9ef9a0b8ab65`: LoL live total kills, 43 positions, $3,118 PnL, 38 stale pre-snap winner buys, 0.792 pre-snap rate, average entry $0.523.
- `0x67e65bd160df9306af70f0f4d211b2308e6686ed`: LoL first blood, 52 positions, $4,406 PnL, 51 stale pre-snap winner buys, 0.622 pre-snap rate, average entry $0.528.
- `0x509db41b6314a8840f08e8ff4db7ee9e04300a6b`: LoL live total kills, 114 positions, $47,097 PnL, 638 stale pre-snap winner buys, average entry $0.575.
- `0x1f03e496b56883fc02ef30e5ed30be2f9547b1be`: LoL live total kills, 91 positions, $27,714 PnL, 137 stale pre-snap winner buys, average entry $0.582.

These lower-ranked names sacrifice headline win rate, but they may be more important for latency validation because the stale-entry count is much denser.

## Cell-Level Standouts

The strongest individual cells are concentrated in LoL live total kills, LoL objective props, and Dota 2 objective/live-kill props.

| rank | address | game | market type | positions | markets | gross USD | PnL | win rate | profit factor | winner buys | pre buys | near-300 buys | pre rate | near-300 rate | edge to 90c | score |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0x6fdddf25b92251ed1515703cda43bf8ff5f5d385` | LoL | live_total_kills | 6,561 | 3,372 | $382,718 | $19,536 | 0.994 | 11.08 | 5,883 | 29 | 2,557 | 0.005 | 0.435 | 50.6c | 0.861 |
| 2 | `0xb8415ce9e03af2618cf6f6c549c26e8953613cfb` | LoL | objective_or_multikill_prop | 3,134 | 1,572 | $126,594 | $3,187 | 0.994 | 4.81 | 1,965 | 9 | 1,100 | 0.005 | 0.560 | 58.7c | 0.861 |
| 3 | `0xb8415ce9e03af2618cf6f6c549c26e8953613cfb` | Dota 2 | objective_or_multikill_prop | 1,489 | 748 | $55,898 | $1,751 | 0.992 | 4.20 | 950 | 1 | 639 | 0.001 | 0.673 | 89.0c | 0.855 |
| 4 | `0x642c673dd63f9de86715a9566c55671eb52821a7` | Dota 2 | objective_or_multikill_prop | 2,130 | 1,080 | $81,993 | $1,585 | 0.982 | 17.08 | 1,149 | 9 | 959 | 0.008 | 0.835 | 46.3c | 0.851 |
| 5 | `0xe303b593e9055265e9c9ba0db2ed70f98d6b401a` | Dota 2 | live_total_kills | 1,751 | 877 | $18,603 | $510 | 0.999 | 10205.75 | 941 | 1 | 887 | 0.001 | 0.943 | 47.2c | 0.839 |
| 6 | `0x033eeb3efed4a42059faace467c5c3f6eb2e1bb9` | Dota 2 | live_total_kills | 832 | 419 | $35,562 | $467 | 0.976 | 2.56 | 625 | 2 | 365 | 0.003 | 0.584 | 85.2c | 0.837 |
| 7 | `0x1a6df901e63422055a1658b0ef2b2a871507cf4a` | LoL | live_total_kills | 357 | 182 | $23,599 | $449 | 0.947 | 11.56 | 308 | 3 | 90 | 0.010 | 0.292 | 88.7c | 0.830 |
| 8 | `0x5a05e30ef2ae848927c7a724b93292bf8cd14c5a` | LoL | first_blood | 166 | 83 | $5,660 | $844 | 0.952 | 3.42 | 113 | 8 | 54 | 0.071 | 0.478 | 44.8c | 0.829 |
| 9 | `0x033eeb3efed4a42059faace467c5c3f6eb2e1bb9` | Dota 2 | first_blood | 89 | 45 | $12,930 | $309 | 0.944 | 5.74 | 101 | 1 | 39 | 0.010 | 0.386 | 89.0c | 0.827 |
| 10 | `0x5f4738cf7db322c772c4b6ce2ace6fd6d677f611` | Dota 2 | objective_or_multikill_prop | 30 | 23 | $2,059 | $420 | 0.900 | 84.91 | 24 | 16 | 5 | 0.667 | 0.208 | 27.9c | 0.821 |

## Control Read

The control buckets also surface high-scoring traders, especially CS2 moneyline/game-winner and LoL/Dota game-winner cells. That is expected: terminal markets also snap to certainty. The controls are useful because they warn against treating the price-snap proxy as a pure latency signal.

Top control examples:

- `0x3a959c179cf018395db289eb553d09e70e0f343b`: CS2 series/moneyline, 64 positions, $111,738 gross, $21,293 PnL, 0.859 win rate.
- `0x3954e1c6b7bc5645d9d3425a6f4490889ef122d9`: LoL game winner, 18 positions, $254,055 gross, $24,580 PnL, 0.889 win rate.
- `0x310f6937ed43dfb00722198ff787ca4a61fc4e42`: LoL game winner, 70 positions, $915,618 gross, $92,353 PnL, 0.771 win rate.

These may still be valuable traders to profile, but they are less clean for intragame latency arb than threshold/objective props.

## Interpretation

Three groups matter:

1. **High-WR prop specialists:** `0xb8415...`, `0x90a509...`, `0x5a05e...`, `0x6fdddd...`, `0x642c...`, `0xe303...`. These wallets are repeat winners across many LoL/Dota props. Their near-snap buying is strong, but cheap pre-snap capture is usually a small percentage of winner buys.
2. **Latency-shaped stale-entry wallets:** `0xfe1efa...`, `0x5f473...`, `0x6e323...`, `0x67e65...`, `0x509db...`, `0x1f03...`. These have more direct evidence of buying eventual winners before the $0.90 snap at still-stale prices, but often with smaller sample sizes or lower win rates.
3. **Terminal-market winners:** CS2 and game-winner controls show that high PnL and price-snap timing can also come from ordinary correct prediction, market-making, or end-of-game repricing. They are useful controls, not primary latency targets.

The strongest immediate research target is not a single wallet. It is a wallet-market-event tuple:

```text
address x event_slug x market_slug x winning_token x fill_timestamp x first_90c_timestamp x external_event_timestamp
```

That tuple lets us separate:

- buying before the in-game event,
- buying after the event but before Polymarket reprices,
- buying after Polymarket already moved but before final settlement,
- and passive maker fills that were merely lifted by someone else.

## Recommended Next Pass

1. Pull full event-level Gamma metadata for the top LoL/Dota markets in these wallets, including event slug, game number, prop line, and resolution text.
2. For the top stale-entry wallets, export fill-level rows around each pre-snap buy: 10 minutes before to 5 minutes after the first $0.90 print.
3. Add external timestamps for a small hand-labelled sample: kill threshold crossing, first blood, Baron/Roshan, tower/inhibitor/barracks, multikill.
4. Re-score only fills where `external_event_timestamp < fill_timestamp < first_90c_timestamp`.
5. Split taker buys from passive maker fills. A taker buy after the external event is much stronger evidence than a resting maker order filled during repricing.
6. Cluster linked wallets by shared markets, synchronized timestamps, and opposite-side / transfer-like behaviour before doing copyability analysis.

The companion notebook loads the cached screen artifacts and visualises the score distribution, stale-entry subset, PnL/volume profile, and market-type concentration.
