# Polymarket Esports Market Map
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


**Generated:** 2026-05-19  
**Purpose:** grid-search reference for esports latency-arb feasibility analysis  
**Primary local data:** `data/trades/*.parquet`, `data/markets/markets_2026-05-06.parquet`, `data/closed_positions.parquet`  
**Companion notebook:** `notebooks/esports_market_map.ipynb`

This note maps the live esports surface from Gamma and ties it back to the existing local fill and closed-position stack. The key adjustment versus a pure API scrape is that this repo already has the fill universe, including closed trades, so feasibility work should join Gamma metadata onto historical fills instead of starting from market discovery.

## Repo And Handoff Context

Read before using this report:

- `CLAUDE.md`: research is offline-first, DuckDB over parquet, append-only data, no Postgres, no live trading in the research module.
- `README.md`: phases 1-4 are complete; the local raw fill layer covers 1,064,500,317 fills through 2026-04-24 UTC, with `closed_positions.parquet` already materialised.
- `notes/overview/data_quality/validation_report.md`: orphans are low overall, self-trades are dropped at view level, Gamma token ordering validated, and redemption synthesis is required because most positions settle by redemption rather than on-book sell.
- `notes/copytrade/phase5_design.md`: latency/slippage analysis should use next-fill or ASOF-style matching, not row-by-row lookups; maker/taker distinction matters because taker fills are latency-sensitive.
- `docs/METRICS_REFERENCE.md`: `raw_trades`, `joined_fills`, `trader_actions`, `closed_positions`, and `traders.parquet` formulas and caveats.

Important local-data caveat: `data/markets/markets_2026-05-06.parquet` is a market-level Gamma snapshot. It has `id`, `condition_id`, `question`, `slug`, `outcomes`, `outcome_prices`, `volume`, `liquidity`, `active`, `closed`, `end_date`, `neg_risk`, and `clob_token_ids`. It does **not** store event slug, `series_id`, tournament, line, or `sportsMarketType`. Use a fresh Gamma event pull for those fields, then join by `conditionId` / market slug / token IDs.

## Coverage Summary

Gamma's `active=true, closed=false` universe includes stale unresolved events. For latency work, I split that from a near-term slate defined as `endDate >= 2026-05-19 00:00:00 UTC`.

| game | series_id | series slug | Gamma open events | Gamma open sub-markets | near-term events | near-term sub-markets | series 24h vol | near-term vol |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Counter Strike 2 | 10310 | `counter-strike` | 487 | 4,358 | 67 | 474 | $2,660,548 | $1,053,997 |
| Dota 2 | 10309 | `dota-2` | 63 | 3,724 | 13 | 443 | $7,968,809 | $6,087,388 |
| League of Legends | 10311 | `league-of-legends` | 105 | 3,602 | 60 | 2,128 | $5,214,908 | $4,359,430 |
| Rocket League | 10433 | `rocket-league` | 12 | 108 | 12 | 108 | $15 | $550 |
| Valorant | 10369 | `valorant` | 18 | 94 | 17 | 89 | $553,497 | $357,437 |
| Mobile Legends: Bang Bang | 10426 | `mobile-legends-bang-bang` | 16 | 80 | 16 | 80 | $109 | $463 |
| StarCraft 2 | 10435 | `starcraft-2` | 4 | 36 | 4 | 36 | $2,009 | $2,369 |
| Honor of Kings | 10434 | `honor-of-kings` | 1 | 15 | 1 | 15 | $192 | $213 |
| Rainbow Six Siege | 10432 | `rainbow-six-siege` | 1 | 5 | 0 | 0 |  | $0 |
| Overwatch | 10430 | `overwatch` | 1 | 4 | 0 | 0 |  | $0 |
| Call of Duty | 10427 | `call-of-duty` | 0 | 0 | 0 | 0 |  | $0 |
| League of Legends: Wild Rift | 10429 | `league-of-legends-wild-rift` | 0 | 0 | 0 | 0 |  | $0 |
| PUBG | 10431 | `pubg` | 0 | 0 | 0 | 0 |  | $0 |
| StarCraft: Brood War | 10436 | `starcraft-brood-war` | 0 | 0 | 0 | 0 |  | $0 |

Near-term market count is dominated by LoL because a single BO3 can spawn dozens of kill/objective props. Raw open count is dominated by CS2 because many unresolved older events remain `active`.

## Historical Fill Surface

The local snapshot tags 145,636 esports markets by market slug/question regex. The raw fill aggregates below scan `data/trades/trades_seed.parquet` plus all `data/trades/trades_delta_shard*.parquet`; therefore they include closed trades, not just currently open markets.

| game | local markets | closed markets | raw fills | raw notional | fill cadence median | fill cadence p90 |
|---|---:|---:|---:|---:|---:|---:|
| League of Legends | 39,778 | 37,198 | 6,999,616 | $1,082,999,494 | 0s | 72s |
| Counter Strike 2 | 41,373 | 40,026 | 7,559,020 | $725,182,304 | 2s | 96s |
| Dota 2 | 42,763 | 42,324 | 3,229,473 | $343,254,158 | 2s | 108s |
| Valorant | 8,613 | 8,493 | 1,247,172 | $77,817,377 | 2s | 246s |
| Call of Duty | 893 | 828 | 98,184 | $4,987,831 | 6s | 1178s |
| Honor of Kings | 5,377 | 5,325 | 112,678 | $4,155,735 | 0s | 1054s |
| Rainbow Six Siege | 1,733 | 1,725 | 45,971 | $1,532,903 | 4s | 1922s |
| Mobile Legends: Bang Bang | 1,779 | 1,602 | 41,912 | $1,530,185 | 26s | 3645s |
| StarCraft 2 | 1,448 | 1,418 | 14,392 | $668,013 | 2s | 2955s |
| Rocket League | 1,087 | 1,087 | 16,735 | $615,355 | 6s | 2040s |
| Overwatch | 791 | 788 | 11,076 | $317,695 | 6s | 4247s |

Fill cadence is a next-fill proxy by outcome token, not a proof of exploitable latency. It says how quickly another fill historically appears after a fill in the same outcome. A 0 second median means timestamp ties / same-block clusters are common; it does not imply sub-second reaction is available from the public data path.

## Sub-Market Types

Current near-term Gamma market types:

- **LoL:** `kill_over_under_game`, `lol_both_teams_baron`, `lol_both_teams_dragon`, `lol_both_teams_inhibitors`, `lol_quadra_kill`, `lol_penta_kill`, `lol_odd_even_total_kills`, `first_blood_game`, game winner, moneyline, handicap, total games.
- **Dota 2:** `kill_over_under_game`, `dota2_game_ends_daytime`, `dota2_both_teams_roshan`, `dota2_both_teams_barracks`, `dota2_ultra_kill`, `dota2_rampage`, game winner, moneyline, handicap, total games.
- **CS2:** moneyline, child moneyline, map handicap, total games, `cs2_odd_even_total_kills`, `cs2_odd_even_total_rounds`.
- **Valorant / MLBB / Rocket League / SC2 / R6 / Overwatch / HoK:** mostly moneyline, child moneyline, total games, and map/game handicap.

Historical notional by market-type proxy confirms where the money has been:

| rank | game | market-type proxy | fills | markets traded | raw notional |
|---:|---|---|---:|---:|---:|
| 1 | League of Legends | series moneyline or other | 3,805,512 | 2,232 | $589,492,200 |
| 2 | Counter Strike 2 | series moneyline or other | 4,779,573 | 6,453 | $478,659,000 |
| 3 | League of Legends | game winner | 2,668,217 | 2,148 | $450,008,300 |
| 4 | Counter Strike 2 | game winner | 2,293,439 | 7,869 | $201,723,200 |
| 5 | Dota 2 | game winner | 1,315,083 | 2,653 | $170,087,700 |
| 6 | Dota 2 | series moneyline or other | 1,630,791 | 3,505 | $162,317,500 |
| 7 | Valorant | series moneyline or other | 841,821 | 1,988 | $53,652,600 |
| 8 | League of Legends | handicap | 162,487 | 1,806 | $21,981,300 |
| 9 | Counter Strike 2 | handicap | 170,755 | 4,726 | $20,772,000 |
| 10 | Valorant | game winner | 333,746 | 2,544 | $19,887,500 |
| 11 | Counter Strike 2 | series total games | 241,431 | 3,968 | $17,431,100 |
| 12 | League of Legends | live total kills | 132,459 | 11,528 | $6,662,400 |
| 13 | Dota 2 | live total kills | 161,853 | 18,268 | $4,907,200 |

For latency-arb, raw notional alone is not enough. The best candidates combine high notional, event-driven resolution, frequent intragame trigger events, and an external data source faster than Polymarket participants.

## Resolution Sources

Observed current sources by game:

- **LoL:** current event descriptions overwhelmingly mention `https://gol.gg/esports/home`; many event-level `resolutionSource` fields point to streams. For settlement, treat `gol.gg` plus stream/video fallback as the relevant source stack.
- **CS2:** largely `hltv.org`, Kick/Twitch/YouTube streams, and tournament channels.
- **Valorant:** `vlr.gg` plus official / regional streams.
- **MLBB, Rocket League, SC2, R6, Overwatch, HoK:** Liquipedia is common in descriptions, often with stream-specific event-level sources.

Operational implication: the data race is not "Polymarket vs Liquipedia" generically. It is game-specific:

- LoL: official Riot/live-feed path, `gol.gg` ingestion, stream delay, and CLOB participant delay.
- CS2: HLTV / stream / server scoreboard timing.
- Dota 2: match API / stream / third-party scoreboard timing.
- Valorant: VLR and official broadcast timing.
- Lower-liquidity games: source may be available, but market depth and refresh cadence are usually the bottleneck.

## Feasibility Ranking

| rank | game / market class | why it fits | main blocker |
|---:|---|---|---|
| 1 | LoL live kill/objective props | Many near-term submarkets, high historical notional, explicit kill/objective props, `gol.gg` source path. | Need faster-than-market kill feed; public stream delay may be too slow. |
| 2 | Dota 2 live kill/objective props | Similar intragame trigger shape to LoL; current DreamLeague volume is large. | Fewer events than LoL, source/timing path must be benchmarked. |
| 3 | LoL / CS2 game winner | Very high notional and fast fill cadence. | Resolves at map/game end; edge window may be much smaller and crowded. |
| 4 | CS2 odd/even kills / rounds | Lots of current submarkets and liquidity, easy state variable. | Odd/even does not resolve at threshold crossing; it resolves at map end, so less clean than O/U threshold props. |
| 5 | Valorant map / match markets | Meaningful but smaller volume. | Mostly winner/handicap/totals; fewer intragame snap-to-one markets. |
| 6 | Rocket League / MLBB / SC2 / HoK / R6 / Overwatch | Searchable and worth keeping in the grid. | Current liquidity and prop density are thin; useful mostly as low-cost experiments. |
| watchlist | Call of Duty, Wild Rift, PUBG, Brood War | Gamma series exist. | No near-term active markets as of this pull. |

## What To Grid Search

Use the notebook to materialise these from local parquet:

1. **Game x market-type fill density:** raw fills, distinct markets, notional, and average fill size.
2. **Intragame prop candidate set:** `kill_over_under_game`, objective props, first blood, game winner, map winner.
3. **Next-fill latency proxy:** for each `(game, market_type, outcome_token_id)`, compute gap percentiles and fraction with another fill inside 5s / 30s / 5m.
4. **Closed-position outcome behaviour:** join candidate market IDs to `closed_positions.parquet` to see realised PnL distribution, held-to-resolution share, and whether historical markets actually resolved cleanly.
5. **External-source timestamp benchmark:** manually record event time from stream/API/source vs earliest Polymarket fill / price snap. This is the only piece the current parquet cannot infer by itself.

Suggested first-pass filters:

```sql
game IN ('League of Legends', 'Dota 2', 'Counter Strike 2')
AND market_type_proxy IN ('live_total_kills', 'objective_or_multikill_prop', 'game_winner')
AND raw_notional_usd > 10000
AND pct_gap_le_30s > 0.50
```

Then split the result by whether the trigger is threshold-based (`kill_over_under_game`) or terminal (`game_winner`, `series_moneyline`, `total_games`). Threshold-based markets are the clean latency-arb candidates because the truth value changes during the game.

## Near-Term Event Slugs

These are event-level slugs with at least one sub-market ending on or after 2026-05-19 00:00 UTC. The companion notebook fetches the full current list and all sub-market slugs directly from Gamma.

### Counter Strike 2

`cs2-2007-lilmix-2026-05-21`, `cs2-3dmax-mgc-2026-05-27`, `cs2-3dmax-mibr-2026-05-20`, `cs2-9daplu-alz-2026-05-18`, `cs2-9z-shk-2026-05-27`, `cs2-algo1-eac-2026-05-20`, `cs2-algo1-inf1-2026-05-20`, `cs2-all-wal2-2026-05-22`, `cs2-ast-cyb-2026-05-19`, `cs2-atreid-ence-2026-05-23`, `cs2-b8-nip-2026-05-20`, `cs2-b8-tyloo-2026-06-02`, `cs2-bb3-gg5-2026-06-02`, `cs2-bg1-ff-2026-05-14`, `cs2-bhe-gls1-2026-05-21`, `cs2-big5-tl1-2026-06-02`, `cs2-brawls-clutch1-2026-05-18`, `cs2-brawls-pure-2026-05-19`, `cs2-bsta-reda-2026-05-19`, `cs2-bw-ep-2026-05-18`, `cs2-don-ep-2026-05-19`, `cs2-don-fal-2026-05-18`, `cs2-esb-vortex-2026-05-19`, `cs2-fal2-bcg-2026-05-20`, `cs2-faze-all-2026-05-27`, `cs2-fnc-fal-2026-05-19`, `cs2-fokus-oxuji-2026-05-19`, `cs2-ge2-auryb-2026-05-19`, `cs2-gl1-nrg-2026-06-02`, `cs2-hero-nip-2026-05-27`, `cs2-hero-shk-2026-06-02`, `cs2-jjh-fpr-2026-05-23`, `cs2-keyd-isg-2026-05-22`, `cs2-kinoa-biga-2026-05-20`, `cs2-kinoa-pre-2026-05-20`, `cs2-krespo-paina-2026-05-19`, `cs2-ldp-yaw-2026-05-22`, `cs2-lgc-nrg-2026-05-19`, `cs2-lph-mana-2026-05-20`, `cs2-m80-lvg-2026-06-02`, `cs2-melill-haven-2026-05-19`, `cs2-mglz-lvg-2026-05-20`, `cs2-mibr-thunde-2026-06-02`, `cs2-mibra-odka-2026-05-18`, `cs2-mibra-pcy-2026-05-19`, `cs2-mouz-tyloo-2026-05-19`, `cs2-msc-projec-2026-05-19`, `cs2-mw-furiaf-2026-05-18`, `cs2-newvis-b8acad-2026-05-19`, `cs2-old1-newvis-2026-05-19`, `cs2-pain-m80-2026-05-20`, `cs2-paina-ge3-2026-05-19`, `cs2-prv-tl1-2026-05-20`, `cs2-pure-bojong-2026-05-19`, `cs2-quin-vexa-2026-05-18`, `cs2-r2-mw-2026-05-19`, `cs2-rt1-enr1-2026-05-19`, `cs2-rt1-mana-2026-05-19`, `cs2-run2-bw-2026-05-21`, `cs2-sashi-eye-2026-05-22`, `cs2-sin2-fly-2026-06-02`, `cs2-sng-inf1-2026-05-19`, `cs2-tlr-eac-2026-05-23`, `cs2-unity-bmb-2026-05-23`, `cs2-ursa-algo1-2026-05-19`, `cs2-wraith-algo1-2026-05-19`, `cs2-zerote-co-2026-05-21`

### Dota 2

`dota2-bb4-vg-2026-05-20`, `dota2-flc-ts8-2026-05-21`, `dota2-flc-tundra-2026-05-19`, `dota2-liquid-xtreme-2026-05-20`, `dota2-miposh-bot-2026-05-19`, `dota2-navi-aur1-2026-05-19`, `dota2-ns2-nix-2026-05-19`, `dota2-ns2-recren-2026-05-19`, `dota2-pari-liquid-2026-05-19`, `dota2-st-bot-2026-05-19`, `dota2-tt1-miposh-2026-05-18`, `dota2-tt1-st-2026-05-19`, `dota2-tundra-vp-2026-05-20`

### League of Legends

`lol-ban2-sec-2026-05-19`, `lol-bar-hrts-2026-05-20`, `lol-bce-dv1-2026-05-21`, `lol-bro1-t1a-2026-05-22`, `lol-bro2-t1-2026-05-24`, `lol-c9-fly-2026-05-23`, `lol-ccg-wu-2026-05-20`, `lol-dfma-rck-2026-05-22`, `lol-dk-bro2-2026-05-21`, `lol-dk-fox1-2026-05-23`, `lol-dk-t1-2026-05-25`, `lol-dnf-drx-2026-05-22`, `lol-dnf-gen-2026-05-24`, `lol-dnsc-hle-2026-05-21`, `lol-doc-dv1-2026-05-19`, `lol-drxc-dkc-2026-05-21`, `lol-fec-dyn1-2026-05-19`, `lol-fox1-hle1-2026-05-21`, `lol-fox1-ns-2026-05-19`, `lol-fsk-doc-2026-05-21`, `lol-g2nord-use1-2026-05-21`, `lol-glr-lds-2026-05-20`, `lol-gxp-flk-2026-05-21`, `lol-hle1-bro2-2026-05-19`, `lol-ig1-tt-2026-05-23`, `lol-jdg-al-2026-05-21`, `lol-kbm-vksa-2026-05-26`, `lol-kc-g2-2026-05-24`, `lol-kt-gen-2026-05-22`, `lol-ktc-foxy-2026-05-22`, `lol-lguide-arb1-2026-05-22`, `lol-lua-ub-2026-05-20`, `lol-ly-tl2-2026-05-24`, `lol-me1-ozo-2026-05-21`, `lol-mvk-cfo-2026-05-23`, `lol-mvu-cnv-2026-05-20`, `lol-newmet-rg4-2026-05-21`, `lol-nip-edg-2026-05-24`, `lol-ns-hle1-2026-05-23`, `lol-ns-hle1-2026-05-25`, `lol-ns-kt-2026-05-20`, `lol-nsea-genga-2026-05-21`, `lol-oa-bombat-2026-05-20`, `lol-pcific-bw-2026-05-20`, `lol-pnga-itz-2026-05-19`, `lol-red-fur-2026-05-24`, `lol-shg-dcg-2026-05-24`, `lol-sly-gal-2026-05-20`, `lol-su-bgt-2026-05-19`, `lol-t1-drx-2026-05-20`, `lol-tlnpir-kcb-2026-05-20`, `lol-tog-ewi-2026-05-20`, `lol-tsw-gam-2026-05-20`, `lol-ucam1-koia-2026-05-21`, `lol-uwinks-fnl-2026-05-21`, `lol-vit-mkoi-2026-05-23`, `lol-vks-los-2026-05-23`, `lol-wb-blg-2026-05-21`, `lol-wb-lgd-2026-05-24`, `lol-we-lng-2026-05-23`

### Valorant

`val-fks1-rbn-2026-05-19`, `val-fnc1-bbl1-2026-05-22`, `val-g2-bar2-2026-05-20`, `val-gxgc-tmo-2026-05-20`, `val-kru-krspar-2026-05-19`, `val-lev1-g21-2026-05-22`, `val-mibra-fura-2026-05-19`, `val-mir-wip-2026-05-19`, `val-navij-cgn-2026-05-19`, `val-nrg-100t1-2026-05-22`, `val-nv2-sen-2026-05-19`, `val-ox-bst1-2026-05-19`, `val-rrq1-drx1-2026-05-22`, `val-tbk1-f4t-2026-05-19`, `val-tl1-gm-2026-05-22`, `val-tla-bar-2026-05-19`, `val-ts-peek-2026-05-19`

### Mobile Legends: Bang Bang

`mlbb-ae-rrq-2026-05-23`, `mlbb-af-yndx-2026-05-26`, `mlbb-btr1-dewa-2026-05-22`, `mlbb-btr1-onic-2026-05-23`, `mlbb-ch-ts-2026-05-26`, `mlbb-darkph-aterio-2026-05-25`, `mlbb-forzee-vp-2026-05-26`, `mlbb-geek-tlid-2026-05-23`, `mlbb-hardgr-omnix-2026-05-25`, `mlbb-navi-geek-2026-05-24`, `mlbb-onic-evos-2026-05-24`, `mlbb-rrq-dewa-2026-05-24`, `mlbb-sup1-tst-2026-05-25`, `mlbb-thewor-tsaw-2026-05-25`, `mlbb-tlid-ae-2026-05-22`, `mlbb-ver-mag-2026-05-26`

### Rocket League

`rl-kc-fut-2026-05-20`, `rl-m8-furia-2026-05-20`, `rl-mce-5f-2026-05-20`, `rl-mce-mibr-2026-05-20`, `rl-nip-tsm1-2026-05-20`, `rl-nrg-5f-2026-05-20`, `rl-nrg-mibr-2026-05-20`, `rl-ssg-r8-2026-05-20`, `rl-twis-sr-2026-05-20`, `rl-vit-kc-2026-05-20`, `rl-vit-wc-2026-05-20`, `rl-wc-fut-2026-05-20`

### Honor of Kings

`hok-agsp-wol-2026-05-23`

### StarCraft 2

`sc2-classi-shin-2026-05-23`, `sc2-clem-lambo-2026-05-24`, `sc2-rogue-bunny-2026-05-23`, `sc2-solar-hero-2026-05-24`

## Gamma Pull Reference

Use event endpoints for sub-market metadata:

```bash
curl -s "https://gamma-api.polymarket.com/events/slug/lol-t1-ns-2026-05-13" | python3 -m json.tool
curl -s "https://gamma-api.polymarket.com/events?series_id=10311&active=true&closed=false&limit=100&offset=0" | python3 -m json.tool
```

Use the local parquet for fills:

```sql
SELECT *
FROM read_parquet('data/trades/trades_delta_shard*.parquet')
UNION ALL BY NAME
SELECT *
FROM read_parquet('data/trades/trades_seed.parquet');
```

Do not rebuild the full research stack for this analysis. Start with small Gamma metadata pulls, classify market IDs, then scan the raw fill parquet only for candidate esports markets.
