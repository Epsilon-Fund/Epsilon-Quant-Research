---
title: "Copytrade attribution repartition — exchange-internal active legs, style-ratio reclassification, and the Block B TFI emit-path re-run"
tags: [copytrade, dali, block-b, attribution, active-order-leg, results]
created: 2026-06-10
status: complete
owner: justin
project: polymarket
para: resource
hubs:
  - COWORK
  - POLYMARKET_BRAIN
---

# Copytrade attribution repartition — who was actually the aggressor on each fill, and what that changes (style labels yes, PnL no)

> Hub: [[COWORK]]

> Table terms: [[polymarket_table_dictionary]]

Backlinks: [[copytrade_relayer_implications]] · [[block_b_reinterpretation]] · [[block_b_findings]] · [[relayer_dig_findings]] · [[POLYMARKET_BRAIN]]

## Plain-English Summary

- Polymarket's exchange contract, when it matches two resting orders against an incoming aggressive order (`_matchOrders`), emits one extra "internal leg" event in which the aggressive trader's wallet sits in the `maker` column and the exchange contract itself sits in the `taker` column. Our raw fill data therefore labels some genuinely AGGRESSIVE trades as "maker" fills. **PnL and position attribution are unaffected — this is style framing only** — but maker/taker style ratios and any trade-flow signal built on maker-side signs are contaminated.
- This note delivers the two cheap pre-registered fixes: (1) an `active_order_leg` flag in the SQL view layer plus a recompute of the audited copytrade leaders' maker:taker style ratios, and (2) a re-run of the Block B trade-flow-imbalance (TFI) hit rates with the historical fills partitioned by emit path (internal-leg vs everything else), all from cached local data.
- Style result: the pre-registered anchors reproduce exactly (Domah 7.89 → 5.67, still maker-heavy; leader_top_leaderboard 5.15 → 3.46, flips maker_heavy → mixed). One additional leader flips (leader_dthreed8b71, mixed → taker_heavy). Domah's `macro / maker / 18-24` smoke cell survives reclassification.
- TFI result: the sign-semantics check confirms that on internal-leg rows `maker_side` is the aggressor's own side, so `historical_to_aggressor()` (which inverts `maker_side`) was sign-inverted on that subset. Verdict: the equity_index 58.8% number reproduces exactly and survives as a real, correctly-measured property of the single-sided population (the per-market paired control rules out market composition) — but the lift is an **attribution correction**, not evidence that a cleaner trader population trades more predictively: in equity the MM-bot part of the operator filter contributed 0.0pp, and in crypto the entire lift was the sign convention. Cite it as "single-sided-partition TFI with the internal-leg artifact removed". No Block B trading decision changes.

## Why this note exists

[[block_b_reinterpretation]] left three follow-ups: add an `active_order_leg` role flag to the view layer, sanity-check `historical_to_aggressor()` on the `_matchOrders` subset, and re-run Block B's TFI hit rates per emit-path partition. [[copytrade_relayer_implications]] had already established the mechanism (CTF Exchange `Trading.sol` `_matchOrders` emits the internal active leg with `maker = takerOrder.maker`, `taker = address(this)`) and verified it continues on the v2 contracts post 2026-04-28. This note executes those follow-ups. The mechanism is taken as established; it is only verified here, not re-derived.

The four exchange-internal-leg contract addresses (source of truth: `data_infra/operator_denylist.py` `EXCHANGE_INTERNAL_LEG`):

| address | version |
|---|---|
| `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` | CTF Exchange v1 standard |
| `0xc5d563a36ae78145c45a50134d48a1215220f80a` | CTF Exchange v1 neg-risk |
| `0xe111180000d2663c0091e4f400237545b87b996b` | CTF Exchange v2 standard (post 2026-04-28) |
| `0xe2222d279d744050d28e00520010520000310f59` | CTF Exchange v2 neg-risk (post 2026-04-28) |

## A practical example — one Domah fill

Suppose Domah submits an aggressive buy of 10,000 YES tokens at $0.60 and the exchange crosses it against two resting sell orders. The chain emits three `OrderFilled` events: two normal legs (`maker` = each resting seller, `taker` = Domah) and one internal leg (`maker` = Domah, `taker` = the exchange contract, $6,000 notional). Before this change, our `trader_actions` view counted that third row as a Domah **maker** fill, inflating his maker:taker style ratio. With the new flag, that row gets `active_order_leg = TRUE` on Domah's maker-role row: his economics are identical (he still paid $6,000 for 10,000 tokens — `token_delta`/`usd_delta` never depended on the label), but style metrics can now move that fill to the active side, and signal work can stop double-counting the bundle.

---

# Sub-task 1 — `active_order_leg` flag and style-ratio recompute

## Design

**View change** (`sql/views.sql`, loaded by `data_infra/views.py`):

- `joined_fills` gains `exchange_internal_match` (BOOLEAN): TRUE iff the fill's `taker` is one of the 4 exchange-internal-leg contracts.
- `trader_actions` gains `active_order_leg` (BOOLEAN) = "this row's address signed the ACTIVE (aggressor) order of the fill":
  - **maker-role row:** TRUE iff `exchange_internal_match` (the wallet labelled maker was the aggressor); FALSE = genuine passive maker.
  - **taker-role row:** TRUE for normal fills (the taker is the aggressor by definition); FALSE only when the row's address IS the exchange contract — an artifact leg, not a trader, to be excluded from per-trader style metrics (those addresses are already in the operator denylist).

**Migration note:** views are `CREATE OR REPLACE`, loaded fresh by `data_infra/views.py::load_views()` at every connection — no stored data migrates; every consumer of `trader_actions`/`joined_fills` gets the new column on its next load. The 4 addresses are hardcoded in the SQL with a comment marking `data_infra/operator_denylist.py` (`EXCHANGE_INTERNAL_LEG`) as the source of truth that must be kept in sync. Column additions are backward-compatible for all existing consumers — `build_closed_positions.py`, `build_traders_table.py` and every script checked reference the views by named columns, none by position. `scripts/smoke_test_views.py` on the updated views: Domah stats and the per-transaction sign-symmetry invariant pass cleanly; the row-count band check (`joined_fills ≥ 95% of raw_trades`) reports 91.6% — a **pre-existing data-freshness issue, not caused by this change** (the change adds SELECT-list columns only; no WHERE/JOIN touched, so row counts are mathematically identical). Cause: the markets snapshot is `markets_2026-05-06.parquet` while trade shards now extend to 2026-05-26, so fills on markets created after the snapshot fail the `markets_tokens` join. Fix is a Gamma markets re-snapshot, out of scope here. Separately, a bounded one-day smoke slice of `trader_actions` (2026-04-20) verified the flag's internal consistency: 2,457,702 maker-role rows with `active_order_leg = TRUE` exactly equal the taker-role artifact rows, and every artifact row's address is an exchange contract.

**Style recompute** (`scripts/copytrade_attribution_repartition.py`): for each audited leader, reclassify internal-leg maker fills to the active/taker side of the style ratio:

`new_ratio = (maker_fills − internal_legs) / (taker_fills + internal_legs)`

Style categories match `scripts/copytrade_relayer_implications.py`: ratio > 4 → `maker_heavy`, ratio < 0.25 → `taker_heavy`, else `mixed`.

Two variants are reported so the pre-registered anchors are reproducible AND the current data is reflected:

- **anchor** — stored `traders.parquet` style counts + the v1-only internal counts from the 2026-05-28 relayer-implications run (this is the exact definition behind the "7.89 → 5.67" numbers in [[copytrade_relayer_implications]] Step 3/5).
- **fresh** — a single streaming DuckDB aggregate over the current `joined_fills` with the new `exchange_internal_match` flag (all 4 contracts, v1+v2; current shards extend ~1 month past the `traders.parquet` build snapshot).

## Results — leader classification table

Audited-leader set: Domah + the seven leaders audited under `data/analysis/` (`domah_followups/leader_ee00ba`, `leader_dthreed8b71_investigation`, `leader_high_conviction`, `leader_negrisk_directional_1/2`, `leader_top_leaderboard`, `leader_ultra_maker`) + the remaining Step-3 manual candidate from [[copytrade_relayer_implications]] (`0x17db3f…`). Full table: `data/analysis/csv_outputs/copytrade/copytrade_style_reclass_leaders.csv`.

| leader | address | old ratio | new ratio (anchor) | new ratio (fresh) | old category | new category | internal share of maker fills |
|---|---|---:|---:|---:|---|---|---:|
| leader_ultra_maker | `0x2005d1…75ea` | 9.08 | 5.18 | 5.51 | maker_heavy | maker_heavy | 6.4% |
| leader_negrisk_directional_2 | `0x5bffcf…ffbe` | 8.98 | 7.18 | 6.99 | maker_heavy | maker_heavy | 2.5% |
| **Domah** | `0x9d84ce…1344` | **7.89** | **5.67** | 5.66 | maker_heavy | **maker_heavy** | 4.2% |
| **leader_top_leaderboard** | `0x6a72f6…33ee` | **5.15** | **3.46** | 3.40 | maker_heavy | **mixed (FLIP)** | 7.4% |
| leader_ee00ba | `0xee00ba…cea1` | 3.82 | 2.49 | 2.49 | mixed | mixed | 9.9% |
| leader_negrisk_directional_1 | `0x629bc4…995a` | 3.57 | 2.65 | 2.65 | mixed | mixed | 6.9% |
| step-3 candidate | `0x17db3f…5f6d` | 1.45 | 1.17 | 1.17 | mixed | mixed | 8.9% |
| leader_high_conviction | `0x204f72…5e14` | 1.32 | 0.31 | 0.36 | mixed | mixed | 54.6% |
| **leader_dthreed8b71** | `0xd38b71…5029` | **0.28** | **0.15** | 0.15 | mixed | **taker_heavy (FLIP)** | 39.8% |

Column glossary: **old ratio** = `style_maker_taker_ratio` stored in `traders.parquet` (maker fills / taker fills over all of `joined_fills`); **new ratio (anchor)** = reclassified ratio using the stored counts + v1-only internal counts from the 2026-05-28 run; **new ratio (fresh)** = reclassified ratio recomputed today from `joined_fills` with all 4 contracts; **new category** is assigned on the fresh ratio; **internal share of maker fills** = fraction of the leader's maker-column fills that are actually `_matchOrders` active legs (fresh).

**Anchor check: exact reproduction.** Domah 7.89 → 5.67 and `0x6a72…` 5.15 → 3.46 match the pre-registered anchors to the second decimal. The fresh variant differs only in the third decimal-ish range (e.g., Domah 5.656 vs 5.674) for two definitional reasons, both verified: (a) the anchor's internal counts came from `raw_trades` (includes orphan fills that `joined_fills` drops) and were v1-only; (b) the fresh scan includes ~1 month of newer shards plus v2-contract internal legs. Neither moves any leader across a category boundary that the anchor variant doesn't also cross.

**Read:** two of nine audited leaders change style category. `leader_top_leaderboard` (the cohort's biggest PnL, $15.0M) loses its maker_heavy label — its "maker-ness" was partly an artifact of aggressive orders being labelled maker. `leader_dthreed8b71` was already an HFT-profile oddity and becomes formally taker_heavy. Domah, the ultra-maker, and negrisk_directional_2 stay maker_heavy with room to spare. `leader_high_conviction` is striking: 54.6% of its maker-column fills are actually aggressive internal legs, so its true style is strongly taker-leaning (ratio 0.36) — anyone reading its old 1.32 ratio as "balanced" was misled. **PnL, win rates, volumes and every cohort PnL metric are unchanged by construction — only the style label moves.**

## Does anything downstream depend on the maker-biased label?

- **Domah `macro / maker / 18-24` smoke cell:** recomputed below — survives. See next subsection.
- **TODO § copytrade smoke-target re-ranking** excludes on `split_position_signature > 60%` — that metric is built from position fragmentation, not style ratio: **unaffected**.
- **Cohort screens** (grep of `scripts/build_cohorts.py`, `build_copyability_metrics.py`, `cross_leader_synthesis_v2.py`): only one of the six cohort pools keys off a style-role metric — `patient_accumulators` requires `style_role_balance > 0.7` (maker share of fills). That screen is maker-biased in the same way and would lose borderline members under reclassification; none of our audited leaders enter or exit it (it also requires holding > 168h). `cross_leader_synthesis_v2` uses `style_role_balance` as a correlation anchor (ρ −0.64 claim) — diagnostic only, flagged for re-read but not load-bearing for any gate.
- **`is_operator_like` heuristic** uses extreme ratio thresholds (> 50, < 0.02) computed on the biased counts. Reclassification only moves ratios *down* (toward taker), so no genuine trader gains the > 50 flag; a pure-MM bot near the boundary could in principle drop below it, but the hardcoded denylist (not the heuristic) carries the known bots. No action needed.

## Domah role-slice recompute — the maker sub-cell survives

Method: the Domah copy-audit (`scripts/domah_copy_audit.py`) slices positions by the role of the position's **first fill**. We recomputed that slice from the cached audit artifacts (`domah_audit_fragments.parquet`, 170,005 fills; `domah_audit_positions.parquet`, 5,890 positions — same position-replay outputs as the original audit, no new simulation) with the reclassified role: a maker-role fill whose fill-level taker is an exchange contract counts as **active** (taker side). Only 4,319 of Domah's 147,123 maker-role audit fills (2.94%, $9.06M notional) reclassify; inside the macro/18-24 smoke cell it is just 104 of 5,903 maker-role fills (1.76%). Full table: `data/analysis/csv_outputs/copytrade/copytrade_domah_role_slice_reclass.csv`.

| scope | variant | slice | n positions | leader PnL | A_real PnL | C_real PnL |
|---|---|---|---:|---:|---:|---:|
| all families | old role | maker | 4,609 | $703,061 | **$95,602** | **$258,279** |
| all families | reclassified | maker | 3,998 | $529,739 | **$125,942** | **$256,115** |
| all families | old role | taker | 1,281 | $238,061 | −$93,468 | −$24,815 |
| all families | reclassified | taker | 1,892 | $411,383 | −$123,808 | −$22,652 |
| macro | old role | maker | 372 | $233,734 | $156,990 | $196,594 |
| macro | reclassified | maker | 341 | $158,120 | $128,243 | $166,635 |
| macro/18-24 | old role | maker | 94 | $179,524 | $181,090 | $188,943 |
| macro/18-24 | reclassified | maker | 85 | $118,165 | $147,779 | $156,396 |

Column glossary: **slice** = role of the position's first fill (the audit's original attribution rule); **leader PnL** = Domah's own realised PnL on those positions; **A_real PnL** = the role-mirrored copy strategy under the realistic fill model; **C_real PnL** = the pure-maker copy strategy under the realistic fill model (the two deployable branches from the audit; see [[copytrade_relayer_implications]] Step 5). The CSV carries all five branches (A_opt/A_real/B/C_opt/C_real) and the taker-side rows for every scope.

**Read:** the maker sub-cell does not just survive — the all-families maker slice's A_real PnL *improves* ($95.6k → $125.9k) once the ~600 positions that actually began with an aggressive internal-leg fill are moved to the taker slice (where copy-PnL was always negative, and gets more negative: −$93k → −$124k). C_real is essentially unchanged. Within the smoke cell (macro, 18-24h UTC) the maker slice keeps positions 94 → 85 and stays strongly positive on both deployable branches (A_real $181k → $148k, C_real $189k → $156k — the modest decline is 9 positions leaving the slice, not sign damage). **Verdict: cell intact, green light unchanged.** The reclassification actually sharpens the cell's story: the copyable part of Domah is his genuinely-passive maker behaviour; his aggressive legs are the uncopyable part.

---

# Sub-task 2 — Block B TFI hit-rate re-run partitioned by emit path

## Design

Same cached inputs, same methodology as the original Block B deep dive (`scripts/dali_tfi_deep_dive.py`, results in [[block_b_findings]]): market-second bars of maker-side signed flow, lookahead-free forward returns via as-of joins (timestamp-filtered before aggregating), min |signal| $25, future gap ≤ 300s, last 600s before market end excluded, magnitude deciles fit per family×horizon, hit rate = share of bars where the signed flow predicted the forward VWAP move. The headline cell is the top-decile magnitude bucket at the 300s horizon under the `inverse_maker_side` convention — that is where the original 58.8% (equity_index) and 52.0% (crypto) operator-filtered numbers live. All from cached fills parquets (`dali_tfi_*_fills.parquet`, 0.16–1.64M rows per family, all pre-v2-cutover); no new capture, no Goldsky backfill. New script: `scripts/copytrade_tfi_emit_path_partition.py`.

Four populations per family:

| population | definition | role in this note |
|---|---|---|
| `all_fills_recon` | the cached mixed-population eval (original "raw" variant) | reconciliation row — must reproduce Block B |
| `operator_removed_recon` | drop fills touching any `OPERATOR_ADDRESSES` (original "operator-filtered" variant) | reconciliation row — must reproduce 58.8% / 52.0% |
| `match_orders_leg` | fills whose `taker` ∈ the 4 exchange contracts | pre-registered partition: the `_matchOrders` internal active legs |
| `single_sided` | all other fills | pre-registered partition: conventional maker/taker semantics |

Note one nuance the partition names inherit from the pre-registration: `single_sided` means "not an internal-leg row". The normal maker legs of a `_matchOrders` bundle (real wallet on both sides) land in `single_sided` too — the sign check below makes this explicit. `operator_removed` ≈ `single_sided` minus rows touching the denylisted MM bots/HFT addresses.

## `historical_to_aggressor()` sanity check on the match_orders partition — INVERTED, as suspected

`historical_to_aggressor()` (`lib/trade_sign_normalization.py`) maps the fill's `maker_side` to the aggressor's token side by **inverting** it, on the premise that `maker_side` is the passive maker's side. On internal-leg rows the maker column holds the AGGRESSOR, so `maker_side` is the aggressor's own side and inverting it flips the sign. Empirical check (per family, within `(transaction_hash, market_id)` bundles that contain an internal leg): the internal leg's `maker_side` should be the inverse of its sibling normal legs' sides, and its `usd_amount` should equal the sum of the sibling legs (it is the whole taker order, re-emitted once).

Per-family results over every checkable bundle (exactly 1 internal leg + ≥1 sibling normal legs in the same `(transaction_hash, market_id)`); full CSV: `copytrade_tfi_match_orders_sign_check.csv`:

| family | internal-leg rows | checkable bundles | % all sibling sides inverse | median USD ratio (siblings ÷ internal) | % USD ratio within 2% of 1 |
|---|---:|---:|---:|---:|---:|
| crypto | 565,540 | 565,289 | 16.9% | 1.00 | 19.3% |
| equity_index | 65,817 | 65,793 | 36.0% | 1.00 | 37.5% |
| ai_product | 136,159 | 136,103 | 36.3% | 1.00 | 36.9% |
| sports | 120,793 | 120,747 | 24.4% | 1.00 | 26.0% |

At first glance the side-inversion percentages look too low for a clean confirmation. They are not — the raw check pools two mechanically different bundle types. A follow-up diagnostic (inline extension of the same `sign_check()` bundle loop, splitting bundles by whether the sibling legs are in the **same outcome token** as the internal leg or in the **complementary token** — a `_matchOrders` mint/merge, where a buy of YES is crossed against resting buys of NO) resolves it completely:

| family | bundle type | n bundles | % of bundles | % sides inverse | % sides same | % USD ratio within 2% of 1 | median USD ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| crypto | same-token | 95,651 | 16.9% | **100.0** | 0.0 | **99.97** | 1.00 |
| crypto | cross-token (mint/merge) | 427,466 | 75.6% | 0.0 | **100.0** | 2.6 | 0.79 |
| crypto | mixed | 42,172 | 7.5% | 0.0 | 0.0 | 5.5 | 1.00 |
| equity_index | same-token | 23,705 | 36.0% | **100.0** | 0.0 | **99.99** | 1.00 |
| equity_index | cross-token (mint/merge) | 34,433 | 52.3% | 0.0 | **100.0** | 1.6 | 0.67 |
| equity_index | mixed | 7,655 | 11.6% | 0.0 | 0.0 | 5.8 | 1.02 |
| ai_product | same-token | 49,470 | 36.4% | **100.0** | 0.0 | **99.97** | 1.00 |
| ai_product | cross-token (mint/merge) | 66,407 | 48.8% | 0.0 | **100.0** | 0.4 | 0.27 |
| ai_product | mixed | 20,226 | 14.9% | 0.0 | 0.0 | 2.8 | 1.10 |
| sports | same-token | 29,428 | 24.4% | **100.0** | 0.0 | **99.97** | 1.00 |
| sports | cross-token (mint/merge) | 83,120 | 68.8% | 0.0 | **100.0** | 2.0 | 0.72 |
| sports | mixed | 8,199 | 6.8% | 0.0 | 0.0 | 4.4 | 0.88 |

Column glossary: **same-token** = every sibling leg's outcome token (the fill's non-USDC asset id) equals the internal leg's token — a conventional cross within one order book; **cross-token** = every sibling is in the complementary token — a mint/merge match; **mixed** = siblings in both tokens. **% sides inverse / same** = share of bundles where ALL sibling `maker_side` values are the opposite of / equal to the internal leg's `maker_side`.

**Read: confirmed, with the dilution fully explained.** On same-token bundles the prediction holds essentially perfectly — sibling sides 100.0% inverse and the internal leg's notional equal to the sum of its siblings (USD ratio 1.0, ≥99.97% within 2%): the internal leg is the whole taker order re-emitted once, with `maker_side` = the **aggressor's own side**. The headline `pct_sides_inverse` numbers (16.9–36.3%) are exactly the same-token bundle shares — the rest are mint/merge bundles where sides are mechanically *equal* (everyone is a buyer in a mint, a seller in a merge) and the sibling USD sum is `(1−p)/p` times the internal leg's notional, where `p` is the internal leg's price (the observed median ratios 0.27–0.79 back out internal-leg prices of ~0.56–0.79 — sensible mid-range prices). In both bundle types `maker_side` on the internal leg is the aggressor's own side of the aggressor's own token, so **`historical_to_aggressor()` (which inverts `maker_side`) is sign-inverted on the entire `match_orders_leg` partition — the old Block B signs were inverted on that subset.** Two downstream consequences: (a) within a second-bar, a same-token bundle's internal leg and siblings carry equal-and-opposite signed flow and **self-cancel**; (b) a cross-token bundle's internal leg is **uncancelled** (its siblings sit in the complementary token, partly with same-sign flow), so mixed-population bars are flooded with aggressor-signed flow that the `inverse_maker_side` convention misreads — crypto, with 75.6% cross-token bundles and a 34.5% internal-leg fill share, is where the mixed-population number gets destroyed (34.31%, i.e. 65.61% under `maker_side`). The recompute below therefore reports both conventions per partition and flags the corrected one.

The token split also surfaces one construction caveat that applies equally to the original Block B run and to this re-run: both outcome tokens of a market appear under one `market_id`, and bars aggregate raw per-fill `maker_side` flow across them (a BUY of Down is signed like a BUY of Up). This is inherited, affects every partition identically, and does not touch the artifact-vs-composition question — but a per-token bar reconstruction is the obvious cheap **never-run gate** if fill-only TFI is ever revisited.

## Results — per-partition TFI hit rates (top decile, 300s)

Each row scores its partition with the **corrected aggressor convention** for that emit path: `inverse_maker_side` everywhere except `match_orders_leg`, which uses `maker_side` (per the sign check above). Unit of observation: one market-second bar in the top signed-flow-magnitude decile (deciles fit per family × horizon), 300s horizon, min |signal| $25, future gap ≤ 300s, last 600s before market end excluded — identical filters to [[block_b_findings]]. **Hit %** = share of bars where the aggressor-signed flow predicted the direction of the forward VWAP move; **Wilson 95% CI** = binomial interval on that share; **mean fwd move** = mean signed forward move in cents under the same convention (pre-cost; positive = continuation profitable). Reconciliation rows reproduce the original Block B operator-filter table (block_b_findings § operator filter) to the fourth decimal in all four families.

| family | partition | convention | n bars | hit % | Wilson 95% CI | mean fwd move (¢) |
|---|---|---|---:|---:|---|---:|
| equity_index | all_fills_recon | inverse | 1,376 | 47.09 | [44.47, 49.73] | +5.14 |
| equity_index | operator_removed_recon | inverse | 1,290 | **58.76** | [56.05, 61.42] | +8.02 |
| equity_index | single_sided | inverse | 1,290 | **58.76** | [56.05, 61.42] | +8.02 |
| equity_index | match_orders_leg | **maker (corrected)** | 1,359 | **45.55** | [42.92, 48.20] | −13.27 |
| crypto | all_fills_recon | inverse | 1,262 | 34.31 | [31.74, 36.97] | −3.03 |
| crypto | operator_removed_recon | inverse | 819 | **52.01** | [48.59, 55.42] | +7.58 |
| crypto | single_sided | inverse | 861 | 52.61 | [49.27, 55.93] | +7.00 |
| crypto | match_orders_leg | **maker (corrected)** | 903 | **47.51** | [44.27, 50.77] | −6.57 |
| ai_product | all_fills_recon | inverse | 1,234 | 50.73 | [47.94, 53.51] | +5.62 |
| ai_product | operator_removed_recon | inverse | 1,042 | 53.07 | [50.04, 56.08] | +4.84 |
| ai_product | single_sided | inverse | 1,134 | 53.79 | [50.88, 56.68] | +5.85 |
| ai_product | match_orders_leg | **maker (corrected)** | 1,203 | 39.90 | [37.17, 42.70] | −12.29 |
| sports | all_fills_recon | inverse | 2,121 | 44.27 | [42.17, 46.39] | +0.49 |
| sports | operator_removed_recon | inverse | 1,327 | 44.54 | [41.88, 47.22] | +2.20 |
| sports | single_sided | inverse | 1,339 | 44.73 | [42.09, 47.41] | +2.32 |
| sports | match_orders_leg | **maker (corrected)** | 1,924 | 37.99 | [35.85, 40.18] | −2.83 |

**Read — four things to notice:**

1. **Reconciliation is exact.** `all_fills_recon` and `operator_removed_recon` reproduce the original Block B numbers (equity_index 47.09 → 58.76, crypto 34.31 → 52.01, ai_product 50.73 → 53.07, sports 44.27 → 44.54) to the fourth decimal. The pre-registered anchor (equity_index operator-filtered 58.8% @300s) is the 58.76 row.
2. **The "operator filter" was de facto an emit-path filter.** In equity_index, `single_sided` and `operator_removed_recon` are **identical — same 1,290 bars, same 58.76% — at every horizon (30s/120s/300s)**. The MM-bot/HFT portion of the denylist contributed exactly nothing to the equity lift; removing the internal legs is the whole effect. In the other families the MM-bot portion shifts cells by ≤ 92 bars and ≤ 0.7pp (e.g., crypto 861 → 819 bars, 52.61 → 52.01%).
3. **Internal-leg fills are a third to two-fifths of every family's fills.** match_orders_leg rows are 34.5% of crypto fills (565,540 of 1,641,436), 40.1% of equity_index (65,817 of 164,003), 35.8% of ai_product (136,159 of 380,287), 41.4% of sports (120,793 of 291,991). This is why mislabelling them matters: it is not a fringe population.
4. **Correctly-signed internal-leg aggressor flow REVERTS.** Under its true convention the match_orders_leg partition is below 50% in every family at every horizon (equity 41.3–45.6%, crypto 44.7–47.7%, ai 39.9–40.5%, sports 35.7–39.0%). Under the old Block B `inverse_maker_side` reading those same cells showed 50.1–57.2% at the 300s headline horizon (48.1–57.2% across all horizons) — i.e., the old convention read genuine reversion as apparent continuation. The crypto mixed-population 34.31% is the flip side of the same coin: 65.61% under `maker_side`, because uncancelled internal legs dominate the mixed bars' signed flow and carry aggressor-side semantics that the inverse convention misreads.

## Composition control — per-market paired comparison

The lift could in principle be **market composition**: maybe the bars that survive internal-leg removal simply live in different (more predictable) markets. Control: hold the market fixed and compare the two emit-path partitions *within* it. Unit of observation: one market with ≥ 30 qualifying bars in **both** partitions at the 300s horizon, all qualifying bars (no magnitude bucketing — per-market top-decile cells are too thin; this is therefore an all-bars control for a top-decile headline, stated as such). **gap** = (single_sided hit − match_orders_leg hit) within the same market, averaged unweighted across markets, with a 2,000-draw bootstrap CI over markets. Two scorings:

- `raw_inverse` — both partitions scored with the old Block B `inverse_maker_side` convention. If the single-sided advantage survives within markets, the lift is **not** market composition.
- `corrected` — each partition scored with its true aggressor convention (`single_sided` → inverse, `match_orders_leg` → maker). If the gap closes here, the partitions' difference was pure sign bookkeeping; if it persists, the internal-leg flow genuinely behaves differently.

| family | scoring | n paired markets | mean within-market gap (pp) | bootstrap 95% CI | single_sided hit % | match_orders hit % |
|---|---|---:|---:|---|---:|---:|
| equity_index | raw_inverse | 70 | **+3.00** | [+1.33, +4.92] | 53.06 | 50.05 |
| equity_index | corrected | 70 | **+6.83** | [+5.07, +8.50] | 53.06 | 46.23 |
| crypto | raw_inverse | 102 | **+7.58** | [+4.37, +10.68] | 52.49 | 44.91 |
| crypto | corrected | 102 | −2.03 | [−5.01, +0.84] | 52.49 | 54.52 |
| ai_product | raw_inverse | 38 | −1.58 | [−4.83, +1.68] | 48.27 | 49.84 |
| ai_product | corrected | 38 | +3.60 | [+1.32, +5.64] | 48.27 | 44.66 |
| sports | raw_inverse | 53 | −3.88 | [−7.85, +0.19] | 43.80 | 47.68 |
| sports | corrected | 53 | +5.19 | [+2.95, +7.53] | 43.80 | 38.61 |

**Read:** in the two families that actually showed an operator-filter lift, the `raw_inverse` within-market gap is positive with a CI excluding zero (equity +3.0pp, crypto +7.6pp) — **the lift survives with the market held fixed, so it is not market composition.** The `corrected` row then separates two different stories: in **equity_index** the gap *widens* to +6.8pp — beyond the sign fix, internal-leg aggressor flow genuinely behaves differently (reverts) inside the very same markets where single-sided flow continues. In **crypto** the corrected gap collapses to ≈ 0 (CI spans zero, point estimate −2.0pp): at the all-bars level, once each emit path is signed correctly the two paths carry the *same* continuation information (52.5% vs 54.5%) — the entire apparent single-sided advantage in crypto was the sign convention. ai_product and sports had little/no lift to explain (raw gaps ≈ 0 or negative, CIs spanning or touching zero), and their corrected gaps are positive for the same reversion reason as equity.

## Verdict — artifact or real?

**The equity-index "operator-filtered" 58.8% lift is a real, correctly-measured property of the single-sided fill population — it is NOT a market-composition artifact. But the original story attached to it ("remove bot traders → cleaner population → more predictive flow") is dead: the lift is an attribution correction, produced almost entirely by purging duplicate, oppositely-signed aggressor legs that the old sign convention misread.** Specifically:

1. **Reconciliation is exact.** The re-run reproduces the original Block B operator-filter table to the fourth decimal in all four families; the pre-registered 58.8% anchor is the 58.76% `operator_removed_recon` row (n = 1,290 bars).
2. **The filter was an emit-path filter wearing an operator costume.** In equity_index, `single_sided` ≡ `operator_removed` exactly (same bars, same hit rate, every horizon): the MM-bot/HFT part of the denylist contributed **0.0pp**; dropping the `_matchOrders` internal legs is the entire lift. In crypto the MM-bot part moves the cell by only −0.6pp (single_sided 52.61% → operator_removed 52.01%) of a 17.7pp total swing.
3. **Not composition.** The within-market paired control keeps the single-sided advantage with the market held fixed (equity +3.0pp [1.3, 4.9]; crypto +7.6pp [4.4, 10.7] under the original convention).
4. **The mechanism is confirmed, not assumed.** Internal legs are whole taker orders re-emitted with `maker` = the aggressor and `maker_side` = the aggressor's own side (same-token bundles: 100.0% side-inverse vs siblings, USD ratio 1.00). The old Block B signs **were inverted on that subset**. In the mixed population, same-token bundles self-cancel and cross-token (mint/merge) bundles inject uncancelled aggressor-signed flow that the inverse convention misreads — crypto, 75.6% cross-token, is the worst hit, which is exactly why its mixed-population number (34.31%) sat so far below 50% and "recovered" to 52.01% after filtering.
5. **What the 58.8% means now.** It survives as the correct measurement of single-sided (genuinely passive-maker-labelled) TFI continuation in equity_index at 300s/top-decile, and should be cited as **"single-sided-partition TFI with the internal-leg artifact removed"** — never as evidence about trader quality. Per-family: equity_index keeps a real single-sided continuation signature (58.76% [56.05, 61.42]); crypto's lift was pure sign bookkeeping (corrected paired gap ≈ 0); ai_product's small lift (50.7 → 53.1) has a raw paired gap CI spanning zero — treat as noise; sports never had a lift.
6. **A new diagnostic, not a new trade:** correctly-signed internal-leg aggressor flow *reverts* — top-decile hit below 50% in every family, with Wilson CIs entirely below 50% in equity_index, ai_product, and sports (crypto's upper bound touches 50.8%). Pre-cost, no L2 context, single capture window — this is a description of `_matchOrders` sweep behaviour, not a tradability claim. Block B's Outcome 3 ("no tradable fill-only TFI rule") is unchanged in both directions.

**Realism calibration check (what is fair vs harsh here):** sample sizes are stated on every claim (819–2,290 bars per headline cell; 38–102 paired markets); Wilson CIs on hit rates and bootstrap CIs on within-market gaps are reported throughout. No OOS split is claimed or needed — this is a *measurement audit* of cached single-capture-window data, not an edge claim, so all data is treated as in-sample and said so (CODEX realism rule 1). No borrowed baselines enter any gate. One deliberately harsh-avoidance call: crypto `match_orders_leg` corrected (47.51% [44.27, 50.77]) is reported as "no continuation, point estimate reverting", not "reversion", because its CI spans 50%.

---

## Assumption ledger (per [[CODEX]] § Realism calibration)

- **Modeled assumptions:** historical-fill bars with no L2 spread/depth (cost columns remain conservative tick proxies, exactly as in [[block_b_findings]]); the emit-path partition is identified purely by the 4 contract addresses in the `taker` slot (verified mechanism, [[copytrade_relayer_implications]], re-verified empirically by the same-token bundle check at 100.0% / USD ratio 1.00); second-bars aggregate raw per-fill `maker_side` flow across **both** outcome tokens of a market (inherited unchanged from the original Block B construction, surfaced by the cross-token diagnostic — affects every partition identically, so comparisons within this note are unaffected); style categories use the pre-existing 4 / 0.25 ratio thresholds.
- **Live-only unknowns:** none introduced here — both sub-tasks are reinterpretations of cached data. The live WS feed normalizes side semantics and never exposes the internal-leg artifact ([[block_b_reinterpretation]]), so nothing in this note changes live-capture plans.
- Nothing in this note is a tradability claim. Block B Headline #1 ("raw TFI is not tradeable in any family/horizon/condition combination") stands.

## Decision and next step

- **Style:** adopt `active_order_leg` for all future style claims; treat `style_maker_taker_ratio` in `traders.parquet` as maker-biased until the next `build_traders_table.py` rebuild optionally adds reclassified counts (documented in `docs/METRICS_REFERENCE.md`). Two leaders re-label (top_leaderboard → mixed, dthreed8b71 → taker_heavy); no smoke-target or cohort gate changes.
- **Domah smoke plan:** unchanged — the `macro / maker / 18-24` cell survives with the maker slice slightly *stronger* on A_real. Proceed per `brain/TODO.md` (parent session updates that file, not this note).
- **Block B:** the operator-filtered numbers should be cited as "single-sided-partition TFI with the internal-leg artifact removed", per the verdict above. No reopening of the Block B trading decision — the outcome remains OUTCOME 3 (no tradable rule; historical fills lack L2 context). If fill-only TFI is ever revisited, the cheap **never-run gates** are, in order: (1) per-token bar reconstruction (fixing the cross-token flow mixing surfaced by the sign-check diagnostic), then (2) a re-read of the internal-leg reversion diagnostic on those clean bars. Neither is scheduled.

## Artifacts

| artifact | path |
|---|---|
| View change | `polymarket/research/sql/views.sql` (`exchange_internal_match`, `active_order_leg`) |
| Sub-task 1 script | `polymarket/research/scripts/copytrade_attribution_repartition.py` |
| Sub-task 2 script | `polymarket/research/scripts/copytrade_tfi_emit_path_partition.py` |
| Leader style table | `data/analysis/csv_outputs/copytrade/copytrade_style_reclass_leaders.csv` |
| Domah role-slice table | `data/analysis/csv_outputs/copytrade/copytrade_domah_role_slice_reclass.csv` |
| TFI partition metrics (all buckets) | `data/analysis/csv_outputs/copytrade/copytrade_tfi_emit_path_partition.csv` |
| TFI headline (top decile @300s) | `data/analysis/csv_outputs/copytrade/copytrade_tfi_emit_path_headline.csv` |
| Sign check | `data/analysis/csv_outputs/copytrade/copytrade_tfi_match_orders_sign_check.csv` |
| Per-market paired control | `data/analysis/csv_outputs/copytrade/copytrade_tfi_emit_path_paired_by_market.csv` |
| Column docs | `polymarket/research/docs/METRICS_REFERENCE.md` §B.3/§B.4/§2.5 |
