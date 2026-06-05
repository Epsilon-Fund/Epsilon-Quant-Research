---
tags: [dali, block-a1, audit, methodology]
---
> Hub: [[COWORK]]


# Block A1 Methodology Audit

This note documents the current A1 replay and diagnostic implementation as inspected on 2026-05-28. It is an audit note only: no optimization or strategy decision is implied.

Primary files inspected:

- `polymarket/research/lib/clob_book.py`
- `polymarket/research/scripts/dali_clob_replay_features.py`
- `polymarket/research/scripts/dali_block_a1_replay_batch.py`
- `polymarket/research/scripts/dali_block_a1_analyze.py`
- `polymarket/research/scripts/dali_block_a11_segment_l2.py`
- `polymarket/research/tests/test_ofi_calculation.py`

A1.1 sidecar outputs added after the original A1 audit:

- `polymarket/research/data/analysis/csv_outputs/dali/block_a11_segment_surface.csv`
- `polymarket/research/data/analysis/csv_outputs/dali/block_a11_ofi_component_sweep.csv`
- `polymarket/research/data/analysis/block_a11_plots/`
- `polymarket/research/notes/dali/block_a11_plan_and_diagnostics.md`

## Confirmed Assumptions

- Current A1 uses **top-of-book OFI only**. Deeper book levels are maintained for book state and top-N stats, but OFI only compares previous and new best bid/best ask. See `bid_ofi`, `ask_ofi`, `ClobBook.replace`, and `ClobBook.update_level` in `lib/clob_book.py:34-73` and `lib/clob_book.py:106-140`.
- "TOB" means **top of book**. In this project, "TOB OFI" is the same as A1's canonical L1 CKS-style OFI: an event-flow measure from changes in best bid and best ask price/size. It is not the same as `tob_imbalance`, which is a standing-book level imbalance.
- `book` events are full snapshots. The first `book` anchors state and returns zero OFI because there is no prior anchored top-of-book. Later `book` events compare new top vs prior top. See `ClobBook.replace` in `lib/clob_book.py:106-122` and replay handling in `scripts/dali_clob_replay_features.py:301-314`.
- `price_change` events mutate executable book state and create the event-level OFI contribution. See `scripts/dali_clob_replay_features.py:340-362`.
- `best_bid_ask` events are telemetry only and never mutate executable book state. See the module docstring in `scripts/dali_clob_replay_features.py:1-9`, replay handling in `scripts/dali_clob_replay_features.py:316-323`, and regression test `test_best_bid_ask_does_not_mutate_executable_book` in `tests/test_ofi_calculation.py:55-93`.
- A1 groups per-market diagnostics by `(run_id, market_key)`, where `market_key` is `market_id` when present and otherwise `asset_id`. See `load_features` in `scripts/dali_block_a1_analyze.py:173-202` and `market_panel` in `scripts/dali_block_a1_analyze.py:466-470`.
- A1 treats live `last_trade_price.side` as aggressor-side token direction and writes `sign_status = live_reported_side_is_aggressor`. See `scripts/dali_block_a1_analyze.py:186-193` and `scripts/dali_block_a1_analyze.py:549`.
- YES/NO normalization is implemented with `direction_factor = +1` when `outcome_index == 0` and `-1` otherwise. It flips directional mid, OFI, and TFI into market-direction space. See `scripts/dali_block_a1_analyze.py:195-224`.
- `block_a1_decile_aggregate.csv` bins absolute depth-normalized OFI magnitude. Decile 10 means largest absolute normalized OFI observations, not most bullish observations. See `decile_aggregate` in `scripts/dali_block_a1_analyze.py:599-648`.
- A1.1 is a sidecar over `block_a1_features.parquet`, not a raw JSONL replay. It reuses A1's canonical TOB/L1 OFI plus top-5 book-state snapshots already present in the feature parquet. See `scripts/dali_block_a11_segment_l2.py:1-10`, `scripts/dali_block_a11_segment_l2.py:100-193`, and `scripts/dali_block_a11_segment_l2.py:208-266`.
- A1.1 tested five horizons: 1s, 5s, 10s, 30s, and 300s. It did not rerun the full 1/2/3/5/10/15/30/60/120/300 horizon surface from the A1 note. See `HORIZONS` in `scripts/dali_block_a11_segment_l2.py:33`.
- A1.1 component tests include combined OFI, bid-only OFI, ask-only OFI, combined OFI normalized by instantaneous touch depth, top-5 depth-pressure proxy, TOB imbalance level, and top-5 imbalance level. See `COMPONENTS` in `scripts/dali_block_a11_segment_l2.py:37-45`.

## Corrected Assumptions

- `future_return_bps` is **not always** `(future_mid - current_mid) / current_mid`. It is based on `directional_mid`, where NO-side tokens are transformed as `1 - mid`. Formula is in `scripts/dali_block_a1_analyze.py:215-224`.
- Per-market top-decile hit rate in `block_a1_results.csv` is not directly `sign(OFI) == sign(return)`. Per-market metrics first fit a univariate OLS model, then take the sign of the OLS prediction. See `ols_prediction`, `directional_hit`, and `metric_from_xy` in `scripts/dali_block_a1_analyze.py:243-283`.
- Aggregate decile hit rate in `block_a1_decile_aggregate.csv` **does** directly compare `sign(OFI_scaled)` against `sign(future_return_bps)`. See `scripts/dali_block_a1_analyze.py:631-644`.
- Public metric columns are not the raw calculation. Sparse/degenerate values are suppressed into `NaN`; corresponding `_raw` columns retain unsuppressed values. See `metric_reportable`, `public_value`, and row construction in `scripts/dali_block_a1_analyze.py:378-396` and `scripts/dali_block_a1_analyze.py:520-593`.
- A1's cost overlay is not executable entry/exit PnL. It is mid-return alpha minus an approximate taker fee, half-spread, and latency-slippage overlay. It does not simulate entering at the ask and exiting at the bid.
- A1.1's L2 additions are **L2 proxies**, not true multi-level OFI. The current feature parquet has top-5 shares/notional and top-5 imbalance snapshots, but it does not persist previous/new quantities for each individual depth level. True MLOFI still requires a raw replay sidecar or A2 replay.

## Unknowns / Ambiguous Code Paths

- Fee rates are hard-coded by category in `FEE_BY_CATEGORY` (`scripts/dali_block_a1_analyze.py:89-103`). A1 does not query per-market `feesEnabled`, `makerBaseFee`, or `takerBaseFee`. Before treating costs as final, A1.1 should reconcile category assumptions against market-level fee config.
- The diagnostic treats a negative market-direction signal as a symmetric short/opposite-market abstraction. It does not decide whether the real executable action should be sell YES, buy NO, reduce existing exposure, or do nothing because of inventory constraints.
- `future_mid` uses the last observed mid at or before `t + horizon`, not interpolation and not the next update strictly after `t + horizon`. See `scripts/dali_block_a1_analyze.py:231-240`.
- Row weighting is event-row weighting, not clock-time weighting. Active markets and bursts with many `price_change` rows receive more weight.
- Market-level diagnostics group YES and NO assets together after normalization. This is intended, but it can produce repeated or mixed market states when both assets emit updates close together.
- `market_resolved` rows are handled in the batch wrapper by dropping rows after first resolution timestamp for affected assets. See `scripts/dali_block_a1_replay_batch.py:91-115` and `scripts/dali_block_a1_replay_batch.py:145-153`. There is not yet a dedicated pre-resolution time-to-resolution bucket in A1.
- A1.1 adds a pre-resolution segmentation bucket, but only for diagnostics over A1 features. It does not change A1's canonical result rows. See `time_to_resolution_bucket` creation in `scripts/dali_block_a11_segment_l2.py:186-192`.
- Stale-book telemetry exists (`book_staleness_seconds`) but A1 currently does not filter on staleness beyond requiring `is_book_state_complete`, non-null mid, and non-null forward return. See `scripts/dali_clob_replay_features.py:152-158` and `scripts/dali_block_a1_analyze.py:511-517`.

## Exact Formulas

### Event-Level Top-of-Book OFI

TOB OFI means top-of-book OFI. It is an L1 projection of the L2-maintained book: the state machine maintains all price levels it has seen, but the OFI calculation only uses the previous and new best bid/best ask. A deeper level contributes to OFI only if it becomes the new touch after an update.

Combined OFI:

```text
OFI_event = bid_ofi + ask_ofi
```

Bid-side contribution from `lib/clob_book.py:34-52`:

```text
if prev_bid is None and new_bid is None: 0
if prev_bid is None:                    +new_bid_size
if new_bid is None:                     -prev_bid_size
if new_bid_price > prev_bid_price:      +new_bid_size
if new_bid_price < prev_bid_price:      -prev_bid_size
if new_bid_price == prev_bid_price:     new_bid_size - prev_bid_size
```

Ask-side contribution from `lib/clob_book.py:55-73`:

```text
if prev_ask is None and new_ask is None: 0
if prev_ask is None:                    -new_ask_size
if new_ask is None:                     +prev_ask_size
if new_ask_price < prev_ask_price:      -new_ask_size
if new_ask_price > prev_ask_price:      +prev_ask_size
if new_ask_price == prev_ask_price:     prev_ask_size - new_ask_size
```

Size changes at unchanged top price are included. Bid size increase is positive; ask size increase is negative.

Terminology guardrail:

```text
TOB OFI / L1 OFI:
    flow/change measure from previous top-of-book to new top-of-book

tob_imbalance:
    standing-book level measure from current best bid/ask sizes
    (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
```

The A1.1 component `tob_imbalance_level` is the second object above. It is not another OFI variant.

### Rolling OFI and TFI

A1 computes rolling windows per `(run_id, asset_id)` in `add_signal_frames`:

```text
ofi_hs = rolling_sum(ofi_combined_event, h seconds)
tfi_hs = rolling_sum(signed_trade_size_live, h seconds)
```

See `scripts/dali_block_a1_analyze.py:205-228`.

### YES/NO Normalization

```text
direction_factor = +1 if outcome_index == 0 else -1
directional_mid = mid if direction_factor > 0 else 1 - mid
ofi_market_hs = direction_factor * ofi_hs
tfi_market_hs = direction_factor * tfi_hs
future_directional_mid = future_mid if direction_factor > 0 else 1 - future_mid
future_return_bps = 10000 * (future_directional_mid - directional_mid) / directional_mid
```

See `scripts/dali_block_a1_analyze.py:195-224`.

### Decile Aggregate

The depth normalizer is computed from per-market result rows:

```text
mean_depth_at_touch = mean(best_bid_size + best_ask_size) for each (run_id, market_key)
```

Touch depth is created in `load_features` at `scripts/dali_block_a1_analyze.py:180`. Per-market mean depth is computed in `market_panel` at `scripts/dali_block_a1_analyze.py:487` and merged into `decile_aggregate` at `scripts/dali_block_a1_analyze.py:599-612`.

For each horizon:

```text
OFI_scaled = ofi_market_hs / market_mean_depth
abs_signal = abs(OFI_scaled)
decile = pd.qcut(abs_signal, 10, labels=False, duplicates="drop") + 1
mean_abs_ofi_scaled = mean(abs_signal within horizon/decile)
```

See `scripts/dali_block_a1_analyze.py:613-648`.

Important interpretation:

```text
decile 10 = largest 10% absolute normalized OFI observations
decile 10 != most bullish observations
```

The sign is used after binning:

```text
hit_rate = mean(sign(OFI_scaled) == sign(future_return_bps))
directional_return_bps = mean(sign(OFI_scaled) * future_return_bps)
mean_next_mid_return_bps = mean(future_return_bps)
```

This is why `mean_next_mid_return_bps` can be negative while `directional_return_bps` is positive: the raw average future return can be down, while the OFI sign correctly points short/opposite direction often enough.

### A1.1 Segment and L2 Proxy Components

A1.1 loads selected columns from `block_a1_features.parquet`, including `ofi_bid_event`, `ofi_ask_event`, `ofi_combined_event`, `tob_imbalance`, `book_imbalance_top_n`, `bid_top5_shares`, `ask_top5_shares`, and `market_resolved_at`. See `scripts/dali_block_a11_segment_l2.py:100-193`.

It computes these shared fields:

```text
touch_depth = best_bid_size + best_ask_size
top5_depth = bid_top5_shares + ask_top5_shares
market_mean_depth = mean(touch_depth) per (run_id, market_key)
relative_depth = touch_depth / market_mean_depth
direction_factor = +1 for outcome_index == 0 else -1
directional_mid = mid if YES side else 1 - mid
future_return_bps_h = 10000 * (future_directional_mid_h - directional_mid) / directional_mid
```

See `scripts/dali_block_a11_segment_l2.py:161-176` and `scripts/dali_block_a11_segment_l2.py:233-241`.

For each horizon `h`, A1.1 computes:

```text
ofi_bid_h = rolling_sum(ofi_bid_event, h seconds)
ofi_ask_h = rolling_sum(ofi_ask_event, h seconds)
ofi_combined_h = rolling_sum(ofi_combined_event, h seconds)

signal_ofi_combined_mean_depth_h =
    direction_factor * ofi_combined_h / market_mean_depth

signal_ofi_bid_mean_depth_h =
    direction_factor * ofi_bid_h / market_mean_depth

signal_ofi_ask_mean_depth_h =
    direction_factor * ofi_ask_h / market_mean_depth

signal_ofi_combined_instant_depth_h =
    direction_factor * ofi_combined_h / touch_depth
```

See `scripts/dali_block_a11_segment_l2.py:242-259`.

Top-5 L2 proxy:

```text
bid_top5_delta_event = current_bid_top5_shares - prior_bid_top5_shares
ask_top5_delta_event = current_ask_top5_shares - prior_ask_top5_shares
top5_depth_pressure_event = bid_top5_delta_event - ask_top5_delta_event

signal_top5_depth_pressure_mean_depth_h =
    direction_factor * rolling_sum(top5_depth_pressure_event, h seconds) / market_mean_depth
```

See `scripts/dali_block_a11_segment_l2.py:227-231` and `scripts/dali_block_a11_segment_l2.py:260-262`.

Book-state imbalance proxies:

```text
tob_imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
book_imbalance_top_n = (bid_top5_shares - ask_top5_shares) / (bid_top5_shares + ask_top5_shares)

signal_tob_imbalance_level_h =
    direction_factor * rolling_mean(tob_imbalance, h seconds)

signal_top5_imbalance_level_h =
    direction_factor * rolling_mean(book_imbalance_top_n, h seconds)
```

The raw imbalance columns are created in `scripts/dali_clob_replay_features.py:198-213`; A1.1 rolling versions are created in `scripts/dali_block_a11_segment_l2.py:246-264`.

A1.1 bins every component by absolute signal magnitude:

```text
abs_signal = abs(signal_component_h)
decile = pd.qcut(abs_signal, 10, labels=False, duplicates="drop") + 1
```

Then it computes:

```text
hit_rate = mean(sign(signal_component_h) == sign(future_return_bps_h))
directional_return_bps = mean(sign(signal_component_h) * future_return_bps_h)
```

See `assign_deciles` and `summarize_rows` in `scripts/dali_block_a11_segment_l2.py:269-304`.

A1.1 segment heatmaps use the canonical `signal_ofi_combined_mean_depth_h` only, not every component, and segment by:

```text
family
market
spread_bucket
relative_depth_bucket
run_id
resolved_in_capture
time_to_resolution_bucket
```

See `SEGMENT_TYPES` in `scripts/dali_block_a11_segment_l2.py:47-55` and `segment_surface` in `scripts/dali_block_a11_segment_l2.py:334-363`.

### Per-Market OFI Metrics

For each market/horizon:

```text
valid rows = complete book state, non-null mid, non-null future return
ofi_eval = valid rows where abs(ofi_market_hs) > 1e-12
OLS: future_return_bps ~ ofi_market_hs
prediction = alpha + beta * ofi_market_hs
top decile = largest 10% abs(prediction)
ofi_hit_rate_top_decile_raw = mean(sign(prediction) == sign(future_return_bps)) in top decile
ofi_directional_return_top_decile_bps_raw = mean(sign(prediction) * future_return_bps) in top decile
```

See `ols_prediction`, `directional_hit`, `metric_from_xy`, `compute_metric`, and `market_panel` in `scripts/dali_block_a1_analyze.py:243-352` and `scripts/dali_block_a1_analyze.py:507-519`.

## Public-vs-Raw Suppression Rules

Constants:

```text
MIN_REPORTABLE_CLASSIFIABLE = 30
MIN_REPORTABLE_TOP_DECILE_N = 30
MIN_REPORTABLE_MAKER_FILLS = 30
```

See `scripts/dali_block_a1_analyze.py:40-43`.

`sample_size_label` from `scripts/dali_block_a1_analyze.py:355-360`:

```text
n_classifiable < 30      -> insufficient
30 <= n_classifiable < 200 -> thin_wide_CI
n_classifiable >= 200    -> primary_read
```

`ofi_metric_reportable` from `scripts/dali_block_a1_analyze.py:378-385`:

```text
n_classifiable >= 30
metric.n_eval >= 30
metric.top_decile_n >= 30
hit_rate is finite
0.0 < hit_rate < 1.0
```

The last condition suppresses perfect-hit artifacts from public columns even if they have enough top-decile rows.

Suppressed public columns:

- `ofi_r2`
- `ofi_r2_ci_lo`
- `ofi_r2_ci_hi`
- `ofi_hit_rate_top_decile`
- `ofi_hit_rate_ci_lo`
- `ofi_hit_rate_ci_hi`
- `ofi_hit_rate_ci_width`
- `ofi_directional_return_top_decile_bps`
- `edge_after_cost_bps`

Raw columns retain unsuppressed values:

- `ofi_hit_rate_top_decile_raw`
- `ofi_directional_return_top_decile_bps_raw`
- `edge_after_cost_bps_raw`

`maker_metric_reportable` from `scripts/dali_block_a1_analyze.py:388-392`:

```text
n_classifiable >= 30
maker_fill_count >= 30
```

`maker_net_edge_bps` is suppressed unless maker metric is reportable; `maker_net_edge_bps_raw` retains the raw proxy. See `scripts/dali_block_a1_analyze.py:496-503` and `scripts/dali_block_a1_analyze.py:583-585`.

Suppression also affects verdict logic because `verdict` is called with public `ofi_hit`, public `ofi_directional_return`, and public `edge`. See `scripts/dali_block_a1_analyze.py:531-536` and `scripts/dali_block_a1_analyze.py:593`.

## Cost-Overlay Formula

Category fee bps:

```text
category_taker_fee_bps =
    fee_rate * p * (1 - p) / directional_mid * 10000
```

where `p` is token mid and denominator is directional mid. See `taker_fee_bps` in `scripts/dali_block_a1_analyze.py:147-155` and market usage in `scripts/dali_block_a1_analyze.py:489-493`.

Half-spread cost:

```text
directional_spread_bps = spread / directional_mid * 10000
half_spread_cost_bps = 0.5 * mean(directional_spread_bps)
```

See `scripts/dali_block_a1_analyze.py:197-200` and `scripts/dali_block_a1_analyze.py:485-486`.

Latency slippage:

```text
ws_latency_ms_assumed = 100
h = latency_ms / 1000
target_time = received_at + h
future_index = last observed row at or before target_time
ws_latency_slippage_bps = mean(abs((directional_mid_at_target - directional_mid_now) / directional_mid_now * 10000))
```

See constants at `scripts/dali_block_a1_analyze.py:40`, `latency_slippage_bps` at `scripts/dali_block_a1_analyze.py:403-427`, and usage at `scripts/dali_block_a1_analyze.py:488`.

Important confirmed fix:

- Latency uses `to_numpy(dtype="datetime64[ns]").astype("int64")`, so the previous microsecond/nanosecond mismatch is fixed in current code. See `scripts/dali_block_a1_analyze.py:412`.
- Regression coverage exists in `tests/test_ofi_calculation.py:121-163`.

Edge after cost:

```text
edge_after_cost_bps =
    ofi_directional_return_top_decile_bps
    - category_taker_fee_bps
    - half_spread_cost_bps
    - ws_latency_slippage_bps
```

The raw calculation is in `scripts/dali_block_a1_analyze.py:523-530`; public suppression is applied at `scripts/dali_block_a1_analyze.py:531`.

Limitations:

- The cost overlay is horizon-independent per market except for the OFI directional-return term.
- Latency slippage is averaged over all rows in the market, not only top-decile signal rows.
- Latency slippage is absolute mid movement and not aligned with OFI sign.
- The overlay is not side-aware executable PnL; it does not enter at ask and exit at bid.

## TFI Calculation

Trade prints come from `last_trade_price` events in replay. See `scripts/dali_clob_replay_features.py:325-338`.

In A1:

```text
aggressor_side = trade_side or last_trade_side
BUY -> +1
SELL -> -1
signed_trade_size_live = is_trade * side_sign * trade_size
tfi_hs = rolling_sum(signed_trade_size_live, h seconds)
tfi_market_hs = direction_factor * tfi_hs
```

See `scripts/dali_block_a1_analyze.py:186-193` and `scripts/dali_block_a1_analyze.py:212-221`.

`n_classifiable` is the count of trade rows with `aggressor_side` in `{"BUY", "SELL"}`. See `scripts/dali_block_a1_analyze.py:478-480`.

`sign_status = live_reported_side_is_aggressor` means A1 treats reported live `BUY` as aggressive buyer of the token and `SELL` as aggressive seller of the token. That convention was established in the sign-convention audit and is hard-coded as a result label at `scripts/dali_block_a1_analyze.py:549`.

## Maker Proxy

The maker proxy is implemented in `maker_fills` at `scripts/dali_block_a1_analyze.py:430-463`.

It filters trade rows with complete mid, trade price, positive trade size, and BUY/SELL side. A simulated touch fill occurs when:

```text
maker BUY fill:
    aggressor_side == SELL
    trade_price <= best_bid + 1e-9

maker SELL fill:
    aggressor_side == BUY
    trade_price >= best_ask - 1e-9
```

Queue position is ignored.

Adverse selection:

```text
if maker_side == BUY:
    adverse_selection_bps_h = (fill_price - future_mid_h) / fill_price * 10000
else maker_side == SELL:
    adverse_selection_bps_h = (future_mid_h - fill_price) / fill_price * 10000
```

See `scripts/dali_block_a1_analyze.py:450-457`.

`maker_net_edge_bps_raw`:

```text
maker_rebate = mean(maker_rebate_bps(category, fill_price))
maker_net_edge_bps_raw = maker_rebate - mean(adverse_selection_bps_5s)
```

See `scripts/dali_block_a1_analyze.py:496-503`.

Maker values appear repeated across horizon rows because maker fill count, 5s adverse selection, and maker net are computed once per market and copied onto each horizon row. The CSV includes `maker_adverse_selection_bps_1s`, `5s`, and `30s` in the internal result frame, but the public schema currently includes only `maker_adverse_selection_bps_5s` and `maker_net_edge_bps`.

## Bootstrap Confidence Intervals

Per-market CIs are computed by `bootstrap_ci` in `scripts/dali_block_a1_analyze.py:286-319`.

- Block size: 300 seconds via `BOOTSTRAP_CHUNK_SECONDS = 300` at `scripts/dali_block_a1_analyze.py:38`.
- Blocks are contiguous in time within the passed `sub` dataframe.
- Per-market calls pass one market/horizon subset, so per-market CIs are within that subset.
- Resamples: `BOOTSTRAP_SAMPLES = 200` at `scripts/dali_block_a1_analyze.py:37`.
- CIs are computed for R2 and hit rate in `compute_metric` at `scripts/dali_block_a1_analyze.py:340-341`.
- Aggregate decile hit-rate CIs are computed by `block_bootstrap_hit` at `scripts/dali_block_a1_analyze.py:721-742`.
- CIs are not currently used for suppression except indirectly via public output. The explicit reportability guard does not check CI width.

Fallback behavior:

- If fewer than four 300-second blocks exist in `bootstrap_ci`, it creates row-count blocks of `max(5, len(clean) // 10)` rows. See `scripts/dali_block_a1_analyze.py:297-302`.
- The aggregate `block_bootstrap_hit` has a similar fallback at `scripts/dali_block_a1_analyze.py:725-729`.

## Verdict Logic

`verdict` is assigned by `scripts/dali_block_a1_analyze.py:363-375`.

```text
if n_classifiable < 30:
    data_thin
elif public edge_after_cost_bps is finite and > 0:
    signal_present_post_cost
elif public hit is finite and public directional return is finite
     and hit >= 0.53 and directional return > 0:
    signal_present_pre_cost
else:
    absent
```

Verdict uses public/suppressed metrics, not raw metrics. Therefore a suppressed public metric prevents a pre-cost or post-cost signal verdict even when `_raw` values look high.

## Hidden Assumptions to Keep in Mind

- **Event-time row weighting:** Active markets and periods with many updates dominate pooled aggregates. There is no clock-time resampling.
- **Repeated states:** Non-book events can carry current book state; if OFI is zero these rows are filtered out for OFI eval, but state rows still exist in the feature table.
- **Forward horizon alignment:** `future_mid` uses the last observed row at or before target time. If target is beyond the asset's final timestamp, output is NaN. See `scripts/dali_block_a1_analyze.py:231-240`.
- **Missing sides:** Missing trade side maps to zero TFI and is not classifiable. See `scripts/dali_block_a1_analyze.py:186-193` and `scripts/dali_block_a1_analyze.py:478-480`.
- **NaN handling:** Numeric parsing coerces malformed values to NaN; OFI combined event NaNs are filled to zero in `scripts/dali_block_a1_analyze.py:194`.
- **Resolution filtering:** The wrapper drops rows after `market_resolved_at`, but A1 does not yet bucket pre-resolution rows by proximity to resolution.
- **A1.1 top-5 pressure is not per-level OFI:** It is a top-5 aggregate depth-change proxy. It can detect broader depth support/depletion, but it cannot tell whether L2, L3, L4, or L5 individually drove the move.
- **A1.1 imbalance components are stock variables:** `tob_imbalance_level` and `top5_imbalance_level` use standing book imbalance averaged over the horizon, while OFI components use flow/change over the horizon. Their hit rates should not be interpreted as OFI hit rates.
- **A1.1 segments use global deciles within each horizon/component:** Segment rows inherit global absolute-signal deciles; they are not within-family or within-market deciles. This is good for comparing where the same global extreme signal lands, but thin segments still need row-count heatmaps.
- **Fee mapping:** Category fee mapping is local and coarse. It should be cross-checked against market-level fee parameters before any cost-sensitive decision.
- **Market-vs-asset grouping:** Rolling signals are per asset; market panel groups rows across assets by market key after sign normalization.
- **Public vs raw values:** Fast CSV scans should prefer public columns and check `ofi_metric_reportable`; raw columns are audit-only.

## Recommended Docstring / README Clarifications

- State explicitly that A1 OFI is top-of-book CKS-style OFI, not multi-level OFI.
- Define "TOB OFI" as top-of-book flow/change OFI, and distinguish it from `tob_imbalance`, a standing-book size-imbalance feature.
- State that per-market metrics use OLS prediction sign, while aggregate deciles use OFI sign directly.
- State that deciles are global per horizon across pooled markets/runs and based on absolute `OFI_scaled`.
- State that returns are directional-mid returns after YES/NO normalization, not raw token-mid returns.
- State that `edge_after_cost_bps` is a diagnostic overlay, not executable bid/ask backtest PnL.
- State that maker proxy is queue-blind and repeats market-level maker stats across horizons.
- Add a short table of public-vs-raw suppression rules near the CSV schema.
- State that A1.1's L2 work is top-5 proxy diagnostics from the existing feature parquet. True multi-level OFI still requires a raw replay that writes per-level OFI columns.

## A1.1 Sanity Checks Completed

Completed in `scripts/dali_block_a11_segment_l2.py`:

1. Segment heatmaps/surfaces by:
   - family
   - market
   - spread bucket
   - relative depth bucket
   - run_id
   - resolved/unresolved
   - time-to-resolution bucket

2. Global horizon-specific OFI decile thresholds for segment heatmaps so decile 10 is comparable across segments.

3. Row-count heatmaps beside every hit-rate heatmap.

4. 300s decomposition by family, market, run, depth, spread, resolved status, and time-to-resolution. This is enough for a composition check, but the note still needs a human read of the specific 300s low-OFI cells.

5. Side-component and L2-proxy diagnostics:
   - bid-only OFI
   - ask-only OFI
   - combined OFI
   - TOB imbalance level
   - top-5 imbalance level
   - top-5 depth-pressure proxy
   - instantaneous-depth normalization vs mean-depth normalization

## Remaining A1.1 / A2 Checks

1. Add executable taker scenarios:
   - enter at ask, exit at bid
   - enter at ask, mark exit at mid
   - buy complement instead of selling token where appropriate
   - include trade-size or walk-the-book assumptions

2. Cross-check fee rates against per-market fee config rather than relying only on family mapping.

3. Add stale-book and event-time/clock-time sensitivity checks.

4. Add tests covering:
   - unchanged ask size increase gives negative OFI
   - unchanged ask size decrease gives positive OFI
   - NO-side OFI and returns flip together
   - aggregate decile 10 is absolute magnitude, not bullishness
   - public verdict ignores `_raw` high-hit values when reportability fails

5. If we want true L2 OFI before A2, replay raw JSONL into a new sidecar feature table with:
   - `bid_ofi_l1..l10`
   - `ask_ofi_l1..l10`
   - `combined_ofi_l1..l10`
   - depth-weighted OFI
   - exponentially weighted OFI
   - integrated/PCA-style OFI

This is not available from the current A1 parquet alone.
