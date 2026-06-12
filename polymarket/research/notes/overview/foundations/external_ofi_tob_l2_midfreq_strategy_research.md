---
title: "External Research Note — OFI/TOB/L2 Mid-Frequency Strategy Library"
tags: [dali, external-research, strategy-library, archive]
created: 2026-05-29
source: user-conducted external research (web-chat / external LLM session), uploaded into Cowork
status: ARCHIVED REFERENCE — see block_a1x_external_note_reconciliation.md (v2) for the tested-vs-open triage
caveat: |
  This note was written from a pre-A1.4 snapshot. A1.4–A1.7 since ran the *directional-continuation* use
  of the local OFI/TOB signal (taker + maker-at-mid) and closed that framing. BUT the signal itself is real
  (73.7% hit at 5s, A1.3) — what A1.x falsified is one way of using it, not the signal. Several ideas here
  are genuinely-untested *framings*: continuous rolling-rank sizing (#1), explicit mean-reversion-to-
  microprice (#4 + fade side of #9/#16/#17), true-L2 features (#5/#6/#7/#15), and off-book cross-market
  lead-lag (#21). Read the reconciliation note's v2 bucket table before treating anything here as either
  dead or promising.
---
> Hub: [[COWORK]]


## Summary

- Scope: OFI / TOB / L2 Mid-Frequency Trading Strategy Research Note in the research area.
- Existing takeaway/status: The best strategy direction is not “trade every OFI spike.” It is:
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
# OFI / TOB / L2 Mid-Frequency Trading Strategy Research Note

Prepared for: Polymarket / DALI A1-A2 research
Date: 2026-05-29
Scope: mid-frequency strategies using L1/L2 order-book and trade-flow data, roughly seconds-to-minutes, not millisecond HFT.

---

## Executive summary

The best strategy direction is not “trade every OFI spike.” It is:

```text
market selection + spread/depth regime filter + TOB confirmation + OFI/MLOFI trigger + executable cost gate
```

Your A1/A1.1 work already points to this structure:

- A1 canonical OFI is **top-of-book / L1 OFI**, not full MLOFI.
- A1.1 top-5 work is useful but still **proxy L2**, not true per-level OFI.
- The strongest current empirical sleeve is **crypto direction markets**, especially BTC/ETH daily and tight/deep states.
- TOB imbalance looks like a high-hit **state/filter** signal.
- OFI looks like a lower-hit but larger-bps **entry trigger**.
- The next robust step is not ML first; it is executable bid/ask PnL and true MLOFI replay.

The most transferable external idea is the Aperiodic-style **rolling-rank transform**: convert any arbitrary microstructure metric into a bounded position signal by rolling percentile rank, then sweep features/windows. For your setup, that trick should be applied to OFI, TOB imbalance, top-5 pressure, MLOFI levels, spread/depth, TFI, and cross-market reference signals.

---

## Source map

### Academic / research anchors

1. **Cont, Kukanov & Stoikov — The Price Impact of Order Book Events**  
   Key idea: short-interval price changes are driven by order-flow imbalance at best bid/ask; impact slope is inversely related to market depth.  
   Source: https://arxiv.org/abs/1011.6402

2. **Xu, Gould & Howison — Multi-Level Order-Flow Imbalance in a Limit Order Book**  
   Key idea: MLOFI is a vector of imbalance at multiple book levels; adding deeper levels improves out-of-sample fit versus L1-only OFI.  
   Source: https://arxiv.org/abs/1907.06230

3. **Gould & Bonart — Queue Imbalance as a One-Tick-Ahead Price Predictor**  
   Key idea: bid/ask queue imbalance predicts next mid-price move direction, especially in large-tick assets.  
   Source: https://arxiv.org/abs/1512.03492

4. **ClusterLOB — Enhancing Trading Strategies by Clustering Orders in Limit Order Books**  
   Key idea: cluster order events into participant-behavior groups, compute cluster-specific OFI in 30-minute buckets, and select strategies by Sharpe.  
   Source: https://arxiv.org/abs/2504.20349

5. **Deep Attentive Survival Analysis in LOBs**  
   Key idea: fill probability is central to choosing passive vs aggressive execution; estimate limit-order fill times from time-varying LOB features.  
   Source: https://arxiv.org/abs/2306.05479

6. **Asynchronous Deep Double Dueling Q-Learning for Trading-Signal Execution**  
   Key idea: use an external alpha signal plus LOB state to learn order placement / inventory control, rather than using RL to discover alpha directly.  
   Source: https://arxiv.org/abs/2301.08688

7. **DeepLOB**  
   Key idea: CNN/LSTM models can learn spatial and temporal structure from LOB snapshots; useful later as a feature extractor, not as A1.1/A2 first step.  
   Source: https://arxiv.org/abs/1808.03668

### Practical / code / data anchors

1. **Aperiodic — Most Predictive notebook**  
   Seed idea from user: rolling percentile rank of an L2 microstructure metric, squashed to `[-1, +1]`, traded as next-bar position size.  
   Source: https://aperiodic.io/notebooks/most-predictive

2. **Aperiodic L1 / L2 data pages**  
   Useful metric inventory: L1 weighted mid, top-of-book imbalance, spread/depth, update frequency, L2 imbalance at 5/10/20/25 levels, L2 liquidity depth.  
   Sources: https://aperiodic.io/metrics/l1 and https://aperiodic.io/metrics/l2

3. **Aperiodic LLM reference**  
   Confirms intervals and fields for flow, impact, slippage, L1/L2 imbalance, and L2 liquidity.  
   Source: https://aperiodic.io/llms-full.txt

4. **AWS sample order-flow pipeline**  
   Useful design pattern: bar aggregation, per-level quantity imbalance, volume imbalance, volatility, trends, and trade imbalance features for ML training.  
   Source: https://github.com/aws-samples/quant-research-sample-using-amazon-ecs-and-aws-batch

5. **Crypto LOB data pipeline**  
   Useful lightweight pattern: stream raw crypto LOB data, compute mid price and normalized depth imbalance in real time.  
   Source: https://github.com/kostyafarber/crypto-lob-data-pipeline

6. **Crypto market microstructure analyzer**  
   Useful lightweight pattern: detect OFI vectors, hidden liquidity / iceberg-like replenishment, and z-score thresholds for volatility adjustment.  
   Source: https://github.com/SpookyJumpyBeans/crypto-market-microstructure-analyzer

---

## Definitions used in this note

### TOB imbalance

Top-of-book state imbalance:

```text
TOB_imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
```

This is a **state** signal. It tells you how the visible touch is leaning right now.

### OFI

Order-flow imbalance is a **change/flow** signal. It measures whether book updates add buy pressure or sell pressure.

Simplified L1 version:

```text
OFI_event = bid_OFI_event + ask_OFI_event
rolling_OFI_w = sum(OFI_event over last w seconds or bars)
```

Positive OFI = buy pressure.  
Negative OFI = sell pressure.

### MLOFI

Multi-level OFI computes OFI separately at multiple depths:

```text
OFI_L1, OFI_L2, ..., OFI_L10
```

Then combines them as a vector, weighted sum, PCA component, or ML feature set.

### Mid-frequency framing

For this project, “mid-frequency” means:

```text
1s, 2s, 5s, 10s, 15s, 30s, 60s, 120s, 300s
```

It explicitly does **not** assume colocated microsecond execution.

---

# Strategy library

## 1. Rolling-rank L2 imbalance position sizing

### Idea

This is the direct extension of the Aperiodic-style trick you mentioned.

Take any microstructure metric, compute its rolling percentile rank over the last `w` bars, squash it to `[-1, +1]`, and use that as the position size for the next bar.

### Feature candidates

```text
L1 imbalance
L2 imbalance 5 levels
L2 imbalance 10 levels
L2 imbalance 20/25 levels
OFI_scaled
MLOFI weighted score
top5_depth_pressure
TFI / volume delta
weighted_mid_minus_mid
spread_depth_ratio
```

### Signal

```text
rank_t = percentile_rank(metric_t over trailing w bars)
position_t = 2 * rank_t - 1
pnl_{t+1} = position_t * return_{t+1} - costs
```

For symmetric markets:

```text
position_t > 0 => long / buy market direction
position_t < 0 => short / sell / buy complement / reduce exposure
```

### Why it is useful

- No calibration to raw units.
- Works across metrics with different scales.
- Easy to sweep across features and windows.
- Converts “diagnostic” signals into direct strategy candidates.

### Polymarket adaptation

Use market-direction-normalized features:

```text
metric_t = OFI_market_direction_scaled
return_t = directional_mid_return or executable ask/bid return
```

Recommended windows:

```text
w = 30s, 60s, 300s, 900s, 1800s
horizon = 5s, 10s, 30s, 60s
```

### Gate

Only keep variants where:

```text
ask-entry/bid-exit PnL > 0
or passive-entry expected PnL > 0 after fill probability and adverse selection
```

---

## 2. Extreme normalized OFI momentum

### Idea

Trade only when rolling OFI is unusually large relative to local liquidity.

This is closest to your A1/A1.1 result.

### Feature

```text
OFI_scaled_w = rolling_sum(OFI_event, w) / depth_normalizer
```

Depth normalizers to test:

```text
mean_touch_depth_by_market
instant_touch_depth_t
median_touch_depth_by_market
top5_depth_t
EWMA_touch_depth_t
```

### Signal

```text
if abs(OFI_scaled_w) >= q90_abs_OFI_scaled_w:
    position = sign(OFI_scaled_w)
else:
    position = 0
```

### Strategy variants

```text
V1: taker entry, ask/bid exit
V2: taker entry, mid-mark exit
V3: passive entry only when TOB agrees
V4: complement-route execution for binary markets
```

### Why it is useful

It is simple, interpretable, and maps directly to A1/A1.1. It also avoids weak middle-decile noise.

### Main risk

Taker costs can dominate. This should be an executable-PnL test, not a mid-return test.

---

## 3. TOB imbalance filter + OFI trigger

### Idea

Use TOB imbalance as the book-state filter and OFI as the entry trigger.

```text
TOB = should I believe the setup?
OFI = should I enter now?
```

### Feature

```text
TOB_imbalance_t = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
OFI_scaled_w = rolling OFI / depth
```

### Long condition

```text
TOB_imbalance_t > -threshold_neutral
AND OFI_scaled_w > q90
AND spread_bps < spread_cap
AND depth > depth_floor
```

### Short / opposite condition

```text
TOB_imbalance_t < threshold_neutral
AND OFI_scaled_w < q10
AND spread_bps < spread_cap
AND depth > depth_floor
```

### Why it is useful

Your A1.1 result suggests TOB imbalance has high hit rate but smaller bps, while OFI has lower hit rate but larger bps. That points to TOB as a filter and OFI as the trigger.

### Polymarket adaptation

For negative market-direction signal, map the action explicitly:

```text
if holding YES: sell YES
if flat and complement available: buy NO
if complement route is wide: skip
```

---

## 4. Weighted-mid / microprice drift

### Idea

When bid size is much larger than ask size, the “fair” short-term price may be above the simple midpoint. When ask size dominates, fair price may be below midpoint.

### Feature

One common version:

```text
weighted_mid = (best_ask_price * best_bid_size + best_bid_price * best_ask_size)
               / (best_bid_size + best_ask_size)

microprice_edge_bps = 10000 * (weighted_mid - mid) / mid
```

### Signal

```text
position = clip(microprice_edge_bps / scale, -1, +1)
```

or thresholded:

```text
if microprice_edge_bps > threshold: long
if microprice_edge_bps < -threshold: short
```

### Why it is useful

It turns a book-state imbalance into a pseudo-fair-value shift. It is usually weaker than OFI but helpful as a confirmation/regime feature.

### Polymarket adaptation

Use it as a pre-trade filter:

```text
only take positive OFI if weighted_mid >= mid
only take negative OFI if weighted_mid <= mid
```

---

## 5. L1-L2 divergence strategy

### Idea

Compare top-of-book imbalance to deeper-book imbalance.

Aperiodic’s L2 guide emphasizes that L1 and L2 can tell contradictory stories: L1 may look strong while deeper levels are empty, or L1 may look weak while deeper support exists.

### Features

```text
L1_imbalance = (bid_l1 - ask_l1) / (bid_l1 + ask_l1)
L5_imbalance = (sum_bid_l1_l5 - sum_ask_l1_l5) / (sum_bid_l1_l5 + sum_ask_l1_l5)
L10_imbalance = same over 10 levels

divergence_5 = L1_imbalance - L5_imbalance
divergence_10 = L1_imbalance - L10_imbalance
```

### Strategy patterns

#### Fake support / thin ladder

```text
L1_bid_heavy but L5/L10 not bid_heavy
=> do not chase long OFI
=> possible short if OFI flips negative
```

#### Hidden support / pullback buy

```text
L1 looks weak but L5/L10 bid support is strong
=> long only after OFI turns positive
```

#### Thin ask ladder / breakout

```text
L1 neutral, L5/L10 ask depth thin, OFI positive
=> long breakout candidate
```

### Why it is useful

This is the most natural next step from A1.1 top-5 proxy diagnostics into A2 true MLOFI.

---

## 6. MLOFI linear score

### Idea

Instead of compressing the whole book into one imbalance, compute per-level OFI and fit a simple linear model.

### Feature

```text
x_t = [OFI_L1, OFI_L2, ..., OFI_L10]
```

### Model

```text
future_return_bps_{t,h} = beta_0 + beta_1*OFI_L1 + ... + beta_10*OFI_L10 + error
```

Then trade the prediction:

```text
pred_t = beta dot x_t
position_t = sign(pred_t) * min(1, abs(pred_t) / scale)
```

### Constraints

Keep this deliberately simple:

```text
ridge regression only
walk-forward fit
no huge parameter search
market/family-segmented coefficients
```

### Why it is useful

MLOFI literature finds deeper levels add predictive power. But for your data, the key question is whether true per-level OFI beats L1 OFI after executable costs.

---

## 7. MLOFI PCA / integrated pressure score

### Idea

Convert the per-level MLOFI vector into one or two robust components.

### Feature

```text
MLOFI_matrix = rows x levels
PC1 = broad book pressure
PC2 = near-touch vs deep-book divergence
```

### Signal

```text
position = sign(PC1) if abs(PC1) is top decile
```

or:

```text
trade only when PC1 and L1_OFI agree
skip when PC2 says pressure is isolated at the touch
```

### Why it is useful

It avoids overfitting each level separately and gives interpretable decomposition:

```text
PC1 = whole ladder leaning
PC2 = touch-only versus depth-backed pressure
```

---

## 8. EWMA OFI / decay kernel

### Idea

Boxcar windows treat a 5-second-old event the same as a fresh event. EWMA gives more weight to recent events.

### Feature

```text
EWMA_OFI_t = sum_i OFI_i * exp(-(t - t_i) / tau)
```

Candidate half-lives:

```text
1s, 2s, 5s, 10s, 30s, 60s
```

### Signal

```text
position = sign(EWMA_OFI_t) when abs(EWMA_OFI_t / depth_t) is top decile
```

### Why it is useful

A1/A1.1 tested fixed windows. EWMA may capture smoother pressure and avoid window-boundary artifacts.

---

## 9. Ask-depletion / bid-support decomposition

### Idea

Do not just ask whether combined OFI works. Ask which side of the book carries the signal.

### Components

```text
bid_add_support
bid_cancel_depletion
bid_price_improvement
ask_add_resistance
ask_cancel_depletion
ask_price_improvement_down
```

### Strategy examples

#### Ask depletion long

```text
ask depth falling rapidly
AND bid support stable/increasing
AND spread not widening too much
=> long / buy market direction
```

#### Bid depletion short

```text
bid depth falling rapidly
AND ask depth stable/increasing
=> short / sell / buy complement
```

#### Replenishment fade

```text
large taker buy consumes ask
ask replenishes quickly
price fails to move
=> fade / avoid long
```

### Why it is useful

In prediction markets, the economic action differs depending on whether you are buying, selling, or using the complement. Side-aware decomposition tells you which route is viable.

---

## 10. OFI + TFI confirmation

### Idea

Book pressure and trade pressure are different. Require them to agree.

### Features

```text
OFI_scaled_w = rolling book-flow imbalance / depth
TFI_scaled_w = rolling signed trade-flow imbalance / trade_volume_or_depth
```

### Signal

```text
if OFI_scaled_w > q90 and TFI_scaled_w > q70:
    long
elif OFI_scaled_w < q10 and TFI_scaled_w < q30:
    short
else:
    skip
```

### Variants

```text
OFI leads TFI: book moves first, tape confirms later
TFI leads OFI: aggressive trades pressure book, book follows
OFI-only: passive book pressure without trade confirmation
TFI-only: aggressive tape pressure without book support
```

### Why it is useful

A1 showed TFI is also informative in crypto. This strategy tests whether combined signals improve cost-adjusted edge.

---

## 11. Flow toxicity / avoid-the-maker-trap

### Idea

Do not place passive orders when the order flow looks informed or toxic.

### Features

```text
abs(OFI_scaled)
abs(TFI_scaled)
flow_toxicity_score
impact_per_notional
spread widening
mid volatility
quote update burst
```

### Passive-entry rule

```text
post passive only when:
    spread is attractive
    TOB imbalance is not strongly adverse
    OFI/TFI toxicity is below threshold
    expected fill probability is high enough
    adverse-selection estimate is low enough
```

### Taker-entry rule

```text
if flow toxicity is high and directional OFI is strong:
    prefer taker/urgent entry if costs clear
else:
    avoid chasing
```

### Why it is useful

Your current maker proxy is queue-blind. A toxicity filter is a simple intermediate step before full fill-probability modeling.

---

## 12. Fill-probability gated passive strategy

### Idea

Use LOB state to decide whether passive entry is likely to fill before the alpha decays.

### Features

```text
queue_ahead_estimate
trade_rate
opposite-side market order rate
TOB imbalance
OFI direction
spread
depth
quote update rate
time_to_resolution
```

### Expected value

```text
EV_passive = fill_prob * (expected_alpha_after_fill + spread_capture - adverse_selection)
             - (1 - fill_prob) * opportunity_cost
```

### Rule

```text
if EV_passive > EV_taker and fill_prob is sufficient:
    post passive
elif EV_taker > 0:
    cross spread
else:
    skip
```

### Why it is useful

This directly targets the main A1/A1.1 issue: the signal is interesting, but taker costs are often brutal.

---

## 13. Spread/depth regime-gated OFI

### Idea

Only trade OFI when the market is in a regime where execution can plausibly clear costs.

### Regime filters

```text
spread_bps <= q25 or q50
relative_depth >= q50 or q75
book_staleness <= threshold
quote_update_rate >= threshold
trade_count >= threshold
```

### Signal

```text
if regime_good and abs(OFI_scaled_w) is extreme:
    trade sign(OFI_scaled_w)
else:
    skip
```

### Why it is useful

A1.1 showed tighter and deeper buckets had better hit/return properties. It also reduces false positives in wide/spiky markets.

---

## 14. Market-selection / similarity strategy

### Idea

Before trading, rank markets by similarity to known winners.

### Winner fingerprint from current work

```text
crypto direction market
high book update rate
high classifiable trade rate
reasonable spread
healthy touch depth
strong TOB hit rate
strong 5s/10s OFI directional return
decent unresolved-window behavior
```

### Candidate score

```text
score = z(update_rate)
      + z(trade_rate)
      - z(spread_bps)
      + z(relative_depth)
      + z(OFI_extreme_frequency)
      + z(TOB_signal_stability)
      - penalty(stale_book)
```

### Use

```text
A0c ranks markets.
A1.2 replays ranked markets.
A2 allocates raw L2 capture only to top candidates + negative controls.
```

### Why it is useful

It keeps A2 capture budget concentrated on markets structurally similar to BTC/ETH daily winners, rather than all “crypto” markets.

---

## 15. L2 liquidity vacuum / breakout strategy

### Idea

Trade when one side of the book has little depth beyond the touch, so a modest pressure event could move price several ticks.

### Features

```text
ask_ladder_depth_5
bid_ladder_depth_5
ask_ladder_depth_10
bid_ladder_depth_10
gap_to_next_levels
spread_bps
OFI direction
TFI direction
```

### Long setup

```text
ask_depth_5 is low
ask_depth_10 is low
OFI positive
TFI non-negative
spread not too wide
```

### Short setup

```text
bid_depth_5 is low
bid_depth_10 is low
OFI negative
TFI non-positive
```

### Why it is useful

This is not “book is bid-heavy.” It is “the opposing side is thin enough that pressure may travel.” This can be more relevant for taker execution.

---

## 16. Replenishment / resilience strategy

### Idea

After one side is consumed, measure how quickly liquidity comes back. Fast replenishment usually means the move may fail; slow replenishment means the move may continue.

### Features

```text
post_trade_depth_replenishment_1s
post_trade_depth_replenishment_5s
ask_replenishment_after_buy_pressure
bid_replenishment_after_sell_pressure
```

### Rule

```text
if positive OFI consumed ask and ask does not replenish:
    long continuation

if positive OFI consumed ask and ask replenishes immediately:
    avoid long or fade
```

### Why it is useful

This separates genuine pressure from spoof-like / temporary touch depletion.

---

## 17. Large-trade impact and reversal strategy

### Idea

Use trade-level impact metrics to decide whether aggressive flow continues or reverses.

### Features

```text
large_trade_volume
large_trade_percentage
impact_per_notional
large_trade_reversal
directional_impact_asymmetry
TFI
OFI_after_trade
```

### Continuation condition

```text
large buy trade
AND impact_per_notional high
AND OFI remains positive
AND ask does not replenish
=> long continuation
```

### Reversal condition

```text
large buy trade
AND price snaps back
AND ask replenishes
AND OFI flips negative
=> fade / avoid long
```

### Why it is useful

Aperiodic-style trade metrics include impact and large-trade reversal. For Polymarket, this can help distinguish informed directional trades from one-off liquidity consumption.

---

## 18. Queue-imbalance logistic classifier

### Idea

Use TOB or queue imbalance to estimate probability of next move.

### Model

```text
P(up over h seconds) = sigmoid(alpha + beta * TOB_imbalance + controls)
```

Controls:

```text
spread_bps
relative_depth
trade_rate
volatility
time_to_resolution
family
run_id
```

### Signal

```text
if P(up) > 0.55 and executable_edge > 0:
    long
elif P(up) < 0.45 and executable_edge > 0:
    short
else:
    skip
```

### Why it is useful

It is interpretable and grounded in queue-imbalance literature. It is also less likely to overfit than a deep model.

---

## 19. Clustered OFI / participant-regime strategy

### Idea

Cluster order events into behavioral types, then compute OFI separately by cluster.

Cluster examples:

```text
directional / urgent
opportunistic
market-making / replenishment
```

### Feature

```text
cluster_OFI_directional
cluster_OFI_opportunistic
cluster_OFI_market_maker
```

### Signal

```text
long if directional_cluster_OFI positive
and market_maker_replenishment is not opposing it
```

### Why it is useful

ClusterLOB found that cluster-specific OFI can be used as trading signals and compared strategies by Sharpe. For your use case, this is an A2/A3 idea after true MLOFI and executable PnL are in place.

---

## 20. Order-event decomposition strategy

### Idea

Separate OFI by event type:

```text
limit adds
cancellations
market/trade consumption
price-level moves
```

### Feature examples

```text
add_bid_pressure
cancel_bid_pressure
add_ask_pressure
cancel_ask_pressure
trade_buy_pressure
trade_sell_pressure
```

### Signal examples

```text
ask cancellations + buy trades => stronger long continuation
bid additions alone => weaker long, maybe passive support only
ask additions after buy trades => resistance / fade
```

### Why it is useful

Combined OFI can hide the mechanism. Event decomposition tells whether pressure came from active taking, passive replenishment, or cancellations.

---

## 21. Cross-market reference lead-lag strategy

### Idea

Prediction markets may lag underlying crypto venues. Use BTC/ETH perp L1/L2 pressure as a reference signal for Polymarket crypto direction markets.

### Features

```text
binance_or_okx_BTC_OFI
binance_or_okx_BTC_TFI
binance_or_okx_weighted_mid_move
polymarket_OFI
polymarket_spread_depth
basis / funding / OI as slow regime filters
```

### Signal

```text
if reference_BTC_OFI strongly positive
and reference_mid_return positive
and Polymarket crypto market has not moved enough:
    buy UP / sell DOWN / buy complement according to route costs
```

### Why it is useful

This may be more valuable than purely local Polymarket OFI if the venue is less efficient or thinner than major crypto perps.

### Caveat

Must handle timestamp alignment and look-ahead rigorously. Use exchange timestamps where possible.

---

## 22. Funding / OI regime + OFI trigger

### Idea

Use slower derivatives metrics as context, not direct entry.

### Features

```text
funding_rate
funding_rate_change
open_interest_change
basis_bps
OFI_scaled_w
TFI_scaled_w
```

### Rule

```text
if funding/OI regime indicates crowded long
and local OFI flips negative:
    short signals get higher weight

if OI rising + positive OFI + positive TFI:
    long signals get higher weight
```

### Why it is useful

This moves from pure microstructure to “microstructure inside a derivatives regime,” which may improve robustness at 30s-300s horizons.

---

## 23. Uncertainty-weighted ML classifier

### Idea

Train a model to predict future return direction/magnitude, but use uncertainty to size or skip trades.

### Inputs

```text
L1 imbalance
weighted mid edge
OFI windows
MLOFI levels
TFI windows
spread/depth
trade rate
update rate
staleness
time to resolution
reference-market signals
```

### Output

```text
predicted_return_bps
prob_up
uncertainty
```

### Position

```text
position = sign(predicted_return) * confidence * cost_gate
```

where:

```text
confidence = 1 - normalized_uncertainty
cost_gate = 1 if predicted_edge > executable_cost else 0
```

### Why it is useful

This borrows from Bayesian LOB work: do not just predict; size down or skip when uncertainty is high.

### Timing

Do this after A2 feature integrity is clean. Not before.

---

## 24. RL execution overlay, not RL alpha discovery

### Idea

Do not ask RL to discover alpha. Give it a validated OFI/MLOFI alpha signal and ask it how to execute.

### State

```text
alpha_signal
spread
depth
TOB imbalance
queue estimate
inventory
time since signal
time to resolution
available YES/NO routes
```

### Actions

```text
cross ask
post bid
cancel
sell bid
buy complement
reduce inventory
wait
```

### Reward

```text
realized executable PnL - inventory penalty - missed-fill penalty - adverse-selection penalty
```

### Why it is useful

The RL execution literature is most relevant for translating a signal into orders. It is not the right first tool for proving that OFI alpha exists.

---

# Strategy priority for your A2 path

## Highest priority now

1. **Extreme normalized OFI momentum**  
   Directly validates A1/A1.1 at 5s/10s.

2. **TOB filter + OFI trigger**  
   Best interpretation of the current TOB-vs-OFI evidence.

3. **Executable ask/bid PnL for OFI**  
   Mandatory before any strategy claim.

4. **L1-L2 divergence**  
   Best immediate use of true A2 L2 data.

5. **MLOFI linear score**  
   Cleanest way to test whether deeper book levels add value.

6. **OFI + TFI confirmation**  
   Tests whether book pressure and trade pressure together improve cost-adjusted edge.

7. **Spread/depth regime-gated OFI**  
   Keeps trades in viable execution regimes.

8. **Market-selection similarity scan**  
   Needed before spending capture budget broadly.

## Medium priority

```text
EWMA OFI
ask-depletion/bid-support decomposition
liquidity vacuum breakout
replenishment/resilience
large-trade continuation/reversal
queue imbalance logistic classifier
```

## Later / A3 priority

```text
Clustered OFI
DeepLOB-style models
Bayesian uncertainty sizing
RL execution overlay
funding/OI regime integration
```

---

# Concrete A2 feature table proposal

## Base identifiers

```text
run_id
market_id
asset_id
family
outcome_index
timestamp_exchange
timestamp_received
time_to_resolution
resolved_in_capture
```

## L1 / TOB features

```text
best_bid
best_ask
mid
spread
spread_bps
best_bid_size
best_ask_size
touch_depth
tob_imbalance
weighted_mid
weighted_mid_edge_bps
quote_update_rate
book_staleness_seconds
```

## OFI features

```text
ofi_l1_event
bid_ofi_l1_event
ask_ofi_l1_event
ofi_l1_1s
ofi_l1_2s
ofi_l1_5s
ofi_l1_10s
ofi_l1_30s
ofi_l1_60s
ofi_l1_300s
ofi_l1_ewma_tau_1s
ofi_l1_ewma_tau_5s
ofi_l1_ewma_tau_30s
```

## MLOFI features

```text
bid_ofi_l1..bid_ofi_l10
ask_ofi_l1..ask_ofi_l10
combined_ofi_l1..combined_ofi_l10
mlofi_weighted_sum_near
mlofi_weighted_sum_deep
mlofi_pc1
mlofi_pc2
l1_l5_divergence
l1_l10_divergence
```

## L2 state features

```text
bid_depth_l5
ask_depth_l5
bid_depth_l10
ask_depth_l10
imbalance_l5
imbalance_l10
liquidity_vacuum_bid_l5
liquidity_vacuum_ask_l5
book_slope_bid
book_slope_ask
```

## Trade-flow features

```text
trade_count
classifiable_trade_count
signed_trade_size
signed_trade_notional
tfi_1s
 tfi_5s
tfi_10s
tfi_30s
buy_run_length
sell_run_length
large_trade_flag
impact_per_notional
large_trade_reversal
```

## Cost / execution features

```text
ask_entry_bid_exit_return_bps
ask_entry_mid_exit_return_bps
bid_passive_mid_exit_return_bps
sell_bid_exit_return_bps
complement_route_return_bps
fee_bps_actual
spread_cost_bps
latency_slippage_bps_signal_rows
```

---

# Backtest rules that matter

## Use executable returns, not only mid returns

For a long taker:

```text
entry = ask_t
exit = bid_{t+h}
return_bps = 10000 * (exit - entry) / entry - fees
```

For a short / sell existing exposure:

```text
entry = bid_t  # sale price
mark/exit according to inventory route
```

For binary complement route:

```text
compare sell YES vs buy NO after all fees/spreads
```

## Separate alpha horizon from holding horizon

The signal horizon may be 5s/10s, but execution may need:

```text
entry timeout
fill timeout
exit rule
inventory cap
stop conditions
```

## Avoid overfitting windows

Use coarse windows only:

```text
1s, 2s, 5s, 10s, 15s, 30s, 60s, 120s, 300s
```

Avoid testing dozens of arbitrary windows until the core effect survives costs.

## Use market-balanced and clock-time tests

For aggregate claims, compare:

```text
event-row weighting
clock-time 1 row / second / market
market-balanced sampling
family-balanced sampling
```

This is especially important for the 300s low-OFI anomaly.

---

# Suggested Codex prompt for this research branch

```text
You are in the epsilon-quant-research repo. Create an A2 strategy-research branch focused on mid-frequency L1/L2 order-book strategies for Polymarket crypto direction markets.

Do not implement live trading. Build research outputs only.

Inputs:
- existing A1/A1.1 feature and result files
- raw CLOB JSONL where available
- market metadata and fee config

Implement research modules for:

1. Rolling-rank transform strategies
   - Apply rolling percentile-rank transform to OFI, TOB imbalance, top5 pressure, TFI, spread/depth, and future MLOFI features.
   - Convert rank to position in [-1, +1].
   - Evaluate next-horizon executable returns, not only mid returns.

2. TOB filter + OFI trigger
   - Treat TOB imbalance as state filter and OFI as trigger.
   - Test 5s/10s/30s horizons.
   - Segment by spread/depth/family/market/run/time-to-resolution.

3. True MLOFI replay
   - Compute bid_ofi_l1..l10, ask_ofi_l1..l10, combined_ofi_l1..l10 from raw book mutations.
   - Add weighted-sum and PCA/integrated MLOFI variants.
   - Compare against current L1 OFI baseline.

4. L1-L2 divergence
   - Compare L1 imbalance to L5/L10 imbalance.
   - Test fake-support, hidden-support, and liquidity-vacuum patterns.

5. OFI + TFI confirmation
   - Compare OFI-only, TFI-only, and OFI+TFI gates.

6. Executable cost scenarios
   - ask entry / bid exit
   - ask entry / mid mark
   - passive bid entry proxy / mid or bid exit
   - complement route for YES/NO markets
   - actual fee config where available

Outputs:
- data/analysis/csv_outputs/dali/a2_strategy_surface.csv
- data/analysis/csv_outputs/dali/a2_strategy_by_market.csv
- data/analysis/csv_outputs/dali/a2_strategy_cost_audit.csv
- data/analysis/csv_outputs/dali/a2_strategy_feature_rank.csv
- data/analysis/a2_plots/*.png
- notes/dali/a2_strategy_research_note.md

Important controls:
- no look-ahead
- use exchange/received timestamp alignment explicitly
- include row-count heatmaps
- include market-balanced and clock-time sensitivity
- include stale-book filters
- do not let raw or sparse perfect-hit artifacts drive conclusions
```

---

# Recommended conclusion

The best immediate trading-strategy research path is:

```text
A0c: find more markets like BTC/ETH daily
A1.2: run rolling-rank + TOB/OFI strategy surfaces on those markets
A2: true MLOFI + executable bid/ask PnL
A3: ML/RL only after A2 proves cost-surviving alpha
```

The first deployable-looking family of strategies, if any, will probably be:

```text
tight/deep crypto direction markets
5s/10s horizon
TOB-confirmed extreme OFI
with executable ask/bid or passive-entry route passing costs
```

Do not treat the current L2 proxy work as proof that deeper L2 adds alpha yet. Treat it as evidence that A2 should test true MLOFI carefully against the current L1 OFI baseline.
