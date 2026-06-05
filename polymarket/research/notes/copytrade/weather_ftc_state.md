# Weather FTC TP — state of the strategy (2026-05-15)
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


Short-form summary of where the FTC take-profit weather strategy stands after the bookkeeping fixes and the next-fill slippage work. Copy/paste into chat or doc as needed. Numbers throughout are for the canonical cell `(p_in=0.60, p_out=0.90, policy='all')` unless noted; n=8,938 entries.

## TL;DR

- **Pre-slippage bookkeeping bugs fixed in `ftc_tp_sizing.py`** shrank headline "total return" from a fictional 2,272,346% to a coherent 628.6%; max DD −43% → −10%.
- **Under taker execution** (cross spread, next-fill proxy) the canonical edge is negative (−2.4% ROI), and fallback-dominated.
- **Under realistic WS-passive execution** (`entry_model='sticky'` in §11 — bot posts at p_in cap, can't dodge crash hits because of queue/latency), **EVERY (p_in, p_out) cell loses money** — between −427% and −941% ROI on $10k bankroll at 5% sizing over 12 months. Best (least bad) sticky cell: `(0.85, 0.90)` at −$42.7k / −427%.
- **Under unrealistic track-down execution** (`entry_model='track_down'` — bot continuously cancels & reposts to track best bid in both directions) every cell is positive — best is `(0.80, 0.90)` at +$304,800 / +3,048% on the same bankroll. **This is an UPPER BOUND**, not deployable. Requires sub-second cancel-replace + queue priority + interpretation of `bid_nf` as a fair price (not just a one-side-wide-spread artifact).
- **The strategy is NOT deployable on analysis alone.** The realistic-vs-upper-bound gap ($90k-$360k per cell on $10k bankroll) is the execution-quality sensitivity. Whether any deployment is profitable depends entirely on which execution model your bot actually achieves in production.
- **Recommended next step**: **execution-calibration live test, not profit test.** See live-test design below.

---

## What was fixed in the bookkeeping (Session 1)

Localised changes in [`polymarket/research/scripts/backtest/ftc_tp_sizing.py`](polymarket/research/scripts/backtest/ftc_tp_sizing.py); see commit-equivalent diff in chat history.

| # | bug | direction | effect |
|---|---|---|---|
| 1 | `backtest()` defaulted `max_notional_pct=0.05` while docstring + `grid_backtest` used 0.02 | inflate | `avg_notional_per_trade` 3.69% → 1.98% |
| 2 | `one_trade_per_city` deduped on `entry_ts.floor("D")` instead of `end_ts.floor("D")` | both ways | n_trades 2836 → 2871 in canonical; correctly catches straddle-midnight dups + un-merges different-resolution-day siblings |
| 3 | NaN resolution silently became `0` (chop) | inflate losses | n_dropped in this dataset = 0 (no NaN), but the guardrail is now in place |
| 4 | Sharpe / CAGR span varied per (p_in, p_out) cell in `grid_backtest` | makes grid cells incomparable | now uses universe-wide span; one constant `annualization_days=345` |
| 5 | `equity = cumprod(1+ret)` while `pnl_pct` was computed on flat $1 (no compounding) | huge inflation at scale | total_return 2.27M% → 628.6%; CAGR 4.06M% → 665.0%; max DD −43% → −10% |

Sanity check: `eval_pair` (the original data-pipeline edge metric) was unaffected — exact same `n_entries=6219` and `edge=0.0548` before/after.

---

## What was built for slippage (Sessions 2 + 2.5 + 2.6)

### Data plumbing

- Added `first_cross_maker`, `first_cross_taker`, `first_cross_maker_side` to the per-instance parquet via additive changes in [`scripts/weather_tail_analysis.py:compute_crossings`](polymarket/research/scripts/weather_tail_analysis.py). Detection logic for *which fills are first crosses* did not change (sanity checks at p=0.70 still produce chop_rate=0.287, edge=+0.0131 exactly as before).
- `pivot_inst_to_wide` now also emits `mk_NNN`, `tk_NNN`, `ms_NNN` columns per barrier.

### Next-fill price proxy

[`data_infra/weather_analysis.py:lookup_next_fills_batch`](polymarket/research/data_infra/weather_analysis.py) does one SQL pass over the 1.06B-row trades parquet per call. Returns, per anchor:

- BID next-fill (BUY-maker print) in (anchor + min_seconds, anchor + max_seconds]
- ASK next-fill (SELL-maker print) over the same window
- Different-counterparty filter: anchor's maker/taker excluded from both sides of the next fill

Direction mapping is per-leg-relative:
- entry (we buy):  `next_same_dir = ASK` (next buy-aggressor),  `next_opp_dir = BID`
- exit  (we sell): `next_same_dir = BID` (next sell-aggressor), `next_opp_dir = ASK`

Cross-aware fallback when no qualifying next-fill exists: `fallback_cents=3¢` baseline + `assumed_spread_cents=2¢` for the other side.

### Three fill scenarios

`compute_fill_scenarios()` adds three PnL columns:

- **`pnl_next_same_dir`** — cross-the-spread proxy on both legs (taker model)
- **`pnl_next_opp_dir`** — passive-fill proxy on both legs (maker-best model)
- **`pnl_midpoint`** — midpoint of the two, clipped per leg to never be worse than `next_same_dir` (handles crossed-market cases)

Invariants enforced + tested:  `pnl_next_same_dir <= pnl_midpoint <= pnl_next_opp_dir` on every row.

### Slippage diagnostic

`slippage_diagnostic()` characterises what the proxy is actually measuring. Per (leg × direction):

- `n_total / n_fallback / fallback_pct`
- slip percentiles (next_fill − trigger, in cents): p10, p25, median, p75, p90
- lag percentiles in seconds (real-fill rows only)
- shape flag: `unimodal | long_tail | bimodal_or_thin_data`
- **`interpretation`** field: emits **`"constant_slippage_with_labelling"`** when `fallback_pct > 50%`

Plus per-leg spread-estimate distribution + per-leg `crossed_market` flag computed on raw (pre-fallback) prices.

### Window comparison + raw activity diagnostic

`compare_windows_diagnostic()` reruns lookup at multiple window caps (default 300s and 600s) and reports fallback drop + edge change.

`time_to_first_any_fill_diagnostic()` is the rawest measure — for each cross, time to ANY next fill by a different trader, ignoring side. Tells you whether the markets are simply illiquid (yes) or just need a wider window (no).

### Strategy wiring

[`ftc_tp_sizing.py:build_trades`](polymarket/research/scripts/backtest/ftc_tp_sizing.py) now uses next-fill prices for `entry_ask` and `exit_bid`. The old `slip` parameter is gone; new params are `min_seconds=15`, `max_seconds=300`, `fallback_cents=3.0`, `assumed_spread_cents=2.0`. Per-share PnL uses actual `entry_ask_price` as the share-conversion denominator (not `p_in`). Summary outputs gain `mean_entry_ask_cents_above_p_in`, `entry_ask_fallback_rate`, etc.

---

## Headline numbers (canonical cell)

```
n_entries = 8938   bucket distribution: p_tp=0.376  p_hold_win=0.295  p_hold_chop=0.329

scenario          edge_$     ROI_%   sd       Sharpe_per_trade
next_same_dir    -0.0142    -2.36%   0.4437   -0.032
midpoint         +0.0078    +1.30%   0.4447   +0.018
next_opp_dir     +0.0298    +4.97%   0.4485   +0.066

operational lever (next_opp_dir − next_same_dir): +4.40¢/share

bucket-conditional next_same_dir PnL:
  tp         n=3360  +0.2366
  hold_win   n=2637  +0.3547
  hold_chop  n=2941  -0.6313
```

## Diagnostic readout

```
fallback rates per leg/direction:
  entry/same_dir : 87.5%  →  constant_slippage_with_labelling   (shape: bimodal_or_thin_data)
  entry/opp_dir  : 73.9%  →  constant_slippage_with_labelling
  exit /same_dir : 44.8%  →  noisy_taker_proxy                  ← only data-supported leg
  exit /opp_dir  : 65.4%  →  constant_slippage_with_labelling

spread estimates (cents):
  entry  median +2.0¢   p10 +2.0¢   p90 +3.0¢   crossed_share 2.9%
  exit   median −2.0¢   p10 −8.0¢   p90  0.0¢   crossed_share 8.1%
  (medians at exactly ±2¢ = the assumed_spread_cents fallback. Confirms the
   distribution is dominated by synthetic-spread fallback rows, not data.)
```

## Window-extension test

```
window  fb_e_same  fb_e_opp  fb_x_same  fb_x_opp  edge_same  edge_mid  edge_opp
 300s    87.5%     73.9%     44.8%     65.4%     -0.0142   +0.0078   +0.0298
 600s    82.1%     64.1%     32.4%     55.0%     -0.0210   +0.0028   +0.0266

Δ 300s → 600s:
  fallback drop entry_same_dir: -5.4 pp
  fallback drop entry_opp_dir : -9.9 pp
  edge change same_dir  : -0.69¢/share  ← WORSE with longer window
  edge change midpoint  : -0.50¢/share  ← WORSE
  edge change opp_dir   : -0.32¢/share  ← WORSE
```

## Raw market activity post-cross

```
cap = 30 min, n = 8938
cumulative share of crosses with ANY next fill within X:
   <30s  :  5.6%
   <60s  : 11.0%
  <120s  : 17.2%
  <300s  : 28.7%   ← only ~29% have anything within 5 min
  <600s  : 38.4%
 none<=1800s: 42.7%   ← 43% have NO follow-up in 30 minutes

lag percentiles (conditional on having any fill):
  p10/p25/p50/p75/p90/p99 = 30 / 86 / 298 / 788 / 1315 / 1752 s
```

---

## Interpretation

1. **The pre-slippage backtest was a fiction.** The compounding bug alone moved the headline total return by 4 orders of magnitude; the entry-date dedup bug had two-way effects that masked themselves; the NaN-resolution silent-coercion was a latent landmine.
2. **The taker-scenario edge of −1.42¢ is mostly the 3¢ fallback constant.** Three of four legs have fallback rates above 50%; the diagnostic correctly flags them as "constant slippage with extra labelling." The number is therefore not a data-derived estimate of cost — it's the consequence of the choice of `fallback_cents`.
3. **The maker-scenario edge of +2.98¢ is similarly fallback-dominated.** Only `exit/same_dir` (44.8% fallback) clears the threshold; even there, the lag distribution shows median 66s and p90 likely well above 4 minutes — a loose proxy.
4. **A wider window does not save the proxy.** 10 min vs 5 min only drops fallback by 5-10pp on entry, and edges get worse — later fills carry drift-information, not execution-cost information. The any-fill diagnostic confirms 43% of crosses are followed by literal silence for 30 minutes.
5. **The qualitative conclusion is robust:** under realistic execution, the canonical FTC(0.60, 0.90) cell hovers around break-even ± a few cents. The precise sign and magnitude depend on what `fallback_cents` you choose, which is a modelling assumption, not a measurement.

## What's left / what to consider next

- **Sensitivity sweep on `fallback_cents`** ∈ {1, 2, 3, 5, 8}. Since fallback dominates, this is the most informative test of edge sign. Until that's done, "the edge is −1.42¢" should be read as "−1.42¢ under our fallback choice of 3¢."
- **Filter the universe to crosses with a real next-fill in 5 min** (~29% of anchors). Re-running the scenarios on that subset gives a smaller-N but cleaner read. The remaining 71% are markets where no proxy is going to recover real information.
- **Decide whether to even bother with the rest of the grid.** All grid cells share the same fallback-dominance problem; running the full 21-pair sweep doesn't tell you anything that's not already implied by the canonical cell + the same-direction edge ranking from the (no-slippage) pooled metrics.
- **If the strategy is ever to be deployed,** the execution-side latency assumption needs to drive `fallback_cents` (per the table in [`phase5_design.md`](phase5_design.md) §3.6). Polymarket WebSocket → 2-3¢; Goldsky polling → ≥5¢.

## Proposal B — data-supported subset analysis

Question: does filtering to entries where a real next-fill actually existed in 5 min (i.e. dropping the fallback-dominated rows) recover the edge? Answer: **no — it makes it worse.**

### 4-row summary table (canonical p_in=0.60, p_out=0.90)

| subset | n | share | p_tp | edge_same_dir | edge_midpoint | edge_opp_dir | mean_entry_slip¢ |
|---|---|---|---|---|---|---|---|
| baseline       | 8938 | 100.0% | 0.376 | −0.0142 (−2.4%) | +0.0078 (+1.3%) | +0.0298 (+5.0%) | 4.32 |
| taker_filt     | 1120 |  12.5% | 0.494 | **−0.0517** (−8.6%) | −0.0272 (−4.5%) | −0.0027 (−0.5%) | **5.49** |
| maker_filt     | 2330 |  26.1% | 0.445 | **−0.0816** (−13.6%) | −0.0439 (−7.3%) | −0.0062 (−1.0%) | **7.56** |
| intersection   |  881 |   9.9% | 0.507 | −0.0565 (−9.4%) | −0.0304 (−5.1%) | −0.0042 (−0.7%) | 4.89 |

**The crucial number is `mean_entry_slip_cents`**: the data-supported subsets show real entry slippage of **5-7¢**, vs the 3¢ assumed in the fallback model on the rest of the universe. The "active" markets aren't easier to fill — they're markets where price was *moving against us* (which is exactly why a follow-up print exists at all).

p_tp goes *up* on the subset (0.45-0.51 vs 0.376), but the higher slippage swamps the higher TP rate. Bucket dist isn't the bottleneck — pricing is.

### Selection bias (maker_filt subset, n=2330 = 26.1% of baseline)

| dimension | baseline | maker_filt | reading |
|---|---|---|---|
| top cities | London 12.7%, Seoul 5.9%, **NYC 5.9%**, Wellington 3.8%, **Ankara 3.8%** | London 10.3%, Seoul 8.2%, NYC **0.0%**, Wellington 5.5%, Ankara 4.6%, Shanghai 4.8% | NYC drops out entirely; Asian markets gain |
| hours-to-resolution at entry (p10/p50/p90) | 4.1 / 21.6 / 24.0 h | 2.0 / 11.0 / 24.0 h | subset trades much closer to expiry |
| hour-UTC mode | 12 UTC (40%) | 12 UTC (28%) + 04-06 UTC bump | Asian-session activity shifts in |

The data-supported subset is essentially **"Asian-session weather markets near resolution"** — a real operational filter, but one whose markets are exactly the most expensive to execute.

### Top 5 grid cells on `taker_filt` (the best-edge_midpoint subset)

```
sorted by edge_midpoint:
 p_in  p_out    n   p_tp  edge_same_dir  edge_midpoint  edge_opp_dir  roi_mid%
 0.90  0.95  1557  0.687    -0.0118       +0.0129       +0.0376      +1.44
 0.80  0.90  1388  0.664    -0.0235       +0.0072       +0.0379      +0.90
 0.85  0.90  1475  0.625    -0.0220       +0.0054       +0.0328      +0.63
 0.80  0.85  1388  0.570    -0.0240       +0.0046       +0.0332      +0.57
 0.80  0.95  1388  0.660    -0.0244       +0.0041       +0.0327      +0.51

sorted by edge_next_opp_dir (maker-best, optimistic):
 p_in  p_out    n   edge_same_dir  edge_midpoint  edge_opp_dir  roi_opp%
 0.80  0.90  1388     -0.0235       +0.0072       +0.0379      +4.73
 0.90  0.95  1557     -0.0118       +0.0129       +0.0376      +4.18
 0.80  0.85  1388     -0.0240       +0.0046       +0.0332      +4.15
```

Even on the data-supported subset, **only one cell shows marginal-positive midpoint edge**: (p_in=0.90, p_out=0.95) at +1.29¢/share (+1.4% ROI). Every taker-mode edge across the entire grid is negative. The opp_dir (maker-best) ROIs sit in the +3-5% range — comparable to baseline — but those are also fallback-dominated assumptions on the exit leg and shouldn't be relied on.

### Deployability verdict (3 sentences)

**No, the strategy isn't deployable at the canonical parameterisation.** The data-supported subset isn't a "cleaner" subset of the same edge — it's the subset where execution cost is *highest* (5-7¢ entry slippage observed vs 3¢ assumed), because the markets that print follow-up fills are the markets where price is actively drifting. The lone marginally-positive cell (p_in=0.90, p_out=0.95 on `taker_filt`, midpoint scenario, +1.4% ROI on ~1,500 trades/yr) is too thin and too contingent on the optimistic-execution assumption to bet capital on; shelve the FTC-TP weather strategy and revisit only if a fundamentally better fill model becomes available (e.g. CLOB-WebSocket-derived bid/ask quotes, not next-fill proxies).

## WS-passive execution model (revises the Proposal B verdict)

**Context change.** The original verdict assumed taker-style execution (cross the spread). With a CLOB WebSocket pipeline that can detect price and post passive limit orders at the touch, **the next-fill price proxies don't apply at all** — they were measuring the price other takers paid, not the price your posted limit would have filled at. Your posted bid at `p_in` is the best bid by construction; if any aggressive sell arrives within the holding window, it fills you at exactly `p_in`. Same on the exit leg at `p_out`.

The right metric for this case isn't "next-fill price" — it's **fill rate** combined with the per-trade math at exact prices.

### Two exit assumptions

- **optimistic**: if price reached `p_out` (`tp_fires=True`), assume our passive ask at `p_out` was lifted. Reasonable when our WS detection beats other passive sellers to posting.
- **strict**: also require a real aggressive-buy print in the exit window. Closer to "we got beaten in the queue at exit." Probably too pessimistic if you have WS-speed posting.

True performance lies somewhere between; it depends on your actual queue priority.

### Canonical (p_in=0.60, p_out=0.90) headline

| model | n_trades | edge_per_trade | ROI | notes |
|---|---|---|---|---|
| proxy taker (cross spread) | 8938 | −0.0142 | −2.4% | what we had |
| proxy maker_best (fallback-heavy) | 8938 | +0.0298 | +5.0% | not real — 74% fallback |
| **WS-passive (optimistic exit)** | **2330** | **+0.0052** | **+0.87%** | filled trades only; 26.1% fill rate |
| WS-passive (strict exit) | 2330 | −0.0152 | −2.5% | if always queued out at exit |

Bucket distribution among filled trades:
- optimistic: tp 0.445, hold_win 0.204, hold_chop 0.350
- strict:     tp 0.246, hold_win 0.363, hold_chop 0.391

Going from optimistic→strict drops the TP rate by 20 pp (0.445 → 0.246) — that's the "queue priority on the exit ask" cost.

### 21-pair grid scan under WS-passive (optimistic exit)

Top 5 cells by `edge_per_filled` (ROI on actual trades):

```
 p_in  p_out  n_filled  fill_rate  p_tp_of_filled  edge_per_filled  ROI%   $PnL/yr
 0.50  0.90    2283      0.275      0.382          +0.0300         +5.99   +68.4
 0.50  0.85    2283      0.275      0.410          +0.0271         +5.43   +61.9
 0.50  0.80    2283      0.275      0.431          +0.0240         +4.79   +54.7
 0.70  0.90    2419      0.268      0.561          +0.0186         +2.66   +45.1
 0.50  0.95    2283      0.275      0.332          +0.0169         +3.39   +38.6
```

The top three cells all share **p_in = 0.50**, which makes sense: lower entry price = more cross events to trade on (fill rate is similar at ~27.5%) AND more headroom to p_out, so more TPs land. The (0.50, 0.90) cell shows **+6% ROI per filled trade on 2,283 fills/yr ≈ 6/day**.

Grid output saved to [`data/analysis/passive_grid_canonical.parquet`](polymarket/research/data/analysis/passive_grid_canonical.parquet) for downstream use.

### Revised deployability verdict (3-4 sentences)

**Yes, there is a deployable signal — but it lives in a specific corner of the grid and is sensitive to your actual queue priority.** Under WS-passive execution at `(p_in=0.50, p_out=0.90)`, the canonical year yields ~2,283 fills with +6% ROI per trade (optimistic exit). The optimistic→strict gap (+0.87% → −2.5% ROI on canonical 0.60/0.90) shows queue priority on the exit is the dominant deployability question — recommend running small live for 2-4 weeks to measure your fill mix vs the data's exit_opp_dir distribution before scaling. Cities to monitor: Seoul, Shanghai, Tokyo, Wellington (the Asian-session names that dominate the active subset); the strategy effectively doesn't trade NYC/Ankara weather markets because they're too inactive post-cross.

### What this revises

- Previous verdict ("shelve the FTC-TP weather strategy") was premised on taker execution. With WS-passive, the strategy isn't dead — the (0.50, 0.90) corner is interesting and defensible, conditional on real fill-priority performance matching the optimistic assumption.
- The proxy-based numbers from Session 2.6 don't generalise to your case; they're a sensitivity range for crossing-the-spread execution, which you're not doing.
- The Proposal B finding still holds for *taker* execution: filtering to "active markets" makes taker performance worse, not better. For passive execution, "active markets" is just a description of where fills happen — by definition the trading set.

## Chase-best-bid cutoff sweep (realistic queue model)

The previous WS-passive section (above) treated "any aggressive sell in window = our fill." That's optimistic about queue priority. The honest model: our bid is only filled when the print actually executes at our price level. Under "chase up to p_in + N¢" execution:

- Cutoff = 0¢: bot stays at p_in. Fills only when the next bid-side print was AT p_in (or below — we wouldn't lower).
- Cutoff = +5¢: bot chases the best bid up to p_in + 5¢. Fills at the next bid-side print's price, if ≤ p_in + 5¢.
- Etc.

### Canonical (p_in=0.60, p_out=0.90) sweep — optimistic TP exit assumption

```
 cutoff¢  n_filled  fill_rate  p_tp_filled  mean_entry¢_above_p_in  edge_per_filled¢  ROI%   $PnL/yr
    0       1057      0.118       0.359              0.00                 -8.81      -14.7   -93.1
    1       1175      0.131       0.367              0.09                 -8.10      -13.5   -95.2
    2       1261      0.141       0.371              0.21                 -7.78      -13.0   -98.1
    3       1334      0.149       0.371              0.35                 -7.99      -13.3  -106.6
    5       1479      0.165       0.387              0.73                 -7.27      -12.1  -107.5
    8       1653      0.185       0.399              1.36                 -6.91      -11.5  -114.2
   12       1845      0.206       0.412              2.28                 -6.19      -10.3  -114.2
   20       2067      0.231       0.430              3.79                 -5.72       -9.5  -118.2
   30       2225      0.249       0.449              5.29                 -5.78       -9.6  -128.5
```

### Reading

- **The realistic stick-at-p_in fill rate is 11.8%, not 26%.** The earlier number counted prints at higher levels as our fills, which they aren't.
- **No positive cutoff exists.** Chasing higher does buy more fills (24.9% at cutoff +30¢) and a higher TP rate (44.9%), but each filled trade has a worse expected outcome: the chop drag at the higher entry price (`−entry_price` on chop) more than offsets the higher TP frequency.
- **The earlier +0.87% / +6% WS-passive headlines were artefacts of the optimistic queue assumption.** The real numbers at canonical are −10% to −15% ROI per filled trade under realistic queue priority.

### Revised verdict

**Do not deploy at canonical (0.60, 0.90).** The chase-bid sweep is unambiguous: every cutoff produces negative ROI under realistic execution. The strategy needs either:

- A different barrier pair where lower entry price (p_in = 0.50) creates more headroom for chop drag — run `chase_bid_cutoff_sweep` on an audit at p_in=0.50 to confirm.
- A different execution edge (genuine queue priority via co-location / order flow agreement) that closes the gap between the optimistic and realistic models.
- A different signal altogether — the level-break detection appears to be too late (price has already moved in the direction we're chasing).

## User-calibrated grid (sticky vs track-down) — the final read

Execution-model parameter on `eval_pair_user_passive` / `grid_user_passive`:

- **`entry_model='sticky'` (REALISTIC, default)** — bot posts at p_in cap, chases up only. In a crash, our high bid is hit FIRST (queue/latency mean we can't cancel fast enough). Fill = p_in. Captures adverse selection.
- **`entry_model='track_down'` (UPPER BOUND)** — bot continuously cancels and re-posts at best bid in both directions. Fill = bid_nf_price (the touch at fill time). Requires sub-second cancel-replace + queue priority. Caveat: a low `bid_nf` could mean a real crash (narrow spread at low level) OR a widened spread (lonely bid, ask still high) — under the latter, the "cheap fill" is into a one-sided dump, not a fair-price entry.

Common to both: exit ACTIVE (hit bid at p_out, receive p_out − 2¢ spread); sizing fixed at **5% of $10k initial bankroll per filled trade**.

### Sticky grid (realistic) — all 21 cells lose money

```
 p_in  p_out  n_filled  fill_rate  p_tp_filled  edge_per_filled  $PnL/yr    ROI%
 0.85  0.90    1,500     0.162      0.659       -0.0484         -$42,682  -427%
 0.85  0.95    1,500     0.162      0.619       -0.0507         -$44,724  -447%
 0.90  0.95    1,719     0.175      0.720       -0.0493         -$47,089  -471%
 0.80  0.90    1,398     0.155      0.599       -0.0622         -$54,350  -544%
 0.70  0.80    1,116     0.124      0.591       -0.0718         -$57,271  -573%
 ... (16 more cells, all negative) ...
 0.60  0.95    1,057     0.118      0.307       -0.1069         -$94,125  -941%
```

**Best (least bad) sticky cell: `(0.85, 0.90)` at −$42.7k / −427% ROI / Sharpe −5.04.** Note: bankroll is $10k; total loss of $42k means the model bankrupts you 4× over (capped in reality at full bankroll loss = −100%).

### Track-down grid (upper bound) — all 21 cells positive

```
 p_in  p_out  n_filled  p_tp_filled  edge_per_filled  $PnL/yr      ROI%
 0.80  0.90    1,398     0.599        +0.0321         $304,800     +3,048%
 0.70  0.90    1,116     0.510        +0.0214         $290,010     +2,900%
 0.85  0.90    1,500     0.659        +0.0356         $267,734     +2,677%
 0.60  0.90    1,057     0.359        +0.0100         $244,968     +2,450%
 ... (17 more cells, all positive) ...
```

**Best track-down cell: `(0.80, 0.90)` at +$304,800 / +3,048% ROI / Sharpe 1.41.**

### Sticky-vs-track-down gap = execution sensitivity

Per-cell gap ranges from $88k (at 0.90/0.95) to $359k (at 0.80/0.90). The gap is what your live test needs to resolve.

## LIVE TEST DESIGN — execution calibration, not profit

The deployment question is no longer "is the strategy profitable?" — it's "**does my bot achieve track-down behaviour often enough to flip the cell positive?**" Answer with a measurement test, not a profit test.

### Phase 1 (2-4 weeks, MEASUREMENT only)

- **Cell**: `(p_in=0.85, p_out=0.90)` — highest fill rate (16%), smallest sticky loss, smallest track-down win → narrow gap, fastest sample, bounded downside.
- **Size**: $10-50 per filled trade. *Not* 5% of bankroll. Goal is observation. Worst-case Phase 1 loss ≈ $50 × 30 losing trades = $1,500.
- **Logging** (ms resolution per trade):
  - `detection_ts`, `post_ts`, every `cancel_ts` / `repost_ts`, `fill_ts`, `fill_price`
  - Order-book snapshot (best bid, best ask, your queue position) every 5s
- **Per-fill computation**:
  - was your bid at the *current touch* or *stale above touch* at fill?
  - historical `bid_nf_price` at that ts (= sticky reference) — was your fill above, at, or below?
  - did you get queue priority, or were you behind another maker?

**Phase 1 success criterion: ≥50 filled trades.**

### Phase 1 readouts (the only numbers that matter)

| metric | sticky predicts | track-down predicts | what to do |
|---|---|---|---|
| `fill_rate` | ~16% | ~16% | sanity check — should match |
| `mean_entry_price` | ≈ 0.85 | ≈ 0.75 (−10¢) | which direction did your fills go? |
| `cancel_replace_latency_p90` | irrelevant | < 2s | if > 2s, you can't track-down |
| `pct_fills_at_or_below_market_touch` | ≈ 0% | ≈ 100% | direct mix measurement |

### Phase 1 decision rule

- **≥70% sticky-style fills**: shelve. Realistic numbers say -400% to -900% ROI/yr.
- **≥70% track-down-style fills**: proceed to Phase 2 at `(0.80, 0.90)` (max track-down PnL cell).
- **Mixed (40-70%)**: compute `expected_pnl = mix × td_pnl + (1−mix) × sticky_pnl` per cell. Pick the cell where measured mix ≥ breakeven mix; if no cell qualifies, shelve.

### Phase 2 (only if Phase 1 says go)

Scale to 5% sizing on a small sandbox bankroll ($1-2k) at the chosen cell for 4-8 weeks. Track measured PnL vs predicted track-down. If measured > 50% of predicted track-down, scale to full bankroll. Otherwise pull back — your execution doesn't generalise across the full universe.

### What you DON'T do

- Don't run a profit test at 5% sizing on day 1. Realistic numbers say bankroll is gone in ~130 trades.
- Don't pick the canonical (0.60, 0.90) cell for testing — it has the noisiest signal and largest sticky loss.
- Don't conclude after <50 fills.

## Tooling produced this round

- **Notebook**: [`notebooks/fill_scenarios_diagnostic.ipynb`](polymarket/research/notebooks/fill_scenarios_diagnostic.ipynb) — 13 cells, 8 charts + caveat, calls existing functions only. ~20s for the canonical eval_pair, ~50s for the window+any-fill diagnostics, plus the subset section (no extra parquet scans — reuses `scen`).
- **Script**: [`scripts/compare_fill_scenarios.py`](polymarket/research/scripts/compare_fill_scenarios.py) — text-only canonical report with full diagnostics + caveat; `--grid` flag does the 21-pair sweep.
- **Script**: [`scripts/subset_analysis.py`](polymarket/research/scripts/subset_analysis.py) — Proposal B orchestration: 4-subset summary, selection bias per subset, 21-pair grid scan on the highest-edge subset. ~7 min end-to-end.
- **Script**: [`scripts/test_next_fill_slippage.py`](polymarket/research/scripts/test_next_fill_slippage.py) — 5-anchor spot-check showing per-anchor next-fill found / not-found and lag.
- **Smoke test**: [`scripts/test_weather_module.py`](polymarket/research/scripts/test_weather_module.py) — full module smoke test, updated for the Session 2.6 renames; all 9 sections pass.
- **Helpers added to `data_infra/weather_analysis.py`** (Proposal B): `subset_pnl_summary`, `subset_selection_bias`, `grid_subset`.
- **Helpers added (WS-passive)**: `passive_pnl_from_audit`, `eval_pair_passive`, `grid_passive`. Audit-level function operates on the existing `scen` DataFrame — no extra parquet scans for the canonical view.
- **Script**: [`scripts/passive_analysis.py`](polymarket/research/scripts/passive_analysis.py) — canonical comparison (optimistic + strict exit) + 21-pair passive grid. ~7 min end-to-end. Persists grid output to `data/analysis/passive_grid_canonical.parquet`.
- **Function added (realistic queue)**: `wa.chase_bid_cutoff_sweep(audit, cutoffs_cents=...)` — operates on an existing audit log (no extra parquet scan). Models "chase best bid up to p_in + N¢" execution and produces fill rate / edge / total PnL vs cutoff. **This is the most honest read of the strategy** under the user's CLOB-WS execution capability.
- **`ftc_tp_sizing.py` extended (earlier this round)**: `backtest()` and `build_trades()` now accept `pnl_model='taker'|'passive'` and `exit_passive='optimistic'|'strict'`. Same equity / Sharpe / max DD / daily-return pipeline runs under both models.
- **`wa.eval_pair_user_passive` + `wa.grid_user_passive`**: user-calibrated execution with `entry_model='sticky'|'track_down'`, sized at fixed-fractional 5% of bankroll. Default sticky (realistic). See docstrings for the additional caveat about wide-spread interpretation of `bid_nf_price`.
- **`scripts/user_model_analysis.py`**: runs both sticky and track-down grids, persists to `data/analysis/user_passive_grid_{sticky,track_down}.parquet`. ~14 min if both grids need recomputing; instant if cached.
- **Notebook**: `notebooks/weather_tail_analysis.ipynb` rebuilt with §11 (user-calibrated grid + sticky-vs-track-down bar chart) and §10 (live-test design). 33 cells total.
