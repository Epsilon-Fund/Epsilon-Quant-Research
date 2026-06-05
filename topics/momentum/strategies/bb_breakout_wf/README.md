# Stage 1 — 4H Setup Detection
> Hub: [[STRATEGY_REFERENCE]]

## Notebook Map

- [[topics/momentum/strategies/bb_breakout_wf/ADA.ipynb|ADA walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/AVAX.ipynb|AVAX walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/BTC.ipynb|BTC walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/DOT.ipynb|DOT walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/ETH.ipynb|ETH walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/LINK.ipynb|LINK walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/MATIC.ipynb|MATIC walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/NEAR.ipynb|NEAR walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/SOL.ipynb|SOL walk-forward]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC.ipynb|BTC strategy design]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_2.ipynb|BTC strategy design 2]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_3.ipynb|BTC strategy design 3]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_3.2.ipynb|BTC strategy design 3.2]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_3long.ipynb|BTC strategy design 3 long]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_4.ipynb|BTC strategy design 4]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC_5.ipynb|BTC strategy design 5]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC34.ipynb|BTC34 strategy design]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC34_2.ipynb|BTC34 strategy design 2]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC345.ipynb|BTC345 strategy design]]
- [[topics/momentum/strategies/bb_breakout_wf/strategy_design/BTC345_2.ipynb|BTC345 strategy design 2]]


The 4H timeframe acts as the **"Engine Room"** to filter for high-conviction momentum. Three conditions must align:

## Condition 1: Persistent Volatility Breakout
Unlike static ATR multiples, this version uses a percentile-based threshold (`breakout_pct` over a lookback).  
Two consecutive 4H candles must be **"Big"** (exceeding that percentile) and directional:
- Both green for long  
- Both red for short  

## Condition 2: BB Expansion
The Bollinger Band width must be greater than its recent rolling average (`bb_exp_window`).  
This ensures entry during **"volatility expansion"** rather than a late-stage pinch.

## Condition 3: Scale-Invariant Slope
The 4H SMA must be sloping in the direction of the trade (within a `slope_epsilon` tolerance).  
The slope is normalized by price to maintain consistency across different price regimes.

---

# Stage 2 — 1H Entry State Machine

Once the 4H momentum is confirmed, the strategy shifts to the 1H timeframe to **"buy the dip."**

## Expiry (Invalidation) Conditions
The setup is discarded if any of these **"spoiler"** conditions are met:

- **E1 — Time Decay:** More than `max_1h_bars` pass without an entry.  
- **E2 — Volatility Spike:** A pullback candle's range exceeds `pullback_atr_mult * ATR`.  
- **E3 — Structural Break:** Price overshoots the 1H SMA by more than `overshoot_bps`.  

## Entry Conditions

- **P1 — The Entry Zone:** Price must be within `entry_zone_bps` of the 1H SMA.  
- **P2 — Momentum Resumption:** The 1H candle must close in the direction of the trade:  
  - Higher close for long  
  - Lower close for short  

---

# Stage 3 — Trade Management & Exits

## Unified Stop & Trail
The strategy uses a **"ratchet" trailing stop** based on `trail_atr_mult * ATR`.  
It updates bar-by-bar and **never moves backward**.

## Regime-Aware Profit Taking
The strategy switches its exit logic based on the macro trend (Trend MA + ADX):

- **Strong Bull Regime:**  
  The Take Profit is removed (`0.0`).  
  The strategy rides the move until the trailing stop is hit.

- **Chop/Bear Regime:**  
  The strategy applies a fixed **6:1 Reward-to-Risk ratio** to harvest gains before a potential reversal.

## Bull-Market Short Filter
Short setups are ignored if:
- Price is above the Macro Trend MA, or  
- ADX indicates a strong bullish trend  


Once a position exits, setup_active needs to fire true again, so requires 2 4h candles satisfying step 1 before moving on. This is done to prevent overfitting and revenge trading, repeatedly entering on a bad 4h signal.
---

# Position Sizing

## The Formula
$$
\text{Position Size} = \frac{\text{Risk Per Trade}}{\frac{\text{ATR}}{\text{Price}}}
$$

## How It Works

- **Volatility Scaling:**  
  Uses the 1H ATR to calculate the "volatility percentage."  
  - High volatility → lower leverage  
  - Low volatility → higher leverage  

- **Leverage Cap:**  
  The final size is clipped between **0.1x** and your `max_leverage` parameter to prevent extreme exposure.
