## Stage 1 — 4H Setup Detection

The 4H timeframe acts as the "Engine Room." It filters for high-conviction momentum. Three conditions must align:

### Condition 1: Persistent Volatility Breakout
Unlike the original which used a static ATR multiple, this version uses a percentile-based threshold (`breakout_pct` over a lookback). Two consecutive 4H candles must be "Big" (exceeding that percentile) and directional (both green for long, both red for short).

### Condition 2: BB Expansion
The Bollinger Band width must be greater than its recent rolling average. This ensures you are entering as the "jaws" of the market open, rather than during a late-stage blow-off where bands are starting to pinch.

### Condition 3: Scale-Invariant Slope
The 4H SMA must be sloping in the direction of the trade. The slope is normalized by price, ensuring the filter remains effective whether BTC is at $20k or $100k.

---

## Stage 2 — 1H Entry State Machine

Once the 4H momentum is confirmed, the strategy shifts to the 1H timeframe to "buy the dip."

### Expiry (Invalidation) Conditions

Before looking for an entry, the state machine checks if the opportunity has "spoiled":

- **E1 — Time Decay:**  
  If `max_1h_bars` pass without an entry, the setup is discarded.

- **E2 — Volatility Spike:**  
  If a pullback candle is too large (`pullback_atr_mult`), it indicates a panic reversal rather than a controlled drift.

- **E3 — Structural Break:**  
  If the price overshoots the 1H SMA by too many basis points (`overshoot_bps`), the mean-reversion has turned into a trend change.

### Entry Conditions

- **P1 — The Entry Zone:**  
  Price must be within a specific basis-point distance (`entry_zone_bps`) of the 1H SMA.

- **P2 — Momentum Resumption:**  
  Instead of complex candle patterns, the strategy now looks for a directional close.  
  - For a long: the current 1H close must be higher than the previous 1H close.  
  - For a short: the current 1H close must be lower than the previous 1H close.  

This confirms the "drift" has ended and momentum is resuming.

---

## Stage 3 — Trade Management & Exits

This is where the most significant simplifications occurred, moving to a "set-and-trail" philosophy.

### Unified Stop & Trail
There is no longer a separate "hard stop" and "trailing stop." At entry, an ATR-based stop is set (`trail_atr_mult`). As price moves in your favor, this stop ratchets up (or down) bar-by-bar. It never moves backward.

### Regime-Aware Profit Taking

The strategy switches its exit logic based on the macro trend:

- **Strong Bull Regime:**  
  *(Price > Trend MA + Strong ADX)*  
  The strategy removes the Take Profit entirely. It assumes a "moon bag" stance, riding the move until the trailing stop is eventually hit.

- **Chop/Bear Regime:**  
  The strategy applies a fixed 5:1 Reward-to-Risk ratio. It harvests gains quickly, assuming that follow-through will be limited.

### One-Shot Logic
The re-entry mechanism is removed. Once a trade is closed via Stop or TP, the 4H setup is considered "spent." You must wait for a brand new 4H breakout signal to trade again, preventing "revenge trading" in a failing zone.

## Position Sizing

### The Formula

The code calculates the raw position size as:

$$
\text{Position Size} = \frac{\text{Risk Per Trade}}{\frac{\text{ATR}}{\text{Price}}}
$$

### How It Works (Plain English)

- **ATR / Price:**  
  This calculates the *volatility percentage*. How much asset moves as fraction of its cost.
  *(e.g., if ATR is $5 and price is $100, volatility is 5%)*

- **Risk per Trade:**  
  Desired loss/exposure, e.g if you want to risk 1% of your account, and the coin moves 5% on average:  
  $$
  1 / 5 = 0.2x \text{ leverage}
  $$

In this strategy, the sizing is calculated on the **1H timeframe** to match the "pullback" entry logic.

### Volatility % Chunk (P-BB)
It uses the `h1_atr` (the 1H Average True Range) to determine how "noisy" the current market is.

### Context
Since this strategy enters during a pullback, volatility is often **expanding**, as it typically follows a 4H breakout.


### The Safety Buffer

The result is then clipped:

```python
.clip(0.1, params['max_leverage'])
