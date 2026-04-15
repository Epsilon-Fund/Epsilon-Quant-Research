## Trend Follower with Caution Logic

### Entry Conditions (The Gatekeepers)

- **E1 — EMA Filter:**  
  The price must be trending up. Specifically, `Close > EMA`.  
  If you aren't above the mean, there’s no trade.

- **E2 — Volume Validation:**  
  `Volume > Volume_MA`.  
  This ensures the move has "gas in the tank" and isn't just a low-liquidity drift.

- **E3 — The Caution Gate:**  
  Entry is blocked if `Caution_Long` is `True`.  
  However, there is a **"Power Clause"**: if `ADX > adx_override`, the strategy ignores the caution and enters anyway, betting on pure parabolic momentum.

- **E4 — Structural Validity:**  
  The system checks that indicators like Swing Highs and ATRs are actually calculated (non-NaN) before firing.

---

## Stop Loss Logic (The Ratchet)

Note the distinct logic between the moment of entry and the management of the trade afterward.

- **S1 — Entry Day Stop (The Anchor):**  
  On the bar of entry, the stop is calculated once based on the current state:

  $$
  \text{Stop Loss} = \text{Swing\_Hi\_Stp} - (\text{ATR\_Stp} \times \text{Multiplier} \times \text{Scale})
  $$

  The multiplier is selected from three options (**normal, caution, or both**) depending on the market's "nervousness" at that exact moment.

- **S2 — Position Day Stop (The Trailing Ratchet):**  
  For every bar after entry, the stop is dynamic.

  - It uses a `"pos"` (position) multiplier instead of an `"ent"` (entry) multiplier.  
  - If `Caution_Long` triggers while you are in the trade, the multiplier switches to `stop_mult_pos_caution` (usually tighter).  

  **The Ratchet Rule:**  
  The stop can only move up. It takes the `max()` of the previous stop and the new calculation.

---

## The Caution Logic (Internal Circuit Breakers)

- **C1 — OBV Divergence:**  
  If the price is making new highs but the On-Balance Volume is below its average, the move is considered "fake" or unsupported.

- **C2 — Overextension:**  
  If the distance between the recent high and current low exceeds $1.5 \times ATR$, the rubber band is stretched too far.  
  Entry is flagged as risky.

---

##Position Sizing

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

In this strategy, the sizing logic is more **modular and customizable**.

###  Volatility % Chunk (P-TF)
It uses a specific parameter called `ATR_Sz`, calculated via:

```python
params['atr_size']
```

### The Safety Buffer

The result is then clipped:

```python
.clip(0.1, params['max_leverage'])