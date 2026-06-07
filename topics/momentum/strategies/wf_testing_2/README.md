---
title: "Readme"
created: 2026-04-15
status: active
owner: justin
project: crypto
para: resource
hubs:
  - STRATEGY_REFERENCE
tags:
  - crypto
  - research
  - momentum
---
## Trend Follower with Caution Logic
> Hub: [[STRATEGY_REFERENCE]]

## Notebook Map

- [[topics/momentum/strategies/wf_testing_2/momentumADA_wf.ipynb|ADA walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumAVAX_wf.ipynb|AVAX walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumBNB_wf.ipynb|BNB walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumBTC_wf.ipynb|BTC walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumDOGE_wf.ipynb|DOGE walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumETH_wf.ipynb|ETH walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumLINK_wf.ipynb|LINK walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumSOL_wf.ipynb|SOL walk-forward]]
- [[topics/momentum/strategies/wf_testing_2/momentumXRP_wf.ipynb|XRP walk-forward]]


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
- **C1 — Caution Long (Overextension):**  
  Detects **"buying exhaustion."**  
  **OBV Divergence:** Price is rising, but On-Balance Volume is below its average (hollow move).  
  **Price Stretch:** Current Low is > $1.5 \times ATR$ away from the Swing High (mean-reversion risk).

- **C2 — Caution Short (Bear Exhaustion):**  
  Signals that **shorting is now risky.**  
  **Trend Breach:** Price has climbed above the EMA.  
  **Downside Stretch:** Current High is > $1.5 \times ATR$ away from the Swing Low.  
  **Role:** Acts as a *"Structural Floor"* indicating a shift back to bullish/neutral bias.

- **C3 — Caution "Both" (Volatility/Chop):**  
  Triggered when **C1 and C2 are active simultaneously.**  
  Indicates extreme, directionless volatility and "long wicks" in both directions.  
  **Action:** Applies a conservative, wider stop multiplier (`stop_mult_ent_both`) to survive market noise.
---

## C3 — Caution "Both" (Volatility/Chop)
Triggered when **C1 and C2 are active simultaneously.**

- Indicates extreme, directionless volatility and "long wicks" in both directions.

**Action:** Applies a conservative, wider stop multiplier (`stop_mult_ent_both`) to survive market noise.
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
