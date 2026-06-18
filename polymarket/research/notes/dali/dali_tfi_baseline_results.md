---
title: "Dali TFI Baseline Results"
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - dali
---
# Dali TFI Baseline Results
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


## Summary

- Scope: Dali TFI Baseline Results in the Dali research lineage area.
- Existing takeaway/status: Baseline TFI results note covering daily crypto up/down, equity index up/down, and AI/product markets. It records method, outputs, headline results, caveats, and the next step rather than introducing a deployable verdict.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Generated: 2026-05-23

Purpose: first non-ML test of whether historical Polymarket trade-flow
imbalance predicts future Polymarket price moves.

Script:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py --family daily_crypto_up_down
```

## Method

The script:

1. Selects a bounded set of markets from the cached market universe screen.
2. Materialises only those markets' raw `OrderFilled` rows.
3. Aggregates fills to market-second bars.
4. Tests whether signed flow predicts future VWAP price changes at
   30s, 120s, and 300s horizons.

Historical signed flow uses `maker_side`:

- `maker_side = BUY` -> positive signed maker USD.
- `maker_side = SELL` -> negative signed maker USD.

This is a proxy, not guaranteed aggressor direction. The output therefore
reports both:

- `maker_side`
- `inverse_maker_side`

If the inverse wins, the first interpretation should be "sign convention needs
live CLOB validation," not "we found an edge."

## Outputs

Crypto baseline:

- `data/analysis/csv_outputs/dali/dali_tfi_crypto_250_candidates.csv`
- `data/analysis/dali_tfi_crypto_250_seconds.parquet`
- `data/analysis/dali_tfi_crypto_250_eval.parquet`
- `data/analysis/csv_outputs/dali/dali_tfi_crypto_250_summary.csv`

Crypto, stricter end-buffer:

- `data/analysis/csv_outputs/dali/dali_tfi_crypto_250_exlast600_summary.csv`

Equity index:

- `data/analysis/csv_outputs/dali/dali_tfi_equity_index_100_candidates.csv`
- `data/analysis/csv_outputs/dali/dali_tfi_equity_index_100_summary.csv`

AI/product:

- `data/analysis/csv_outputs/dali/dali_tfi_ai_product_100_candidates.csv`
- `data/analysis/csv_outputs/dali/dali_tfi_ai_product_100_summary.csv`

## Headline Results

### Daily Crypto Up/Down

Run:

```bash
PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py \
  --family daily_crypto_up_down \
  --max-markets 250 \
  --prefix dali_tfi_crypto_250 \
  --exclude-last-seconds 120
```

Scale:

- 250 markets.
- 1,641,436 fills.
- 82,120 market-second bars.
- 237,330 horizon eval rows.

Summary with 120-second end buffer:

| horizon | best sign | obs | edge cents/share | hit rate |
|---:|---|---:|---:|---:|
| 30s | maker_side | 32,406 | +0.41 | 58.75% |
| 120s | maker_side | 31,815 | +5.26 | 67.36% |
| 300s | maker_side | 18,596 | +7.74 | 70.49% |

With a stricter 600-second end buffer:

| horizon | best sign | obs | edge cents/share | hit rate |
|---:|---|---:|---:|---:|
| 30s | inverse_maker_side | 12,766 | +1.01 | 42.94% |
| 120s | maker_side | 12,716 | +0.14 | 59.16% |
| 300s | maker_side | 12,615 | +3.23 | 65.82% |

Interpretation:

- Daily crypto is a good pipeline baseline.
- The naive signal is strongly contaminated by resolution-window dynamics.
- A large amount of the signal likely reflects external crypto price movement
  already being incorporated into Polymarket, especially near market end.
- This still supports using crypto as a harness, but it is not yet proof of
  alpha after costs.

### Equity Index Up/Down

Run:

```bash
PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py \
  --family daily_equity_index \
  --max-markets 100 \
  --prefix dali_tfi_equity_index_100 \
  --exclude-last-seconds 600
```

Scale:

- 100 markets.
- 164,003 fills.
- 55,519 market-second bars.
- 153,687 horizon eval rows.

Summary:

| horizon | best sign | obs | edge cents/share | hit rate |
|---:|---|---:|---:|---:|
| 30s | inverse_maker_side | 14,496 | +4.20 | 47.72% |
| 120s | inverse_maker_side | 14,124 | +4.41 | 48.34% |
| 300s | inverse_maker_side | 13,753 | +4.66 | 48.64% |

Interpretation:

- The sign convention flips versus crypto.
- Hit rate is below 50%, but the average signed move is positive because the
  payoff distribution is skewed.
- Do not trust this until live CLOB trade sign is validated and external SPX/SPY
  reference data is aligned.

### AI/Product

Run:

```bash
PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py \
  --family ai_product \
  --max-markets 100 \
  --prefix dali_tfi_ai_product_100 \
  --exclude-last-seconds 600
```

Scale:

- 100 markets.
- 380,287 fills.
- 117,975 market-second bars.
- 326,337 horizon eval rows.

Summary:

| horizon | best sign | obs | edge cents/share | hit rate |
|---:|---|---:|---:|---:|
| 30s | inverse_maker_side | 14,850 | +3.70 | 46.84% |
| 120s | inverse_maker_side | 13,426 | +3.48 | 46.58% |
| 300s | inverse_maker_side | 12,339 | +3.87 | 47.57% |

Interpretation:

- This looks structurally similar to equity-index in sign direction.
- Positive average signed move with sub-50% hit rate means skew/tails dominate.
- This family is still attractive, but the next test needs per-market and
  walk-forward splits, not just pooled averages.

## Caveats

1. **No costs yet.** These are price-change metrics, not tradable PnL.
2. **No spread/depth yet.** Historical fills do not include historical L2.
3. **Sign proxy is not validated.** Maker-side sign is not guaranteed aggressor
   direction.
4. **Resolution dynamics leak into short-horizon tests.** End buffers reduce
   this but do not fully solve it.
5. **Pooled results can hide market-level concentration.** Next pass needs
   per-market/family splits and time splits.

## Next Step

The next script should add:

- per-market summary rows,
- walk-forward date split,
- simple cost proxy,
- optional restriction to specific templates like `btc-updown-15m`,
- live CLOB sign validation against `last_trade_price` events.

This is enough to justify continuing the TFI/OFI research, but not enough to
justify live-money execution.
