---
title: "Dali Live L2 Capture Plan"
created: 2026-06-05
status: watching
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - research
  - dali
---
# Dali Live L2 Capture Plan
> Hub: [[COWORK]]


## Summary

- Scope: Dali Live L2 Capture Plan in the Dali research lineage area.
- Existing takeaway/status: Plan and runbook for Dali live L2 capture, connecting prior A0/A0b capture work to the next live smoke and deferring parameter search until capture semantics are stable.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
Generated: 2026-05-23

This note is the concrete version of the prior recommendation: use historical
fills for fast TFI research, but start live CLOB market-state capture now
because historical data does not contain true L2 order book state.

## What This References

The earlier recommendation was:

- market universe screen first,
- fill-only TFI baseline second,
- live L2 capture immediately in parallel,
- paper-trade a simple baseline before ML,
- only then move to LightGBM or richer external-data models.

The bottleneck is now live market-state truth: book depth, top-of-book spread,
trade side, market-channel latency, and whether our historical `maker_side`
sign proxy maps to the live `last_trade_price.side` field.

## What To Do Now

2026-06-06 collection caveat: public PM CLOB L2 is anonymous aggregate price-level data, not wallet/order-owner truth. Before using live JSONL for cancel-vs-take attribution, read [[mm_clob_capture_semantics]] and report clean vs ambiguous reconstruction rates.

1. Pick a small watchlist from the current Dali universe file.

   Start with `ai_product` or curated geopolitics/policy for alpha research,
   and use crypto only as a pipeline stress test.

2. Start CLOB market-channel capture.

   ```bash
   cd polymarket/research
   PYTHONPATH=. uv run python scripts/dali_live_clob_capture.py \
     --family ai_product \
     --max-markets 5 \
     --duration-seconds 300
   ```

   Output lands under `data/live_clob/` as JSONL plus a manifest. It is
   intentionally raw so we can replay events as the feature logic changes.

3. Run the fill-only TFI baseline on the same family.

   ```bash
   cd polymarket/research
   PYTHONPATH=. uv run python scripts/dali_tfi_baseline.py \
     --family ai_product \
     --max-markets 100 \
     --prefix dali_tfi_ai_product_100 \
     --exclude-last-seconds 600
   ```

4. Build the first replay feature pass from the live JSONL.

   ```bash
   cd polymarket/research
   PYTHONPATH=. uv run python scripts/dali_clob_replay_features.py --latest
   ```

   Required features:

   - `best_bid`, `best_ask`, `spread`,
   - top-N bid/ask depth,
   - OFI over 5s/15s/60s windows,
   - `last_trade_price.side`, `price`, `size`,
   - event receive lag versus exchange timestamp,
   - price bucket and time-to-end from Gamma metadata.

5. Paper trade a dumb rule before ML.

   Candidate rule: only signal when signed flow, short price momentum, and OFI
   agree; skip wide spreads, micro-price boundary markets, and low-depth books.
   Evaluate after spread/slippage, not just future price delta.

## Why This Is The Right Order

The historical TFI results are promising enough to continue, but not clean
enough to trade. They are missing spread/depth/costs and the sign convention
flips across families. Live CLOB capture directly resolves those questions.

The model should come after this. LightGBM is useful once we know the event log
has stable labels and costs; before that, it mostly hides data-quality problems.

## First Live Smoke Result

Run on 2026-05-23:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/dali_live_clob_capture.py \
  --family ai_product \
  --max-markets 5 \
  --duration-seconds 300
```

Output:

- Raw capture: `data/live_clob/dali_clob_ai_product_20260523T160956Z.jsonl`
- Manifest: `data/live_clob/dali_clob_ai_product_20260523T160956Z.manifest.json`
- Event count: 123 raw messages.
- Event mix: 12 `book`, 21 `new_market`, 87 `price_change`,
  2 `best_bid_ask`, 1 `last_trade_price`.
- Watchlist: Mistral, xAI, OpenAI IPO, Anthropic, and Google AI-model markets.

Replay command:

```bash
PYTHONPATH=. uv run python scripts/dali_clob_replay_features.py --latest
```

Replay output:

- Feature file:
  `data/analysis/dali_clob_features_ai_product_20260523T160956Z.parquet`
- Rows: 189 per-asset feature rows.
- Assets: 10 CLOB token IDs.
- Replay event mix: 174 `price_change` rows, 12 `book` rows,
  2 `best_bid_ask` rows, 1 `last_trade_price` row.
- Price-change receive lag was tight: median about -71 ms, p90 about 262 ms,
  p99 about 873 ms. The negative median is local/exchange clock skew, not
  negative latency.
- Initial `book` snapshot timestamps can be stale, so latency reporting should
  separate initial snapshots from live top-of-book/trade updates.

Current replay limitations:

- Full depth is refreshed only from `book` events.
- `price_change` and `best_bid_ask` update top-of-book only; they are not a
  full depth reconstruction.
- True OFI needs a longer capture with repeated depth/top-of-book transitions.
  This first replay is enough to validate ingestion, spread/depth fields,
  trade side, and rough receive-lag handling.

## Parameter Search Is Deferred

Do not run grid search or Optuna yet. Task 5 starts only after the trigger
conditions in `notes/overview/data_quality/task5_trigger_conditions.md` are met: at least 3 captured
families, 24h+ capture per family, 200+ combined `last_trade_price` events, and
enough classifiable trades to establish or deliberately avoid live sign
normalization.
