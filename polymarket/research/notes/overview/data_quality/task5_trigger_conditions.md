---
title: Dali Task 5 Trigger Conditions
created: 2026-05-23
status: watching
owner: justin
project: polymarket
para: project
hubs:
  - COWORK
tags:
  - dali
  - trigger-conditions
  - parameter-search
  - research
---

# Dali Task 5 Trigger Conditions

> Hub: [[COWORK]]

Generated: 2026-05-23

## Summary

This note defines when Dali Task 5 parameter search should start. The task is deliberately deferred until capture coverage, trade events, and sign-normalization evidence reach stated thresholds. Until then, Dali work should stay focused on capture volume, OFI correctness, sign normalization, and backtest-engine correctness.

Task 5, parameter search over dumb-baseline rules, is deliberately deferred.
The current live sample is too small: 5 minutes, 123 raw messages, and only one
`last_trade_price` event.

## Trigger Conditions

Do not start parameter search until all are true:

- At least 3 market families captured.
- At least 24 hours total capture per family.
- At least 200 `last_trade_price` events in the combined dataset.
- Sign-convention audit has at least 50 classifiable live trades, or the
  strategy avoids live trade-side normalization entirely.

## Future Search Shape

When triggered:

- Use Optuna with a TPE sampler, not a fixed 2160-cell grid.
- Start with 200-500 trials.
- Use chronological train/validation/test split: 60% / 20% / 20%.
- Objective: net-of-cost Sharpe.
- Require at least 100 test-set trades before treating Sharpe as meaningful.
- Report test-set results once; do not optimize on the test set.

Until these conditions are met, Dali should focus on capture volume, sign
normalization, OFI correctness, and backtest-engine correctness.
