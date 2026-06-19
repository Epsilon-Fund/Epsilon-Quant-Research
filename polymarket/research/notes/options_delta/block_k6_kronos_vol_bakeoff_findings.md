---
title: Block K6 Kronos Vol Bake-Off Gate
created: 2026-06-05
status: closed
owner: justin
project: polymarket
para: archive
hubs:
  - COWORK
  - strat_options_delta
tags:
  - options-delta
  - block-k
  - volatility
  - kronos
  - research
---

# Block K6 Kronos Vol Bake-Off Gate

> **Strat:** [[strat_options_delta]] (Options-Delta). Sibling: [[strat_market_making]]. Arc: [[block_k_plain_english_synthesis]].

## Summary

This note checks whether the Kronos/HAR/EWMA forward-vol bake-off should run. It stops at the prerequisite gate because the repo did not yet contain a cleared static-hedge Strategy-A OOS result, and the closest K6 hedge artifact was negative. The later static-hedge note exists, but it also fails the gate, so the bake-off remains blocked.

## Headline

Stopped at the prerequisite gate. I did not run HAR-RV, GARCH, or Kronos because the repo does not contain an already-cleared static-hedge Strategy-A OOS result on the K6/K3 panel.

The available K6 artifact is the banded-rehedge diagnostic, not a static-hedge OOS Strategy-A result. In that artifact, zero strict-source bucket/config cells have positive net-of-cost lower CI after Polymarket fee plus Binance hedge turnover.

## Gate Evidence

- Searched local notes, scripts, and analysis outputs for `Strategy-A`, `Strategy A`, `static hedge`, `static-hedge`, `OOS`, and K6/Kronos terms.
- Found K6 vol artifacts:
  - `data/analysis/csv_outputs/options_delta/k6_vol_gap.csv`
  - `data/analysis/csv_outputs/options_delta/k6_gamma_scalp_trades.csv`
  - `data/analysis/k6_vol_gap_panel.parquet`
  - `notes/options_delta/block_k6_vol_findings.md`
- Found no static-hedge Strategy-A OOS result artifact.
- Current K6 note headline: no strict-source `(|z|, tau)` bucket clears zero after Polymarket fee plus banded Binance hedge turnover.
- Best strict K6 bucket/config in the available artifact is `far_absz_ge1|late_lt30m`, latency `5s`, entry gap `20 vol pts`, band `10c`: mean net `-9.39c`, CI `[-19.05c, -1.13c]`.

## Decision

Do not run Kronos. The requested decision rule says Kronos should only be tested after the static-hedge Strategy-A variant has already cleared net-of-cost OOS. That prerequisite is absent locally, and the closest existing K6 net-of-cost hedge result is negative. A forward-vol forecaster cannot rescue a strategy whose failure is driven by hedge/cost mechanics rather than estimator ranking.

## What Would Be Needed Before Kronos

Run or provide an explicit static-hedge Strategy-A OOS ledger on the K6/K3 panel with:

- non-overlap entries,
- PM taker fee at entry only,
- one static Binance hedge set at entry and closed at resolution,
- strict source filter,
- bucket table by `|z| x tau`,
- OOS split or an artifact that clearly labels the OOS sample,
- far/late `far_absz_ge1|late_lt30m` lower CI above zero net of costs.

Only after that gate clears should the bake-off compare EWMA, HAR-RV/GARCH, and Kronos on net-of-cost PnL and tail calibration.

## Recheck

Repeated the gate check after the bake-off request was reissued. I still found no static-hedge Strategy-A OOS ledger or note showing a cleared net-of-cost result on the K6/K3 panel.

Additional local evidence:

- `notes/overview/synthesis/block_k_plain_english_synthesis.md` frames Strategy A as the remaining carry-to-resolution/static-hedge hypothesis, but explicitly says the static hedge has not yet been tested and still needs a rigorous OOS-gated test.
- `data/analysis/csv_outputs/options_delta/k6_vol_gap.csv` has 228 strict `gamma_bucket` rows from the available K6 diagnostic; 0 have positive lower CI with `n_trades >= 5`.
- The best available strict K6 bucket/config remains `far_absz_ge1|late_lt30m`, latency `5s`, entry gap `20 vol pts`, band `10c`: mean net `-9.39c`, CI `[-19.05c, -1.13c]`.

Decision unchanged: Kronos/HAR/EWMA bake-off remains gated off until the static-hedge Strategy-A prerequisite clears.

## Static-Hedge Gate Result

The missing static-hedge Strategy A test now exists in `notes/options_delta/block_k6_strategy_a_static_hedge_findings.md`.

Result: the gate still does not clear. Strict OOS `far_absz_ge1|late_lt30m` has n=11, mean net `+1.07c`, CI `[-1.04c, +3.99c]`. The mean is positive after replacing continuous rehedging with one static Binance hedge, but the lower CI is below zero, so it does not satisfy the preregistered far/late OOS lower-CI gate.

Decision unchanged: do not run the EWMA vs HAR-RV vs Kronos forward-vol bake-off.
