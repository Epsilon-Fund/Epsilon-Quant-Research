---
title: "OD Methodology Realism Audit: Where The Pricing Work Can Fool Us"
created: 2026-06-07
status: active
owner: justin
project: polymarket
para: project
hubs:
  - strat_options_delta
  - COWORK
tags:
  - research
  - options-delta
---
# OD Methodology Realism Audit: Where The Pricing Work Can Fool Us

> Hub: [[strat_options_delta]] · [[COWORK]]
> Related: [[od_strategy_a_v3_findings]] · [[od_strategy_a_realism_reaudit_findings]] · [[od_conditional_prob_calibration_findings]] · [[od_pricing_model_form_findings]] · [[od_equities_index_pricing_scope_findings]] · [[block_k6_vol_findings]]
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- The main OD pricing issue is that much of the old "fair" language means **realized-vol / physical-probability fair**, not **option-implied / risk-neutral fair**.
- `token_model_fair` in Strategy A is mostly a causal EWMA realized-vol probability (`P(resolve above strike)`), so calling it option fair can overstate whether a Polymarket quote is truly rich or cheap.
- `pm_iv_annualized` is useful as a diagnostic, but it is just the Polymarket midpoint rewritten as an implied vol. It is circular if used as the fair price for the same PM market.
- The surviving OD use case is not standalone "vol mispricing alpha"; it is a cautious valuation / sizing overlay for source-clean passive MM and carry measurement.
- Before reopening OD live, the code and notes should separate three objects: `physical_prob_fair`, `external_iv_fair`, and `pm_mid_implied_vol`.

## Audit Scope

This audit reviewed the OD pricing path and the adjacent realism notes/scripts:

- `dali_block_k3v3h_hedged_basis.py`
- `dali_block_k6_vol_gap.py`
- `od_strategy_a_v3.py`
- `od_strategy_a_v3_pnl_risk.py`
- `od_v4_calibration_gate.py`
- `od_v4_queue_replay.py`
- `od_conditional_prob_calibration.py`
- `od_pricing_model_form.py`
- `od_same_day_crypto_pricing_gate.py`
- SPX / Deribit / equities-index OD notes

## Methodology Issues To Fix Or Guardrail

| Issue | Where it shows up | Why it is unrealistic or subtly wrong | Guardrail |
| --- | --- | --- | --- |
| RV fair mistaken for option fair | K3/K6/Strategy A v3/PnL risk path | `p_model = N(z)` from EWMA realized vol is a physical forecast, not a risk-neutral external option price. It can be a useful forecast, but it is not the same thing as option-implied fair. | Rename outputs to `physical_prob_fair` or `rv_model_prob`; only call something option fair when it comes from a settlement-aligned external IV/surface or a proven calibrated option model. |
| PM implied vol circularity | K6 `pm_iv_annualized` and vol gap diagnostics | Inverting the PM midpoint into IV explains the PM price in vol units. Using that same value as fair makes edge mostly a restatement of the quote/spread. | Use PM IV only as a diagnostic: compare PM-implied vol against forecast sigma or external IV, never as the fair for the same market. |
| N(z), N(d2), and `ewma_nd2` label drift | Older K3/K3v2 naming versus later K3v3/Strategy A | Some labels imply Black-Scholes `d2`, but later code/prose uses a simpler normalized distance `z`. Mixed labels make backtests look more comparable than they are. | Standardize model names before comparing rows: `ewma_nz`, `ewma_nd2`, `empirical_conditional`, `external_iv_surface`, etc. |
| Fair-value-scaled sizing inherits model error | `od_strategy_a_v3_pnl_risk.py` sizing | If `claim_edge` is measured against RV fair, then "fair-value-scaled" size is only as good as that RV model. Bad fair means bad size. | Rename to `rv_edge_scaled` until an external/calibrated fair exists; cap sizing and shadow-run alternative fair definitions. |
| Source-clean lifecycle owns more edge than valuation | v3/v4 source-vs-valuation diagnostics | The strict rich-short minus strict-source diagnostic was negative on average (`-40.47c/episode`, CI roughly `[-120.93c, 1.49c]`). Selection edge did not clearly beat lifecycle/source cleaning. | Treat OD as a selection/sizing overlay, not independent alpha, until incremental edge survives source-clean controls. |
| Small sample and concentration | Strategy A v4 and touch/Arm T work | Some positive point estimates are built from a few fills or a dominant market. The biggest market sometimes explains most PnL. | Report market-cluster CI, leave-one-market-out, and per-underlying splits; do not promote concentrated estimates to live rules. |
| Global/per-asset embargo confusion | Strategy A v4 calibration gate | One-position-global and per-asset embargoes answer different capital questions. A weak per-asset result can be underpowered, not proof of no edge. | Label capital assumption separately from signal validity; report both if sample size allows. |
| Borrowed queue baseline as kill switch | Queue replay / realism gate | A `1.98c` queue baseline borrowed from crypto-4h MM is not necessarily valid for OD touch/Strategy A fills. | Keep borrowed baselines as diagnostics until the same instrument, venue, queue, and time-to-expiry family has its own baseline. |
| Capacity haircuts as hard law | Liquidity/capacity notes | Non-top3 share and fill-ratio assumptions are useful, but they are not laws of nature. Treating a haircut as fixed can over-kill viable tiny strategies or over-trust optimistic variants. | Publish a sensitivity ladder: 1%, 2%, 5%, 10% share; top-3 and non-top3; live-capture-only unknowns clearly separated. |
| Actual fills versus executable quotes | SPX replay, K-PEG lessons, PM historical fills | Last trade, midpoint, or historical matched fills are not the same as executable best bid/ask with queue priority. | Require bid/ask/depth/queue evidence for executable claims; otherwise label as indicative pricing only. |
| External reference and settlement mismatch | Binance/Chainlink, Deribit/PM settlement, SPX/PM close windows | The source used for modeling can differ from the settlement source or settlement timestamp. This creates hidden basis risk. | Add source-basis filters and settlement-aligned capture before calling an edge tradable. |
| 5m history versus 1s live state | Conditional probability calibration | Multi-year 5m truth tables are strong priors, but they miss local jump risk, source lag, OFI, and queue conditions. | Use historical 5m as the prior/control and captured 1s live data as validation. |
| Barrier/touch tails under-sampled | Touch/Arm T audit | Touch outcomes are sparse and can have near-total loss given default. An all-winner sample is not enough to infer a live loop. | Keep touch size tiny; require explicit barrier/touch acceleration logs and pessimistic loss assumptions. |
| Hedge overlays can hide cost | Hedged basis / gamma-scalp variants | Continuous and banded hedges can manufacture theoretical variance reduction while turnover destroys edge. | Do not count a hedge as alpha unless net-of-turnover CI is positive. Static hedges should stay a variance footnote unless proven otherwise. |
| Mark-to-mid and last-trade substitution | K-PEG, SPX, and OD replay lessons | Midpoint or last-trade marks can make a strategy appear smoother than what could actually be executed or held to resolution. | Prefer resolution PnL, actual fills, or contemporaneous executable quotes. Label mark-to-mid as diagnostics only. |

## RV vs IV Specific Finding

The OD code did **not** accidentally use PM IV as the main fair price. The main Strategy A path used realized-vol forecast probabilities:

- K3/K3v3 build `p_model = N(z)` from causal EWMA annualized sigma.
- Strategy A v3 carries that into `token_model_fair`.
- PnL/risk sizing then uses `claim_model_fair` and `claim_edge`.
- K6 separately inverts PM midpoint into `pm_iv_annualized`, mostly as a diagnostic.

That means the core mistake is semantic and methodological: the system often said "fair" where it meant "my physical RV forecast probability." That can be useful, but it is not the same as risk-neutral option fair.

A cached strict-source far-short diagnostic showed the practical consequence:

- Strict far-short rows: `157`
- Rich by at least `1c` versus RV model fair: `23` fills across `8` markets
- Rich by at least `1c` versus PM-mid / PM-IV proxy: `31` fills across `9` markets
- Overlap between both definitions: `21`
- RV-only rich fills: `2`
- PM-mid-only rich fills: `10`

The two RV-only examples were SOL high-probability shorts where RV fair was around `98.6c-98.8c`, PM midpoint was around `99.85c`, and both resolved to `1`. That is exactly the danger: RV fair can declare a quote "rich" even when the market-implied price is higher and the realized outcome agrees with the market.

PM-mid / PM-IV is not an external fair either. It is the market's own price in different units. The right comparison set is:

- `physical_prob_fair`: causal forecast probability from RV / empirical / jump model.
- `external_iv_fair`: settlement-aligned option-implied probability from Deribit/CME/etc., when available and correctly mapped.
- `pm_mid_implied_vol`: Polymarket price inverted into IV for diagnostics only.

## Naming And Code Hygiene Recommendations

- `token_model_fair` -> `physical_prob_fair` or `rv_model_prob`
- `claim_model_fair` -> `claim_physical_prob_fair`
- `claim_edge` -> `edge_vs_physical_prob_fair`
- `pm_iv_annualized` -> `pm_mid_implied_vol`
- `vol_premium_ewma` -> `pm_iv_minus_forecast_sigma`
- `arm_a_ewma_nd2` -> `arm_a_ewma_nz` unless the code truly uses a Black-Scholes `d2`
- `fair_value_scaled` -> `rv_edge_scaled` until an external/calibrated fair exists

## Decision

- Do **not** reopen standalone OD on this evidence.
- Keep Strategy A only as source-clean passive MM / carry measurement with OD valuation as a cautious sizing or selection overlay.
- Before live use, update dashboards/scripts/notes so physical probability fair, external IV fair, and PM implied vol cannot be confused.
- The next useful audit is a replacement-fair sensitivity: rerun the same selected fills and sizes using empirical conditional probability, HAR/Kou, or external-IV fair instead of EWMA RV fair.
