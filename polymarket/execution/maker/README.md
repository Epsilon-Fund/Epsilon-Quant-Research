# Politics NegRisk Maker Module

> Hub: [[strat_market_making]] · [[COWORK]]

Phase 1 builds the prerequisites for a one-market politics NegRisk live measurement loop. It is not a production quoting system and it does not implement K-PEG chasing, external pricing, multi-market orchestration, or news scraping.

## Modules

- `event_calendar.py` loads the manual `politics_events.yaml` calendar and exposes `is_event_proximate(...)` for telemetry.
- `negrisk_inventory.py` polls Data API activity for `SPLIT,MERGE,REDEEM,CONVERSION`, applies basket-level deltas, and persists replayable JSONL state.
- `resolution_handler.py` polls Gamma for market closure, polls Data API for redeemable positions, logs resolution/redemption events, and attempts `NegRiskAdapter.redeemPositions(bytes32,uint256[])` via lazy web3.
- `maker_engine.py` places one bid and one ask on the configured condition, refreshes only when the book moves by more than one tick, cancels on resolution, and logs fill/missed-fill telemetry.

## Phase-2 Measurement Loop

Set `POLYMARKET_MAKER_CONDITION_ID` to the target market condition id. The engine looks up the YES token from Gamma, joins the current best bid and best ask with one-contract GTC quotes, and records the data needed to decide whether politics adverse selection is dodgeable live:

- quote placements and cancels
- fills and missed fills
- `top_maker_rank_at_fill`
- `post_fill_price_drift_60s`
- `news_proximate`
- `fill_share_this_market`

The basket inventory cap is hardcoded at 10 contracts by default. If `get_basket_exposure(condition_id) >= 10`, the engine skips the bid side so it does not add to the basket.

## Operational Notes

The existing real-venue safety harness still applies: `POLYMARKET_MAX_REAL_ORDERS` and `POLYMARKET_REQUIRE_OPERATOR_CONFIRM` gate real venue submits.

Self-redemption is best effort. If `POLYMARKET_RPC_URL` and a usable private key are configured, the resolution handler attempts a direct web3 transaction against the NegRiskAdapter. Failures are logged and do not block quoting shutdown or the rest of the loop.
