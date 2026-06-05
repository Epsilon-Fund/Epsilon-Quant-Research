# Politics NegRisk maker smoke procedure - live ladder

> Hub: [[strat_market_making]] / [[COWORK]]
> Design: [[mm_politics_negrisk_live_loop_design]]
> Companion runbook: [[polymarket/execution/scripts/SMOKE_REAL|SMOKE_REAL]]

Operator runbook for the first live smoke of the politics NegRisk passive maker loop, listed beside [[polymarket/execution/scripts/SMOKE_REAL|SMOKE_REAL]]. Read top to bottom before doing anything. This is a one-market measurement loop, not a production quoting system.

## Prerequisites

1. **Funded proxy wallet.** Use a bounded Polymarket proxy wallet with enough USDC.e for two 1-contract resting quotes plus cushion, and a small POL balance for any manual on-chain cleanup. Do not use a broad research wallet unless that wallet is explicitly assigned to the smoke.

2. **Real `.env`.** Fill `polymarket/execution/.env` from `.env.example`:
   - `POLYMARKET_PRIVATE_KEY` - EOA private key that owns the proxy wallet.
   - `POLYMARKET_FUNDER` - proxy wallet address, lowercase `0x` address.
   - `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE` - Polymarket CLOB credentials.
   - `POLYMARKET_JOURNAL_DIR` - stable journal/state directory for maker telemetry and NegRisk inventory replay.

   The real-venue builder refuses placeholder-looking credentials. If auth fails, stop and fix credentials before any observe or quote run.

3. **Non-geo-blocked region.** The UK dev box gets 403 from the Data API. Run from a non-blocked VPS/VPN and verify reads before any live rung:

   ```bash
   curl -s ifconfig.me
   curl -i "https://data-api.polymarket.com/positions?user=<your_funder>&redeemable=true"
   curl -i "https://gamma-api.polymarket.com/markets?condition_ids=<selected_condition_id>"
   ```

   You want HTTP 200. A 403 is a region/network problem; move the run, do not quote from that box.

4. **One tick=0.01 politics NegRisk market.** Select the market with the five-screen rule from [[mm_politics_negrisk_live_loop_design]]:
   - Gamma says `negRisk: true`.
   - Bucket is Non-US elections, Trump personnel/policy, Other politics, or 2026 US races/midterms. Exclude 2028 presidential outrights for Phase 2.
   - Corrected-carry cache shows at least 5% non-top3 headroom.
   - Prefer uninformed flow: top makers have below-median directional score.
   - Resolution is clear, scheduled, and objective.

   Confirm Gamma metadata before launch:

   ```bash
   export POLYMARKET_MAKER_CONDITION_ID=<selected_condition_id>
   curl -s "https://gamma-api.polymarket.com/markets?condition_ids=${POLYMARKET_MAKER_CONDITION_ID}" \
     | jq '.[0] | {question, conditionId, negRisk, orderPriceMinTickSize, tick_size, tickSize, active, closed, bestBid, bestAsk}'
   ```

   **Abort if the market is not tick=0.01.** Sub-penny markets (`tick=0.001`) are out of scope for the first smoke. A sub-penny market mispriced on a 0.01 fallback can be badly wrong; the maker has an allow-list guard, but operator selection must still be tick=0.01 only.

5. **Event calendar sanity.** Check `polymarket/execution/maker/politics_events.yaml` for scheduled events near the selected market. Phase 2 logs event proximity; it does not yet suppress quotes during event windows.

## Common setup

Run from repo root:

```bash
cd /Users/justiniturregui/Desktop/github/epsilon-quant-research
set -a
source polymarket/execution/.env
set +a
```

Use these baseline maker settings unless a rung overrides them:

```bash
export POLYMARKET_MAKER_CONDITION_ID=<selected_tick_0_01_negrisk_condition_id>
export MAKER_SIZE_CONTRACTS=1
export POLYMARKET_MAKER_REFRESH_SECONDS=30
export POLYMARKET_MAKER_ORDER_TYPE=GTC
export POLYMARKET_MAKER_BASKET_EXPOSURE_CAP=10
export POLYMARKET_MAKER_ALLOWED_TICKS=0.01
export POLYMARKET_RPC_URL=""  # blank for first smoke unless intentionally testing self-redemption
```

Tail the maker journal in another terminal:

```bash
tail -f "${POLYMARKET_JOURNAL_DIR:-journal_logs}/maker-$(date -u +%Y-%m-%d).jsonl" | jq .
```

Run command template:

```bash
PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
  python -m polymarket.execution --mode maker
```

If self-redemption is intentionally being tested in the same ephemeral uv runtime, add `--with web3` and set `POLYMARKET_RPC_URL` to a Polygon mainnet RPC URL. For the first smoke, blank RPC and manual redemption fallback are cleaner.

## Rung 0 - auth-only read

Purpose: prove real CLOB credentials can perform an open-orders read. This rung submits nothing, does not start the maker engine, and does not require `POLYMARKET_MAKER_CONDITION_ID`.

```bash
export POLYMARKET_VENUE=real
export POLYMARKET_MAX_REAL_ORDERS=0
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true

PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
  python -m polymarket.execution --mode maker --check-auth
```

Expected:

```text
[maker:auth] open_orders=<count>
```

Expected result:
- Exit code `0`.
- One open-orders read.
- Zero submits.
- No `MAKER_SESSION_STARTED`; this is not a maker session.

Failure case:
- Exit code `5` means the auth read failed. Stop here. Re-derive or re-check CLOB credentials, verify private key/funder linkage, verify the region is not blocked, then rerun Rung 0.

## Rung 1 - real observe-only pricing

Purpose: wire the real venue, read Gamma/CLOB/Data API, price the would-be quotes, and prove the real safety gate blocks submits. This rung **does not exercise NegRisk signing** because `POLYMARKET_MAX_REAL_ORDERS=0` halts before `venue.submit_order`.

```bash
export POLYMARKET_VENUE=real
export POLYMARKET_MAX_REAL_ORDERS=0
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false
export POLYMARKET_MAKER_MAX_RUNTIME_SECONDS=120

PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
  python -m polymarket.execution --mode maker
```

Expected journal:
- `MAKER_SESSION_STARTED`
- `RISK_HALT reason="max_real_orders"` on the first would-be real submit.
- `MAKER_QUOTE_SKIPPED reason="max_real_orders"` for each side that would have quoted.
- Other `MAKER_QUOTE_SKIPPED` rows only if market lookup, book, tick, resolution, or basket-cap checks block pricing.
- `MAKER_SESSION_STOPPED reason="max_runtime_reached"`
- Zero `ORDER_SUBMITTED`, zero `ORDER_ACKNOWLEDGED`, zero `MAKER_QUOTE_PLACED` on the real venue.

Pass condition: the loop keeps running until `POLYMARKET_MAKER_MAX_RUNTIME_SECONDS` while the safety gate blocks submits. If it exits early or throws, stop and inspect before Rung 2.

Optional fake rehearsal:

```bash
export POLYMARKET_VENUE=fake
export POLYMARKET_MAX_REAL_ORDERS=0
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false
export POLYMARKET_MAKER_MAX_RUNTIME_SECONDS=60

PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
  python -m polymarket.execution --mode maker
```

Fake venue may journal `ORDER_SUBMITTED` and `MAKER_QUOTE_PLACED` because the in-process fake accepts quotes. Those are not exchange submits.

## Rung 2 - first real orders

Purpose: first real passive quotes and first NegRisk-signing test. Budget is exactly two submit attempts: one bid and one ask, both size 1, tick=0.01 only. Operator confirmation is mandatory.

```bash
export POLYMARKET_VENUE=real
export POLYMARKET_MAX_REAL_ORDERS=2
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true
export MAKER_SIZE_CONTRACTS=1
export POLYMARKET_MAKER_REFRESH_SECONDS=30
export POLYMARKET_MAKER_MAX_RUNTIME_SECONDS=1800
export POLYMARKET_MAKER_ALLOWED_TICKS=0.01
export POLYMARKET_RPC_URL=""

PYTHONPATH=. uv run --no-project --with py-clob-client --with websockets \
  python -m polymarket.execution --mode maker
```

Expected operator prompts:
- One BUY and one SELL quote at the current best bid / best ask.
- Size is `1`.
- Order type is `GTC`.
- Condition id and asset id match the selected market.
- Price is on a 0.01 tick.

Type `yes` only after eyeballing every field. Decline anything surprising.

Expected journal:
- `MAKER_SESSION_STARTED`
- `ORDER_SUBMITTED` for each confirmed real submit.
- `ORDER_ACKNOWLEDGED` if the venue accepts.
- `MAKER_QUOTE_PLACED` for each accepted quote.
- `RISK_HALT reason="max_real_orders"` and `MAKER_QUOTE_SKIPPED reason="max_real_orders"` after the two-submit budget is exhausted.
- `MAKER_FILL_TELEMETRY` if any quote fills.
- `MAKER_MISSED_FILL` if the market trades at our resting price but our order is not filled.
- `MAKER_QUOTE_CANCELED reason="shutdown"` when the run stops.
- `MAKER_SESSION_STOPPED`

Failure signatures to watch immediately:
- `ORDER_REJECTED reason="invalid_signature"` or an `ORDER_REJECTED` detail mentioning `invalid signature` - stop; this is the NegRisk signing path or credential mismatch.
- `ORDER_REJECTED reason="cannot_classify_market"` - stop; Gamma metadata did not classify NegRisk/tick safely.
- `MAKER_QUOTE_SKIPPED reason="tick_size_not_allowed"` - selected market is not allowed tick; choose tick=0.01.
- Geo/network 403 in curl/stdout/stderr or repeated `market_lookup_failed` - move region/network before retrying.
- `RISK_HALT reason="ambiguous_submit"` - manually reconcile UI/open-orders read before relaunching.

After shutdown, rerun Rung 0 and inspect the Polymarket UI. There should be no unintended open quotes.

## Phase-2 campaign and gates

Do not change quoting rules mid-sample. The central question from [[mm_politics_negrisk_live_loop_design]] is whether a non-incumbent can dodge politics adverse selection well enough to preserve the corrected-carry edge.

Minimum sample:
- At least **30 settled markets** before any scale decision.
- Or **90 calendar days**. If fewer than 30 markets settle in 90 days, that is itself a liquidity/fill-share finding.

Pre-registered gates:

| gate | measurement | scale-eligible read | close / redesign read |
|---|---|---|---|
| Fill share | Our fills as a share of market fills | > 0% in at least 5 markets | 0 fills after 30+ resolved markets |
| Adverse selection | 60-second post-fill drift, market-cluster CI | CI crosses zero or lower bound > -500 bps | Lower bound < -2,000 bps |
| News-tagged loss concentration | Adverse fills near scheduled events | < 50% of adverse fills are scheduled-event-proximate | > 70% cluster near events |
| Net-of-cost PnL | Realized PnL / gross notional, market-cluster CI | Lower CI > 0 over at least 30 settled markets | Lower CI < -500 bps |
| Resolution drag | Delayed/disputed/failed recovery share | < 10% of markets | > 20% |

Settlement and redemption drag are part of the strategy economics. Track `MARKET_RESOLVED`, `POSITION_REDEEMABLE`, `POSITION_REDEEMED`, and `REDEMPTION_FAILED`; do not count capital as recovered until the wallet balance/position state confirms it.

## Telemetry interpretation

`MAKER_QUOTE_PLACED` means a quote was accepted by the venue. It is not a fill.

`MAKER_QUOTE_CANCELED` means the loop canceled a tracked quote because of shutdown, price move, market resolution, or guard logic.

`MAKER_FILL_TELEMETRY` means Data API trades showed a fill by our funder wallet. Fields:
- `client_order_id` - maker quote id if the fill matched a live tracked quote; blank if the fill was ours but no quote remained in local state.
- `condition_id` / `asset_id` - market and token that filled.
- `side` - side from the Data API trade row.
- `size` / `price` - fill size and price.
- `top_maker_rank_at_fill` - proxy rank among same-price fills in a 5-second window; not true queue position.
- `post_fill_price_drift_60s` - first trade price at or after fill time + 60 seconds minus fill price. Negative drift is adverse selection.
- `news_proximate` - scheduled-event proximity from `maker/politics_events.yaml`; Phase-2 analysis can recompute broader windows from timestamps.
- `fill_share_this_market` - our fills divided by total market fills over the session.

`MAKER_MISSED_FILL` means the market traded at our quoted asset and price but another wallet filled instead. It is queue/capacity evidence, not a realized loss by itself.

`MARKET_RESOLVED` means Gamma flipped the market closed/inactive. The maker should cancel quotes and leave recovery to the resolution handler.

`POSITION_REDEEMABLE`, `POSITION_REDEEMED`, and `REDEMPTION_FAILED` are settlement/capital-recovery events. Redemption failure is non-blocking for the loop but blocking for declaring capital recovered.

## Failure-mode matrix

Check the journal first, then stdout/stderr, then Polymarket UI/open-orders.

| journal/stdout signal | likely cause | operator action |
|---|---|---|
| `[maker:auth] auth read failed`, exit 5 | CLOB auth failure, bad API credentials, private key/funder mismatch, or blocked network | Stop. Re-derive credentials, verify wallet linkage, verify region, rerun Rung 0. |
| Data API or Gamma curl returns 403 | Geo-blocked region, especially UK dev box | Move to non-blocked VPS/VPN. Do not quote from this region. |
| `MAKER_QUOTE_SKIPPED reason="market_lookup_failed"` | Gamma unreachable, bad condition id, or metadata not ready | Verify condition id and Gamma row. Wait/retry only after Gamma returns the market. |
| `MAKER_QUOTE_SKIPPED reason="no_book"` | Empty CLOB book or wrong/stale token id | Pick another market or wait. Do not force a quote without a book. |
| `MAKER_QUOTE_SKIPPED reason="tick_size_not_allowed"` | Market tick not in `POLYMARKET_MAKER_ALLOWED_TICKS`, usually 0.001 | Expected guard. Pick tick=0.01. Do not widen allowed ticks for first smoke. |
| `MAKER_QUOTE_SKIPPED reason="basket_exposure_cap"` | Existing NegRisk basket exposure is already at/above cap | Confirm inventory JSONL and wallet positions. Reduce exposure or choose another market. |
| `RISK_HALT reason="max_real_orders"` | Submit budget exhausted | Expected in Rung 1 and after two submit attempts in Rung 2. Raise only after review. |
| `RISK_HALT reason="operator_aborted"` | Operator declined the prompt | Expected if any prompt field looked wrong. Fix cause before relaunch. |
| `RISK_HALT reason="ambiguous_submit"` | Venue timeout; order may or may not exist | Stop. Check UI and Rung 0 open-orders read. Cancel unknown quotes manually before relaunch. |
| `ORDER_REJECTED reason="cannot_classify_market"` | RealVenueAdapter could not get safe Gamma NegRisk/tick metadata | Stop. Confirm Gamma reachability and condition id. Do not submit on fallback metadata. |
| `ORDER_REJECTED reason="invalid_signature"` or detail contains `invalid signature` | NegRisk signing path, selected market classification, or CLOB credential mismatch | Stop immediately. Confirm Gamma `negRisk:true`, inspect signing path, do not retry blindly. |
| `ORDER_REJECTED reason="maker_submit_exception"` | Local submit exception or network transport failure | Stop if real venue. Inspect detail/stderr and rerun Rung 0 before relaunch. |
| `ORDER_REJECTED reason="maker_cancel_exception"` | Cancel path threw or venue/network failed during cancel | Check UI/open-orders read. Cancel manually if any quote remains. |
| Repeated `MAKER_MISSED_FILL`, no `MAKER_FILL_TELEMETRY` | We are behind queue or non-top3 headroom is worse live than historical | Keep logging; this is Phase-2 evidence. Do not chase or improve price mid-sample. |
| `MARKET_RESOLVED` | Target resolved while loop was live | Let maker cancel. Verify no open quotes, then watch redemption state. |
| `POSITION_REDEEMABLE` followed by `REDEMPTION_FAILED` | Blank/missing `POLYMARKET_RPC_URL`, missing `web3`, insufficient gas, or unsupported proxy redemption path | Redeem manually in UI if needed. If testing self-redemption, configure RPC/runtime/gas and rerun only after manual state check. |

## Env-var reference

These are the environment variables read by the maker path or by `ExecutionConfig`, which maker loads before startup. Some copytrade fields are still required because the shared execution config validates them.

| variable | rung value / default | purpose |
|---|---|---|
| `POLYMARKET_VENUE` | Rung 0/1/2: `real`; optional rehearsal: `fake` | Selects fake or real venue adapter. |
| `POLYMARKET_MAKER_CONDITION_ID` | Required for Rung 1/2; not required for `--check-auth` | One market to quote. Must be politics NegRisk tick=0.01. |
| `MAKER_SIZE_CONTRACTS` | Rung 2: `1`; default `1` | Quote size in contracts. |
| `POLYMARKET_MAKER_REFRESH_SECONDS` | Rung 1 short runs may use `30`; Rung 2 `30`; default `30` | Maker refresh/cancel-replace cadence. |
| `POLYMARKET_MAKER_MAX_RUNTIME_SECONDS` | Rung 1 `120`; Rung 2 `1800`; unset/`0` runs until signal | Bounded smoke runtime. |
| `POLYMARKET_MAKER_ORDER_TYPE` | `GTC` | Maker quote order type. |
| `POLYMARKET_MAKER_BASKET_EXPOSURE_CAP` | `10` | Hard basket exposure cap before placing more BUY exposure. |
| `POLYMARKET_MAKER_ALLOWED_TICKS` | First smoke: `0.01`; default `0.01` | Tick-size allow-list. Do not widen for first smoke. |
| `POLYMARKET_MAX_REAL_ORDERS` | Rung 0/1: `0`; Rung 2: `2`; default `5` | Real submit cap. `0` means observe-only: real venue wired, no submits. |
| `POLYMARKET_REQUIRE_OPERATOR_CONFIRM` | Rung 0 `true`; Rung 1 `false`; Rung 2 `true`; default `false` | Per-submit stdin confirmation for real venue. |
| `POLYMARKET_RPC_URL` | First smoke: blank | Optional Polygon RPC for self-redemption. Blank means redemption attempts are not configured. |
| `POLYMARKET_PRIVATE_KEY` | Real `.env` | EOA private key for CLOB signing and optional redemption. |
| `POLYMARKET_FUNDER` | Real `.env`, lowercase `0x` | Proxy wallet / user address for CLOB, Data API reads, and telemetry filtering. |
| `POLYMARKET_API_KEY` | Real `.env` | CLOB API credential. |
| `POLYMARKET_API_SECRET` | Real `.env` | CLOB API credential. |
| `POLYMARKET_PASSPHRASE` | Real `.env` | CLOB API credential. |
| `POLYMARKET_LEADER_ADDRESS` | Any valid lowercase `0x` address in maker mode | Required by shared `ExecutionConfig`; not used by maker quoting. |
| `POLYMARKET_CHAIN_ID` | `137` | Polygon chain id. |
| `POLYMARKET_SIGNATURE_TYPE` | `1` | CLOB signing signature type. |
| `POLYMARKET_CLOB_URL` | `https://clob.polymarket.com` | CLOB book/auth/order endpoint base. |
| `POLYMARKET_GAMMA_URL` | `https://gamma-api.polymarket.com` | Gamma market metadata base. |
| `POLYMARKET_DATA_URL` | `https://data-api.polymarket.com` | Data API trades/positions/activity base. |
| `POLYMARKET_WS_URL` | Default from `.env.example` | Read by shared config; maker does not use RTDS. |
| `POLYMARKET_JOURNAL_DIR` | Stable smoke directory | Maker journal and NegRisk inventory state path. |
| `POLYMARKET_LOG_LEVEL` | `INFO` | Shared config field. |
| `POLYMARKET_MAX_CAPITAL_USD` | Keep default for maker smoke | Shared config field; maker uses its own 1-contract sizing and basket cap. |
| `POLYMARKET_PER_TRADE_CAP_USD` | Keep default for maker smoke | Shared config field. |
| `POLYMARKET_PER_MARKET_CAP_USD` | Keep default for maker smoke | Shared config field. |
| `POLYMARKET_SIZING_USD` | Keep default for maker smoke | Required by shared config; not maker quote size. |
| `POLYMARKET_MAX_OPEN_POSITIONS` | Keep default for maker smoke | Shared config field. |
| `POLYMARKET_DEFAULT_ORDER_TYPE` | Keep default `FOK` | Copytrade default; maker uses `POLYMARKET_MAKER_ORDER_TYPE`. |
| `POLYMARKET_PRICING_MODE` | Keep default `leader_fill` | Copytrade pricing mode; not used by maker. |
| `POLYMARKET_PRICE_DEVIATION_PCT` | Keep default | Shared config field. |
| `POLYMARKET_DAILY_LOSS_HALT_USD` | Keep default | Shared config field. |
| `POLYMARKET_KILLSWITCH_PATH` | Default `/tmp/polymarket_killswitch` | Shared config field. |

## Final smoke review

Before raising `POLYMARKET_MAX_REAL_ORDERS` above 2, collect:

- Selected market Gamma row showing `negRisk:true`, tick=0.01, active, and not closed.
- Rung 0 auth-only output with open-order count.
- Rung 1 observe-only journal showing pricing checks plus `max_real_orders` gate and zero submits.
- Rung 2 journal slice from `MAKER_SESSION_STARTED` through `MAKER_SESSION_STOPPED`.
- Polymarket UI or open-orders read showing no unintended open quotes after shutdown.
- Any `MAKER_FILL_TELEMETRY`, `MAKER_MISSED_FILL`, `MARKET_RESOLVED`, or redemption events.

Only after review is clean should Phase 2 continue market-by-market under the frozen 1-contract measurement design.
