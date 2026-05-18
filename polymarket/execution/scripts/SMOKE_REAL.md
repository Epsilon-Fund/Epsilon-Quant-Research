# Real-money smoke procedure — first run after wiring lands

Operator runbook for the first end-to-end real-venue test of the
Polymarket copy-trading bot. Read top to bottom before doing
anything; every step matters.

## Prerequisites

1. **Funded account.** A Polymarket proxy wallet with at least $5 USDC.e on
   Polygon mainnet. Plus a small POL balance for gas (~$1 worth is fine).

2. **Real credentials.** Fill in `.env` from `.env.example`:
   - `POLYMARKET_PRIVATE_KEY` — the EOA private key that owns the proxy
     wallet. Hex-encoded, 0x-prefixed, 64 chars.
   - `POLYMARKET_FUNDER` — the proxy wallet address (where the USDC.e
     lives). Lowercase, 0x-prefixed, 40 chars.
   - `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE` —
     created via Polymarket UI or via `midas/scripts/derive_api_keys.py`.

   **The bot refuses to start in real mode if any credential looks like
   a placeholder** (empty, `dummy`, all-zeros, `placeholder`, etc.).

3. **Geo-unblocked region.** Polymarket order submission is geo-blocked
   in several regions (UK confirmed). Use a VPS in a non-blocked region
   (US East / Frankfurt / Tokyo work) or a VPN routed there. Confirm via:
   ```bash
   curl -s ifconfig.me                                              # your IP
   curl -s "https://data-api.polymarket.com/trades?user=…&limit=1"  # 200, not 403
   ```

4. **Pick an active leader.** The leader has to actually be trading during
   your run — pick from the recent winners on
   `polymarket.com/leaderboard?period=1h` or the bot's earlier soak
   findings (gravia, domah, elPolloLoco). If they go idle for the whole
   window the bot won't have anything to mirror — that's a leader
   problem, not a bot problem.

## First run — paranoid mode

Set in your shell before launching:

```bash
export POLYMARKET_VENUE=real
export POLYMARKET_MAX_REAL_ORDERS=1               # halt after one submit
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true   # blocks on stdin per order
export POLYMARKET_LEADER_ADDRESS=<your active leader>
export POLYMARKET_SIZING_USD=5                    # tiny; minimum is 5 shares ≈ $0.05–$5
export POLYMARKET_PRICING_MODE=leader_fill        # avoid current_book latency
```

Launch:

```bash
PYTHONPATH=. python3 -m polymarket.execution.cli
```

Expected sequence:

1. **Banner.** Config redacted. Venue: real. Watcher started against the
   leader address.
2. **Leader fills come in via RTDS.** Watcher journals
   `LEADER_FILL_OBSERVED`. Classifier emits `MIRROR_SIGNAL_EMITTED`.
3. **Risk + harness checks pass.** Mirror_engine builds a candidate.
   Operator prompt appears on stdout:

   ```
   [operator confirm] Submit order:
     market   : 0x…
     asset    : …
     side     : BUY
     shares   : 12.195122
     price    : $0.4100
     type     : FOK
     size_usd : $5.00
   Type 'yes' to proceed:
   ```

   Eyeball *every line* — especially that `market` and `asset` IDs
   correspond to whatever the leader actually traded, and `size_usd`
   matches your `POLYMARKET_SIZING_USD`.

4. **Type `yes` + ENTER.** Bot calls `submit_order`. Wire format
   (decoded by the substitute HTTP client):
   - `size = "12.195122"` (decimal shares)
   - `price = "0.41"` (dollar price)
5. **`ORDER_ACKNOWLEDGED` lands in the journal** with the
   `venue_order_id`. The polling thread (~700 ms cadence) picks up
   the fill and writes `FILL_RECORDED`.
6. **Bot halts** (`max_real_orders=1` reached). Final stats line on stdout.
   Exit code 1 (halted).

## Verify

Open https://polymarket.com/profile/<your_funder_address> in a browser:

- The `transactionHash` from your `FILL_RECORDED` (synthetic
  `<coid>:fill:<ts_ns>` for now — until we wire the on-chain hash)
  won't match the UI's tx hash directly. Cross-check on **size, price,
  market, timestamp** instead.
- Your USDC.e balance is reduced by approximately `size_shares × price`.
- The position appears in your "Active Positions" list.

If all that lines up, the wiring is good.

## Second run — moderate paranoia

Set:

```bash
export POLYMARKET_MAX_REAL_ORDERS=5
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false
```

Re-launch. Bot will now submit up to 5 orders without prompting before
halting. Watch the journal in real-time:

```bash
tail -f journal_logs/execution-$(date -u +%Y-%m-%d).jsonl | jq .
```

Once you've seen 5 successful round-trips on the second run and
nothing looked weird, the bot is ready for unattended operation —
remove `MAX_REAL_ORDERS` (or set it to a daily-budget value like 50).

## Failure modes — what to look for

Check `ORDER_REJECTED` and `RISK_HALT` events in the journal first:

| journal entry | likely cause | fix |
|---|---|---|
| `OrderRejected reason="no_orderbook_price"` | `current_book` mode + thin asks | switch to `leader_fill` mode |
| `OrderRejected reason="venue_rejected"` detail mentioning auth | API key / signature mismatch | re-derive API key, verify funder ↔ private_key linkage |
| `OrderRejected reason="venue_rejected"` detail mentioning size/price | min_order_size violated, or tick rounding | bump `POLYMARKET_SIZING_USD`, check tick_size for the market |
| `RiskHalt reason="kill_switch"` | `/tmp/polymarket_killswitch` exists | `rm /tmp/polymarket_killswitch` |
| `RiskHalt reason="size_cap"` | `sizing_usd > per_trade_cap_usd` | raise per_trade_cap or shrink sizing |
| `RiskHalt reason="max_real_orders"` | hit the harness limit | expected; raise `POLYMARKET_MAX_REAL_ORDERS` |
| `AmbiguousSubmit` | venue timeout — order may or may not have landed | **manually check Polymarket UI** before unhalting; orders submitted while ambiguous should be reconciled before further trading |

If the bot dies with an uncaught exception (rare; the harness should
catch all common cases), the stack trace goes to stderr. Capture it.

## Notes

- The synthetic `transaction_hash` on `FILL_RECORDED` events is a
  placeholder (`<client_order_id>:fill:<ts_ns>`). The on-chain tx hash
  is not currently exposed by the kernel's `VenueFillEvent`. This means
  cross-referencing journal fills to PolygonScan needs a manual lookup
  via the venue order id. Tracked as a v2 enhancement.
- Polling thread cadence is 700 ms. A leader fill → bot fill round-trip
  is bounded by RTDS-to-bot latency (~200 ms) + bot's own submit
  latency (kernel timeout 500 ms) + this poll interval. Real-world
  end-to-end is typically 1–2 seconds.
- The polling thread is a daemon thread — it dies with the main
  process. `wrapper.stop()` joins it cleanly during shutdown; if it
  doesn't terminate within 5 s, the cli logs a warning and continues.
