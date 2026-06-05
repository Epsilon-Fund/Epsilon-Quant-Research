# NegRisk markets — investigation findings
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


**Investigation date:** 2026-05-10. Read-only — no orders submitted, no on-chain state changed.
**Probed market:** `elc-sot-mid-2026-05-12` (NegRisk, 3 binary sub-markets, ~$2.2M 24h volume).
**Most-active trader observed:** `0x84ad9c5c547a82ec9a08547b94bd922446e5bfb7` (19 fills in 90 s on the target event).
**Tooling produced:** `polymarket/execution/tests/probes/negrisk_trades_probe.py`.

## TL;DR

Existing bot is **95% NegRisk-ready**: the watcher → signal → risk pipeline handles NegRisk markets transparently because position keying is `(condition_id, asset_id)` and each NegRisk sub-market is a normal binary condition. **One critical gap blocks real-money submission on NegRisk markets**: the order signer must encode `verifyingContract = NegRiskCtfExchange` (`0xC5d563…f80a`) instead of `CtfExchange` (`0x4bFb41…8982E`), which py-clob-client supports via `PartialCreateOrderOptions(neg_risk=True)` — but our kernel signer's `_build_order_input` doesn't currently plumb that flag through. Submitting a NegRisk order without the flag yields `invalid signature` from the CLOB. Estimated scope to fix: **small (~80 LOC + tests)**.

---

## 1. NegRisk market structure

### Gamma API — events and markets

A NegRisk *event* bundles N binary *markets* (one per mutually-exclusive outcome) into a single trading universe with a protocol-enforced "at most one outcome resolves YES" invariant.

Gamma response fields on the event row:
- `negRisk: true` — boolean flag, camelCase. **This is the discriminator.**
- `enableNegRisk: true` — paired with `negRisk`; always true if `negRisk` is true.
- `negRiskAugmented: true` (optional) — event can add new outcomes post-launch.

Per-market row (the binary sub-markets within a NegRisk event):
- Each market has its own `conditionId`.
- Each market has exactly **2 entries** in `clobTokenIds` (YES + NO of that outcome's binary).
- A market row also carries `negRisk: true`.

**Implication:** "more than 2 tokens" is *not* the discriminator. NegRisk markets *look* binary at the market level. The `negRisk` flag is the only reliable discriminator at query time.

### Concrete sample (probed target)

```
event_slug:    elc-sot-mid-2026-05-12
event_id:      (3 markets)
negRisk:       True
negRiskAugmented: False
3 sub-markets, each binary, condition_ids:
  0x...  "Will Southampton FC win on 2026-05-12?"
  0x...  "Will Sheffield United win on 2026-05-12?"
  0x...  "Will the match draw?"
```

The probed event was actually `elc-sot-mid` — a Sheffield United vs Southampton game with three mutually-exclusive outcomes (home win / away win / draw). Standard NegRisk pattern.

---

## 2. RTDS behaviour for NegRisk trades

### Wire format — identical to binary

Verbatim sample from the live probe (truncated for brevity):

```json
{
  "connection_id": "dRDE0c0ALPECE9A=",
  "topic": "activity",
  "type": "trades",
  "timestamp": 1778613995772,
  "payload": {
    "asset": "84682847701819061397340022291614341301784642590416131103103478837706660199008",
    "conditionId": "0x0e7f698cc672e0b913b4ea4304ad374477d865f82b005c86123a7040c4866794",
    "eventSlug": "elc-sot-mid-2026-05-12",
    "fee": 0.03225,
    "outcome": "No",
    "outcomeIndex": 1,
    "price": 0.84,
    "proxyWallet": "0xd7F9d0eF6CABe0F2419827effCa57463F5aa29A9",
    "side": "BUY",
    "size": 8,
    "slug": "elc-sot-mid-2026-05-12-sot",
    "timestamp": 1778613995,
    "title": "Will Southampton FC win on 2026-05-12?",
    "transactionHash": "0xe71a3b9752bcdf372aa8132317a7a40482054e578ace8f34ba2529b6239d955b",
    "name": "levytskyy5",
    "pseudonym": "Melodic-Chocolate",
    "bio": "",
    "icon": "...",
    "profileImage": ""
  }
}
```

### Field-by-field comparison vs binary trades

All required fields the watcher uses today (`transactionHash`, `conditionId`, `asset`, `side`, `size`, `price`, `proxyWallet`, `timestamp`) are present and have the same types. **The bot's watcher needs zero changes to detect NegRisk trades.**

One new field worth noting: `fee` (3.225% in the sample). I don't recall this in earlier binary-trade samples. Could be a recent protocol addition or NegRisk-specific. Either way it's optional — the watcher doesn't depend on it.

### NegRisk-specific events on RTDS

**There are none.** RTDS's documented topics are `activity/trades`, `crypto_prices`, `equity_prices`, `comments`. Split, merge, redeem, convert, and resolution events are **all silent on RTDS**. Empirically confirmed by 90 s of subscription with no filter — only `activity/trades` messages arrived.

To detect those operations, the bot must poll the Data API `/activity` endpoint or watch on-chain transactions against `NegRiskAdapter` (`0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`).

---

## 3. Data API behaviour for NegRisk

### `/positions?user=<addr>` — clear discriminator

The flag is **`negativeRisk: true | false`** (full word, **different casing from Gamma's `negRisk`**). This is the single footgun across the two endpoints.

**Side-by-side sample (both rows from the same trader's recent positions):**

NegRisk position:
```json
{
  "proxyWallet": "0x84ad9c5c…bfb7",
  "asset": "11658844…349716",
  "conditionId": "0xb09c9ce6…9c10",
  "size": 34270.179,
  "avgPrice": 0.0724,
  "initialValue": 2481.33,
  "currentValue": 0,
  "redeemable": true,
  "mergeable": false,
  "outcome": "Yes",
  "outcomeIndex": 0,
  "oppositeOutcome": "No",
  "oppositeAsset": "15912828…789639",
  "eventSlug": "fl1-sbr-rcl-2026-04-24",
  "negativeRisk": true       ← discriminator
}
```

Binary position:
```json
{
  "proxyWallet": "0x84ad9c5c…bfb7",
  "asset": "72120578…262608",
  "conditionId": "0x64c212ec…0e72",
  "size": 20864.93,
  "avgPrice": 0.188,
  "initialValue": 3923.96,
  "currentValue": 4277.31,
  "redeemable": false,
  "mergeable": true,
  "outcome": "BetBoom Team",
  "outcomeIndex": 0,
  "oppositeOutcome": "Vitality",
  "eventSlug": "cs2-bb3-vit-2026-05-12",
  "negativeRisk": false      ← discriminator
}
```

**Field schema is identical** — only `negativeRisk` differs. Position keying as `(proxyWallet, asset)` or `(proxyWallet, conditionId, outcomeIndex)` works for both. Bot's existing position-rebuild logic is NegRisk-compatible.

Lifecycle flags:
- `redeemable: true` — market is resolved and this position can be claimed via `redeemPositions`.
- `mergeable: true` — trader holds both YES and NO of the same condition; can call `mergePositions` to recover USDC without crossing the spread.

### `/trades?user=<addr>` — no discriminator

Field set: `proxyWallet, side, asset, conditionId, size, price, timestamp, title, slug, icon, eventSlug, outcome, outcomeIndex, name, pseudonym, bio, profileImage, profileImageOptimized, transactionHash`.

**No `negativeRisk` / `negRisk` field.** To classify a trade as NegRisk, look up `conditionId` (or `eventSlug`) against Gamma. Cache the result — Gamma calls are slow.

### `/activity?user=<addr>&type=<…>` — where split/merge/redeem/convert live

Per the supporting research, `/activity` accepts a `type` query parameter with values: **`TRADE`, `SPLIT`, `MERGE`, `REDEEM`, `CONVERSION`, `REWARD`** (comma-separated). This is the **only off-chain way** to observe a leader's non-trade NegRisk operations.

The probed trader's recent 20 activity rows were all `TRADE` — they don't currently use split/merge/redeem/convert. A more representative leader for those operations would be one running a NegRisk arb strategy.

---

## 4. Split / merge / redeem / convert

### What they are (plain English)

| Operation | Mechanics | When used |
|---|---|---|
| **Split** | Burn 1 USDC, mint 1 YES + 1 NO of a single binary condition. | Market makers minting inventory at par. |
| **Merge** | Burn 1 YES + 1 NO of the same condition, receive 1 USDC. | Closing both sides without paying spread. |
| **Redeem** | After resolution, claim USDC for winning-side tokens. | Post-settlement clean-up. |
| **Convert** *(NegRisk-only)* | Burn 1 NO on each of m outcomes within an event with N outcomes → receive (m − 1) USDC + 1 YES on each of the remaining (N − m) outcomes. | NegRisk-specific arbitrage when the NO basket is mispriced. |

The "guaranteed dollar" property of NegRisk comes from **holding one NO per outcome** — that's economically equivalent to "none of these will win" which, when the universe is complete, is the empty event. The adapter pays (m − 1) USDC to close it out.

### How they appear in the wire

- **RTDS**: not at all. Silent.
- **CLOB**: not at all. These don't route through `/order`.
- **Data API `/activity`**: yes, with explicit `type` values (`SPLIT`, `MERGE`, `REDEEM`, `CONVERSION`).
- **On-chain**: as direct calls to the `NegRiskAdapter` contract (`0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`).

### py-clob-client coverage

**None.** Confirmed via the supporting research — `py_clob_client/client.py` exposes `create_order`, `post_order`, `cancel_order`, `get_*` query methods, but no `split_position`, `merge_positions`, `redeem_positions`, or `convert_positions`. A bot that wants to perform these operations must use web3.py + the `NegRiskAdapter` ABI directly.

### Should the copy-trading bot mirror them?

**Pragmatic recommendation: NO** for the PoC. They are inventory-management operations, not directional signals. A leader doing a Convert is rebalancing their book, not "calling a direction" the bot should mirror. The bot should:

1. **Ignore** leader SPLIT/MERGE/CONVERT activity (don't even watch for it).
2. **Self-execute REDEEM** opportunistically: when the bot's own `/positions` shows `redeemable: true` on any row, call `redeemPositions` to recover USDC. Skipping this is harmless — winnings sit as redeemable tokens until claimed — but it eats into the bot's deployable capital.

---

## 5. Resolution detection

RTDS does not emit resolution events for either NegRisk or binary markets. Realistic detection paths:

| Option | Mechanism | Auth | Freshness | Notes |
|---|---|---|---|---|
| **A. Gamma poll** | `GET /markets?condition_ids=…` and check `closed`/`active` per condition | None | ~minutes (Gamma is cached) | Cheapest. One call per known condition_id per cycle. Recommended primary. |
| **B. Data API `/positions` poll** | Look for `redeemable: true` rows for the bot's own funder | None | ~minutes | Works only for the bot's own positions, not the leader's open ones. Use this to drive redemption. |
| C. CLOB market WS | Subscribe to `wss://ws-subscriptions-clob.polymarket.com/ws/market` with all relevant `asset_ids`; emits `market_resolved` events | None | Real-time | Per the supporting research; payload shape not empirically verified. Adds a second WS connection + per-asset subscription churn. |
| D. On-chain | Watch the UMA-CTF-Adapter / NegRisk resolution oracle on Polygon RPC | None (public RPC) | ~seconds | Most complex; needs web3.py + event ABI decoding. Overkill for PoC. |

**Recommendation for the bot**: option A (Gamma poll for closed status on known condition_ids) + option B (Data API poll for own redeemable positions). Together they cover both "did the leader's market close" and "do I need to redeem my own position".

---

## 6. Implications for the bot's existing modules

| Module | Change needed? | Reason |
|---|---|---|
| **watcher/** | **None.** | NegRisk trades arrive on `activity/trades` with the same payload shape as binary. The probe confirmed every field the watcher reads (`transactionHash`, `conditionId`, `asset`, `side`, `size`, `price`, `proxyWallet`, `timestamp`) is present and correctly typed. |
| **signal/classifier** | **None.** | Position keying as `(condition_id, asset_id)` works identically. Each NegRisk binary sub-market is just one more condition. |
| **signal/dedup** | **None.** | `transactionHash` is unique per fill; same dedup logic applies. |
| **risk/** breakers | **None.** | USD caps apply the same. The price-deviation breaker also works the same — a NegRisk price is still in [0, 1]. |
| **mirror/mirror_engine** | **None at the engine level**, but the venue layer needs the flag (see below). |
| **mirror/clob_http_client** + **kernel signer** | **YES — small change required.** The substitute HTTP client and the kernel's `PyClobClientOrderSigner` together produce the EIP-712 order signature. For NegRisk markets the `verifyingContract` must be `NegRiskCtfExchange` (`0xC5d563…f80a`), not `CtfExchange`. py-clob-client supports this via `PartialCreateOrderOptions(neg_risk=True)`, but the kernel signer's `_build_order_input` ignores the flag — see §7. **Without this, NegRisk orders will be rejected with `invalid signature`.** |
| **journal/** | **No new event types required.** Optionally add `MarketResolved` and `PositionRedeemed` if you implement option-A/B from §5. |
| **cli/** | **None** required for the order path. Optionally: add the Gamma `negRisk` lookup as part of credential/preflight validation, with caching. |

---

## 7. Concrete design proposal for NegRisk handling

### The single load-bearing change

When mirror_engine constructs a `VenueOrderIntent` for the kernel adapter, the chain needs to know whether the target market is NegRisk. The fix is one of three shapes, in increasing order of invasiveness:

1. **Lightweight: per-asset NegRisk cache in `RealVenueAdapter`** (recommended).
   - Add a `_negrisk_cache: dict[str, bool]` keyed by `asset_id` (or `condition_id`).
   - First time we see an asset, call `gamma-api.polymarket.com/markets?condition_ids=<cid>` and read the `negRisk` field. Cache forever (NegRisk-ness doesn't change post-launch).
   - Extend the kernel signer call path so the wrapper can pass `neg_risk=True` through to py-clob-client's `PartialCreateOrderOptions`.
   - The kernel signer's `_build_order_input` currently builds a plain dict — modify the substitute HTTP client to set a `neg_risk` key on the unsigned dict, and patch our copy of the signer (NOT the vendored `_kernel/polymarket_sdk_signer.py`) to wrap the result in `PartialCreateOrderOptions`.
   - **Scope: ~80 LOC** across `mirror/real_venue_adapter.py` (cache + lookup) + a new `mirror/clob_signer.py` that subclasses or replaces the kernel signer.

2. **Medium: explicit per-market config knob.**
   - Operator pre-classifies markets they care about, sets `POLYMARKET_NEGRISK_MARKETS=cid1,cid2,…` in `.env`.
   - Simpler code-wise, but means a manual config update every time a new market is targeted. Bad for unattended runs.

3. **Heavy: re-vendor py-clob-client integration.**
   - Replace the kernel's signer entirely with a direct py-clob-client wrapper. Future-proofs against the kernel's vendored signer drifting further from upstream.
   - **Scope: ~300 LOC.** Over-engineered for the PoC.

**Recommended: option 1.** Plumb the flag through one well-defined boundary; cache Gamma lookups. Tests can mock the cache to exercise both paths.

### Optional: redemption helper

When the bot starts up, after `_rebuild_state_from_journal`, additionally fetch `data-api/positions?user=<config.funder>&redeemable=true`. For each row, log `PositionRedeemable` (new event) and optionally call the NegRiskAdapter contract directly to redeem.

- **Without this**, winnings sit as ERC1155 tokens on the proxy wallet. Polymarket's UI can redeem them with one click. So this is a quality-of-life feature, not blocking.
- **With this**, the bot needs web3.py + the NegRiskAdapter ABI. Scope: another ~150 LOC.

For the first real-money smoke: **don't implement redemption automation**. After the smoke, log into the Polymarket UI and redeem manually.

### Optional: leader activity polling

Poll `data-api/activity?user=<leader>&type=SPLIT,MERGE,REDEEM,CONVERSION` periodically. Document the observed events in the journal as `LeaderNegRiskActivity`. Don't act on them — purely observational.

- **Scope: ~50 LOC** (new poller in `watcher/`).
- **Value**: lets you analyse leader behaviour patterns retrospectively, but doesn't affect the bot's trading.
- For the PoC: skip.

### Net implementation scope

| Component | Scope |
|---|---|
| NegRisk flag plumbing through to py-clob-client | **~80 LOC + tests** (required for any NegRisk trading) |
| Self-redemption automation | ~150 LOC (optional, post-smoke) |
| Leader activity polling | ~50 LOC (optional, observational) |
| **Total minimum to safely trade NegRisk** | **Small (~100 LOC including tests)** |

### What stays unchanged

The full data-plane (watcher, signal, risk, journal-replay, position math) needs zero modification. The change is localised to the order-submission path. Tests added: 3–4 unit tests covering the Gamma lookup + cache + flag-flipping; no integration test against a real NegRisk venue (same constraint as binary — no live-venue tests, ever).

---

## Appendices

### A. Sources used

- `gamma-api.polymarket.com/events` + `/markets` — live empirical probes.
- `data-api.polymarket.com/positions` + `/trades` + `/activity` — live empirical probes.
- `wss://ws-live-data.polymarket.com` — 90 s probe against `elc-sot-mid-2026-05-12`.
- Polymarket docs: `docs.polymarket.com/developers/neg-risk/overview` and `/advanced/neg-risk`.
- NegRiskAdapter contracts and ABI: `github.com/Polymarket/neg-risk-ctf-adapter` (`docs/NegRiskAdapter.md`, `addresses.json`).
- ChainSecurity audit (Apr 2024): `https://old.chainsecurity.com/wp-content/uploads/2024/04/ChainSecurity_Polymarket_NegRiskAdapter_audit.pdf`.
- py-clob-client source — `create_order` NegRisk routing logic; signer field plumbing.
- Allowance-setup gist: `gist.github.com/poly-rodr/44313920481de58d5a3f6d1f8226bd5e`.

### B. Contract addresses (Polygon, chainId 137)

| Contract | Address | Purpose |
|---|---|---|
| CTF (Gnosis ConditionalTokens) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | Token wrapper for both binary and NegRisk |
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` | Binary order venue |
| **NegRisk CTF Exchange** | `0xC5d563A36AE78145C45a50134d48A1215220f80a` | NegRisk order venue — `verifyingContract` for signatures |
| **NegRiskAdapter** | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` | Split/merge/redeem/convert |
| NegRiskUmaCtfAdapter | `0x2F5e3684cb1F318ec51b00Edba38d79Ac2c0aA9d` | Resolution oracle |
| USDC | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` | Settlement asset |

### C. Field-name footguns

| Surface | Field name | Type |
|---|---|---|
| Gamma `events`/`markets` | `negRisk` | bool |
| Gamma `events` | `enableNegRisk`, `negRiskAugmented` | bool |
| Data API `/positions` | `negativeRisk` *(full word!)* | bool |
| Data API `/trades` | *(not present — must look up)* | — |
| RTDS payload | *(not present — must look up)* | — |

If the bot ever needs to check both, the safest pattern is a single helper:

```python
def is_negrisk(field_or_row) -> bool:
    if isinstance(field_or_row, dict):
        return bool(field_or_row.get("negRisk") or field_or_row.get("negativeRisk"))
    ...
```

### D. Probe script

`polymarket/execution/tests/probes/negrisk_trades_probe.py` — adapted from the existing WS probe. Filters RTDS `activity/trades` client-side on `payload.eventSlug == TARGET_EVENT_SLUG`. Run with default `websockets>=12`. Output: per-trade log lines + summary with top traders by fill count + full verbatim of first matched message. Self-terminates after 90 s.

The target event slug is hardcoded at the top of the script (line ~28). To re-probe a different NegRisk market, update that constant and re-run. Before re-targeting, confirm the event is currently trading via Gamma:

```bash
curl -s "https://gamma-api.polymarket.com/events?slug=<your-slug>" | python3 -c 'import sys,json; e=json.load(sys.stdin)[0]; print(f"negRisk={e.get(\"negRisk\")} vol24={e.get(\"volume24hr\")} active={e.get(\"active\")}")'
```

A NegRisk event with $1M+ in 24h volume isn't a guarantee of in-window activity (we observed this with `who-will-be-confirmed-as-fed-chair`: $4M 24h but zero fills in 90 s). Pick events where the 24h volume *and* the half-life of trading activity suggest current liquidity (sports markets near event time are best for short probe windows).
