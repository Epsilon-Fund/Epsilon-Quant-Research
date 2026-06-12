---
title: "MM Politics NegRisk Live Loop — Design Decisions"
tags: [market-making, negrisk, politics, live-loop, measurement, design]
created: 2026-06-03
status: phase-1-ready — decisions locked, Codex prompt drafted
---

# MM Politics NegRisk Live Loop — Design Decisions

> Hub: [[strat_market_making]] · [[COWORK]]
> Anchor: [[mm_politics_negrisk_accounting_findings]] (2,290 bps, CI [1,020, 3,621])
> Handoff that spawned this: [[2026-06-03_politics_negrisk_live_loop]]

## Summary

- Scope: MM Politics NegRisk Live Loop — Design Decisions in the MM/market-making area.
- Existing takeaway/status: This note records the four design decisions made in the kickoff chat. It is the pre-registration anchor for Phase-2 measurement.
- Evidence lives in the detailed sections below; this summary is only a navigation layer over the existing note.
This note records the four design decisions made in the kickoff chat. It is the pre-registration anchor for Phase-2 measurement.

---

## Decision 1 — Proven-bucket cell list and market-selection rule

### Proven settled buckets (Phase 2 priority order)

| bucket | settled bps | CI | 2026 non-top3 flow share |
|---|---|---|---|
| Non-US elections | +3,673 | [1,671, 6,978] | 19% |
| Trump personnel/policy | +1,018 | [174, 1,182] | 24% |
| Other politics | +1,046 | [216, 2,245] | 16% |
| 2028 US presidential outrights | **no settled PnL** | forward-only | 36% |
| 2026 US races/midterms | in scope (scheduled, objective) | — | 5% |

**2028 US presidential outrights are excluded from Phase 2.** 36% of 2026 flow but zero settled-only evidence and multi-year capital lockup. Quote them only after Phase 2 fills validate the dodgeability hypothesis in the three proven buckets.

**2026 US races/midterms stay in scope.** Scheduled resolution dates, objective criteria (official election results), structurally closer to Non-US elections than to the 2028 outrights.

### Per-market five-screen filter

Applied at market selection time (before quoting any new market):

1. **`negRisk: true`** from Gamma API — the only reliable discriminator. Do NOT use `negativeRisk` from Data API `/positions` as a substitute (different field name, different surface).
2. **Event bucket** = Non-US elections, Trump personnel/policy, Other politics, or 2026 US races/midterms. Exclude 2028 outrights.
3. **Non-top3 headroom**: from the corrected-carry wallet-market cache (`mm_politics_negrisk_corrected_carry_recut_wallet_market.parquet`), require ≥5% historical non-top3 fill share in that specific market.
4. **Uninformed-flow preference**: cross-reference `traders_directionality.parquet` — prefer markets where the observed top-3 makers have below-median directional score (retail-directional flow we're providing against, not specialists front-running us).
5. **Resolution clarity**: prefer markets with a scheduled resolution date and objective criterion (official election result, confirmed appointment). Deprioritize UMA-dispute-likely or subjective resolution.

---

## Decision 2 — Phase-1 exec build scope

### What Midas already has

- Full copytrade pipeline (watcher → signal → risk → mirror_engine → RealVenueAdapter → kernel → py-clob-client).
- Safety harness: `MAX_REAL_ORDERS`, `REQUIRE_OPERATOR_CONFIRM`, per-trade/per-market/daily caps.
- Position keying as `(condition_id, asset_id)` — already NegRisk-compatible at the data layer.
- `mirror/clob_signer.py` — the substitute signer (kernel signer is frozen).

### Gaps and build order

| gap | what | LOC est | blocking |
|---|---|---|---|
| 1 | NegRisk signature flag in `clob_signer.py` + `real_venue_adapter.py` | ~80 + tests | YES — no NegRisk order can submit without this |
| 2 | Composite NegRisk inventory tracker (poller on `/activity` for SPLIT/MERGE/REDEEM/CONVERSION) | ~150–200 | needed before sizing beyond 1 contract |
| 3 | Resolution handler (Gamma poll + `/positions?redeemable=true` + web3.py self-redemption) | ~100 | needed before any carry trade |
| 4 | Passive maker engine (two-sided GTC/GTD quotes, cancel-replace, 1-contract Phase-2 scaffold) | ~200–300 | the actual MM quoting loop |
| E | Scheduled-event calendar (static YAML + `is_event_proximate()`) | ~50 | Phase-2 telemetry `news_proximate` field |

**Build order: 1 → 3 → 2 → 4 (+ E alongside 4)**

Gap 3 (resolution handler) before Gap 2 (inventory tracker) because carry-to-resolution requires capital recovery at settlement, and self-redemption is simpler than the full inventory tracker.

### Architecture note

The maker engine is a NEW module (`maker/maker_engine.py`) — not an extension of `mirror_engine.py`. The copytrade bot is a FOK taker that mirrors a leader; the MM loop places passive GTC/GTD limit orders independently. They share the safety harness, journal, risk module, and venue adapter's order-submission path but have different quoting logic.

### What the maker engine does NOT do (Phase 2)

- No dynamic peg-chasing (K-PEG style). Carry-to-resolution is the edge; chasing imports the exit-spread tax.
- No Binance anchor or vol model. This loop does not use OD pricing.
- No leader-mirroring. Independent quoting only.
- No X/Twitter or LLM news feed. Scheduled calendar only.
- No multi-market orchestration. One market at a time via `POLYMARKET_MAKER_CONDITION_ID` env var.

---

## Decision 3 — Pre-registered Phase-2 measurement success criteria

**Central question:** can a non-incumbent dodge politics adverse selection well enough to preserve the 2,290 bps historical edge?

**Reference baseline from K5-STRESS adverse-selection table:**
- `politics_negrisk` 60s markout: **−336 bps** (vs crypto_4h **−1,886 bps**). The ratio is ~5.6×. Phase 2 tests whether we maintain this structural difference in live conditions.

### Pre-registered gates

| metric | measure | "dodgeable → scale" | "picked-off → close" |
|---|---|---|---|
| Fill share | % of non-top3 fills in target markets we capture | > 0% in ≥5 markets | 0 fills after 30+ resolved markets |
| Adverse selection (post-fill 60s drift) | mean bps, market-cluster CI | CI crosses zero OR lower bound > −500 bps | Lower bound < −2,000 bps (approaching crypto levels) |
| News-tagged loss concentration | % of adverse fills within 30 min of a scheduled event | < 50% (losses spread, not news-clustered) | > 70% (losses cluster on events = need real-time feed to dodge) |
| Net-of-cost PnL per resolved market | realized PnL / gross notional, market-cluster CI | Lower CI > 0 over ≥30 settled markets | Lower CI < −500 bps |
| Resolution drag | % of markets with delayed/disputed resolution | < 10% | > 20% = carry cost exceeds historical numbers |

### Sample size and duration

- Minimum: **30 settled markets** before any scale decision.
- Maximum duration: **90 calendar days**. If < 30 settled markets in 90 days, that is itself a finding (fill share / liquidity lower than modeled).
- Capital commitment: ~$30 notional at 1-contract per market.

### Integrity constraint

Quoting parameters are frozen for the duration of Phase 2. If adverse selection looks bad at 10 markets, log it but do not change the quoting logic mid-loop. The only clean reading is from the full pre-registered sample.

---

## Decision 4 — News-feed approach

**Decision: scheduled-event calendar only for Phase 2. Gate the breaking-news feed on Phase-2 telemetry.**

The news-tagged adverse-selection gate (Decision 3) will tell us whether losses cluster on news events. If < 50% of adverse fills are news-proximate, a real-time news feed wouldn't help even if we had it. Build the expensive thing only if the cheap telemetry proves it's the binding loss channel.

**Phase 2 calendar (Deliverable E):** static YAML with scheduled events for the next 30–60 days, refreshed manually every few weeks. Coverage: election dates, scheduled court hearings, Congressional confirmation votes, major policy announcement dates. ~30 minutes of ops work per refresh.

**Conditional Phase 3 gate for breaking-news feed:** only revisit if Phase-2 telemetry shows `news_proximate > 50%` of adverse fills AND the net-of-cost CI is close to "picked-off" verdict. Options then (cheapest first): (a) Polymarket Gamma event API for pre-published resolution dates, (b) lightweight RSS/Google News scrape for politics keywords, (c) X API (now paid/rate-limited — only if (a)/(b) insufficient and adverse selection is demonstrably news-driven).

**Explicit defer:** do NOT build any breaking-news pipeline before Phase-2 telemetry justifies it.

---

## Phase map

| phase | what | status |
|---|---|---|
| Phase 0 | copytrade smoke (non-NegRisk), proves basic plumbing | live default per [[TODO]] § copytrade |
| Phase 1 | exec build: NegRisk flag → resolution handler → composite inventory → maker engine + calendar | **READY TO BUILD** — Codex prompt below |
| Phase 2 | measurement-only deployment, 1-contract, ≥30 settled markets, full telemetry | blocked on Phase 1 |
| Phase 3 | scale within proven buckets IF Phase-2 adverse-selection gate passes | blocked on Phase 2 |

---

## Codex Phase-1 prompt

Paste this into a Codex chat to build Phase 1. Do not commit this prompt to the repo — it lives in chat history.

```markdown
Before doing anything else, read:
1. `brain/CODEX.md`
2. `brain/TODO.md`
3. `brain/COWORK.md`
4. `brain/POLYMARKET_BRAIN.md`
5. `polymarket/research/notes/market_making/strat_market_making.md`
6. `polymarket/research/notes/market_making/mm_politics_negrisk_accounting_findings.md`
7. `polymarket/execution/tests/probes/NEGRISK_FINDINGS.md`
8. `polymarket/execution/PLAN.md`
9. `polymarket/execution/CLAUDE.md`

---

## Task: Politics NegRisk MM — Phase-1 exec build

Build the prerequisites for Phase-2 live measurement quoting on politics NegRisk markets.
All work goes inside `polymarket/execution/`. Follow PLAN.md architecture and CLAUDE.md constraints
(kernel is frozen, no pip install, uv only).

### Background

The copytrade bot submits FOK taker orders. The MM loop needs to submit passive GTC/GTD maker orders
on NegRisk politics markets. Three gaps block it:

1. NegRisk orders get `invalid signature` because `_build_order_input` doesn't set `neg_risk=True`.
2. No composite basket inventory tracking — merge/split/redeem/convert events are silent on RTDS.
3. No resolution detection or self-redemption — carry-to-resolution requires capital recovery at settlement.

---

### Deliverable A — NegRisk signature fix (Gap 1)

Per `NEGRISK_FINDINGS.md` §7 option 1 (recommended):

- Add `_negrisk_cache: dict[str, bool]` to `mirror/real_venue_adapter.py`, keyed by `condition_id`.
- On first order for a condition, call `gamma-api.polymarket.com/markets?condition_ids=<cid>` and read `negRisk`
  field. Cache forever (NegRisk-ness is immutable post-launch).
- Patch `mirror/clob_signer.py` (NOT `_kernel/polymarket_sdk_signer.py` — kernel is frozen) so it passes
  `neg_risk=True` into `PartialCreateOrderOptions` when the cache says so.
- The `VenueOrderIntent` dataclass (or equivalent) should carry a `neg_risk: bool` field.
- Write 3–4 unit tests: cache miss → Gamma lookup → cache hit, flag-true path, flag-false path,
  invalid-signature guard.

Acceptance: a NegRisk order intent with `neg_risk=True` produces a signature using
`verifyingContract = 0xC5d563A36AE78145C45a50134d48A1215220f80a` (NegRiskCtfExchange),
not `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` (CtfExchange).

---

### Deliverable B — Resolution handler (Gap 3)

New module `maker/resolution_handler.py`:

- Background thread polling every 60s.
- **Gamma poll**: for all condition_ids in the bot's current open position set, call
  `gamma-api.polymarket.com/markets?condition_ids=<comma-separated>` and check `closed`/`active` per row.
  Log a `MarketResolved` journal event when a market flips to closed.
- **Redemption poll**: call `data-api.polymarket.com/positions?user=<config.funder>&redeemable=true`
  and collect rows with `redeemable: true`. Log `PositionRedeemable` for each.
- **Self-redemption**: using web3.py + NegRiskAdapter ABI (`0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296`),
  call `redeemPositions` on redeemable rows. Wrap in try/except — failed redemption is non-blocking, just log.
  The `redeemPositions` selector is the minimum ABI needed; hardcode it if the full ABI isn't available.

Acceptance: when a market in our position set resolves, the handler logs `MarketResolved` and within two
poll cycles logs `PositionRedeemable` + attempts `redeemPositions`.

---

### Deliverable C — Composite NegRisk inventory tracker (Gap 2)

New module `maker/negrisk_inventory.py`:

- Background poller (every 30s) on
  `data-api.polymarket.com/activity?user=<config.funder>&type=SPLIT,MERGE,REDEEM,CONVERSION`.
- Parse each row and apply the delta to an in-memory `basket_inventory: dict[condition_id, float]`
  tracking net basket-level exposure (not just per-token fills from the CLOB journal).
- Persist state to a JSONL file (same pattern as `journal/`) so it survives restarts.
- Expose `get_basket_exposure(condition_id) -> float` so the maker engine can query effective exposure
  before placing a new quote.
- Note: RTDS is confirmed silent for split/merge/redeem/convert (NEGRISK_FINDINGS §2). This poller is the
  only off-chain non-contract path to see these events.

Acceptance: after a MERGE event on our funder wallet, `get_basket_exposure` reflects the reduced position
within one poll cycle.

---

### Deliverable D — Minimal passive maker engine (Gap 4)

New module `maker/maker_engine.py`:

Phase-2 measurement scaffold — NOT a production quoting system.

- One market at a time via `POLYMARKET_MAKER_CONDITION_ID` env var.
- Two-sided quoting: one GTC/GTD bid + one ask, sized at 1 contract (`MAKER_SIZE_CONTRACTS=1` env var).
- Quote placement: join the best bid and ask. No improvement, no chasing. (Carry is the edge; chasing
  imports the K-PEG exit-spread tax.)
- Cancel-and-replace: every `MAKER_REFRESH_SEC` seconds (default 30), cancel existing quotes and replace
  if the book has moved > 1 tick.
- Safety harness: same `MAX_REAL_ORDERS` and `REQUIRE_OPERATOR_CONFIRM` as the copytrade bot.
- Basket cap: if `negrisk_inventory.get_basket_exposure(condition_id) >= 10`, skip quoting. Hard Phase-2 cap.
- **Telemetry** (the whole point of Phase 2): log every quote placement, cancel, fill, and missed fill with:
  - `top_maker_rank_at_fill`: our rank among all fills at that price in a 5s window
  - `post_fill_price_drift_60s`: price 60s after fill minus fill price
  - `news_proximate`: bool from `event_calendar.is_event_proximate(fill_time, window_minutes=30)`
    (or `None` if calendar not loaded)
  - `fill_share_this_market`: our fills / total market fills over the session
- On resolution: cancel all open quotes, let the resolution handler redeem.

Acceptance: places a two-sided passive quote, refreshes it, cancels on resolution, logs all telemetry fields.

---

### Deliverable E — Scheduled-event calendar

New file `maker/event_calendar.py` + `maker/politics_events.yaml`:

- YAML format: list of `{date, description, window_minutes}` entries.
- Starter content: next 30 days of scheduled elections, major court dates, Congressional confirmation votes.
  Five to ten entries is fine — this is a manually-maintained ops file.
- Python module exposes `is_event_proximate(timestamp: datetime, window_minutes: int = 30) -> bool`.

---

### What NOT to build

- No dynamic peg-chasing. No K-PEG style logic.
- No Binance anchor, vol model, or OD pricing.
- No leader-mirroring logic.
- No X/Twitter or LLM news feed. Scheduled calendar only.
- No multi-market orchestration. One market at a time.
- No changes to `_kernel/` (frozen).

---

### Output

- `polymarket/execution/maker/` directory with the five modules above.
- Tests in `polymarket/execution/tests/maker/` following existing test patterns.
- Update `polymarket/execution/PLAN.md` with Phase-1 decisions and maker engine architecture.
- Write `polymarket/execution/maker/README.md` describing the module and Phase-2 purpose.
- Do NOT write a handoff note — Cowork handles that.
```
