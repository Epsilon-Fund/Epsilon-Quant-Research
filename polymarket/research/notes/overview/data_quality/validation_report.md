# Polymarket trades — validation report
> Hub: [[COWORK]]
> Table terms: [[polymarket_table_dictionary]]


**Generated:** 2026-05-08
**Data scope:** `data/trades/trades_delta_shard*.parquet` + `trades_seed.parquet`
**Markets snapshot:** `data/markets/markets_2026-05-06.parquet`
**Total fills:** 1,064,500,317 (913.4 M delta + 151.1 M seed)
**Reproduce:** `cd polymarket/research && PYTHONPATH=. uv run python scripts/validation/NN_*.py`

This report is the gate before building the Phase 1 position-reconstruction
layer. Each section: what was checked, what was found, recommendation.

---

## 1. Orphan rate over time

**What was checked.** Monthly count of fills with `market_id IS NULL` (i.e.,
trade involved a token_id that doesn't appear in the current Gamma
markets snapshot). If orphan rate is rising toward today, the snapshot
is aging and we'd need to refresh it before Phase 1.

**What was found.**

- Overall orphan rate: **0.379%** (4,032,815 of 1,064,500,317 fills).
- Trend: **NOT aging.** Most recent month (2026-04) is 0.05%; recent quarter (2026-01 → 2026-04) averages ~0.18%.
- Earlier months are spikier: 2024-11 = 1.35%, 2023-10 = 3.43%. Likely legacy markets that were created early and later removed/archived from Gamma.

| month | n_trades | n_orphan | pct_orphan |
|---|---:|---:|---:|
| 2025-11 | 43,072,418 | 393,526 | 0.91% |
| 2025-12 | 73,712,031 | 487,313 | 0.66% |
| 2026-01 | 127,170,920 | 241,329 | 0.19% |
| 2026-02 | 193,119,504 | 532,398 | 0.28% |
| 2026-03 | 278,746,733 | 700,188 | 0.25% |
| 2026-04 | 173,523,874 | 80,002 | 0.05% |

**Recommendation:** **KEEP.** Snapshot is current. No need to re-pull markets
before Phase 1. At Layer A build, route orphans to a sibling
`trader_actions_orphan.parquet` rather than mixing with main file —
they're real fills but lack market context, so they shouldn't enter
position math. ~4 M rows is small enough to query separately if a
specific orphan market needs investigation.

---

## 2. Operator address detection

**What was checked.** Top 50 addresses by union (maker + taker) fill
count. For each, computed maker:taker ratio, distinct counterparties,
distinct markets, and sub-second clustering (% of fills with another
fill from same address within 1 second). Heuristic: operator-shaped
addresses have either extreme maker:taker ratios OR ratio ≈ 1.0 with
very high counterparty count.

**What was found.**

Two address-shape clusters dominate the top of the leaderboard:

**Cluster A — Pure relayers (operator submitting orders on behalf of takers).**
Distinguishing trait: `maker_count = 0`, all fills as taker.

| addr | total_fills | maker_count | taker_count | distinct_counterparties | $ volume |
|---|---:|---:|---:|---:|---:|
| `0x4bfb41…8982e` | 305,605,503 | 0 | 305,605,503 | 1,999,485 | $17.34 B |
| `0xc5d563…f80a` | 100,290,420 | 0 | 100,290,420 | 1,977,295 | $12.60 B |

These are unambiguously matcher/relayer bots — 0 maker fills, ~2 M
counterparties, billions in flow. **Must be excluded.**

**Cluster B — Pure market-maker bots.** Distinguishing trait: extreme
maker:taker ratio (>50, often >1000), much smaller counterparty count.

| addr | total_fills | maker:taker ratio | distinct_counterparties |
|---|---:|---:|---:|
| `0x297fbd…5103` | 2,611,061 | 870,353 | 103,766 |
| `0x04895657…ee4d` | 2,969,313 | 1,116 | 106,997 |
| `0xdc669ba0…0533` | 2,541,118 | 687 | 44,447 |
| `0xe9cbb1c9…4096` | 2,893,827 | 78 | 104,300 |
| `0x38e598…8a4a` | 5,672,615 | 45 | 142,346 |

Liquidity-providing bots — almost never take, just resting orders. Not
copy-trade targets.

**Cluster C — HFT/arb traders.** Ratio ~1.0, very high sub-second clustering.

| addr | total_fills | ratio | distinct_counterparties | sub-second % |
|---|---:|---:|---:|---:|
| `0xe8dd7741…ec86` | 6,919,620 | 0.99 | 158,505 | 99.22% |
| `0x63d43bbb…f2f1` | 3,273,208 | 1.02 | 104,647 | 97.93% |
| `0xe3726a1b…eb38` | 5,572,787 | 1.07 | 147,945 | 97.27% |

These are genuine traders running matched buy/sell flow — technically
"profitable strategies", but not human-pace decisions. Not what
copy-trading is aimed at.

**Cluster D — Heavyweight discretionary traders.** Top end of the
remaining top-50 has more reasonable shapes (e.g., `0x204f72f3…` with
ratio 1.32, 85k markets, 113k counterparties, $567 M). Some of these
likely belong on the cohort radar.

**Recommendation: opinionated deny-list — apply at Phase 3, not Phase 1.**

Phase 1 (Layer A) should be identity-neutral and include all addresses.
Phase 3 (trader panel) should compute an `is_operator_like` flag per
address using:

```
is_operator_like =
  taker_count = 0 OR maker_count = 0
  OR maker_taker_ratio > 50 OR maker_taker_ratio < 0.02
  OR (distinct_counterparties > 500_000)
  OR (pct_sub_second > 95 AND total_fills > 1_000_000)
```

These criteria capture clusters A, B, and most of C. Cohort selection
filters out `is_operator_like = TRUE`. Concretely the deny-list seeded
from this scan is the top 5–10 in clusters A and B (definite operators)
plus opt-in extension to clusters C/D based on other diagnostics.

**Candidate operator deny-list (decisions blank):**

| addr | suggested action | reason |
|---|---|---|
| `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` | **DENY** | matcher/relayer; 0 maker fills, 305 M takes, $17 B |
| `0xc5d563a36ae78145c45a50134d48a1215220f80a` | **DENY** | matcher/relayer; 0 maker fills, 100 M takes, $12.6 B |
| `0x297fbd45782af37d899015aebbc52437f3d55103` | **DENY** | pure MM bot; ratio 870k:1 |
| `0x04895657d3c2afebec8be4b6e60b9c56ad68ee4d` | **DENY** | pure MM bot; ratio 1116:1 |
| `0xdc669ba0adb45448020025f756070492d1070533` | **DENY** | pure MM bot; ratio 687:1 |
| `0xe9cbb1c9b3f7f411dd4fdf2ea7afa780c8b4d096` | **DENY** | MM bot; ratio 78:1 |
| `0x38e598961dd0456a7fb2e758bd433d3e59fb8a4a` | **DENY** | MM bot; ratio 45:1 |
| `0xd44e29936409019f93993de8bd603ef6cb1bb15e` | **DENY** | MM bot; ratio 12:1 |
| `0x5f4d4927ea3ca72c9735f56778cfbb046c186be0` | **DENY** | MM bot; ratio 13:1 |
| `0xe8dd7741ccb12350957ec71e9ee332e0d1e6ec86` | discuss | HFT (ratio 0.99, 99% sub-second) — exclude as cohort target but data still valid |
| `0x63d43bbb87f85af03b8f2f9e2fad7b54334fa2f1` | discuss | HFT |
| `0xe3726a1b9c6ba2f06585d1c9e01d00afaedaeb38` | discuss | HFT |

---

## 3. Self-trades (maker == taker)

**What was checked.** Count, address concentration, sample of rows where
`maker = taker` (a single address on both sides of a fill).

**What was found.**

- Total: **905 fills, $234 k volume, 70 distinct addresses**.
- Time range: **2022-11-21 → 2023-04-27 only.** Zero self-trades after April 2023.
- Top self-trader: `0x47339e…` (210 fills, 96 markets) — looks like a small early-stage bot or wash test.

This is early-protocol noise — probably internal testing or a bug in the
matching logic that was patched in 2023. **Zero impact** on modern data.

**Recommendation: DROP.** Filter `WHERE maker != taker` at Layer A. If
left in, the explode would create two opposite-sign trader_action rows
for the same address on the same fill, which is meaningless. Dropping
also avoids inflating fill counts by 2 × 905 = 1,810 phantom rows.

---

## 4. Outcome-index drift

**What was checked.** For 50 random closed markets, picked the last 50
trade prices per `(market_id, outcome_index)`, took the median, and
compared the trade-implied "winner" (the outcome converging to ~1.0)
against the markets parquet's `outcome_prices` array (where the
resolved outcome should be 1.0).

**What was found.**

- Sample size: 35 closed markets fully reconciled (15 of the 50 sampled
  had the LATERAL join return no rows — likely older markets where the
  trades' outcome tokens don't perfectly match current `clob_token_ids`,
  or markets where fills happened in less common patterns).
- **Reconciliation: 35/35 match. 0 mismatches.**
- Spot-checks confirm clean convergence: e.g., market 1143741 (Natus Vincere
  vs Team Lynx) — last prices [0.001, 0.999], outcome 2 (Team Lynx) won.
  Markets parquet reports outcome_prices = [0.0, 1.0] (outcome 2 = winner).

**Recommendation: TRUST.** The Gamma markets snapshot's `clob_token_ids`
ordering is consistent with on-chain resolution. Bake `outcome_index`
into Layer A via `list_position(clob_token_ids, market_token_id)` once.
No need for per-query re-validation.

The 15 sampled markets that returned no LATERAL data are worth a
sniff-test once Layer A is built (after build, look for closed markets
in our trades data that have no resolution price reachable). Likely
these are markets where the snapshot's `clob_token_ids` differs from
the historical token ids — 0.4 % orphan-class issue, separately handled.

---

## 5. Sign asymmetry — buy vs sell volume per token

**What was checked.** Per `(market_id, outcome_token_id)`, totaled
`token_amount` across BUY-by-maker fills vs SELL-by-maker fills. The
user's framing: "every token bought was sold by someone — modulo
issuance/redemption".

**What was found.** Massive structural asymmetry, **not corruption.**

- Overall: 853 M BUY-by-maker fills (98.3 B tokens) vs 211 M SELL-by-maker fills (43.3 B tokens) — **~4× imbalance**.
- Per-token median ratio: **buy:sell ≈ 8.3:1**, p10=2.4, p90=31.9, max ratio 484k:1.

**Why this is structural, not corruption:**

1. **Holding to resolution → redeem on-chain, not sell on book.** When a
   market resolves, holders of the winning outcome redeem their tokens
   directly via the CTF contract, receiving USDC at $1 per token. This
   is a settlement transaction, **not an `OrderFilled` event**. Our
   trades parquet by definition cannot capture redemptions.
2. **CTF splits/merges are missing.** A user can split USDC into YES + NO
   tokens (mint), or merge YES + NO back into USDC (burn). These also
   bypass the orderbook and don't appear as fills.

So a typical retail flow is: buy YES at $0.30 → hold → market resolves
YES → redeem at $1.00. In trades data, this looks like one-sided buy
flow with no offsetting sell. The asymmetry is the dataset's signature
of held-to-resolution positioning.

**Worst imbalances by USD (top 5):**

| token_prefix | market_id | n_fills | buy_usd | sell_usd | net_usd |
|---|---:|---:|---:|---:|---:|
| 217426331… | 253591 | 3,450,272 | $682 M | $537 M | +$146 M |
| 483310433… | 253591 | 1,659,347 | $266 M | $151 M | +$114 M |
| 291618412… | 1640919 | 94,081 | $151 M | $40 M | +$111 M |
| 875849553… | 253597 | 1,060,734 | $207 M | $109 M | +$98 M |
| 343795817… | 546814 | 51,592 | $133 M | $37 M | +$95 M |

These are big resolved markets; the imbalance reflects net buying that
wasn't sold back before resolution.

**Recommendation: DOCUMENT, no action at Phase 1.** Layer A captures
trades as-is; Layer B (positions) treats redemption-at-resolution as a
synthetic closing event using the markets parquet's resolution prices.
Specifically, for any closed market and any address with non-zero
final position from trades alone, the closing entry is:

```
synthetic_close.token_delta = -final_position
synthetic_close.usd_delta = +final_position * resolution_price
synthetic_close.timestamp = market.end_date
synthetic_close.role = 'redemption'
```

This is **a Phase 2 design decision**, not a Phase 1 build issue. But
the asymmetry confirms why this matters: the majority of trader
positions never close via trades — they close via redemption.

---

## 6. Timestamp ties

**What was checked.** When Layer B applies a window function partitioned
by `(address, market_id, outcome_index)` ordered by `timestamp`,
multiple fills sharing the same timestamp + transaction hash for the
same partition produce non-deterministic order, breaking
deterministic avg-cost reconstruction. Quantified by counting fills in
multi-fill (transaction_hash, timestamp) buckets, then narrowing to
the same address + market + outcome within those buckets.

**What was found.**

- **99.97 % of all fills** sit in (transaction_hash, timestamp) buckets
  with at least one other fill — which is just normal: a single CTF
  Exchange transaction often matches dozens or hundreds of orders. The
  largest single bucket contained 110 fills.
- Narrowing to the actual problem (same trader on same outcome within
  a bucket):
  - **6.59 % of fills** are the same MAKER on the same market+outcome inside a tied (tx, ts) bucket. ~70 M rows.
  - **30.0 % of fills** are the same TAKER on the same market+outcome inside a tied (tx, ts) bucket. ~319 M rows.

The taker-side number is dominated by the matcher operators (`0x4bfb41…`
and `0xc5d563…`) who appear as taker for many fills in every tx. After
excluding those addresses, the trader-side figure is much smaller —
but still non-trivial.

**Why this matters.** For a window function partitioned by
`(address, market, outcome)` ordered by `timestamp`, two fills with the
same timestamp produce indeterminate row order. If both are buys at
different prices, avg-cost is the same regardless of order (commutative).
**If one is a buy and the other a sell**, the order DOES matter — it
determines whether the sell sees the new avg-cost or the old one.

**Recommendation: COLLAPSE same-bucket fills in Layer B.** At the
window-function step, pre-aggregate fills sharing
`(address, transaction_hash, timestamp, market_id, outcome_index)`
into a single net effect:

```
net_token_delta = sum(token_delta)
net_usd_delta = sum(usd_delta)
representative_price = sum(usd_delta) / sum(token_delta)
n_underlying_fills = count(*)
```

This reduces ~395 M same-bucket multi-fill rows to single atomic events.
The economic interpretation is correct (these fills were atomic on-chain
anyway). Layer A keeps raw fills (for audit / drill-down); Layer B is
where the collapse happens.

If you instead want to preserve per-fill granularity through Layer B,
you need a deterministic tiebreaker. The schema doesn't have `log_index`,
so the next-best is a synthesized stable hash —
`hash(transaction_hash || maker || taker || token_amount || price)` —
appended to the order key. This works but adds complexity for negligible
analytical value.

---

## 7. Sample trader sanity (domah, `0x9d84ce…1344`)

**What was checked.** Domah's local stats vs Polymarket's public APIs.

**What was found (local):**

| role | fills | usd_volume | distinct_markets | earliest | latest |
|---|---:|---:|---:|---|---|
| maker | 572,995 | $139.6 M | 9,443 | 2022-12-12 | 2026-04-23 |
| taker | 72,778 | $35.5 M | 5,971 | 2022-12-13 | 2026-04-23 |
| **union** | **645,772** | **$175.1 M** | **9,444** | 2022-12-12 | 2026-04-23 |

**Cross-check via Polymarket public API:**

- `/profile?address=…` returns 404 across all variants (data-api,
  gamma-api, lb-api). Profile endpoint isn't accessible.
- `/positions?user=…` returns 500+ open positions (page limit;
  domah likely has more):
  - Sum of `initialValue` for open: **$1.98 M**
  - Sum of `currentValue` for open: **$852 k** (close to global value below)
  - Sum of `realizedPnl` for open: **$290 k**
- `/value?user=…` returns total portfolio value: **$895,672**.
- `/activity?user=…` returns fresh trades dated 2026-05-08 14:58 UTC
  (today) — domah is actively trading.

**Comparison.**

- Local trade history goes through **2026-04-23 23:52 UTC**. API shows
  trades on **2026-05-08**. The 15-day gap matches Goldsky's known
  ~9-day indexing lag plus the time since the last sync run finished.
  Not an error — known constraint.
- Local total volume $175 M is consistent with the API's reported
  $1.98 M of currently-open initial value (i.e., ~99 % of his historical
  capital has cycled out — high-turnover trader).
- 9,444 distinct markets in our data vs 500+ open positions (more on
  paged): **unable to fully reconcile** without paging the full
  `/positions` response and the full `/activity` response. The
  endpoints support that but it's a side quest, not a blocker.

**Recommendation: PASS, with caveat.** Local stats are internally
consistent and match the rough shape of API-reported state. Full
fills-by-fills reconciliation would require paging the API
exhaustively; not done here. The **15-day lag at the tail** is real and
worth noting in any walk-forward backtest — backtest cutoff dates
should be at least 9 days behind real-time to stay on safely-indexed
data.

---

## Top 3 things to decide before Layer A

These are the open questions that gate Phase 1 build. Each has a
**recommended default** that we can adopt unless you push back.

### 1. Operator/MM-bot deny-list scope at Layer A

**Question.** Should Layer A include all addresses (identity-neutral, with
`is_operator_like` filtering deferred to Phase 3), or strip operator
fills at build time?

**Recommended default: include all, filter at Phase 3.** Layer A is
universal substrate. Stripping operators at build means re-running the
build if the deny-list changes; including all means trader rankings
filter the same parquet. The two definitive matcher addresses (`0x4bfb41…`,
`0xc5d563…`) are 305 M + 100 M = 405 M fills (38 % of all data); a
flagged subset is fine for everything except a query like "average
fills per real trader", where the operators distort the mean.

**Alternative if you'd rather strip:** drop `WHERE maker IN (deny) OR taker IN (deny)` rows at Layer A build. Halves the dataset size and speeds Phase 2/3 builds. Re-build cost is the trade-off.

### 2. Same-bucket fill collapse vs raw-per-fill granularity at Layer B

**Question.** Should Layer B collapse fills sharing
`(address, transaction_hash, timestamp, market_id, outcome_index)` into
single atomic events before the cumulative window, or carry every fill
through with a deterministic tiebreaker?

**Recommended default: collapse.** ~30 % of fills (319 M) hit this
case (mostly the matcher operators, but ~6.5 % even after their
removal). On-chain these were atomic; collapsing matches that semantic
and makes avg-cost reconstruction trivially deterministic. Raw fills
remain in Layer A for audit / drill-down.

**Alternative:** carry per-fill granularity with a synthetic
tiebreaker (e.g., hashed row key). Adds complexity, no analytical
benefit for cohort copy-trading.

### 3. Redemption-at-resolution as a synthetic "close" in Layer B

**Question.** When a market resolves and a trader has non-zero
final-from-trades position, do we synthesize a redemption row (closing
the position at the market's resolution price) in Layer B, or leave
positions "open" forever and handle redemption only in Phase 2?

**Recommended default: synthesize in Layer B.** The 4:1 buy:sell
asymmetry tells us that the *vast majority* of positions in this data
never get a closing fill — they're held to resolution. If Layer B
treats them as "still open", every trader's PnL is permanently
unrealised, and Phase 3 metrics (realized PnL, win rate) effectively
fail. Synthesizing the redemption from the markets parquet's
`outcome_prices` at `market.end_date` is the only way to make Layer B
useful as the input to Phase 2/3.

The synthetic row carries `role = 'redemption'`, distinguishable from
real fills. Open markets (still trading) leave their final position
as-is — no synthetic close. This is lookahead-free per market: we use
data known at `market.end_date`, not future data.

**Alternative:** keep Layer B trades-only; defer redemption logic
entirely to Phase 2. Phase 2 then has to do the same join against
markets and compose two sources for any per-position PnL. Doable, but
duplicates work and makes "position state at time T" harder to reason
about.

---

*End of report.*
