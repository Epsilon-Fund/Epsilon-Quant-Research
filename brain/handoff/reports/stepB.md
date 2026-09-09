# STEP B — Scope the Gamma fetch   [STOPPED FOR DECISION]

Written by Claude Code, 2026-08-27. Scope-only; no full fetch. Reports into the handoff structure per `PROTOCOL.md`.

**Headline:** the procedure is sound but needs three fixes before it survives the real API, and the target is **~10× larger than the brief predicts** (15,386 conditions vs the brief's "~3,000 tokens"). Both are stop triggers under `PROTOCOL.md`, so I stopped at the end of B instead of chaining into C.

---

## WHAT I RAN

- **Sizing (metadata-only, no bulk download):** DuckDB `httpfs` over R2, reading **only the `market` column** of every `book` file across all 64 days (projection pushdown). R2 credentials read from `rclone.conf` inside the script, never printed. One-day latency test first (5.5 s), then the full 64-day scan (234 s ≈ 3.9 min).
- **API probe:** ~10 real condition ids (politics + esports, incl. 3 resolved esports and 2 resolved politics), against `https://gamma-api.polymarket.com/markets`. Tested param name, batching, page-size, closed-market behaviour, encoding, event nesting, resolution fields, rate limits, and a round-trip against the archive.

## WHAT I FOUND

### 1 · Size of the target
- **15,386 distinct `condition_id`** across 64 days — **esports 13,959, politics 1,427**.
- **Every condition has exactly 2 `asset_id`. Zero anomalies** (distribution `{2: 15386}`). → ~30,772 tokens.
- Driver: esports has **many sub-markets per match** (moneyline, map winner, game handicap, …), each its own condition. Politics (1,427) is in line with expectation; esports is the multiplier.
- Artefact written: `_reports/stepB/condition_target_list.csv` (one row per condition: `condition_id, universe, n_assets, asset_ids`). This is program #1 ("build target list").

### 2 · The API, as it actually is
- **Endpoint:** `GET /markets?condition_ids=<0x…>` → JSON array of market objects. Repeat the param to batch (`condition_ids=a&condition_ids=b`).
- **Param must be plural `condition_ids`.** Singular `condition_id` is **silently ignored** (returns an unfiltered page — looks like success, wrong data).
- **A `User-Agent` header is required** — default library UA gets `403`; curl/browser UA works.
- **Closed markets are excluded by default.** A resolved condition returns `[]` until you add `&closed=true`. Open markets return only under the default (no `closed`). To capture everything you must do **two passes per batch** (default → open, `closed=true` → closed) and union.
- **Batching limits:** max **100 ids per call** (120 → HTTP 422). Default page size is **20** — passing 100 ids with no `limit` returns only 20, **silently**. Must set `limit` ≥ batch size. Comma-joined ids return **0** (silent), not an error — must use the repeated param.
- **Encoding:** `outcomes`, `clobTokenIds`, `outcomePrices` come back as **JSON-encoded strings**, needing a second `json.loads`. Token ids inside are 77-digit — keep as text.
- **`outcomes` content:** politics = literally `["Yes","No"]`; **esports = team names**, e.g. `["Hanwha Life Esports","Dplus KIA"]`. So the YES/NO model is politics-only; esports labels are teams (NEXT.md C3 already anticipates this).
- **Event is nested** in the market: `market.events[0]` carries `id, title, slug, endDate, negRisk, ticker, series, gameId, …`. Event grouping needs **no separate fetch**.
- **Resolution** (for a closed market): `closed=true`, `umaResolutionStatus="resolved"`, `outcomePrices=["1","0"]` names the winner by position; `closedTime`, `resolvedBy` present. Closed markets **are** served (with `closed=true`).
- **Round-trip check (Step D #1 feasibility):** Gamma `clobTokenIds` were **set-equal to the archive `asset_ids`** for both probed conditions (politics open + esports resolved). Concretely verified, both universes.
- **Rate limits:** no rate-limit headers advertised; 25 rapid calls at ~**8 req/s**, zero 429s. Generous; be polite anyway.
- Full raw responses saved for reading: `_reports/stepB/raw_market_politics_open.json`, `_reports/stepB/raw_market_esports_resolved.json`.

**Two concrete mapped examples (outcome[i] ↔ clobTokenIds[i]):**
- politics `0xac02…b830` "Will the Fed decrease rates by 25bps…" · event *"Fed Decision in September?"* (`481717`) · `Yes→5774…`, `No→2823…` · endDate 2026-09-16 · negRisk true · **open**.
- esports `0x764f…217b` "Game Handicap: HLE (-1.5) vs Dplus KIA (+1.5)" · event *"LoL: Dplus KIA vs Hanwha Life Esports (BO3)"* (`850529`) · `Hanwha Life Esports→2232…`, `Dplus KIA→3319…` · outcomePrices `["1","0"]` (Hanwha won) · **resolved**.

### 3 · The plan, and where the procedure breaks
**Fetch time estimate:** 15,386 conditions ÷ 100 per call × 2 passes ≈ **308 calls** → ~**5–15 min** with polite pacing, plus writing 15,386 verbatim JSON files. Resumable via skip-if-exists.
**Cache layout:** `gamma_cache/<condition_id>.json`, one unmodified response per condition, in a gitignored data dir (propose `polymarket/research/data/gamma_cache/`). Not R2, never committed.

**Where the four-program procedure does NOT survive contact — must be fixed before C runs:**
1. **C2 "empty → `.notfound`, not retried" will misfire.** The default query returns empty for *closed* markets, so ~14k resolved esports (+ resolved politics) would be wrongly marked not-found and never fetched. **Fix:** status-agnostic fetch — two passes per batch (default + `closed=true`), union; write `.notfound` only when **both** passes are empty.
2. **Batch semantics are silent-failure-prone.** Comma-join → 0; default `limit=20` → truncates. **Fix:** repeated `condition_ids` param, ≤100 ids/call, `limit` = batch size.
3. **Scale is ~10× the brief.** 15,386 conditions ≈ 30,772 tokens vs the brief's ~3,000. Cheap in time/bytes, but it changes expectations and should be confirmed before committing.
Plus: esports `outcome` is a team name (≈91% of rows), so the `outcome=YES/NO` column is null for most tokens and Step D **check 3 (NegRisk YES-sum) is politics-only**; esports orientation must lean on check 4 (resolution) + check 2 (pairing). And a `User-Agent` header is mandatory.

## WHAT SURPRISED ME
- **15,386 conditions, not ~3,000** — esports sub-markets per match dominate. Materially off from the brief.
- **The default `/markets` query hides every resolved market.** Following C2 literally would silently drop the majority of the dataset.
- Everything else (esports team-name outcomes, JSON-string encoding, nested event, exact-2-asset universality, round-trip equality) matched or refined what the brief/Step A already flagged.

## WHAT I DID NOT DO
- Did not fetch the full set (no Step C). Did not proceed to Step D. Built no data table.
- No writes outside `_reports/stepB/` and the handoff files. No R2 writes. No secrets in any command string.

## FILES CHANGED
- `_reports/stepB/condition_target_list.csv` (15,386 rows), `fullscan_out.txt`, `raw_market_politics_open.json`, `raw_market_esports_resolved.json`.
- Handoff: this report, `STATUS.md` (overwritten), `LOG.md` (appended).

## FIGURES
None — this step is scoping. Artefacts above are text/CSV/JSON.

## PROPOSED NEXT
Operator/Cowork decision, then C→D:
1. **Confirm the 10× scale** (15,386 conditions / ~30,772 tokens) is expected and acceptable.
2. **Approve the corrected fetch** for C2: two-pass (default + `closed=true`) union, repeated `condition_ids` param, ≤100 ids/call with `limit`=batch, mandatory `User-Agent`, and `.notfound` only when both passes are empty.
Then Step C fetch is ~5–15 min; parse + Step D verification minutes more. I stopped here per `PROTOCOL.md` (materially-off number + a fetch instruction that would mislabel the majority) rather than chain into C.
