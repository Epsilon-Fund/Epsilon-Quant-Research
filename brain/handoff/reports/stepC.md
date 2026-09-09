# STEP C — Fetch Gamma, build `tokens_raw.parquet`   [DONE]

Written by Claude Code, 2026-08-27. Four programs, file boundaries between them, per the approved design. Chained into D (report below).

## WHAT I RAN
- `data_layer/c1_targets.py` — promote the Step-B condition scan to `targets.parquet` (one row per condition+asset) + `targets_rollup.parquet`.
- `data_layer/c2_fetch.py` — two-pass Gamma fetch (open + `closed=true`), batches of 100, repeated `condition_ids`, `limit`=batch, `User-Agent`; one verbatim market object per condition to `gamma_cache/<cid>.json`, `.notfound` only when both passes empty; `fetch_log.csv` + `fetch_batchlog.csv`.
- `data_layer/c3_parse.py` — cache-only parse (no network) → `tokens_raw.parquet`.
All outputs in the gitignored `polymarket/research/data/gamma/`.

## WHAT I FOUND
- **C1:** `targets.parquet` = **30,772 tokens / 15,386 conditions**, all exactly 2 assets (matches Step B).
- **C2:** **15,351 cached, 35 not-found**, in **164 s**. Reconcile **exact**: 15,351 + 35 = 15,386. The ignored-param assertion (every returned `conditionId` ∈ batch) never fired. Miss rate **0.23%**, all in esports — well under the 5% contingency.
- **C3:** `tokens_raw.parquet` = **30,772 rows** — resolved **30,702**, unresolved **70** (the 35 not-found × 2), **0 quarantined** (every parsed market had length-2, position-aligned `outcomes`/`clobTokenIds`). Outcome labels: **YES 2,418 / NO 2,418** (all 1,427 politics markets + ~991 Yes/No-style esports props), the remaining 25,936 tokens carry team/other labels (`outcome` left null, `outcome_label` kept verbatim). All ids stored as text.

## WHAT SURPRISED ME
- **~991 esports markets are literally Yes/No** ("Both teams slay Baron?", tweet-range props), not team-vs-team — so YES/NO is not purely a politics thing, though team-name outcomes still dominate esports.
- **All 35 not-found are esports.** Consistent with the esports long tail of obscure/delisted sub-markets. Kept as `unresolved`, not dropped.

## WHAT I DID NOT DO
- Built **no** data table (`tokens.parquet` and the four time-series tables are out of scope for this chain). Only `tokens_raw.parquet`, the identity table.
- No R2 writes; no secrets in any command string (Gamma is public).

## FILES CHANGED
- `polymarket/research/data/gamma/`: `targets.parquet`, `targets_rollup.parquet`, `gamma_cache/` (15,351 `.json` + 35 `.notfound`), `fetch_log.csv`, `fetch_batchlog.csv`, `tokens_raw.parquet`.
- `polymarket/research/data_layer/`: `c1_targets.py`, `c2_fetch.py`, `c3_parse.py` (code, uncommitted).

## FIGURES
None (identity step). Verification figure is in Step D.

## PROPOSED NEXT
Step D — verify the mapping (chained; report follows).
