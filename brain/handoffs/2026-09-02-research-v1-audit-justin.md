---
title: "Handoff — research_v1 independent audit: the mapping is sound, its evidence isn't; one real build defect"
created: 2026-09-02
status: open
owner: justin
project: copytrade
para: area
hubs:
  - COWORK
  - POLYMARKET_BRAIN
  - TODO
tags:
  - handoff
  - audit
  - data-quality
  - polymarket
  - research-v1
---

# Handoff 2026-09-02 — research_v1 independent audit

> Hubs: [[COWORK]] · [[POLYMARKET_BRAIN]] · [[TODO]] · Dataset docs: `polymarket/research/epsilon_data/README.md`, `polymarket/research/HANDOVER.md`
> Full audit, committed with this note: [`polymarket/research/docs/audit_2026-09/`](../../polymarket/research/docs/audit_2026-09/) — [01_digest.md](../../polymarket/research/docs/audit_2026-09/01_digest.md) · [02_dashboard_walkthrough.md](../../polymarket/research/docs/audit_2026-09/02_dashboard_walkthrough.md) · [04_audit_plan.md](../../polymarket/research/docs/audit_2026-09/04_audit_plan.md) · [03_numbers.json](../../polymarket/research/docs/audit_2026-09/03_numbers.json) · raw outputs in [04_results/](../../polymarket/research/docs/audit_2026-09/04_results/)
> Scripts: `polymarket/research/scripts/audit_checks/` (now in the index; see its README)

## Plain-English Summary

- **What this is.** An independent audit of `research_v1` — the 2026-06-19 → 08-21 Polymarket order-book capture (30,772 tokens, 101 M L1 rows, 7.2 M trades) — against the two questions that decide whether it can be traded on: *is the token ↔ identity mapping right*, and *was anything silently dropped between raw capture and built library*. Audited repo state: `alvaro` @ `9810c2b`. Read-only against R2 throughout (19 `GetObject`, zero writes).
- **The two answers.** The **mapping is sound** on every one of 110 independently-checked tokens/markets across both universes and two live oracles — including every token the authors' own columns flag as failed. **Nothing was dropped** on any sampled hour: 15,313 trades, hash sets identical, zero missing, zero extra.
- **But the authors' evidence for both is not what the docs say it is.** The code that built `tokens.parquet` and its five `check*` columns is **not in the repo at any commit**; `check3_negrisk` is undefined and False on 72% of what it covers; `check5_indep` covers 0.38% of tokens; `check2_pairing` and the audit's "pair sums to 1" are an algebraic identity that cannot fail; and the shipped reconciliation is **circular** — it compares a column derived from `l1` to a count of `l1`.
- **One real build defect found.** The `l1` dedup's "previous row" state **resets at part-file boundaries**, injecting **195,984 rows (0.19%, esports August 0.40%)** that are not touch-moves — contradicting `epsilon_data/README.md:129`. Not data loss; an undocumented *addition* that biases any touch-move rate on esports Jul/Aug upward and inflates `n_l1_events`.
- **Net read.** The data is **better than its documentation**; the documentation's evidence is **worse than it reads**. Nothing here blocks using `research_v1` — but three of the five identity checks should stop being cited as evidence, and the part-boundary reset should be fixed in the build.

---

## 1. Verdict

| question | verdict | one line |
|---|---|---|
| **Is the token ↔ identity mapping right?** | **Sound on every sample; the authors' own evidence for it is not.** | 60/60 random tokens + 30/30 deliberately chosen `check3`-failed/passed tokens match two fresh oracles on all 8 fields; 20/20 resolved markets end on the oracle's winner. But the code that produced the five check columns was never committed, `check3` is undefined and 72% False, `check5` covered 0.38% of tokens, and the "pair sums to 1" check is an algebraic identity. |
| **Was anything silently dropped between raw and library?** | **No drops found; one undocumented *addition* found.** | On 5 hourly shards across both universes, every trade is present (hash sets identical, 15,313 trades) and `l1` reproduces from raw under a rule the docs never state. The shipped reconciliation is circular. And the dedup resets at part-file boundaries, injecting **195,984 rows (0.19%) that are not touch-moves** — 0.40% of esports August — contradicting `README:129`. |

Checks run, with results: `04_audit_plan.md` §4. Seven of the twelve ranked checks ran this session; #8 (full 15,386-market CLOB sweep) and #9 (raw trades total) were left for the build — see the asks below.

---

## 2. The two real defects

### 2.1 The l1 dedup resets at part-file boundaries — 195,984 non-touch-move rows

`epsilon_data/README.md:129` says "A row exists only where best_bid/best_ask changed." A library-wide scan (`check_l1_dedup_integrity.py`, 13 s) says otherwise:

| universe | month | rows | non-touch-move rows | share | at a part boundary | parts |
|---|---|---:|---:|---:|---:|---:|
| politics | 06 / 07 / 08 | 1.6M / 8.4M / 3.8M | 8 / 10 / 0 | 0.000% | 0 | 1 each |
| esports | 2026-06 | 10,611,146 | 1,414 | 0.013% | 1,264 | 12 (daily) |
| esports | 2026-07 | 44,684,318 | 65,924 | 0.148% | 65,732 | 283 (daily → hourly) |
| esports | 2026-08 | 31,920,440 | **128,628** | **0.403%** | 128,628 | **481 (hourly)** |
| **total** | | 101,051,502 | **195,984** | **0.194%** | 195,624 | |

**Mechanism** (confirmed on the 2026-08-17 h12 shard): esports `l1` is cut into hourly `part-YYYY-MM-DD-HH.parquet` files from late July, and the dedup's "previous row" state resets per part. Every asset's first row in every part is kept whether or not the touch moved — 274 of that hour's 298 assets had a quote in the previous hour; all 298 first rows were kept. Politics is one part per month and shows 18 such rows in 14 M.

**Consequences.** `n_l1_events` for esports tokens is inflated by roughly (#hourly parts the token was quoted in). Any touch-move rate or quotes-per-hour statistic on esports Jul/Aug is biased up by 0.15–0.40%, concentrated at :00 of every hour. The anti-drift and audit tools cannot see it — they compare the library to itself.

**Fix.** Carry LAG state across parts in the build, or drop post hoc the rows where `(bb,ba)` equals the previous row (195,984 rows). Related and undocumented: `l1` partitioning is inconsistent — monthly for politics, daily for esports Jun/early-Jul, hourly from late Jul.

### 2.2 `negrisk_sum()` returns a 20-hour-stale composite, documented as instantaneous

`catalog.py:139-146` promises a sum "**at a common timestamp** — never a sum of per-candidate medians (which is invalid)". The path is `negrisk_sum` → `load_event` → `align_mids` (`_internal.py:219-231`), which floors to 1-second buckets and **`.ffill()`s with no limit**.

Measured on `elon-musk-of-tweets-july-21-july-28` (25 YES legs, 135,931 one-second buckets):

| | value |
|---|---:|
| mean legs actually updating in a given 1s bucket | **1.31 of 25** |
| median age of the quote being summed | **72,836 s = 20.2 hours** |
| p90 / max age | 135 h / **10.1 days** |
| fraction of summed values older than 1 hour | **67.7%** |
| yes_sum median as `negrisk_sum` returns it | **4.528** |
| yes_sum median, true instantaneous (no ffill) | **0.270** |

Three defects, all documentation: (1) "at a common timestamp" is a misnomer — it is the *same class* of error the docstring condemns, with a finer stale clock; (2) `n_live` is `notna().sum()` **after** ffill, so it counts candidates that have *ever* quoted, not candidates quoting then (44.15 vs 1.01 on the 2028 event); (3) `.attrs['note']` says missing candidates pull the sum down *only* — ffill makes dead candidates pull it **up**, which is exactly the right tail `HANDOVER.md:83-85` is puzzling over.

The honest statement: **for these events the instantaneous NegRisk sum is not measurable from this capture**, because fewer than 2 of 25 legs quote in any given second. Fixed in this branch as docs-only (no behaviour change) — see commit 2.

**Downstream.** `HANDOVER.md:83-85`'s replacement figure ("Elon-tweet-range events ~2.35") does not reproduce: those ten events have medians of **1.99–4.53** and maxima to 6.06 under the library's own `negrisk_sum()`, and 0.270 computed truly instantaneously. Treat `~2.35` as a stale draft figure of the same family as the four condemned ones (`01_digest.md` §9). The four condemned figures themselves appear **nowhere** in the repo — that sweep is clean.

---

## 3. Found, not admitted

`01_digest.md` §7 lists the **20 gaps the authors admit**. The 22 items below were found in code or data and are in none of them. **Bold = would change a conclusion someone draws from this data.**

| # | found, not admitted | where |
|---:|---|---|
| 1 | **The code that built `tokens.parquet` and its five checks is not in the repo at any commit**; `built_by_commit` points at the dashboard commit | 04 §1.1 |
| 2 | **`check3_negrisk` is False on 71.9% of the tokens it covers and has no definition**; the mapping is fine regardless | 03 §3; 04 §1.2 |
| 3 | **`check5_indep` covers 116 tokens (0.38%)**; `check3` covers 8.59% — "every token was identity-verified" is not true of those two | 03 §3 |
| 4 | **`check2_pairing`, the audit pair-sum and `test_load_pair…` are an algebraic identity** and cannot fail (worst deviation 1.1e-16 over 162,041 points) | 03 §4; 04 §1.1 |
| 5 | **The shipped reconciliation is circular** — `n_l1_events` *is* `COUNT(l1)` by construction, for all 30,732 assets | 04 §2.1 |
| 6 | **The l1 dedup rule is undocumented**; it is "(bb,ba) differs from the previous row of the same asset, ordered (timestamp_ms, received_ns)" | 04 §2.2 R1 |
| 7 | **195,984 l1 rows are not touch-moves** (0.19%; esports Aug 0.40%) — `README:129` is false at the margin; `n_l1_events` inflated for esports | 04 §2.2 R2 |
| 8 | **`l1` partitioning is inconsistent** — monthly politics, daily esports Jun/early-Jul, hourly from late Jul — and undocumented | 04 §2.2 |
| 9 | **`negrisk_sum()` is a 20-hour-stale composite**, `n_live` counts ever-quoted legs, `.attrs['note']` states the wrong direction | 01 §8.1 |
| 10 | **HANDOVER's replacement NegRisk figure (~2.35) doesn't reproduce** (Elon events 2.0–4.5); 147 of the 320 NegRisk events are esports, which the NegRisk panel text says don't exist | 01 §9; 02 §9; 03 §6 |
| 11 | **The outage is 14:03→08:29 (18 h 26 m), not 15:00→08:29**; two files also say 08:00. Anyone filtering the documented window leaves 57 min of dead capture in their data | 01 §8.3 |
| 12 | **Gamma's `condition_ids` filter silently hides closed markets** unless `closed=true` — anyone re-verifying the mapping the obvious way gets 90% "not found" (27 of my first 30) | 04 §1.2 M1 |
| 13 | **`audit_market()` has no plausibility check** — passes a 579,445-buy / 8-sell market at 0.1¢ as "looks fine" | 02 §6 |
| 14 | **`search("fed july")` returns nothing**; it is the documented example in four places | 02 §1 |
| 15 | `check_setup.py` and the loader never read `.env`; every doc says to use one, and `python-dotenv` is shipped | 01 §8.4 |
| 16 | 24 crossed-book rows (spread to −65.9¢) — undocumented; assume-non-negative code trips | 02 §3 |
| 17 | `METRICS_REFERENCE.md` and `RESEARCH_FINDINGS.md` describe a **different dataset** (copytrade positions, 2022→2026-04), one the repo's own canon audit condemns, with no banner | 01 §8.2 |
| 18 | `brain/handoff/reports/` (the build history `HANDOVER.md:108` points at) does not exist; no `step*.md`, no `LOG.md` anywhere in `brain/` | 01 §8.9 |
| 19 | `pc_file_manifest.txt` does not drive `coverage()` (nothing reads it); `e2_counts.csv` — the only raw ledger — is undocumented; `coverage.parquet` missing from the README file table | 01 §8.6 |
| 20 | The anti-drift test covers 5 tokens, one column, no trades; the property holds broadly (40/40 both, widened) but the shipped test doesn't show it | 04 §3 |
| 21 | NegRisk tab ignores the window selector; 25 s render; audit 12–14 s | 02 §8-9 |
| 22 | `events()` returns a NaN event row; `search()` doesn't apply exclusions while `catalog()` does; `first_seen`/`last_seen` are DOUBLE not int | 01 §8.10, §3 |

Items 1–8 are the ones that matter for trusting the dataset.

---

## 4. Asks for Alvaro

In order:

1. **The build code** that produced `tokens.parquet` and the five `check*` columns — or a statement that it is gone. `_manifest.json.built_by_commit = 043d64f` is the Step I *dashboard* commit; every `build_*.py` present at that commit is the copytrade pipeline. Without it the five checks are unreproducible and the "Step D mapping gate" exists only as `HANDOVER.md` prose.
2. **The definition of `check3_negrisk`.** It is False on 1,900 of the 2,642 tokens it covers, and those tokens map exactly as well as the True ones (20/20 vs 10/10 against both oracles). Until we know what it measured we cannot tell whether any of the 1,900 are informative. If the definition is gone, **delete or redefine the column** rather than ship it.
3. **Is the part-boundary dedup reset known?** (§2.1.) If it was a deliberate trade-off, say so in the docs; if not, the fix is carrying LAG state across parts in the build.
4. **Permission to run checks #8 and #9 once inside the build**, so the two "not run" rows become gates rather than trust: #8 is the full CLOB sweep of all 15,386 markets (~30–60 min, rate-limit risk) and #9 is the raw trades total against 7,227,528 (3,072 `get_object`, ~450 MB — cheap in bytes, expensive in calls). Both are trivial from inside the build, awkward from outside it.

## 5. What landed in this branch

- **Commit 1** — this note, `polymarket/research/docs/audit_2026-09/` (the four audit documents + raw check outputs), and `polymarket/research/scripts/audit_checks/` moved from `.git/info/exclude` into the index with a README mapping each script to the plan section it backs.
- **Commit 2** — mechanical fixes only, no design changes: `.env` loading via `load_dotenv()` (shell env wins); the outage window corrected to `14:03:30Z → 08:29:10Z` in all six places plus a tightened containment test in `audit.py`; the `.venv/bin/pip` install command, the dead `brain/handoff/reports/` pointer, the `"fed july"` search example and the README file table; dataset banners on `METRICS_REFERENCE.md` / `RESEARCH_FINDINGS.md`; the widened anti-drift test and three other test corrections; `negrisk_sum()` docstring/`attrs` corrections (docs only); a universe radio on the Distributions panel.

**Not done, deliberately:** the part-boundary dedup fix (a build change, not a library change — needs the build code, ask #1) and anything touching `negrisk_sum()`'s behaviour.
