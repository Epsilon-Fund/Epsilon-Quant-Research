# STEP E (E0 only) — Which source is authoritative for the L1 touch?   [STOPPED FOR DECISION]

Written by Claude Code, 2026-08-27. E0 is the open question that decides E2's 1.7 GB `l1` table. I stopped here — the esports result fits **none of E0's three branches cleanly**, and the finding changes the procedure's premise. No table built.

## Recommendation (one line)
**Build `l1` from deduped `price_change` for BOTH universes.** The evidence that carries the decision is the **genuinely independent leg: `price_change` ↔ `bba` = 98.4%** once a few ms of clock-skew slack is allowed. The book-reconstruction agreement (99.4%) is a strong *internal-consistency* check — it proves `price_change`'s summary `best_bid`/`best_ask` fields agree with its own level deltas — but it is **not** independent (the reconstruction is built from those same `price_change` deltas), so it cannot corroborate the feed against the outside world. `bba` is a differently-timed emission of the same L1, not a more-truthful tiebreaker. Awaiting your go before the immutable build.

> **Wording correction (Cowork, 2026-08-27):** an earlier draft called the book reconstruction "independent." It is not — it shares the `price_change` stream. The independent number is `price_change`↔`bba` @ 98.4% (50 ms slack). Framing corrected throughout; conclusion unchanged.

## What I ran
`data_layer/e0_source_check.py` + `e0b_timing.py` — bounded sample, 20 assets/universe (10 liquid + 10 thin) across 2026-08-20 hours 10–15 where `price_change`, `bba`, `book` all exist. Read via DuckDB httpfs (predicate pushdown, no bulk download). Book touch reconstructed from snapshot + `price_change` deltas, probed at the comparison timestamps. Tolerance 1¢.

## Three-way agreement (within 1¢)
| pair | politics | esports |
|---|---|---|
| (1) `price_change` vs `bba` | **100.0%** (exact 99.6%), n=143k | **60.1%** (exact 59.3%), n=363k |
| (2) reconstructed `book` vs `bba` | **100.0%**, n=144k | **61.8%**, n=364k |
| (3) `price_change` vs reconstructed `book` | 99.8%, n=39k | **99.4%**, n=30k |

- **Politics:** all three agree → E0 branch 1, clean.
- **Esports:** `price_change` and the book reconstruction agree with **each other** (99.4%) — but that is an *internal-consistency* agreement within the `price_change` stream, not independent corroboration. The independent comparison is against `bba`, which disagrees ~40% **at exact timestamps** — resolved below as clock skew.

## Why esports `bba` looks like it disagrees — it's timing, not data (E0b)
Re-aligning `price_change` onto `bba` with a little slack (nearest within N ms):

| alignment slack | esports pc↔bba within 1¢ |
|---|---|
| 0 ms (exact) | 59.5% |
| **50 ms** | **98.4%** |
| 200 ms | 98.4% |
| 1000 ms | 98.4% |
| 5000 ms | 98.3% |

The gap collapses at **50 ms** and is flat thereafter. On fast esports books, `bba` and `price_change` carry the same L1 but their receive timestamps are offset by tens of ms, so an exact-timestamp join picks a stale `price_change` and reports a false mismatch. Politics books move slowly, so exact alignment already gives 100%. (The n drop from 363k→211k at tolerance>0 is `bba` rows with no `price_change` within the window — bba-only ticks — which is exactly where exact-join was pairing a stale delta.)

**Conclusion:** `price_change`'s `best_bid`/`best_ask` is trustworthy in both universes — established by the **independent** `bba` leg at 98.4% (50 ms slack), with the book reconstruction (99.4%) confirming internal consistency. The Step-3 "29% esports disagreement" was the same exact-alignment / staleness timing effect, not a defect in `price_change`.

## The procedure note E0 asked for
The plan treated `bba` as the ground-truth tiebreaker. **For fast esports books that assumption is wrong** — `bba` is just a lower-cadence, differently-timed copy of the same L1. It remains a useful cross-check but should not be the arbiter. This is the thing that "does not survive contact"; nothing else in E0 does.

## Bonus: esports "near 0.5" cross-tab (from `tokens_raw`, no new compute)
Of **11,189** esports resolved conditions ending with both tokens near 0.5: **391 actually voided** (`outcomePrices=["0.5","0.5"]`) and **10,798 settled 1/0 but never traded to the extreme** (thin props the book left mid-range). So "near 0.5" is overwhelmingly *thin-but-settled*, not *voided* — worth stating precisely in the manifest.

## Three short answers
- **Why this way, not the obvious alternative?** I could have believed `bba` (the named tiebreaker) and declared `price_change` unreliable for esports — which would have forced a book-reconstruction build (slow, and the book is a 15-min anchor). Instead I tested the disagreement and found it was alignment, not data, so full-resolution `price_change` is safe. The alternative would have thrown away the 1.68 GB full-res tape for no reason.
- **What breaks first if this is wrong, loud or silent?** If `price_change` were actually wrong and I build `l1` from it, every spread/mid in the viewer is quietly off on esports — silent. That is why the decision rests on the *independent* `bba` leg (98.4% at 50 ms), with the book reconstruction (99.4%) as an internal-consistency confirmation, not the primary evidence.
- **What would a new person misunderstand?** That "esports pc↔bba = 60%" means the data is bad. It means the two feeds are clock-skewed by tens of ms; with 50 ms slack they agree 98.4%.

## STOP
`reports/stepE.md`, `STATUS.md`, `LOG.md` written. **No `tokens.parquet`, no `l1`/`trades` built.** E0 asked me to stop and report on the non-clean branch; I'm doing that with a recommendation. One word from Cowork ("build from price_change, proceed") and I run E1 (`tokens.parquet`) → E2 (`l1` + `trades`) and stop before `book`.

---

# STEP E1 + E2 — build the library   [DONE · all gates passed — stopped before `book`]

Written by Claude Code, 2026-08-28. E0 (above) confirmed by Cowork; this section covers the build (E1 `tokens.parquet`, E2 `l1`+`trades`), the addendum's manifest↔inventory completeness proof, reconciliation against true footer counts, and the six-date spot check.

## Tables built (gitignored `polymarket/research/data/research_v1/`)
| table | rows | size | notes |
|---|---|---|---|
| `tokens.parquet` | 30,772 | 6.8 MB | the tree — one row per token, identity + observation + 5 verification columns |
| `l1/` | 101,051,502 | 769 MB | deduped to touch-moving rows; `universe=/month=/` (politics month-level parts, esports per-day + per-hour parts) |
| `trades/` | 7,227,528 | 338 MB | `universe=/month=/` |
| `obs_stats.parquet` | 30,732 assets | 2.6 MB | per-asset first/last_seen, n_days, n_l1_events, medians (feeds `tokens`) |
| `exclusions.csv` | 0 rows | — | created empty (header + comment); stays empty, operator-only at load |

Total library ~1.1 GB (vs the ~1.7 GB plan estimate). `changes` (full tier) deliberately not built.

## E1 — `tokens.parquet` assertions (all pass, with numbers)
- rows **30,772** (=30,772 expected) · asset_id **unique** · every condition_id **exactly twice** (min=max=2) · `complement_asset_id` **symmetric across every pair** · `path` **unique** · **no nulls** in universe/condition_id/asset_id/path · `identity_status='unresolved'` **=70** · per-universe esports 27,918 + politics 2,854 = **30,772**.
- `outcome`: YES 2,418 / NO 2,418 / **null 25,936** (esports team-name labels; `outcome_label` kept verbatim in all).
- `check4_status`: near_half 21,522 · converged_correct 6,650 · not_resolved 2,148 · no_convergence 328 · no_price 40 · **inverted 14** (the upsets from Step D) · n/a 70.
- Every id (`asset_id`, `condition_id`, `event_id`) stored as **text**.

## E2 — manifest ↔ inventory completeness (ADDENDUM)
A fresh independent `rclone` listing vs the build's manifest and the Step-1 inventory:
- **price_change 2,984 files / 60.05 GiB** and **trades 2,982 files / 0.71 GiB** — an **exact** match to Step-1's 60.05 / 0.71 GiB. (Fresh total 11,985 vs Step-1 11,984, diff +1 = legacy hour-less `bba.parquet`/`book.parquet` + the `_processed.txt` marker — non-data, none are price_change/trades, so zero build impact.)
- **Per (universe, month) files READ == files PRESENT for every partition** — politics `pc 259/259, 744/744, 501/501`; esports `pc 259/259, 740/740, 481/481`; trades likewise. **SHORT partitions: NONE — the manifest is complete.** This is the guard the addendum asked for: it fires loudly if a partition processed <100% of its files even while producing plenty of rows.
- Dates per partition **12 / 31 / 21** both universes = the 64-day calendar (June starts 06-19; Aug ends 08-21). Known hole **2026-06-22 15:00 → 06-23 08:29** (17 h, both universes) declared — the one real outage, not thin trading.

## E2 — row reconciliation (true parquet footer counts)
- price_change (archive, pre-dedup): **3,832,521,222** → l1 (touch-moving): **101,051,502 = 2.64%** (Step-3 predicted ~4%; slightly leaner).
  - esports: 2,447,599,174 → 87,215,904 (3.56%), trades 5,632,001
  - politics: 1,384,922,048 → 13,835,598 (1.00%), trades 1,595,527  (politics touch moves less often relative to deep-book churn)
- trades total: **7,227,528**.

**DAEMON-COUNTER CORRECTION (finding — recorded here because `LOG.md` Step A states the wrong figures).** Step A recorded "133.6M price_change / 445k trades" as archive **grand totals**. That is wrong twice over: (1) the daemon's `capture_end.total_counts` **reset every session** — that pair is only the **last** session (08-05→08-26); the 8 sessions sum to ~1.57 B / 5,757,382. (2) The daemon counts price_change **messages**, while the parquet stores one row per fanned-out **entry**. The reconciliation basis is therefore the true archive footer count **3,832,521,222 price_change rows / 7,227,528 trades**, not the daemon figure. Anyone reading the Step A LOG entry should treat its 133.6M/445k as last-session message counts, not archive totals.

## E2 — asset coverage & the empty tree
- **l1 assets not in `tokens`: 0 · trades assets not in `tokens`: 0** — no token traded that we cannot name (a hard-stop condition; passed).
- Empty branches (a report, not a stop): esports **40 tokens fully empty** (no l1, no trades — the 40 without an obs_stats row), **3,886 esports with zero trades** (l1 present), politics **226 with zero trades**, politics **0** with zero l1.

## E2 — the near-0.5 esports split (via `hours_from_last_seen_to_close`)
Of **21,082** settled-near-0.5 esports tokens (20,960 have a `closed_time`): **20,764 had their last observation within ~1 h of settlement** (median ≈ 0 h; slightly negative — last quotes land at/just after Gamma's `closedTime`). So these are **books watched to the end that never moved on an already-decided outcome** — a genuine stale-book-on-a-resolved-outcome phenomenon, market-making relevant — **not** "we stopped watching" (only 196 fall in that bucket). This sharpens the Step-D read: "near 0.5" is overwhelmingly *live-but-unmoved*, not *unobserved*.

## E2 — six-date spot check (built l1 vs bba, nearest within 50 ms)
| date | assets | n | agree |
|---|---|---|---|
| 2026-06-19 (legacy-format day) | 10 | 153,834 | 99.5% |
| 2026-06-23 (post-OOM restart) | 10 | 152,442 | 99.6% |
| 2026-07-01 | 10 | 155,141 | 98.5% |
| 2026-07-15 | 10 | 320,603 | 99.4% |
| 2026-08-01 | 10 | 246,780 | 99.1% |
| 2026-08-20 | 10 | 665,449 | 99.3% |

**Dates below 98%: none** — including the two dates flagged for suspicion. The built l1 matches the independent bba L1 across the whole window.

## Three short answers
- **Why this way, not the obvious alternative?** l1 is built by **per-hour dedup over explicit URIs from an rclone manifest**, not a month-level glob-and-sort. The obvious alternative (one big window-sort per month) took **17 h on politics-August and thrashed** on this machine's volatile free RAM (seen at 0.5 GB); per-hour sorts fit in memory with zero spill and make a sleep/reboot cost one hour, not a month. Four build refinements, each forced by a real failure: silent-timeout-skip → retries+loud-fail+manifest; 17 h month-sort → per-day; RAM wedge → 1 GB cap + SSD spill; external-sort thrash → per-hour.
- **What breaks first if this is wrong, loud or silent?** A **short manifest** (fewer files than the archive holds) would silently drop l1 rows with nothing raising — the same class as the earlier swallowed timeout, one layer up. The addendum's **per-partition files-read == files-present** check makes that failure loud; it passed with exact matches, which is what lets every row count above be trusted.
- **What would a new person misunderstand?** That **101 M l1 rows from 3.83 B raw** signals a bug. It doesn't: 3.83 B is fanned-out price_change **entries** (one row per level change), and 2.64% touch-moving is expected. And that the daemon's "133.6 M total" is authoritative — it is a per-session **message** count, not an archive row total.

## Decision / stop
All gates passed: E1's 8 assertions, the manifest↔inventory completeness (no short partitions), asset coverage (0 unnamed), and the six-date spot check (none <98%). **Stopped before `book`** per NEXT.md — `book` is Step F (needs state carried across hour boundaries). The milestone is reached: with `tokens` + `l1` + `trades`, a market can be opened and looked at. Operator reviews before Step F / publish.
