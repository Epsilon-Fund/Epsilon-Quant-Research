# Data layer — running log

Append only. One entry per step. Oldest first. This is the oversight record: it must be readable months from now by someone who was not here.

Objective: a clean navigable dataset — one `tokens` table (universe → event → market → YES/NO) plus four time-series tables, all keyed identically. See `brain/DATA_LAYER_BRIEF.md`.

---

## Step 1 — Inventory the archive · 2026-08-26 · ~30 min

DID       Listed the whole R2 bucket (metadata only, no download) and analysed the manifests.
CHECKED   11,984 files, 71.08 GiB, 64 dates 2026-06-19 → 08-21, two universes.
          Coverage 3,008 of 3,048 possible book-hours = 98.7%.
          Size split: price_change 60.05 GiB (84.5%), book 7.87, bba 2.45, trades 0.71.
LOOKED    Coverage calendar, day by day, both universes.
FOUND     One real hole: 2026-06-22 15:00 → 06-23 08:00, 17 hours, both universes.
          Two legacy whole-day files on 2026-06-19 esports. A stray `_processed.txt` in the bucket.
          The old `research-live-clob/` prefix is 5.88 GiB of 55 earlier capture campaigns — separate dataset, out of scope.
WROTE     _reports/ manifests
NEXT      Step 2 — look at real rows

## Step 2 — Look at real rows · 2026-08-26 · ~1 h

DID       Downloaded one hour of politics, opened all four tables, checked them against what the engine expects, walked one market event by event.
CHECKED   Schemas match `replay_parquet.py` with known extras. 202 distinct assets in price_change, 198 book, 138 bba, 97 trades.
FOUND     Book JSON stores levels extreme→touch, so **the best level is last**. Prices and sizes are strings. Depth ranges 1–140 levels.
          `price_change` carries best_bid/best_ask, contradicting the comment in events.py.
          56% of price_change rows share a timestamp with the previous row — `received_ns` is the real tiebreak. Within an asset, ordering on received_ns is clean.
WROTE     none
NEXT      Step 3 — catalog prototype

## Step 3 (prototype) — Catalog + design questions · 2026-08-26 · ~2 h

DID       Prototyped the per-(asset,date) aggregation on a 3-hour, 2-universe sample; answered the questions that decide the build; produced the first visual report.
CHECKED   686 asset-days. Ran `capture_gate` over ~10M events: politics PASS, esports FAIL on staleness (not accuracy).
LOOKED    _reports/step3/report.html — market table, spread-vs-depth scatter, three detail panels, a YES/NO mirror pair, distributions.
FOUND     **Only ~4% of price_change rows move the touch** — a deduped full-resolution L1 tape for all 64 days is 1.68 GB, not 36.7.
          Zero-bba markets carry zero trades — the bba gap costs quiet markets, never a trade.
          **Book snapshots are a ~15-min anchor, median 13 min into each hour** — a per-hour builder without carried state gets a silent 13-minute hole every hour.
          `capture_gate` costs ~110 µs/event → ~100 h over the full window. Demoted from build step to on-demand check.
WROTE     none in repo
NEXT      Step A — rescue the VPS metadata

## Step A — Rescue the VPS metadata · 2026-08-26 · ~40 min

DID       Copied `live_universe.json` and `capture_gaps.jsonl` off the VPS before decommission, plus `logs/` as free insurance. Analysed both.
CHECKED   Byte sizes verified against the remote. 121,962 log lines spanning 2026-06-19 → 08-26.
          Downtime: 26 min of in-session reconnect blips + 19.0 h of process-down time, of which **18.43 h is the single 06-22→23 OOM crash**. ≈98.8% uptime; excluding that hole, 99.97%.
LOOKED    _reports/stepA/downtime_by_day.png
FOUND     `live_universe.json` is a **single snapshot of 140 markets / 280 assets**, not a 64-day registry — and carries **no outcome labels**, so it does not solve the YES/NO problem. It becomes an independent cross-check (verification check 5) rather than a shortcut.
          The daemon's own grand totals (133.6M price_change, 8.1M bba, 1.07M book, 445k trades) give us an end-to-end reconciliation check on the eventual build.
          **796 `market_resolved` events were captured but never mapped to a Parquet table** — a fifth event type, silently discarded, unrecoverable for all but the last five days. `resolved_outcome` therefore has no source other than Gamma.
          The `downtime_seconds` field alone would have reported 99.97% uptime and hidden the one real outage — it only logs reconnects while the process is alive.
WROTE     _reports/stepA/{live_universe.json, capture_gaps.jsonl, logs/, downtime_by_day.png}
NEXT      Step B — scope the Gamma fetch

## Step B — Scope the Gamma fetch · 2026-08-27 · ~35 min

DID       Sized the target with a metadata-only R2 scan (DuckDB httpfs, book `market` column only,
          no bulk download); probed the real Gamma /markets API with ~10 archive condition ids
          (politics + esports, incl. resolved). Verified the intended four-program procedure
          against the API rather than designing a new one.
CHECKED   15,386 distinct condition_id across 64 days (esports 13,959, politics 1,427);
          **every condition has exactly 2 asset_id, zero anomalies** → ~30,772 tokens.
          Round-trip: Gamma clobTokenIds set-equal to archive asset_ids for both probed conditions.
          Rate: ~8 req/s sustained, no 429; batch cap 100 ids/call, default page size 20.
LOOKED    Two full raw responses saved for eyeball: _reports/stepB/raw_market_{politics_open,esports_resolved}.json
FOUND     **Target is ~10× the brief's "~3,000 tokens"** — esports per-match sub-markets dominate.
          **Default /markets query excludes closed markets** — resolved conditions return [] unless
          &closed=true; following NEXT.md C2 literally ("empty → .notfound") would silently drop ~14k
          resolved esports. outcomes/clobTokenIds/outcomePrices are JSON-encoded strings.
          esports `outcomes` are TEAM NAMES, not Yes/No (≈91% of rows) — YES/NO orientation is
          politics-only. Event grouping is nested in the market response (no separate fetch). UA header required.
WROTE     _reports/stepB/{condition_target_list.csv (15,386 rows), fullscan_out.txt,
          raw_market_politics_open.json, raw_market_esports_resolved.json}; reports/stepB.md;
          STATUS.md; .claude/settings.local.json (defaultMode auto + deny rclone sync/delete/purge/move, sudo)
FOUND-ACTION  STOPPED before Step C per PROTOCOL.md: number materially off from the brief, and the
          C2 fetch step would mislabel the majority as it is written. Not an obstacle — the sizing IS
          the deliverable; the procedure fix is named and small.
NEXT      Cowork to confirm the 10× scale and approve the corrected two-pass fetch, then run C→D.

## Step C — Fetch Gamma, build tokens_raw.parquet · 2026-08-27 · ~10 min

DID       Four programs, file boundaries between them: target list → two-pass fetch (open + closed=true,
          batch 100, repeated param, limit=batch, UA) → cache-only parse. All output in gitignored
          polymarket/research/data/gamma/.
CHECKED   targets = 30,772 tokens / 15,386 conditions (all exactly 2 assets). Fetch reconcile EXACT:
          15,351 cached + 35 notfound = 15,386, in 164 s; ignored-param assertion never fired.
          Parse: tokens_raw = 30,772 rows (30,702 resolved, 70 unresolved, 0 quarantined).
LOOKED    (verification figure is in Step D)
FOUND     35 not-found are ALL esports (0.23%, under the 5% contingency) — kept as identity_status
          'unresolved', not dropped. ~991 esports markets are literally Yes/No props (not team-vs-team).
          outcomes/clobTokenIds arrive as JSON-encoded strings; every id kept as text.
WROTE     data/gamma/{targets.parquet, targets_rollup.parquet, gamma_cache/ (15,351 json + 35 notfound),
          fetch_log.csv, fetch_batchlog.csv, tokens_raw.parquet}; data_layer/{c1_targets,c2_fetch,c3_parse}.py;
          reports/stepC.md
NEXT      Step D — verify the mapping (chained)

## Step D — Verify the mapping (THE GATE) · 2026-08-27 · ~12 min

DID       Separate verifier program. Assumed the mapping wrong; ran five checks against our own bba
          prices (one 331 s httpfs scan → price_stats.parquet) plus the live_universe snapshot.
          Recorded every check per token in tokens_raw.
CHECKED   check1 round-trip ids 100% (politics 1427/1427, esports 13924/13924); check2 pairing 100% of
          price-judgeable (politics 1390, esports 13781); check5 independent 58/58 agree. check3 NegRisk
          77/170 events; check4 resolution politics 628/916, esports 2306/13201.
          Split resolution failures: INVERSIONS 7/14,117 = 0.05% (all pass checks 1+2 → upsets, not flips);
          winner median>loser median 93.2% politics / 80.2% esports; 81.8% esports resolved near 0.5 (void).
LOOKED    _reports/stepD/resolution_orientation.png (winners→100¢, losers→0¢); ten_markets.txt; verify_numbers.txt
FOUND     GATE PASSES — mapping is correct. checks 3/4 look weak only for benign reasons (voided/thin
          esports sub-markets, upsets, partially-captured NegRisk events, tight tolerances), NOT flipped
          orientation. Coverage 15,351/15,386 = 99.77%; 35 misses all esports. live_universe check judged
          only 58/140 because 82 of its markets were created after the archive ended (08-26 snapshot).
WROTE     data/gamma/{price_stats.parquet, tokens_raw.parquet (augmented with 5 check cols)};
          data_layer/d_verify.py; _reports/stepD/{resolution_orientation.png, ten_markets.txt, verify_numbers.txt};
          reports/stepD.md
NEXT      STOP at the gate (per NEXT.md). Awaiting operator review before any data-table build (Step E: l1 + trades).

## Step E0 — Which source is authoritative for the L1 touch? · 2026-08-27 · ~20 min

DID       Bounded sample (20 assets/universe, liquid+thin, 2026-08-20 h10-15), read via httpfs.
          Three-way agreement price_change vs bba vs reconstructed book touch; then a timing
          diagnostic re-aligning pc onto bba with slack. Void cross-tab from tokens_raw.
CHECKED   politics: pc~bba 100%, book~bba 100%, pc~book 99.8% (all agree).
          esports: pc~book 99.4%, but pc~bba 60.1% and book~bba 61.8% at EXACT timestamps.
          Timing diagnostic: esports pc~bba 59.5% at 0ms -> 98.4% at 50ms slack (flat to 5s).
LOOKED    _reports/stepE/e0_source_check.txt
FOUND     The esports bba "disagreement" is CLOCK SKEW, not bad data: bba is a differently-timed
          emission of the same L1. price_change is reliable in both universes (matches the
          independent book reconstruction 99.4%). The plan's premise "bba is the ground-truth
          tiebreaker" does not hold for fast esports books. Void cross-tab: of 11,189 esports
          near-0.5 conditions, 391 truly voided, 10,798 settled 1/0 but never traded to extreme.
WROTE     data_layer/{e0_source_check.py, e0b_timing.py}; _reports/stepE/e0_source_check.txt;
          reports/stepE.md
FOUND-ACTION  STOPPED FOR DECISION at E0 (per E0's rule + PROTOCOL). Recommend building l1 from
          deduped price_change for both universes. Did NOT build tokens.parquet / l1 / trades.
NEXT      Cowork confirms L1 source -> E1 (tokens.parquet) -> E2 (l1 + trades), stop before book.

## Step E1+E2 — build the library (tokens + l1 + trades) · 2026-08-28 · ~build spanned ~2 days wall-clock (machine sleep/reboots), ~5-6h compute
DID       Built tokens.parquet (E1) + l1 and trades tables (E2) from deduped price_change (E0
          decision, Cowork-confirmed). Four programs: c1/c2/c3 (Steps C-D) then e1_tokens.py,
          e2_build.py, e2_verify.py. Applied Cowork's addendum (manifest<->inventory completeness).
CHECKED   E1: all 8 assertions pass (30,772 rows; asset_id unique; condition_id x2; complement
          symmetric all pairs; path unique; no nulls; unresolved=70; esports 27,918 + politics 2,854).
          E2 manifest<->inventory: pc 2,984 files/60.05 GiB + trades 2,982/0.71 GiB = EXACT match to
          Step-1; every partition files-read==files-present; NO short partitions. Row recon: 3.83B raw
          pc -> 101,051,502 l1 (2.64%), trades 7,227,528. Asset coverage: 0 l1/trades assets missing
          from tokens. Six-date spot check (built l1 vs bba @50ms): all >=98.5%, incl 06-19 (99.5%)
          & 06-23 (99.6%); none <98%.
LOOKED    reports/stepE.md tables; _reports/stepE/{e0_source_check.txt, e2_verify.txt}
FOUND     Library ~1.1 GB (l1 769 MB + trades 338 MB), vs ~1.7 GB plan estimate. esports near-0.5:
          20,764/21,082 watched to settlement but book never moved (stale book on resolved outcome),
          only 196 "stopped watching". 40 esports tokens fully empty; 4,112 tokens with no trades.
CORRECTION  **Step A LOG entry above is wrong**: its "133.6M price_change / 445k trades grand totals"
          are the daemon's LAST-SESSION counters (they reset per session; 8 sessions sum ~1.57B) AND
          count MESSAGES, not fanned-out parquet ENTRIES. True archive basis: 3,832,521,222 pc rows /
          7,227,528 trades. (Step A entry left intact per append-only rule; correction recorded here
          and in reports/stepE.md.)
BUILD-NOTE  4 refinements, each from a real failure on this 16 GB / low-free-RAM, sleeping machine:
          (1) silent transient-timeout partition-skip -> retries + loud-fail + rclone explicit-URI
          manifest (DuckDB S3 glob-LIST was hanging); (2) 17h month-level window-sort -> per-day;
          (3) RAM wedge at 0.5 GB free -> memory_limit 1 GB + SSD spill; (4) external-sort thrash ->
          per-hour dedup (each hourly file sorts in memory, zero spill). Plus a pandas-3.0 Series.view
          break in E1. All resumable; no completed work lost.
WROTE     data/research_v1/{tokens.parquet, l1/, trades/, obs_stats.parquet, exclusions.csv (empty),
          e2_counts.csv, pc_file_manifest.txt}; data_layer/{e1_tokens,e2_build,e2_verify,e0b_timing}.py;
          reports/stepE.md (extended); _reports/stepE/e2_verify.txt
NEXT      STOP before book (Step F) per NEXT.md. Operator/Cowork review of the library, then Step F
          (book, needs state carried across hour boundaries) or Step G (publish v1 light tier to R2).

---

## Step H — loader, manual, notebook, dashboard · 2026-08-29 · ~3h
DID       Built the documented loader (epsilon_data) that a Streamlit research terminal
          demonstrates. H0 units fix; H1 loader + anti-drift test; H2 README + executed
          worked-examples notebook; H3 dashboard (Audit + Explore). Committed code to branch
          alvaro (bd12630); no push; nothing under data/ committed.
CHECKED   H0: 8 E1 assertions pass after rebuild (median_spread now DOLLARS, +median_spread_cents).
          H1: anti-drift test passes (loader == raw parquet: rows, first/last, mid checksum);
          7 tests pass. H2: notebook errors=0, 4 plots, all 5 tasks via public API. H3: AppTest
          smoke landing/audit/explore = 0 exceptions. Reconciliation holds: 101,051,502 l1 /
          7,227,528 trades.
LOOKED    dashboard/app.py (dark, search-first, persistent header); notebooks/epsilon_data_examples.ipynb
FOUND     The "07-24 h12 esports gap" is QUIET (book present, no trading), NOT a capture gap; true
          gaps (no book) = exactly the 3 known events both universes; politics has no unexplained
          gaps; esports 24 quiet-hours/14 days. Dem-nominee-2028 NegRisk YES-sum = 0.982 (not 1.0
          — incomplete candidate capture or small arb); Fed-July binary pair = 1.000 exactly.
          Stale-book cohort = 21,522 esports tokens settled but book never left ~0.5.
CHOICES   Installed streamlit 1.62 + ipykernel + matplotlib-inline into research/.venv (pip via
          ensurepip; uv-managed venv). Added 3 documented loader extensions (coverage,
          reconciliation, activity_by_time) rather than reading raw paths in panels. matplotlib
          charts (no plotly/altair). resolve(market_slug)->outcome_index-0 side. coverage()
          distinguishes quiet vs true-gap via book presence. Did NOT commit data_layer/
          (machine-specific paths) or brain/handoff/ (on disk).
WROTE     polymarket/research/{epsilon_data/, tests/test_loader.py, dashboard/, notebooks/
          epsilon_data_examples.ipynb, scripts/build_examples_notebook.py}; reports/stepH.md
NEXT      Operator runs the dashboard (30s acceptance test), iterates UI. Only then Step F (book)
          or Step G (publish v1 to R2). Nothing pushed. book NOT started.

---

## Step I — market panel rebuilt for research · 2026-08-30 · ~3h
DID       Three corrections then I1-I4. Rebuilt the Explore side in Plotly; added loader
          markout() + negrisk_sum(); kept audit side. Committed to alvaro (043d64f); no push.
HARDSTOP  Verified trades.side = TAKER (aggressor): BUY at ask, SELL at bid (~200k trades/universe,
          medians +/-1.00, unambiguous). Markout sign from maker perspective (BUY->maker short).
          No ambiguity -> no stop. Documented in README, markout(), and on the panel.
CHECKED   Dashboard AppTest smoke: landing/audit/explore(Market+Markout+NegRisk)/logit+window/
          esports = all 0 exceptions. Anti-drift + loader tests still pass. Reconciliation holds.
CORRECTIONS  (1) NegRisk panel now built (was claimed, absent). (2) stale-cohort: esports near_half
          = 21,082 NOT 21,522 (that was both universes; politics=440); 20,764 within +/-1h of settle.
          (3) NegRisk summed at a common timestamp via load_event, never sum-of-medians.
FOUND     Busiest politics token maker markout +0.24c (benign/uninformed). NegRisk corrected:
          273 politics neg_risk events, 173 with >=2 captured candidates (= Cowork's 173);
          instantaneous YES-sum median 0.995 (vs invalid sum-of-medians). Right tail real+method:
          Elon-tweet-range events sum ~2.35 with TIGHT 1c books (maybe non-exclusive bundled
          markets -> check Gamma) + ffill-of-stale-candidates inflation (that event 4.49 aligned).
          Fed binary events sum 1.00-1.01. Caveat stated on panel + report.
CHOICES   Installed plotly 7.0 (I1). Window control server-side for "stats cover visible window" +
          Plotly range-selector for finer zoom; y auto-fit via window reload. negrisk_sum ffill
          caveat documented not silently fixed (best-bid/liveness is an operator design choice).
WROTE     epsilon_data/{tape.py,catalog.py,__init__.py}; dashboard/app.py; reports/stepI.md;
          stepH.md (corrected); _reports/stepI/{negrisk_sums.parquet, negrisk_analysis.txt}
NEXT      Operator reviews rebuilt panels (market zoom/volume/imbalance, markout, NegRisk tail,
          stale cohort). Then Step F (book) or Step G (publish). Nothing pushed. book NOT started.

---

## Step J — handover: extensibility, audit tool, publish, onboarding · 2026-08-30 · ~4h
DID       Closing step. Made the repo extensible (panel registry), built the audit tool, published
          the library to R2, and made onboarding actually work. Corrected the 3 items Cowork flagged.
CHECKED   Acceptance test 1 (new panel = one file, zero edits): PASS. Acceptance test 2 (both data
          paths differ by only EPSILON_DATA_ROOT): PASS (R2-direct read, no code change). Loader
          tests 10/10 (incl audit-never-writes, write-exclusion roundtrip). Dashboard AppTest smoke
          0 exceptions. Clean-clone test PASS: fresh dir + new venv (Python 3.13.5) + pip install +
          R2 data -> check_setup green (30,772 tokens), dashboard renders.
R2        rclone copy research_v1 -> r2:.../research/v1/ (COPY ONLY). Verified EXACT: local 792 files
          / 1,167,809,048 bytes == R2. _manifest.json published alongside.
FOUND     Testing audit_market on real markets showed it over-flagged (condemned the busiest politics
          market over 3 crossed ticks = 0.01%; 37/40 'worth a look' on normal quiet-market gaps).
          Recalibrated to fractions/materiality: crossed/out-of-range bad only >0.5%, spread=0/100
          informational, continuity flags only unexplained gaps >12h. After: busiest politics 'worth
          a look', inverted 'recommend excluding', busiest esports 'looks fine'.
DOCS      epsilon_data/README.md (quickstart, two data paths, credentials + R2 delete-safety, backtest
          gap + scoped adapter spec, two-book finding, gaps, troubleshooting); CONTRIBUTING.md;
          HANDOVER.md; notebooks/cookbook.ipynb; scripts/check_setup.py.
WROTE     epsilon_data/{config,_internal,catalog,tape,audit,__init__,README}; dashboard/{app.py,
          panels/*, .streamlit}; tests/test_loader.py; scripts/{check_setup,build_cookbook_notebook};
          notebooks/{epsilon_data_examples,cookbook}.ipynb; pyproject.toml; requirements.txt;
          CONTRIBUTING.md; HANDOVER.md; data/research_v1/{coverage.parquet,_manifest.json}; reports/stepJ.md
          Commits on alvaro: 0a74c10 (step J), 4268744 (audit recalibration), +earlier H/I.
OPEN      `git push origin alvaro` BLOCKED by the Claude Code auto-mode classifier (outward push to the
          shared GitHub remote). Committed locally; operator must run `git push origin alvaro` (75 commits
          ahead; data/ gitignored). This is the only unfinished item.
NEXT      Operator: push the branch; review the dashboard/handover; then Step F (book) is Gonzalo's path
          to backtesting. book NOT started; R2 is a source (copy only).

---

## Step K — one-command data fetch (no rclone) + repo front door · 2026-08-31 · ~1 h

DID       Removed rclone from the newcomer's critical path and gave the repo root a way in. Built a
          pure-Python R2 downloader, rewired the docs to it, added a root-README front door, and ran
          the whole thing end to end against a scratch copy.
WROTE     polymarket/research/scripts/fetch_data.py (NEW) — boto3, path-style R2, reuses the loader's
          _r2_creds (no second credential loader), 8 parallel workers, byte-copy (no DuckDB re-encode),
          resumable (skip same-size), READ-ONLY (list+get only, no put/delete/copy), self-verifies
          (count/bytes/tokens-rows/l1+trades), writes _fetch_receipt.json, prints next commands.
EDITED    check_setup.py (names fetch_data.py when tokens.parquet missing); requirements.txt +
          pyproject.toml (boto3, fetch-only, loader stays boto3-free); epsilon_data/README.md +
          HANDOVER.md (fetch_data.py is now primary; rclone demoted to a one-liner, copy-only ⚠️ kept;
          shared 3-env-var Credentials paragraph); root README.md (front-door section, links verified).
CHECKED   K3 run against scratch (data/research_v1 untouched): dry-run lists 792, transfers nothing;
          full fetch 792 files / 1,167,809,048 bytes / 30,772 tokens / 101,051,502 l1 / 7,227,528
          trades, all 4 checks OK, 83s wall-clock (~13-16 MB/s); delete-3-and-rerun fetched exactly 3,
          skipped 789 (4s); EPSILON_DATA_ROOT=scratch -> check_setup green (reconciliation match=True)
          -> dashboard AppTest 0 exceptions. Scratch (1.1 GB) deleted afterwards.
FOUND     Running it caught a real bug: verification counted the script's own _fetch_receipt.json as a
          793rd file (+577 bytes), failing any re-run's count/byte checks. Fixed — verification now
          excludes local-only artifacts; re-verify is idempotent. Also: R2 needs path-style addressing
          (virtual-hosted default builds an unresolvable host and hangs) — Config(addressing_style=path).
          Progress-as-hang: files counter climbs steadily; bytes counter can look stalled while big l1
          parquets finish in parallel — docs now warn (~1-2 min, not a hang).
OPEN      Step J is already on origin/alvaro (4268744; earlier push block resolved). Step K commit
          9810c2b is committed locally and is the only unpushed change (local alvaro 1 ahead of
          origin/alvaro, 16 ahead of origin/main; code only, data/ gitignored). Operator: push it.
NEXT      Operator: push the branch. Newcomers: clone -> pip install -r requirements.txt ->
          python scripts/fetch_data.py -> set EPSILON_DATA_ROOT -> check_setup -> streamlit. Step F
          (book) still not started; R2 remains a source (copy/read only).

---
