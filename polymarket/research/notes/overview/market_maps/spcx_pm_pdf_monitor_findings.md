---
title: "SPCX Live Crowd-Distribution Monitor — Polymarket 16-Strike Survivor Fit + Perp Gap, Polled Every 30–60s (Block S5 build + the −$3.3 EV reconciliation)"
tags: [spacex, spcx, ipo, polymarket, hyperliquid, pchip, pdf, monitor, findings, tooling]
created: 2026-06-10
audience: "Justin + Alvaro on listing day (the PEAK trigger + PM tail-sell screens); Cowork/Claude Code sessions extending the Friday stack"
status: "feature-complete + scope-frozen 2026-06-11 (S5 monitor → S5b IEX spot feed → S5c static dashboard → S5d playbook panel → S5e interactive localhost dashboard + design pass; 57 monitor tests, 84 with the calc suite, all green; 39-min live soak PASS); Alpaca keys installed + auth verified incl. SPCX subscription; human steps left: 1-min AAPL feed check during US hours, launch the stack Thursday evening"
---

# SPCX Live Crowd-Distribution Monitor — Polymarket Survivor Fit + Perp Gap (Block S5)

> Hub: [[spacex_ipo_market_map_handoff]] · [[COWORK]] · [[POLYMARKET_BRAIN]]
> Companions: [[spcx_listing_day_gameplan]] (§5.3 screens, §7 Block S5 spec) · [[spacex_ipo_coworker_addendum]] (§ Coworker PCHIP Distribution, § Ambiguity To Reconcile) · [[spacex_pdf_construction_audit]] (why PCHIP shape stats need a health warning) · [[spcx_convergence_calc_findings]] (the basis math this feeds)
> Code: `scripts/spcx_pm_pdf_monitor.py` · tests: `tests/test_spcx_pm_pdf_monitor.py`

## Plain-English Summary

- **What this is.** Block S5 of the listing-day gameplan: a read-only terminal monitor that polls Polymarket's two SpaceX closing-cap markets (the 16-strike "cap above $X" ladder and the 7-bucket "cap between" market) plus the Hyperliquid `xyz:SPCX` perp every 30–60 seconds, refits the crowd's implied closing-price distribution from **executable bid/ask quotes** (never last-trade), and prints the stats that drive day-1 decisions: P(close>$135), mean/median/percentiles in market-cap *and* $/share, the bucket-vs-ladder mispricing table, and the perp-vs-crowd gap.
- **Why it exists.** On listing day the PEAK call (gameplan §5.3) and Alvaro's Polymarket tail-selling key off how the >$2.4T/>$3T tails reprice *while the stock moves* — and nobody can refit a survivor curve by hand mid-session. This is the one dashboard the gameplan deemed worth building (the convergence web dashboard stays declined as infra-before-signal).
- **Live read at first poll (2026-06-10 ~18:00 UTC).** P(close>$135) = **82.2%**, crowd mean **$167.0** / median **$162.8** (13.076B base). Two changes vs the 06-07 coworker snapshot: the famous **$1.5–2.0T bucket overpricing has collapsed (+7.8pp → +1.5pp)** — that diagnostic no longer shows an edge — and the **perp ($163.44) now sits *below* the PM-implied mean (−$3.6)**, vs +$6.2 above on 06-07. The perp-vs-crowd disagreement has closed from both sides.
- **The −$3.3/share mystery is resolved (verdict: bug, do not import).** The coworker PNG's "IPO at 135: EV $−3.3/share" cannot be produced by any defensible EV convention on his own distribution (his own mean − entry = **+$31.9**). A numerical sweep of 11 candidate conventions finds the only near-match is a **"total capital loss if close < $135"** payoff (pwin·(median−135) − plose·135 = **−$3.78**) — an accounting that pretends a $113 close loses all $135/share instead of $22. That is not a hedged-entry convention; it is a wrong payoff floor. **Pre-registered: the monitor prints EV = mean − offer; the PNG EV line is dead.**
- **Status.** Shipped: script + 17 unit/acceptance tests (all green; 44 with the calc suite), live-verified against the real CLOB and HL endpoints, `--html` static page and append-only parquet poll log working. Ready for Thursday-evening-onward duty.

---

## 1. Design — what it computes and from what

**Data, per poll.**

| input | source | what is taken |
|---|---|---|
| 16-strike ladder ("SpaceX IPO closing market cap above $X?") | Polymarket Gamma event `spacex-ipo-closing-market-cap-above` → CLOB `/books` | best **bid and ask** of each YES token (executable quotes; last-trade is never used) |
| 7-bucket market + No-IPO leg ("cap between $X and $Y?") | Gamma event `spacex-ipo-closing-market-cap` → CLOB `/books` | same |
| `xyz:SPCX` perp mark/mid | Hyperliquid info endpoint (`metaAndAssetCtxs`, dex `xyz`) | mark + mid |
| listed price (post-cross) | manual `--spot` flag | compared against the crowd distribution |

**Construction (the survivor fit).** Each ladder YES price is read as the survivor probability S(K) = P(close cap > K) at its strike, on a chosen basis (`--basis mid` default; `ask`/`bid` selectable). The point set is **clipped to non-increasing** (a strike quoted above its left neighbor is an arbitrage-inconsistent quote — it gets clipped and **flagged in the output**, satisfying the monotonicity-enforcement requirement). A **monotone PCHIP (Fritsch–Carlson)** interpolant is fitted through (0, 1.0) + the clipped points + the coworker's tail anchors (4.5T, 0.005), (5.0T, 0.001); the PDF is −dS/dK and the expected cap is the survivor integral **E[cap] = ∫S(K)dK** (cross-checked against ∫K·pdf dK internally).

**Why PCHIP when the audit said its shape lies.** [[spacex_pdf_construction_audit]] showed differentiating a PCHIP through kinky survivor points manufactures spurious PDF humps — but also that the **central stats (P(win), mean, median, P25–P95) are construction-invariant**. The monitor keeps PCHIP so its numbers are directly comparable with Alvaro's pipeline, and prints **mode flagged as shape-fragile** with a pointer to the audit. Decisions should ride on the central stats, never on mode/extreme-tail shape.

**Share bases.** Every per-share figure is printed on the vault-standard **13,075,865,175** S-1/A base *and* on Alvaro's rounded **13.091B** convention as a comparison column, so numbers reconcile against his screens without mental math (the difference is ~0.1%).

**Bucket-vs-ladder table.** Model-free, per the audit's convention: ladder side = S(lo) − S(hi) from a *linear* interpolation of the clipped quotes (so the PCHIP's smoothing can't inject fake gaps); bucket side = that bucket's quote renormalized over the seven IPO-conditional buckets (No-IPO leg shown separately). Gap > ±5pp prints a DIVERGENT flag. This is the addendum's $1.5–2.0T diagnostic, made live.

**Graceful degradation (tested).** One-sided books degrade to the available side (flagged); dead books drop the strike (flagged); the fit refuses to run below 4 usable strikes rather than print nonsense.

**Outputs.** Terminal table (default, every poll); `--html PATH` writes **one static auto-refreshing HTML file** (meta-refresh, no server, glanceable on a phone); `--parquet-log` appends **one shard per poll** under `data/analysis/spcx_convergence/pm_pdf_log/` (append-only, long format: one row per instrument with bid/ask, poll-level stats denormalized onto every row — DuckDB-globbable). No Streamlit, no FastAPI, no DB, no `live_trading/` imports.

**Run (from `polymarket/research/`):**

```bash
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py                 # one poll
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 \
    --html data/analysis/spcx_convergence/pm_pdf_monitor.html --parquet-log
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --spot 168.5    # post-cross
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --reconcile     # the −3.3 sweep
```

---

## 2. Practical example — one poll, read end to end

At the 2026-06-10 18:00 UTC poll the >$2T strike quoted 0.61 bid / 0.66 ask, so the mid says the crowd puts **63.5%** on SpaceX closing day 1 above a $2T cap (= $153/share on 13.076B shares). Stacking all 16 strikes the same way and fitting the survivor gives the full curve; reading it at the $135-offer cap (1.766T) gives **P(close>$135) = 82.2%**, and integrating it gives **E[cap] = 2.184T → mean $167.0/share**. The perp marked $163.44 at the same instant, i.e. **$3.6 below** the crowd mean: the leveraged-money venue and the prediction-market crowd now roughly agree (on 06-07 the perp was $6 *richer* than the crowd). On Friday, if the stock prints $180 and the >$2.4T YES is still offered at 30c while the refit survivor says 45%, that's the PEAK-adjacent signal this monitor exists to surface within one poll cycle.

---

## 3. First live poll — full output (2026-06-10 ~18:00 UTC)

| stat | cap $T | $/sh (13.076B) | $/sh (13.091B, Alvaro) |
|---|---:|---:|---:|
| mean | 2.184 | 167.0 | 166.9 |
| median | 2.129 | 162.8 | 162.6 |
| mode* | 2.260 | 172.9 | 172.7 |
| P25 | 1.869 | 142.9 | 142.8 |
| P75 | 2.372 | 181.4 | 181.2 |
| P90 | 2.858 | 218.5 | 218.3 |
| P95 | 3.122 | 238.8 | 238.5 |

P(close > $135) = **82.2%** · EV vs offer = **+$32.0/sh** · perp $163.44 → vs mean **−$3.6**, vs median **+$0.6**. *Mode is shape-fragile under PCHIP ([[spacex_pdf_construction_audit]]) — printed for continuity with Alvaro's screens, not for decisions.*

| bucket | ladder % | market % | gap (pp) |
|---|---:|---:|---:|
| <1T | 1.6 | 0.9 | −0.6 |
| 1–1.5T | 5.3 | 3.7 | −1.6 |
| 1.5–2T | 29.6 | 31.2 | **+1.5** |
| 2–2.5T | 42.5 | 42.8 | +0.3 |
| 2.5–3T | 14.0 | 15.2 | +1.2 |
| 3–3.5T | 3.9 | 4.1 | +0.2 |
| 3.5T+ | 3.1 | 2.1 | −1.0 |

**Column glossary.** "ladder %" = probability the strike ladder implies for that cap range (S(lo) − S(hi), model-free linear interp of the clipped mids). "market %" = the standalone bucket market's own quote, renormalized over the 7 IPO-conditional buckets (No-IPO leg, 0.2c ask, excluded). "gap" = market − ladder, in probability points; positive = the bucket market is richer than the ladder.

**Read.** Two material changes vs the 2026-06-07 coworker snapshot. (1) The **$1.5–2.0T bucket overpricing — the addendum's candidate NO-expression edge — has collapsed from +7.8pp to +1.5pp**; at this gap, after spread and fees, there is nothing to take; the diagnostic stays on the screen because a fast tape on Friday could reopen it. (2) The perp has bled from $173.5 to $163.4 while the PM crowd barely moved (mean $166.9 → $167.0 on the same share base): the **perp premium-to-crowd has gone from +$6.2 to −$3.6** — consistent with the gameplan's "basis bleeding out" read, and a reminder that the PM crowd, not the perp, is currently the more bullish venue.

---

## 4. The −$3.3/share EV reconciliation (Block S5(d) — closes the addendum's open item)

**The puzzle.** The coworker PNG's summary panel prints `IPO at 135 (~1.77T): P(win): 79.9%, EV: $-3.3/share` — while the same panel's distribution has mean $166.9, which makes the obvious EV = 166.9 − 135 = **+$31.9**. [[spacex_ipo_coworker_addendum]] § Ambiguity To Reconcile flagged it; [[spacex_pdf_construction_audit]] housekeeping item 4 routed it here.

**Method.** Rebuild his exact construction (PCHIP through the 2026-06-07 mids, his anchors, his 13.091B share base — reproduction verified within $0.4 on every table stat, see §5), then evaluate **11 candidate EV conventions on that same distribution** and rank by distance to −3.3. Run it yourself: `--reconcile`.

| convention | EV $/sh | (err vs −3.3) |
|---|---:|---:|
| **D2 pwin·(median−offer) − plose·offer (total loss if lose)** | **−3.78** | **0.48** |
| D3 pwin·(mean−offer) − plose·offer | −1.46 | 1.84 |
| C1 mean − perp mark ($173.53, entry at perp) | −6.43 | 3.13 |
| C2 median − perp mark | −9.33 | 6.03 |
| D1 pwin·avg_gain − plose·offer | +8.32 | 11.62 |
| E1 sign-flipped decomposition | −31.91 | 28.61 |
| B2 mode − offer | +26.33 | 29.63 |
| B1 median − offer | +29.20 | 32.50 |
| A2 pwin·avg_gain − plose·avg_loss (correct, decomposed) | +31.91 | 35.21 |
| A1 mean − offer (correct unhedged EV) | +32.10 | 35.40 |
| F1 survivor integral truncated at 1.0T − offer | −43.98 | 40.68 |

**Column glossary.** "avg_gain/avg_loss" = conditional means above/below the $135-equivalent cap (his construction: +$44.3 / −$17.5, matching the audit). "offer" = $135. The D-family replaces the true loss leg (offer − E[close | close<135] ≈ $17.5/sh) with the **entire $135** — i.e. it models a sub-offer close as losing all capital.

**Verdict.** The correct EV under his own distribution is **+$31.9/share** (A1/A2, agreeing with the audit and the v2 builder). No defensible convention — not taker-ask vs mid, not the 13.076B base, not entry-at-perp, not truncation — lands at −3.3. The only near-match is the **total-capital-loss payoff floor (D2, −$3.78)**: treating "close below $135" as −$135/share instead of −$17.5/share. Whether his `ev_report.py` did exactly D2 or a cousin of it (his source wasn't shared), the class of error is identified and it is **not a legitimate hedged-entry convention; it is a wrong payoff function for owning shares**. Disposition, pre-registered: **the PNG EV line is dead — never import it; the monitor prints EV = mean − offer (A1)**. The addendum's open ambiguity is closed.

---

## 5. Acceptance criteria → evidence

| spec requirement | status | evidence |
|---|---|---|
| Bid/ask (not last) from CLOB, ladder + bucket | ✅ | `fetch_books` parses best bid/ask only; live poll verified |
| Monotone PCHIP survivor, monotonicity enforced (clip + flag) | ✅ | `enforce_monotone` + tests `test_enforce_monotone_*`, `test_pchip_survivor_is_monotone_and_pdf_nonnegative` |
| EV integral E[cap]=∫S dK on synthetic surface | ✅ | uniform + lognormal closed-form tests (`test_ev_integral_*`), pdf-moment cross-check |
| Per-share conversion, both share bases | ✅ | `test_per_share_conversion_both_bases`, `test_analyze_emits_both_share_base_columns` |
| Reproduces the addendum's 06-07 table within tolerance | ✅ | `test_reproduces_addendum_2026_06_07_table`: mean/median within $1, percentiles $1.5–2, P(win) ±0.5pp (actual deltas ≤$0.4 / 0.2pp) |
| Bucket-vs-continuous mispricing table | ✅ | `test_fixture_bucket_overlay_matches_audit` reproduces the audit's +7.8pp on 1.5–2.0T; consistent-market test gives ~0 gaps |
| Perp vs PM gap; `--spot` crowd-vs-traded | ✅ | live poll output; `test_render_text_and_html_smoke` covers the spot block |
| Graceful one-sided/empty-strike degradation | ✅ | `test_one_sided_and_empty_strikes_degrade_gracefully`, `test_too_few_strikes_refuses_to_fit` |
| Terminal default, `--html` single static file, `--parquet-log` append-only shards | ✅ | live run wrote both; `test_parquet_log_appends_shard` proves shard-per-poll append-only |
| −$3.3 EV reconciled + documented | ✅ | §4; `test_ev_sweep_correct_ev_positive_and_total_loss_family_nearest_png` |
| No Streamlit/FastAPI/DB/`live_trading/` imports | ✅ | stdlib + numpy + httpx + pyarrow only (all already in the venv); PCHIP hand-implemented (venv has no scipy) |

Tests: **17 new, all green** (`PYTHONPATH=. uv run pytest tests/test_spcx_pm_pdf_monitor.py -q`); the combined SPCX suite (with the convergence calculator) is 44 green.

---

## 6. Listing-day runbook (where this sits in the gameplan)

- **From Thursday evening (D1):** run `--watch 45 --parquet-log --html ...` alongside the convergence calculator's `--watch` ([[spcx_listing_day_gameplan]] §6 D0–D1). Two terminals, two jobs: the calc watches the *basis/liquidation*; this watches the *crowd distribution*.
- **Pre-cross Friday (D3–D4):** the PM mean/median vs perp gap is the cleanest live "who's right" tension on the screen map (§5.3). A perp collapsing toward $135 while PM holds ~$165 = the crowd disagrees with the leveraged money — pre-agree with Alvaro which side gets believed.
- **Post-cross (D5):** with `--spot-ws alpaca` running (§7), the crowd-vs-traded line updates itself from the IEX last trade; manual `--spot <listed>` remains the override/fallback. The "spot sits at the crowd's P-xx" line plus the >$2.4T/>$3T tail repricing is the shared PEAK input; the bucket gap table is Alvaro's NO-expression screen, live.
- **Afterward:** the parquet poll log feeds the `spcx_listing_postmortem` note (gameplan § Post-day) — every poll's full ladder/bucket/perp state is preserved.

## 7. Block S5b — auto spot feed (Alpaca IEX websocket), added 2026-06-10

**What it adds.** `--spot-ws alpaca` replaces the manual `--spot` retyping post-cross: a background thread holds SPCX's last trade from Alpaca's free real-time IEX stream (`wss://stream.data.alpaca.markets/v2/iex`), and each poll cycle renders the crowd-vs-traded block from it, labeled with the trade's age. Precedence and failure rules, all tested with a socket-free fake-frame injector: **manual `--spot` always overrides the feed**; a trade older than `--spot-stale-secs` (default 120) renders a `STALE` line and suppresses the crowd-vs-traded math; a subscription with no prints yet (the normal state before the IPO cross) renders "no prints yet (normal pre-listing)" — in every degraded state the PM/HL polling continues untouched. Disconnects reconnect forever with capped exponential backoff (1s → 60s), and the feed thread can never raise into the monitor.

**The coverage caveat (printed on every live read).** IEX carries **~2% of consolidated US tape volume**. The label says it verbatim: `IEX (≈2% of tape — signal-grade, not queue-grade)`. It is good for "where is SPCX trading right now ±cents" — which is all the crowd-vs-traded block needs — and not good for microstructure (spreads, queue, prints-per-second). The §5.3 screen map is unchanged: TradingView/the S3 sources stay the chart, this is just the number that feeds the monitor.

**Setup (human, before Friday).** Create a free Alpaca account (paper account is enough — market data keys work without funding), then export `ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` in the shell that runs the monitor (or a local `.env`, which is git-ignored and auto-loaded). Keys are read from the environment only — never hardcoded, never committed. Run: `--watch 45 --spot-ws alpaca --parquet-log --html …`.

**Parquet schema extension.** Poll shards gain four columns — `spot_source` (`manual`/`alpaca_iex`), `spot_status` (`live`/`stale`/`no_prints`), `spot_age_s`, `spot_last_price` (kept even when stale, for the postmortem). Pre-S5b shards stay readable; query the mixed log with `read_parquet('pm_pdf_log/*.parquet', union_by_name=true)` — the backward-compat read is unit-tested against a simulated legacy shard.

**Smoke result (2026-06-10, recorded).** No Alpaca keys exist on this machine yet, so the live smoke verified what is verifiable without them: a real connection to the production endpoint received the protocol greeting `{"T":"success","msg":"connected"}` and answered our auth frame with a well-formed `{"T":"error","code":402,"msg":"auth failed"}` — TLS, endpoint, framing, and the auth/subscribe message schema are all confirmed; the missing-creds path exits with a clean one-line error. **The remaining human step:** once keys exist, run a 1-minute check during US market hours with a liquid symbol — `--watch 30 --spot-ws alpaca --spot-symbol AAPL` — and confirm a `LISTED $…` line with a sub-minute trade age. SPCX itself will print nothing until the cross; "no prints yet" is the expected Friday-morning state.

**Tests.** 8 new (manual-overrides-WS, stale → STALE + monitor alive, no-prints rendering, IEX label + age, backoff cap + on-open reset + auth/subscribe frames, junk-frame immunity, env-var requirement, parquet schema backward-compat). Suite: **25 monitor tests, 52 with the calc — all green.** No new dependency: the feed uses `websocket-client`, already in `pyproject.toml` for other capture work (one-line justification: it's the only WS client already vendored, and its callback API fits a daemon-thread last-value cache exactly).

## 8. Block S5c — rich self-contained HTML dashboard, added 2026-06-10

**What it adds.** `--html` no longer writes a bare text page: it now renders a phone-readable, dark-mode, single-file dashboard, regenerated **atomically** each poll (temp file + rename — the browser can never catch a half-written page) with meta-refresh. The page is fully self-contained — inline CSS, charts as inline matplotlib SVGs, zero scripts/links/CDN references — so it renders with the network down (self-containment is unit-tested by scanning for external `src`/`href`/`url()` constructs; the XML namespace identifiers inside SVGs are not fetches). The terminal output is unchanged. Render cost is ~0.5s/poll, well inside the <2s budget.

**Panels, top to bottom, and how to read them** (every chart carries the same caption on the page):

1. **Status strip** — poll time (CET + UTC) and green/amber/red staleness chips for the PM, HL, and spot fetches (≤90s green, ≤300s amber, else red / "never"), plus any active flags: monotonicity clips, one-sided books, S5b spot-WS STALE.
2. **Headline tiles** — P(close>$135), crowd mean, crowd median, perp mark, perp−mean gap, spot (+ crowd-percentile once printing, "no prints yet" before), and **EV vs offer using the A1 convention (mean − offer) — never the §4 D2 bug**. Each tile shows ▲/▼ deltas vs the previous poll and vs session start.
3. **Chart A — survivor curve** — the PCHIP fit with each strike's quote dot and bid–ask whisker, clipped (arbitrage-inconsistent) points in red, overlaid on a dashed **reference curve** (`--reference fixture` = the 06-07 addendum surface; a snapshot JSON path; default = session-start poll). Red drifting off grey = the crowd repricing. Dual x-axis: cap $T and $/share.
4. **Chart B — implied PDF** — per-share density with the P25–P75 band shaded and verticals at offer/perp/spot; the mode marker is explicitly annotated shape-fragile per [[spacex_pdf_construction_audit]].
5. **Chart C — session time series** — crowd mean/median, perp, spot ($/share, left axis) and P(close>offer) (%, right axis), with `--mark "LABEL[@HH:MM]"` event verticals (PRICED, ALLOC, CROSS…). History is held in memory and **backfilled from today's parquet shards on startup** (a restart keeps the session), capped at 12h with a drop note in the footer.
6. **Chart D — the PEAK/tail panel** — bid (thick) and ask (dotted) of the >$2.2T / >$2.4T / >$3.0T YES strikes in cents, deliberately the most legible chart: it is Alvaro's tail-sell screen, read in a hurry. Tails bid up = crowd chasing the rip; tails fading off highs = PEAK confirming.
7. **Bucket-vs-ladder table** — terminal numbers, |gap| > 5pp rows highlighted DIVERGENT.
8. **Alert ledger** — pre-registered levels ($183 pre-hedge trigger, $162 ceiling, final price once `--offer` differs, $140, $135, $125), each crossed **in its trigger direction relative to the offer** (upside levels when price ≥ level, downside when ≤; series = spot once printing, perp before), with first-crossing time. Display only — alerting stays in TradingView per the gameplan.

**Degradation:** every chart renders independently — a failure becomes a labeled placeholder ("unavailable this poll … retries next poll"), never a dead page; with <2 polls of history, Charts C/D show "collecting…". All tested, including a forced chart exception.

**Live visual check (2026-06-10 23:41 CET, recorded).** Rendered from live data to `data/analysis/spcx_convergence/pm_pdf_dashboard.html` and read back: 4 SVGs, 0 placeholders, tiles P(win) 82.7% / mean $167.2 / perp $163.15 / gap −4.1 / EV +32.2, ledger correctly shows $162 and $140 CROSSED (first touch 19:55 CET, perp series, replayed from the backfilled session) with $183/$135/$125 not crossed, reference overlay labeled "2026-06-07 addendum fixture", 17 polls of history.

**Tests.** 11 new (atomic temp+rename with no leftover, self-containment scan, first-poll "collecting", many-polls/no-spot, missing-tail-strike placeholder, ▲/▼ delta correctness on synthetic up/down two-poll sequences, `--mark` in-range rendering, reference-overlay math reproducing the 06-07 P(win) 79.9%, forced-chart-failure degradation, directional alert-ledger first-touch, parquet backfill). Suite: **36 monitor tests, 63 with the calc — all green.** No new dependencies (matplotlib was already a dev dep of this venv).

## 9. Block S5d — the "what now" playbook panel, added 2026-06-10 (scope freeze)

**What it is.** A NOW card at the top of the S5c dashboard that renders the gameplan's decision tree live: which node the day is in (D2/D3 → D4 → D5 → CLOSE-OUT, inferred from the fed state + the CET clock), the node's pre-registered rule with its live evaluation, and a "watch next" line. It is strictly a **checklist renderer** — every rule line carries its vault anchor (e.g. `[gameplan §6 D2 / S1]`, `[S2 §4]`), it invents no thresholds, computes no new signals beyond the spec, and never suggests orders beyond quoting the gameplan's own frozen rules.

**Day state the human feeds it** (persisted to `--playbook-state` JSON — default `data/analysis/spcx_convergence/playbook_state.json` — so restarts resume): `--offer` (final price), `--fill` (shares allocated; unlocks D2), `--hedged --hedge-entry --hedge-lev` (once the short is on), `--cross HH:MM PRICE` (the first print; unlocks D5), `--sold N` (cumulative residual shares sold, updated after each tranche), plus `--sub-eur/--margin-eur/--eurusd` for the hedge math (EURUSD live-fetched if unset, 1.08 labeled fallback). Missing inputs render as "awaiting --fill / --cross", never as wrong defaults (tested).

**Per node:**

- **D2/D3 (allocation, pre-bell):** the S1 rule evaluated live — hedge = min(fill, margin-cap) at 1× and 1.5× iff net basis (perp − offer − fees + funding) > 0, with the arithmetic **delegated to `spcx_convergence_calc.build_hedge_grid`** (no third implementation; equality with an independent calc call is asserted bit-for-bit in tests). Shows suggested hedge shares at both leverages, € margin needed, gross/net basis, GREEN "hedge sleeve ON" / RED "NO hedge", the §3 fill-row that materialized and the residual sleeve size. D3 watch items: the perp ≤ $140 risk-off flag, the calc's liquidation-buffer math if hedged, and the bell countdown.
- **D4 (display-only):** proxy-stack reminder (perp + newswire; no EU NOII access per S3), "no selling into the cross", the perp-vs-PM-mean tension line, and the "log the cross" prompt.
- **D5 (post-cross):** the tranche clock (minutes since cross → OBSERVE 0–15 / T1 15–60 / T2 60–180 / T3 180+, with target cumulative-sold % vs `--sold` actual, per the S2 calibration); the hard-stop ladder ($140/$135/$125) with %-distance from live spot flipping red when crossed (display only); **anchored VWAP from the S5b IEX stream** (anchored at the first received print — lookahead-free by construction — labeled `IEX ≈2% of tape — confirm on TradingView before acting`, with "lost N min ago, not reclaimed → front-load" state per gameplan §5.3 #1 + S2 §6, and "AVWAP: use TradingView" whenever the feed is absent/stale); the hedge-sleeve **pair-close chip** (green only when |perp−spot| ≤ $2 has held ≥15 min AND ≥60 min since cross, per §5.1, plus the TR-limit-first leg order); and the **PEAK coordination box** (definition restated; live state for AVWAP and the PM tails vs their session highs; volume-divergence and lower-low left as manual TradingView checkboxes — IEX volume is too thin to compute them honestly).
- **CLOSE-OUT (after 21:30 CET):** "don't carry residual overnight (S2: 5/6 day-2 closes lower)" with the remaining-shares count, and the vntl hand-timed close-leg reminder.

**Dry-run transcript (scripted day, from the test fixture — exactly what the card renders, tags stripped):**

```text
--- 08:30 D2 allocation (fill 40 sh, perp $158) ---
NOW: D2/D3
 - rule: hedge = min(fill, margin-cap) iff net basis > 0 → hedge sleeve ON: short ~21 sh
   at 1.5× (~14 at 1×) [gameplan §6 D2 / S1]
 - live: gross basis $+23.00/sh, net $+22.86/sh after fees+funding; margin needed €2000
   at 1.5× / €2000 at 1× (EURUSD 1.080)   ← cap binds: full margin at either leverage
 - fill 40/80 sh requested ≈ 50% → §3 row: 50% · residual sleeve 19 sh at 1.5× [gameplan §3]
 watch next: perp level vs offer (risk-off line $140); bell in 420 min.

--- 19:15 D5 T2 (cross 18:00 @ $165; sold 8/19; spot $160) ---
 - tranche clock: 75 min since cross → phase T2 · sold 8/19 sh (42%) vs target 80%
 - anchored VWAP $164.80 (IEX ≈2% of tape — confirm on TradingView): spot BELOW — lost
   20 min ago, not reclaimed → front-load remaining tranches [gameplan §5.3 #1 / S2 §6]
 - hard stops vs spot: $140 reassess: +12.5% · $135 offer: +15.6% · $125: +21.9%
 - hedge sleeve pair-close: tracking confirmed — pair-close window OPEN · leg order:
   TR limit sell first, then close the perp [gameplan §5.1]
 - PEAK = volume divergence + any of: AVWAP loss / lower low / perp divergence / PM-tail
   fade. Live: AVWAP, PM tails; ☐ volume divergence ☐ lower low manual on TradingView.

--- 21:45 CLOSE-OUT ---
 - don't carry residual overnight — 5/6 mega-IPO day-2 closes were lower; remaining 0 sh
```

The full six-node transcript (07:00 PRE-ALLOC through CLOSE-OUT) regenerates from the test fixtures; the D2 numbers above are byte-identical to the calc's grid output by construction.

**Tests.** 15 new: node inference across the scripted day (including the 15:29/15:30 and 21:29/21:30 boundaries), missing-inputs-render-awaiting, D2 hedge arithmetic equal to an independent `build_hedge_grid` call on all four cell fields at both leverages, GREEN/RED basis paths + the $140 risk-off flag, tranche-phase boundary minutes (14.99/15/59.99/60/179.99/180), sold-vs-target rendering, stop-ladder crossing colors and distances, AVWAP stream-replay anchoring (first print only, later trades never move the anchor), AVWAP lost/reclaim/stale states, pair-close chip requiring both conditions (each violated separately), pair-close only rendering when hedged, CLOSE-OUT residual math, JSON state persistence round-trip, `--cross` parsing (HH:MM and "now"), and panel placement above the tiles. Suite: **51 monitor tests, 78 with the calc — all green.**

**Scope freeze.** Per the block spec, S5 is feature-complete for Friday: S5 (monitor) + S5b (spot feed) + S5c (dashboard) + S5d (playbook panel). Remaining human steps are unchanged: Alpaca keys + AAPL feed check (§7), start the watch Thursday evening with the full stack:

```bash
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 --parquet-log \
    --spot-ws alpaca --html data/analysis/spcx_convergence/pm_pdf_dashboard.html
# Friday morning, after the TR notification:  add  --offer <final> --fill <shares>
# at the print:                               add  --cross HH:MM PRICE
# after each tranche:                         add  --sold <cumulative>
```

## 10. Block S5e — interactive localhost dashboard, added 2026-06-11 (supersedes the S5c page as the primary screen; final scope freeze)

**What changed.** The operator lifted the static-HTML constraint, so the monitor now serves a **live localhost web dashboard**: `--serve` starts an HTTP+websocket server on `127.0.0.1:8642`, every poll is pushed to the browser instantly (no meta-refresh), charts are interactive, and the S5d day-state is settable **from the page itself** as well as the CLI. The S5c static page and the terminal output are unchanged and keep working alongside — they are the fallback, not legacy.

**Launch (the Friday command):**

```bash
PYTHONPATH=. uv run python scripts/spcx_pm_pdf_monitor.py --watch 45 --parquet-log \
    --spot-ws alpaca --serve \
    --html data/analysis/spcx_convergence/pm_pdf_dashboard.html   # static fallback page
# prints: [serve] interactive dashboard: http://127.0.0.1:8642/
```

**Design rationale (free-reign choices, and why):**

- **Server: aiohttp** — one small dependency covering HTTP + websockets in a single background-thread event loop; no framework, no build step. Added via `uv add aiohttp`.
- **Charts: ECharts 5.5.1, vendored** at `scripts/assets/echarts.min.js` (1.03MB, Apache-2.0) — a single self-contained minified file with every needed chart type; the page's only `<script src>` is that local file, so **Friday renders with the internet down** (self-containment unit-tested).
- **Layout for the stressed glance** (post-audit design pass, 2026-06-11): a sticky **masthead** with a color-coded **node chip** (PRE-ALLOC grey → D2/D3 sky → D4 amber → D5 emerald → CLOSE-OUT red — readable from across the room), a live CET clock, and a pulsing connection dot; red DISCONNECTED banner with 2s auto-reconnect + full snapshot replay. Below: staleness chips that **tick every second**, the S5d NOW playbook card (server-rendered, reused verbatim, left-border colored by node), headline tiles **ordered by day-of glance priority** (spot → perp → perp−crowd gap → P(win) → mean → median → EV) with ▲/▼ deltas, a horizontal **pre-registered levels strip** ($183/$162/$140/$135/$125 — dim until crossed, then sky for upside / red for downside with first-cross time), then the **numbered panels** for mid-session comms ("check panel 03"): 01 tail panel (tallest — the PEAK screen), 02 session price-vs-crowd, 03 crowd curve (survivor + PDF), 04 bucket consistency, 05 day-state drawer + one-click PRICED/ALLOC/CROSS marks.
- **Visual discipline:** strict semantic palette — **red is reserved for stops/crossed levels/disconnect** (the crowd series is white, perp amber, spot sky, P(win) emerald, the >$3T tail violet); every number renders in **monospace tabular figures** so columns never wobble between polls; section labels are uppercase microtype, captions one line each with their vault anchor.
- **PLAN tab (operator request, 2026-06-11):** a second tab carries the frozen gameplan rendered for mid-session reading — the two-sleeve structure, the full **D1→CLOSE-OUT decision tree as a node / window / key-move / key-watch table**, the PEAK signal definition, the §5.4 hard risk rules, and a **glossary** defining every term and document shorthand on the page (gameplan/S1–S5, basis, survivor, ladder vs bucket, EV A1, AVWAP, pair gap, each alert level, the node chip). Playbook source anchors were expanded from terse "[gameplan §6 D2]" style to self-describing references; a **perp − spot tile** (the §5.1 pair-close tracking gap, labeled against its ≤$2 band) and a perp-implied-cap line on the survivor chart were added; base typography and form inputs were enlarged for at-distance reading. **Gameplan re-encode (2026-06-11, after the gameplan's §0/S6 update):** the D2 card now implements the **overflow-valve rule** — hedge = clamp(fill − the ~22-share comfort zone, 0, margin cap) iff net basis > 0, with the expected case (fill ≤ comfort) rendering "NO hedge — margin stays free" and a front-load warning when the residual still exceeds comfort after the cap; arithmetic still delegates to `build_hedge_grid` (the overflow is fed as the fill — equality test updated). D5 gained the **21:00 CET forced-flat backstop** line (countdown + the ≈$1.7/sh Cerebras backstop cost), D4 the no-cross-by-20:30 contingency, CLOSE-OUT the "perp 0, shares 0" end-state check, and the pair-close line cites the S6 calibration (gap ≤ $2 at +46 min). The pre-hedge trigger reads ~$183 everywhere, `--comfort` joined the CLI/state file, and the decision-tree PNG + PLAN tab were regenerated to match. A second pass added a dedicated **panel 02 pair chart** (perp + spot lines with the gap area and dashed ±$2 band on a right axis; the perp leg renders alone pre-cross, spot joins at the first print), fixed chart-label clipping via `containLabel`, and condensed the PLAN tab into a **color-coded D1→CLOSE-OUT flow diagram** with a red any-time "$125 → sell everything" strip, every prose block collapsed behind expanders, and the glossary collapsed by default.
- **All numbers are computed server-side** by the same engine and calc functions; the browser only draws. A UI state change POSTs to `/api/state`, persists to the same JSON file the CLI uses, and triggers an **immediate** playbook re-broadcast — no waiting for the next poll.

**Fallback ladder (memorize for Friday):**
1. **Browser tab dies / laptop sleeps** → reload the URL; the socket reconnects and the server replays a full session snapshot (history, marks, state).
2. **Server layer fails** → the same process's **terminal output** and the **S5c static page** (`--html`, atomic writes) continue untouched; a push failure is caught and logged, never stops the engine.
3. **Process dies** → restart the command; session history backfills from the last `--backfill-days` of parquet shards (default today), day-state reloads from `playbook_state.json`. Overnight holes are filled separately by `spcx_backfill_history.py` (§10.2) before relaunch.
4. **Everything dies** → the parquet log is append-only on disk; HL/PM/TradingView remain the primary external screens per gameplan §5.3 rule zero.

**Security posture:** binds 127.0.0.1 only, no auth — acceptable solely because it never leaves localhost; it is also read-only/advisory by construction (no venue auth exists anywhere in the monitor).

**Scripted-day dry run through the live server API (operator-reviewable transcript, abridged — regenerates from the test fixtures):**

```text
BROWSER CONNECTS → snapshot (history 1, state fill=None)
  NOW: PRE-ALLOC — awaiting --fill (D2 unlocks at allocation)
08:00 ALLOCATION — operator types fill=40 in the page form → POST /api/state
  → instant playbook push: NOW: D2/D3 — hedge sleeve ON: short ~19 sh at 1.5×
    (~12 at 1×) [gameplan §6 D2 / S1]   (fixture perp $173.53 → higher mark caps
    fewer shares than the S5d transcript's $158 — same calc arithmetic)
18:05 CROSS — operator clicks "mark CROSS" + sets cross 18:05 @ $165
  → marks broadcast; NOW: D5 — tranche clock 15 min → T1, sold 0/40 (0%) vs 40%
19:30 — operator updates sold=8 → D5 T2, sold 8/40 (20%) vs target 80%
21:45 → NOW: CLOSE-OUT — don't carry residual overnight (5/6 day-2 lower); 32 sh left
state JSON after the day: {offer 135, fill 40, sold 8, cross 18:05 @ 165, …}
```

**Soak test.** A ≥30-minute real-polling soak (`--watch 45 --serve --parquet-log --html`) ran against the live PM/HL endpoints on 2026-06-11; result recorded below in §10.1.

**Tests.** 6 new on top of the S5b–S5d suite: wire-payload serializability + downsampling (20k-point fit → ~400 on the wire), page self-containment + vendored-ECharts serving + path-traversal guard, **websocket snapshot/push/reconnect** against a real ephemeral-port server (client drop → reconnect → fresh full snapshot → pushes resume), `/api/state` round-trip (update → persisted JSON → immediate playbook re-broadcast), `/api/mark` broadcast, and a **static-mode regression** proving `--html`-only runs are unaffected by the server layer. The acceptance-listed S5d tests (state-file round-trip, scripted-day node inference, hedge-arithmetic equality vs the calc, AVWAP stream-replay) remain green and unchanged. Suite: **57 monitor tests, 84 with the calc — all green.**

### 10.1 Soak result (2026-06-11, 00:00–00:36 CET — PASS)

The full stack (`--watch 45 --serve --parquet-log --html`) ran against the live PM CLOB and Hyperliquid endpoints for **~39 minutes**: **46 polls, 46 parquet shards written, 0 push failures, 0 skipped polls, 0 exceptions**; the static fallback page was atomically rewritten through the final poll (00:36 CET) and two real browser-client connections mid-soak received a 49-poll history snapshot and live pushes (P(win) 82.7%, mean $167.2, perp $162.72 at the time). Termination was the test harness's planned SIGALRM (exit 142), not a crash. One robustness finding, fixed: terminal tables were block-buffered when stdout is redirected to a log file — the render print now flushes per poll. The mid-soak design pass (masthead node chip, semantic palette, levels strip — see the layout bullet above) was served live by the running instance, confirming the page is hot-reloadable; the two additive payload fields (node, levels) require the next process start.

## 10.2 Block S5f — overnight gap-filler (2026-06-11)

**The problem the operator hit:** the live monitor only writes a parquet shard while it is actually running, so any window where nobody was polling — overnight, while asleep — is a hole in the dashboard time series. On a fresh calendar morning the charts looked nearly empty even though yesterday's run had logged hundreds of polls.

**Two fixes, one cheap and one real:**

1. **Read-window widening (`--backfill-days N`).** The startup backfill was hard-filtered to *today's* UTC shards, so a cross-midnight restart dropped the prior session even though its shards were still on disk. The flag now replays the last `N` UTC days (default `1` = clean listing-day session; `2+` carries the run-up across a restart), and retention auto-widens to `N·24+6` h so the older shards are not immediately trimmed by the 12 h cap. This recovers data that *was already logged*, nothing more.

2. **Historical reconstruction (`scripts/spcx_backfill_history.py`).** The genuine holes — hours when nothing polled — are filled from each venue's **historical** endpoint, which the live monitor never calls: Hyperliquid `candleSnapshot` for the xyz:SPCX perp mark (24/7 OHLC) and Polymarket CLOB `/prices-history` for each strike/bucket token's midpoint series. The script replays both onto a regular grid (default 10 min), reconstructs a synthetic snapshot at each grid time in the **exact shape `build_snapshot` emits**, and runs it through the **same `analyze()` + `log_parquet()`** the live monitor uses. Result: the shards are byte-compatible, the dashboard's `backfill_from_parquet` loads them with zero changes, and the survivor-fit / EV / percentile math is identical to a live poll.

**Worked example.** At 03:05 UTC on 06-10 — a dead zone with no live poll — the filler reads each ladder token's last CLOB midpoint at-or-before 03:05, assembles a snapshot, fits the survivor, and writes `poll_20260610T030544.parquet` with `mean_ps ≈ $163.9`, `perp ≈ $155.6`. After a 48 h fill the combined history spans a continuous 06-09 → 06-11 with the **largest gap reduced to the 10-min grid spacing** (was a multi-hour overnight hole).

**Lookahead-free by construction.** Each grid point forward-fills only quotes with timestamp `≤ t` (`_asof`); no future quote ever leaks into a past point — the same non-anticipation discipline the rest of the repo enforces.

**Gap-aware and idempotent.** By default the filler **skips grid points already covered by a live shard** (within ±½ grid step), so it fills only genuine holes and never stacks a midpoint reconstruction on top of a denser live best-bid/ask poll; shard filenames key off the historical timestamp, so re-running overwrites rather than duplicates. `--fill-all` overrides the skip; `--dry-run` reports without writing.

**Honesty caveat (fidelity).** Live polls read the best executable **bid/ask** off the live book; the backfill only has the CLOB **midpoint** history, so it sets `bid = ask = mid`. For an overnight *trend* view that is fine, but a backfilled point is a **midpoint reconstruction — one notch below a live best-bid/ask poll**, not a perfect replay. **Spot is not backfilled**: SPCX is not a listed equity until listing day, so there is no overnight price (pre-listing "spot" was only the AAPL feed test).

**Tests.** 6 new (offline, monkeypatched feeds): lookahead-free `_asof`, gap detection, synthetic-snapshot schema parity through the real `analyze()`, dashboard-readable shard columns, idempotency on re-run, and `--fill-all` override. Suite now **63 monitor+backfill tests** (90 with the calc) — all green.

**Run order for a fresh morning:**
```bash
# 1. fill the holes (read-only; reconstructs from venue history)
uv run python scripts/spcx_backfill_history.py --hours 48
# 2. relaunch the dashboard with a window that covers them
uv run python scripts/spcx_pm_pdf_monitor.py --serve 8642 --backfill-days 2 \
    --parquet-log --spot-ws alpaca \
    --html data/analysis/spcx_convergence/pm_pdf_dashboard.html
```

## 10.3 Block S5g — PM tail-sell screen (2026-06-11)

**Why:** the operator's Polymarket leg of the gameplan is to **sell the rich upper-tail "cap above $K" YES into the pop at PEAK** (gameplan §lines 196/199/246 — "when the stock rips the >$2.4T/>$3T tails reprice in minutes, exactly when selling them is the trade"; "YES bids fading = crowd lowering the close" is PEAK signal #6). The dashboard measured the crowd distribution but didn't surface the *trade*. This block turns panel 01 into the tail-sell screen and resolves three operator decisions (basis/liquidity/buckets).

**The key reframe — for SELLING you read the bid, not the ask.** To sell "cap above K" YES you hit the **YES bid**, so the executable sell price and the implied P(cap>K) are the bid. The screen shows, per upper-tail strike (>$2.2 / 2.4 / 2.6 / 2.8 / 3.0T — gameplan's named >$2.4T/>$3T plus resolution around the action):

| column | meaning |
|---|---|
| **sell** | the YES bid in ¢ = the crowd's implied P(cap>K) = what you receive selling here |
| **Δwin** | how that bid repriced over the trailing window (default 15 min) — *the* signal: ripping up = chasing the rip; fading = PEAK |
| **off-hi** | ¢ below the bid's own window high (the fade-from-high read) |
| **depth** | shares resting at the bid right now — sellable size (the tails are thin, so this gates execution) |
| **vs fair** | bid minus the 1/spread²-weighted lognormal fair (a *secondary* dislocation read) |

**The PEAK trigger is repricing, not the model gap.** The composite fires `go` ("PEAK signal #6 firing → SELL the tail") only when tails that **ripped up** are now **fading from their highs** — the gameplan's actual signal. It fires `watch` when tails are ripping (sell zone opening, wait for the fade), else `idle`. This is deliberately *not* driven by "rich vs the lognormal fair," because (see below) that metric is structurally biased in the tail.

**Q2 — liquidity-weighted lognormal fair (the audit's construction), ported numpy-only.** A `1/spread²`-weighted lognormal survivor is fit in cap space by a dependency-free two-stage grid search (erf-based normal CDF, no scipy), mirroring [`spacex_pdf_builder_v2.py`](../../../scripts/spacex_pdf_builder_v2.py). It reproduces the audit on the 06-07 fixture (median cap ≈ 2.14T, weighted RMSE ≈ 1.1¢) and is drawn as a violet dashed "fair" line on the survivor chart. **Honest caveat:** the audit already showed the lognormal sits slightly *above* the kinked market in the deep tail, and the executable bid is half-a-spread below the mid the fit targets — so `vs fair` reads mildly **negative even at fair** (live: −4 to −5¢). It flags *relative* dislocation between strikes, not the sell trigger. The basis question (mid vs best-ask) is moot for the same reason the audit found: best-ask is only ~½ spread from mid (+0.4pp / +$0.6) — immaterial; the actionable price is the bid.

**Q1 — depth is now capturable live (the offline audit couldn't).** `fetch_books` now keeps the size at best bid/ask (the live `/book` carries 70+ levels per strike). Depth is read from the **current snapshot only** — deliberately *not* added to the parquet schema or history (it's a "can I sell size right now" readout), so the log format is unchanged and backfilled shards are unaffected.

**Q3 — buckets removed from the LIVE tab.** The 7-bucket market's only role was the one-time consistency check (done; RMS gap 1.66pp). It's thin (the key 1.5–2.0T bucket has the widest spread in the book) and adds nothing to tail-watching — the 16-strike ladder *is* the finer tail. Panel 05 and the bucket overlay are gone from the interactive LIVE tab (still computed for the parquet log and the terminal/static fallback as a background diagnostic).

**Worked example.** The >$2.6T bid sits at 14¢ all morning, then on a +30% spot rip climbs 14→24¢ over 10 min (Δwin **+5¢**, "ripping → ready, sell on the fade"); ten minutes later it ticks back to 19¢ — now **−5¢ off its high** with ≥2 tails doing the same → composite goes `go`, the row reads "ripped then fading −5.0c off high → **SELL (PEAK)**". You sell the YES at the 19¢ bid into the size shown in **depth**.

**Tests.** 6 new (90 total with the calc): lognormal fit recovers a known (μ,σ) and reproduces the audit median on the fixture; the rip-then-fade synthetic fires the PEAK signal and tags the right strikes SELL; a flat tape stays `idle`; richness sign is correct and missing fit/depth/history degrade to `None` without raising; the LIVE payload carries `tail_trade` + a grid-aligned `lognormal.fair_S` and **no** `buckets`, JSON-serializable. Full suite green; live-booted and verified over the websocket (real depth 1.3k–7.8k sh/strike, fair median 2.13T, "tails quiet" pre-listing as expected).

## 10.4 Block S5g.2 — time-scrub slider for the PDF/survivor (2026-06-11; relabeled — the operator's Block S5h is §10.5)

**What it is:** a slider above panel 04 that replays how the crowd's PDF and survivor curve (and the perp marker on them) looked at any logged moment — the operator asked to *see the evolution*, not just the latest fit. One shared slider drives both charts; a sky vertical marks the scrubbed moment on the pair and session charts so the curve shape can be read against where the perp was trading; a LIVE button (and dragging to the right edge) snaps back to the current poll, which keeps streaming meanwhile.

**How it works (one code path, no new math):** the parquet log already stores the **full 16-strike ladder per poll**, so `CurveIndex` loads every shard in the window (default `--curve-days 7`, ~2.2k polls) into memory and appends each live poll as it lands — the scrub range keeps extending while the monitor runs. Dragging the slider hits `GET /api/curve?ts=…`, which snaps to the nearest logged poll and **refits it through the same `analyze()` a live poll uses** (PCHIP + liquidity-weighted lognormal + raw histogram, identical blocks via the shared `curves_payload()`), off the event loop, ~50–80 ms uncached / ~2 ms cached. The browser draws the response through the same chart functions as a live poll — historical and live curves are pixel-comparable by construction.

**Coverage:** the week backfill (`spcx_backfill_history.py --hours 168`, gap-aware) extended the shard log to **06-04** — venue history supported the full requested week (675 shards written, 1 grid point skipped). Worked read from the scrub itself: 06-04 the crowd mean was **$177.3** (perp $173.3, P(win) 88.7%); by 06-11 it is **$166.8** (perp $165.0, P(win) 85.5%) — the whole distribution drifted ~$10 lower over the week with the perp tracking it.

**Honest caveats:** scrub points inherit the fidelity of their source shard — backfilled overnight points are midpoint reconstructions (§10.2), live-logged points are real best-bid/ask polls. The tail-sell markers and panel-01 screen stay live-only (depth is a now-quantity, not logged). Bucket overlay is not part of the scrub (removed from the live view in S5g).

**Tests:** 4 new (168 total green): CurveIndex shard-load/nearest-snap/refit-equivalence (refit stats match the original poll's), append-only range extension, the `/api/curve` endpoint end-to-end on an ephemeral server (snap-to-edge, cached-identical second hit, 400 on missing ts), and `curve_range` in the websocket payload (null without an index — the slider hides).

## 10.5 Block S5h — listing-eve additions (2026-06-11 evening; freeze reopened by the operator for this block only, RE-FROZEN after)

Four items, landed in priority order, each tests-green and relaunch-verified before the next started. All are **display/advisory only — no order logic exists anywhere in this codebase.**

### Item 1 — hedge-ops panel

One amber-edged card, visible whenever hedged-state is set (`--hedged`/page form; `hedge_ts` is stamped automatically the first time the short goes on, for the funding clock). Shows: entry price, **basis locked** (entry − offer, $/sh and total), **liquidation price + buffer %** with SAFE/WARN/BREACH bands — the arithmetic is **imported from `spcx_convergence_calc`** (`maintenance_margin_frac` / `liq_price_short` / `liq_buffer_summary`; an equality test pins the panel bit-for-bit to the calc module), with the venue's live max-leverage from the HL meta (5× → mmr 0.10 at verification). Margin posted (declared EUR × FX) and **funding accrued ≈ current public HL rate × mark notional × hours-on** (labeled an approximation — read-only monitor, no account API). The **S6 readiness ladder runs as a live countdown**: PRE-CROSS → NO-ZONE (first ~45 min, with minutes-to-the-$2-gate) → GATE-2 (+46 min, $2) → GATE-1 (+61 min, $1) → GREEN (pair-close chip confirms), with the live perp−spot gap chip. **21:00 CET backstop countdown** carries the S6-measured ~$1.7/sh cost note. The **leg-sequencing card quotes gameplan §5.1 verbatim** — default "TR sell first, then perp" plus the NEW fast-tape exception (liq buffer < 25% **or** >2% spot move in 5 min ⇒ perp leg first), both inputs evaluated live each poll and the rule flagged *"pre-registered judgment, not measured"*.

### Item 2 — lean tranche tables + day-shape overrides (§5.2, decided today)

The displayed schedule now follows the lean tables, **auto-selected from fill − hedge sleeve**: ≤10 sh → two tickets 60/40 (skip T3); ~20 → 8/8/5; ~40 → 17/17/9; ~65 → 26/26/13 with T1 placed immediately post-observe (flagged). Off-row residuals show the nearest row verbatim with the **last ticket absorbing the difference** (e.g. fill 40 − hedge 18 = 22 sh → 8/8/6, live-verified). Each ticket renders shares, its post-cross window (T1 +15–60, T2 +60–180, T3 +180→21:30 CET), and DONE / PARTIAL (n/m) / ACTIVE / OVERDUE / opens-at status against cumulative `--sold`. The §5.2 overrides change the **displayed** schedule only: **FADE** halves every remaining window from now (tested: at +70 min, T2's end 180→125, T3's start 180→125) and flags "sell the next tranche NOW (T2 8 sh)"; **RALLY** defers the final ticket and prints the mental stop at 0.9 × session high; **CRASH** voids the table — hard-stop ladder governs; **FLAT** default.

### Item 3 — S7 day-shape classifier + banner (§5.3, pre-registered today)

The state machine exactly as pre-registered: **CRASH** = below the cross print AND below $160, or any hard-stop level hit ($140/$125) — instant, no hysteresis; **FADE** = below anchored VWAP ≥10 min AND >5% off the session high; **RALLY** = above AVWAP AND a new session high within 15 min; **FLAT** = else; non-CRASH switches need the candidate to hold **2 consecutive minutes**. Inputs: IEX spot + AVWAP, cross print, internally-tracked session high, the hard-stop levels — **no volume features** (IEX too thin). Full-width banner: state in its color, the matching **§0 action row verbatim**, minutes-in-state, and a manual override dropdown that **pins the state until released** (computed state stays visible beneath; override always wins, including into the tranche overrides). The banner stays hidden until the first spot print arms the classifier.

**CBRS sanity replay (smoke test, NOT calibration — thresholds untouched).** Classifier run over the cached Cerebras 1m listing-day tape (`ipo_tapes/cbrs_spot_1m_listingday.csv`; cross = first volume bar, AVWAP = running Σpx·vol/Σvol from it):

```text
state timeline (minutes post-cross):
  +0.0   FLAT   (arms at the cross print, $350)
  +27.0  FADE   ← only transition; held for the entire remaining session
  — no RALLY at any point; no CRASH (CBRS trades ~$300+, SPCX's absolute stops inert)
```

One honest note, reported rather than tuned away: the block spec asked for "FADE within the first ~20 min", but the pre-registered rule itself makes that arithmetically unreachable — S2 measured the AVWAP loss at +12 min, the FADE clock needs 10 more minutes below it, and hysteresis adds 2 ⇒ a **≥24-min floor by construction**. The measured +27 min is consistent with the rule as written (the replay's own AVWAP loss lands ~+15 min vs S2's +12 — same construction-sensitivity caveat as the audit). The test asserts ≤30 min with this reasoning in its docstring; **no threshold was adjusted to make the replay prettier.**

### Item 4 — cross-timing poller + indication paste field

**(a)** A background thread polls Nasdaq's public trading-halts RSS (`nasdaqtrader.com/rss.aspx?feed=tradehalts`) for SPCX every 60 s, self-gated to ≥15:00 CET. The cross-timing card shows halt code, **resumption quote time, resumption TRADE time (= when the cross prints)**, and data age. Parsing is defensive (the page is not an API contract): any fetch/parse failure renders **"poller down — watch CNBC"**, never stale data; pre-listing the live feed correctly returns "SPCX not in the halts feed" (verified against the real endpoint). When a resumption trade time first appears, a one-shot alert fires the in-page path (toast + title flash) like a state change. **(b)** An **indication paste field**: the operator types e.g. `148-152` when CNBC/X relays it → timestamped, appended to the persisted day-state (survives restarts), displayed latest + history **next to the perp and PM mean**, broadcast to all clients immediately, and logged onto the parquet shards as `indication_text`/`indication_ts` (schema-extended; older shards lack them — `union_by_name`). **No headline-scraping poller — explicitly out of scope.**

### Tests + soak

19 new tests (191 total, all green): buffer-band equality vs the calc module; fast-tape inputs incl. no-tape degradation; S6 ladder phases incl. GREEN-from-history; tranche row selection with hedge-sleeve subtraction (incl. the verbatim rows at their own sizes and last-ticket absorption); FADE/RALLY window arithmetic + CRASH handover; classifier FADE-with-hysteresis, RALLY-from-stale-high decay, instant CRASH on both legs, override pinning, blip immunity; the CBRS replay assertions above; halts parse of a real-shaped fixture, malformed-input degradation, one-shot alert consumption; indication POST → persisted JSON reload → render → parquet columns → 400 on bad input.

**Live soak (2026-06-11 evening — PASS, with a bonus robustness finding).** The full stack (`--watch 45 --serve --parquet-log --spot-ws alpaca --html`, all four S5h panels live) launched 19:23 CET. Mid-soak the operator's machine went to **OS sleep ~19:26–22:27 CET**; the stack **survived the sleep/wake cycle unattended** — on wake it re-established the CLOB/HL/Alpaca connections itself, shed exactly 6 transient "0 usable ladder strikes" polls at the wake boundary (the refuse-to-fit guard, working as designed, no garbage shards written), and resumed clean polling. The acceptance soak was then measured on the **uninterrupted post-wake window: 22:27–22:50 CET, 26 polls at perfect 45 s cadence, 0 push failures, 0 tracebacks, 0 further skips, process alive throughout** (49 shards total across the evening; last live read perp $168.28, −$2.9 vs PM mean). The same running process also hot-served the PLAN-tab gameplan refresh (fast-tape exception, lean tranche tables, S7 state table, +61 min gate) without a restart — the serve-from-disk design doing its job.

Scope is now **RE-FROZEN** — no further dashboard changes before listing.

## 10.6 Block S5i — REHEARSAL tab: full-day dress rehearsal on a real Nasdaq tape (2026-06-12 pre-dawn; the freeze's single reopening, now FINAL)

**What it is.** A third dashboard tab that replays a real Nasdaq session through the production pipeline, pretending the 9:30 ET open print is the IPO cross — so the operator can drill the day (watch the classifier flip, click tranche tickets, read the hedge-ops ladder) on a tape that actually happened. It is a **rehearsal harness, not a feature**: engine functions are imported read-only, and the whole thing is provably isolated from production state.

**The level-scaling trick (zero engine modification).** SPCX's dollar levels are meaningless on NVDA, and the classifier's $160/$140/$125 constants are pre-registered — they must not be forked. Both problems solve each other: with simulated offer := cross ÷ 1.30, every gameplan level is a *ratio* to the offer, so mapping the rehearsal price into SPCX-dollar space (`px × 135/offer`) before feeding the **production** `DayShapeClassifier` makes its unmodified constants *exactly* the scaled levels. One multiplication; provably the same rules (a test asserts the round-trip: scaled level × mapping = SPCX constant, bit-exact). The scaled mapping is printed on screen (NVDA 06-11: cross $201.33 → sim offer $154.87 → CRASH floor $184, reassess $161, sell-everything $143…).

**Build.** `scripts/spcx_rehearsal.py` (tape fetch Alpaca-historical → Yahoo-1m fallback → CSV cache in `ipo_tapes/`; RTH filter via America/New_York so the first surviving bar is the simulated cross; `build_rehearsal` precomputes close/AVWAP/session-high/perp-proxy and the classifier **state ribbon stepped sequentially — lookahead-free by construction**) + three isolated server endpoints (`/api/rehearsal/load|panel|state`) + the tab UI: price chart with AVWAP, scaled-level lines and the SIM CROSS marker (revealed only up to the playhead), a colored **state ribbon** (future stays dark), play/pause at 1×/10×/60×, scrub bar, "next state change" jump, the production-rendered tranche table with rehearsal-only **mark-sold buttons** (8/16/22 cumulative), the production-rendered hedge-ops panel with a SIMULATED note (entry = cross-time stock + $25 proxy), and the always-on violet banner ("REHEARSAL — symbol date — simulated cross/levels/position"). Panels with no replayable source (PM distribution, real perp, halts, indication) are a "live SPCX — not part of rehearsal" note; their production behavior is untouched.

**Isolation (the acceptance bar) — proven by test, not asserted:** rehearsal day-state lives in `rehearsal_state.json` (its own namespace; the server takes the path as a parameter so tests pin it to a tmp dir); after a full rehearsal lifecycle (load → scrub panels across the session → mark sold → reload) the **production `playbook_state.json` is byte-identical**, the **only new file on disk is the rehearsal state file**, the production websocket payload is unchanged (same inputs → deep-equal output), production `/api/state` stays live, and **10/10 production pushes deliver in order to a connected client while the rehearsal panel endpoint is hammered**. `build_rehearsal` is also tested not to mutate its input bars. No parquet is written anywhere by any rehearsal code path.

**NVDA 2026-06-11 end-to-end (the dress rehearsal of the dress rehearsal).** 390 RTH bars via Alpaca (then cache). Session state timeline through the production classifier:

```text
+0    FLAT   (sim cross $201.33)
+14   RALLY  (new highs above AVWAP — NVDA was an up-day)
+44   FLAT   (high went stale)
+255  RALLY
+272  FLAT
+358  RALLY  (close strength)
```

19 panel renders checked across every state change + 30-bar samples: 0 unexpected (hedge-ops + tranche + banner all consistent with the ribbon at every playhead). Mark-sold T1 → DONE arithmetic verified through the API; the RALLY override (final ticket DEFERRED, 10%-below-high stop) rendered at the RALLY bars. **No FADE/CRASH occurred — NVDA 06-11 was simply not a fade day**; the FADE path is exercised by the synthetic pop-then-fade tape test and the CBRS replay (§10.5), and the operator can load any fading session by date. Visual screenshots are left as the operator's 1-minute check (open the tab, press LOAD + PLAY at 60×); the panel-state evidence above is API-verified.

**Tests:** 12 new (`tests/test_spcx_rehearsal.py` + 2 endpoint guards) — **218 total, all green**. Tape cache/fallback chain, pre-market exclusion + first-RTH-bar cross, scaling ratios + SPCX-space round-trip, **lookahead truncation** (ribbon prefix at cuts 30/60/90 identical with and without future bars), fading-tape reaches FADE, ticket-click arithmetic, input non-mutation, the isolation suite above, 404-before-load / 400-on-missing-index.

**The freeze is now FINAL — nothing else ships before listing.**

## 10.7 Block S5j — two interpretation charts (2026-06-12 pre-dawn; display-only, the last addition)

Two charts that render exclusively from values the server already computes — no new data, state, or rules. Both draw on the LIVE tab **and** in the REHEARSAL tab from replayed values through the **same code path** (`tranche_chart_data` / `hedge_chart_data` are called identically by the live payload and the rehearsal panel), so the dress run exercises them.

**Chart 1 — tranche schedule ("where am I in the sell plan").** Horizontal timeline, cross → 21:30 CET, under the ticket list. Each lean-table ticket is a block spanning its phase window horizontally and its cumulative share range vertically (so the blocks form the sell staircase); solid when marked sold, amber-bordered when active, dashed-grey when RALLY-deferred. Dashed step-line = end-of-window cumulative target; emerald line = actual sold; white playhead = minutes since cross. **The override redraw imports the panel's arithmetic** — the numeric window bounds are emitted by `tranche_schedule` itself (one function, already tested), and an equality test pins the chart block to the same FADE-halved numbers the panel shows (+70 min FADE: T2 end 180→125, T3 start →125). CRASH collapses all bands into a single red "SELL ALL n sh — hard-stop ladder" block.

![[s5j_tranche_chart_rehearsal.png]]
*Mid-rehearsal render (NVDA tape, +120 min, T1 marked sold): T1 solid, T2 active under the NOW playhead, T3 pending; target staircase vs the sold line.*

**Chart 2 — hedge unwind ("can I close now, what does it cost").** Two strips on a shared time axis under the hedge-ops panel. Top: the perp−spot gap against the ±$2/±$1 readiness lines, with the S6 ladder as background zones (0–45 min red no-pair-close, 46–60 amber $2 gate, 61+ green $1 gate), the 21:00 CET backstop as a hard red vertical (~$1.7/sh note), and a READY marker when the readiness chip first goes green (first-green is remembered on the dashboard state). Bottom: liq-buffer % over time with the WARN (<10%) and BREACH (≤0) bands — **each point is the calc module's `liq_buffer_summary` at the entry-fixed liq price** (point-for-point equality test). Below: the live readout *"close both legs now ⇒ locked basis − gap drag = $X total"*, tested equal to (basis − gap) × hedged exactly. Pre-cross: the gap strip shows "awaiting cross" while the buffer strip runs from hedge entry.

![[s5j_hedge_chart_rehearsal.png]]
*Mid-rehearsal render: the SIM perp proxy (stock + $25) puts the gap at ≈$25 — correctly far outside the readiness bands (the proxy is constant by construction); buffer ≈50%, safely above WARN; readout: locked $71.46/sh × 18 = $1,286 − drag $450 = $836.*

**Tests:** 4 new — override-redraw arithmetic equality, close-now + buffer-series equality vs the calc module, empty states (no hedge → no chart; no fill → no chart; no cross → no playhead, gap series null while buffer runs, drag unknown), and carriage in both the live payload and the rehearsal panel (JSON-serializable). **222 total, all green.** Relaunch-verified: both containers and draw functions serve; the PNGs above are rendered from the live rehearsal API mid-session.

**Freeze is FINAL-final.**

## 10.8 Block S5k — EXECUTION tab + day-view audit fixes (2026-06-12 pre-dawn; operator-directed)

**Why:** the operator's audit call on listing-day morning: the LIVE tab had become ~2 screens of execution panels interleaved with ~2 screens of research surfaces; the S5j charts were invisible pre-allocation with nothing saying where they'd appear; the day-state form was unclear ("don't know what to input, buttons don't seem to work" — endpoints verified fine, the failure was *feedback*: a tiny "saved ✓" and no immediate visual change); the CNBC indication paste field was judged unused.

**The restructure (additive + moves, no engine rule changes):**

- **New EXECUTION tab, the default view** — the rehearsal layout fed live: S7 banner → mini tile strip (spot · perp · gap · P(win)) → **E1 live chart** (IEX spot + anchored AVWAP vs every pre-registered level line, CROSS marker, perp line pre-cross) with a **live-building state ribbon** beneath (per-poll classifier states stamped onto the history records and carried through the snapshot — backfilled records honestly lack them, so the ribbon starts with live polling) → playbook NOW card → tranche table + S5j schedule chart → hedge-ops + S5j unwind chart → halts card → **E5 guided day-state card**.
- **E5 day-state card** replaces the old drawer: labeled inputs with a what-to-do-when caption (allocation → fill+SAVE; hedge fields if the overflow rule fires; first print → MARK CROSS NOW; after each ticket → sold+SAVE), offer shown as fixed ($135 priced ✓, no longer an input), and **loud feedback** — every save/mark fires the toast + title flash, and the playbook re-renders instantly as before. SAVE and MARK CROSS verified end-to-end through the API.
- **LIVE tab renamed ANALYSIS** and reduced to the research surfaces: full tiles, levels strip, tail-sell screen, pair, session, crowd distribution + time-scrub. Day-state inputs point to EXECUTION/E5.
- **Indication paste card dropped** from the day view (operator decision — the halts card is the real cross-time alarm and stays). The `/api/indication` endpoint, render function, and parquet columns remain dormant. **Found + fixed during the audit:** the previous night's test indication ("148-152 (test)") was still in production state and being stamped onto every parquet shard — purged.
- **D1 marked RESOLVED** on the PLAN tab and annotated in the gameplan: priced **$135** — basis math holds exactly as computed, the $162 contingency is void, the $183 pre-hedge trigger stands.

**Tests:** 2 new (snapshot carries per-poll shape/avwap for the ribbon; payload no longer carries the indication card while halts stays) — **224 total, all green**; JS syntax-checked; relaunch-verified (EXEC tab default, all elements serving, removed elements absent, clean pre-allocation state restored: offer 135 / fill null / cross null / sold 0).

## 10.9 Block S5k-PM — PM tab + pre-allocation button fix (2026-06-12 listing-day morning; display-only, surgical)

> Naming note: the operator issued this block as "S5k", but §10.8 above already carries that label (the pre-dawn EXECUTION-tab block). Disambiguated here as **S5k-PM**.

**Why.** Two listing-day-morning operator asks, both display-layer: (1) the EXECUTION tab's day-state button "does not respond" in the pre-allocation state; (2) the Polymarket-leg analytics — which the engine computes and the terminal prints every poll — deserved their own tab for Alvaro's PM screens, instead of being scattered between the terminal, ANALYSIS, and a dropped card.

### (1) Button verdict — no code bug reproducible; the failure mode is a dead server; feedback hardened

- **Reproduction attempt (UI path, live server, real browser click):** typed `fill=22` into the E5 form and clicked 💾 SAVE DAY STATE — POST `/api/state` round-trips, the state file persists, the playbook card re-renders D2/D3 instantly, the toast fires, zero console errors. The bare API path (curl) round-trips equally. The button, as coded, works.
- **What the evidence says happened instead:** the production process was **down** when this session started (nothing on port 8642 at 12:21 CET; last parquet shard written 12:18:51 CET — it had been running through the morning, relaunched ~11:49 CET around the S8 arb investigation). A click in a browser tab whose server has died fails *silently* except for a 12-second toast and the masthead DISCONNECTED banner — far from the button a stressed operator is staring at. This matches the §10.8 history: the same complaint shape ("buttons don't seem to work") was diagnosed then as a *feedback* failure, not an endpoint failure.
- **Fix (display-only, ~15 lines of page JS):** on websocket disconnect the E5 buttons now **disable** (greyed, `cursor:not-allowed`) and a red **"SERVER UNREACHABLE — buttons disabled until the feed reconnects (is the monitor process running?)"** prints in the `saveok` slot next to the buttons; everything re-enables on reconnect. Failure toasts now distinguish "server unreachable (is the monitor process running?)" from "server error". No engine, state, or playbook logic touched.
- **Regression test** (`test_api_state_preallocation_save_button_regression`): pins the exact pre-allocation E5 SAVE body (`fill+comfort+hedge_lev+sold`) through a real ephemeral server — 200 + changed-set + JSON persistence + the instant playbook re-broadcast carrying the new state — and the clear-back-to-`null` path (fill must clear, not coerce to 0).

### (2) The PM tab — render what the engine already computes

Tab order is now **EXECUTION | PM | ANALYSIS | PLAN | REHEARSAL** (the four directed tabs in the directed order; PLAN kept where it was). Nothing on the EXECUTION tab moved or changed (structure-tested). The pane, top to bottom:

- **Verbatim header note:** *"Analytics for the PM leg (Alvaro). Display only — no PM trading logic, no edge claims; DIVERGENT = quote gap before spread/fees, check depth before acting."*
- **Perp-vs-crowd strip** — perp mark, gap to PM mean and PM median (the ±$/sh pair from the terminal print), every poll.
- **PM1 — full distribution table**, exactly the terminal block: P(close>$135), E[cap], EV vs offer (A1 = mean − offer, never the dead PNG convention), then mean/median/mode/P25/P75/P90/P95 in cap $T and **both** per-share bases (13.076B + Alvaro's 13.091B), with the mode shape-fragility footnote and the strikes-used count + quote basis.
- **PM2 — bucket-vs-ladder consistency screen** with DIVERGENT rows highlighted. The flag now has **one source of truth**: a new `bucket_divergent()` (±5pp) that the terminal renderer and the PM table both call — equality of the two surfaces is pinned by test, including the strict-`>` boundary semantics. Each row carries a **gap sparkline from today's poll log** (grey zero line, dashed ±5pp guides, end-dot red when divergent) plus a "N pp 2h ago" recency read — a newly opened divergence (like this morning's 2–2.5T) is visibly a fresh move, a stale one is a flat wide line. History points come from the parquet backfill **recomputed through the same `bucket_compare` path a live poll uses** (numeric equality live-vs-backfilled is tested); live polls stamp their gaps onto the dashboard history as they land. The No-IPO leg ask prints under the table.
- **PM3 — the S5g tail-sell screen, moved (not duplicated)** from ANALYSIS panel 01 — signal banner, sell/Δwin/off-hi/depth/vs-fair table, and the bid/ask history chart; a link stub remains on ANALYSIS. Unchanged code, new location.
- **PM4 — the indication paste field, resurrected** from its §10.8-dormant endpoint (this block supersedes that drop): paste `148-152` → timestamped, persisted, broadcast, parquet-logged as before. Found while wiring it: the §10.8 restructure had left `render_indications`' ADD button pointing at a **deleted JS function** (`sendIndication`) — restored with proper toast feedback; the same disconnect-hardening disables it when the server is unreachable.

**Live read at ship (12:41 CET).** The divergence that triggered S8 this morning kept widening while this block was built: the 2–2.5T bucket gap went **+8.1pp (09:48 UTC) → +14.6pp (12:41 CET)**, mirrored by 2.5–3T at **−11.9pp**, with the sparklines showing both as fresh, still-moving opens (+10.3/−7.9 two hours prior) and the 1.5–2T gap flat at ~zero. Read: the bucket market is shifting probability mass from 2.5–3T into 2–2.5T faster than the ladder is. **No edge claim — and S8 (which ran concurrently this morning) has since answered the executability question: NO-ARB.** Its decomposition: most of the flagged gap is the non-traded PCHIP S(2.5T) knot (~±6pp) plus wide-bucket mid-vs-executable spread (~4–5pp); the executable [2,3T) union box is −1.7c net ([[spcx_pm_arb_findings]]). The sparkline stays on screen as a *consistency diagnostic*, exactly per the header note: quote gap before spread/fees, not edge.

![[s5k_pm_tab_production.png]]
*The PM tab server-rendered from the production payload at 12:43 CET (PM1 + PM2 + indication card; PM3 is the unchanged S5g screen and draws client-side). Note the two DIVERGENT rows and their sparklines — both gaps opened this morning and are still widening at ship time.*

### Tests + restart + soak

**Tests:** 5 new + 1 rewritten — PM-panel render smoke on the fixture (both share bases, headline stats, footnotes, perp strip, one sparkline per bucket row); page structure (tab order, verbatim note, moved-not-copied panel ids, ANALYSIS stub, E5 untouched on EXEC); DIVERGENT single-source equality (terminal rows == PM rows == `bucket_divergent` set, boundary pinned); synthetic two-poll parquet log → backfilled gap equality vs live `analyze()` + red end-dot on the freshly opened divergence; the button regression above; and the §10.8 indication-drop test rewritten to assert the card now ships in the payload (superseded by this block). Suites: **138 green** across monitor/rehearsal/calc/backfill + **40 green** across unwind-tape/perp-exit/S8-arb (which imports this monitor) — all SPCX tests on disk pass.

**Restart + state restore:** production relaunched 12:43 CET with the standard stack (`--watch 45 --serve --parquet-log --spot-ws alpaca --backfill-days 2 --html …`). On startup: 16/16 strikes, Alpaca feed attached (SPCX pre-listing "no prints yet" = expected), **1330 polls backfilled** from 2 days of shards, 2445 polls indexed for the time-scrub, `playbook_state.json` restored clean pre-allocation (offer 135 ✓, fill/cross null, sold 0), first poll clean, PM panel + indication card verified present in the live websocket payload.

**Soak:** 10-min live soak PASS (12:43→12:54 CET, against the live PM CLOB + Hyperliquid + Alpaca feeds): **13 parquet shards written at the 45s cadence, 0 skipped polls, 0 push failures, 0 tracebacks, process alive throughout**; the static `--html` fallback page rewrote atomically; the PM panel + indication card served live over the websocket; spot correctly "no prints yet (normal pre-listing)".

**Caveats.** The PM1/PM2 panels are server-rendered HTML — they inherit the engine's per-poll degradation behavior (a failed bucket fetch renders "bucket books unavailable this poll", never a dead pane). Sparkline history spans today (CET) only, by design; backfilled overnight points are midpoint reconstructions one notch below live best-bid/ask polls (§10.2 caveat applies). The static `--html` fallback page is unchanged (terminal-text rendering — the PM numbers are all in it already).

### (3) Chart zoom slider — hard live anchor (operator follow-up, same morning)

**Ask, verbatim:** "charts on the dashboard need to be anchored at live on the slider, I just need to be able to slide backwards to see historical data, never get rid of the live data."

**What was wrong.** The time-series lookback sliders (the ECharts `dataZoom` on the EXECUTION price chart and the ANALYSIS session/pair/tail charts) *intended* the right edge to stay at live (`end:100`), but only re-forced it on each 45s poll redraw. Between polls, dragging the slider window or its right handle backward pushed "now" off the right of the screen, and it only snapped back on the next poll (up to 45s later) — so live data visibly disappeared mid-drag.

**Fix (client JS only, ~8 lines).** The `datazoom` handler now enforces the live anchor in **real time**: any gesture that moves the right edge below 100 (a pan or a right-handle drag) is snapped back to `end:100` instantly, while the user's `start` (lookback depth) is preserved and persisted. Net behavior: the **left handle is the only control** — drag it left to reveal more history, and the latest poll is *always* the rightmost point. The right edge can no longer be dragged off live at all. (Panel 04's PDF/survivor *time-scrub* slider is a separate, deliberate replay tool with its own LIVE button and is unchanged — that one is for freezing on a past moment on purpose.)

**Verified live** (test instance, real backfilled charts): a pan to start=40/end=70 springs to start=40/end=100; a right-handle zoom-in to end=50 springs to end=100 with start preserved; a pure left-handle widen (end already 100) is left untouched — no fighting the drag; same across the EXECUTION and ANALYSIS charts; zero console errors; 95 monitor tests still green. Pure front-end change — the running production process serves the HTML fresh from disk, so it took effect on a **browser reload, no restart** (confirmed: the live-served page now carries the snap-back code).

### (4) Backfill guardrail (operator follow-up: "backfill is broken for a few places — write a guardrail")

**The fragility.** Both parquet-replay paths — `DashboardState.backfill_from_parquet` (feeds every session/PDF/pair/tail chart) and `CurveIndex.load_parquet` (feeds the time-scrub) — did **unguarded direct dict access** on each shard (`first["mean_ps"]`, `_epoch(first["poll_ts"])`, `r["strike_lo_t"]`, …) with the try/except wrapping only the file *read*, not the row processing. So a single shard with a missing/renamed column, a malformed timestamp, or an old/forked schema would **raise straight out of the loop** — losing every shard after it, and (since `backfill_from_parquet` is called unwrapped at launch) potentially failing the whole dashboard start. "Broken in a few places" was silently taking down the whole replay.

**The guardrail (display-layer; the survivor/bucket math is untouched — routed through the same `bucket_compare`/`analyze`).**
- **Per-shard fault isolation.** Each shard's record is now built by a fully-guarded `_history_rec_from_rows()` that returns `None` on any missing column, bad timestamp, or **non-finite session anchor** (`mean_ps`/`median_ps`/`pwin` — caught via `_is_finite`, which also rejects NaN/inf, not just None). The caller **skips + counts** it (`unreadable` / `empty` / `malformed`) and moves on. Same isolation wraps `CurveIndex.load_parquet`'s whole per-shard body. One bad shard now costs one row, never the run.
- **Loud health summary.** After backfill, `backfill_health()` emits a stderr line — `[backfill] dashboard: loaded N, skipped M · reasons … · missing-anchor … · largest hole … min` — and escalates the tag to **`!! BACKFILL WARNING`** when the result is **degenerate**: every shard skipped (`loaded==0, skipped>0`), or a session anchor non-finite on ≥5% of loaded records. `perp`/`spot` being `None` is *deliberately not* a warning — it's the legitimate pre-cross state, so it must never trip the alarm. The difference being surfaced is exactly "pre-listing, nothing logged yet" (fine, quiet) vs "backfill silently broke and the charts are wrong" (loud).

**Verified.** 2 new tests (broken-shard isolation: good + corrupt-file + empty-table + missing-anchor-column + good, in filename order → loads exactly the 2 good, never raises; degenerate-case flagging incl. NaN-anchor and the pre-cross perp/spot-None *non*-warning). **160 SPCX tests green.** This change is engine-startup Python (not hot-reloadable like the HTML), so production was **restarted 13:05 CET** — the guardrail line printed clean on the real log: `[backfill] dashboard: loaded 1359, skipped 0 · largest hole 24 min` (a quiet `[backfill]`, no warning — all real shards well-formed; the 24-min hole is a real sleep-window gap, now surfaced); time-scrub indexed 2472 with no skip warning; state restored clean pre-allocation; short post-restart stability check passed (polls landing, 0 skips/push-fails/tracebacks, PM panel + slider both served).

### (5) Sell-tail shaded zone on the scrubbed PDF (operator follow-up)

**Ask:** "as I slide, I should still be able to see the 'sell tail' shaded area if I wanted." The PDF chart's tail-sell zone (the violet shaded band + the per-strike dots labeled with the sell bid) was fed only from the *live* `tail_trade` payload, which the time-scrub `/api/curve` payload didn't carry — so sliding back made the zone vanish.

**Fix (display-layer).** The tail markers are now computed in `curves_payload` (the block shared by the live websocket payload AND the scrub endpoint) as `pdf.tail_pts` — derived from the curve's **own** strikes, with **sell = the YES bid** at each tail strike (2.2/2.4/2.6/2.8/3.0T), strike position in $/share. `drawPDF` sources the zone from `pdf.tail_pts` on both paths (falling back to the live `tail_trade` rows), so the shaded zone and dots render identically live and while scrubbing — and because they come from each historical poll's own quotes, the scrubbed dots show the **sell bid as it actually was at that moment** (e.g. >$2.2T sell 0.535 overnight vs 0.60 live), not a stale live value. The series stays legend-toggleable, so it's there "if you want it." The live-only panel-01 columns (depth, Δwin repricing, off-high) remain live-only — the PDF marker only needs strike + sell bid.

**Verified.** 1 new test (`pdf.tail_pts` on both the live `build_ws_payload` and a refit historical curve via `curves_payload`, with `sell_bid` == the ladder YES bid). Live check after the 13:15 CET restart: `/api/curve` at an overnight ts returns the 5 tail markers with their historical bids; a browser scrub to 06-07 07:16 CET rendered the tail-sell series (5 dots) + the shaded `markArea` on the PDF (confirmed via the live ECharts option + screenshot). **161 SPCX tests green.**

> **Suite note (not a regression of this work):** `tests/test_spcx_pm_arb_check.py::test_best_executable_arb_one_fee_kills_it` (the separate S8 PM-arb block) fails when run *in isolation* but passes as a full suite and under deterministic order — a pre-existing test-isolation defect in that file (a test reads module state a sibling sets up), exposed only by `pytest-randomly`'s seed. `spcx_pm_arb_check.py` imports nothing from the monitor and was untouched here. Flagged for a separate session.

### (6) Today's last backfill hole filled + PM3 — the executable-arb card (operator follow-ups)

**Backfill hole (the "30-min gap at 12-something").** The 24-min hole 12:18→12:43 CET was the window between the morning production process dying and the §10.9 relaunch — genuinely nothing polled, so the §(4) guardrail correctly *reported* it (`largest hole 24 min`) rather than hiding it. Filled with the §10.2 historical reconstructor (`spcx_backfill_history.py --start 2026-06-12T10:15 --end 2026-06-12T10:45 --step-min 5` — gap-aware: wrote 4 midpoint shards, left the 3 live-covered grid points alone). After the relaunch the health line prints **no hole ≥15 min**; the largest residual gap in that hour is the 6.2-min grid spacing. Usual §10.2 caveat: those 4 points are midpoint reconstructions, one notch below live best-bid/ask polls.

**PM3 — "Ladder↔bucket mismatch — what's executable now (taker-only)".** The PM2 screen shows the *points*; PM3 shows the *cash*: given live L2 depth and one stated taker fee, what a taker could actually pull out of the books before the mispricing closes. Sits directly under PM2; the tail-sell screen and indication card renumber to PM4/PM5 (the drafted prompt predated the tab).

- **Math imported, not reimplemented** — all from S8's `spcx_pm_arb_check`: `fetch_arb_metadata()` once at serve startup (both YES+NO tokens per leg + resolution keys; ~24 GETs); `build_snapshot()` per poll (one batch `POST /books` over all 48 tokens — the monitor's own snapshot is top-of-book and can't size depth); `best_executable_arb()` for the render-ready verdict (walks asks best→next→…, stops at the first net-negative marginal set; ladder↔bucket covers + exact unions only, one resolution key, never mids/curves).
- **One fee, stated on the card, swappable.** fee/share = `(bps/10000)·min(p,1−p)` (the documented CLOB taker formula). **Both** variants — `0 bps` (what fills are observed to pay) and `1000 bps` (the declared `taker_base_fee`, the operator's "costs are fixed for takers" reading) — are computed server-side **every poll** (the walk is pure local math; one fetch serves both) and shipped together; the page toggle is a pure client-side swap, so no extra requests and the choice survives poll re-renders (re-applied in `apply()`) and reloads (localStorage). `--fee-bps` (default 0) sets the initial view.
- **Card contents:** verdict chip (strict palette: **green only when `investable`**, amber `lock exists — dust` — the normal case, neutral `no executable lock`; red never), the leg list as executable actions (`SELL 2-2.5T @ 0.41 (126 sh top) · …` — a NO taker-buy renders as SELL-the-YES), `pay $X/set → locked floor $F` + the free-upside sliver when present, and the headline `EXTRACTABLE: $N net over S sets ($Y notional) — edge closes by {fees/spread | book exhausted}`. Caption ties it to PM2 verbatim: watch this dollar figure, not the pp.
- **Read-only discipline:** no orders, and **no parquet writes from the serve path** (`build_snapshot` is fetch-only; the standalone S8 script remains the books-capture path). Any arb fetch failure degrades the card ("arb books unavailable this poll"), never the poll; a failed startup metadata fetch just leaves PM3 unavailable, loudly logged.

**Live read at ship (13:36 CET, production):** at 0 bps a lock exists and is **dust, breathing poll-to-poll** — `EXTRACTABLE` moved $2.02 → $2.63 → $0.32 across three polls (pay ≈$3.99/set → floor $4 on a 5-leg all-bucket + ladder cover); at 1000 bps **no executable lock**. Exactly S8's NO-ARB verdict, now watchable live next to the gap that generates it (PM2 at ship: 2–2.5T **+16.3pp** and still widening).

![[s5k_pm3_arb_fee0_production.png]]
*PM3 from the production payload at 13:36 CET, fee 0 bps: amber dust verdict, the 5 SELL legs with top-of-book sizes, pay $3.993/set → locked floor $4, EXTRACTABLE $0.32 over 47 sets.*

![[s5k_pm3_arb_fee1000_production.png]]
*Same poll at 1000 bps declared fee: neutral "no executable lock" — the declared taker fee alone erases the lock.*

**Tests:** 2 new (PM3 structural render: card under PM2, BUY/SELL legs with prices, fee formula stated, both variants + toggle present, `--fee-bps` default honored, verdict chip class tracks `investable` with green-only-if-investable pinned; degradation: `arb=None` → unavailable placeholder, no chip, PM1/PM2 unaffected, payload carriage threads through `build_ws_payload`). **Monitor suite 100; all SPCX suites 191 green** (deterministic order). Production relaunched 13:36 CET: `[arb] PM3 armed: 16 ladder + 7 bucket legs`, backfill clean (1404 polls, 0 skipped, no hole warning), both variants verified in the live websocket payload, browser-verified toggle persistence across live poll re-renders.

**Freeze.** FROZEN, absolutely — next change after the close, in the postmortem.

## 11. Decision and next step

- **Gate outcome:** Block S5 is **built, tested, live-verified**. The −$3.3 ambiguity is closed (bug, not convention — use mean−offer). No edge claim is made here: at the 06-10 poll the bucket mispricing is gone (+1.5pp) and the perp-vs-crowd gap is ±$4 — this is a **measurement instrument**, consistent with the thread's merits-a-measurement-loop posture, not a signal.
- **Caveats:** quoted PM spreads remain implausibly tight as a liquidity proxy (audit caveat — no depth weighting without book capture; depth exists in the CLOB response and could be added later if a decision needs it); mode/extreme tails inherit the PCHIP shape fragility and are flagged on every print; the No-IPO leg (0.2c) is excluded from the bucket renormalization by convention.
- **Next:** nothing left to build. Human actions per the runbook — create the free Alpaca keys + run the 1-minute AAPL feed check during US hours (§7), start the watch Thursday evening, log Friday end-to-end, then fold the poll log into the postmortem note.
