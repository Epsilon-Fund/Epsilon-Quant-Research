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

## 11. Decision and next step

- **Gate outcome:** Block S5 is **built, tested, live-verified**. The −$3.3 ambiguity is closed (bug, not convention — use mean−offer). No edge claim is made here: at the 06-10 poll the bucket mispricing is gone (+1.5pp) and the perp-vs-crowd gap is ±$4 — this is a **measurement instrument**, consistent with the thread's merits-a-measurement-loop posture, not a signal.
- **Caveats:** quoted PM spreads remain implausibly tight as a liquidity proxy (audit caveat — no depth weighting without book capture; depth exists in the CLOB response and could be added later if a decision needs it); mode/extreme tails inherit the PCHIP shape fragility and are flagged on every print; the No-IPO leg (0.2c) is excluded from the bucket renormalization by convention.
- **Next:** nothing left to build. Human actions per the runbook — create the free Alpaca keys + run the 1-minute AAPL feed check during US hours (§7), start the watch Thursday evening, log Friday end-to-end, then fold the poll log into the postmortem note.
