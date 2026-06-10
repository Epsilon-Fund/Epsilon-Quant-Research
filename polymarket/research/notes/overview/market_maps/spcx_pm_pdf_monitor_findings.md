---
title: "SPCX Live Crowd-Distribution Monitor — Polymarket 16-Strike Survivor Fit + Perp Gap, Polled Every 30–60s (Block S5 build + the −$3.3 EV reconciliation)"
tags: [spacex, spcx, ipo, polymarket, hyperliquid, pchip, pdf, monitor, findings, tooling]
created: 2026-06-10
audience: "Justin + Alvaro on listing day (the PEAK trigger + PM tail-sell screens); Cowork/Claude Code sessions extending the Friday stack"
status: "built + tested 2026-06-10 (17 new tests green, 44 with the calc suite); live-verified against the real CLOB/HL; −$3.3 EV line reconciled (verdict: payoff-convention bug — do not import)"
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
- **Post-cross (D5):** add `--spot <listed>` (or rerun with it). The "spot sits at the crowd's P-xx" line plus the >$2.4T/>$3T tail repricing is the shared PEAK input; the bucket gap table is Alvaro's NO-expression screen, live.
- **Afterward:** the parquet poll log feeds the `spcx_listing_postmortem` note (gameplan § Post-day) — every poll's full ladder/bucket/perp state is preserved.

## 7. Decision and next step

- **Gate outcome:** Block S5 is **built, tested, live-verified**. The −$3.3 ambiguity is closed (bug, not convention — use mean−offer). No edge claim is made here: at the 06-10 poll the bucket mispricing is gone (+1.5pp) and the perp-vs-crowd gap is ±$4 — this is a **measurement instrument**, consistent with the thread's merits-a-measurement-loop posture, not a signal.
- **Caveats:** quoted PM spreads remain implausibly tight as a liquidity proxy (audit caveat — no depth weighting without book capture; depth exists in the CLOB response and could be added later if a decision needs it); mode/extreme tails inherit the PCHIP shape fragility and are flagged on every print; the No-IPO leg (0.2c) is excluded from the bucket renormalization by convention.
- **Next:** nothing to build. Human action per the runbook — start the watch Thursday evening, log Friday end-to-end, then fold the poll log into the postmortem note.
