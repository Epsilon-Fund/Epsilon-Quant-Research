# STEP I — the market panel, rebuilt for research   [DONE · all gates passed]

Written by Claude Code, 2026-08-30. Three corrections, then I1 (Plotly) → I2 (market panel) → I3 (markout) → I4 (NegRisk). Step H (loader/manual/notebook/audit) kept. `book` (F) and R2 publish (G) not started. Rule 2 held — the dashboard reads only `epsilon_data`; where panels needed more I extended the loader (`markout`, `negrisk_sum`) and documented it. Code committed to `alvaro`; nothing under `data/` committed; nothing pushed.

## The hard stop — `side` convention (verified, no stop needed)
`trades.side` is the **TAKER (aggressor) side.** Asof-joined ~200k trades/universe to the prevailing touch: **BUY prints sit at the ask** (politics median pos +1.00, 77% at/above ask; esports +1.00, 96%), **SELL at the bid** (median −1.00, 95%/89% at/below bid). Unambiguous, consistent across both universes. So a taker BUY means the **maker sold** (short); a taker SELL means the **maker bought** (long). Markout is computed from the maker's perspective with `maker_sign = -1 for BUY, +1 for SELL`; **negative markout = the resting quote was adversely selected.** Documented in the README, in `markout()`'s docstring, and on the panel itself.

## Three corrections
1. **NegRisk panel now exists** (I4) — Step H had none despite the claim; `stepH.md` corrected.
2. **Stale-book figure corrected** — esports `near_half` = **21,082** (politics 440; 21,522 was both combined); of esports, **20,764** within ±1h of settlement, 196 early, 122 no `closed_time`. Fixed in `stepH.md` and the cohort view now shows the split.
3. **NegRisk measured correctly** — summing per-candidate time-medians is invalid; the panel and the analysis now sum **at a common timestamp** via `load_event()`. `stepH.md` corrected.

## Loader extensions (documented, in `epsilon_data`)
- `markout(ref, horizons=(10,30,60))` → per-trade maker-perspective markout (sign convention above).
- `negrisk_sum(event_slug)` → instantaneous YES-sum time series + `n_live`, `n_captured`.

## I1 — Plotly interactivity (Explore panels)
Shared, linked x-axis across all stacked subplots; range-selector buttons (1h/6h/1d/1w/all) + range slider; unified hover (`hovermode="x unified"`); **y autoranges to the data** (a 97–100¢ market fills the panel — the core screenshot fix); a **linear-¢ / logit price toggle** (logit turns 0.99 vs 0.999 into visible distance); a **window control** (all/1w/1d/6h/1h) that reloads the slice so the y-axis and the stat row both describe the visible window (stated on screen); `Scattergl` (WebGL) above 20k points (max ~82k L1/token, no downsampling needed).

## I2 — the market panel (one screen, shared time axis)
- **A · price + touch:** best bid/ask as a shaded band with mid inside it (the band *is* the tradeable spread — previously only mid was shown), the complement side's mid as a second line, and trade prints coloured by side, sized by `size`, with hover (price/size/side/time).
- **B · volume:** buy vs sell size per bucket as a diverging bar (buys up, sells down); bucket adapts to zoom (1m→1h).
- **C · order-flow imbalance** — the "are we being run over" panel: per-bucket (buy−sell) bars plus a cumulative signed-volume line.
- **D · spread:** step line (hv), cents.
- **E · short-term realised vol:** rolling std of 1-min mid returns (1/5/15-min window选).

## I3 — markout (adverse selection)
Per-trade maker markout at Δ ∈ {10,30,60}s: distribution (BUY vs SELL overlaid), mean per side, and a time series with a rolling mean so a spike of toxic flow is locatable — all for the visible window, with the `%` of negative-markout fills. **Finding on a control market:** the busiest politics token shows **positive** maker markout (+0.24¢ at all horizons) — i.e. benign, uninformed flow (the maker would have gained). The sign convention therefore reads sensibly (negative would flag toxic flow).

## I4 — NegRisk panel + the corrected cross-event finding
Panel: every candidate's YES mid on one shared axis, the **instantaneous YES-sum** drawn as a line with a reference at 1.0, plus captured-candidate count and the live sum for the window.

**Corrected distribution across the politics NegRisk events** (`_reports/stepI/negrisk_sums.parquet`):
- `events()` returns **273** politics `neg_risk` event_ids; **173 have ≥2 captured candidates** (the summable set — this reconciles with Cowork's 173; the extra 100 are single-market events).
- Corrected **instantaneous** YES-sum (≥2-candidate events): **median 0.995** — the method now centres on 1, versus the invalid sum-of-medians. But a right tail remains: 95th pct **2.50**, max **4.53**; 28 events > 1.05, 48 < 0.95.
- **The right tail is real and worth chasing, not pure method artefact.** The worst events are the **Elon-Musk tweet-count ranges** (e.g. Aug 14–21: 23 candidates, **1¢ median spread — tight books**, per-candidate median mids of 0.36/0.31/0.28/… summing to **2.35**). Two things stack here: (a) a genuine over-1 sum with tight books and many candidates — possibly the event bundles non-mutually-exclusive markets (cumulative "over N" thresholds rather than disjoint ranges), which would explain a legitimate sum > 1 and should be checked against Gamma's event definition; and (b) my `negrisk_sum` **forward-fills each candidate's last mid**, so once-active candidates that went quiet still contribute a stale value at later timestamps, inflating the aligned sum further (this event's ffilled instantaneous median is 4.49 vs 2.35 by medians). Tight two-sided binary events behave: Fed-decision events sum to **1.00–1.01**.
- **Caveat now stated on the panel and in the docstring:** the YES-sum rests on `mid=(bid+ask)/2` and on ffill of quiet candidates; a future refinement (best-bid instead of mid, or a liveness window on `n_live`) would sharpen it. It is honest as a diagnostic, not yet a clean arbitrage signal.

## What the panels revealed about the data
- **The imbalance panel makes adverse selection legible** — on liquid politics the cumulative-signed line and mid move together but markout stays positive (uninformed flow); this is exactly the "one-sided volume + price following" picture the operator wanted to watch for.
- **NegRisk sum-to-1 is only a clean check for tight two-sided binary events.** For multi-candidate range/threshold events it is dominated by mid-overstatement and ffill-staleness — a caution before anyone treats "sum ≠ 1" as an arbitrage.
- The **173-vs-273** reconciliation confirms Cowork's count referred to ≥2-candidate events.

## Choices where the instruction was silent
- Installed **plotly 7.0** into the research `.venv` (needed for I1; matplotlib kept for the static audit charts).
- **Window control** in Streamlit (server-side) to make "stats cover the visible window" deterministic, with Plotly's range-selector/slider for finer client-side zoom on top; y auto-fit comes from the window reload (manual x-zoom does not rescale y — a Plotly limitation, stated).
- `negrisk_sum` uses ffill alignment (via `load_event`); its inflation caveat is documented rather than silently "fixed", since the right fix (liveness/best-bid) is a design choice for the operator.

## Gates — none tripped
`side` unambiguous (no stop). Dashboard AppTest smoke: landing / Audit / Explore(Market+Markout+NegRisk) / logit+window / esports non-NegRisk — **all 0 exceptions**. Anti-drift and loader tests still pass. Reconciliation still holds (101,051,502 l1 / 7,227,528 trades).

## Then STOP
`reports/stepI.md`, `STATUS.md`, `LOG.md` written; `stepH.md` corrected. Code committed to `alvaro`, not pushed. `book` not started; nothing to R2. Operator reviews the rebuilt panels.

**To run it:** `cd polymarket/research && .venv/Scripts/streamlit run dashboard/app.py`
