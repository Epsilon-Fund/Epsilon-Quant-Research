# Dashboard walkthrough — `dashboard/app.py`

**Audit part A** · 2026-09-01 · repo `alvaro` @ `9810c2b` · data root `polymarket/research/data/research_v1`
Screenshots: `./screenshots/` · capture logs: `screenshots/_capture_log*.json`

---

## 0. Getting it running — what actually happened

The documented command is:

```bash
cd polymarket/research && .venv/bin/streamlit run dashboard/app.py
```

Two things you need to know before it works:

1. **`EPSILON_DATA_ROOT` must be exported in the shell.** Putting it in `.env` does nothing — nothing in `epsilon_data/` or the dashboard calls `load_dotenv()` (see `01_digest.md` §8.4). The command that works:
   ```bash
   cd polymarket/research && EPSILON_DATA_ROOT="$PWD/data/research_v1" .venv/bin/streamlit run dashboard/app.py
   ```
   Without it the loader falls back to `config.py:10`'s default, which happens to be the same path here — so it works by luck on this machine, and would not on a machine where the data lives elsewhere.

2. **Port 8501 was already occupied** by a streamlit process started at **12:03 today** (PID 24112, same `dashboard/app.py`), roughly 8 hours before the data was fetched to this machine at 19:43. Streamlit does not fail loudly — it logs `Port 8501 is not available` and exits, which is easy to miss when backgrounded. I left that process alone and ran mine on **8502**. Worth checking whether that stale instance is yours, Alvaro's, or a leftover: it is almost certainly serving a session that predates the data.

Startup was clean after that: **HTTP 200 in ~1 s**, no tracebacks, and **zero browser console errors or page errors** across the entire capture session (both `_capture_log*.json` files report `browser_errors: 0`).

---

## 1. Finding a market — and the one thing that's broken

### ⚠️ `fed july` returns "No match."

The landing page says *"Search a market in the sidebar (try `fed july`)"*. The search box placeholder is `e.g. fed july`. `HANDOVER.md:51` says `search "fed july"`. `epsilon_data/README.md:9` uses `ed.search("fed")`.

**Typing `fed july` returns "No match."** — screenshot `00_landing.png` → the state after search is visible in the first capture attempt.

Why: `search()` is a **literal contiguous substring** match (`catalog.py:56-58`, `regex=False`) across `event_title`, `question`, `market_slug`, `event_slug`. The string `"fed july"` appears in none of them — the event is `fed-decision-in-july-181` and the question is *"Will there be no change in Fed interest rates after the July 2026 meeting?"*. There is no field where `fed` is immediately followed by ` july`.

Verified directly:

| query | rows |
|---|---:|
| `fed july` | **0** |
| `fed` | 40 |
| `july` | 50 |
| `fed-decision` | 20 |
| `Fed interest` | 4 |

This is the first thing a new person does, in the one place the docs point them. **It fails.** The fix is one line — split the query on whitespace and AND the terms — but as shipped, the documented example is wrong in three files plus the app's own landing copy and placeholder.

Secondary annoyance: `search("fed")` returns 40 tokens of which **14 are esports** — Counter-Strike matches in the *"TESFED League"* and *"Fire Flux Esports"*, which contain the literal substring `fed`. They sort **before** the real Fed markets (table order, not relevance), so the first thing you see under a `fed` search is Counter-Strike. Not a bug, but it makes the documented entry point feel broken even when it works.

### What does work

- Search `fed-decision-in-july` → 5 matches → pick one → **Open →**. (Note: you must press Enter *or* click away for Streamlit to apply the text input; the "Press Enter to apply" hint is easy to miss.)
- Or use the sidebar's **"Or browse the busiest:"** buttons — 15 markets by trade count.
- Or **Browse the tree** expander: universe → event → market.

> **Caveat on "browse the busiest":** the top entry is *"Will Ultra Prime win the LPL 2026 season? · 115890t"*, which is a degenerate market (§4). The busiest-by-trade-count ranking is dominated by sub-cent esports longshots, not by markets you'd want to look at first.

---

## 2. Explore vs Audit

The **Mode** radio in the sidebar switches between two panel sets. `panels/_base.py:44-46` filters the registry by `section`.

| | Explore | Audit |
|---|---|---|
| scope | one selected market | the whole dataset |
| needs a market? | yes — panels declare `needs="market"` | no |
| layout | **tabs** (`app.py:140`) | **stacked with dividers** (`app.py:114-116`, `:144-146`) |
| panels | Market, Markout, NegRisk | Reconciliation, Coverage calendar, Distributions, Outliers, Activity by time, Stale-book cohort |
| window control | yes — `all / 1w / 1d / 6h / 1h`, linear/logit, vol window | none |

Two things worth knowing:

- **A selected market stays selected when you switch to Audit.** `app.py:117-124` renders the market header (and any open audit result) above the dataset-wide panels. So "Audit mode" is not a clean dataset-only view — see `20_audit_mode_full.png`, where the LPL market header sits on top of Reconciliation.
- **Adding a panel is adding a file.** `panels/__init__.py:13-16` auto-discovers any module not starting with `_`; `app.py` names no panel. This part of the design is genuinely good.

---

## 3. The Market panel — the five stacked subplots

Screenshot: `11_politics_explore_0_market.png` (Fed July) and `11_esports_explore_0_market.png` (LPL).
Built in `panels/market.py:14-67`, one `make_subplots(rows=5)` on a **shared, linked x-axis** — zoom one, all five move.

| # | subplot | what it draws | how to read it / when to be suspicious |
|---|---|---|---|
| 1 | **price · bid–ask band · trades** | side-A mid (blue), side-B mid (orange, from `load_pair`), a shaded bid–ask band, and trade prints as dots sized by trade size, green=BUY / red=SELL | The band is the *tradeable* spread. Prints should sit inside or on it. Dots outside the band = trades at prices the touch never showed — see the audit's "trades vs quotes" check. Remember B's mid is `1 − A`'s by construction (`01_digest.md` §7 item 5), so the two lines mirroring is **not** confirmation of anything |
| 2 | **volume (buy↑ / sell↓)** | resampled contract volume, buys up, sells down; bucket auto-scales (`_data.py:47-48`: 1min → 1h by span) | A one-sided wall means one-way flow. On the LPL market this is a solid green block with essentially no red — 579,445 buy vs **8** sell contracts |
| 3 | **order-flow imbalance & cumulative signed** | grey bars = net (buy − sell) per bucket; purple line = cumulative signed volume | The "are we being run over" view. A **monotonic** cumulative line = pure accumulation with no two-way flow. Straight-line ramps (LPL) mean one participant is lifting continuously |
| 4 | **spread** | `spread_c` in cents, step-drawn (`shape="hv"`) | Should be ≥ 0. **On the Fed market the y-axis runs to −10¢** — because 3 rows in that token have `best_bid > best_ask`. Dataset-wide there are **24 such rows out of 101,051,502** (min spread −65.9¢). Negligible in volume, but real, and undocumented in the README traps |
| 5 | **realised vol (mid returns)** | rolling std of 1-minute mid pct-changes × 100, window from the "vol win (min)" selector | Spikes mark repricings. On a deduped L1 tape the 1-min resample forward-fills (`market.py:62`), so a flat stretch means "touch didn't move", not "no activity" |

**Above the chart**: five metrics — median spread, mid range, trades, buy vol, sell vol.

### Defects observed in this panel

1. **Metric values truncate at ≤1600px viewport.** At 1600px wide, "mid range" rendered as `58.5—96.…` and "buy vol" as `12,769,5…`. At 1800px they fit. `st.metric` doesn't shrink or wrap — on a laptop screen two of the five headline numbers are unreadable.
2. **Subplot titles collide with the plot above them.** `vertical_spacing=0.03` across 5 rows with per-row titles (`market.py:15-19`): "volume (buy↑/sell↓)", "spread" and "realised vol" all overlap the traces above. Visible in every market screenshot.
3. **The range-selector buttons (`1h 6h 1d 1w all`) render on top of the spread subplot.** They're attached to row 5 (`market.py:66`) but paint into row 4's area.
4. **The "trades" metric disagrees with the header.** Header says `tr 26,419`; the metric says `26,409`. The window defaults to `end = l1.ts.max()` (`app.py:132`), so the 10 trades that occur after the token's last touch-move are silently outside "all". "all" is not all.

---

## 4. The two markets I walked, and what they show

### Politics — *Will there be no change in Fed interest rates after the July 2026 meeting?*
`fed-decision-in-july-181` · 26,419 trades · 29,755 L1 events · median spread 0.6¢ · RESOLVED · `check4=no_convergence`

A healthy, liquid, two-sided market. Mid ranges 58.5–96.5¢, both sides mirror cleanly, markout is **positive** (+0.244¢ mean at 30s, only 8% of fills negative) — i.e. resting liquidity on this market was *paid*, not picked off. The NegRisk tab shows 5 captured candidates summing to a median **1.004** — textbook. This is the market to show someone first.

### esports — *Will Ultra Prime win the LPL 2026 season?*
`lol-lpl-2026-season-winner` · **115,890 trades** · **510 L1 events** · median spread 0.2¢ · mid **0.001** · open

This is the **busiest esports market in the dataset by trade count** and it is degenerate:

- 115,890 trades against only **510 touch-moves** — 227 trades per book update.
- Buy volume **579,445** contracts, sell volume **8**.
- Price pinned at **0.1–0.7¢** the whole time; the complement side has **zero trades**.
- Cumulative signed volume is a perfectly straight ramp to 600k.

Someone accumulated ~580k contracts of a sub-cent longshot in ~2.5 days with no two-way flow. Whether that's volume farming, a wash pattern, or a genuine lottery-ticket buyer, **it is not a market whose microstructure tells you anything** — and it is the first entry in the dashboard's "browse the busiest" list.

**`audit_market()` passes it green — "looks fine, all checks passed."** That is the most useful thing I found in Part A: see §6.

---

## 5. Markout panel

Screenshot: `11_politics_explore_1_markout.png`. Source `panels/markout.py`, loader `tape.py:48-73`.

Answers: *"if I had been the resting quote, would I have been run over?"*

- `trades.side` is the **taker** side. Maker sign is inverted: taker BUY ⇒ maker sold ⇒ `sign = −1`.
- Markout at horizon Δ = `maker_sign × (mid[t+Δ] − price)`. **Negative = adversely selected.**
- Horizon selector: 10s / 30s / 60s. Three metrics (all / BUY / SELL), a distribution histogram clipped to ±5¢, and a time series with a rolling-50 mean.

**Fed July result:** +0.244¢ mean at 30s, +0.227¢ on BUY, +0.284¢ on SELL, 8% of fills negative. Benign flow.

**Caveat you must carry (from `01_digest.md` §8.8):** the mid at `t+Δ` comes from `merge_asof(direction="backward")` with **no staleness bound**. Under L1 dedup semantics that's usually correct — no row means the touch didn't move. But where a token's tape has genuinely ended, markout still returns a number computed from a *pre-trade* mid instead of `NaN`, and the panel's own empty-state guard (`markout.py:24`) therefore almost never fires. On the three busiest politics tokens, 6.6% / 68.1% / 88.0% of trades have no new L1 row inside a 60s horizon.

---

## 6. The audit button — what `audit_market()` actually checks

Screenshots: `12_politics_audit_market.png` (🟡 worth a look), `20_audit_mode_full.png` top (🟢 looks fine).
Source: `epsilon_data/audit.py:54-213`. **It never writes** — `write_exclusion()` is the only writer, and only on an explicit click (`app.py:98-103`). Cost measured: **~11–14 s** per call end-to-end in the browser (~1.7 s of that is the loader; the rest is Streamlit rerun + Plotly).

Seven checks, each `ok` / `note` / `bad`. Verdict = worst level: any `bad` → "recommend excluding"; any `note` → "worth a look"; else "looks fine".

| check | what it tests | thresholds |
|---|---|---|
| **identity** | `check1_roundtrip`, `check2_pairing`, `check5_indep` not False; complement present; reports `identity_status` + `check4_status` | any False ⇒ `bad` |
| **value sanity** | crossed bid>ask, mid outside (0,1), mid never moves, max 1-step jump >50¢; reports spread=0 and =100¢ fractions as *informational* | **fraction-based**: >0.5% ⇒ `bad`, else `note` (`audit.py:98-109`) |
| **pair sum≈1** | median \|A+B−1\| over the aligned pair | <0.02 ok, <0.05 note, else bad |
| **continuity** | gaps >1h (reported, not penalised) and gaps >12h that don't overlap the known outage | any unexplained >12h ⇒ `note` |
| **trades vs quotes** | trades printing >1¢ outside the prevailing touch | <2% ok, <10% note, else bad |
| **volume shape** | one print >50% of volume; duplicate transaction hashes | either ⇒ `note` |
| **resolution** | `check4_status`: `inverted` ⇒ bad; `near_half`/`no_convergence` ⇒ note | |

Then it prints the **exact `exclusions.csv` lines** you'd need, and recommends **market scope, not token scope** — because excluding one token orphans its complement and breaks pair views and NegRisk sums (`audit.py:200-202`). That reasoning is sound and worth preserving.

### Fed July verdict: 🟡 worth a look
- value sanity `note`: 3 crossed bid>ask (0.01%)
- continuity `note`: **135 gaps >1h; 11 unexplained gaps >12h; max gap 67.2h**
- resolution `note`: resolved but `no_convergence`
- pair sum ok: median \|A+B−1\| = **0.000**, worst 0.000 over 14,723 points
- trades vs quotes ok: 17/26,419 outside touch

> **Wording trap:** "11 unexplained gaps >12h" sounds like missing capture. It isn't. `l1` is deduped to touch-moves, so a "gap" means *the touch did not move for 67 hours* — entirely possible on a quiet resolved market that still trades. The check is measuring quote staleness and calling it continuity. Worth renaming before someone reports a capture incident that never happened.

### ⚠️ The audit tool has no plausibility check
The LPL market — 115,890 trades, 579,445 buy vs 8 sell contracts, price pinned at 0.1¢, one-way monotonic accumulation — returns **🟢 "looks fine — all checks passed"**, including `volume shape: no dominant print / dup hashes`.

That is correct by its own definitions: the volume-shape check only looks for *a single print* >50% of volume, or duplicate tx hashes. 115,890 evenly-sized one-sided prints trip neither.

**Every check in `audit_market()` is about series integrity — is the tape self-consistent — and none is about market plausibility.** A one-sided-flow / near-zero-price / trades-per-touch-move check would cost a few lines and would catch the single most anomalous market in the dataset. Right now the tool's green light means "the data is internally consistent", which a reader will very easily mistake for "this market is fine to analyse".

---

## 7. Audit-mode panels

Screenshots `20_audit_mode_full.png`, `21_audit_*.png`.

| panel | reads | what it showed |
|---|---|---|
| **Reconciliation** | `ed.reconciliation()` | Both identities **MATCH**: 101,051,502 L1 rows, 7,227,528 trades |
| **Coverage calendar** | `ed.coverage()` | 64 rows/universe, hour × date heatmap. Red = true gap: the 06-19 ramp (h00–11), the 06-22→23 outage, the 08-21 tail (h21–23). Politics shows **no grey at all** — it has zero quiet hours; all 24 quiet hours across 14 days are esports (verified) |
| **Distributions** | `cat_all()` | 4 histograms: median spread ¢ (≤50), trades/token (≤500), days alive, median mid. **Unlabelled 91%-esports mix, silent clipping** — see `01_digest.md` §8.5. "days alive" is dominated by a huge 0–2 day bar (esports matches); "median mid" is U-shaped with spikes at 0 and 1 |
| **Outliers** | `cat_all()` | 6 tabs. "widest spreads" is all esports handicap markets at **99.0¢** median spread (one-sided books). The `path` column is cut off at the right edge in every tab |
| **Activity by time** | `ed.activity_by_time(u)` | Two heatmaps, weekday × UTC hour. esports concentrates ~09–15 UTC; politics is flatter with a weekday spike ~14–15 UTC. Genuinely informative |
| **Stale-book cohort** | `cat_all()` | esports `near_half` = **21,082**; **20,764 (98.5%)** had their last observation within **±1h of settlement** and were *still quoting ~0.5 on a decided outcome*; 196 stopped early; 122 have no `closed_time` |

The stale-book panel is the strongest thing in the app. 20,764 tokens watched to the wire, still at ~50¢ on a resolved outcome, is either a large artefact or a large amount of money — and the panel correctly refuses to say which.

---

## 8. Performance — what felt slow

Wall time measured in-browser (includes Streamlit rerun + Plotly render; subtract ~3s of settle margin for the true figure):

| action | measured | verdict |
|---|---|---|
| app cold start | ~1 s to HTTP 200 | fine |
| open a market | 3–4 s | fine |
| Market tab (politics, 29,755 L1 rows) | **3.7 s** | fine |
| Markout tab (politics) | **8.1 s** | noticeable |
| NegRisk tab (politics, 5 legs) | **10.5 s** | noticeable |
| **NegRisk tab (esports LPL, 14 legs)** | **24.7 s** | **feels broken** |
| `audit_market()` (politics) | **14.0 s** | **feels broken** |
| `audit_market()` (esports) | **12.8 s** | **feels broken** |
| Audit mode switch (all 6 panels) | ~4 s | fine |

The NegRisk and audit paths are the problem, and the cause is structural: `load_event()` calls `read_token_tape()` **once per token in a loop** (`tape.py:86-90`) — 14 separate DuckDB connections and queries for the LPL event, 28 for a 14-market event. `audit_market()` similarly loads L1 + trades + the full aligned pair (`audit.py:82`, `:121`).

Neither shows a progress indicator beyond Streamlit's spinner, and `@st.cache_data(show_spinner=True)` on `event_wide`/`negrisk_df`/`audit_result` means the *second* view is instant — but the first is a 25-second stare at a spinner with no explanation. Batching the per-token reads into one `IN (...)` query would fix both.

---

## 9. Defect list from Part A (for Alvaro)

| # | severity | what | where |
|---|---|---|---|
| 1 | **high** | `search("fed july")` — the documented example query in 4 places — returns **0 results**; search is literal-substring only | `catalog.py:56-58`; docs `HANDOVER.md:51`, `README.md:9`, `app.py:112`, `app.py:38` |
| 2 | **high** | `audit_market()` passes a market with 579,445 buy vs 8 sell contracts as "looks fine" — no plausibility/one-sided-flow check exists | `audit.py:164-173` |
| 3 | **med** | NegRisk tab **ignores the window selector** while the caption promises "stats & panels cover this window" — Market tab showed 07-31→08-03, NegRisk tab showed 07-05→08-16 on the same screen | `negrisk.py:18-19` vs `app.py:138` |
| 4 | **med** | NegRisk YES-sum **ramps 0→1 as legs are first captured**, which reads as a huge arb in the left third of the chart; it's the ffill/coverage artifact, not a signal | `11_esports_explore_2_negrisk.png` |
| 5 | **med** | 25 s to render the NegRisk tab, 12–14 s for the audit button; per-token query loop | `tape.py:86-90`, `audit.py:82` |
| 6 | **med** | NegRisk panel's guard text says the sum-to-1 view "applies to NegRisk events (**politics**)" — but it renders for esports NegRisk events too, and did here | `negrisk.py:13` |
| 7 | **low** | Two of five Market metrics truncate to `…` at ≤1600px viewport | `market.py:87-92` |
| 8 | **low** | Subplot titles overlap traces; range-selector buttons paint over the spread subplot | `market.py:15-19`, `:66` |
| 9 | **low** | "trades" metric (26,409) ≠ header (26,419) — the "all" window ends at the last L1 event, dropping later trades | `app.py:132-134` |
| 10 | **low** | Audit "continuity" check calls a 67h quote-staleness stretch an "unexplained gap", which sounds like lost capture | `audit.py:145-147` |
| 11 | **low** | 24 crossed-book rows dataset-wide (spread to −65.9¢) are undocumented in the README traps; the spread subplot silently plots negative | `l1` data; `README.md:115-134` |
| 12 | **info** | Stale streamlit on port 8501 from 12:03 today, predating the data fetch | host process PID 24112 |

**Nothing blank-rendered, nothing errored, and there were zero browser console errors.** The app is in better shape than the doc set around it.
