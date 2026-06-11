---
title: "SPCX IPO Convergence Trade — Units-Matched Basis + Liquidation-Survival Calculator"
tags: [spacex, ipo, spcx, hyperliquid, ventuals, convergence, basis, liquidation, calculator, findings]
created: 2026-06-09
audience: "Cowork/Codex sessions deciding go/no-go on the long-IPO / short-perp SPCX convergence trade"
status: "offline gate built + tested; SPCX venue numbers are a 2026-06-09 00:09 UTC live snapshot; Cerebras lifecycle tape pulled 2026-06-09 (HL 15m perp + Yahoo 5m spot); Block S1 (hedge grid + basis-decay fit + pre-hedge timing rule) added 2026-06-10 17:00 UTC — pre-hedge NOT armed at 06-10 marks, Friday allocation gate is the rule — refresh before trading"
---

# SPCX IPO Convergence Trade — Units-Matched Basis + Liquidation-Survival Calculator

> Hub: [[spacex_ipo_market_map_handoff]] · [[COWORK]] · [[POLYMARKET_BRAIN]]
> Companion: [[spacex_ipo_coworker_addendum]] · Table terms: [[polymarket_table_dictionary]]
> Script: `polymarket/research/scripts/spcx_convergence_calc.py` · Tests: `polymarket/research/tests/test_spcx_convergence_calc.py`

## Plain-English Summary

- **The trade.** Buy SpaceX IPO Class A shares at the fixed **$135** offer (long leg), short a Hyperliquid SpaceX pre-IPO perp (short leg), and hold **both** legs to the perp's first-trading-day settlement to capture the convergence basis. If units match and you hold to settlement, the profit is *locked the moment both legs are on* — it does not depend on which way SpaceX trades.
- **What this note delivers.** A single, pre-registered, unit-tested CLI/calculator that turns Friday-morning inputs into a go/no-go. It normalizes every venue price into the same unit (FDV and $/IPO-share), decomposes P&L by **hedge ratio** (you need not cover 1:1) and by **leverage** (you need not lever), answers the COWORK forcing question, and stress-tests the short against a Cerebras-style melt-up. It also ships a localhost `--watch` live monitor.
- **Live read (2026-06-09 00:09 UTC snapshot).** Both perps trade **rich** to the $135 offer: `xyz:SPCX` mark **$159.92** ⇒ units-matched basis **$24.92/IPO-share**; `vntl:SPACEX` mark **2027.1** (= $2.027T) ⇒ units-matched basis **$20.03/IPO-share**. The gross basis is large and positive.
- **The one-line takeaway (revised).** The basis is real, and the earlier "NO-TRADE" was a **leverage artifact, not a property of the trade**. Sized as the realistic base case — **fully FDV-hedged (h=1) and UNLEVERED (1×)** — the short's liquidation buffer is **+82% (xyz) / +71% (vntl)**, far beyond the +39% Cerebras-high analog, so both legs are **TRADE-ABLE** (net locked ~**$24.9k / $20.3k** on 1,000 long shares, ROC ~**8.4% / 7.0%** over ~3 days). Leverage is what kills it: at 2× the +39% melt-up liquidates; at 5×/3× a +9%/+3% wiggle does. **Don't lever the short.**
- **Two new realistic dimensions.** (1) **Hedge ratio h** (you need not cover 1:1): P&L splits into a **locked** part `h·N·basis` (settle-invariant) plus a **directional residual** `(1−h)·N·(close−135)` on the uncovered fraction — under-hedge (h<1) is a net-long tilt, over-hedge (h>1) is a net-short tilt that *loses on a melt-up*. (2) **Unlevered/over-collateralized short**: a wide but **finite** liquidation buffer (a perp short's loss is unbounded; only margin ≫ notional is truly un-liquidatable), at the cost of tying up the full notional as margin.
- **Status.** Offline gate is **green unlevered and fully-hedged**; everything that decides whether to actually do it (real TR allocation fill, book depth at size, whether richness persists to Friday, oracle/transition behavior at listing) is **live-only**. Per the COWORK *terminus = live* rule, a green gate justifies a minimal instrumented live test, **not** a trading-system build — and a localhost **web** dashboard is declined as infra-before-signal for a ~3-day fuse (a terminal `--watch` loop is the right-sized monitor).
- **Block S1 (added 2026-06-10): hedge grid + pre-hedge timing rule.** The calculator now answers the Friday-shaped question directly: a **fill-price ($135→$162) × fill-fraction (10/25/50/100%) × margin (€2k at 1×/1.5×) hedge grid**, a **basis-decay fit** on HL hourly candles (premium bleeding ~−4.4%/day full-window, ~−11%/day last week — half-life ~16d), and a **pre-registered pre-hedge timing rule** ("hedge X at node Y iff net basis ≥ Z"). Verdict at 06-10 marks (perp ~$162, basis ~$27/sh): **do NOT pre-hedge before allocation** — the margin ceiling is only ~14–21 shares, the expected fill covers it, and 2 days of basis decay (~$1.8/sh) is far cheaper than the naked-short melt-up tail; **the rule is the Friday gate**: hedge min(fill, ~21 sh at 1.5×) at ~8:00 CET iff live net basis > 0. Pre-hedging arms only if the perp spikes ≥ ~$171 first (Z ≈ $36/sh over a $135 fill). See § Block S1 below.

---

## Why this note exists

The [[spacex_ipo_market_map_handoff]] mapped the venues and warned, in "Do Not Make These Mistakes," against (a) comparing `vntl:SPACEX` 2059.5 to `xyz:SPCX` 173.53 without normalizing units, (b) comparing PM close contracts to a perp mark as if they settle on the same thing, and (c) treating "SpaceX is overvalued" as "short now at any price" without timing/carry/liquidation/exit analysis. This calculator operationalizes exactly those warnings into a refusable, testable gate. It is the offline screen *before* any live capital.

---

## Design — what the calculator does, in phases

The script (`spcx_convergence_calc.py`) is pure-Python in its math core (stdlib only, so the tests need no numpy/scipy), with a thin read-only `httpx` fetch for live marks and a snapshot cache. It evaluates each contract in six steps.

### Phase 1 — Units normalization (the core of the tool)

Every quote is restated to **(a) total implied FDV in $** and **(b) per-IPO-share-equivalent on the 13,075,865,175 IPO base** *before* any spread or sizing math. The two unit conventions:

- **`per_share`** (xyz:SPCX): `FDV = mark × share_base`; per-IPO-share-equiv `= FDV / IPO_base`.
- **`valuation_per_1e9`** (vntl:SPACEX): the price *is* valuation/$1e9, so `FDV = mark × 1e9`; per-IPO-share-equiv `= FDV / IPO_base`.

The **denominator ratio** `R = IPO_base / contract_base` is displayed for every contract. `R = 1` means the naive per-share gap (`mark − 135`) is correct; `R ≠ 1` means it mixes share-count conventions and is wrong. **The tool refuses to print a naive `mark − 135` gap for a valuation-unit contract** (2027.1 is not a $/share number).

> **Where the denominator actually matters (correction).** For a **per-share** perp like `xyz:SPCX`, the perp price already *is* the expected per-Class-A-share price, the same unit as the $135 offer — so the basis is simply `mark − 135 ≈ $25/share`, a **1:1 short** (one perp per IPO share) is the correct fully-hedged size, and the 13.08B share base **never enters**. An earlier draft over-applied the denominator warning to `xyz`; it does not apply there at `R=1`. The share-base normalization is genuinely necessary **only** for the **valuation-unit** perp `vntl:SPACEX`, whose mark (~2027) is *$1B-of-valuation units*, not dollars-per-share: you must divide by the share base to compare to $135 (implied per-share `= 2027 × 1e9 / 13.08B = $155`) **and** to size the short (one vntl contract moves `shares/1e9 = $13.08` per $1/share, so 1,000 long shares need **~76.5 contracts, not 1,000**). The only residual denominator for `xyz` is a different mechanism entirely — a **split / dilution / share-class** mismatch between your IPO lot and the converted/listed per-share base (e.g. the ~11.87B hypothesis), which rescales by that factor, not by a valuation-vs-per-share conversion.

### Phase 2 — Hedge ratio + P&L decomposition, gross and net

The short is sized by a FDV-anchored **hedge ratio** `h = short / FDV-neutral` (default 1). P&L splits into **LOCKED** `= h · N · (per-IPO-share-equiv(entry) − 135)` — direction-independent iff held to settlement, units match, and both legs converge to the same FDV — plus a **directional RESIDUAL** `= (1−h) · N · (close − 135)` on the uncovered fraction. Only the locked piece is netted of HL fees, funding (sign + estimate), EUR/USD on the Trade Republic long leg, and stablecoin/USDH basis; the residual is shown across a close-price grid and never counted as locked return. Capital deployed (long notional + short margin) and a locked return-on-capital are reported too. See [§ Shorting without leverage, and not covering 1:1](#shorting-without-leverage-and-not-covering-11).

### Phase 3 — Forcing question (COWORK decision rule)

"If the stock price didn't move between entry and settlement, would the position still make money?" → answered yes/no **with the number**. At h=1 the locked basis is settle-invariant, so the answer is "is net locked basis > 0?". At h≠1 the tool reports the residual slope `(1−h)·N` instead of an unconditional YES, because the position is then directional.

### Phase 4 — Path / liquidation survival (the binding gate)

Given posted margin and leverage, the tool computes the **liquidation price and the % adverse move to it**, then stress-tests a pre-settlement melt-up at **+13%, +26%, +39%** (the Cerebras open/high analogs) and reports **SURVIVE / LIQUIDATED** for each. Liquidation-before-convergence — which leaves you long-only into the spike — is treated as the **primary ruin mode**.

### Phase 5 — Capacity vs open-interest cap

Desired short notional is compared to the contract's OI cap (`xyz:SPCX` $150M per the trade[XYZ] IPOP spec; `vntl:SPACEX` $10M per the handoff) and to the current live-book OI notional, reporting the fraction of each you would be.

### Phase 6 — Verdict

**NO-TRADE** if units-matched net basis ≤ 0, **or** if any default melt-up scenario liquidates the short at the chosen size. Otherwise **TRADE-ABLE**, with the **max short notional that survives all default scenarios** = `posted_margin × max_survivable_leverage`.

---

## Worked example (read this before the tables)

Suppose you are allocated **1,000 IPO shares at $135** ($135,000 long) and you short **$135,000** of `xyz:SPCX` at mark **$159.92** with **$67,500 margin (2× leverage)**.

- **Units:** `xyz:SPCX` is per-share; we default its base to the IPO base (R = 1), so per-IPO-share-equiv = $159.92 and the units-matched basis = $159.92 − $135 = **$24.92/share**. Naive and units-matched agree because R = 1.
- **Locked basis:** $24.92 × 1,000 = **$24,920** gross; net of fees/funding ≈ **$24,890**. The forcing question answers **YES** — if SpaceX never moves between entry and settlement, you still collect ~$24.9k, because the settlement price cancels out of the matched book.
- **But liquidation:** at 2× isolated with maintenance-margin fraction 0.10 (= 1/(2×5), HL's rule for a 5×-max asset), the short is liquidated at a **+36.4%** move (price $218.07). A **+39% melt-up to $222.29 LIQUIDATES** the short before settlement. Verdict for this size: **NO-TRADE**.
- **The fix is less leverage:** the max short that survives a +39% spike at this margin is **$170,132 (≤ 1.89×)**. Below that, the trade is offline-green.

This is the whole thesis in one example: *the basis is real; surviving to collect it requires very low leverage.*

---

## Live results — 2026-06-09 00:09 UTC snapshot

> Snapshot is **dated**. The IPO listing is anticipated **June 12, 2026**; refresh marks (re-run online) the evening of June 11 / morning of June 12 before any decision. Marks below were live-fetched from the Hyperliquid `/info` endpoint (`metaAndAssetCtxs`) and cached to `data/analysis/spcx_convergence/latest.json`.

Legs assumed for the table: **1,000 IPO shares @ $135**, short notional **$135,000** (notionally matched), **2× isolated leverage**, scenarios **+13 / +26 / +39%**.

| field | `xyz:SPCX` | `vntl:SPACEX` |
|---|---:|---:|
| mark | 159.92 (per-share) | 2027.1 (valuation/$1e9) |
| oracle | 159.52 | 1855.2 |
| implied FDV | $2.091T | $2.027T |
| per-IPO-share-equiv | $159.92 | $155.03 |
| denominator ratio R | 1.0000 | 1.0000 |
| naive per-share gap | $24.92 (`mark−135`) | **REFUSED** (valuation units) |
| **units-matched basis** | **$24.92/share** | **$20.03/share** |
| FDV-neutral short count | 1,000.00 contracts | 76.48 contracts |
| naive short count | 1,000.00 | 66.60 (notional-matched) |
| over/under-size error | +0.00% | **−12.92%** (notional-match under-hedges) |
| gross locked basis | $24,920 | $20,026 |
| funding (short receives) | +$30 (72h) | +$336 (72h) |
| net basis | $24,890 ($24.89/sh) | $20,302 ($20.30/sh) |
| forcing question | **YES** | **YES** |
| liq move @ 2× | +36.4% | +28.6% |
| +39% melt-up | **LIQUIDATED** | **LIQUIDATED** |
| OI cap / current book | $150M / $99.3M | $10M / $2.77M |
| short = % of cap / book | 0.09% / 0.14% | 1.35% / 4.88% |
| settlement | **convert-in-place** (flag) | cash → IPO close |
| **verdict @ 2×** | **NO-TRADE** | **NO-TRADE** |

### Column glossary (every non-obvious column defined)

- **implied FDV** — total fully-diluted valuation the quote implies (`mark × base` for per-share; `mark × 1e9` for valuation-units).
- **per-IPO-share-equiv** — the quote restated to dollars per IPO-share on the 13.076B base; the *only* representation in which subtracting $135 is legitimate.
- **denominator ratio R** — `IPO_base / contract_base`; R = 1 means naive = units-matched.
- **naive per-share gap** — `mark − 135`; correct only for a per-share contract at R = 1, refused entirely for valuation-units.
- **units-matched basis** — per-IPO-share-equiv − 135; the real locked basis per share.
- **FDV-neutral short count** — contracts needed so the short's FDV delta equals the long's; **naive short count** is the count a 1:1 (per-share) or 1:1-notional (valuation) trader would use; **over/under-size error** is the % gap between them.
- **funding** — Hyperliquid hourly funding × notional × hours; sign preserved (a positive premium means longs pay shorts, so the short *receives*).
- **liq move @ 2×** — the adverse (price-up) % move that liquidates the isolated short at 2× given the asset's maintenance-margin fraction.
- **settlement** — `cash → IPO close` (vntl cash-settles at 4pm ET first day to basic-shares × close); `convert-in-place` (xyz may convert to a listed-equity perp rather than cash-settle).

### Read

Both perps are **rich** to the $135 offer (basis $20–25/share, ~15–18%), and **costs are immaterial** against that basis — HL fees are a few dollars and funding is *positive carry for the short* (both perps trade above oracle, so the short is paid to wait). So the forcing question is unambiguously **YES** on both: if SpaceX never moves, you still collect the basis. **Costs do not kill this trade.** What kills it is the melt-up path — see the next section. Note also the `vntl` sizing trap: a trader who "shorts $135k against $135k long" **under-hedges by 12.9%** because vntl is a valuation-unit contract, not a per-share one; the FDV-neutral short is 76.48 contracts, not 66.60.

---

## Liquidation survival across leverage

![Liquidation move vs leverage for xyz:SPCX and vntl:SPACEX](../../../data/analysis/plots/spcx_convergence/liq_survival.png)

**Chart read.** X-axis is isolated leverage (1×–5×); Y-axis is the adverse (price-up) % move that liquidates the short. The black curve is `liq_move = (1 + 1/L)/(1 + mmr) − 1`. Dashed lines are the Cerebras +13/+26/+39% analogs (red = the worst, +39%). The **green region** is where the liquidation move exceeds the worst scenario, i.e. the short survives a Cerebras-style spike — it only exists at **low leverage** (≤ ~1.9× for `xyz`, mmr 0.10; ≤ ~1.6× for `vntl`, mmr 0.167, because vntl's 3×-max gives a *higher* maintenance fraction). The blue line marks the example 2× — which sits *below* the +39% line for both, i.e. liquidated.

### Verdict by leverage regime (same legs, swept leverage)

| leverage | `xyz:SPCX` liq move | `xyz` verdict | `vntl:SPACEX` liq move | `vntl` verdict |
|---:|---:|---|---:|---|
| **1.0× (unlevered, the realistic base case)** | **+81.8%** | **TRADE-ABLE** | **+71.4%** | **TRADE-ABLE** |
| 1.5× | +51.5% | **TRADE-ABLE** (max short $170,132 ≤1.89×) | +42.9% | **TRADE-ABLE** (max short $144,772 ≤1.61×) |
| 2.0× | +36.4% | NO-TRADE (+39% liquidates) | +28.6% | NO-TRADE (+39% liquidates) |
| 5.0× / 3.0× (venue max) | +9.1% | NO-TRADE (even +13% liquidates) | +2.9% | NO-TRADE (even +13% liquidates) |

### Read

This is the decision, and it turns entirely on leverage. The basis is collectible **only if the short survives to settlement**. **Unlevered (1×) the short survives a +82% / +71% spike** — far beyond the +39% Cerebras-high analog — so the realistic base case is TRADE-ABLE. At the venues' **max leverage (5×/3×) the short is liquidated by a +9% / +3% wiggle** — catastrophic, given the Cerebras tape ran +39% in ~10 minutes and liquidated ~$2.1M of shorts at the transition. The crossover (survives the +39% analog) is **≤~1.9× for xyz, ≤~1.6× for vntl**. Because liquidation move depends only on leverage (not directly on size), the rule is: **don't lever the short; run ≤1.5×.** The OI cap is non-binding at retail size.

---

## Shorting without leverage, and not covering 1:1

The first version of this tool assumed you short *with* leverage and cover the long exactly 1:1. Both assumptions are now relaxed, because both were unrealistic and the second one hid a real error: the old default `--short-notional = shares × $135` ($135k) only buys ~844 `xyz` contracts at $159.92, **not** the FDV-neutral 1,000 — so the old default was actually an 84%-hedge reported as if fully locked. The calculator now sizes the short by an explicit, FDV-anchored **hedge ratio**.

### Hedge ratio `h` and the locked/residual split

Define `h = short_contracts / FDV-neutral_contracts`. It is anchored in **FDV-delta space**, not dollar-notional, so `h = 1` is truly direction-neutral on *both* the per-share (`xyz`) and valuation-unit (`vntl`) contracts (matching dollar-notional silently under-hedges `vntl` by ~12.9%). The total P&L then splits cleanly into two pieces — this is the algebra two independent derivations and the existing unit tests agree on:

```
total P&L  =  LOCKED              +  RESIDUAL (directional)
           =  h·N·(E_ps − 135)    +  (1 − h)·N·(close − 135)
```

- **LOCKED** `= h · N · (units-matched basis)` — direction-independent (the covered fraction's settlement price cancels). This is the only piece that is "arbitrage," and the only piece routed through the forcing question / net-of-cost gate.
- **RESIDUAL** `= (1 − h) · N · (close − 135)` — the **uncovered fraction is a directional bet**, with the *same* risk profile as buying (h<1) or short-selling (h>1) the stock outright. `h<1` is a net-long tilt; `h>1` is a net-short tilt. It is **not** locked and **not** counted as return.

### Practical example (xyz, 1,000 long shares, basis $24.92/share)

| hedge ratio `h` | what it is | locked P&L | residual at close=$222 (Cerebras high) | total at $222 |
|---:|---|---:|---:|---:|
| 0.0 | no short — pure long | $0 | +$87,000 | +$87,000 |
| 0.5 | half-hedged, net **long** | $12,460 | +$43,500 | +$55,960 |
| 1.0 | fully FDV-hedged (**locked**) | $24,920 | $0 | **$24,920 (flat)** |
| 1.5 | over-hedged, net **short** | $37,380 | −$43,500 | **−$6,120 (loss)** |

![Total P&L vs settlement close, by hedge ratio](../../../data/analysis/plots/spcx_convergence/hedge_pnl_by_close.png)

**Chart read.** X-axis is the first-day settlement close ($/IPO-share); Y-axis is total P&L ($000s). The flat green line is `h=1` (locked — pays ~$25k regardless of close, i.e. real arbitrage). `h=0` (blue) is the unhedged long; `h=0.5` (orange) keeps a net-long tilt; `h=1.5` (red) is a net-short tilt that **goes negative above ~$210**. All lines cross at one point — the perp entry (~$160) — where the close equals the entry and every hedge ratio gives the same P&L. The takeaway: **under-hedging is a deliberate long tilt; over-hedging is a short tilt that can lose on a melt-up. Only `h=1` is locked.** Honest caveat: shorting the perp *because you think SpaceX is rich* and then leaving an uncovered **long** (h<1) is internally contradictory — size the residual as a view you actually hold, or set h=1.

### Shorting without leverage

Leverage and the hedge ratio are independent: the liquidation move depends only on `L` and the maintenance fraction, never on `h`. Running the short **unlevered (`L=1`, margin = full notional)** gives a wide buffer — **+81.8% (xyz)** / **+71.4% (vntl)** — so the Cerebras-style ruin mode the earlier draft flagged is, at `L=1`, gone. Over-collateralize (`L<1`, post margin > notional) and the buffer grows further (`L=0.5` → +172.7% on xyz).

**Honest framing (this is a correctness point, not a nicety):** "unlevered removes liquidation risk" is **false**. A perp short's loss is **unbounded** (price can rise without limit). `L=1` is not liquidation-proof — at maintenance fraction 0.10 a **+82% spike still liquidates it**. Only margin ≫ notional (`L → 0`) is truly un-liquidatable, and the IPO long shares sit at a different venue (Trade Republic), so they **cannot** cross-margin the Hyperliquid short. So: low leverage **widens the buffer, it does not remove the risk**, and the price is capital efficiency — the unlevered short ties up the full notional as margin. The tool reports **capital deployed = long notional + short margin** and a **locked return-on-capital** (e.g. h=1 unlevered xyz: ~8.4% over ~3 days; we report simple, non-compounded annualization because this is a one-shot event, not a repeatable yield).

---

## Live monitor / dashboard — what's worth building

You asked whether a localhost live tracker is worth it. Verdict, after scoring three designs against the repo's discipline (no cross-import with the separate `live_trading/` Streamlit app, no DB server, "no infra before signal," "terminus = live") and the **~3-day fuse** (listing anticipated June 12):

- **A web dashboard (Streamlit/FastAPI) is NOT worth building now** — it is textbook infra-before-signal. ~1 day to build well, never repaid inside a 72-hour one-shot, and it would *re-display* the same un-actionable snapshot more often without resolving any of the live-only unknowns. It also plants the first Streamlit footprint in `polymarket/research/` next to the `live_trading/` app it must never touch.
- **A terminal `--watch` loop IS worth it, and is shipped now** (zero new dependencies). It re-fetches the HL `/info` snapshot every `--interval` seconds, re-runs the already-tested evaluator, clears + redraws, and — when you declare a live short via `--live-entry / --live-short-notional / --live-margin` — front-and-centers a **liquidation-buffer line** (mark → fixed liq price, buffer %, `SAFE/WARN/BREACH` band, terminal bell on a thin buffer via `--alert-buffer-pct`). With `--parquet-log` it writes append-only tick shards under `data/analysis/spcx_convergence/watch_log/` (new shard per flush, never edited in place — honoring the Parquet invariant) so the full pre-listing path is captured for the post-mortem and is DuckDB-queryable.

  ```bash
  cd polymarket/research
  PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --watch 30 \
      --live-entry 159.9 --live-short-notional 50000 --live-margin 50000 --alert-buffer-pct 10 --parquet-log
  ```

- **When the web dashboard *does* become justified** (all three must hold): (a) the gate is still green on a refreshed Friday snapshot **and** a real Trade Republic $135 allocation is **confirmed filled** (a short is actually on); (b) this stops being a one-shot and becomes a **repeatable multi-listing program** (≥3 concurrent pre-IPO convergences to track); and (c) the `--watch` + Parquet + a notebook has already captured ≥1 full listing path and the **monitor itself** is provably the bottleneck (you need real-time multi-position alerting / remote glanceability a terminal can't give). Until then, the terminal loop is the right amount of infra.

---

## Cerebras (CBRS): a real-data analog for the SPCX short

To see what a pre-IPO-perp short actually lives through, here is the real Hyperliquid `xyz:CBRS` path through Cerebras's 2026-05-14 Nasdaq debut, mapped across the IPO lifecycle with exact timestamps (built by `scripts/spcx_cerebras_case_study.py`: HL **15m** perp + Yahoo **5m** cash spot; the **sourced event timeline + lookahead-free phase table** are below). Cerebras priced its IPO at **$185** (above the $150–160 range, ~$5.55B raised), **opened at $350** (~+89%; one source says $385), hit a **cash-equity high ~$385–386** (the perp printed **$392** on 05-14 14:00 UTC), and **closed $311.07 (+68%)** — then fell ~10% the next day and drifted to ~$246 cash (~$236 perp) by 2026-06-09. (These 2026 figures are corroborated across the issuer's pricing release, CNBC, Yahoo Finance and IPOScoop; treat them as scenario-era reporting. The genuinely pre-scenario record is just Cerebras's 2024 S-1 and a CFIUS/G42 delay.)

![Cerebras perp full arc: run-up, listing spike, settle](../../../data/analysis/plots/spcx_convergence/cerebras_full_arc.png)

![Cerebras listing-window intraday (15m)](../../../data/analysis/plots/spcx_convergence/cerebras_intraday_listing.png)

**Chart read.** Y-axis is the perp price ($/share-equivalent); horizontal dash-dot lines are the liquidation price of a **$277 perp short** at 1× / 2× / 5× (mmr 0.10). The perp ran from ~$178 (May 1) up through the listing, spiking to **$392** on 05-14, then fell back below the short entry to ~$236. A short entered at $277 would have been **liquidated at the spike at 2× (liq $378) and at 5× (liq $302), and survived only unlevered (1× liq $504)** — and the unlevered short, having survived, then converged *profitably* as the price fell to ~$236. This is the SPCX calculator's central claim, confirmed on real tape: **leverage, not the trade, is the ruin mode.**

**Important correction (no short squeeze).** The day-one move was a demand/oversubscription pop (>20× oversubscribed), **not** a short squeeze — a fresh IPO has essentially no day-one cash short interest to squeeze, and no squeeze/liquidation event is documented in the IPO reporting. The shorts at risk on the spike are **levered pre-IPO perp shorts** (exactly what the liq-lines above model), not cash-equity shorts. The task's "~$2.1M of shorts liquidated" figure is uncorroborated and is not relied on here.

### Perp vs spot on the listing day — how it gaps

A natural question: do the perp and the *cash equity (spot)* trade together on the listing day, and how does it gap? Using real HL perp 15m candles + real Yahoo cash 5m candles for 2026-05-14:

![Cerebras listing day: perp vs spot, the gap](../../../data/analysis/plots/spcx_convergence/cerebras_perp_vs_spot_listingday.png)

**Chart read (the key mechanic).** *Before listing there is no spot — only the perp trades.* On 05-14 the cash equity was merely *referenced* at the $185 offer (no trades, the flat grey-noted level) until it actually **opened for trading at ~16:55 UTC at ~$385** — a **+108% gap** over the offer. But the **perp had already run to ~$392 by ~14:45 UTC, ~2 hours before the cash open** — i.e. the **pre-IPO perp *pre-discovered* the open price**, and the cash equity gapped *up to meet where the perp already was*. After the open, perp and spot track together. So the "gap" is not perp-vs-spot — it is the **cash equity gapping from its $185 reference to the perp's discovered ~$385**, with the perp leading.

![Cerebras after listing: perp vs spot daily + basis](../../../data/analysis/plots/spcx_convergence/cerebras_perp_vs_spot_multiday.png)

**Chart read (post-listing basis).** Once both trade, the converted perp and the cash equity move in lockstep: the daily basis `perp − spot` was ~$18 on day 1 (perp still richer) and then collapsed to within ~$1–4 for the rest of the window, funding-tied. **Implication for the SPCX trade:** the dangerous, un-hedged moment is the *transition* (the perp can spike to ~2× the offer pre-cash-open, taking out levered shorts); once the cash equity exists and the perp converts, the two are tightly coupled and the convergence the calculator assumes holds.

**The SPCX analog, today.** SPCX is on the *pre-listing* leg of the same journey, but moving the other way: its perp premium to the $135 offer has **compressed from ~+48% in mid-May to ~+18% now** (~$160), i.e. the basis is fading as listing nears — a live illustration of the "richness may not persist" unknown.

![SpaceX perp today vs the $135 offer](../../../data/analysis/plots/spcx_convergence/spcx_vs_offer_today.png)

### Lifecycle event timeline (sourced timestamps, ET + UTC)

The whole point of a lifecycle map is *when* each thing happened, so we can place a short on the curve. Every row below carries a source URL; times are US/Eastern (EDT = UTC−4 in May 2026) and UTC. Where only the date is sourced, the clock time is labeled **(time unconfirmed)** — not guessed.

| event | date | ET | UTC | price / size | confirmed? | source |
|---|---|---|---|---|---|---|
| S-1 initial range | 2026-05-04 | ~06:26 (article) | ~10:26 | $115–125, 28M sh | range ✓, **clock unconfirmed** (SEC filing time) | [investing.com](https://www.investing.com/news/stock-market-news/cerebras-systems-plans-ipo-with-shares-priced-between-115125-432SI-4655311) |
| Range raise + upsize | 2026-05-11 (Mon) | — | — | →$150–160, →30M sh | range/upsize ✓, **time unconfirmed** | [CNBC](https://www.cnbc.com/2026/05/11/cerebras-raises-ipo-range.html) |
| **Pricing night** | 2026-05-13 | **19:45** | **23:45** | priced **$185**, 30M sh, ~$5.55B, +4.5M greenshoe | ✓ **time confirmed** (wire) | [GlobeNewswire](https://www.globenewswire.com/news-release/2026/05/13/3294565/0/en/cerebras-systems-announces-pricing-of-initial-public-offering.html) |
| Allocation window | 05-13 → 05-14 | 19:45 → 12:59 | 23:45 → 16:59 | overnight; no single print | window (not a print) | (pricing → open) |
| **Listing / first cash trade** | 2026-05-14 | **12:59** | **16:59** | open **$350 (majority) vs $385 (TechCrunch)**; perp high ~$392 | first-trade time **single-source**; **open disputed** | [IPOScoop](https://www.iposcoop.com/the-ipo-buzz-cerebras-cbrs-prices-ipo-at-185-25-above-range-2/) · [TechCrunch](https://techcrunch.com/2026/05/14/cerebras-raises-5-5b-kicking-off-2026s-ipo-season-with-a-bang/) |
| +1 day close (Fri 05-15) | 2026-05-15 | 16:00 | 20:00 | **$279.72** (−10.1%) | close from price API; news "down ~10%" ✓ | [Yahoo](https://finance.yahoo.com/markets/article/cerebras-stock-slides-after-near-70-surge-in-biggest-ipo-of-2026-130757084.html) |
| +2 trading days (Mon 05-18) | 2026-05-18 | 16:00 | 20:00 | **$296.65** (+6.1% vs Fri) | 05-16 is Sat (no session); **news "+17%" does NOT reconcile** with the Yahoo close — flagged | [Yahoo daily] |

![Cerebras IPO lifecycle: event timeline, perp vs spot](../../../data/analysis/plots/spcx_convergence/cerebras_event_timeline.png)

**Chart read.** Perp (blue, HL 15m) trades the entire window 24/7; spot (red, Yahoo 5m) only exists from the listing open and **only during regular Nasdaq hours** — it is plotted on a continuous 24/7 axis to overlay the perp, so each RTH session is a **solid** red line and the **dotted** red connectors are the overnight/weekend **closures** (market closed, *not* missing data). Vertical markers are the Phase-A events — **solid = sourced clock time, dotted = date-only**; the shaded band is the overnight allocation window. The perp was already trading **far above the $185 offer for the whole pre-listing period**, spiked at listing, then perp and spot drift down in lockstep (red sits right on blue during each session).

### Phase-aligned readout (lookahead-free)

For each event, the perp/spot value is the candle **at or before** that timestamp (never after — enforced by an assertion and a unit test), with a staleness guard so a data gap shows blank rather than a stale price. Columns: `perp→offer%` = perp richness over the $185 offer; `perp→spot$` = perp minus cash (only once spot exists).

| event | ET | perp | spot | perp→offer % | perp→spot $ |
|---|---|---:|---:|---:|---:|
| S-1 initial range | 05-04 06:26? | $282.40 | (none) | +52.6% | — |
| Range raise + upsize | 05-11 12:00? | $273.33 | (none) | +47.7% | — |
| Pricing night ($185) | 05-13 19:45 | $289.00 | (none) | +56.2% | — |
| Allocation window | 05-14 00:00? | $286.15 | (none) | +54.7% | — |
| Listing open | 05-14 12:59 | $376.73 | $385.00 | +103.6% | −$8.3 |
| +1 day close (05-15) | 05-15 16:00 | $281.13 | $280.19 | +52.0% | +$0.9 |
| +2 days close (05-18) | 05-18 16:00 | $296.30 | $296.53 | +60.2% | −$0.2 |

![Cerebras listing window, annotated (15m perp + 5m spot)](../../../data/analysis/plots/spcx_convergence/cerebras_listing_window_annotated.png)

**Read — the number that matters for a short.** The perp was **already +47% to +56% over the $185 offer at the range-raise and at pricing night** — it had priced in most of the pop *weeks before listing*. A short opened at the **pricing-night perp ($289)** faced a **+35.8% adverse move to the $392 spike** before settlement; algebraically that liquidates any short at **≳2.0× isolated** (mmr 0.10) — the same conclusion as the SPCX calculator, now on real Cerebras tape. After the open, `perp→spot` is tiny (−$8.3 at the open, +$0.9 / −$0.2 at the +1/+2 closes), confirming the two converge once both trade. *Data caveat:* HL has **purged 1m/5m** history for this date and Yahoo's 1m window has passed, so **15m perp / 5m spot are the finest available** — flagged, not interpolated.

### Access & demand: Cerebras vs SPCX (the live difference that decides the trade)

The convergence trade needs a **basis-preserving long** — i.e., primary allocation at the offer. Whether that exists is the real fork between the two names:

- **Cerebras was institutional-skewed and retail was effectively shut out of the $185 primary.** No source documents ordinary-retail (or European-retail) allocation at $185; coverage describes the standard hot-IPO path (request via syndicate/IPO-access, pro-rated, "not guaranteed"), and at **~20× oversubscribed** retail fill was negligible — retail bought in the **aftermarket at ~$350**. Sources: [techstackipo](https://www.techstackipo.com/ipo/cerebras/trading-day), [techi](https://www.techi.com/cerebras-ipo/), [Yahoo](https://finance.yahoo.com/markets/article/cerebras-stock-slides-after-near-70-surge-in-biggest-ipo-of-2026-130757084.html), [StockTwits](https://stocktwits.com/news-articles/markets/equity/cerebras-ipo-priced-above-range-20x-oversubscribed-what-you-need-to-know-about-nvidia-challenger-going-public/cZX1aQ2ReKx).
- **What that implies:** if you cannot get primary at $185, the long leg can only be sourced **post-open (~$350)**, with no $135-style basis — at which point shorting the ~$385 perp against a ~$350 aftermarket long is a **directional bet, not a locked basis**. *For retail, the Cerebras convergence trade did not exist.*
- **SPCX is structurally the opposite, which is why the trade is even plausible there:** SpaceX reserved **up to ~30% for retail** and a **~55.6M-share European retail tranche**, expected **$135** with a **$162.00 maximum** and a **"no allocation below $135"** threshold, per the **BaFin-approved EU prospectus dated 2026-06-05** (reported ~06-09; this corrects the "June 8" reference — the document is dated **June 5**). SPCX is only **~2× oversubscribed** (~$150B orders vs ~$75B). Sources: [Euronews](https://www.euronews.com/business/2026/06/09/spacex-ipo-how-european-retail-investors-can-buy-shares-and-the-risks-to-be-aware-of), [SeekingAlpha](https://seekingalpha.com/news/4601213-spacex-ipo-over-two-times-oversubscribed), [EU prospectus PDF](https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/SpaceX%20-%20EU%20Prospectus%20(Approved%20by%20Bafin)%20-%20June%205,%202026.pdf).
- **Implication for the SPCX convergence trade:** the basis-preserving long **is plausibly attainable** for European retail near $135 (subject to two live caveats: the offer is a **$135–$162 band**, so if it prices toward $162 the basis shrinks toward zero, and ~2× oversubscription means **partial pro-rata fills**). This is the single most important live difference — Cerebras shows that *without* primary access the convergence trade collapses into a directional aftermarket bet; SPCX's retail/EU tranche is what makes a real basis trade possible at all.

**Two cautions the Cerebras tape makes concrete:** (1) the listing-day path can spike ~2× the offer intraday before settling far lower, so any levered short is in real liquidation danger precisely at the transition; (2) the perp ($392) and cash-equity ($385) prints differed at the peak and the post-listing levels differ (~$236 perp vs ~$246 cash), so perp/cash basis is real — do not assume the converted perp tracks the cash close tick-for-tick.

---

## Block S1 — hedge grid + pre-hedge timing rule (2026-06-10)

> Task: [[spcx_listing_day_gameplan]] §7 Block S1 (highest priority, run before Friday). All marks in this section are a **2026-06-10 16:57 UTC live snapshot** (`xyz:SPCX` mark **$162.21**, funding **−0.000223%/hr**, EURUSD **1.1558** — Yahoo `EURUSD=X` intraday, flagged in every output). Re-run `--decision` at each decision node; the printed numbers refresh, the rule logic does not.

### What S1 adds, in plain English

The original calculator answered "is the basis real and what leverage survives" for a generic 1,000-share lot. Block S1 answers the **actual Friday decisions** for Justin's €10k Trade Republic subscription with <€2k of Hyperliquid margin:

1. **(a) Hedge grid** — for every final price ($135→$162) × fill fraction (10/25/50/100%) × leverage (1×/1.5×): how many shares can the margin actually hedge, what does the locked sleeve net (after taker fees both sides — xyz pair-close pays an exit fee — and funding carry), what is the return on the hedge-sleeve capital, and how many shares are left naked (the Frame-B residual sleeve of the gameplan).
2. **(b) Basis-decay fit** — how fast is the premium over $135 bleeding, fitted causally on HL hourly candles with bootstrap uncertainty, so "wait vs hedge now" can be priced rather than vibed.
3. **(c) Pre-hedge timing rule** — a single pre-registered trigger per fill-price row: *"hedge X shares at node Y iff live net basis ≥ Z"*, for nodes NOW / D1 pricing-night (Thu 22:00 CEST) / D2 allocation (Fri 8:00 CEST), where Z prices **both** risks of hedging before the fill is known: the naked-short melt-up tail **and** the option to simply wait and hedge risk-free at allocation.

**Practical example (one cell, hand-checkable).** Suppose the 424B prints $135, the fill comes in at 25%, and you run 1.5×. €10k × 1.1558 / $135 ≈ 85.6 shares requested → 21.4 filled. The €2k margin (= $2,312) at 1.5× supports $3,468 of short notional = 21.4 contracts at $162.21 — exactly covering the fill. Locked: 21.4 sh × ($162.21 − $135) ≈ $582 gross, minus ~$3 of taker fees (entry + pair-close exit at 4.5 bps/side) and ~$0.4 of funding paid (funding is currently slightly *negative* for the short) ≈ **$578 net**, on capital of 21.4×$135 + $2,312 ≈ $5,202 → **ROC ≈ 11.1%** over ~2 days. The same row at a $162 final price locks **≈ $0** — the basis is gone by construction.

### (a) The hedge grid (1.5× shown; 1× ceiling is 14.3 sh)

Cell = hedged shares | net locked $ | ROC | naked residual shares. Fill-price axis $135→$160 step $5 plus the $162 EU-prospectus corner; full grid (both leverages, all 56 cells) in `data/analysis/csv_outputs/market_maps/spcx_s1_hedge_grid.csv`.

| fill $ | fill 10% | fill 25% | fill 50% | fill 100% |
|---:|---|---|---|---|
| 135 | 8.6 sh, +$232, 11.1%, r0 | 21.4 sh, +$578, 11.1%, r0 | 21.4 sh, +$578, 11.1%, r21.4 | 21.4 sh, +$578, 11.1%, r64.2 |
| 140 | 8.3 sh, +$182, 8.9%, r0 | 20.6 sh, +$455, 8.9%, r0 | 21.4 sh, +$471, 8.9%, r19.9 | 21.4 sh, +$471, 8.9%, r61.2 |
| 145 | 8.0 sh, +$136, 6.7%, r0 | 19.9 sh, +$340, 6.7%, r0 | 21.4 sh, +$364, 6.7%, r18.5 | 21.4 sh, +$364, 6.7%, r58.3 |
| 150 | 7.7 sh, +$93, 4.7%, r0 | 19.3 sh, +$232, 4.7%, r0 | 21.4 sh, +$257, 4.7%, r17.2 | 21.4 sh, +$257, 4.7%, r55.7 |
| 155 | 7.5 sh, +$53, 2.7%, r0 | 18.6 sh, +$131, 2.7%, r0 | 21.4 sh, +$151, 2.7%, r15.9 | 21.4 sh, +$151, 2.7%, r53.2 |
| 160 | 7.2 sh, +$15, 0.8%, r0 | 18.1 sh, +$37, 0.8%, r0 | 21.4 sh, +$44, 0.8%, r14.7 | 21.4 sh, +$44, 0.8%, r50.9 |
| 162 | 7.1 sh, +$0, 0.0%, r0 | 17.8 sh, +$1, 0.0%, r0 | 21.4 sh, +$1, 0.0%, r14.3 | 21.4 sh, +$1, 0.0%, r50.0 |

**Column glossary.** *hedged shares* = min(shares filled, margin ceiling), where the ceiling = margin × leverage / mark (21.4 sh at 1.5×, 14.3 sh at 1×, at the $162.21 mark); *net locked $* = hedged × (mark − fill price) − taker fees (both sides for xyz, whose convert-in-place settlement needs a pair-close exit) + funding carry to settlement at the current hourly rate; *ROC* = net locked / (hedged × fill price + margin used) — hedge-sleeve capital only, the naked residual is a directional bet and is deliberately excluded; *r* = residual naked shares (the gameplan's Frame-B sleeve).

**Read.** Three facts the gameplan needs: (1) at the expected case (fill ≤ 25%) the position is **fully lockable at 1.5×** — residual ≈ 0; the margin constraint only binds at ≥50% fills, where the extra shares are house-money Frame-B exposure anyway. (2) The locked sleeve is worth **~$580 max** (25%+ fill, $135 price, 1.5×) and decays roughly linearly to **zero at a $162 print** — if the 424B prints near the EU cap, the hedge sleeve is dead on arrival, exactly as the gameplan's hard rule anticipated. (3) ROC on the locked sleeve is a healthy 11.1%/~2d at $135 but the *dollar* numbers are small: this sleeve is a measurement loop with positive carry, not a payday — consistent with the [[CODEX]] materiality rule.

### (b) Basis-decay fit — the premium is bleeding, and faster lately

![SPCX premium decay fit with rate-only projection](../../../data/analysis/plots/spcx_convergence/spcx_s1_decay_fit.png)

**Chart read.** Blue = hourly `xyz:SPCX` level premium over $135 (left edge ~$90 on 05-17, right edge ~$27 today); dotted black = the fitted full-window exponential trend (diagnostic); red = the **decision projection** — today's live premium decayed at the fitted rate — with the shaded band the daily-block-bootstrap 95% CI on the decay rate; dashed verticals = the D1 / D2 / settlement decision nodes. Note the two-regime shape: roughly flat $60–70 until ~06-03, then a fast bleed.

| fit | window | b (per day) | half-life | D2 (Fri 8:00 CEST) projected mark |
|---|---|---:|---:|---:|
| full-window | 05-17 → 06-10, 570 hourly closes | **−0.0436** (naive se 0.0013) | 15.9 d | **160.44** [159.90, 161.00] |
| recent-window (sensitivity) | last 7 d, 167 closes | **−0.1107** | 6.3 d | **157.94** |

**Design notes (assumptions stated).** Model is `ln(mark − 135) = a + b·days`, OLS on hourly closes whose **close time precedes the fit cutoff** (the lookahead guard is inside the fitter and unit-tested — appending future bars cannot change a fit). CI is a **daily-block bootstrap** (n=500, seeded) because hourly residuals are autocorrelated and the naive OLS se lies. Projections are **rate-only from the live premium**: the level of a traded price is taken as given and only the decay *rate* is applied forward — projecting from the fitted trend line instead would smuggle a mean-reversion-to-trend bet into a risk rule (the live premium currently sits −15% *below* the trend, so trend-projection would claim the basis will *grow* by waiting; refused). HL purged hourly bars before 05-17 (requested since 05-14 — flagged in output, immaterial). No `--watch` parquet shards exist yet, so the watch-log cross-check reports "absent" — the fit rests on candles alone, which the gameplan explicitly blessed.

**Read.** The premium is decaying with a half-life of **~16 days full-window, ~6 days in the last week** — the regime shift (flagged automatically when the two rates differ by >2 se) means model-form uncertainty exceeds the bootstrap band, and the honest projection range for Friday 8:00 is a mark of **~$158–160.4**, i.e. a net basis of **~$23–25.4/share if it prices at $135**. Waiting from now to allocation costs ~**$1.8/sh** (full rate) to ~**$4.3/sh** (recent rate) of expected basis. That number is what the timing rule must beat.

### (c) The pre-registered pre-hedge rule

The decision at D0/D1 is not "hedge vs never hedge" — it is "hedge **now**, unfilled, vs **wait** and hedge risk-free at allocation." Pre-hedging pays the richer current basis but risks being naked-short into a melt-up if the fill disappoints; waiting pays the decayed basis with zero naked risk (and keeps the option to skip if the basis dies). Per pre-hedged share, with `p = P(fill ≥ pre-hedge size)`:

```
EV(pre-hedge at node) = p·B_node − (1−p)·NL          (task formula, vs never hedging)
EV(wait)              = p·max(B_alloc, 0)
pre-hedge beats waiting  ⇔  B_node > Z* = (1−p)/p·NL + max(B_alloc, 0)
```

where `NL` (naked loss) = mark × E[melt-up] + exit fee, with the melt-up distribution taken from the Cerebras analogs (+13/+26/+39%, equal weights → E[move] = +26%), and `B_alloc` = today's decay-fit projection of the allocation-node basis, **frozen at run time** so Z* is a constant trigger, not a moving target. On a shortfall the *whole* pre-hedge is conservatively treated as naked and bought back at the melted-up price.

The rule table at 06-10 marks (X = pre-hedge shares; Z = trigger $/sh; B = basis at the node, live at NOW / projected later; full table with CIs in `data/analysis/csv_outputs/market_maps/spcx_s1_decision_table.csv`):

| fill $ | NOW | D1 pricing-night (Thu 22:00 CEST) | D2 allocation (Fri 8:00 CEST) |
|---:|---|---|---|
| 135 | X 8.6, Z 35.93, B +27.12 → **wait** | X 8.6, Z 35.84, B +25.82 → **wait** | X 21.4, Z 0, B +25.36 → **HEDGE** |
| 140 | X 8.3, Z 30.93, B +22.12 → wait | X 8.3, Z 30.84, B +20.82 → wait | X 21.4, Z 0, B +20.36 → **HEDGE** |
| 145 | X 8.0, Z 25.93, B +17.12 → wait | X 8.0, Z 25.84, B +15.82 → wait | X 21.4, Z 0, B +15.36 → **HEDGE** |
| 150 | X 7.7, Z 20.93, B +12.12 → wait | X 7.7, Z 20.84, B +10.82 → wait | X 21.4, Z 0, B +10.36 → **HEDGE** |
| 155 | X 7.5, Z 15.93, B +7.12 → wait | X 7.5, Z 15.84, B +5.82 → wait | X 21.4, Z 0, B +5.36 → **HEDGE** |
| 160 | X 7.2, Z 10.93, B +2.12 → wait | X 7.2, Z 10.84, B +0.82 → wait | X 21.4, Z 0, B +0.36 → **HEDGE** |
| 162 | X 7.1, Z 10.57, B +0.12 → wait | X 7.1, Z 10.48, B −1.18 → wait | X 21.4, Z 0, B −1.64 → **wait** |

**Column glossary.** *X* = pre-hedge size = min(pessimistic-fill shares, margin ceiling); pre-nodes size to the **10%-fill row only** (the gameplan's "never above the 10%-fill row" cap); at D2 the fill is known, so X = min(fill, 21.4-sh ceiling). *Z* = the pre-registered trigger: `(1−p)/p × naked-loss + projected allocation basis` at pre-nodes; **0 at D2** (no naked risk, no wait option left — any positive net basis is lockable). *B* = net basis $/sh = mark − fill price − entry fee + funding-to-settle; HEDGE/wait = whether **today's** B clears Z — the rule itself binds on the basis **observed live at the node**.

And the EV matrix at the $135 anchor price (total $ on the tranche; "vs never hedging" task formula):

| pessimistic fill | NOW | D1 | D2 (fill known, p=1) |
|---:|---:|---:|---:|
| 0% | $0 (the f=0 trivial corner: zero size → never hedge) | $0 | $0 |
| 10% (X=8.6 sh, p=0.80 assumed) | +$113 | +$105 | **+$217** |
| 25% (X=21.4 sh, p=0.50 assumed) | −$162 | −$172 | **+$542** |

**Read — the rule in one paragraph.** At current marks **no pre-hedge node arms, at any fill price**. The reason is structural, not marginal: the basis lost by waiting ~2 days (~$1.8–4.3/sh) is an order of magnitude smaller than the naked-melt-up risk premium (~$10.6/sh at p=0.8, plus the ~$25 wait-comparator), and the EV matrix says the same thing in dollars (+$217 waiting vs +$113 pre-hedging the 10% tranche). **The pre-registered rule is therefore: do nothing until the allocation e-mail; at Fri ~8:00 CEST hedge min(fill, 21.4 sh at 1.5× / 14.3 sh at 1×) iff live net basis > 0.** Pre-hedging earlier becomes correct only if the perp spikes through **Z* ≈ $36/sh over a $135 fill (perp ≥ ~$171)** before allocation — i.e., a pre-listing melt-up so violent that locking it beats any plausible Friday basis; that trigger is exactly what the frozen Z column encodes, and it doubles as the "naked-short loss" the gameplan's §5.4 hard rule worries about. This also resolves the gameplan's D1 decision: at these marks, **D1 = do nothing** unless the 424B prints AND the perp is ≥ ~$171.

### S1 assumption ledger (modeled vs live-only)

**Modeled (fair, declared, CLI-overridable):** P(fill≥10%) = 0.80 and P(fill≥25%) = 0.50 (pure assumptions — TR pro-rata is unmodelable offline; `--p-fill`); melt-up = equal-weighted Cerebras analogs +13/+26/+39% applied un-scaled to every node's naked window (conservative for the short D1→D2 window; `--meltup-dist`); total-shortfall-on-miss (whole pre-hedge naked — conservative); exponential level-premium decay, rate-only projection; funding frozen at the current hourly rate (currently *slightly negative* for the short: ~−$0.02/sh over 51h — the positive-carry line from 06-09 has flipped, immaterial at this size); EURUSD 1.1558 Yahoo intraday.

**Live-only (the rule cannot resolve; measure):** the actual fill; the actual basis at D1/D2 (the rule binds on it); book depth at 14–21 contracts; funding path; `xyz` convert-in-place oracle behavior; TR day-1 sell capability (Block S4 / human in-app check).

### S1 acceptance (all green)

16 new tests extend the original 11 (**27 passing**, same file): grid axes contain $135/$162 corners and the $5 steps; a hand-computed grid cell reproduces exactly (sizing, fees, ROC); the margin ceiling binds at high fills with the residual counted; **zero margin → X=0 → never hedge** and **zero basis → never hedge** (the two required trivial corners, plus negative-basis rows never arm); the decay fit recovers a synthetic −2%/day slope with the bootstrap CI bracketing truth, drops non-positive premiums counted, and is **provably lookahead-free** (appending post-cutoff bars, including a synthetic melt-up, leaves the fit bit-identical); EV/threshold algebra (EV=0 exactly at Z; p=1→Z=0; p=0→Z=∞; Z* = Z_naked + max(0, projected allocation basis)); wait-dominance when no decay information exists; early arming when fill odds are near-certain and the melt-up tail is tiny; and a source-inspection test that the whole S1 decision path (`build_decision_table`, EV/threshold functions, the grid) never references a realized settlement price and cannot reach raw candles (it only receives the causal fit object).

### S1 decision and next step

- **Decision:** pre-registered and frozen — **no pre-hedge at current marks; the Friday allocation gate is the rule** (hedge min(fill, margin ceiling) iff live net basis > 0, sized by the grid row that materializes). The spike-trigger column (Z ≈ $36/sh over $135) stays armed as the only path to an earlier hedge.
- **Next:** start `--watch --parquet-log` Thursday evening per the gameplan (D0 note: the logger is mandatory from pricing night onward — it also feeds the watch-log cross-check this run reported as absent); re-run `--decision` after the 424B prints (`--offer <final price>`) and again at allocation; post-day, feed the realized basis-at-each-node back into this section per the gameplan's post-mortem step.

---

## Pre-registration / acceptance criteria (all green)

The calculator was built against a frozen set of acceptance tests in `tests/test_spcx_convergence_calc.py` (**originally 11 tests; now 27 with Block S1's 16 — all passing**, `uv run pytest`). The original five criteria:

1. **Units (i):** when `perp_base == IPO_base` the naive and units-matched gaps are equal; when bases differ they diverge by exactly the denominator ratio R (verified: `per-IPO-share-equiv == mark / R`, and a naive 1:1 short over-hedges by exactly `R − 1`). Valuation-unit FDV and per-share equiv verified separately.
2. **Direction-neutrality (ii):** locked P&L is invariant to the assumed settlement price across a sweep of settle prices $80–$320 (variance < 1e-9, all equal to `entry − offer`).
3. **Liquidation algebra (iii):** with mmr = 0 the liq adverse fraction equals `1/L` exactly; with mmr = 0.1 it flips SURVIVE→LIQUIDATED at the boundary `(1+1/L)/(1+mmr) − 1`; `max_survivable_leverage` inverts it correctly.
4. **No look-ahead:** a source-inspection test asserts the decision/verdict path (`ContractEval.__post_init__`) never references the realized settlement price; `realized_pnl_*_at_settlement` is used **only** for the invariance sweep.
5. **Hedge decomposition (v):** locked is 0 at h=0 and `N·basis` at h=1; the residual slope is +ve (net long) for h<1, 0 at h=1, −ve (net short) for h>1; total reduces to pure-long at h=0 and to the flat locked basis at h=1; over-hedge (h=1.5) total falls as the close rises (net-short). Plus `capital_deployed` / `return_on_capital` / **simple** (non-compounded) annualization, and the `liq_buffer_summary` SAFE/WARN/BREACH band boundaries.

> The original direction-neutrality proof only holds at h=1; the partial-hedge enhancement was explicitly built so the residual is **never** routed through the locked/forcing-question machinery — the forcing question reports the residual slope `(1−h)·N` when h≠1 rather than an unconditional YES.

### Cerebras regression fixture (reproduced exactly)

The task's regression case is encoded and passes: **short @277 vs long @185, settle @311 → locked +92** (and invariant to settle), the **+39% path flags LIQUIDATED at ≥2×** (mmr 0.10 from CBRS's 5× max → liq move +36.4% at 2×, below +39%), and **short @340 → locked +155 and SURVIVE at low (1×) leverage**.

---

## How to reproduce

```bash
cd polymarket/research
# realistic default: live-fetch both, FDV-neutral (h=1) + UNLEVERED (1x), + both charts:
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py \
    --chart data/analysis/plots/spcx_convergence/liq_survival.png \
    --hedge-chart data/analysis/plots/spcx_convergence/hedge_pnl_by_close.png
# partial hedge (half-hedged, net-long residual):   over-hedge (net-short):
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --hedge-ratio 0.5
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --hedge-ratio 1.5
# levered (riskier) sensitivity, and the venue-max danger case:
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --leverage 2
# stress the xyz split-adjusted-base hypothesis (R != 1, naive gap becomes wrong):
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --xyz-base 11870000000
# localhost live monitor with a declared live short (liq-buffer panel + append-only parquet):
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --watch 30 \
    --live-entry 159.9 --live-short-notional 50000 --live-margin 50000 --parquet-log
# cached dated snapshot; machine-readable JSON; acceptance tests:
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --offline --json
PYTHONPATH=. uv run pytest tests/test_spcx_convergence_calc.py -q
# Block S1 — hedge grid / decay fit / pre-registered pre-hedge rule (one page + CSVs + chart):
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --decision \
    --decay-chart data/analysis/plots/spcx_convergence/spcx_s1_decay_fit.png
# S1 pieces standalone, and the post-424B re-run with the final price:
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --grid
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --decay-fit
PYTHONPATH=. uv run python scripts/spcx_convergence_calc.py --contract xyz --decision --offer 140
# Cerebras lifecycle: pulls HL 15m perp + Yahoo 5m spot, writes the event-timeline + listing-window
# charts, the phase-aligned readout, and 3 CSVs (perp 15m, spot 5m, phase table); then its tests:
PYTHONPATH=. uv run python scripts/spcx_cerebras_case_study.py
PYTHONPATH=. uv run pytest tests/test_cerebras_lifecycle.py -q
```

Snapshots cache to `data/analysis/spcx_convergence/` (`hl_snapshot_<ts>.json` + `latest.json`); `--watch --parquet-log` appends tick shards under `data/analysis/spcx_convergence/watch_log/`. Charts write to `data/analysis/plots/spcx_convergence/`; Cerebras CSVs to `data/analysis/csv_outputs/market_maps/` (`cerebras_cbrs_perp_15m.csv`, `cerebras_cbrs_spot_5m_listingwindow.csv`, `cerebras_lifecycle_phase_table.csv`).

---

## Decision and next step

- **Verdict:** the long-IPO / short-perp convergence basis is **real and large** ($20–25/IPO-share gross, costs immaterial, positive funding carry), and sized realistically — **fully FDV-hedged (h=1) and UNLEVERED** — it is **TRADE-ABLE offline** (survives a +71–82% spike, ROC ~7–8% over ~3 days). The earlier NO-TRADE was a **leverage artifact**: at ≥2× a +39% melt-up liquidates, at venue max a +3–9% wiggle does. **The rule is: don't lever the short (≤1.5×), and decide your hedge ratio deliberately** — h=1 is locked arbitrage, h<1 leaves a net-long directional tilt, h>1 a net-short tilt that loses on a melt-up.
- **Concrete next action (if pursued):** a *minimal instrumented live test* — confirm a real TR allocation fill at $135, post a small **unlevered (or ≤1.5×)** short, run the `--watch` monitor (liq-buffer panel + Parquet tick log), and capture book depth at size, realized funding, and oracle/mark behavior into the listing transition. **Do not scale, do not build a web dashboard yet.** Refresh the snapshot the night before listing; if the perp's richness has collapsed by Friday morning, the basis is gone and there is no trade.
- **Timing (Block S1, 2026-06-10):** the pre-registered rule says **do not pre-hedge before the allocation e-mail** — at 06-10 marks the basis decay forfeited by waiting (~$1.8–4.3/sh over 2 days) is an order of magnitude cheaper than the naked-melt-up tail on an unfilled short. **Hedge at Fri ~8:00 CEST: min(fill, ~21 sh at 1.5× / ~14 sh at 1×) iff live net basis > 0**; pre-hedge earlier only if the perp first spikes ≥ ~$171 (the frozen Z* ≈ $36/sh trigger). See § Block S1.

---

## Realism framing — offline-resolvable vs LIVE-ONLY

Per [[CODEX]] § Realism calibration and the COWORK *terminus = live* rule, here is the honest two-part ledger.

**Offline-resolvable — and resolved by this calculator (modeled assumptions, fair):**
- **Units normalization** — FDV and per-IPO-share-equiv on the 13.076B base; the denominator ratio and over/under-sizing error. (Assumption: `xyz:SPCX` base = IPO base; trade[XYZ] publishes none — see caveat below.)
- **Locked basis** — direction-neutral algebra, proven invariant to settlement price (at h=1).
- **Hedge decomposition** — locked vs directional residual for any hedge ratio; the residual is honestly labeled a directional bet, never folded into the locked/forcing-question path.
- **Liquidation algebra & the leverage trade-off** — exact adverse-move-to-liquidation per leverage (validated against the Cerebras fixture); unlevered gives a wide but **finite** buffer (a perp short's loss is unbounded — not "zero risk").
- **Capacity vs OI cap** and **capital / return-on-capital** — desired notional as a fraction of cap and live book; locked return on (long notional + short margin), simple-annualized.

**LIVE-ONLY — this gate cannot answer these (graduate to a measurement loop, do not assume):**
- **Real TR allocation fill** — whether you actually get $135 shares, at what size, with what day-1 sell/flipping constraints (the [[spacex_ipo_coworker_addendum]] flags these as unconfirmed).
- **Real book depth at size** — the snapshot OI is not executable depth; slippage at your size is unknown.
- **Richness persistence** — whether the perp is still rich vs $135 on Friday morning; the basis can evaporate.
- **Oracle / transition behavior at listing** — especially for `xyz:SPCX`, which may **convert in place** to a listed-equity perp rather than cash-settle, so its convergence reference is the post-listing trading price, and the conversion-moment mark/oracle behavior is a live unknown and a gap risk, not just a valuation question.

**Caveat on the `xyz:SPCX` share base (a borrowed/assumed knob, flagged).** trade[XYZ] explicitly disclaims a share count / FDV for SPCX, so its per-IPO-share-equiv is only clean **if** the listed/converted Class A stock trades on the same 13.076B total-share market-cap convention (the default, R = 1). If the realized listed base differs (splits, option exercise, a diluted convention — e.g. the ~11.87B hypothesis), the basis rescales by R: at 11.87B the units-matched basis falls from **$24.92 to $10.17/share** and a naive 1:1 short **over-hedges by 10.2%**. Confirm the post-listing share base before trading the `xyz` leg; `vntl:SPACEX` carries no such ambiguity because Ventuals settles to basic-shares × close on the stated base.

**Disposition.** This is a **"merits a live MEASUREMENT loop," not "merits a trading system"** result: offline-green at low leverage, with every remaining unknown live-only. The calculator is the gate that, if still green on a refreshed Friday snapshot, justifies one tiny instrumented short — nothing more.
