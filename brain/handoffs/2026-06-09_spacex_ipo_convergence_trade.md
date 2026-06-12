---
title: "SPCX IPO Convergence Trade — Session Snapshot"
tags: [spacex, spcx, ipo, hyperliquid, ventuals, convergence, basis, handoff]
created: 2026-06-09
audience: "Cowork/Codex sessions on the SpaceX IPO convergence trade"
status: "live thread; listing ~2026-06-12 — refresh all marks/terms before any decision"
---

# SPCX IPO Convergence Trade — Session Snapshot (2026-06-09)

> Hub: [[spacex_ipo_market_map_handoff]] · Calculator + decision note: [[spcx_convergence_calc_findings]] · Companion: [[spacex_ipo_coworker_addendum]] · Master list: [[TODO]] § SPCX

## The trade

Long SpaceX IPO Class A at the offer; short a Hyperliquid pre-IPO perp; hold **both** legs to the perp's first-trading-day settlement to capture the convergence basis. If units match and you hold to settlement, locked P&L = `(perp_entry_in_IPO_units − offer_price) × shares`, **direction-independent** — the settlement price cancels.

## Verdict (offline gate)

**Basis is real and large** (~$20–25/IPO-share, ~15–18% rich on the 2026-06-09 snapshot; HL fees immaterial; funding is *positive carry to the short*). **TRADE-ABLE offline only UNLEVERED / ≤1.5×.** Leverage is the ruin mode: at ≥2× a +39% Cerebras-style pre-settlement melt-up liquidates the short before convergence; at venue max (5×/3×) a +3–9% wiggle does. Don't lever the short. Hedge ratio is a deliberate choice: `h=1` is locked arbitrage, `h<1` a net-long tilt, `h>1` a net-short tilt that loses on a melt-up.

## Five corrections this session changed the picture

1. **Convergence reference = first-day CLOSE, not the open** (for `vntl:SPACEX`: cash-settles 4pm ET to basic-shares × close via `haltTrading`). `xyz:SPCX` may **convert-in-place** to a listed-equity perp instead — different settlement, live unknown. "Trade the gap at the open" is the wrong exit; equality is enforced at the close.

2. **Cerebras precedent reframed (and partly refuted).** The "$340 perp converged within ~6% of the open" framing was wrong in detail: the synthetic *pre-discovered* the open (perp ~$392 ~2h before cash opened ~$385), and earlier in the day it *undershot* (~$277 morning-of vs $350 open). Lesson on real tape: **leverage, not the trade, is the ruin mode** — an unlevered $277 short survived the spike and converged profitably; a 2×/5× short was liquidated. No documented day-1 *cash* short squeeze; the "$2.1M shorts liquidated" figure is uncorroborated.

3. **Denominator critique corrected.** It is NOT about perp supply or a fixed valuation. It is the **share-count baked into the price quote**. It applies to **`vntl:SPACEX`** (valuation-units = valuation/$1e9; must divide by share base to compare to $135 and to size the short — 1,000 long shares ≈ 76.5 vntl contracts, not 1,000). It does **not** apply to **`xyz:SPCX`** (per-share, R=1: basis = `mark − 135` directly, 1:1 short). `xyz` residual risk is only a split/share-class/converted-base mismatch.

4. **$135 is NOT a fixed price in the filing.** EDGAR (CIK 0001181412): the June-3 **S-1/A is a preliminary red-herring**; $135 is *"expected,"* the underwriting-agreement price is a blank `$[•]`, and there is **no no-adjust/fixed language**. Binding single global price is set ~June 11 in a **424B (not yet filed)**. Reuters' "told its banks it wouldn't adjust it" was *reported intent*, not a filing commitment. **EU retail: maximum offering price $162.00; orders below $135 "should not expect allocation."** ⇒ upward-revision risk is **bounded at $162, not foreclosed** — made improbable (not impossible) by the soft ~2× book. The buyer's lever is the **Trade Republic order price limit**.

5. **Allocation reality is the binding live risk, not price.** SpaceX reserves "up to 30%" retail + a ~55.6M-share EU tranche (7 countries); Trade Republic allocates pro-rata at the offer price — you can be scaled to a fraction or zero, with full subscription cash locked during the window. Contrast Cerebras: ~20× oversubscribed, priced $185 ($25 above the *raised* $150–160 range), retail largely shut out → forced to chase the ~$350 open. The SPCX long leg's attainability at a basis-preserving price is the thing that decides the trade, and it isn't known until ~June 11–12.

## Offline-resolvable vs LIVE-ONLY

**Resolved offline (in [[spcx_convergence_calc_findings]]):** units normalization, locked-basis algebra (proven settle-invariant at h=1), hedge-ratio decomposition, liquidation-vs-leverage math (validated on the Cerebras fixture), capacity vs OI caps, ROC.

**LIVE-ONLY (graduate to a measurement loop, do not assume):** real TR $135 allocation fill + size + price-limit outcome; book depth at size; whether perp richness persists to Friday (premium compressed ~+48% mid-May → ~+18% now); oracle/transition behaviour at listing (esp. `xyz` convert-in-place); TR day-1 flipping rule.

## Open work (see [[TODO]] § SPCX)

- Claude Code: Cerebras event-aligned perp+spot timeline (exact ET+UTC timestamps across range-raise/pricing/allocation/listing/+1–2d), fold into [[spcx_convergence_calc_findings]]; + retail-vs-institutional + oversubscription/European analysis.
- Claude Code: calculator fill-price ($135→$162) × fill-size grid + cash-lockup-vs-margin capital plan.
- Friday-morning live gate before any capital.

> Not investment advice — structural analysis. Many venues are geo/KYC-restricted; refresh every number before acting.
