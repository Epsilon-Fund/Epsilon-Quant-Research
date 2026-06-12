---
title: "EU Retail IPO Subscription — Broker Comparison Model + Capital-Split Recommender"
tags: [ipo, subscription, brokers, revolut, trade-republic, degiro, interactive-brokers, allocation, pro-rata, spacex, spcx, decision-model]
created: 2026-06-09
status: "decision model built + tested; broker facts are mostly UNKNOWN (vault only documents Trade Republic) and must be filled in by hand"
audience: "Justin / Cowork / Codex deciding how to split capital across EU brokers for a retail IPO subscription"
---

# EU Retail IPO Subscription — Broker Comparison Model + Capital-Split Recommender

> Hub: [[POLYMARKET_BRAIN]] · [[COWORK]]
> SpaceX cluster: [[spacex_ipo_market_map_handoff]] · [[spacex_ipo_coworker_addendum]] · [[spcx_convergence_calc_findings]] · [[spacex_pdf_construction_audit]]
> Tracking system: `IPO Subscriptions/_template.md` · `IPO Subscriptions/SpaceX_SPCX_2026-06.md` · `IPO Subscriptions/README.md`
> Tool: `polymarket/research/scripts/eu_ipo_capital_split.py` · Tests: `polymarket/research/tests/test_eu_ipo_capital_split.py`
> Table terms: [[polymarket_table_dictionary]]

## Plain-English Summary

- **What this is.** A reusable decision model for subscribing to a retail IPO across the European brokers Justin uses — **Revolut and Trade Republic** (DEGIRO and Interactive Brokers dropped per user, 2026-06-09) — plus a runnable recommender that turns a capital budget into a per-broker split, and a tracking system that learns each broker's realised fill rate over time.
- **Why it exists.** The live anchor deal is the **SpaceX IPO (ticker `SPCX`, expected $135, listing anticipated 2026-06-12)**. The user wants a repeatable way to decide *how much to subscribe at which broker*, anchored on facts in the vault (plus facts the user supplies), and to build a per-broker fill-rate track record for the next deals (OpenAI, Anthropic, …).
- **What we actually know.** **Trade Republic** (vault): in-app IPO subscriptions, official price, **pro-rata** if oversubscribed, **1 EUR** fee — see [[spacex_ipo_coworker_addendum]]. **Revolut** (user-provided, 2026-06-09): offers the SPCX subscription, **no fee**, **$500 min** (USD). Remaining blanks are `UNKNOWN` and must be confirmed by hand. Nothing here is invented.
- **The allocation model encoded.** *Within* a broker, fill is **flat pro-rata**: everyone gets the same fill fraction `phi`, and your subscription *size* scales your absolute shares, **not** your fill %. *Across* brokers, fills are **independent draws on separate sealed sub-allocations** — you can't see the sub-allocations or internal demand. Independence is what makes spreading reduce variance.
- **The recommendation logic (the point).** With flat pro-rata, `shares = fill_rate × subscription`, so we **estimate each broker's fill rate and send more money where it's higher** — the split is *proportional to fill rate* (`weight_b ∝ fill_rate_b ^ tilt`). The fill estimate is the best of three tiers: your **realised** logs → a **researched deal-level prior** → a gut **maturity** prior. For SpaceX the research prior is *equal* across brokers (no per-broker evidence), so today's split is **50/50**. Respect each broker's min/max; keep total subscription **== C** so a full fill never exceeds your budget. See [§ The core idea](#the-core-idea--estimate-the-fill-rate-go-where-you-get-filled-more).
- **One-line takeaway / status.** The model and tool are built and tested (10/10 tests green). Researching the fill rate (web, 2026-06-09) found **no per-broker fill difference** for SpaceX and a low deal-level fill (~5%), so the honest split is **50/50** with **expected fill ≈ €500 per €10k subscribed**. Fill-rate logic can't steer until deal #1 resolves (you were right); any Revolut lean is an **operational** choice (no fee, $500 min, day-1 sell), not a fill edge. Remaining work is operational: TR eligibility/min, and the **USD ($500 Revolut floor) vs EUR** reconciliation.

This is research/decision tooling, not investment advice. Eligibility and product availability are jurisdiction-specific and unverified.

---

## STEP 1 — What the vault already knows, per broker (with citations)

I searched the whole vault (filenames, `#tags`, body text) for `{Revolut, Trade Republic, DEGIRO, Interactive Brokers, IBKR, IPO, subscription, allocation, pro rata, fill rate, prospectus}`. The broker knowledge lives **entirely** in the SpaceX IPO cluster under `polymarket/research/notes/overview/market_maps/` and `.../data_quality/`. Findings per broker:

### Trade Republic — the only broker with vault facts

| fact | value | source |
|---|---|---|
| Offers retail IPO subscription | **Yes** — European customers can subscribe to selected IPOs directly in-app *before* trading starts | [[spacex_ipo_coworker_addendum]] ("TradingView And Trade Republic"); [[spacex_ipo_market_map_handoff]] (Current Tradable Surfaces) |
| Allocation price | The **official allocation price** (not a market price) | [[spacex_ipo_coworker_addendum]] |
| Allocation method | **Pro-rata, proportional to subscription volume** — now confirmed by TR's own launch press and German coverage (not just "medium-high") | TR press / EQS-News; boerse-online; telepolis (web, 2026-06-09) |
| Fee | **1 EUR settlement fee per order** (re-confirmed) | [[spacex_ipo_coworker_addendum]]; TR press (web, 2026-06-09) |
| SPCX retail tranche / float | Press: **up to ~30% of SPCX reserved for retail**, but **free float ~5%** ⇒ **expect heavy oversubscription / low fill**. Worked example in coverage: subscribe €10,000 → maybe ~€500 or nothing. | boerse-online; telepolis (web, 2026-06-09) |
| Maximum subscription | **None documented** in TR's launch coverage (consistent with user: no max). Absence of evidence, not a guarantee. | web search, 2026-06-09 (user concurs) |
| Day-1 sell restriction | **None documented** for TR (the 30-day "flipping" bans found in search are US-broker policies, e.g. Robinhood, not TR). Consistent with user: no day-1 limit. | web search, 2026-06-09 (user concurs) |
| Eligibility | "European retail investors" / "European customers" (Justin's *specific* account eligibility not stated) | [[spacex_ipo_market_map_handoff]]; TR press |
| TradingView routing | **Not** a TradingView-integrated broker; subscription is a separate in-app workflow | [[spacex_ipo_coworker_addendum]] |
| **Still UNKNOWN / unconfirmed** | Justin's account eligibility; **minimum** subscription; exact subscription deadline; whether the **full subscription cash is blocked** during the book-build; SPCX-specific participation detail (TR says "more details in the coming days") | TR press; web, 2026-06-09 |

### Revolut — user-provided facts (2026-06-09)

The vault has **no** Revolut facts (a body-text grep returned zero matches). The following are **user-provided** (allowed under the no-invention rule), recorded 2026-06-09:

| fact | value | source |
|---|---|---|
| Offers retail IPO subscription for SPCX | **Yes** | user, 2026-06-09 |
| Fee | **None** | user, 2026-06-09 |
| Minimum subscription | **$500** (USD-denominated) | user, 2026-06-09 |
| Maximum subscription | **None** | user, 2026-06-09 |
| Day-1 sell restriction | **None** (can sell on first trading day) | user, 2026-06-09 |
| Eligibility | Usable by Justin (implied) | user, 2026-06-09 |
| Operational maturity (trust prior) | **0.90** — user's most-trusted broker | user prior (editable) |
| **Still UNKNOWN** | confirmed allocation method (model assumes flat pro-rata) | — |

### DEGIRO and Interactive Brokers — out of scope (user, 2026-06-09)

The user dropped these two ("idc about"), so they are **removed from the candidate set and the recommender**. They were considered and are noted here only so a future reader knows they were deliberately excluded, not overlooked.

### Contradictions and gaps

- **No hard contradictions** were found.
- **The candidate set is now Revolut + Trade Republic.** Both have facts (Revolut user-provided, TR vault-documented), so the model is no longer data-starved on the brokers that matter — the main remaining unknowns are Revolut's max + day-1 sell, and TR's min/max + eligibility + day-1 sell.
- **A currency gap to reconcile:** Revolut's **$500 min is USD-denominated**, while Trade Republic settles in **EUR** (1 EUR fee). The recommender works in a single currency and does **not** do FX — decide whether your budget `C` is in EUR or USD and convert the $500 floor yourself.
- **A soft tension worth flagging:** the user trusts **Revolut most** (prior 0.90), yet the *vault's* only independently-evidenced IPO-subscription product is **Trade Republic's** — Revolut's facts come from the user, not the vault. That is fine; it just means the maturity prior is doing real work.
- **An EV ambiguity inherited from the cluster (not broker-specific):** the colleague's PNG shows a `-$3.3/share` IPO EV that contradicts the `+$32/share` mean-minus-entry arithmetic; see [[spacex_ipo_coworker_addendum]] and [[spacex_pdf_construction_audit]]. The corrected, audited day-1 distribution is **P(close > $135) ≈ 80%, mean ≈ $168, median ≈ $164**. Use those, not the chart's EV line.

### Sources (web, 2026-06-09)

Trade Republic facts above were re-verified by web search on 2026-06-09. Note that **minimum/maximum subscription** and **day-1 sell** are *not stated* in any of these — "no max / no day-1 limit" is an **absence of documented restriction** corroborated by the user, not a published guarantee. TR's own communication says fuller terms are coming in the days before listing.

- [Trade Republic — "Starting today … direct access to IPOs" (EQS-News)](https://www.eqs-news.com/news/corporate/starting-today-trade-republic-gives-european-retail-investors-direct-access-to-ipos/cb7d9263-f112-4539-a30e-2a87dcf950f7_en) — official price, **pro-rata by subscription volume**, **1 EUR** fee.
- [boerse-online — Trade Republic lets clients join the SpaceX IPO](https://www.boerse-online.de/nachrichten/aktien/trade-republic-laesst-kunden-beim-spacex-ipo-mitmachen-was-jetzt-wichtig-ist-20402250.html) — ~30% retail reservation vs ~5% float; allocation depends on demand; no min/max/day-1 terms stated.
- [telepolis — "Trade Republic opens IPOs to everyone"](https://www.telepolis.de/article/Trade-Republic-oeffnet-IPOs-fuer-alle-das-sollten-Anleger-wissen-11320544.html) — pro-rata proportional to subscription volume; worked example €10,000 subscribed → ~€500 or nothing.
- [brokerchooser — Trade Republic fees 2026](https://brokerchooser.com/broker-reviews/trade-republic-review/trade-republic-fees) — fee schedule context.

---

## STEP 2 — The broker comparison model

### The comparison table

`UNKNOWN` means *not in the vault and not provided* — do not guess; confirm before funding. `maturity` is the **only** subjective field: a user-provided execution-trust prior in `(0,1]`, editable in the script's `default_brokers()`.

| broker | fee | min | max | day-1 sell | FX / cash-block | allocation method | eligibility (country) | operational maturity (user prior) | notes |
|---|---|---|---|---|---|---|---|---:|---|
| **Revolut** | **None** (user) | **$500** (USD, user) | **None** (user) | **No limit** (user) | $500 min is USD-denominated; reconcile FX vs a EUR budget | UNKNOWN (model assumes flat pro-rata) | usable (user) | **0.90** | User facts 2026-06-09: offers SPCX, no fee, $500 min, no max, day-1 sell allowed. |
| **Trade Republic** | **1 EUR fixed** | UNKNOWN | **None documented** | **None documented** | EUR settlement (1 EUR fee implies EUR cash); cash-block-during-book-build UNKNOWN | **flat pro-rata by subscription volume** (web-confirmed) | "European retail" (Justin's exact eligibility UNKNOWN) | 0.45 | Pro-rata + 1 EUR fee web-confirmed; ~5% float ⇒ expect low fill. Confirm min + eligibility. |
| ~~DEGIRO~~ · ~~Interactive Brokers~~ | — | — | — | — | — | — | — | — | **Out of scope (user, 2026-06-09)** — removed from the candidate set. |

**Column meaning.** *fee* = subscription/settlement cost; *min/max* = the smallest/largest subscription the broker accepts; *FX/cash-block* = currency the subscription is funded in and how cash is reserved during the book-build; *allocation method* = how the broker rations an oversubscribed book (here, all relevant brokers are flat pro-rata by assumption, vault-confirmed only for TR); *eligibility* = whether Justin's country/account can subscribe; *operational maturity* = a `(0,1]` execution-trust prior (higher = more proven to the user / lower chance of a delivery failure); *notes* = open confirmations.

**Read.** Only one row (Trade Republic) carries real facts, and even it is missing min/max and day-1 sell. The table's job right now is to make the **unknowns explicit and uniform** so the recommender treats them honestly and the fact-finding list writes itself.

### The allocation logic, encoded

Two rules, exactly as specified:

**1. Within a broker — flat pro-rata.** Every subscriber at broker `b` is filled at the **same** fraction `phi_b ∈ (0, 1]`. Your absolute shares are `phi_b × (your subscription size)`. So **size scales absolute shares, not your fill percentage** — subscribing 2× more does *not* raise `phi_b`; it doubles the shares you receive at whatever `phi_b` the broker prints.

> *Worked micro-example.* You subscribe for 100 shares at Trade Republic. The book is 4× oversubscribed, so the broker fills everyone at `phi_TR = 0.25`. You receive **25 shares**. Your neighbour who subscribed for 400 shares receives 100 — same 25% fraction, more absolute shares. If instead the book had been undersubscribed, `phi_TR = 1.0` and you'd receive **all 100** shares you asked for (this is the danger in STEP 3's warning).

**2. Across brokers — independent draws on separate sealed sub-allocations.** Each broker receives its **own sealed sub-allocation** from the underwriter, so `phi_Revolut, phi_TR, phi_DEGIRO, phi_IBKR` are **independent** random variables. The sub-allocations and each broker's internal demand are **not observable** in advance, so every `phi_b` is unknown when you decide the split.

Let `w_b` be the fraction of your subscription budget placed at broker `b` (`Σ w_b = 1`), so the subscription at `b` is `S_b = w_b · B` with `B = oversubscribe · C`. The total filled notional is:

```text
F      = Σ_b phi_b · S_b = B · Σ_b phi_b · w_b
E[F]   = B · Σ_b w_b · mu_b                      # mu_b = E[phi_b]
Var[F] = B² · Σ_b w_b² · sigma_b²                # independence ⇒ NO covariance terms
```

The disappearance of the covariance terms (because the draws are independent) is *why spreading across brokers hedges fill-rate noise* — but it is only the tie-breaker. The real driver is the fill rate itself; see [§ The core idea](#the-core-idea--estimate-the-fill-rate-go-where-you-get-filled-more) in STEP 3.

---

## STEP 3 — Capital-split recommender

### Inputs and outputs

- **Input:** total capital `C` you are willing to block; an execution-risk **`tilt ≥ 0`** (default `1.0`, which leans toward Revolut over the unproven TR); an optional **`oversubscribe`** multiple (default `1.0` = safe).
- **Output:** a recommended euro split across the *usable* brokers, respecting each broker's min/max, with the reasoning, the worst-case funded amount, and warnings.

Run it:

```bash
cd polymarket/research
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital 10000 --tilt 1.0
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital 10000 --oversubscribe 1.5   # warns
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --track-record                        # rolling fill rates
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --selftest                            # 9 built-in checks
# stdlib-only: plain `python3 scripts/eu_ipo_capital_split.py ...` also works.
```

### The core idea — estimate the fill rate, go where you get filled more

This is the whole methodology in one line. With flat pro-rata:

```text
shares_at_broker = fill_rate_at_broker × your_subscription_at_broker
```

So for a given euro, the **only** thing that decides how many shares you get is the broker's **fill rate**. A concrete fill-rate difference:

> Suppose **Revolut fills 25%** and **Trade Republic fills 10%** on the same deal. Then **€1,000 at Revolut → 250 shares**, but **€1,000 at Trade Republic → 100 shares**. *Same money, 2.5× the shares at Revolut.* You want more of your budget where the fill rate is higher.

So the recommended split is simply **proportional to fill rate**, with a `tilt` knob for how hard to chase it:

```text
weight_b  ∝  fill_rate_b ** tilt
  tilt = 0  → ignore fill rate, split EQUALLY (when you have no data / no reason to prefer one)
  tilt = 1  → split PROPORTIONAL to fill rate (double the fill rate ⇒ double the money)   ← default
  tilt > 1  → concentrate harder on the higher-fill broker (tilt → ∞ = all at the best one)
```

**Where the fill rate comes from (3 tiers, best available wins).**

1. **Realised fill rate** — the mean of your past realised fill fractions (`filled ÷ requested`) from the IPO Subscriptions logs. The real thing, once you've done a deal.
2. **Research-based prior** — a *deal-level* estimate from public evidence when you have no logs yet. For SpaceX we researched it (see next subsection): the evidence gives **~5% and shows no difference between Revolut and TR**, so the prior is **equal across brokers → a 50/50 split**.
3. **Maturity prior** (gut trust) — only used if there is no fill estimate at all. The output always labels which tier drove the split (`WEIGHTED BY: …`), so you never confuse data, researched evidence, and gut.

### How the verdict is computed (step by step)

The recommended euro split is produced by exactly five steps — nothing hidden:

1. **Score each broker.** `score_b` = realised fill rate if logged for all brokers; else the research prior if set for all; else the maturity prior. *(One tier at a time, so all scores are on the same scale.)*
2. **Weight by score.** `weight_b ∝ score_b ** tilt`, normalised to sum to 1.
3. **Scale to budget.** `subscription_b = weight_b × B`, where `B = oversubscribe × C` (default `B = C`).
4. **Respect min/max.** Clip each subscription to the broker's `[min, max]`; redistribute any residual to the unconstrained brokers. (Revolut's `$500` floor binds on small budgets.)
5. **Report.** The euro split, the worst-case funded amount, expected fill `E[F]`, and any flags/warnings.

**Trace of today's default** (`C = 10,000`, `tilt = 1`, no logs yet → step 1 uses the **research prior**):

```text
scores   = {Revolut: 0.05, Trade Republic: 0.05}          (research prior; equal, no per-broker evidence)
weights  = score^1 normalised = {0.05/0.10, 0.05/0.10} = {0.50, 0.50}
amounts  = weights × 10,000 = {Revolut 5,000, Trade Republic 5,000}
E[F]     = 0.05×5,000 + 0.05×5,000 = 500            (expect ~€500 of shares on a €10k subscription)
```

That is the verdict: **50/50, expect ~5% filled** — and it will *change to fill-rate-driven the moment you log a real deal*.

### The researched prior — why today's split is 50/50, not a Revolut tilt

This is the honest answer to "can we do anything before we have realised fills?" We researched it (web, 2026-06-09) instead of guessing:

- **Neither broker has a usable fill-rate track record.** Trade Republic's IPO product launched ~2026-06-06 (SpaceX is among its first deals); Revolut is a named access route but has **no documented past-allocation fill data**. So a *per-broker* fill prior cannot be grounded in evidence.
- **The deal-level fill is estimable and low.** SpaceX reserves up to **~30% for retail** (record-high vs the usual 5–10%), is **~2× oversubscribed** ($150B orders vs ~$75B), with a European retail allocation of **~55.6M shares** (~10% of the 555.6M offered). The one concrete retail illustration (German coverage): **subscribe €10k → ~€500, ≈ 5% fill**.
- **No source shows a fill difference *between* Revolut and TR** (Euronews: no comparative rationing data, no per-investor caps stated).

**Conclusion (this validates the earlier skepticism):** with no per-broker evidence, the fill-rate prior is **equal** across the two brokers, so **fill-rate logic does NOT justify steering — the split is 50/50.** The `~5%` level is a rough central estimate with wide uncertainty; it does not change the *split* (only the expected-fill `E[F]`). Sources: [Reuters/Yahoo — well oversubscribed](https://finance.yahoo.com/markets/stocks/articles/spacex-ipo-said-well-oversubscribed-154906500.html); [Motley Fool — what oversubscription means for retail](https://www.fool.com/investing/2026/06/09/spacexs-historic-ipo-may-be-oversubscribed-heres-w/); [Euronews — how EU retail can buy, risks](https://www.euronews.com/business/2026/06/09/spacex-ipo-how-european-retail-investors-can-buy-shares-and-the-risks-to-be-aware-of); [Finextra — record retail allocation](https://www.finextra.com/newsarticle/47871/spacex-sets-aside-record-ipo-allocation-for-retail-investors).

**So is Revolut preferred at all?** Only on **operational** grounds, not fill: no fee (vs TR's trivial €1), $500 min, confirmed day-1 sell, and your higher trust. If you want to act on that, lean Revolut deliberately as an *operational* choice (e.g. 60/40) — just know it is **not** a fill-rate edge. The tool's maturity tilt (Revolut 0.90 > TR 0.45) only re-activates if you clear the research prior (`--prior-fill-rate ""`-style: remove it), keeping the two rationales cleanly separate.

### Worked example A — today (research prior; no realised fills yet)

Actual tool output (`WEIGHTED BY: fill-rate (research-based prior — no realised data yet)`):

| broker | fill rate | basis | weight | amount | fee |
|---|---:|---|---:|---:|---:|
| Revolut | 5.0% | prior | 50.0% | 5,000.00 | none |
| Trade Republic | 5.0% | prior | 50.0% | 5,000.00 | 1 EUR fixed |
| **TOTAL** | | | **100.0%** | **10,000.00** | |

- **Equal 5.0% prior → 50/50 split.** The research found no per-broker fill difference, so there is no fill-rate reason to favour either broker.
- **Expected fill `E[F]` = 500.00** (research prior — rough, wide uncertainty): subscribe €10k total, expect *roughly* €500 of shares to actually fill, given the heavy oversubscription.
- **Flags:** only **Trade Republic eligibility** is `UNKNOWN`.
- **Basis `prior`** flags that this is researched evidence, not your own realised history — it will switch to `realised` after the first logged deal.

### Worked example B — once you have one logged deal (fill-rate-driven)

If the SpaceX deal resolves with **Revolut filling 30%** and **Trade Republic 10%**, re-running with `--use-track-record` gives (`WEIGHTED BY: fill-rate (realised track record)`):

| broker | fill rate | basis | weight | amount | fee |
|---|---:|---|---:|---:|---:|
| Revolut | 30.0% | realised | 75.0% | 7,500.00 | none |
| Trade Republic | 10.0% | realised | 25.0% | 2,500.00 | 1 EUR fixed |
| **TOTAL** | | | **100.0%** | **10,000.00** | |

- **Expected fill `E[F]` = 0.30×7,500 + 0.10×2,500 = 2,500.00** — the system now estimates how many euros of shares you'll actually get.
- **The split moved from 50/50 → 75/25** purely because the *measured* fill-rate gap (30% vs 10%) replaced the equal research prior. That is the system "learning where you get filled more" — the thing it genuinely can't do until deal #1 resolves.

**Column meaning.** *fill rate* = estimated pro-rata fill; *basis* = where it came from (`realised` log, `prior` research, or `—`); *weight* = share of the budget; *amount* = currency to subscribe; *fee* = from the comparison table.

### Why not just put 100% at the higher-fill broker?

Two caveats, both small — they are the *tie-breaker*, not the objective:

1. **The estimate is noisy.** One or two deals is a tiny sample; today's "best" broker may not be best next time.
2. **Each broker is a single sealed draw.** You can't see sub-allocations or demand, so `phi_b` is random; going 100% bets everything on one allocation.

Spreading some money hedges both. (Formally, with independent draws the variance of total fill is `Var[F] = B²·Σ w_b²·σ_b²` — no covariance terms — which equal weights minimise. That's *why* `tilt` doesn't slam everything onto the top broker, but the **objective is still: go where you get filled more.**) The `tilt` knob lets you choose: `tilt = 0` ignores the gap and spreads evenly (max diversification); higher `tilt` chases the higher fill rate harder.

> *Small-budget example — the $500 Revolut min binds.* At `C = 600, tilt = 1`, the unconstrained split would put ≈$400 at Revolut, below its $500 floor. The recommender clips Revolut **up to $500** and gives Trade Republic the **$100** residual (total still $600). If `C < $500`, the tool prints an **INFEASIBLE** warning (Revolut's minimum alone exceeds the budget). Remember the $500 is USD; if `C` is in EUR, convert the floor first.

### The over-subscription warning (built into the verdict)

Flat pro-rata creates a trap. When `phi_b` is low you receive few shares, so it is tempting to **subscribe for more than you actually want** to scale the absolute fill back up. But `phi_b` is unobservable and **can come in high (even 1.0)**. If you over-subscribe (total subscription `> C`) and the fill lands full, you are **fully allocated at the inflated size** — you must fund and hold far more exposure than you intended, and you blocked more cash than planned.

The safe default is therefore **`oversubscribe = 1.0`: total subscription == C**, so the worst case (`phi = 1` everywhere) buys *exactly* your budget. Any `oversubscribe > 1.0` is allowed but **loudly flagged** with the worst-case funded amount `= oversubscribe · C` (e.g. `--oversubscribe 1.5` on `C = 10,000` prints "if the fill lands FULL you must fund 15,000"). **The amount you subscribe is the amount you might actually have to buy.**

**SpaceX-specific reality (sharpens the warning, both ways).** Press coverage expects **heavy oversubscription** — ~30% of SPCX reserved for retail but only ~5% free float — with a worked example of *subscribe €10,000 → receive ~€500 or nothing* (web sources above). So `phi` is *likely* low, which makes over-subscribing *tempting* (low expected fill ⇒ inflate to scale absolute shares). Two cautions: (1) the tail still bites — if allocation comes in unexpectedly generous you are funded at the inflated size; and (2) **whether Trade Republic blocks the full subscription cash during the book-build is UNKNOWN** — if it does, an inflated subscription ties up far more cash than your expected fill for the whole window, for no extra expected shares once you account for the block. Net: keep `oversubscribe = 1.0` unless you have *confirmed* the cash-block behaviour and can fund the full-fill tail.

### Realism ledger (per [[CODEX]] § Realism calibration)

- **Modelled / fair assumptions:** flat pro-rata within a broker (`shares = fill_rate × subscription`); **fill-rate-proportional weighting** (with the maturity prior as a clearly-labelled placeholder until logs exist); independence of cross-broker draws (the spread-to-hedge tie-breaker); maturity priors as the *only* subjective input; total-subscription == C as the safe bound.
- **Live-only / must-confirm-by-hand (remaining unknowns, now few):** Trade Republic's **eligibility for Justin's account**, **minimum** subscription, **exact deadline**, and **whether full subscription cash is blocked during the book-build**; the **EUR/USD reconciliation** of Revolut's $500 (USD) floor against a EUR budget; and the actual `phi_b` (only observable *after* allocation — exactly what the tracking system below captures). Revolut max + day-1 sell are user-confirmed; TR max + day-1 sell are *undocumented-restriction* (absence of evidence + user concurrence), not published guarantees.
- **Honest status:** with both brokers' headline terms now in hand and a realistic *low* expected fill, this **merits a small instrumented subscription + measurement loop**: subscribe modestly, log the realised fills, and let the track record (not priors) drive the next split.

---

## STEP 4 — Reusable subscription log + rolling fill-rate track record

A dedicated, human-navigable folder **`IPO Subscriptions/`** (vault root, per the user's explicit request) holds the operational tracking system:

- **`_template.md`** — the reusable per-deal template: deal, ticker, prospectus terms (expected/max price, tranche size, per-broker deadline), subscription per broker, and post-allocation fields (realised fill, effective price, day-1 open/close, realised P&L). The realised-fill table uses fixed column names so the tool can parse it.
- **`SpaceX_SPCX_2026-06.md`** — the template **instantiated for SpaceX**, pre-filled with the known prospectus terms ($135 expected, `SPCX`, 555,555,555 Class A offered, listing anticipated 2026-06-12, ~13.076B total shares ⇒ ~$1.765T at offer). Subscription/fill fields are left blank to fill in.
- **`README.md`** — how to add a deal and how the rolling per-broker fill-rate track record is computed.

**How the rolling track record feeds back.** After each deal you record the realised fill per broker in that note's `Realised Fill (per broker)` table. Then:

```bash
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --track-record        # per-broker mean/sd/min/max phi
PYTHONPATH=. uv run python scripts/eu_ipo_capital_split.py --capital 10000 --use-track-record   # fold it into the split
```

`--track-record` scans every deal note, extracts resolved `fill_fraction` rows, and reports each broker's `n / mean / sd / min / max`. `--use-track-record` then **switches the split from the maturity-prior placeholder to the measured fill rates** — money flows to the broker that actually fills you more — and reports an actual **`E[F]`**. The loop is: *recommend → subscribe → log realised fill → learn where you get filled → recommend better next time* (OpenAI, Anthropic, …).

---

## Decision and next step

- **Verdict:** the decision model is **built, tested (10/10), and runnable**; the candidate set is **Revolut + Trade Republic** (DEGIRO/IBKR dropped per user). The researched fill prior shows **no per-broker edge**, so with `C = 10,000` the split is **50/50** and **E[F] ≈ €500**. A Revolut lean is defensible only on operational grounds, not fill.
- **Concrete next action (operational, not modelling):** confirm the last unknowns — **TR eligibility for your account, TR minimum, exact deadline, and TR cash-block-during-book-build** — and **reconcile the currency** (Revolut $500 is USD; TR is EUR). Enter any new facts into `default_brokers()` and re-run. Then subscribe modestly, and **log the realised fill** in the SpaceX deal note so the next deal's split is **fill-rate-weighted on real data** instead of the prior.
- **Do not** over-subscribe unless you have confirmed TR's cash-block behaviour and can fund the full-fill tail; convert Revolut's $500 floor into your budget currency before sizing.
