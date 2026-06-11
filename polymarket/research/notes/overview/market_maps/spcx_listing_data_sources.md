---
title: "SPCX Listing-Day Screens Runbook — Where to Watch the IPO Cross, Real-Time Prints, and L2 from the EU (Block S3)"
tags: [spacex, spcx, ipo, nasdaq, noii, tradingview, trade-republic, ibkr, data-sources, findings]
created: 2026-06-10
status: "done — desk research verified 2026-06-10; HUMAN-CHECK items listed in §7"
audience: "Justin + Alvaro, Friday 2026-06-12 listing day; replaces §6 of Alvaro's SpaceX_Execution_Playbook.docx"
---

# SPCX Listing-Day Screens Runbook — Where to Watch the IPO Cross, Real-Time Prints, and L2 from the EU

> Hub: [[spacex_ipo_market_map_handoff]] · [[POLYMARKET_BRAIN]] · [[COWORK]]
> Companions: [[spcx_listing_day_gameplan]] (§4 cross mechanics, §5.3 interim screen map — this note supersedes that table) · [[spcx_convergence_calc_findings]] (S1 thresholds used in the alert list)
> This is **Block S3** of the gameplan's §7. Desk research only; every claim carries a source + access date (all accessed **2026-06-10** unless stated). Unverifiable items are marked **HUMAN-CHECK** with the exact question to resolve.

## Plain-English Summary

- **What this is.** The verified "which screen do I look at, and what does it cost" runbook for SPCX listing day (Fri 2026-06-12), for an EU retail trader (Trade Republic + Hyperliquid + free web tools). It replaces §6 of Alvaro's playbook, whose screen list assumed feeds we either can't access from the EU or that are delayed.
- **Headline answers.** (a) There is **no practical free EU retail screen for the official Nasdaq NOII / indicative clearing price** — it lives inside Nasdaq TotalView, and the retail brokers that surface it (Webull, IBKR) either don't serve Justin's country or need an already-funded account. The working proxy is **xyz:SPCX perp + CNBC/newswire "indicated to open $X" headlines**, which relayed indications in near-real-time for Cerebras. (b) TradingView free tier **is** 15-min delayed for Nasdaq primary data, but **every account gets real-time Cboe BZX prints free** — good enough seconds after the cross; the true Nasdaq real-time add-on is **$3/mo but requires a paid plan or the 30-day trial**. Trade Republic's in-app quote is real-time but from LS Exchange (order-entry venue only — rule zero stands). (c) Only realistic EU L2 is **IBKR TotalView (~$15/mo non-pro)** *if an IBKR account already exists*; otherwise no L2 — use the Hyperliquid book as the depth proxy. (d) Anchored VWAP from the first print is a **free TradingView drawing tool** (not the session VWAP indicator); exact setup in §5.
- **D1 input (Thursday night):** the final price will hit **newswires minutes after pricing Thursday evening; the 424B itself often only lands on EDGAR Friday pre-market** — watch CNBC/Reuters first, keep an EDGAR RSS alert as backstop (§6).
- **Status.** Block S3 complete. Seven HUMAN-CHECK items remain (§7) — all are "open the app and look" checks, none block the plan.

---

## 1. The day in feeds (the runbook proper — replaces Alvaro §6 and gameplan §5.3 table)

All times CET. **Rule zero unchanged:** Trade Republic is order-entry only, never a price feed or chart.

| window | primary screen | backup / cross-check | what you're reading |
|---|---|---|---|
| 8:00–15:30 (allocation → bell) | Hyperliquid `xyz:SPCX` — app.hyperliquid.xyz/trade/SPCX, free, no account needed for the chart/book | `spcx_convergence_calc.py --watch` terminal; Polymarket SPCX markets (S5 monitor) | Only live SpaceX price. Level vs $135/final; liq buffer if short is on |
| 15:30 → cross (display-only period) | **CNBC live / newswire headlines for "indicated to open"** + xyz perp | TR app: check if LS Exchange shows a pre-print SPCX quote (HUMAN-CHECK #5); IBKR TWS NOII *if* account + TotalView (HUMAN-CHECK #3) | Indication range vs perp vs PM mean = best available pre-trade truth (we cannot see official NOII directly — §2) |
| cross → 22:00 | TradingView SPCX, 1m + 5m, **Cboe BZX real-time feed (free)** — setup in §5 | Hyperliquid perp vs listed price; TR app only to place sell tickets | Anchored VWAP from first print, volume at highs, lower lows, spread, perp–spot divergence (gameplan §5.3 indicator set) |
| all day | TradingView alert list (§5.4) on phone + desktop | — | $135 / final / $162 / $171 S1 trigger / $140 / $125 levels |

Read: the only structural blind spot is the official NOII indicative price during the display-only window. §2 shows why, and why the proxy stack (perp + TV headlines) is the honest substitute rather than a degraded one.

---

## 2. (a) NOII / indicative clearing price — can EU retail see it?

**What NOII is and where it lives.** The Net Order Imbalance Indicator (paired shares, imbalance, indicative clearing price) is disseminated **only inside Nasdaq TotalView**; during an IPO quoting period it updates every second from 1 second after the quote period starts ([Nasdaq Order Imbalance Snapshot spec v2.2](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NOIS_v2.2.pdf), accessed 2026-06-10; [Nasdaq TotalView product page](https://www.nasdaq.com/solutions/data/equities/nasdaq-totalview), accessed 2026-06-10). It is not on any free public Nasdaq webpage.

**Checked routes, in order of practicality:**

| route | NOII? | EU-workable? | cost | verdict |
|---|---|---|---|---|
| Nasdaq's own "IPO Indicator" web tool | yes (order-level) | **no — member firms only** (access via Nasdaq IPO Execution desk) | n/a | dead end ([classic.nasdaqtrader.com IPO Indicator](https://classic.nasdaqtrader.com/trader.aspx?id=IPOIndicator), accessed 2026-06-10) |
| Webull L2 (TotalView incl. NOII panel) | yes — NOII shown below the order book with Level 2 Advance | **no for Justin** — Webull EU requires **Dutch residency** (UK separately); not offered in Germany/rest-of-EU | free promo/cheap in served markets | dead end unless NL-resident ([Webull NOII courseware](https://www.webull.com/learn/courseware/yFYqot/How-does-Auction-Crossing-NOII-Work); [investingintheweb on Webull EU availability](https://investingintheweb.com/investment-apps/webull-europe/), both accessed 2026-06-10) |
| IBKR TWS + Nasdaq TotalView sub | TotalView **carries** NOII; whether TWS *displays* the IPO indicative-price fields retail-side is not confirmed in public docs | yes (IBIE serves EU; data subs need ≥$500 equity) | "NASDAQ TotalView-OpenView" ~**$15/mo non-pro** (older sources; the $3.50 "BX TotalView" line is the small Boston book, not this) | **only viable EU route, and only if an IBKR account already exists** — a new account won't be open+funded by Friday. HUMAN-CHECK #3 on display + exact price ([brokerage-review IBKR L2 pricing](https://www.brokerage-review.com/expert/level2/interactive-brokers-level-2-quotes.aspx), accessed 2026-06-10) |
| eSignal / pro terminals | yes | yes | $$$ (pro-tier) | not worth it for one day ([eSignal NOII KB](https://kb.esignal.com/hc/en-us/articles/6362096311323-Net-Order-Imbalance), accessed 2026-06-10) |
| moomoo (free TotalView in its markets) | yes | **no — not offered in the EU** | — | dead end |
| Financial TV / newswires | indirectly — anchors relay "indicated to open $X–$Y" during mega-IPO display periods | yes, free | free | **the practical proxy.** Cerebras precedent: indication updates circulated through the morning before the 12:59 ET first print ([techstackipo Cerebras trading-day log](https://www.techstackipo.com/ipo/cerebras/trading-day), accessed 2026-06-09/10 — already cited in [[spcx_listing_day_gameplan]]) |

**Verdict (a):** nowhere practical, free or cheap, for the official NOII from the EU by Friday. **Pre-registered proxy stack:** (1) `xyz:SPCX` perp — Cerebras's perp pre-discovered the cross within ~2% from ~2h out (vault, [[spcx_convergence_calc_findings]]); (2) CNBC/Bloomberg/Reuters/X "indicated to open" headlines, which are journalists reading the same NOII we can't see; (3) if Justin has a live IBKR account, spend the ~$15 and try TWS (HUMAN-CHECK #3). A practical example of what this looks like: at ~16:40 CET a CNBC anchor says "SpaceX indicated $148–$152, next update 17:05" while the perp prints $151 — that pair of numbers is our working indicative price; treat a perp far outside the relayed indication range as the perp being wrong, not the auction.

---

## 3. (b) Real-time SPCX prints

**TradingView — confirmed:** on the free plan, primary-exchange data (Nasdaq/NYSE) **is delayed up to 15 minutes**, but **every account, free included, streams real-time Cboe BZX prints for US stocks** at no cost. True Nasdaq-primary real-time is a **$3.00/mo non-pro add-on** (or $9.95/mo US bundle), and the add-on marketplace is **only purchasable on a paid plan or during the 30-day free trial** ([financialtechwiz TradingView real-time data guide, updated 2026-04-16](https://www.financialtechwiz.com/post/tradingview-real-time-data/); [TradingView: purchasing additional market data](https://www.tradingview.com/support/solutions/43000471705-how-to-purchase-additional-market-data/), both accessed 2026-06-10).

What this means for Friday, concretely:

- **Free BZX is good enough post-cross.** The opening cross itself is a single Nasdaq print, but continuous trading starts on all venues the moment the stock is released — BZX prints will track the tape within seconds and pennies. For reading anchored VWAP, volume divergence, and lower lows on 1m/5m bars, BZX real-time ≫ Nasdaq 15-min-delayed.
- **Pre-cross, every chart is empty anyway** — the stock is halted on all venues during the display-only period, so no feed tier changes anything before the first print.
- **Belt-and-braces option (~€0):** activate the TradingView 30-day trial Thursday, add the $3 Nasdaq feed → true primary real-time incl. the official opening print, cancel after. Recommended if alerts matter (free tier is also alert-starved, §5.4).
- One known free-tier gap: the **official opening-print price itself** appears on the delayed Nasdaq feed 15 min late; you'll see the immediate post-cross BZX prints in real time but should take the official cross price from the newswire headline (it's reported instantly for an IPO this size).

**Trade Republic app — confirmed with a caveat:** TR shows **free real-time quotes** in-app ([TR support: "Why don't my quotes update?"](https://support.traderepublic.com/en-sk/756-Why-don't-my-quotes-update), accessed 2026-06-10), but the quote is the **LS Exchange (Lang & Schwarz, Hamburg) market-maker quote**, not the Nasdaq tape ([bankeronwheels TR review](https://www.bankeronwheels.com/trade-republic-review/); [brokerchooser TR stocks review](https://brokerchooser.com/broker-reviews/trade-republic-review/trade-republic-stocks), both accessed 2026-06-10). It's real-time *to LS*, mirrors the US reference price during US hours, and is fine for placing limit sells — but spreads and prints are the market maker's, so it stays banned as a chart/tape source (rule zero). Latency app-side is connection-dependent (TR's own support note); no published quantitative latency figure exists — treated as "real-time, venue-shifted."

**Possible bonus, unverified:** German market makers have historically quoted hot US IPOs **before** the US first print (gray-market "per Erscheinen"-style quoting). If LS quotes SPCX between 15:30 and the cross, the TR app becomes a second free indicative-price proxy. Not confirmable from published docs → **HUMAN-CHECK #5** (zero effort: open the app Friday at 15:35 and look).

---

## 4. (c) L2 depth that works from the EU

Short list after eliminating non-EU brokers (§2 table):

1. **IBKR (Interactive Brokers Ireland for EU residents)** — the only realistic retail L2: Nasdaq TotalView non-pro ~$15/mo (plus possibly Level 1 UTP ~$1.50), needs ≥$500 account equity for data subscriptions, activates from Client Portal usually within minutes ([brokerage-review](https://www.brokerage-review.com/expert/level2/interactive-brokers-level-2-quotes.aspx), accessed 2026-06-10). **Only actionable if an account already exists and is funded** — HUMAN-CHECK #4.
2. **Webull** — free/cheap TotalView incl. NOII, but EU = Netherlands-resident only; UK separate entity. Dead end for Germany.
3. **moomoo** — free TotalView in its served markets; not available in the EU. Dead end.
4. **TradingView** — sells real-time *prints*, not depth; its DOM panel needs a connected broker. Not an L2 route here.

**Verdict (c):** IBKR or nothing. If nothing: the **Hyperliquid `xyz:SPCX` order book** (free, browser) is the only live depth we can see all day — it's depth in the perp, not the stock, but for the decisions this plan actually takes (spread-widening warning, perp-side liquidity for the pair-close) it is the relevant book anyway. L2 on the stock is a nice-to-have, not a gate, for both sleeves: no rule in [[spcx_listing_day_gameplan]] §5 requires stock-side depth — spread width (signal 4) is readable from the TV 1m chart's bid/ask line or the newswire.

---

## 5. (d) Exact TradingView setup (do this Thursday evening)

### 5.1 Account + data

- Log in (create free account if needed). **Recommended:** start the 30-day trial of any paid tier Thursday → unlocks (i) the $3/mo Nasdaq real-time add-on, (ii) multi-chart layout, (iii) a usable alert budget. Cancel after the weekend. Sources as in §3.
- If staying free: accept that the chart runs on real-time **Cboe BZX** (fine) and budget only ~3 alerts (§5.4).

### 5.2 Charts

- Symbol: **NASDAQ:SPCX** (delayed feed) / **BZX:SPCX** equivalent is selected automatically for free real-time — check the data-source label at the bottom of the chart. The symbol may not resolve until listing morning → HUMAN-CHECK #6.
- Layout: **1m and 5m side-by-side** if on trial/paid (multi-chart layout). On free (1 chart per tab): open **two browser windows**, one 1m + one 5m.
- Indicators per chart (free tier allows 2): keep it to **Volume** (default pane) and nothing else — the trend filter is the anchored VWAP *drawing*, which does not consume an indicator slot.

### 5.3 Anchored VWAP from the first print — the critical step

Standard session VWAP anchors at 9:30 ET and is **wrong for an IPO** (it would average over nothing until the cross, then behave like cross-anchored — but auto-anchored "Session" variants and screenshots people share will mislead; make the anchor explicit):

1. Left-hand **drawing toolbar** → search/select **"Anchored VWAP"** (it is a *drawing tool*, available on **all plans incl. free**, not an Indicators-menu item) ([TradingView: Anchored VWAP drawing tool](https://www.tradingview.com/support/solutions/43000669764-anchored-vwap-drawing-tool/); [optimusfutures how-to](https://optimusfutures.com/blog/anchored-vwap-tradingview/), both accessed 2026-06-10).
2. On the **1m chart, click the first printed candle** (the cross print) the moment bars start. That candle *is* the IPO cross + first continuous minute — the anchor we pre-registered in [[spcx_listing_day_gameplan]] §5.2.
3. Repeat on the 5m chart (click the first 5m bar).
4. Do **not** use the "VWAP Auto Anchored" indicator with anchor = Session for this ([TradingView: VWAP Auto Anchored](https://www.tradingview.com/support/solutions/43000652199-vwap-auto-anchored/), accessed 2026-06-10) — explicit click-anchor only, so we know exactly what the line means.

Worked example of the read: cross prints $151 at 17:42 CET; by 18:30 the 1m anchored VWAP sits at $153 and price is $149 after a lower low — that is "price below anchored-VWAP + structure broken" = fade underway → accelerate residual-sleeve tranches (gameplan §5.3 signals 1+3).

### 5.4 Alert list

Free tier is alert-starved — recent sources put the free budget at **~3 active price alerts** (older pages say 1–5; TradingView has tightened this over time; [tv-hub guide, 2026](https://www.tv-hub.org/guide/tradingview-alerts-setup); [TradingView pricing](https://www.tradingview.com/pricing/), accessed 2026-06-10). Exact current free budget = HUMAN-CHECK #7 (visible the moment you create alerts). Priority-ordered list — on free, set only the top 3; on trial/paid, set all:

| priority | alert level | meaning / source of threshold |
|---|---|---|
| 1 | **$171** (crossing up, pre-Friday on xyz perp via TV's HL listing if available, else set on SPCX after listing) | S1 frozen pre-hedge trigger Z* ≈ $36 over $135 ([[spcx_convergence_calc_findings]] § Block S1) |
| 2 | **$135** (crossing down) | offer price = residual-sleeve underwater line; near Alvaro's $140/$125 ladder |
| 3 | **final price** (set Thursday night once the 424B/newswire number is known) | D1 input; re-anchors every basis calc |
| 4 | $162 (crossing up) | EU prospectus ceiling — basis≈0 line for the hedge sleeve |
| 5 | $140 (crossing down) | Alvaro "reassess" stop |
| 6 | $125 (crossing down) | Alvaro hard stop — sell everything |
| 7 | opening print ±10% (set manually minutes after the cross) | tranche-clock context for S2 windows |

Phone + desktop notifications on; TR app is where the resulting order gets placed, never where the signal is read.

---

## 6. D1 input — fastest way to see the final price Thursday night

**Newswire beats EDGAR.** Pricing is decided Thursday evening and hits **CNBC/Bloomberg/Reuters within minutes** (Cerebras's $185 pricing was evening-news, not filing-first), while the binding **424B(b)(4) prospectus may legally follow up to 2 business days later and typically lands on EDGAR pre-market the next morning**. EDGAR itself disseminates accepted filings in near-real-time (acceptance window runs to 22:00 ET) and exposes a per-company RSS feed ([SEC RSS feeds](https://www.sec.gov/about/rss-feeds), accessed 2026-06-10; real-time third-party alert layers: [kfilings](https://kfilings.com/), [StockTitan live SEC feed](https://www.stocktitan.net/sec-filings/live.html), both accessed 2026-06-10). **Do both:** watch CNBC/Reuters/X from ~22:00 CET Thursday for the headline number, and set the EDGAR backstop — company filings RSS for **CIK 0001181412 filtered to form 424B** (URL pattern: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001181412&type=424B&output=atom`) — to catch the binding document and any surprise terms. The headline is the D1 trigger; the filing is the verification.

---

## 7. HUMAN-CHECK list (exact questions, none blocking)

1. **CNBC availability:** do you have a way to watch CNBC live (TV package / cnbc.com stream) Friday 16:00–20:00 CET? If not, X/Twitter search "`SPCX indicated`" is the fallback headline feed.
2. **IBKR NOII display:** *if* an IBKR account exists — after subscribing to "NASDAQ TotalView-OpenView" in Client Portal, does TWS show the imbalance/indicative-price fields for a halted/IPO symbol? (Test Thursday on any halted stock or Friday pre-cross on SPCX.) Public docs confirm TotalView *carries* NOII but not the retail TWS display path.
3. **IBKR TotalView exact 2026 price:** Client Portal → Settings → Market Data Subscriptions — confirm the non-pro price (~$15/mo per older sources; the $3.50 item is BX TotalView, the wrong book).
4. **Does Justin have a funded IBKR account at all?** If no → §4 verdict collapses to "no stock L2; use the HL book," which the plan tolerates.
5. **LS Exchange pre-print quote:** open the TR app Friday 15:35 CET — is there a live SPCX quote before the Nasdaq cross? If yes, log it next to the perp; it's a free second indicative price.
6. **TradingView symbol resolution:** does NASDAQ:SPCX resolve Thursday night or only Friday? (Set the alert list as soon as it resolves.)
7. **Current free-tier alert budget:** TradingView has repeatedly tightened free alerts (sources conflict, 1–5); the cap shows when you create the 2nd/4th alert. If <4, take the trial.

## 8. Decision and next step

- **Gate outcome:** Block S3 complete — the §1 runbook replaces Alvaro's §6 and the interim table in [[spcx_listing_day_gameplan]] §5.3. No paid subscription is *required*: the zero-cost stack (HL perp + newswire indications + TradingView free BZX real-time + anchored-VWAP drawing) covers every pre-registered decision rule. The one cheap upgrade with real value is the **TradingView 30-day trial + $3 Nasdaq add-on** (alerts + primary feed), recommended Thursday.
- **Next:** Blocks S2 (unwind tape calibration), S4 (TR execution mechanics), S5 (PM-PDF monitor) per [[spcx_listing_day_gameplan]] §7; the human runs the §7 HUMAN-CHECK list (items 1, 5, 6 are Friday-morning glances; 2–4 only matter if IBKR exists).

> Not investment advice — operational data-source notes for a personal-size position. All access dates 2026-06-10; re-verify prices/policies after the SPCX event window.
