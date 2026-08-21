---
title: "MM Join-2 C5 — Operator Guide: running the live 5-share fill loop yourself"
created: 2026-07-09
status: active
owner: justin
project: polymarket-mm
hubs:
  - strat_market_making
  - COWORK
tags:
  - market_making
  - runbook
  - execution
  - live-loop
---
# MM Join-2 C5 — Operator Guide: running the live 5-share fill loop yourself

> Hub: [[strat_market_making]] · [[COWORK]] · Parent runbook: [[MM_JOIN2_RUNBOOK]] · Fragility/verdict: [[mm_join2_model_fragility_findings]]

> ⚠️ **SUPERSEDED PENDING REDESIGN (2026-07-12).** A live Step-1 run proved the order path works (a real 5-share bid rested then auto-cancelled) but surfaced two design flaws: **(A)** you can't rest an ask without inventory, so real two-sided event MM needs a **YES-bid + NO-bid** paired quoter, not both sides of one token; **(B)** the quoter **churns the bid away on every mid tick** so it never persists. **Do not run Step 2 as written.** Step 1 below is still valid as a *"prove a real order rests"* smoke (it did). The redesign is a Cowork consolidation → see [[2026-07-12_mm_join2_c5_live_session_and_step1_redesign]].

## What this is (read first)

This is the **one step the agent will not run for you**: placing **real limit orders that can fill** is executing a trade, so **you** run C5 in your terminal. Everything else is done — the order path is rebuilt + tested, latency is measured (160 ms), the market is picked. C5 places a **5-share** two-sided quote (the venue minimum) on **one politics market** and waits for a fill. Money at risk is a **few dollars**, bounded by a hard 5-contract inventory cap.

**You run these commands in a terminal** (Terminal.app, or a Claude Code session running in your terminal — either way your hand is on it). Copy-paste the blocks verbatim.

- **Step 1** = place your first real resting quote, eyeball it on the site, cancel. Proves the live path works end-to-end. ~30 seconds of your attention.
- **Step 2** = let it re-quote for a while to actually catch a fill or two. Minutes to a while, manually stopped.
- When you have fills, tell the agent — it does C6 (calibration) from the journal.

## The market

Two candidates in the **Brazil Presidential Election** NegRisk event (both mid-life, resolve 2026-10-04, active, tradable). Pick ONE and use its two `export` lines in the commands below.

| | **Lula (RECOMMENDED — wider spread)** | Flávio (alternative — lower toxicity) |
|---|---|---|
| condition_id | `0xdf8e2dc5860027decbe6164555c3c1c9645c3bd33e16b9dc57ca87125047d4a8` | `0x1a01bf78f56a507fcb666d564d8c8b91b0750679163ed6e96746102c9b7d285d` |
| YES token | `30630994248667897740988010928640156931882346081873066002335460180076741328029` | `109876868437950584369987384406356259939519193117253465815665152916226511121427` |
| touch | bid 0.60 / ask 0.61 (**1.0¢ spread**) | bid 0.232 / ask 0.233 (0.1¢ spread) |
| tick | **0.01** → `HALF_SPREAD=0.005` | 0.001 → `HALF_SPREAD=0.0005` |
| 5-share cost | BUY ~$3.00 / SELL ~$3.05 | BUY ~$1.16 / SELL ~$1.17 |
| maker half-spread to capture | **0.5¢** (10× Flávio) | 0.05¢ (razor-thin) |
| caveat | ~60¢ frontrunner → **more adverse selection** (news-driven); deep queue (~51k ahead) | thin edge, but ~23¢ and less informed flow |

**Why Lula for a measurement loop:** you want to observe *real* half-spread capture vs *real* adverse selection — a 1¢-spread frontrunner is where you actually learn something. A 0.1¢ spread has nothing to measure. If you'd rather minimise toxicity for the very first fill, use Flávio.

## Before you start — 10-second sanity

- You have ~$109 on the Polymarket account (`@jamonator`). The account can trade on the site.
- The kill-switch file does **not** exist: `ls /tmp/polymarket_killswitch` should say "No such file". (To emergency-halt at any time: `touch /tmp/polymarket_killswitch` — no new orders go out, existing ones can still cancel.)
- You're in the repo root: `cd /Users/justiniturregui/Desktop/github/epsilon-quant-research`

---

## STEP 1 — place your first REAL resting quote (safe, confirm-per-order)

This places **one two-sided quote set** (a bid + an ask), asking you to type `yes` for each. After 2 orders the budget is spent and it just holds the quotes resting. You eyeball them on the site, then Ctrl-C.

```bash
cd /Users/justiniturregui/Desktop/github/epsilon-quant-research
set -a; . polymarket/execution/.env; set +a

# --- market: LULA (recommended). For Flávio, swap these two lines per the table. ---
export POLYMARKET_MAKER_CONDITION_ID=0xdf8e2dc5860027decbe6164555c3c1c9645c3bd33e16b9dc57ca87125047d4a8
export POLYMARKET_MM_BRIDGE_ASSET_ID=30630994248667897740988010928640156931882346081873066002335460180076741328029
export POLYMARKET_MM_BRIDGE_HALF_SPREAD=0.005
export POLYMARKET_MM_BRIDGE_TICK=0.01
# --- (Flávio instead: CONDITION_ID=0x1a01bf..7d285d, ASSET_ID=10987686..1427, HALF_SPREAD=0.0005, TICK=0.001) ---

export POLYMARKET_VENUE=real
export MAKER_SIZE_CONTRACTS=5
export POLYMARKET_MM_BRIDGE_LATENCY_MS=160
export POLYMARKET_MM_BRIDGE_MAX_INVENTORY=5
export POLYMARKET_MM_BRIDGE_RECONCILE_EVERY=100
export POLYMARKET_MM_BRIDGE_THROTTLE_MS=1000
export POLYMARKET_MAX_REAL_ORDERS=2
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true

PYTHONPATH=. uv run --no-project --with py-clob-client --with websocket-client --with requests \
    python -m polymarket.execution --mode mm_bridge
```

**What you'll see and do:**

1. A **startup banner**. Check it says `venue=real … size=5.0 … latency_ms=160.0 max_inventory=5.0 operator_confirm=True`. **If you see any `WARNING` line, Ctrl-C and tell the agent.**
2. A **confirm prompt** for the first order, e.g. `[operator confirm] mm_bridge order: BUY 5 @ 0.60 …` → type `yes` and Enter to place it (costs ~$3), or anything else to decline.
3. A second prompt for the `SELL 5 @ 0.61` side → `yes`.
4. After those two, the budget is spent — it holds the quotes and stops prompting.
5. **Open polymarket.com → the Lula market → your open orders.** You should see your **5-share bid at 0.60 and ask at 0.61**. That is the proof the live path works.
6. Press **Ctrl-C**. It cancels the resting quotes. Refresh the site: **0 open orders.**

✅ If you saw your orders on the site and they cancelled cleanly, Step 1 is done — the real order path is proven.

⛔ **If the first order is rejected** with something like `not enough balance` / `not enough allowance`: stop, and paste the exact `errorMsg` to the agent. It likely means the on-chain USDC **allowance** for the exchange isn't set (only you can set it, on the site — usually the first manual trade / an "enable trading" prompt does it). Don't retry until it's resolved.

---

## STEP 2 — catch a first fill or two

A resting quote only fills when someone trades through it, and it drops off when the mid moves (Step 1's budget of 2 goes flat after one move). To actually land a fill, let it **keep re-quoting** — raise the order budget and run it a while. Pick ONE of the two modes below.

### Mode A — you approve every re-quote (safest, more typing)

Same as Step 1 but with a bigger budget. Change **only** these two lines:

```bash
export POLYMARKET_MAX_REAL_ORDERS=30
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=true
```

Then run the same `uv run … --mode mm_bridge` command. You'll get a `yes/no` prompt every time the quote refreshes (throttled to ~once/second). Tedious, but you approve every dollar. Keep going until you get a fill, then Ctrl-C.

### Mode B — autonomous for the session, you watch and stop (less typing, your risk call)

This lets it re-quote on its own, bounded by the order budget **and** the hard 5-contract inventory cap. Change these two lines:

```bash
export POLYMARKET_MAX_REAL_ORDERS=40
export POLYMARKET_REQUIRE_OPERATOR_CONFIRM=false
```

Then run the same command. It quotes/re-quotes automatically. **You watch the terminal.** Your safety net is: worst-case position is exactly ±5 contracts (~$3), the order budget caps total submits, and `touch /tmp/polymarket_killswitch` (in another terminal) halts it instantly. Stop with Ctrl-C the moment you have a fill or two.

> Mode B is a genuine "put a few real dollars at risk unattended-ish" decision — it's yours. The caps make the worst case a few dollars, but only you can authorise running without per-order confirmation.

### When a fill happens

You'll see a `MAKER_FILL_TELEMETRY` line in the terminal and the position show up on the site. Once you have **one or two fills**, press Ctrl-C (check the site shows 0 open orders afterwards) and **tell the agent "got fills"**. It will read `journal_logs/mm_bridge-<today>.jsonl`, run the first-fill audit, and do the C6 calibration.

## Stop conditions (halt if any of these)

- The banner shows an unexpected `WARNING`, or a `RISK_HALT` you didn't expect — especially `ambiguous_submit` (a submit whose outcome is unclear). If you see it: `touch /tmp/polymarket_killswitch`, Ctrl-C, check the site for stray orders, and tell the agent before restarting.
- Inventory pinned at the cap (one side stops quoting) for a long stretch — the book is one-way; stop.
- Fills clustering right after a Brazil-election news headline — that's adverse selection; note it and stop.
- Any order on the site you don't recognise. The periodic reconcile should cancel orphans, but verify.

**Emergency halt, always:** `touch /tmp/polymarket_killswitch` then Ctrl-C. To resume later: `rm /tmp/polymarket_killswitch`.

## What's bounded (so you can relax about the downside)

- **Order size:** 5 shares, always. ~$3 per order on Lula.
- **Inventory:** hard cap of 5 contracts. Worst-case position ≈ ±$3, regardless of how long it runs.
- **Order count:** `POLYMARKET_MAX_REAL_ORDERS` is a hard per-run ceiling on submits.
- **Kill switch:** one `touch` command stops all new orders instantly.
- **Frozen safety:** the real-order gate (real venue ∧ raised budget ∧ — in Mode A — per-order confirm) is enforced in code, not by this doc.
