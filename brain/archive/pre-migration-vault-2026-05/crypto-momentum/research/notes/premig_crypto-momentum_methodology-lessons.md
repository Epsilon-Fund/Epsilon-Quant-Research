---
title: "Crypto momentum — methodology lessons ledger (May 2026)"
created: 2026-05-11
archived: 2026-09-09
status: archived — pre-migration vault (May 2026); historical, not current
owner: justin
project: crypto
para: archive
hubs:
  - pre-migration-vault-2026-05
tags:
  - archive
  - pre-migration-vault
  - crypto
---

> **ARCHIVED — PRE-MIGRATION VAULT (May 2026).** Historical record, imported 2026-09-09 from the standalone Cowork vault that predates this repo becoming the brain. Index: [[pre-migration-vault-2026-05]].
> **PARKED (2026-08-25).** The crypto momentum book is live, but this note is a May-2026 snapshot of it, not its current state. The live record is `docs/STRATEGY_REFERENCE.md` and [[TODO]] § Crypto — do not build on this note or quote its numbers as current.

# methodology lessons

Append-only ledger of methodology lessons that survive any single strategy decision. When a strategy gets scrapped, the *strategy* dies — these lessons should not. New entries go on top. Each entry: the principle, the concrete trigger that taught it, the source thread.

---

## 2026-05-11 — disk + session state must mutate atomically

**Principle.** Any disk write that mutates position / order state must also mutate the in-memory session_state copy in the same operation. Disk and memory must stay in sync atomically — otherwise UI fragments that re-render from session_state will display stale values until a full page reload triggers a fresh disk read.

**Concrete trigger.** Stop-loss auto-ratchet wrote `pending_stop` to `positions.json` but didn't update the shared `pos` dict the live-price fragment renders from. Fragment re-renders every 30s from session_state; showed pre-ratchet stop until full reload. Highest-impact recent production bug.

**Fix pattern.** After the disk write: `pos['pending_stop'] = sugg_stop` on the in-memory dict immediately. Treat the two writes as one atomic operation in the code, even if they're not transactionally atomic at the OS level.

**Source.** `dashboard-status.md` §3.

---

## 2026-05-11 — historical-P&L values must be snapshotted at the event

**Principle.** Any value that affects historical P&L must be snapshotted at the time of the event and read back directly. Never derive it from live config — that retroactively rewrites history when config changes.

**Concrete trigger.** `build_trade_pairs()` was calling `get_coin_capital()` which read live config. Adding a new coin to `ACTIVE_ASSETS` or changing `CAPITAL` retroactively recalculated historical trade P&L.

**Fix pattern.** Freeze `size_usd`, `coin_capital`, `capital_total`, `coin_weight` at entry time in `trades.json`. Read them back from the trade record, never from current config. Generalises: capital snapshot, leverage at fill, fee rate at fill, position size limit at fill — all snapshotted at event.

**Cross-reference.** Same principle in polymarket data: `est_bankroll_usd_30d_max_approx` is lifetime peak and explicitly NOT to be used for forward-looking sizing (would leak future capacity into past decisions). Both are forms of "don't let later state poison earlier history."

**Source.** `dashboard-status.md` §3.

---

## 2026-05-11 — schema migrations need an explicit migration script at deploy time

**Principle.** When you change the on-disk schema of a state file (positions.json, trades.json, etc.), code changes alone are not deployment-ready. The next environment that picks up the new code with old data files will silently fail — silently because the new code may parse the old format partially without erroring.

**Concrete trigger.** VPS was deployed with `positions.json` from before the FIFO position ID refactor. Code expected `{SYMBOL}_{YYYYMMDD}_{seq:03d}` keys; found legacy symbol-only keys. Silent failures in stop loss display and decision logic — no error raised, no log line, just wrong behaviour.

**Fix pattern.** Migration script runs as part of deploy. Detects old schema, transforms in place (with backup), confirms new schema. Pattern from the database world that applies just as well to JSON state files.

**Source.** `dashboard-status.md` §3.

---

## 2026-05-11 — minimise the number of running processes

**Principle.** Every additional server in a deployment is a failure point and an operational burden. When you can consolidate into a single process, do.

**Concrete trigger.** `journal_server.py` ran as a Flask app on port 5001 to handle trade logging. Conflicted with prior processes on the same port. Resolution: eliminate Flask entirely, consolidate logging into Streamlit.

**Source.** `dashboard-status.md` §3.

---

## 2026-05-11 — drawdown calcs on equity curves starting at zero must use absolute dollar drawdown

**Principle.** `max_drawdown / peak_equity` divides by zero at curve start. Compute drawdown in absolute dollars first, find the worst point, *then* express as a percentage of peak at that point — with an explicit `peak == 0` guard.

**Concrete trigger.** Max drawdown displayed as 0.00% because division by peak (starting at 0) produced `inf`/`nan`, defaulting to zero downstream.

**Source.** `dashboard-status.md` §3.

---

## 2026-05-11 — infrastructure params vs strategy params

**Principle.** Never put data-trustworthiness parameters into Optuna alongside strategy parameters. They are categorically different and mixing them is a subtle but severe overfitting vector.

- **Infrastructure params** answer "what data is trustworthy?" — universe size, min volume / age / liquidity floors, regime filter windows (SMA period, ADX threshold).
- **Strategy params** answer "how do we use that data?" — signal lookbacks, holding period, stop multipliers.

Infra params get sensitivity-tested manually at tight / base / loose configs and then frozen. Strategy params go into Optuna.

**Concrete trigger.** XS thread considered putting `TOP_N`, `MIN_VOLUME`, `MIN_AGE`, `VOLUME_WINDOW`, `SMA window`, `ADX threshold` into the parameter space. Caught and fixed before runs.

**Source.** `xs-momentum-recap.md` §3 "Parameter discipline".

---

## 2026-05-11 — filter on the data dimension you'll execute in

**Principle.** If you'll trade in venue X, filter by venue X liquidity. Cross-venue volume isn't a reliable proxy.

**Concrete trigger.** XS thread initially planned to filter the universe by spot volume but trade in perps. Spot liquidity ≠ perp liquidity; cross-venue arbitrage gaps make spot volume a noisy proxy for perp tradability. Fix: perp volume only.

**Source.** `xs-momentum-recap.md` §3 "Universe construction".

---

## 2026-05-11 — theoretical purity vs infrastructure complexity

**Principle.** When the practical difference between "clean" and "convenient" is sub-basis-point on the bars you actually trade, take convenient. Dual pipelines for marginal correctness aren't worth their maintenance cost.

**Concrete trigger.** XS thread initially proposed parallel spot+perp caches (spot for clean prices, perp for execution). Daily-bar price difference on majors is negligible; consolidated to perps-only.

**Source.** `xs-momentum-recap.md` §3 "Universe construction".

---

## 2026-05-11 — cache once, filter many

**Principle.** API-bound operations inside hot loops (especially walk-forward / CPCV inner loops) are a performance and rate-limit trap. Two-layer pattern: offline cache build → in-notebook filter reads from cache.

**Concrete trigger.** XS universe filter initially proposed to query Binance live inside the WF loop. Fix: cache parquet, filter function reads cache.

**Source.** `xs-momentum-recap.md` §3 "Universe construction".

---

## 2026-05-11 — residual momentum needs rolling beta, not single-regression-over-window

**Principle.** Blitz/Huij/Martens (2011) residual momentum uses rolling beta — each bar gets its own β from its own J-bar window, producing one scalar residual per bar. The Sharpe is then a *second* J-bar window over those scalar residuals. Warmup is therefore 2J, not J.

The naïve construction (one regression over the full J-bar window producing J residuals, then Sharpe of those) misrepresents time-varying market exposure and is not the literature-standard signal.

**Concrete trigger.** XS thread caught this misconception during the residual-Sharpe signal build.

**Empirical aside.** Residual alpha had the strongest perturbation result of the three signals tested (raw / rolling Sharpe / residual). Even if the XS strategy doesn't ship, the empirical evidence is: rolling-beta residual constructions are unusually structurally robust to parameter perturbation in crypto cross-section.

**Source.** `xs-momentum-recap.md` §3 "Signal construction".

---

## general — universe filter constrains parameter lookback

**Constraint.** `J_max ≤ UF_MIN_AGE_DAYS - 30`. Coins need a buffer of post-listing settling time before their data is trustworthy for momentum signal computation. Universe filter constrains parameter lookback, not vice versa.

**Source.** `xs-momentum-recap.md` §3 "Parameter discipline".

---

*From pre-cowork memory (preserved from prior project memory file, not always sourced):*

## pre-cowork — never blanket `dropna()` before the engine

**Principle.** Use `dropna(subset=[indicator_cols])` paired with `fillna(0)` on the position column. Blanket `dropna()` silently drops flat-position bars and corrupts Sharpe / total-return calculations.

**Reinforcement.** Enforced in `wf_engine.py:352-356` and `cpcv_engine.py:360-363` — engines never call `df.dropna()` without `subset=`. Strategies must return `(df, indicator_cols)` tuple so the engine knows which warmup NaNs are acceptable.

**Source.** Crypto-momentum engine convention (`strategy-reference.md` §G.2).

## pre-cowork — 1-bar forward shift on regime filters and signals

**Principle.** Any value derived from bar `t`'s observable data is used to size or direct trading on bar `t+1`. Enforced in `portfolio_metrics.py:83-87` and inside `xs_strategy.py:508-514` / `:651-658`.

**Source.** `strategy-reference.md` §G.1.

## pre-cowork — Sharpe / Calmar annualisation discipline

**Principle.** `sqrt(periods_per_year)` for Sharpe; Calmar uses *annualised* return (CAGR / |max DD|). The scoring-side Calmar in `_default_score` is *not* annualised — it's a hand-tuned saturation cap, do not quote it as Calmar.

**Source.** `strategy-reference.md` §D.2, §D.4, §G.3, §G.4.

## pre-cowork — Optuna trial budget heuristic

**Principle.** ≥10–20 trials per free parameter. Engine default of 400 is comfortable for ~10 free params; scale `n_trials` if you add or unfreeze parameters.

**Source.** `strategy-reference.md` §G.5.

## pre-cowork — cap iteration cycles at 3–4

**Principle.** The CPCV → parameter analysis → narrow/fix → re-CPCV loop should run at most 3–4 times per strategy/asset before declaring convergence. Beyond that, you're overfitting to the CPCV path distribution itself.

**Source.** `strategy-reference.md` §G.6.
