---
title: "Crypto momentum — todos (May 2026)"
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

# crypto momentum — todos

## now — production (live trading)

1. [ ] **Fix VPS positions.json + verify stop loss end-to-end.** Mate scp's the four corrected JSON files onto VPS (positions, realised_capital for momentum + clean state for statarb and bbbreakout). After scp: verify the stop loss confirm button persists correctly across page refreshes on the live VPS (not just locally). Only remaining known production bug.
2. [ ] **Run full BB Breakout optimisation and activate.** `python3 dashboards/bbbreakout/optimise.py` for each coin in the BB universe with production `n_trials`. Commit resulting `live_params.json`. Push to VPS. Verify dashboard shows correct signals. Makes the second strategy live.
3. [ ] **VPS security hardening.** nginx reverse proxy + Let's Encrypt SSL + basic auth. Single focused Claude Code prompt covering `nginx.conf`, `docker-compose.yml` update, certbot setup. Required before treating dashboard as production-grade or sharing access.
4. [ ] **Commit the uncommitted cc-crypto work** (from 2026-05-11 evening session). `git status` / `git add` / commit — covers live-tendency fragment, BB live-WS manual arm, WS resilience pass, ENTRY backfills, `build_trade_pairs` position_id matching, display dp tightening.
5. [ ] **BB BTC price formatting fix.** Pass `symbol` to `_fmt_price` in bb's Decisions table cells so BTC renders at 2dp instead of 3dp. Trivial follow-up.
6. [ ] **Capture WS "stuck 🟡" terminal slice next recurrence.** Reconnect button + `force_reset_shared_ws()` already in place. Next time the persistent 🟡 fires, grab the terminal output around the failure to decide auto-recovery strategy (longer join timeout vs N-failure auto-reset depends on which error message appears). Investigation currently paused.
7. [ ] **Binance API read-only sync — still on hold.** Per Justin's call. Recommendation stands: stages 1–2 are pure upside, stage 3 (submit-mode) is the scary one. Revisit when ready.

## now — research

- [ ] **Parameter derivation** — collapse related stop multipliers into 2-3 derived params, reducing momentum WF from 18 → ~10 free params. Blocking a clean final re-optimisation; current `live_params.json` was generated with the bloated set. Plateau analysis already showed 9/15 returning N/A + 40-55% perturbation degradation — strong evidence the fix is needed.
- [ ] **Execution-hour P&L verification.** Test the `execution_cumulative` toggle in portfolio page against real trades (4 open positions + closed history now available). Pre-req: confirm hourly cache is populated for entry dates May 6-10.
- [ ] **BNB diagnostic.** Present in `momentum_swing` strategy registry but absent from `ACTIVE_ASSETS`. Decide: re-include or remove from registry.

## next

- [ ] **275-day OOS window** for final momentum validation — gated on parameter derivation completing first
- [ ] **Trade log statistics** — limited real history right now; sections will become meaningful with more trades
- [ ] **Fund-level portfolio chart** — Prompt 6 ran but theoretical-toggle on fund equity intentionally skipped; full multi-strategy fund portfolio not yet verified
- [ ] **Website integration** — GitHub Pages portfolio summary push

## later

- [ ] Stat arb strategy implementation (currently empty shell)
- [ ] Hyperliquid vault deployment workflow (gated on champion strategy with sustained live track + clean CPCV)
- [ ] Intraday flow / imbalance signals (cumulative delta, order book imbalance) — backlog idea
- [ ] Dynamic portfolio re-weighting on multi-sleeve combiner (currently static across OOS window)

## done

- [x] WF engine + CPCV engine + portfolio modules — stable testing infra (`strategy-reference.md`)
- [x] STRATEGY_REFERENCE.md authored by Claude Code (2026-05-11)
- [x] Live trading dashboard built and deployed to VPS (momentum strategy active, 4 open positions)
- [x] FIFO position tracking + capital snapshot consistency + realised capital tracking shipped
- [x] xs-momentum-recap.md captured — preserves thread reasoning + empirical findings even if scrapped
- [x] methodology-lessons.md extracted — durable lessons that survive strategy decisions
- [x] dashboard-status.md captured (2026-05-11) — operational state of VPS-deployed system
