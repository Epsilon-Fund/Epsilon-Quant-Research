# Epsilon Skills Library (working name: `rigorkit` — provisional)

A public-facing library of research-rigor tooling extracted from Epsilon's
internal quantitative research stack: portable **agent-skill bundles**
(SKILL.md, [Agent Skills spec](https://agentskills.io)) and **pip-installable
Python modules**. Built here in a subfolder, designed to split into its own
repository (`git subtree split library/`) once mature.

## The one non-negotiable rule

**The library never imports Epsilon internals** — no `infrastructure.*`,
`polymarket.*`, `live_trading.*`, `topics.*`, or vault paths. Each package
imports only the stdlib and its own declared third-party deps. Epsilon consumes
the library (one-way), never the reverse. Enforced per package by a
`tests/test_decoupling.py`.

## Packages

| package | what it does | status |
|---|---|---|
| [`changepoint/`](changepoint/) — `rigorkit-changepoint` | causal, lookahead-free structural-break detection (CUSUM / Page-Hinkley / BOCPD) + purged-CV embargo, causal regime features, trend gate; ships the `changepoint-audit` skill bundle | v0.1.0 — extracted, tests green, dogfooded, scrub APPROVED |
| [`calibrate/`](calibrate/) — `rigorkit-calibrate` | Brier + Murphy decomposition, reliability diagrams with Wilson bands, ECE/MCE, Spiegelhalter's Z, isotonic/Platt recalibration, market-odds edge; ships the `calibrate` skill bundle | v0.1.0 — extracted, tests green, dogfooded, **scrub PENDING** |
| `data-contract` (planned) | schema + append-only/lookahead invariant + drift gate | Phase 2 (3.10 blocker cleared 2026-07-05; sequenced after calibrate) |
| overfitting harness / purged CV (planned) | deflated Sharpe, PBO (CSCV), White's Reality Check; CPCV/walk-forward engines | Phase 3 |

## Publishing status

**Not yet published.** Everything here is pre-release until the human
IP/strategy scrub signs off (hard gate: no strategy logic, alpha, thresholds,
data, or addresses leave the private repo). The library name `rigorkit` is a
working name — final naming happens at the same checkpoint. License: Apache-2.0
(patent grant + NOTICE attribution — the right default for a company-published
methodology library; recorded 2026-07-04).

## Conventions

- Each package: own `pyproject.toml`, tests, README, LICENSE, NOTICE.
- Skill bundles ship *inside* the pip package with a
  `python -m <pkg>.skills install` CLI (pattern credit: goldmansachs/gs-quant,
  Apache-2.0), so bundle + module are one artifact.
- License-honesty: MIT/Apache upstream patterns credited in NOTICE;
  AGPL is never vendored — reimplement from the math.
