# lemma — the Epsilon skills library

A public library of research-rigor tooling extracted from Epsilon's internal
quantitative research stack. Built here in a subfolder, designed to split into
its own repository (`git subtree split library/` → `Epsilon-Fund/lemma`) once
the human release step is taken.

## How distribution works (no package registry)

**The repo is the distribution.** Nothing here is published to PyPI or any
other registry — you consume lemma the way you consume BuilderIO/skills or
kepano/obsidian-skills:

1. **Copy a skill bundle** into your agent's skills directory:

   ```bash
   cp -r changepoint/src/lemma/changepoint/skills/changepoint-audit  .claude/skills/
   cp -r calibrate/src/lemma/calibrate/skills/calibrate              .claude/skills/
   cp -r skills/reflection-prompt                                    .claude/skills/
   cp -r skills/prd-scaffold                                         .claude/skills/
   ```

2. **Optionally install a Python package from git** when you want the engine
   importable (each package also bundles its skills plus a
   `python -m lemma.<pkg>.skills install` installer):

   ```bash
   pip install "lemma-changepoint @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=changepoint"
   pip install "lemma-calibrate   @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=calibrate"
   ```

   (Until the subtree split, the subdirectory is `library/<pkg>` in the
   private monorepo — collaborators use the editable installs in the repo TODO.)

## The one non-negotiable rule

**The library never imports Epsilon internals** — no `infrastructure.*`,
`polymarket.*`, `live_trading.*`, `topics.*`, or vault paths. Each package
imports only the stdlib and its own declared third-party deps. Epsilon consumes
the library (one-way), never the reverse. Enforced per package by a
`tests/test_decoupling.py`.

## What's here

Two kinds of entry, one catalog (`catalog.json`, machine-readable — the
Epsilon website's `/library` page reads a copy of it):

**Python packages** (engine + CLI + bundled agent skills + demo + tests):

| package | what it does | status |
|---|---|---|
| [`changepoint/`](changepoint/) — `lemma-changepoint` | causal, lookahead-free structural-break detection (CUSUM / Page-Hinkley / BOCPD) + purged-CV embargo, causal regime features, trend gate; ships the `changepoint-audit` skill bundle | v0.1.0 — extracted, tests green, dogfooded, scrub APPROVED |
| [`calibrate/`](calibrate/) — `lemma-calibrate` | Brier + Murphy decomposition, reliability diagrams with Wilson bands, ECE/MCE, Spiegelhalter's Z, isotonic/Platt recalibration, market-odds edge; ships the `calibrate` skill bundle | v0.1.0 — extracted, tests green, dogfooded, scrub APPROVED |
| `data-contract` (planned) | schema + append-only/lookahead invariant + drift gate | next in line |
| overfitting harness / purged CV (planned) | deflated Sharpe, PBO (CSCV), White's Reality Check; CPCV/walk-forward engines | later |

**Standalone skill bundles** (`skills/` — prompt-ware, no engine; copy the
folder and go):

| bundle | what it does | status |
|---|---|---|
| [`skills/reflection-prompt/`](skills/reflection-prompt/) | mine your own recent sessions for recurring pain → cluster → score recurrence × build-cost → decide build/automate/fix/nothing, logging every decision (including "nothing", with a reason) | v0.1.0 — scrub PENDING |
| [`skills/prd-scaffold/`](skills/prd-scaffold/) | co-author a PRD through structured Q&A, then emit a single self-contained goal prompt an implementation agent can execute | v0.1.0 — scrub PENDING |

## Publishing status

**Not yet public.** The remaining release step is deliberate and human-only:
`git subtree split` into `Epsilon-Fund/lemma` + the catalog copy onto the
Epsilon website. Per-package IP/strategy scrubs gate every entry (one SCRUB.md
each; nothing with a pending verdict leaves the private repo). License:
Apache-2.0 (patent grant + NOTICE attribution; recorded 2026-07-04). The
library brand is **lemma** (decided 2026-07-05; no registry name needed since
nothing is published to one).

## Conventions

- Each Python package: own `pyproject.toml`, tests (incl. the decoupling
  test), README, LICENSE, NOTICE, SCRUB.md, a seeded runnable
  `examples/demo.py` with a smoke test, and its skill bundles shipped
  *inside* the package with a `python -m lemma.<pkg>.skills install` CLI
  (pattern credit: goldmansachs/gs-quant, Apache-2.0).
- Each standalone bundle: `SKILL.md` (agentskills.io spec) + README + worked
  example + LICENSE + NOTICE + SCRUB.md.
- License-honesty: MIT/Apache upstream patterns credited in NOTICE;
  AGPL is never vendored — reimplement from the math.
