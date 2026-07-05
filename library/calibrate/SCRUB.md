# IP / Strategy Scrub — rigorkit-calibrate v0.1.0

**VERDICT: PENDING HUMAN SIGN-OFF** — checklist executed 2026-07-05 by the
implementation agent with recorded evidence; **no flags found**. Unlike the
changepoint scrub (operator-delegated in chat on 2026-07-04), no delegation was
given for this package, so the verdict line stays PENDING until Justin flips it
to APPROVED. The catalog generator keys on the literal approved verdict, so
this package reports `scrub_status: pending-human-review` until then.

## Checklist (per skills_library_build_plan § Cross-cutting gates)

| check | result | evidence |
|---|---|---|
| Strategy logic / alpha | NONE — textbook scoring methodology (Murphy 1973 decomposition, Wilson 1927 intervals, Spiegelhalter 1986 Z, PAV isotonic regression, Platt scaling, proportional devig) | code review; no trading rules, signals, or selection logic anywhere |
| Tuned thresholds | NONE — the only constants are literature-standard (z=1.96 Wilson default, 10 bins default, numeric epsilons) | core.py defaults |
| Proprietary data / data paths | NONE — the internal book→repo-path ledger mapping was **removed** in extraction (`resolve_ledger_dir` now takes an explicit path / env var / caller-supplied mapping); examples and tests use synthetic seeded data only | grep for live_trading/, polymarket, R2/Hetzner/Binance: 0 hits |
| Wallet addresses / keys / secrets | NONE | grep `0x[a-f0-9]{16,}`, api-key/secret/token patterns: 0 hits |
| Internal naming | Only (a) NOTICE attribution "extracted from the Epsilon internal quantitative research codebase" (intended provenance, mirrors changepoint) and (b) the decoupling test's forbidden-module list (the enforcement mechanism) | NOTICE; tests/test_decoupling.py |
| Upstream licences | Apache-2.0 package; gs-quant installer pattern (Apache-2.0) credited in NOTICE; the superforecasting ledger *format* is consumed read-only with **no code included** from that MIT project (credited in NOTICE anyway); no AGPL anywhere | NOTICE |

## Scope

Covers: `library/calibrate/**` and its catalog entry in `library/catalog.json`.
The website copy step (`epsilon-webs1te` `data/skills-catalog.json`) is the
publish gate and stays blocked until the verdict above is APPROVED **and** the
site deploy is a deliberate human action.

## Open at publish time

- Same as changepoint: library name `rigorkit` provisional; PyPI/repo-split not
  done; the `pip install rigorkit-calibrate` line is aspirational until upload.
