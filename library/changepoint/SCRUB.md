# IP / Strategy Scrub — rigorkit-changepoint v0.1.0

**VERDICT: APPROVED** — 2026-07-04, review delegated by Justin (chat, 2026-07-04:
"you have git access… you have access and permission"), executed by the
implementation agent with recorded evidence.

## Checklist (per skills_library_build_plan § Cross-cutting gates)

| check | result | evidence |
|---|---|---|
| Strategy logic / alpha | NONE — package is textbook methodology (CUSUM, Page-Hinkley, BOCPD per Adams & MacKay 2007) | code review; the only "alpha/beta" tokens are Normal-Gamma prior parameters |
| Tuned thresholds | NONE — all defaults are literature-standard (k=0.5, h=5.0, λ=8.0, hazard 250), not fitted trading parameters | detectors.py defaults |
| Proprietary data / data paths | NONE — examples use generic `prices.parquet`; benchmarks are synthetic series | grep for live_trading/, polymarket data paths, R2/Hetzner/Binance: 0 hits |
| Wallet addresses / keys / secrets | NONE | grep `0x[a-f0-9]{16,}`, api-key/secret/token patterns: 0 hits |
| Internal naming | Only (a) NOTICE attribution "extracted from infrastructure/changepoint" (intended provenance) and (b) the decoupling test's forbidden-module list (the enforcement mechanism — reveals only internal top-level package names) | test_decoupling.py:15 |
| Upstream licences | Apache-2.0 package; gs-quant installer pattern (Apache-2.0) credited in NOTICE; no AGPL anywhere | NOTICE, radar audit |

## Scope

Covers: `library/changepoint/**`, `library/README.md`, `library/catalog.json`,
and the website surface (`epsilon-webs1te` `/library` page + `data/skills-catalog.json`).

## Addendum — 2026-07-05

`examples/demo.py` + `tests/test_demo.py` + README demo section added after the
scrub. Content check against the same table: synthetic seeded data only, all
detector parameters are the package's literature-standard defaults, no internal
paths/names beyond what the scrub already covers. No new content class — the
APPROVED verdict stands.

## Open at publish time

- Library name `rigorkit` remains provisional — rename (if desired) before any
  PyPI publish; a rename after PyPI is painful.
- PyPI publish itself not yet done: the page's `pip install` line goes live in
  spirit only until the package is uploaded or the repo is split public.
