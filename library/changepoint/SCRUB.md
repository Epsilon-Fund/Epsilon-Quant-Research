# IP / Strategy Scrub — lemma-changepoint v0.1.0

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

- **Resolved 2026-07-05:** library brand = `lemma` (renamed from the
  provisional `rigorkit` while nothing was public — this scrub's content
  verdict is unaffected: same code bodies, names only). **No PyPI** — the
  distribution is the public repo itself (copy the skill bundle, or
  `pip install` from git); no registry name is needed.
- Remaining release step (human-only): `git subtree split library/` →
  `Epsilon-Fund/lemma` public repo + the deliberate catalog copy onto the
  website. Until then the documented git-install URL does not resolve.
