---
title: "Handoff — rigorkit-calibrate extracted to library/ (RC-024 built); scrub PENDING; localhost preview live"
created: 2026-07-05
status: complete — one human gate open (calibrate scrub sign-off)
owner: justin
project: infra
para: area
hubs:
  - SKILL_MAP
  - CODEX
  - TODO
tags:
  - handoff
  - skills
  - library
  - calibration
  - infra
---

# Handoff — calibrate library extraction (2026-07-05, operator-go build pass)

> Hub: [[SKILL_MAP]] · law: [[CODEX]] · backlog: [[candidates]] · prior passes: [[2026-07-05_skills_easywins_pass]] · [[2026-07-04_skills_lifecycle_phase1]]

## Plain-English Summary

- **What this is:** the RC-024 proposal executed on Justin's go ("build the rest, to a localhost-viewable point"). The `calibrate` engine is now the library's **second package** — `library/calibrate/` = `rigorkit-calibrate` v0.1.0 (Apache-2.0) — proving the changepoint packaging pattern repeats cleanly.
- **Where to look:** the `/library` page preview is live on **http://localhost:3000/library** (the already-running `pnpm dev` in `epsilon-webs1te` hot-reloaded the new catalog); both packages render, calibrate carries a "Review pending — not for deploy" chip.
- **The one open gate:** the calibrate **IP scrub is PENDING** — checklist executed with zero flags, but unlike changepoint no delegation was given, so the verdict line in `library/calibrate/SCRUB.md` awaits Justin. The website repo changes are deliberately **uncommitted** (the copy/commit/deploy step *is* the publish gate).
- **Deliberately not built** (operator asked for reasons): RC-025 data-contract package, RC-026 Obsidian starter-kit, RC-027 reflection/PRD-scaffold skills, and any website commit/deploy — reasons in § What was NOT built.

## What shipped (verified)

1. **Package** — `library/calibrate/`: engine (`core.py`) + CLI (`rigorkit-calibrate --ledger <dir> score|table|report`) + in-package `calibrate` skill bundle with the gs-quant installer (`python -m rigorkit.calibrate.skills install`) + seeded runnable demo (`examples/demo.py`: well-calibrated vs over-confident forecaster, Murphy decomposition separating the failure modes, out-of-sample isotonic fix, markets layer, optional reliability PNG) + README/LICENSE/NOTICE/SCRUB.
2. **Generalization with a smaller scrub surface:** the epsilon book→repo-path ledger mapping was **removed from the package** — `resolve_ledger_dir` now takes an explicit path / `$SF_LEDGER_DIR` / a caller-supplied `books` mapping. The mapping lives only in the epsilon shims. The superforecasting ledger **state machine stays out** (MIT upstream, re-pointed in NOTICE; the package is a read-only scorer of the `events.jsonl` format).
3. **Tests:** 12 passed in the repo venv AND 12 passed standalone in a **fresh venv without sklearn** — the pure-numpy PAV/IRLS recalibration fallbacks are proven, not assumed. Decoupling enforced by the same ast-based test as changepoint.
4. **Dogfood:** `infrastructure/calibration/` and `polymarket/research/lib/calibration/` are now same-API **shims** (still byte-identical to each other, `cmp`-verified). Crypto regression **7/7** — including the `ml_metrics.calibration_table` byte-for-byte reproduction gate — and PM **6/6 + expected skip**, both through the shims. Historical CLI invocation lines unchanged (`--book crypto|polymarket` still works; smoke-tested on both real ledgers — correct "no scored forecasts yet" state, the Observatory's forecasts being live but unresolved).
5. **Catalog + preview:** `tools/skills_catalog.py` → 2 library entries (changepoint `approved`, calibrate `pending-human-review`); `library/catalog.json` committed; copied to `epsilon-webs1te/data/skills-catalog.json` (uncommitted); `page.tsx` gained a minimal scrub-pending chip and dropped calibration from the "in preparation" footer (uncommitted).
6. **Registry:** [[SKILL_MAP]] calibrate row + `.agents/skills/calibrate/SKILL.md` provenance comment updated to the shim reality; `library/README.md` packages table updated.

## What was NOT built, and why (operator asked for reasons)

- **RC-025 `data-contract` package** — its 3.10 blocker is cleared, but it has **no public consumer** (calibrate has the Observatory), a heavier dependency surface (`pandera.polars`), and the real work is API design: the PM/crypto contracts are the epsilon-specific part and genericizing contract-authoring deserves its own deliberate pass, not a same-day second extraction. Sequenced next once calibrate's pattern settles through the scrub.
- **RC-026 Obsidian starter-kit** — L-cost rewrite (the law files embed strategy context throughout, so it is NOT an extraction), the largest scrub surface of any candidate, and its own naming/positioning decision. A scoping session with Justin first.
- **RC-027 reflection-prompt / PRD-scaffold skills** — prompt-ware whose publish value depends on a distribution channel that doesn't exist yet (PyPI/repo-split still human-gated). Building the bundles now would produce artifacts that sit; the generalization is cheap whenever the channel decision lands.
- **Website commit / Vercel deploy** — the copy step into `epsilon-webs1te` is the **IP-scrub publish gate** by design, and deploy is a deliberate human action (`npx vercel --prod`). The preview needs neither.

## Next gates

- **Justin:** review `library/calibrate/SCRUB.md` → flip to APPROVED (or flag) → rerun `python3 tools/skills_catalog.py`, re-copy `library/catalog.json` to the website repo, and commit/deploy there when ready. Standing release mechanics (final `rigorkit` naming, PyPI/split) unchanged.
- **Collaborators:** after pulling, install the package into both venvs (lines in [[TODO]] § Skills Lifecycle).
- **Engine:** Monday's reflection pass treats RC-024 as at-library-stage (never re-propose).
