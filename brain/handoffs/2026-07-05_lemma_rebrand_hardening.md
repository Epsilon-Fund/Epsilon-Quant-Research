---
title: "Handoff — lemma rebrand + no-PyPI distribution model + RC-027 bundles built; two bundle scrubs pending"
created: 2026-07-05
status: complete — human gates: 2 bundle scrub sign-offs, repo split, website copy/deploy
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
  - lemma
  - infra
---

# Handoff — lemma rebrand + distribution hardening (2026-07-05, third pass)

> Hub: [[SKILL_MAP]] · law: [[CODEX]] · backlog: [[candidates]] · prior: [[2026-07-05_calibrate_library_extraction]] · [[2026-07-04_skills_lifecycle_phase1]]

## Plain-English Summary

- **What this is:** the operator's distribution + rebrand decisions applied to the skills library. The library brand is now **lemma** (was provisional `rigorkit`); there will be **no PyPI publish** — the public repo itself is the distribution (copy a skill bundle into `.claude/skills/`, the BuilderIO / kepano / gs-quant model), with an optional `pip install` straight from git for the Python engines.
- **Also in this pass:** RC-027's two prompt-ware skills were built as standalone copyable bundles (`library/skills/reflection-prompt/`, `library/skills/prd-scaffold/` — clean-authored, scrubs PENDING); RC-026 was reframed from a code extraction to a written setup-guide article and shelved on the docket; the catalog tool now indexes standalone bundles.
- **Everything re-verified after the rename:** changepoint 23 tests, calibrate 12, crypto regressions 7/7 + 20 through the shims, PM 6 + expected skip; both venvs run the renamed editable installs.
- **Human gates open:** the two bundle scrub sign-offs; then the standing release mechanics (`git subtree split library/` → `Epsilon-Fund/lemma`, the deliberate catalog copy into epsilon-webs1te, manual Vercel deploy). Per instruction, the website repo was NOT touched this pass — its working tree still holds the pre-rename catalog copy + pending-chip page edit from the previous session, uncommitted.

## What changed

1. **Rename (namespace + names, zero code-body changes):** `src/rigorkit/` → `src/lemma/` in both packages (git mv), package names `lemma-changepoint` / `lemma-calibrate`, console scripts, installer module paths, the epsilon shims' imports, NOTICEs, READMEs, SCRUB titles. Both venvs: rigorkit-* uninstalled, lemma-* editable-installed. The approved scrubs carry explicit rename notes (names only — content verdicts unaffected).
2. **Distribution text:** every `pip install <name>` line replaced by the two real paths — copy the bundle (primary) or `pip install "<pkg> @ git+https://github.com/Epsilon-Fund/lemma.git#subdirectory=<pkg>"` (optional). SCRUB "open at publish time" sections now record the resolved decisions (brand lemma, no PyPI) and the one remaining step (repo split; the git URL does not resolve until then).
3. **`library/README.md` rewritten as the lemma index:** distribution model up top, packages table + standalone-bundles table, conventions per entry kind.
4. **RC-027 built:** two standalone bundles, each SKILL.md (agentskills-spec) + README + fictional worked EXAMPLE.md + LICENSE + SCRUB.md. No tests by design — prompt-ware has nothing to run; the worked example is the demo. `reflection-prompt` = mine-your-own-sessions → recurrence × build-cost rubric → decide/log with "nothing-with-a-reason"; `prd-scaffold` = PRD-via-Q&A → one self-contained goal prompt (with the validate-before-build gate and the read-it-back-cold step).
5. **Catalog hardened:** `tools/skills_catalog.py` gains `library-bundle` entries (from `library/skills/*/SKILL.md`, scrub-aware, copy-line invocation) and emits git-install invocations for packages; public-feed note no longer references PyPI. Current feed: 47 entries — `lemma-changepoint` (approved), `lemma-calibrate` (approved), `prd-scaffold` (pending), `reflection-prompt` (pending).

## Deliberately NOT done

- **Website repo untouched** (catalog copy + commit + deploy are the human publish gate; explicitly excluded by the operator).
- **RC-026 not built** — reframed to an article and shelved on the docket per instruction.
- **RC-025 data-contract** — unchanged (next in line; no new go).
- **No NOTICE files for the prompt-ware bundles** — nothing borrowed to attribute; LICENSE covers copyright. Packages keep NOTICEs (they carry real attribution).

## Next gates

- **Justin:** flip the two bundle SCRUB verdicts (or flag), rerun `python3 tools/skills_catalog.py`; when ready to publish: repo split → `Epsilon-Fund/lemma`, re-copy `library/catalog.json` → `epsilon-webs1te/data/skills-catalog.json` (replacing the stale pre-rename copy sitting uncommitted there), commit + `npx vercel --prod`.
- **Collaborators:** after pulling, reinstall — `uv pip install -e "library/changepoint[dev]" -e "library/calibrate[dev]" --python .venv/bin/python` and `uv pip install -e "library/calibrate" --python polymarket/research/.venv/bin/python` (the old rigorkit editable installs must be removed: `uv pip uninstall rigorkit-changepoint rigorkit-calibrate --python <venv>`).
- **Engine:** Monday's reflection pass — RC-026 is shelved-docket and RC-027 is at scrub stage; neither is re-proposable.
