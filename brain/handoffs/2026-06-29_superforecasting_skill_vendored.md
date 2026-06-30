---
title: "Vendored the superforecasting skill (resolvable, Brier-scored forecasts) + forked its ledger to a repo-controlled, one-per-book path"
tags: [handoff, skills, tooling, provenance, superforecasting, forecasting, calibration]
created: 2026-06-29
status: complete — vendored, forked, wired, all DoD gates passed (provenance/tooling step only; no strategy or signal code)
owner: justin
hubs:
  - SKILL_MAP
  - VAULT_MAP
  - CODEX
relationship: >
  Provenance + tooling step. Vendors deusyu/superforecasting-skill (MIT) into the repo skills tree
  and forks its global ledger into two repo-controlled, per-book ledgers (polymarket, crypto) so the
  two projects share no state ([[CODEX]] § no cross-import). Registered in [[SKILL_MAP]] § Agent
  runtime skills alongside the other vendored skills (`cost-mode`, `efficient-fable`,
  `stay-within-limits`). No strategy code, no signal work.
---

# Vendored the superforecasting skill + forked its ledger path

## Plain-English Summary

- **What:** Vendored the open-source **superforecasting skill** (MIT, `deusyu/superforecasting-skill`) into `.agents/skills/superforecasting/`, symlinked it into both runtimes (`.claude/skills/`, `~/.codex/skills/`), and recorded it in `skills-lock.json` + [[SKILL_MAP]].
- **Why:** Gives both agents a prompt-invoked tool to turn a vague claim/decision into a **resolvable, dated, Brier-scored probability**, then settle and review calibration over time — with a built-in **anti-post-hoc guard** (you cannot rewrite a probability after the outcome is known).
- **The one required local patch:** upstream logs everything to a single global `~/.superforecast/` directory. We forked the ledger path to be **repo-controlled, one ledger per book** so the polymarket and crypto projects never pool state.
- **Status / takeaway:** Done. Security re-vet passed on the pinned commit (stdlib-only, no network/shell/secrets, writes confined to the ledger dir); the smoke forecast logs to the forked per-book path (not `~/.superforecast`); the anti-post-hoc guard still rejects edits to a settled forecast after the fork. This is a tooling/provenance step only — it does not touch any strategy.

## Source + pin

- **Upstream:** [`deusyu/superforecasting-skill`](https://github.com/deusyu/superforecasting-skill) — License **MIT**.
- **Pinned commit:** `8913b0847ab63b7aefa0599bdffe00362ee360f2` (2026-06-24). Recorded at the top of the vendored `SKILL.md` and in `skills-lock.json` (`sourceCommit`).

## STEP 1 — Security re-vet (on the pinned commit)

Re-confirmed line-by-line that `scripts/sf.py` still holds the previously-vetted properties:

- **Imports are stdlib-only:** `argparse, json, re, sys, datetime, pathlib` (upstream). `grep` for `os / subprocess / socket / requests / urllib / os.system / os.environ / getenv / eval / exec` returned **none** in the upstream file.
  - No network (no `requests`/`urllib`/`socket`), no shell-out (no `subprocess`/`os.system`), no credential reads (no `os.environ`/`getenv`; `os` was not even imported upstream).
- **Writes are confined to the ledger dir:** `events.jsonl` is **append-only** (`EVENTS_FILE.open("a")`); `active.json`, rendered cards, and calibration reports are derived projections rewritten in place — all strictly under `LEDGER_DIR`. No writes anywhere else.

Verdict: safe to vendor. (The fork in STEP 3 deliberately adds one narrow `os.environ.get` read — for the ledger path only, never credentials — documented below.)

## STEP 2 — Vendored files (exactly the green-lit set)

Into `.agents/skills/superforecasting/` (12 files):

```
SKILL.md
scripts/sf.py
references/{workflow,scoring,superforecasting_concepts,examples}.md
schemas/{forecast_event,forecast_card}.schema.json
docs/{skill_design,concept_understanding}.md
AGENTS.md
LICENSE
```

Left upstream (not vendored): `README.md`, `README.zh-CN.md`, `.github/FUNDING.yml`, `.gitignore`.

> **Dir name vs skill name:** the directory is `superforecasting` (matches the task + the source repo name); the `SKILL.md` frontmatter `name:` is `superforecast` (kept as upstream). Both runtimes load it — Claude Code listed it as `superforecasting` (by directory), Codex/OpenClaw key off the frontmatter `name`. Benign; noted so it isn't a surprise.

## STEP 3 — The ledger fork (the required local patch)

**Problem:** upstream hard-codes `LEDGER_DIR = Path.home() / ".superforecast"` — one global log pooled across every project. That violates [[CODEX]] § no-cross-import between the polymarket and crypto books.

**Fix:** `LEDGER_DIR` is now resolved at import time, **one ledger per book**:

1. `$SF_LEDGER_DIR` — explicit path override (wins if set; used for isolated tests).
2. `$SF_BOOK` ∈ {`polymarket`, `crypto`} → repo-relative per-book ledger.
3. Neither set → **hard error** ("sf refuses to guess"). Prevents silently pooling the two books.

The repo root is found via `Path(__file__).resolve().parents[4]`, which follows symlinks, so resolution is correct whether `sf.py` is invoked through `.agents/`, the `.claude/skills/` symlink, or the absolute `~/.codex/skills/` symlink (all verified). The fork adds `import os`, used **only** for the two `os.environ.get` calls above — no credentials, no network, no shell. The docstring and the ledger-path mentions in `SKILL.md`, `AGENTS.md`, and `schemas/forecast_event.schema.json` were updated to match.

**Untouched (the point of the engine):** the `TRANSITIONS` state machine, Brier-at-settle (`(final_p - outcome)²`), and the append-only event write are byte-for-byte upstream.

### The two ledger paths (per [[VAULT_MAP]] § Where to write things / the data manifests)

| Book | `SF_BOOK` | Ledger path | Already git-ignored? |
|---|---|---|---|
| Polymarket | `polymarket` | `polymarket/research/data/superforecast/` | Yes — covered by `polymarket/research/data/` |
| Crypto | `crypto` | `live_trading/data/superforecast/` | Yes — new `.gitignore` rule added |

Layout inside each: `forecasts/events.jsonl` (append-only source of truth), `forecasts/active.json` (derived state), `forecasts/rendered/<id>.md` (cards), `reports/calibration_*.md`.

**Gitignore policy — treat like `trades.json`.** The ledger is append-only **runtime data**, not source-of-truth code, so it is git-ignored and never committed (matching the repo's "never commit live state" rule for `trades.json`/`positions.json`). A repo-relative path still gives the intended sharing: both agents (Codex + Claude Code) operating in the **same checkout and same book** see the same ledger, while the two books stay separate. It is not git-synced across machines (correct for append-only logs — avoids merge churn).

## STEP 4 — Wired in

- **Symlinks** (same style as `cost-mode` / `efficient-fable`): `.claude/skills/superforecasting → ../../.agents/skills/superforecasting` (relative, in-repo) and `~/.codex/skills/superforecasting → <repo>/.agents/skills/superforecasting` (absolute, since `~/.codex` is outside the repo).
- **`skills-lock.json`** — added the `superforecasting` entry mirroring the `cost-mode` shape plus the required `localPatch`, the pinned `sourceCommit`, and a `hashMethod` note (the hash semantics differ from `cost-mode`, see below):

```json
"superforecasting": {
  "source": "deusyu/superforecasting-skill",
  "sourceType": "github",
  "sourceCommit": "8913b0847ab63b7aefa0599bdffe00362ee360f2",
  "skillPath": "SKILL.md",
  "computedHash": "e047852d2aeb6e0fda9cb4ecfefed1c1f57f30ef6e75f97643b5985091786c00",
  "localPatch": "LEDGER_DIR -> repo-controlled path, one per book ..."
}
```

  - **`computedHash` method (reproducible):** `cost-mode`'s hash is a single-file sha256; this entry's hash is a **tree hash of the vendored, patched tree** (the task asked for the patched-tree hash, since the patch lives in `sf.py`, not `SKILL.md`). Recipe:
    ```bash
    cd .agents/skills/superforecasting
    find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | while read -r f; do
      printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "${f#./}"
    done | shasum -a 256 | awk '{print $1}'
    # => e047852d2aeb6e0fda9cb4ecfefed1c1f57f30ef6e75f97643b5985091786c00
    ```
- **`brain/SKILL_MAP.md`** — registered a row in § Agent runtime skills (prompt-invoked: log a forecast, settle/score via Brier, review calibration; `SF_BOOK` first) plus a provenance/guardrail note linking back here.

## DoD gates — all passed

Run with `uv run --no-project python` (stdlib only; no `pip`).

| Gate | Result |
|---|---|
| `sf.py` runs stdlib-only; full lifecycle `new → scope → set-prob → update → settle → render → list → review` | ✓ |
| Brier correct: `(0.65 − 1)² = 0.1225` | ✓ `0.1225` |
| Smoke logs to the forked per-book path, **not** `~/.superforecast` | ✓ events under `…/superforecast/forecasts/events.jsonl`; `~/.superforecast` never created |
| Resolution correct via all 3 entry paths (`.agents`, `.claude` symlink, `~/.codex` symlink) | ✓ all → repo per-book paths |
| No-book guard | ✓ errors instead of guessing |
| **Anti-post-hoc guard fires after the fork** — mutate a SETTLED/SCORED forecast's probability | ✓ both `sf update` and `sf set-prob` rejected with "illegal transition"; final_p/Brier unchanged |
| Skill loads/lists in **Claude Code** | ✓ appears in the available-skills list as `superforecasting` |
| Skill loads/lists in **Codex** | symlink + valid single-line frontmatter in place under `~/.codex/skills/` (Codex not run in this session; reachable + parseable verified) |

### Worked example of the guardrail (anti-post-hoc)

A forecast `sf-2026-001` set at `p=0.65`, settled `--outcome 1`, is auto-scored `Brier=0.1225` and moves to state `SCORED`. Trying to "improve" it after the fact — `sf update sf-2026-001 --p 0.99` — is refused: `SCORED` is not an allowed source state for `evidence_update` (allowed: `ACTIVE`, `UPDATED`). Same for `sf set-prob`. The recorded probability and Brier cannot be rewritten with hindsight. This is the discipline the skill exists to enforce.

## Decision / next step

Done — the skill is installed, forked, and verified. No follow-up code is required. When using it: **always `export SF_BOOK=polymarket` or `SF_BOOK=crypto` first** (or `SF_LEDGER_DIR` for a throwaway), or `sf` will refuse to run. Future maintenance: to re-verify the vendored tree against the lock, re-run the `computedHash` recipe above and compare to `skills-lock.json`. Upstream README attribution lives in the source repo (we did not vendor the READMEs).
