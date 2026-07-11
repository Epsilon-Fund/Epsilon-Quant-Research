---
name: find-skills
description: >
  Surface the most relevant local skills for the current task (the Sherpa
  router). Use at session start, whenever the task shifts to a new kind of
  work, or when the user asks "is there a skill for this / what skill should I
  use / how do I do X here". Runs tools/sherpa.py over this repo's skill catalog
  and returns the top-N skills with a one-line "use when" so the right skill
  loads without anyone naming it. Local + deterministic (keyword scoring, plus
  local-embedding semantics when Ollama is up). Complements Claude Code's native
  description-triggering — it catches skills the agent would not have triggered
  on and ranks when several match.
---

<!--
Source: epsilon-quant-research (branch justin). First-party. This is the skill
wrapper around tools/sherpa.py — the Sherpa skill router. Sherpa itself is
scope: internal this pass (a public-library entry is a later decision). The
engine is tools/sherpa.py; this bundle just tells the agent how and when to run
it. Cross-repo: a copy of tools/sherpa.py + this bundle is vendored into the
"is this the bottom" repo so that repo routes over its OWN local skills.
-->

# find-skills — Sherpa skill router

Given the current task, score every skill installed in this repo and surface the
top matches, each with a one-line "use when". This is the proactive layer on top
of Claude Code's native auto-triggering: it fires the skills the agent might not
have triggered on, and ranks when several are plausible.

## When to run it

- **At session start** and **whenever the task shifts** to a new kind of work
  (this is also wired into the Agent Bootstrap in `brain/VAULT_MAP.md`,
  `brain/CODEX.md`, and `brain/COWORK.md`).
- When the user asks "is there a skill for X", "what should I use here",
  "how do I do X in this repo".

## How to run it

From the repo root:

```bash
python3 tools/sherpa.py "<the task or prompt, in a sentence or two>"
```

Useful flags:

- `--top N` — how many skills to surface (default 5).
- `--json` — machine-readable output (name, scope, score, use_when, source).
- `--scope shareable|internal|unknown` — restrict to one scope.
- `--no-semantic` — keyword-only (skip the local embedder).
- `--list` — dump the whole indexed catalog with scope tags.

Example:

```
$ python3 tools/sherpa.py --top 3 "how well calibrated is my forecast model?"
1. calibrate  (internal, score 0.28)
   → Use when asked how well-calibrated a model or forecaster is …
2. superforecast  (internal, score 0.24)
   → Use when the user asks "will X happen", "should I do Y" …
```

## How to act on the output

1. Read the surfaced skills' one-line "use when". For any that clearly fit the
   task, **load that skill** (invoke it) — do not wait to be told.
2. Sherpa **surfaces and ranks**; it does not force-invoke. A low top score
   (everything under ~0.1) means no skill is a strong fit — proceed without one.
3. Sherpa never blocks. It is a router, not a gate. The one hard gate in this
   repo is `data-contract` (fail-closed before a real run); Sherpa may surface
   it, but running it is still your call per its own rules.

## How the matcher works (local + deterministic)

- **Keyword/description scoring** — always on, offline, deterministic. Weighted
  overlap of the task's terms with each skill's trigger keywords, name, and
  description.
- **Semantic** — an optional booster. If a local embedder is running (Ollama
  `nomic-embed-text` on `localhost:11434`, the same model gbrain uses), Sherpa
  blends cosine similarity of the task against each skill description. Nothing
  leaves the machine. If Ollama is down, Sherpa is keyword-only and says so.
- The index is rebuilt from SKILL.md frontmatter on every run, so it is always
  current when skills change. Embeddings are cached by description hash and
  re-embedded only when a description changes.

## Scope tags

Each skill carries a `scope`: `internal` (epsilon-wired, never leaves the repo),
`shareable` (scrubbed / generic, safe to vendor into another repo), or `unknown`
(needs a human tag). See `tools/sherpa_scope.json` for repo-local overrides.
