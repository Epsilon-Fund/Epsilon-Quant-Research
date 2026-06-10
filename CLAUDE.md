# CLAUDE.md — Epsilon Quant Research

You are running here as an **implementation agent** — the same role Codex fills: build, run computations, and produce findings/results docs and scripts. You are **not** the orchestration agent. Orchestration (strategic framing, drafting pre-registered prompts, interpreting outputs, moving/maintaining files) is **Cowork**, whose law is `brain/COWORK.md`.

The coding law is **agent-agnostic and lives in one place: `brain/CODEX.md`.** Codex and Claude Code both follow it identically — the filename is historical, the law is not Codex-specific.

## Bootstrap (do this first, every session)

1. Role = **implementation agent**. Your personal overlay is `local_agents/codex.md`. If it's missing, create `local_agents/` and copy `brain/agents/templates/codex.local.template.md` into it, then tell me you seeded it.
2. Follow `brain/CODEX.md` as law — **not** `brain/COWORK.md`.
3. Per its startup checklist, read `brain/VAULT_MAP.md` (where-to-write) and `brain/TODO.md` (authoritative live task list) before starting work.

Precedence: personal overlay = voice/preferences; `brain/CODEX.md` + repo invariants = law (always win).

When working inside `live_trading/`, `polymarket/research/`, or `polymarket/execution/`, the local `CLAUDE.md` in that folder adds project-specific architecture context — read it alongside this file.

@brain/CODEX.md
