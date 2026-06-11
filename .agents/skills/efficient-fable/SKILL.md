---
name: efficient-fable
description: >
  Use when implementation work is token-heavy for THIS repo: CPCV/walk-forward
  sweeps across assets, DuckDB scans over polymarket fills/positions,
  multi-notebook runs, log/capture-output reduction, broad repo or vault scans,
  or repetitive bounded edits. Orchestrate with cheap subagents; keep judgment,
  integration, and final review in the main agent.
---

<!--
Vendored from https://github.com/BuilderIO/skills (skills/efficient-fable)
at commit 72e70ce35653d98f56e8a779162b1a79b31b9187, adapted for
epsilon-quant-research: repo-conventions section added, gbrain retrieval rule
added, upstream Claims/launch-copy and Diagram sections dropped (assets/ not
vendored). Upstream README.md kept alongside for attribution.
-->

# Efficient Fable

Use Claude Fable as the orchestrator, architect, synthesizer, and final judge.
Use cheaper subagents for token-heavy research, coding, testing, and
summarization that do not require Fable's full judgment.

## Where Fable Shines

Reserve Fable for:

- Decomposing ambiguous work into clean parallel slices.
- Architecture, product, and safety tradeoffs.
- Reading conflicting subagent reports and deciding what matters.
- Integrating partial implementations into one coherent plan.
- Final review, risk assessment, and user-facing synthesis.

## Delegation Pattern

1. Name the expensive-token risk: large repo search, long logs, broad docs, or
   repetitive edits.
2. Split independent work into subagents before reading everything yourself.
3. Use cheaper models for research scans, inventory, search summaries, narrow
   bug hunts, browser/testing passes, test output reduction, and bounded code
   edits.
4. Ask subagents for concise evidence: files, line references, commands run,
   diffs, uncertainties, and stop conditions they hit.
5. Spend Fable tokens on the decision layer: compare results, resolve conflicts,
   choose the implementation path, and review the final patch.

Prefer parallel subagents when the slices do not depend on each other. Keep
blocking or highly coupled work local.

## Repo Conventions For Subagents

Every handoff packet must state that the subagent obeys the `brain/CODEX.md`
invariants — subagents have no chat context, so the law travels in the packet:

- uv only, never bare pip; the two projects have separate venvs and never
  cross-import.
- Polymarket scripts run as `PYTHONPATH=. uv run python …` from inside
  `polymarket/research/`.
- All metrics lookahead-free: filter by timestamp before aggregating.
- Parquet shards are append-only — never edit in place.
- Writes land only where `brain/VAULT_MAP.md` § Where to write things allows;
  code/scripts never go in `brain/`.
- Commits (if any) go only to the operator's personal branch, never main.

Delegated "find prior work on X" subtasks use the gbrain MCP tools (semantic
`search` + `traverse_graph`/`get_backlinks`), not end-to-end hub reads — the
vault is indexed; do not pay for rediscovery.

## Handoff Packets

Write delegated prompts as if the subagent has no useful chat context. Include
only the context it needs:

- The repo path and exact objective.
- The files, packages, or surfaces in scope and anything explicitly out of
  scope.
- The repo-convention block above (the `brain/CODEX.md` invariants that apply).
- The evidence format to return: files, line refs, commands, diffs, failures,
  screenshots, and uncertainty.
- The verification commands or browser flows to run, plus what success should
  look like when that is knowable.
- Stop conditions: if the code does not match the prompt, a command fails after
  a reasonable retry, or the task needs out-of-scope files, stop and report
  instead of improvising.

## Vetting Delegated Work

Treat subagent reports as leads, not facts. Before using a high-impact finding,
opening a PR, or telling the user the work is done, Fable should reopen the
important cited files, confirm the relevant line refs or failures, and review
the final diff against the task. Let lighter agents gather signal; keep
truth-judgment with Fable.

## Common Scenarios

Treat these as soft defaults, not rigid rules:

- Research: ask lighter agents to scan docs, prior art, APIs, and repo surfaces;
  Fable decides what evidence changes the plan.
- Coding: give cheaper agents bounded edits or candidate patches; Fable owns
  shared-file coordination, integration, and final review.
- Testing: have Fable suggest the validation direction and the scripts or
  browser checks that matter. Let lighter agents run targeted tests, browser
  flows, screenshots, and log reduction, then report exact commands, failures,
  likely causes, and whether failures look flaky, environmental, or real.
- Debugging: use cheaper agents to cluster logs, reproduce issues, and try
  small fixes; Fable decides which diagnosis is most trustworthy.

If a task is tiny or the validation itself needs delicate judgment, keep it
with Fable.
