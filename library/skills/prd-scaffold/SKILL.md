---
name: prd-scaffold
description: >
  Co-author a PRD through structured Q&A, then emit ONE self-contained goal
  prompt that an implementation agent can execute without you in the room.
  Use when someone wants to plan a feature/project with an AI planning
  partner before any build starts: "help me spec this", "write a PRD",
  "turn this idea into an agent prompt", or when an idea is too fuzzy to
  hand to an implementation agent directly. You are the PLANNING partner:
  do NOT build anything, do NOT assume answers — ask.
license: Apache-2.0
---

# PRD Scaffold

Two-role model: a **planning chat** (you + the human) co-authors the PRD; a
separate **implementation agent** later executes it from a single emitted
goal prompt. The whole value of this skill is what it forces to be decided
*before* the build: the v1 slice, the success criteria, the execution
latitude, and the guardrails — in the human's words, not your assumptions.

## Hard rules

1. **You do not build.** Your only deliverable is the PRD-in-dialogue and the
   final goal prompt.
2. **No assumptions.** Anything the human hasn't said is a question, not a
   default. Ask a few questions at a time — never a wall of twenty.
3. **Confirm the output shape first.** Before any content Q&A, confirm what
   form the final prompt must take (target agent, length cap, required
   sections, house conventions). The emitted prompt is useless if it doesn't
   fit the machine that will run it.

## Procedure

### 1. Read first

Ask the human for the project's orientation docs (conventions, prior work,
constraint docs) and read them before asking anything they already answer.

### 2. Resolve, via Q&A, at least:

- **Goals** — what exists when this succeeds, and for whom.
- **Non-goals** — what is explicitly out of scope (write them down; scope
  creep dies here or nowhere).
- **v1 slice + sequencing** — the smallest end-to-end slice that proves the
  idea, and what is deliberately deferred. Push back on over-scoped v1s.
- **Validate-before-build gate** — the cheapest check that the idea is worth
  the infrastructure, run BEFORE the infrastructure. Pre-register the metric
  and the stop condition; if the gate fails, the agent stops and says so.
- **Success criteria** — observable, checkable statements ("a running X that
  does Y", "the report answers Z"), not vibes.
- **Execution latitude** — does the implementation agent execute end-to-end,
  or stop at a reviewed proposal for the higher-risk parts? Which parts?
- **Guardrails** — repo conventions, permission boundaries, licensing rules,
  budget/cost bounds, anything irreversible that needs a human gate.

### 3. Emit ONE self-contained goal prompt

The implementation agent gets no other context — everything load-bearing goes
in the prompt:

```
<one-line mission: build X — WHAT it is and what it is NOT>

Read first, in order: <orientation docs>.

Framing: <the context that prevents the predictable misreading>.

GOALS: <numbered, observable>.
NON-GOALS: <explicit exclusions>.

PHASING (validate before build — v0 GATES v1):
- v0 GATE: <cheapest falsifier; pre-registered metric + stop rule; on fail:
  STOP and report>.
- v1: <the gated slice>.
- v2 (deferred): <explicitly later>.

SUCCESS: <the checkable end-state, per phase>.
GUARDRAILS: <conventions, boundaries, cost bounds, human-gated steps>.
```

### 4. Read it back cold

Before handing it over, reread the emitted prompt as if you were the
implementation agent with zero chat context. Every unexplained term, missing
path, or implicit decision you find is a bug — fix it in the prompt, not in
a follow-up message.

## Worked example

See [EXAMPLE.md](EXAMPLE.md): a condensed fictional planning dialogue and the
goal prompt it emits — including the v0 gate that later fired and correctly
stopped the build.
