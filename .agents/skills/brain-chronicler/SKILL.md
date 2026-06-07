---
name: brain-chronicler
description: >
  Use for epsilon-quant-research end-of-session records, decisions, closures,
  handoffs, status changes, and collaborator updates. Trigger when the user
  says Chronicler, write a handoff, record what changed, capture decisions,
  end-of-day handoff, update TODO/hub after a decision, or before handing work
  to Codex/Cowork.
---

# Brain Chronicler

Use this skill to record why work changed, not to narrate every intermediate step.

## Report-First Workflow

1. Read `brain/SKILL_MAP.md`, `brain/VAULT_MAP.md`, `brain/TODO.md`, and the relevant hub.
2. Gather recent context: today's diffs, recent commits, changed notes, scratch lane, and user decisions from the session.
3. Decide the output surface:
   - `brain/handoffs/YYYY-MM-DD_<topic>.md` for a real handoff.
   - A short status line in the relevant hub when the decision belongs there.
   - `brain/TODO.md` when next gates or ownership changed.
4. Record decisions, closures, blockers, next gates, and where evidence lives.
5. Keep the note concise enough to be useful in the next session.

## Guardrails

- Do not turn a handoff into a transcript.
- Do not change strategy status unless the evidence or user decision supports it.
- Do not auto-edit many hubs. One or two precise updates beat broad churn.
- If two agents are active, one Chronicler pass should own the durable record.

## Good Output

- A future session can answer: what changed, why it matters, what is blocked, and what to do next.
- Links point to the evidence, findings note, or TODO entry.
