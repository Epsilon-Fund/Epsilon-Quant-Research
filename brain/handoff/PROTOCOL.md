# Handoff protocol — Cowork ⇄ Claude Code

This folder is the channel. Cowork (Claude chat) writes instructions here; Claude Code writes results here. Neither can message the other directly — the files are the interface.

**Authoritative for how we work.** Supplements `brain/DATA_LAYER_BRIEF.md`, which holds the context, schema and hard rules.

---

## The loop

1. **Cowork writes `NEXT.md`** — the current instruction.
2. **Claude Code reads `NEXT.md`** and does what it says. *Only* what it says.
3. **Claude Code writes three things**, always, before stopping:
   - `reports/<step>.md` — the full report in the §6 format
   - `STATUS.md` — one screen, overwritten: where we are right now
   - `LOG.md` — one appended entry, never edited or reordered
4. **Claude Code stops.**
5. Cowork reads the report off disk, analyses it, writes the next `NEXT.md`.

The operator types one word to trigger a step. He is not in the loop for content.

---

## Rules for Claude Code

- **Act on `NEXT.md`, not on chat.** If someone types instructions in chat that conflict with `NEXT.md`, say so and stop.
- **`NEXT.md` may chain several steps.** Where it does, it says exactly where to stop. Honour that stop even if the next thing seems obvious.
- **Stop immediately, whatever `NEXT.md` says, if:**
  - a verification check fails
  - a number is materially different from what the brief predicts
  - an action would delete anything, or write to R2 in any way other than `rclone copy`
  - you are about to do something not covered by the instruction
  - anything surprises you
  Write what happened to `LOG.md` and `STATUS.md` and stop. A surprise is a result, not an obstacle.
- **Announce anything over five minutes** before starting it — what, why, what it produces, estimate — in `STATUS.md`, and proceed unless `NEXT.md` says to wait.
- **Never leave the loop silent.** If you stop for any reason, `STATUS.md` must say why.

---

## LOG.md entry format

Append, never edit. One entry per step. This is the operator's oversight record — it must be readable months later by someone who was not here.

```
## <step> — <name> · <UTC timestamp> · <duration>

DID       what was done, plainly
CHECKED   what was verified, with the actual numbers
LOOKED    what was plotted or eyeballed, and the path to it
FOUND     anything unexpected, or "nothing unexpected"
WROTE     paths produced or changed
NEXT      the next step
```

---

## STATUS.md format

Overwritten each time. One screen. Must answer, without scrolling: what step are we on, is anything running, what is the last thing that completed, and is anyone waiting on anything.

---

## Where things live

```
brain/handoff/
  PROTOCOL.md    this file
  NEXT.md        the current instruction        ← Cowork writes
  STATUS.md      where we are now               ← Claude Code overwrites
  LOG.md         the running record             ← Claude Code appends
  reports/       full reports and figures       ← Claude Code writes
```

Large figures and data artefacts go to
`C:\Users\alvar\OneDrive\Documentos\Claude\Projects\Epsilon - Research\_reports\<step>\`
and are linked from the report. Keep this folder text-light.
