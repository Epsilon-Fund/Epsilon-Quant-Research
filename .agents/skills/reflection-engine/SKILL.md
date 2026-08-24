---
name: reflection-engine
description: >
  Discovery front of the skills lifecycle: mine recent work for recurring pain
  and turn it into build decisions. Use on the weekly scheduled pass, or
  on-demand when asked to "run a reflection pass", "what should we build/automate
  next", "review recent sessions for friction", or after a burst of repetitive
  manual work. Gathers signals via read-only subagents (git history, scratch
  lanes, session transcripts, hygiene/graph reports), clusters pain, scores
  recurrence × build-cost, logs every decision to brain/reflection/candidates.md
  (including "nothing" with a reason), executes accepted candidates, and runs
  the external-repo radar.
---

<!--
First-party skill (NOT vendored). Built 2026-07-04 as part of the skills
lifecycle (brain/reflection/candidates.md RC-002). Law: brain/CODEX.md.
The backlog it maintains is brain/reflection/candidates.md (tracked,
authoritative). SKILL_MAP + library/ stay canonical for built/published skills.
-->

# Reflection Engine

One pipeline: **discover → decide (recurrence × build-cost) → build → validate →
package → publish**. This skill is the *discover→decide→build* front. It exists
so recurring pain gets found systematically instead of ad hoc, and so rejected
ideas stay rejected (logged with reasons).

## Cadence

- **Weekly pass** — scheduled (`reflection-weekly`, Mondays after the hygiene
  scan so `brain/generated/` reports are fresh).
- **On-demand** — any time the operator asks, or after a session with obvious
  repeated toil.
- **Radar** — the external-repo sweep runs INSIDE this pass, monthly (first
  pass of the month), or when new seed repos are named.

## Procedure

### 1. Gather signals (read-only subagents — efficient-fable pattern)

Spawn parallel read-only subagents; keep synthesis central. Never let a
subagent edit files. Standard four:

1. **git history** — commits since the last pass (see the Pass log in
   candidates.md): churn hotspots, fix/again/rework messages, repeated chores
   (normalization, hygiene, vendoring), merge pain, test churn. Evidence =
   hashes + dates + counts.
2. **scratch lanes** — `scratch/codex/` + `scratch/cowork/`: explicit
   complaints, planned-but-unbuilt items, deferred chores.
3. **session transcripts (best-effort)** — `~/.claude/projects/<this repo>/`:
   grep-sample friction markers (errors, retries, repeated commands,
   corrections). Sampling only; never full reads; state confidence honestly.
   The gotchas below are cumulative and **all mandatory** for an accurate
   scan — each was a real accuracy bug this skill shipped with, found by the
   pass it bit (RC-029 2026-07-06 · RC-030 2026-07-17 · RC-033 2026-07-21 ·
   RC-037 2026-07-27 · RC-042 2026-08-03). Report your compliance with each:
   - **Exclude EVERY reflection-pass transcript, not just the current one.** A
     reflection pass writes this SKILL's text, the gathering prompt's pattern
     list, and past candidates-table incident counts into its own `.jsonl`, so
     grepping any of them back inflates friction counts with the engine's own
     vocabulary. Dropping only the current-session file (the original RC-029
     fix) is insufficient — prior weekly-pass transcripts accumulate and each
     one re-contaminates every future scan. Before counting, drop **(a)** the
     newest/self file (by name or mtime) AND **(b)** any transcript that is
     itself a reflection output.

     **Detecting (b) — use the RELIABLE discriminators, NOT a bare path match.**
     Do **not** drop a file just because it matches the backlog path
     `brain/reflection/candidates.md`. That path is embedded verbatim in this
     skill's own catalog `description:`, which Claude Code loads into the system
     context of **every** session in this repo — so a `grep` for it fires ~1×
     as boilerplate on *genuine work* sessions too (verified RC-033: 1 hit on a
     real audit session vs 49 on the reflection pass). A `≥1` path threshold
     therefore wrongly DROPS real work sessions — an under-count, the opposite
     failure mode from RC-030's over-count. Instead, per file, drop it as a
     reflection output if **either**:
     - **(b1) Opening-prompt signal** — the transcript's first records contain
       the reflection scheduled-task header. `head -c 4000 "$f" | grep -qE
       'reflection-weekly|scheduled-task name=.reflection|WEEKLY pass'`. This is
       the strongest, cleanest signal (RC-033: fired on the pass file, silent on
       both work files). Or:
     - **(b2) Trio co-occurrence fingerprint** — the file matches **all three**
       of `gtimeout` AND `nbstripout` AND `index.lock` (a trio a normal dev
       session rarely names together). RC-033: reflection pass = 45/112/62;
       genuine work = 8 nbstripout but **0** gtimeout and **0** index.lock, so
       the AND correctly did not fire. Require all three, not any one.
     - (Optional b3) a HIGH-VOLUME path threshold (e.g. `candidates.md` ≥5) can
       corroborate, but never as the sole criterion — the boilerplate hit makes
       `≥1` unsafe.

     Report which files you dropped and by which signal. (Evidence RC-030: on
     the 2026-07-17 pass, contamination made `gtimeout` and "still failing" 100%
     artifacts and inflated ModuleNotFoundError 13→39, index.lock 34→60,
     nbstripout 125→207 across just two prior-pass files. Evidence RC-033: on
     2026-07-21 the bare path-sentinel fired on 4/4 in-scope files including both
     genuine work sessions; b1+b2 separated the 2 real from the 2 reflection
     files cleanly.)
   - **Grep per file, not multi-file.** A combined `grep -c pattern $(find …)`
     over these very-long single-line JSONL transcripts intermittently returns
     false-zero counts here; a `while read f; do grep … "$f"; done` loop (one
     grep invocation per file) is the reliable pattern.
   - **Call `/usr/bin/grep` by absolute path, and corroborate with two engines.**
     (Root cause found the 2026-07-27 pass — RC-037.) Bare `grep` in this
     environment is **not** GNU/BSD grep: it is a Claude Code shell *function*
     (verified: `type grep` → "shell function from …/shell-snapshots/…") that
     dispatches to a bundled engine carrying `-I` (skip files it deems binary)
     and `--ignore-files`. Huge single-line JSONL transcripts can trip the
     binary heuristic — that is the real mechanism behind the false zeros the
     per-file loop was working around. Run each count through `/usr/bin/grep`
     and, for any pattern you are about to score on, run it a second time
     through the wrapper and require the two to agree before trusting it.
   - **Anchor every pattern; never case-fold a word that occurs in English
     prose.** (RC-037.) Bare `OOM` matches "room", "bloomy", "zoom" — verified
     3 hits vs 1 for `\bOOM\b` on a 3-line fixture, and it manufactured **276
     phantom OOM hits** in one 35MB transcript, which is why earlier passes
     believed this repo had memory pressure it does not have. Likewise
     case-insensitive `FAILED` matches any prose "failed". Use `\b…\b`
     anchors, prefer case-sensitive for acronyms and log-level tokens, and
     re-check any count that looks surprisingly high before scoring it.
     This repo is saturated with skill machinery, so `kill` is especially
     dangerous here: verified 2026-08-03, `-i kill` = **16** records vs
     `\bkilled\b` = **0** on the same file — every hit was "skill"/"skills".
   - **Count RECORDS, not occurrences: `/usr/bin/grep -c`, never
     `grep -o … | wc -l`.** (RC-042, verified live 2026-08-03.) Every tool
     result is stored **twice inside the same JSONL record** — once in the
     `tool_result` content block and once mirrored under the record's
     `toolUseResult` key. So an occurrence count double-counts every
     tool-sourced marker, while `grep -c` (records matched) yields the correct
     distinct-event count. Verified: `No module named 'scipy'` = **4**
     occurrences but **2** records, and both copies sit on the same line
     (`occurrences_in_this_record=2`). Historic counts in this backlog that
     were reported occurrence-style (e.g. "nbstripout ×172") are therefore
     inflated up to 2× — do not compare a new record-count against an old
     occurrence-count and call the difference a trend.

   **Scan TOP-LEVEL transcripts only; nested `subagents/` files are the
   pass's own exhaust.** (RC-042, verified 2026-08-03.) The project dir holds
   both `~/.claude/projects/<repo>/*.jsonl` (real sessions, 54) and
   `~/.claude/projects/<repo>/<session-uuid>/subagents/agent-<id>.jsonl`
   (**228** nested subagent transcripts). Use a top-level glob (`*.jsonl`),
   **never** a bare `find … -name '*.jsonl'`. The reason is not cost, it is
   contamination: on this pass, **all 9** in-window nested files were this
   reflection run's *own* gathering subagents, and **3 of those 5 gathering
   subagents trip NEITHER RC-033 discriminator** (b1 opening-header and b2
   trio both silent for the scratch-lane, radar, and reports agents — their
   prompts never name `reflection-weekly` and never mention all three of
   `gtimeout`/`nbstripout`/`index.lock`). A future pass that widens the glob
   "to be thorough" would therefore ingest the engine's own friction-pattern
   vocabulary as if it were genuine work — the RC-029/RC-030 failure mode,
   amplified per-subagent. If you ever do scan nested files, attribute each to
   its parent session directory and apply the **parent's** include/exclude
   verdict; the layout makes that attribution trivial. (Corollary checked and
   dismissed: this is a contamination vector, not a missed-signal gap — the
   one genuine work session this window spawned **zero** in-window subagents,
   having done its work inline.)

   **Scope files by CONTENT date, not mtime** (RC-037, the highest-cost bug
   found on the 2026-07-27 pass). A transcript's mtime moves when a session is
   *resumed or compacted*, so `ls -lt` promotes long-finished sessions into the
   window and the pass re-mines material earlier passes already scored. Verified
   that pass: **5 of 15 mtime-in-scope files held zero records in the window** —
   e.g. `76d4c3e6` had mtime 07-21 15:30 but content spanning 07-02→07-04 and
   `grep -c '"timestamp":"2026-07-2[1-7]'` = **0**. Those 5 files carried the
   only `index.lock` hits and one of two `timeout`-not-found hits, i.e. the
   stale files were generating the friction "signal". So: use mtime only to
   build a candidate list, then per file confirm at least one record dated after
   the last pass (`grep -c '"timestamp":"<in-window date prefixes>"' "$f"`) and
   DROP any file with zero. Report the mtime-window vs content-window split —
   it is the difference between a real evidence base and a rerun of last month.
4. **maps + reports** — [[SKILL_MAP]] § Future skills, `brain/generated/`
   hygiene/graph/daily-brief reports, `brain/TODO.md` blockers.

### 2. Cluster and verify

Cluster raw signals into named pains ACROSS sessions (one incident ≠ a
cluster). Subagent reports are leads, not facts — **verify on the live system
before accepting evidence** (e.g. the 2026-07-04 pass found "nbstripout
missing" was actually a PATH-context bug plus a TCC-blocked launchd job;
verification changed the fix entirely).

### 3. Score and decide

Use the rubric in `brain/reflection/candidates.md` § Scoring rubric
(recurrence LOW/MED/HIGH × cost S/M/L). You MAY revise the rubric without
asking, but RECORD the rule used in that section. Per cluster decide:
**new skill / automation / fix / nothing**.

### 4. Log — everything

Append to `brain/reflection/candidates.md`: new RC-nnn ids, evidence,
score, decision, status; update the Pass log. Rules:

- "nothing" gets a **reason** (so it never re-surfaces as new).
- Never re-propose: items at a later lifecycle stage (e.g. PACKAGING),
  SKILL_MAP-deferred items (Sherpa/Indeaverse), or closed-nothing entries —
  absent genuinely NEW evidence (then say what changed).
- Radar verdicts: Adopt / Borrow-pattern / Reference / Skip, with licence +
  integration cost (AGPL → reimplement, never vendor).

### 5. Execute accepted candidates end-to-end

Build-now items get built in the same session where feasible: code + passing
tests, registered in [[SKILL_MAP]] with an invocation line. Human-gated items
get status `blocked-human` with the exact step the human must take.

### 6. Graduate

candidate → built (tests green) → SKILL_MAP (registered) → `library/` entry
(generalized, decoupled, IP-scrubbed, licensed). The **human IP/strategy scrub
blocks any public publish** — no exceptions. Everything lands on the operator's
personal branch, never main.

## Guardrails

- No infra before signal; no ML before a rule-based baseline; no cross-import
  polymarket ↔ crypto (brain/CODEX.md — always in force).
- Token discipline: gathering is subagent work; a reflection pass should not
  read the vault end-to-end centrally.
- The engine proposes and builds *tooling*; it never edits research
  conclusions, numbers, or strategy decisions.
