# Worked example — fictional PRD chat → emitted goal prompt

A condensed planning dialogue for an invented project ("changelog digest
bot"), showing the question cadence, the pushback on scope, and the emitted
prompt. Everything is fictional.

## The dialogue (condensed)

**Planner:** Before content: what runs the final prompt? Any length cap or
required sections?
**Human:** Our CLI coding agent; keep it under one screen; we always start
prompts with a read-order line.

**Planner:** Goals — when this works, what exists? Who reads the output?
**Human:** A weekly digest of merged PRs, posted to the team channel. Readers
are non-engineers.

**Planner:** Non-goals worth writing down? For example: does it summarize
code diffs, or only PR titles/descriptions?
**Human:** Only titles/descriptions — no diff reading. And no real-time
posting, weekly is fine.

**Planner:** v1 slice: I'd argue posting can wait — the risky part is whether
the digests are actually readable to non-engineers. Cheapest gate: generate
digests for the last 3 real weeks, show them to two non-engineers, ask if
they'd read them weekly. If not, stop before any bot infrastructure. Agree?
**Human:** Agreed. If both say no, kill it.

**Planner:** Execution latitude: may the agent create the channel webhook
itself, or is that yours?
**Human:** Mine — secrets stay human-managed.

## The emitted goal prompt

```
Build "changelog digest bot": a weekly plain-English digest of merged PRs for
non-engineers, posted to the team channel. It reads PR titles/descriptions
ONLY — never diffs — and it is NOT a real-time notifier.

Read first, in order: CONTRIBUTING.md, docs/tone-guide.md.

GOALS: (1) a digest generator over merged PRs (title+description only);
(2) weekly automated posting via the channel webhook.
NON-GOALS: diff summarization; real-time posting; any secret handling.

PHASING (validate before build — v0 GATES v1):
- v0 GATE: generate digests for the last 3 real weeks. Show both reviewers
  named by the operator; pre-registered bar: BOTH say they would read it
  weekly. If not: STOP, report verbatim feedback, build nothing further.
- v1: the generator as a scheduled job, dry-run output to a file.
- v2 (deferred): channel posting — blocked on the operator creating the
  webhook (secrets are human-managed).

SUCCESS: v0 = the two-reviewer verdict, documented. v1 = a scheduled dry-run
producing a digest file weekly with no manual step.
GUARDRAILS: tone per docs/tone-guide.md; no secrets in the repo; the webhook
step is human-only; total scheduled-job cost < $5/month.
```

## What happened (fictional epilogue, the point of the gate)

One reviewer said the digests read like commit logs. The v0 gate fired, the
agent stopped exactly as pre-registered, and the feedback ("group by user
impact, not by PR") became the input to the next planning round — total cost
of the false start: one afternoon, zero infrastructure.
