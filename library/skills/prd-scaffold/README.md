# prd-scaffold

A copyable agent skill for the planning half of agent-assisted building:
co-author a PRD through structured Q&A (no assumptions — ask), force the
decisions that actually matter (v1 slice, validate-before-build gate,
success criteria, execution latitude, guardrails), then emit **one
self-contained goal prompt** an implementation agent can execute without the
planning chat in the room.

Prompt-ware only: no code, no dependencies. The worked example in
[EXAMPLE.md](EXAMPLE.md) — dialogue → emitted prompt → the gate firing — is
the demo.

## Install

```bash
cp -r prd-scaffold  your-project/.claude/skills/
```

(or `~/.claude/skills/` to make it available everywhere)

## Why the two-role split

The planning chat holds context the implementation agent will never see, so
everything load-bearing must survive the handoff *inside the emitted prompt*.
The skill's final step — reread the prompt cold, as the implementation agent
— is where most handoff bugs die.

## Provenance

Generalized from the PRD → goal-prompt flow Epsilon uses to hand research
projects to implementation agents (both uses to date shipped, one via a
pre-registered v0 gate that correctly STOPPED a build). Authored clean — the
flow is the content, not Epsilon's PRDs.

## License

Apache-2.0 — see [LICENSE](LICENSE).
