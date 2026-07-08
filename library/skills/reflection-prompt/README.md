# reflection-prompt

A copyable agent skill: mine your own recent work (git history, notes,
session transcripts) for **recurring** pain, score each cluster by
recurrence × build-cost, decide *build / automate / fix / nothing* — and log
every decision, including "nothing" **with a reason**, to a persistent
backlog so rejected ideas stay rejected.

Prompt-ware only: no code, no dependencies. The worked example in
[EXAMPLE.md](EXAMPLE.md) is the demo.

## Install

```bash
cp -r reflection-prompt  your-project/.claude/skills/
```

(or `~/.claude/skills/` to make it available everywhere)

## What it produces

One backlog file in your project (suggested: `reflection/candidates.md`) with
stable candidate ids, evidence, scores, decisions, lifecycle statuses, and a
pass log. The two rules that make the loop converge: *"nothing" always gets a
recorded reason*, and *nothing is ever re-proposed without new evidence*.

## Provenance

Generalized from the reflection engine Epsilon runs weekly over its own
research monorepo (first pass logged 22 candidates and shipped 3 fixes the
same day). This public bundle is authored clean — the rubric and the
discipline are the content, not Epsilon's backlog.

## License

Apache-2.0 — see [LICENSE](LICENSE).
