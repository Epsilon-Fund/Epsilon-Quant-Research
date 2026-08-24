# Worked example — a fictional backlog after two passes

Everything below is invented to show the mechanics: ids, evidence, the
recurrence × build-cost score, decisions, and the "nothing with a reason"
discipline.

## Scoring rubric (recorded)

v1 rule as in SKILL.md § 3. No revisions yet.

## Candidates

| id | pain / cluster | evidence | rec | cost | decision | status |
|---|---|---|---|---|---|---|
| RC-001 | Test suite needs a hand-set env var; every fresh session trips on it | 6 transcript hits across 4 sessions; commits `a1b2c3d`, `d4e5f6a` ("fix env again") | HIGH | S | **fix** — default the var in `conftest.py`, document the override | built (2026-03-02) |
| RC-002 | Release notes are hand-assembled from git log every release | 3 releases, ~30 min each (release checklists in notes/) | MED | M | **plan** — changelog generator; schedule with the next release, don't build mid-sprint | planned |
| RC-003 | "We should switch task runners" comes up in retros | 1 retro note | LOW | L | **nothing** — single mention, no measured pain; re-open only if a concrete failure is traced to the current runner | closed-nothing |
| RC-004 | Agent keeps re-asking where fixtures live | 2 sessions | MED | S | **fix** — one line in CONTRIBUTING.md § layout ("fixtures: tests/fixtures/, one dir per module") | built (2026-03-02) |

## Pass log

| date | signals used | outcome |
|---|---|---|
| 2026-03-02 | git log 02-01→03-02 · notes/ · ~12 transcripts (sampled) | 4 candidates: 2 built, 1 planned, 1 closed-nothing |
| 2026-03-09 | git log 03-02→03-09 · notes/ | 0 new candidates. RC-003 re-surfaced in a retro note — **not re-proposed** (closed-nothing, no new evidence; the recorded reason did its job) |
