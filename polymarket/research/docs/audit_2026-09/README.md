# research_v1 independent audit — 2026-09-01/02

Verbatim copies of the audit session's artefacts, committed so the findings and the numbers behind
them live in the repo rather than in a chat transcript. Audited state: branch `alvaro` @ `9810c2b`,
data `polymarket/research/data/research_v1`. **Read-only against R2 throughout** — 19 `GetObject`
(157 MB, all under `parquet/`), zero put/delete/copy.

| file | what it is |
|---|---|
| `01_digest.md` | the audit digest: what the dataset contains (re-derived, not copied), and §8 "what I found wrong" — the ten defect classes, each measured |
| `02_dashboard_walkthrough.md` | the Streamlit dashboard driven panel by panel, with what each panel actually shows vs what it claims |
| `03_numbers.json` | every number quoted in the write-ups, with the query that produced it |
| `04_audit_plan.md` | **start here.** §0 is the two verdicts; §1–2 the independent mapping and reconciliation checks; §3 what the anti-drift test does and doesn't prove; §4 the twelve ranked checks with results; §5 the 22 "found, not admitted" items |
| `04_results/*.txt` | raw stdout from the check scripts in `../../scripts/audit_checks/` |

Not copied: `03_dataset_overview.html` (8 MB, generated from `03_numbers.json`) and the dashboard
screenshots.

One correction to the copies: `04_audit_plan.md` describes `scripts/audit_checks/` as git-excluded
scratch. As of this branch the scripts **are** in the index — see
[`../../scripts/audit_checks/README.md`](../../scripts/audit_checks/README.md).

Handoff note, including the four asks for Alvaro:
[`brain/handoffs/2026-09-02-research-v1-audit-justin.md`](../../../../brain/handoffs/2026-09-02-research-v1-audit-justin.md).
