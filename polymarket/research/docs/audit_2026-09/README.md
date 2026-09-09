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
| `03_dataset_overview.html` | **generated artefact** (8 MB) — the dataset overview rendered as a standalone page; regenerates from `03_numbers.json`, so treat the JSON as the source of truth if the two ever disagree |
| `04_audit_plan.md` | **start here.** §0 is the two verdicts; §1–2 the independent mapping and reconciliation checks; §3 what the anti-drift test does and doesn't prove; §4 the twelve ranked checks with results; §5 the 22 "found, not admitted" items |
| `04_results/*.txt` | raw stdout from the check scripts in `../../scripts/audit_checks/` |
| `research_v1_audit.html` | **generated artefact** (17 MB) — the whole audit as one self-contained offline page (verdicts, dataset, dashboard, mapping, reconciliation, anti-drift, plan, findings, raw output), built from the Markdown and JSON in this folder |
| `screenshots/` | 39 files — 38 PNG captures plus `_capture_log_audit.json` (what was captured, when). The top level is the live Streamlit dashboard panel by panel (landing, esports/politics market headers, explore and audit modes, the six audit tabs, the overview panels); `site_check/` is 15 captures of `research_v1_audit.html` itself, taken to verify the generated page renders every section |

One correction to the copies: `04_audit_plan.md` describes `scripts/audit_checks/` as git-excluded
scratch. As of this branch the scripts **are** in the index — see
[`../../scripts/audit_checks/README.md`](../../scripts/audit_checks/README.md).

Handoff note, including the four asks for Alvaro:
[`brain/handoffs/2026-09-02-research-v1-audit-justin.md`](../../../../brain/handoffs/2026-09-02-research-v1-audit-justin.md).
