#!/usr/bin/env python3
"""
wip_audit.py — read-only "what work exists only on this disk?" reporter.

Finds exposure; it does NOT fix it. Committing, staging, pushing and branch
pruning stay deliberate human/agent decisions (see brain/MERGE_PROTOCOL.md).
This tool never mutates git state: no add, commit, stash, checkout, push or gc.

The gap it closes: tools/brain_hygiene.py audits *notes* and
tools/brain_graph_audit.py audits the *link graph*, but nothing reports that a
research or execution thread has been sitting uncommitted for weeks, or that a
new capability exists in no git object at all. Three consecutive reflection
passes re-derived that picture by hand (brain/reflection/candidates.md RC-036).

Reports:
  - uncommitted working-tree paths, bucketed by age, grouped by thread
  - never-versioned paths (untracked: present in NO commit, so a lost disk
    loses them outright) — the highest-exposure class
  - deletions whose basename still exists elsewhere in the tree, i.e. the
    likely unstaged tail of a large rename/reorg
  - local commits absent from every remote (backup exposure), per branch,
    including branches with no upstream at all
  - `status: active` handoff/PRD notes that are untracked (context that exists
    only on one machine)

Writes (under brain/generated/, which is git-ignored and regenerable):
  brain/generated/wip_audit.md

Usage:
  python3 tools/wip_audit.py                 # scan repo, write the report
  python3 tools/wip_audit.py --dry-run       # print summary, write nothing
  python3 tools/wip_audit.py --stale-days 21 # change the stale-WIP threshold
  python3 tools/wip_audit.py --repo PATH     # audit another clone (used by tests)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------- config
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent

STALE_DAYS = 14          # WIP older than this is flagged as stale drift
WARN_DAYS = 7            # WIP older than this is worth a look
REPORT_REL = "brain/generated/wip_audit.md"

# Paths that are dirty by design and are not "work at risk".
IGNORE_PREFIXES = (
    ".obsidian/",        # editor/plugin state, not work product
)

# Frontmatter status values that mark a note as live context worth protecting.
ACTIVE_STATUS_DIRS = ("brain/handoffs/", "brain/reflection/")


# ----------------------------------------------------------------------------- git helpers
def _git(repo: Path, *args: str) -> str:
    """Run a read-only git command; return stdout (empty string on failure)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout if out.returncode == 0 else ""


def parse_status(repo: Path) -> list[tuple[str, str]]:
    """
    Parse `git status --porcelain=v1 -z --untracked-files=all`.

    Returns [(xy, path)] where xy is the two-char status code. Rename/copy
    records carry a second NUL-separated origin path, which we consume and
    report against the *new* path (the origin shows up as a D only if the
    rename was left unstaged, which is exactly the case we want to surface).
    """
    raw = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not raw:
        return []
    records = [r for r in raw.split("\0") if r]
    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(records):
        rec = records[i]
        if len(rec) < 4:
            i += 1
            continue
        xy, path = rec[:2], rec[3:]
        if xy[0] in ("R", "C"):
            i += 1  # the following record is the origin path; skip it
        entries.append((xy, path))
        i += 1
    return entries


def path_age_days(repo: Path, path: str, now: float) -> float | None:
    """
    Age of a working-tree path in days.

    Uses the file mtime when the path exists. For a deleted path there is no
    mtime, so fall back to the last commit that touched it — that is the age of
    the state the deletion is diverging from. Returns None if neither is known.
    """
    fp = repo / path
    if fp.exists():
        return max(0.0, (now - fp.stat().st_mtime) / 86400.0)
    ts = _git(repo, "log", "-1", "--format=%ct", "--", path).strip()
    if ts.isdigit():
        return max(0.0, (now - int(ts)) / 86400.0)
    return None


def thread_of(path: str) -> str:
    """
    Group paths into a coarse 'thread' label.

    Two leading directory components when there are that many (so
    `polymarket/execution/cli.py` → `polymarket/execution`), otherwise the
    containing directory. Never the file itself: labelling
    `meetings/README.md` as its own thread is noise, not a grouping.
    """
    parts = Path(path).parts
    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) == 2:
        return parts[0]
    return "(root)"


def find_twins(repo: Path, path: str, tracked: dict[str, list[str]]) -> list[dict]:
    """
    Other copies of a deleted path's basename still present in the tree.

    Checks tracked files first, then falls back to a filesystem search — because
    the most informative case is a file moved into a **git-ignored** tree (the
    reorg kept the bytes but dropped them out of version control entirely), and
    `git ls-files` is blind to exactly that. Deletions are few, so a bounded
    per-path search is cheap.
    """
    name = Path(path).name
    twins = [
        {"path": t, "tracked": True, "ignored": False}
        for t in tracked.get(name, [])
        if t != path
    ]
    if twins:
        return twins

    try:
        found = subprocess.run(
            [
                "find", ".", "-name", name,
                "-not", "-path", "./.git/*",
                "-not", "-path", "*/.venv/*",
                "-not", "-path", "*/node_modules/*",
                "-not", "-path", "*/__pycache__/*",
            ],
            cwd=str(repo), capture_output=True, text=True, check=False, timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []

    for line in found.splitlines():
        rel = line[2:] if line.startswith("./") else line
        if not rel or rel == path:
            continue
        ignored = bool(_git(repo, "check-ignore", rel).strip())
        twins.append({"path": rel, "tracked": False, "ignored": ignored})
    return twins


def unpushed_branches(repo: Path) -> list[dict]:
    """
    Local commits that exist on no remote, per local branch.

    `--not --remotes` is the honest measure of backup exposure: a branch can be
    'in sync' with a missing upstream yet still have commits no remote holds.
    Branches with no upstream configured are reported explicitly.
    """
    fmt = "%(refname:short)\t%(upstream:short)"
    rows = _git(repo, "for-each-ref", "--format=" + fmt, "refs/heads").splitlines()
    out: list[dict] = []
    for row in rows:
        if not row.strip():
            continue
        parts = row.split("\t")
        branch = parts[0].strip()
        upstream = parts[1].strip() if len(parts) > 1 else ""
        n_raw = _git(repo, "rev-list", "--count", branch, "--not", "--remotes").strip()
        n_unremoted = int(n_raw) if n_raw.isdigit() else 0
        ahead = None
        if upstream:
            a_raw = _git(repo, "rev-list", "--count", f"{upstream}..{branch}").strip()
            ahead = int(a_raw) if a_raw.isdigit() else None
        out.append(
            {
                "branch": branch,
                "upstream": upstream,
                "ahead_of_upstream": ahead,
                "commits_on_no_remote": n_unremoted,
            }
        )
    return out


def tracked_basenames(repo: Path) -> dict[str, list[str]]:
    """basename -> tracked paths, used to spot stranded rename tails."""
    index: dict[str, list[str]] = defaultdict(list)
    for line in _git(repo, "ls-files").splitlines():
        if line.strip():
            index[Path(line).name].append(line)
    return index


def active_untracked_notes(repo: Path, untracked: list[str]) -> list[str]:
    """Untracked notes under handoff/reflection dirs whose frontmatter is active."""
    hits = []
    for p in untracked:
        if not p.endswith(".md") or not p.startswith(ACTIVE_STATUS_DIRS):
            continue
        fp = repo / p
        try:
            head = fp.read_text(encoding="utf-8", errors="replace")[:800]
        except OSError:
            continue
        for line in head.splitlines():
            if line.strip().lower().startswith("status:"):
                if "active" in line.lower():
                    hits.append(p)
                break
    return hits


# ----------------------------------------------------------------------------- audit
def audit(repo: Path, stale_days: int = STALE_DAYS, now: float | None = None) -> dict:
    """Collect the full read-only picture. Pure: mutates nothing."""
    now = time.time() if now is None else now
    entries = parse_status(repo)

    kept = [
        (xy, p) for xy, p in entries
        if not p.startswith(IGNORE_PREFIXES)
    ]
    ignored = len(entries) - len(kept)

    basenames = tracked_basenames(repo)

    untracked: list[str] = []
    items: list[dict] = []
    rename_tails: list[dict] = []

    for xy, path in kept:
        age = path_age_days(repo, path, now)
        # A path is "never versioned" only when git has no record of it at all.
        never_versioned = xy == "??"
        if never_versioned:
            untracked.append(path)
        item = {
            "xy": xy,
            "path": path,
            "age_days": age,
            "never_versioned": never_versioned,
            "thread": thread_of(path),
        }
        items.append(item)

        if "D" in xy:
            twins = find_twins(repo, path, basenames)
            if twins:
                rename_tails.append(
                    {
                        "path": path,
                        "twins": twins[:3],
                        "age_days": age,
                        # A twin that is itself ignored means the bytes survived the
                        # move but left version control — the deletion is not the
                        # whole story, the destination is unversioned too.
                        "dest_unversioned": all(
                            (not t["tracked"]) for t in twins
                        ),
                    }
                )

    aged = [i for i in items if i["age_days"] is not None]
    oldest = max((i["age_days"] for i in aged), default=None)

    stale = [i for i in aged if i["age_days"] >= stale_days]
    warn = [i for i in aged if warn_band(i["age_days"], stale_days)]

    by_thread: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_thread[i["thread"]].append(i)

    branches = unpushed_branches(repo)
    commits_at_risk = sum(b["commits_on_no_remote"] for b in branches)

    return {
        "repo": str(repo),
        "generated_utc": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stale_days": stale_days,
        "items": items,
        "ignored_count": ignored,
        "untracked": untracked,
        "never_versioned_count": len(untracked),
        "oldest_age_days": oldest,
        "stale": stale,
        "warn": warn,
        "by_thread": dict(by_thread),
        "rename_tails": rename_tails,
        "branches": branches,
        "commits_on_no_remote": commits_at_risk,
        "active_untracked_notes": active_untracked_notes(repo, untracked),
    }


def warn_band(age: float, stale_days: int) -> bool:
    """True for the 'worth a look' band: WARN_DAYS <= age < stale_days."""
    return WARN_DAYS <= age < stale_days


def headline(a: dict) -> str:
    """One-line summary, suitable for the daily brief."""
    oldest = a["oldest_age_days"]
    oldest_txt = f"{oldest:.0f}d" if oldest is not None else "n/a"
    return (
        f"{len(a['items'])} uncommitted paths "
        f"({a['never_versioned_count']} never versioned, oldest {oldest_txt}); "
        f"{a['commits_on_no_remote']} commits on no remote"
    )


# ----------------------------------------------------------------------------- report
def _age(i: dict) -> str:
    return "n/a" if i["age_days"] is None else f"{i['age_days']:.0f}d"


def render(a: dict) -> str:
    L: list[str] = []
    L += [
        "---",
        "title: WIP Exposure Audit",
        f"created: {a['generated_utc'][:10]}",
        "status: generated",
        "owner: wip_audit.py",
        "tags: [git, brain, generated, wip, backup]",
        "---",
        "",
        "# WIP Exposure Audit",
        "",
        "> Auto-generated by `tools/wip_audit.py`. **Finds exposure; does not fix it.**",
        "> Read-only: this tool never stages, commits, pushes or prunes anything.",
        f"> Last refreshed: {a['generated_utc']}.",
        "",
        "Hub links: [[VAULT_MAP]] | [[MERGE_PROTOCOL]] | [[GENERATED_INDEX]]",
        "",
        "## Summary",
        "",
        f"- {headline(a)}",
        f"- Stale WIP (>= {a['stale_days']}d): **{len(a['stale'])}** paths · "
        f"ageing ({WARN_DAYS}-{a['stale_days']}d): {len(a['warn'])}",
        f"- Likely unstaged rename tails: {len(a['rename_tails'])}",
        f"- Untracked `status: active` context notes: {len(a['active_untracked_notes'])}",
    ]
    if a["ignored_count"]:
        L.append(f"- Ignored by design (editor/plugin state): {a['ignored_count']}")
    L.append("")

    # --- never versioned
    L += [f"## Never versioned ({a['never_versioned_count']})", ""]
    if not a["untracked"]:
        L += ["_None — every working-tree path is at least known to git._", ""]
    else:
        L += [
            "Present in **no commit**: losing this disk loses them outright.",
            "",
            "| age | path |",
            "|---|---|",
        ]
        u = [i for i in a["items"] if i["never_versioned"]]
        u.sort(key=lambda i: -(i["age_days"] or 0))
        L += [f"| {_age(i)} | `{i['path']}` |" for i in u]
        L.append("")

    # --- stale WIP
    L += [f"## Stale WIP (>= {a['stale_days']} days) ({len(a['stale'])})", ""]
    if not a["stale"]:
        L += ["_None — no working-tree path has been sitting that long._", ""]
    else:
        L += ["| age | status | path |", "|---|---|---|"]
        for i in sorted(a["stale"], key=lambda i: -(i["age_days"] or 0)):
            L.append(f"| {_age(i)} | `{i['xy']}` | `{i['path']}` |")
        L.append("")

    # --- by thread
    L += ["## Uncommitted paths by thread", ""]
    if not a["by_thread"]:
        L += ["_Clean working tree._", ""]
    else:
        L += ["| thread | paths | oldest | never versioned |", "|---|---|---|---|"]
        rows = []
        for thread, items in a["by_thread"].items():
            ages = [i["age_days"] for i in items if i["age_days"] is not None]
            oldest = max(ages) if ages else None
            rows.append(
                (
                    oldest if oldest is not None else -1,
                    f"| `{thread}` | {len(items)} | "
                    f"{'n/a' if oldest is None else f'{oldest:.0f}d'} | "
                    f"{sum(1 for i in items if i['never_versioned'])} |",
                )
            )
        L += [r[1] for r in sorted(rows, key=lambda r: -r[0])]
        L.append("")

    # --- rename tails
    L += [f"## Likely unstaged rename tails ({len(a['rename_tails'])})", ""]
    if not a["rename_tails"]:
        L += ["_None._", ""]
    else:
        L += [
            "A deletion whose basename still exists elsewhere in the tree — usually",
            "the correct-but-unstaged tail of a large move/reorg commit. `dest`",
            "flags whether the destination copy is itself under version control:",
            "`unversioned` means the bytes survived the move but left git entirely",
            "(e.g. moved into a git-ignored data tree), so staging the deletion",
            "would drop the last versioned copy.",
            "",
            "| age | deleted path | same basename now at | dest |",
            "|---|---|---|---|",
        ]
        for r in a["rename_tails"]:
            age = "n/a" if r["age_days"] is None else f"{r['age_days']:.0f}d"
            t = r["twins"][0]
            dest = "tracked" if t["tracked"] else (
                "**unversioned (ignored)**" if t["ignored"] else "**untracked**"
            )
            L.append(f"| {age} | `{r['path']}` | `{t['path']}` | {dest} |")
        L.append("")

    # --- active untracked notes
    L += [
        f"## Untracked `status: active` context notes ({len(a['active_untracked_notes'])})",
        "",
    ]
    if not a["active_untracked_notes"]:
        L += ["_None._", ""]
    else:
        L += ["Live handoff/reflection context that exists on one machine only.", ""]
        L += [f"- `{p}`" for p in a["active_untracked_notes"]]
        L.append("")

    # --- branches
    L += ["## Commits on no remote", ""]
    L += ["| branch | upstream | ahead of upstream | on no remote |", "|---|---|---|---|"]
    for b in sorted(a["branches"], key=lambda b: -b["commits_on_no_remote"]):
        up = b["upstream"] or "_(none)_"
        ahead = "n/a" if b["ahead_of_upstream"] is None else str(b["ahead_of_upstream"])
        L.append(f"| `{b['branch']}` | {up} | {ahead} | **{b['commits_on_no_remote']}** |")
    L += [
        "",
        "`on no remote` counts commits reachable from the branch but from no remote ref",
        "(`git rev-list --count <branch> --not --remotes`) — the honest backup measure,",
        "since a branch with a missing upstream can read as 'in sync' yet be unbacked.",
        "",
        "> Pruning branches and pushing are **human decisions** — see [[MERGE_PROTOCOL]].",
        "> This report deliberately stops at reporting.",
        "",
    ]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only WIP/backup exposure audit.")
    ap.add_argument("--repo", default=str(DEFAULT_REPO_ROOT), help="repo to audit")
    ap.add_argument("--stale-days", type=int, default=STALE_DAYS)
    ap.add_argument("--dry-run", action="store_true", help="print summary, write nothing")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"not a git repo: {repo}", file=sys.stderr)
        return 2

    a = audit(repo, stale_days=args.stale_days)
    print(headline(a))
    if a["stale"]:
        print(f"  stale (>= {a['stale_days']}d): {len(a['stale'])} paths")
    if a["active_untracked_notes"]:
        print(f"  untracked active context notes: {len(a['active_untracked_notes'])}")

    if args.dry_run:
        return 0

    out = repo / REPORT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a), encoding="utf-8")
    print(f"wrote: {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
