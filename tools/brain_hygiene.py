#!/usr/bin/env python3
"""
brain_hygiene.py — Obsidian/brain hygiene scanner for the epsilon-quant-research vault.

Finds issues; it does NOT fix them. Cleanup is a deliberate, reviewable Janitor pass
(usually handed to Codex). See brain/OPERATING_RHYTHMS.md and brain/SKILL_MAP.md.

Scans all tracked-ish Markdown (excludes .venv, .git, node_modules, .obsidian, and the
generated/ output dir itself) and reports:
  - duplicate basenames (break wikilink navigation)
  - broken wikilinks (target note/path not found)
  - orphan notes (no inbound wikilinks)
  - findings/results notes missing a hub backlink
  - notes missing YAML frontmatter
  - durable notes missing a Summary section
  - stale TODO checkbox items (in files not modified in STALE_DAYS)
  - recently-changed notes (last RECENT_DAYS)

Writes (all under brain/generated/, which is git-ignored and regenerable):
  brain/generated/GENERATED_INDEX.md — navigation index by folder + recent changes
  brain/generated/hygiene_report.md  — the issue report
  brain/generated/stale_notes.md     — stale / unlinked notes + stale TODOs

Usage:
  python tools/brain_hygiene.py            # scan whole repo, write reports
  python tools/brain_hygiene.py --dry-run  # print summary, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------- config
REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", ".git", "node_modules", ".obsidian", ".tmp", "__pycache__",
                ".claude", ".pytest_cache", ".ipynb_checkpoints", ".agents"}
EXCLUDE_REL_PREFIXES = ("brain/generated",)  # don't analyse our own output
# Per-folder convention files where a shared basename is expected and not a navigation bug.
EXPECTED_DUP_BASENAMES = {"README", "CLAUDE", "INDEX", "__init__", "PLAN", "TODO"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf", ".webp", ".canvas"}
HUB_BASENAMES = {
    "COWORK", "CODEX", "TODO", "POLYMARKET_BRAIN", "VAULT_MAP",
    "strat_market_making", "strat_options_delta", "glossary",
    "codex_lane", "cowork_lane",
}
SUMMARY_HEADINGS = ("## summary", "## plain-english summary", "## plain english summary")
STALE_DAYS = 30
RECENT_DAYS = 7

WIKILINK_RE = re.compile(r"!?\[\[([^\]]+)\]\]")
CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[ \]\s+(.*)$")


# --------------------------------------------------------------------------- helpers
def discover_md_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*.md"):
        rel = p.relative_to(root).as_posix()
        if any(part in EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        if rel.startswith(EXCLUDE_REL_PREFIXES):
            continue
        files.append(p)
    return sorted(files)


def has_frontmatter(text: str) -> bool:
    return text.lstrip().startswith("---")


def has_summary(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in SUMMARY_HEADINGS)


def parse_wikilinks(text: str) -> list[str]:
    """Return raw wikilink targets (without alias/heading)."""
    out = []
    for raw in WIKILINK_RE.findall(text):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            out.append(target)
    return out


def is_findings_note(rel: str) -> bool:
    name = rel.lower()
    return (
        name.endswith("_findings.md")
        or name.endswith("_results.md")
        or "/notes/" in name
    )


def age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400.0


# ------------------------------------------------------------------------------ scan
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print summary, write nothing")
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root to scan")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = discover_md_files(root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # indexes
    basename_to_paths: dict[str, list[str]] = defaultdict(list)
    rel_paths: set[str] = set()
    rel_paths_noext: set[str] = set()
    texts: dict[str, str] = {}

    for p in files:
        rel = p.relative_to(root).as_posix()
        rel_paths.add(rel)
        rel_paths_noext.add(rel[:-3] if rel.endswith(".md") else rel)
        basename_to_paths[p.stem].append(rel)
        try:
            texts[rel] = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # pragma: no cover
            texts[rel] = ""
            print(f"warn: could not read {rel}: {e}", file=sys.stderr)

    # also index non-md files (attachments) so embeds/links to them aren't "broken"
    other_paths: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix == ".md":
            continue
        if any(part in EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        other_paths.add(p.relative_to(root).as_posix())
    other_basenames = {Path(x).stem for x in other_paths}

    known_basenames = set(basename_to_paths.keys())

    # link graph + broken links
    inbound: dict[str, set[str]] = defaultdict(set)  # basename -> set of source rels
    broken: list[tuple[str, str]] = []  # (source rel, target)

    def link_resolves(target: str) -> bool:
        # heading-only or empty
        if not target or target.startswith("#"):
            return True
        base = Path(target).stem
        ext = Path(target).suffix.lower()
        # attachment / image link
        if ext in IMAGE_EXTS:
            return target in other_paths or base in other_basenames or target in rel_paths
        # path-style link (has a slash): check file existence with/without .md
        if "/" in target:
            cand = target
            if cand in rel_paths or cand in rel_paths_noext or cand in other_paths:
                return True
            if (cand + ".md") in rel_paths:
                return True
            # fall back to basename match
            return base in known_basenames or base in other_basenames
        # plain basename link
        return target in known_basenames or target in other_basenames

    for rel, text in texts.items():
        for target in parse_wikilinks(text):
            base = Path(target.split("#", 1)[0]).stem
            if base in known_basenames:
                inbound[base].add(rel)
            if not link_resolves(target):
                broken.append((rel, target))

    # duplicate basenames (ignore conventional per-folder files like README/CLAUDE)
    duplicates = {b: ps for b, ps in basename_to_paths.items()
                  if len(ps) > 1 and b not in EXPECTED_DUP_BASENAMES}

    # orphans: notes whose basename has no inbound links and that aren't hubs/generated index
    NEVER_ORPHAN = HUB_BASENAMES | {"GENERATED_INDEX", "SKILL_MAP", "OPERATING_RHYTHMS",
                                    "OBSIDIAN_INFRA_ROADMAP", "README", "INDEX"}
    orphans: list[str] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        if p.stem in NEVER_ORPHAN:
            continue
        if not inbound.get(p.stem):
            orphans.append(rel)

    # missing hub backlink (findings/results/notes only)
    missing_hub: list[str] = []
    for rel, text in texts.items():
        if not is_findings_note(rel):
            continue
        links = {Path(t.split("#", 1)[0]).stem for t in parse_wikilinks(text)}
        if not (links & HUB_BASENAMES):
            missing_hub.append(rel)

    # missing frontmatter / summary
    no_frontmatter: list[str] = []
    no_summary: list[str] = []
    for rel, text in texts.items():
        if not has_frontmatter(text):
            no_frontmatter.append(rel)
        if is_findings_note(rel) and not has_summary(text):
            no_summary.append(rel)

    # stale TODOs: unchecked checkbox items in files not modified in STALE_DAYS
    stale_todos: dict[str, list[str]] = {}
    all_open_todo_count = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        items = [m.group(1).strip() for line in texts[rel].splitlines()
                 if (m := CHECKBOX_RE.match(line))]
        all_open_todo_count += len(items)
        if items and age_days(p) > STALE_DAYS:
            stale_todos[rel] = items

    # recently changed
    recent: list[tuple[str, float]] = []
    for p in files:
        d = age_days(p)
        if d <= RECENT_DAYS:
            recent.append((p.relative_to(root).as_posix(), d))
    recent.sort(key=lambda x: x[1])

    # ----------------------------------------------------------------- summary print
    summary = [
        f"brain_hygiene scan @ {now}",
        f"  markdown files scanned : {len(files)}",
        f"  duplicate basenames    : {len(duplicates)}",
        f"  broken wikilinks       : {len(broken)}",
        f"  orphan notes           : {len(orphans)}",
        f"  missing hub backlink   : {len(missing_hub)}",
        f"  no frontmatter         : {len(no_frontmatter)}",
        f"  findings w/o summary   : {len(no_summary)}",
        f"  files w/ stale TODOs   : {len(stale_todos)}",
        f"  open TODO items total  : {all_open_todo_count}",
        f"  changed last {RECENT_DAYS}d       : {len(recent)}",
    ]
    print("\n".join(summary))

    if args.dry_run:
        print("\n(dry-run: no files written)")
        return 0

    # ----------------------------------------------------------------- write reports
    gen_dir = root / "brain" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # GENERATED_INDEX.md — navigation index by folder
    by_folder: dict[str, list[str]] = defaultdict(list)
    for p in files:
        rel = p.relative_to(root).as_posix()
        folder = str(Path(rel).parent)
        by_folder[folder].append(rel)

    idx = [
        "---",
        "title: Generated Index",
        f"created: {datetime.now().strftime('%Y-%m-%d')}",
        "status: generated",
        "owner: brain_hygiene.py",
        "tags: [obsidian, brain, generated, index]",
        "---",
        "",
        "# Generated Index",
        "",
        "> Auto-generated by `tools/brain_hygiene.py`. Do not edit by hand — rerun the script.",
        f"> Last refreshed: {now}. Files scanned: {len(files)}.",
        "",
        "Hub links: [[VAULT_MAP]] | [[SKILL_MAP]] | [[OPERATING_RHYTHMS]]",
        "",
        "## Health summary",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| Markdown files | {len(files)} |",
        f"| Duplicate basenames | {len(duplicates)} |",
        f"| Broken wikilinks | {len(broken)} |",
        f"| Orphan notes | {len(orphans)} |",
        f"| Missing hub backlink | {len(missing_hub)} |",
        f"| No frontmatter | {len(no_frontmatter)} |",
        f"| Findings without summary | {len(no_summary)} |",
        f"| Files with stale TODOs | {len(stale_todos)} |",
        "",
        f"Detail: `brain/generated/hygiene_report.md` · `brain/generated/stale_notes.md`",
        "",
        f"## Recently changed (last {RECENT_DAYS} days)",
        "",
    ]
    if recent:
        for rel, d in recent:
            idx.append(f"- `{rel}` — {d:.1f}d ago")
    else:
        idx.append("_None._")
    idx += ["", "## All notes by folder", ""]
    for folder in sorted(by_folder):
        idx.append(f"### `{folder}/` ({len(by_folder[folder])})")
        idx.append("")
        for rel in sorted(by_folder[folder]):
            idx.append(f"- [[{Path(rel).stem}]]")
        idx.append("")

    (gen_dir / "GENERATED_INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    # hygiene_report.md
    rep = [
        "---",
        "title: Hygiene Report",
        f"created: {datetime.now().strftime('%Y-%m-%d')}",
        "status: generated",
        "owner: brain_hygiene.py",
        "tags: [obsidian, brain, generated, hygiene]",
        "---",
        "",
        "# Hygiene Report",
        "",
        "> Auto-generated by `tools/brain_hygiene.py`. **Finds issues; does not fix them.**",
        "> Hand to Codex for a focused Janitor pass — see [[OPERATING_RHYTHMS]] and [[SKILL_MAP]].",
        f"> Last refreshed: {now}.",
        "",
        "Hub links: [[VAULT_MAP]] | [[GENERATED_INDEX]]",
        "",
    ]

    def section(title: str, rows: list[str], empty="_None — clean._"):
        rep.append(f"## {title} ({len(rows)})")
        rep.append("")
        rep.extend(rows if rows else [empty])
        rep.append("")

    section("Duplicate basenames", [
        f"- **{b}** → " + ", ".join(f"`{x}`" for x in ps) for b, ps in sorted(duplicates.items())
    ])
    section("Broken wikilinks", [
        f"- `{src}` → `[[{tgt}]]`" for src, tgt in sorted(broken)
    ])
    section("Orphan notes (no inbound wikilinks)", [f"- `{r}`" for r in sorted(orphans)])
    section("Findings/notes missing a hub backlink", [f"- `{r}`" for r in sorted(missing_hub)])
    section("Notes missing YAML frontmatter", [f"- `{r}`" for r in sorted(no_frontmatter)])
    section("Findings notes missing a Summary section", [f"- `{r}`" for r in sorted(no_summary)])

    (gen_dir / "hygiene_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    # stale_notes.md
    stale = [
        "---",
        "title: Stale Notes",
        f"created: {datetime.now().strftime('%Y-%m-%d')}",
        "status: generated",
        "owner: brain_hygiene.py",
        "tags: [obsidian, brain, generated, stale]",
        "---",
        "",
        "# Stale Notes",
        "",
        "> Auto-generated by `tools/brain_hygiene.py`.",
        f"> 'Stale' = file not modified in {STALE_DAYS}+ days. Last refreshed: {now}.",
        "",
        "Hub links: [[VAULT_MAP]] | [[GENERATED_INDEX]]",
        "",
        f"## Stale notes (not modified in {STALE_DAYS}+ days)",
        "",
    ]
    stale_files = sorted(
        ((p.relative_to(root).as_posix(), age_days(p)) for p in files if age_days(p) > STALE_DAYS),
        key=lambda x: -x[1],
    )
    if stale_files:
        stale += [f"- `{rel}` — {d:.0f}d" for rel, d in stale_files]
    else:
        stale.append("_None._")
    stale += ["", f"## Stale open TODOs (in files {STALE_DAYS}+ days old)", ""]
    if stale_todos:
        for rel in sorted(stale_todos):
            stale.append(f"### `{rel}`")
            stale += [f"- [ ] {item}" for item in stale_todos[rel][:25]]
            if len(stale_todos[rel]) > 25:
                stale.append(f"- … +{len(stale_todos[rel]) - 25} more")
            stale.append("")
    else:
        stale.append("_None._")

    (gen_dir / "stale_notes.md").write_text("\n".join(stale) + "\n", encoding="utf-8")

    print(f"\nwrote: brain/generated/GENERATED_INDEX.md, brain/generated/hygiene_report.md, "
          f"brain/generated/stale_notes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
