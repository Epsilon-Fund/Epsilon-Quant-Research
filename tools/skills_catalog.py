#!/usr/bin/env python3
"""
skills_catalog.py — machine-readable catalog + minimal local dashboard for the
skills lifecycle (brain/reflection/candidates.md RC-003).

Reads three canonical surfaces:
  1. .agents/skills/*/SKILL.md          — installed agent skills (frontmatter)
  2. brain/SKILL_MAP.md                 — brain workflow / runtime skill tables
  3. library/*/pyproject.toml (+ bundled src/**/skills/*/SKILL.md)
                                        — extracted public-candidate packages

Emits:
  brain/generated/skills_catalog.json   — FULL internal catalog (git-ignored, regenerable)
  brain/generated/skills_dashboard.html — minimal local dashboard (git-ignored)
  library/catalog.json                  — PUBLIC-CANDIDATE subset (committed; every
                                          entry carries scrub_status until the human
                                          IP/strategy scrub signs off — nothing here
                                          is published)

Run from the repo root:  python3 tools/skills_catalog.py
Stdlib only (3.10+); the website colleague consumes library/catalog.json.
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "brain" / "generated"


# ── SKILL.md frontmatter ─────────────────────────────────────────────────────
def parse_skill_md(path: Path) -> dict:
    """name + description from YAML frontmatter (handles `key: >` folded blocks)."""
    out = {"name": path.parent.name, "description": ""}
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return out
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        km = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if km:
            key, val = km.group(1), km.group(2).strip()
            if val in (">", "|", ">-", "|-"):
                block = []
                i += 1
                while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                    block.append(lines[i].strip())
                    i += 1
                out[key] = " ".join(b for b in block if b)
                continue
            out[key] = val
        i += 1
    return out


def agent_skills() -> list[dict]:
    entries = []
    for skill_md in sorted((ROOT / ".agents" / "skills").glob("*/SKILL.md")):
        meta = parse_skill_md(skill_md)
        body = skill_md.read_text(encoding="utf-8")
        first_party = bool(re.search(r"first-party", body, re.IGNORECASE))
        vendored = bool(re.search(r"vendor", body, re.IGNORECASE)) and not first_party
        entries.append({
            "id": meta["name"],
            "kind": "agent-skill",
            "summary": meta.get("description", ""),
            "provenance": "first-party" if first_party else ("vendored" if vendored else "unspecified"),
            "source": str(skill_md.relative_to(ROOT)),
            "invocation": f"Skill: {meta['name']} (auto/prompt-invoked per description)",
        })
    return entries


# ── SKILL_MAP tables ─────────────────────────────────────────────────────────
def _table_rows(section_text: str) -> list[list[str]]:
    rows = []
    for line in section_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows[1:] if rows else []          # drop header row


def _clean(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def skill_map_entries() -> list[dict]:
    text = (ROOT / "brain" / "SKILL_MAP.md").read_text(encoding="utf-8")
    sections = re.split(r"^(#{2,3} .+)$", text, flags=re.MULTILINE)
    titled = {}
    for i in range(1, len(sections) - 1, 2):
        titled[sections[i].lstrip("# ").strip()] = sections[i + 1]

    entries = []
    for row in _table_rows(titled.get("Core skills", "")):
        if len(row) >= 2 and row[0]:
            entries.append({
                "id": _clean(row[0]), "kind": "brain-workflow",
                "summary": f"trigger: {_clean(row[1])}; produces: {_clean(row[3]) if len(row) > 3 else ''}",
                "provenance": "first-party", "source": "brain/SKILL_MAP.md § Core skills",
                "invocation": f'prompt: "Run a {_clean(row[0])} pass" (see SKILL_MAP prompt pack)',
            })
    for row in _table_rows(titled.get("Agent runtime skills", "")):
        if len(row) >= 2 and row[0]:
            entries.append({
                "id": _clean(row[0]), "kind": "runtime",
                "summary": _clean(row[1]),
                "provenance": "see SKILL_MAP", "source": "brain/SKILL_MAP.md § Agent runtime skills",
                "invocation": f"mention `{_clean(row[0])}` in the prompt when it matters",
            })
    for row in _table_rows(titled.get("Runtime efficiency skills", "")):
        if len(row) >= 2 and row[0]:
            entries.append({
                "id": _clean(row[0]).split(" ")[0], "kind": "runtime-efficiency",
                "summary": f"auto-trigger: {_clean(row[1])}",
                "provenance": "vendored-or-first-party (see SKILL_MAP)",
                "source": "brain/SKILL_MAP.md § Runtime efficiency skills",
                "invocation": "auto-triggered in Claude Code by description match",
            })
    fut = titled.get("Future skills (deferred)", "")
    for m in re.finditer(r"^- \*\*(.+?)\*\* — (.+)$", fut, re.MULTILINE):
        entries.append({
            "id": m.group(1).strip(), "kind": "future-deferred",
            "summary": m.group(2).strip(),
            "provenance": "unbuilt", "source": "brain/SKILL_MAP.md § Future skills",
            "invocation": "n/a (deferred by design — reflection engine must not re-propose)",
        })
    return entries


# ── library packages ─────────────────────────────────────────────────────────
def library_entries() -> list[dict]:
    entries = []
    lib = ROOT / "library"
    if not lib.is_dir():
        return entries
    for pyproject in sorted(lib.glob("*/pyproject.toml")):
        text = pyproject.read_text(encoding="utf-8")
        name = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
        version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        desc = re.search(r'^description\s*=\s*"([^"]+)"', text, re.MULTILINE)
        lic = re.search(r'^license\s*=\s*"([^"]+)"', text, re.MULTILINE)
        scrub = pyproject.parent / "SCRUB.md"
        scrub_status = "pending-human-review"
        if scrub.is_file() and re.search(r"^\*\*VERDICT: APPROVED\*\*", scrub.read_text(encoding="utf-8"), re.MULTILINE):
            scrub_status = "approved"
        pkg = {
            "id": name.group(1) if name else pyproject.parent.name,
            "kind": "library-package",
            "version": version.group(1) if version else None,
            "summary": desc.group(1) if desc else "",
            "license": lic.group(1) if lic else None,
            "source": str(pyproject.parent.relative_to(ROOT)),
            "invocation": f"pip install {name.group(1) if name else pyproject.parent.name}",
            "published": False,
            "scrub_status": scrub_status,
            "bundled_skills": [],
        }
        for skill_md in sorted(pyproject.parent.glob("src/**/skills/*/SKILL.md")):
            meta = parse_skill_md(skill_md)
            pkg["bundled_skills"].append({
                "id": meta["name"],
                "summary": meta.get("description", ""),
                "install": f"python -m {'.'.join(skill_md.relative_to(pyproject.parent / 'src').parts[:-2])} install",
            })
        entries.append(pkg)
    return entries


# ── dashboard ────────────────────────────────────────────────────────────────
def render_dashboard(catalog: dict) -> str:
    kinds = {}
    for e in catalog["entries"]:
        kinds.setdefault(e["kind"], []).append(e)
    order = ["library-package", "agent-skill", "runtime-efficiency", "brain-workflow",
             "runtime", "future-deferred"]
    titles = {
        "library-package": "Library packages (public candidates — scrub pending)",
        "agent-skill": "Installed agent skills (.agents/skills)",
        "runtime-efficiency": "Runtime efficiency skills (auto-triggered)",
        "brain-workflow": "Brain workflow passes",
        "runtime": "Runtime skills",
        "future-deferred": "Future (deferred by design)",
    }
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Epsilon Skills Catalog</title>",
        "<style>body{font:14px/1.5 -apple-system,sans-serif;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a}h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:2rem;"
        "border-bottom:1px solid #ddd;padding-bottom:.3rem}table{border-collapse:collapse;width:100%}"
        "td,th{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #eee;vertical-align:top}"
        "th{font-size:.8rem;text-transform:uppercase;color:#666}code{background:#f4f4f4;"
        "padding:.1rem .3rem;border-radius:3px;font-size:.85em}.chip{display:inline-block;"
        "padding:.05rem .5rem;border-radius:9px;font-size:.75rem;background:#eef}"
        ".warn{background:#fe9;} .muted{color:#888}</style>",
        f"<h1>Epsilon Skills Catalog</h1><p class='muted'>generated {catalog['generated_at']} · "
        f"{len(catalog['entries'])} entries · source of truth: brain/SKILL_MAP.md + library/ · "
        "backlog: brain/reflection/candidates.md · regenerate: <code>python3 tools/skills_catalog.py</code></p>",
    ]
    for kind in order:
        rows = kinds.pop(kind, [])
        if not rows:
            continue
        parts.append(f"<h2>{titles.get(kind, kind)} <span class='chip'>{len(rows)}</span></h2>")
        parts.append("<table><tr><th>skill</th><th>summary</th><th>invocation</th><th>meta</th></tr>")
        for e in rows:
            meta_bits = [e.get("provenance") or ""]
            if e.get("version"):
                meta_bits.append(f"v{e['version']}")
            if e.get("license"):
                meta_bits.append(e["license"])
            if e.get("scrub_status"):
                meta_bits.append(f"<span class='chip warn'>{e['scrub_status']}</span>")
            bundled = "".join(
                f"<br><span class='muted'>bundles skill: <code>{html.escape(b['id'])}</code></span>"
                for b in e.get("bundled_skills", []))
            parts.append(
                f"<tr><td><code>{html.escape(e['id'])}</code>{bundled}</td>"
                f"<td>{html.escape((e.get('summary') or '')[:280])}</td>"
                f"<td><code>{html.escape(e.get('invocation') or '')}</code></td>"
                f"<td>{' · '.join(b for b in meta_bits if b)}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def main() -> int:
    entries = library_entries() + agent_skills() + skill_map_entries()
    catalog = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/skills_catalog.py",
        "sources": [".agents/skills/*/SKILL.md", "brain/SKILL_MAP.md", "library/*/pyproject.toml"],
        "entries": entries,
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "skills_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    (GENERATED / "skills_dashboard.html").write_text(render_dashboard(catalog), encoding="utf-8")

    public = {
        "generated_at": catalog["generated_at"],
        "generator": catalog["generator"],
        "note": ("PUBLIC-CANDIDATE catalog. Only entries with scrub_status 'approved' "
                 "(per-package SCRUB.md, human-authorized) may be surfaced publicly; "
                 "'published' flips when the package is actually released (PyPI/repo split)."),
        "entries": [e for e in entries if e["kind"] == "library-package"],
    }
    (ROOT / "library" / "catalog.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")

    print(f"entries: {len(entries)}  "
          f"(library {sum(1 for e in entries if e['kind'] == 'library-package')}, "
          f"agent {sum(1 for e in entries if e['kind'] == 'agent-skill')}, "
          f"map {sum(1 for e in entries if e['kind'] in ('brain-workflow', 'runtime', 'runtime-efficiency', 'future-deferred'))})")
    print(f"wrote: {GENERATED / 'skills_catalog.json'}")
    print(f"wrote: {GENERATED / 'skills_dashboard.html'}")
    print(f"wrote: {ROOT / 'library' / 'catalog.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
