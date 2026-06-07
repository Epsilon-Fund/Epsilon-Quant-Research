#!/usr/bin/env python3
"""
brain_graph_audit.py — Graphify-style structural audit of the brain's wikilink graph.

Complements brain_hygiene.py: where hygiene checks per-note correctness (frontmatter,
summaries, broken links), this looks at the SHAPE of the link graph and turns structural
smells into actionable cleanup signals. Finds issues; does not fix them.

Reports:
  - Authorities      — most linked-TO notes (natural hubs)
  - Index hubs       — most linking-FROM notes (maps / overviews)
  - Over-connected   — very high total degree (candidate to split)
  - Orphans          — 0 inbound links (hard to discover)
  - Dead-ends        — 0 outbound links (don't route anywhere)
  - Topic islands     — small weakly-connected components, isolated from the main graph
  - Graph summary    — node/edge counts, density, component count

Writes: brain/generated/graph_audit.md   (git-ignored, regenerable)

Usage:
  python tools/brain_graph_audit.py
  python tools/brain_graph_audit.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", ".git", "node_modules", ".obsidian", ".tmp", "__pycache__",
                ".claude", ".pytest_cache", ".ipynb_checkpoints", ".agents"}
EXCLUDE_REL_PREFIXES = ("brain/generated",)
WIKILINK_RE = re.compile(r"!?\[\[([^\]]+)\]\]")

# Notes that are *meant* to be highly connected — don't flag them as smells.
KNOWN_HUBS = {"VAULT_MAP", "COWORK", "CODEX", "TODO", "POLYMARKET_BRAIN", "glossary",
              "SKILL_MAP", "OPERATING_RHYTHMS", "ONBOARDING", "INDEX",
              "strat_market_making", "strat_options_delta", "OBSIDIAN_INFRA_ROADMAP"}
OVERCONNECTED_DEGREE = 40   # total degree above which a non-hub note is worth a look
ISLAND_MAX_SIZE = 3         # weakly-connected components this small are "islands"
TOP_N = 15


def discover(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        if p.relative_to(root).as_posix().startswith(EXCLUDE_REL_PREFIXES):
            continue
        out.append(p)
    return sorted(out)


def parse_targets(text: str) -> list[str]:
    out = []
    for raw in WIKILINK_RE.findall(text):
        base = Path(raw.split("|", 1)[0].split("#", 1)[0].strip()).stem
        if base:
            out.append(base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=str(REPO_ROOT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = discover(root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # node = basename; keep one representative rel-path per basename
    stem_to_rel: dict[str, str] = {}
    for p in files:
        stem_to_rel.setdefault(p.stem, p.relative_to(root).as_posix())
    nodes = set(stem_to_rel)

    out_edges: dict[str, set[str]] = defaultdict(set)
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    adj: dict[str, set[str]] = defaultdict(set)  # undirected, for components

    edge_count = 0
    for p in files:
        src = p.stem
        try:
            targets = parse_targets(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            targets = []
        for tgt in targets:
            if tgt in nodes and tgt != src and tgt not in out_edges[src]:
                out_edges[src].add(tgt)
                in_deg[tgt] += 1
                out_deg[src] += 1
                adj[src].add(tgt)
                adj[tgt].add(src)
                edge_count += 1

    # weakly-connected components (undirected)
    seen: set[str] = set()
    components: list[list[str]] = []
    for n in sorted(nodes):
        if n in seen:
            continue
        comp, dq = [], deque([n])
        seen.add(n)
        while dq:
            cur = dq.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        components.append(sorted(comp))
    components.sort(key=len, reverse=True)

    authorities = sorted(nodes, key=lambda n: (-in_deg[n], n))[:TOP_N]
    index_hubs = sorted(nodes, key=lambda n: (-out_deg[n], n))[:TOP_N]
    overconnected = sorted(
        (n for n in nodes if (in_deg[n] + out_deg[n]) >= OVERCONNECTED_DEGREE and n not in KNOWN_HUBS),
        key=lambda n: -(in_deg[n] + out_deg[n]),
    )
    orphans = sorted(n for n in nodes if in_deg[n] == 0 and n not in KNOWN_HUBS)
    dead_ends = sorted(n for n in nodes if out_deg[n] == 0 and n not in KNOWN_HUBS)
    islands = [c for c in components if len(c) <= ISLAND_MAX_SIZE]
    main_comp = len(components[0]) if components else 0

    n_nodes = len(nodes)
    density = (edge_count / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

    print(f"graph audit @ {now}: {n_nodes} nodes, {edge_count} edges, "
          f"{len(components)} components (largest {main_comp}), "
          f"{len(orphans)} orphans, {len(dead_ends)} dead-ends, {len(islands)} islands")

    if args.dry_run:
        print("(dry-run: no file written)")
        return 0

    def deg(n):
        return f"in {in_deg[n]} / out {out_deg[n]}"

    L = [
        "---",
        "title: Graph Audit",
        f"created: {datetime.now().strftime('%Y-%m-%d')}",
        "status: generated",
        "owner: brain_graph_audit.py",
        "tags: [obsidian, brain, generated, graph]",
        "---",
        "",
        "# Graph Audit",
        "",
        "> Auto-generated by `tools/brain_graph_audit.py`. **Finds structural smells; does not fix them.**",
        "> Turn flagged items into a Codex link-repair pass if warranted. See [[OPERATING_RHYTHMS]].",
        f"> Last refreshed: {now}.",
        "",
        "Hub links: [[VAULT_MAP]] | [[SKILL_MAP]]",
        "",
        "## Graph summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Notes (nodes) | {n_nodes} |",
        f"| Wikilinks (edges) | {edge_count} |",
        f"| Connected components | {len(components)} |",
        f"| Largest component | {main_comp} ({main_comp / n_nodes:.0%} of graph) |",
        f"| Graph density | {density:.4f} |",
        f"| Orphans (0 inbound) | {len(orphans)} |",
        f"| Dead-ends (0 outbound) | {len(dead_ends)} |",
        f"| Topic islands (≤{ISLAND_MAX_SIZE} notes) | {len(islands)} |",
        "",
        "## Authorities — most linked-to (natural hubs)",
        "",
        "| Note | Degree |",
        "|---|---|",
    ]
    L += [f"| [[{n}]] | {deg(n)} |" for n in authorities]
    L += ["", "## Index hubs — most linking-from (maps / overviews)", "",
          "| Note | Degree |", "|---|---|"]
    L += [f"| [[{n}]] | {deg(n)} |" for n in index_hubs]

    L += ["", f"## Over-connected non-hub notes (total degree ≥ {OVERCONNECTED_DEGREE})", "",
          "> Candidates to split into smaller notes, or to promote to an explicit hub.", ""]
    L += ([f"- [[{n}]] — {deg(n)}" for n in overconnected] or ["_None._"])

    L += ["", "## Orphans — no inbound links (hard to discover)", "",
          "> Add a backlink from the relevant hub, or archive if dead.", ""]
    L += ([f"- [[{n}]] — `{stem_to_rel[n]}`" for n in orphans] or ["_None — clean._"])

    L += ["", "## Dead-ends — no outbound links (route nowhere)", "",
          "> Add at least a hub backlink so readers can navigate onward.", ""]
    L += ([f"- [[{n}]] — `{stem_to_rel[n]}`" for n in dead_ends[:60]]
          + ([f"- … +{len(dead_ends) - 60} more"] if len(dead_ends) > 60 else [])
          or ["_None._"])

    L += ["", f"## Topic islands — isolated clusters of ≤{ISLAND_MAX_SIZE} notes", "",
          "> Disconnected from the main graph. Link into the relevant cluster hub or confirm intentional.", ""]
    if islands:
        for c in islands:
            L.append("- " + ", ".join(f"[[{n}]]" for n in c))
    else:
        L.append("_None — every cluster connects to the main graph._")

    L.append("")
    (root / "brain" / "generated").mkdir(parents=True, exist_ok=True)
    (root / "brain" / "generated" / "graph_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote: brain/generated/graph_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
