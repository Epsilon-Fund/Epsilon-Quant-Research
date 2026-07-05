"""
Install the agent-skill bundles shipped inside this package into an agent's
skills directory (pattern credit: goldmansachs/gs-quant `gs_quant/skills`,
Apache-2.0).

    python -m lemma.calibrate.skills list
    python -m lemma.calibrate.skills install --project   # ./.claude/skills/
    python -m lemma.calibrate.skills install --global    # ~/.claude/skills/
    python -m lemma.calibrate.skills install --target DIR
    python -m lemma.calibrate.skills uninstall --project

A "skill" is any directory next to this file containing a SKILL.md
(agentskills.io Agent Skills spec). Install = copy the directory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _bundles() -> list[Path]:
    return sorted(d for d in _HERE.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())


def _target_dir(args) -> Path:
    if args.target:
        return Path(args.target).expanduser()
    if getattr(args, "global_", False):
        return Path.home() / ".claude" / "skills"
    return Path.cwd() / ".claude" / "skills"


def _description(skill_md: Path) -> str:
    """First line of the frontmatter description (handles `description: >` folds)."""
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("description:"):
            continue
        val = line.split(":", 1)[1].strip()
        if val in (">", "|", ">-", "|-"):  # folded/literal block: first indented line
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
            return ""
        return val
    return ""


def _cmd_list(_args) -> int:
    for b in _bundles():
        print(f"{b.name}: {_description(b / 'SKILL.md')[:100]}")
    return 0


def _cmd_install(args) -> int:
    dst_root = _target_dir(args)
    dst_root.mkdir(parents=True, exist_ok=True)
    for b in _bundles():
        dst = dst_root / b.name
        if dst.exists():
            if not args.force:
                print(f"skip (exists, use --force): {dst}")
                continue
            shutil.rmtree(dst)
        shutil.copytree(b, dst)
        print(f"installed: {dst}")
    return 0


def _cmd_uninstall(args) -> int:
    dst_root = _target_dir(args)
    for b in _bundles():
        dst = dst_root / b.name
        if dst.exists():
            shutil.rmtree(dst)
            print(f"removed: {dst}")
        else:
            print(f"not installed: {dst}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lemma.calibrate.skills")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("list", _cmd_list), ("install", _cmd_install), ("uninstall", _cmd_uninstall)):
        p = sub.add_parser(name)
        p.add_argument("--global", dest="global_", action="store_true",
                       help="~/.claude/skills instead of ./.claude/skills")
        p.add_argument("--project", action="store_true",
                       help="./.claude/skills (the default; flag kept for explicitness)")
        p.add_argument("--target", default=None, help="explicit skills directory")
        if name == "install":
            p.add_argument("--force", action="store_true", help="overwrite an existing bundle")
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
