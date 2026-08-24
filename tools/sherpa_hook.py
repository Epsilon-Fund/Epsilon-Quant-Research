#!/usr/bin/env python3
"""
sherpa_hook.py — OPTIONAL Claude Code UserPromptSubmit hook (CC-only bonus).

Runs the Sherpa router on the user's prompt and injects the top matching skills
as additional context, so the right skills surface hands-free on the Claude Code
path. Nothing depends on this — the reliable, agent-agnostic mechanism is the
Sherpa bootstrap step in brain/VAULT_MAP.md / CODEX.md / COWORK.md. Hooks are
CC-only and not guaranteed to fire, so this is purely a convenience layer.

Fail-safe by construction: any error, timeout, or weak match ⇒ exit 0 with no
output (never blocks or delays the prompt). Keyword-only (no embedder call) to
keep per-prompt latency negligible.

Wire-up lives in .claude/settings.json under hooks.UserPromptSubmit.
"""
import json
import subprocess
import sys
from pathlib import Path

SCORE_FLOOR = 0.08          # below this, no skill is a strong enough fit to surface
TOP_N = 3


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    prompt = (data.get("prompt") or "").strip()
    if len(prompt) < 8:                          # trivial prompts: skip
        return 0

    root = Path(__file__).resolve().parents[1]
    sherpa = root / "tools" / "sherpa.py"
    if not sherpa.is_file():
        return 0
    try:
        proc = subprocess.run(
            [sys.executable, str(sherpa), "--json", "--top", str(TOP_N),
             "--no-semantic", prompt],
            capture_output=True, text=True, timeout=10)
        res = json.loads(proc.stdout)
    except Exception:
        return 0

    hits = [r for r in res.get("results", []) if r.get("score", 0) >= SCORE_FLOOR]
    if not hits:
        return 0

    lines = ["Sherpa surfaced these installed skills for this task "
             "(auto-ranked, not hand-picked — load any that fit; a skill is a "
             "suggestion, not a gate):"]
    for r in hits:
        lines.append(f"- {r['name']} ({r['scope']}): {r['use_when']}")
    ctx = "\n".join(lines)

    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": ctx,
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
