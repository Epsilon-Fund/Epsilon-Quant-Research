#!/usr/bin/env bash
# Install the epsilon guard hooks into THIS clone (.git/hooks). Opt-in, per clone.
# Usage: bash tools/git_hooks/install.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Guard: a stale core.hooksPath (e.g. from a moved clone) silently disables ALL hooks.
HP="$(git config --get core.hooksPath || true)"
if [ -n "$HP" ] && [ ! -d "$HP" ]; then
  echo "WARNING: core.hooksPath points at a nonexistent dir: $HP"
  echo "         unsetting it (restores the default .git/hooks)"
  git config --unset core.hooksPath
fi

HOOKS_DIR="$(git rev-parse --git-path hooks)"
SRC="tools/git_hooks/pre-commit"
DST="$HOOKS_DIR/pre-commit"

[ -f "$SRC" ] || { echo "ERROR: $SRC not found (run from inside the repo)"; exit 1; }

if [ -e "$DST" ] && ! cmp -s "$SRC" "$DST"; then
  cp "$DST" "$DST.backup.$(date +%Y%m%d%H%M%S)"
  echo "existing pre-commit hook backed up alongside it"
fi

cp "$SRC" "$DST"
chmod +x "$DST"
echo "installed: $DST"
echo "bypass when needed with: git commit --no-verify"
