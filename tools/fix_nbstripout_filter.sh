#!/usr/bin/env bash
# Pin the nbstripout git filter to an ABSOLUTE interpreter path (RC-004, brain/reflection/candidates.md).
#
# Why: the filter was configured as `python3 -m nbstripout`, which resolves per-context.
# Scheduled/sandboxed contexts (launchd, agent sandboxes) get a minimal PATH where
# `python3` -> /usr/bin/python3, which has no nbstripout -> every git operation touching
# a notebook aborts ("No module named nbstripout"). Pinning an absolute path makes the
# filter work identically in every context on this machine.
#
# Usage: bash tools/fix_nbstripout_filter.sh   (idempotent; run once per clone/machine)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Candidate interpreters, most specific first. `command -v python3` covers the
# interactive default; the rest cover launchd-style minimal-PATH contexts.
CANDIDATES=(
  "$(command -v python3 || true)"
  /Library/Frameworks/Python.framework/Versions/3.14/bin/python3
  /opt/homebrew/bin/python3
  /usr/local/bin/python3
  "$(pwd)/.venv/bin/python"
  /usr/bin/python3
)

PYBIN=""
for p in "${CANDIDATES[@]}"; do
  [ -n "$p" ] && [ -x "$p" ] || continue
  if "$p" -c "import nbstripout" >/dev/null 2>&1; then
    PYBIN="$p"
    break
  fi
done

if [ -z "$PYBIN" ]; then
  echo "ERROR: no interpreter with nbstripout found." >&2
  echo "Install it first, e.g.:  python3 -m pip install nbstripout  (or: uv pip install nbstripout)" >&2
  exit 1
fi

# Resolve symlinks so the pin survives PATH changes.
PYBIN="$("$PYBIN" -c 'import sys; print(sys.executable)')"

git config filter.nbstripout.clean "$PYBIN -m nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true

echo "pinned: filter.nbstripout.clean = $PYBIN -m nbstripout"
# Prove it works the way git will call it, outside any venv PATH.
echo '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}' | $PYBIN -m nbstripout >/dev/null \
  && echo "verify: filter executes OK" \
  || { echo "verify: FILTER STILL FAILING" >&2; exit 1; }
