#!/usr/bin/env bash
# Daily brain/notes commit + push to the CURRENT personal branch.
# Runs on the Mac via launchd (real git environment). Markdown/brain only; never main; never force.
# Replaces the Cowork sandbox scheduled task, which cannot manage .git (lock/worktree EPERM).
set -uo pipefail

# launchd runs with a minimal PATH — set a sane one so git (and anything it calls) resolves.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

REPO="/Users/justiniturregui/Desktop/github/epsilon-quant-research"
LOG="$REPO/brain/generated/brain_commit_push.log"
SCOPE=(brain/ polymarket/research/notes/ docs/ README.md tools/)

# Neutralize the nbstripout filter for THIS markdown-only job: launchd's minimal env may not find
# the filter binary, and this script never commits notebooks, so a passthrough is safe and bulletproof.
GIT=(git -c filter.nbstripout.clean=cat -c filter.nbstripout.smudge=cat -c filter.nbstripout.required=false)

mkdir -p "$REPO/brain/generated"
exec >>"$LOG" 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') brain-commit-push ====="

cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

# Self-heal a stale lock / stale worktrees (works fine on the real Mac).
[ -f .git/index.lock ] && { echo "removing stale .git/index.lock"; rm -f .git/index.lock; }
"${GIT[@]}" worktree prune 2>/dev/null || true

BRANCH="$("${GIT[@]}" rev-parse --abbrev-ref HEAD)"
echo "branch: $BRANCH"
case "$BRANCH" in
  main|master)
    echo "ABORT: on '$BRANCH' — daily sync must run on a personal branch (e.g. justin). Checkout your branch and retry."
    exit 1 ;;
esac

"${GIT[@]}" add -A -- "${SCOPE[@]}"
if "${GIT[@]}" diff --cached --quiet; then
  echo "nothing to sync"; exit 0
fi

# EOL guard: refuse to commit CRLF in staged TEXT files (-I skips binary).
# Backstop to .gitattributes (eol=lf); matters most if this ever runs on Windows.
CR_HITS="$("${GIT[@]}" grep --cached -I -l -F "$(printf '\r')" -- "${SCOPE[@]}" 2>/dev/null || true)"
if [ -n "$CR_HITS" ]; then
  echo "ABORT: CRLF detected in staged text files — refusing to commit mixed endings:"
  echo "$CR_HITS"
  echo "Fix: run 'git add --renormalize .' (LF via .gitattributes), then retry. See brain/MERGE_PROTOCOL.md § 6."
  exit 1
fi

N="$("${GIT[@]}" diff --cached --name-only | wc -l | tr -d ' ')"
if ! "${GIT[@]}" commit -m "chore(brain): daily sync $(date '+%Y-%m-%d') ($BRANCH, $N files)"; then
  echo "commit failed"; exit 1
fi

# Keep your OWN branch in sync with its remote (NOT main). Catching up from main is a deliberate merge step.
if ! "${GIT[@]}" pull --no-rebase origin "$BRANCH"; then
  echo "pull failed (possible conflict) — aborting merge, NOT pushing. Resolve via brain/MERGE_PROTOCOL.md."
  "${GIT[@]}" merge --abort 2>/dev/null || true
  exit 1
fi

if "${GIT[@]}" push origin "$BRANCH"; then
  echo "pushed to origin/$BRANCH ($N files). done."
else
  echo "push failed (check auth/remote)"; exit 1
fi
