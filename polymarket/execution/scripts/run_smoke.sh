#!/usr/bin/env bash
# Operator smoke-run launcher. Sets PYTHONPATH, defaults to the fake
# venue, optionally sources .env, and runs the bot.
set -euo pipefail

# Resolve repo root: scripts/ → execution/ → polymarket/ → <repo root>
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=.
export POLYMARKET_VENUE="${POLYMARKET_VENUE:-fake}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python3 -m polymarket.execution.cli
