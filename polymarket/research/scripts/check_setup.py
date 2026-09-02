"""check_setup.py — verify the environment and say, in a sentence, what to fix.

    python scripts/check_setup.py

Checks Python version, required packages, EPSILON_DATA_ROOT, R2 credentials (if the root is
s3://), and that the data is actually readable through the loader. Exits non-zero if anything
would stop the dashboard from starting.

Settings come from the shell first and then from `polymarket/research/.env` (loaded via
epsilon_data.config.load_env — shell env always wins). No value is ever printed except
EPSILON_DATA_ROOT, which is a path, not a secret.
"""
from __future__ import annotations
import os
import sys
import pathlib

OK, BAD = "  OK  ", " FAIL "
problems = []


def check(cond, ok_msg, fix_msg):
    print(f"[{OK if cond else BAD}] {ok_msg if cond else fix_msg}")
    if not cond:
        problems.append(fix_msg)
    return cond


# make epsilon_data importable from a checkout
root = next((p for p in pathlib.Path(__file__).resolve().parents if (p / "epsilon_data").is_dir()), None)
if root and str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Load polymarket/research/.env BEFORE reading any env var below — the docs tell you to put
# EPSILON_DATA_ROOT and the R2 credentials there. Shell env wins; nothing is printed.
env_loaded = False
try:
    from epsilon_data.config import ENV_FILE, load_env
    env_loaded = load_env()
except Exception:
    ENV_FILE = (root / ".env") if root else None

print("Epsilon research — setup check\n" + "-" * 40)
print(f"[{OK}] read {ENV_FILE}" if env_loaded else f"[      ] no .env at {ENV_FILE} (using shell environment only)")

check(sys.version_info >= (3, 10),
      f"Python {sys.version.split()[0]}",
      f"Python {sys.version.split()[0]} is too old — need >= 3.10 (the venv targets 3.14).")

missing = []
for mod in ("duckdb", "pandas", "pyarrow", "numpy", "streamlit", "plotly", "matplotlib"):
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
check(not missing,
      "all required packages importable",
      f"missing packages: {', '.join(missing)} — run  pip install -r requirements.txt  (or `uv sync`).")

dr = os.environ.get("EPSILON_DATA_ROOT")
check(dr is not None,
      f"EPSILON_DATA_ROOT = {dr}",
      "EPSILON_DATA_ROOT is not set — export it, or put it in polymarket/research/.env, "
      "pointing at your local research_v1/ dir or the s3:// bucket.")

is_s3 = bool(dr) and dr.startswith("s3://")
if is_s3:
    try:
        from epsilon_data._internal import _r2_creds
        creds = _r2_creds()
    except Exception:
        creds = None
    check(creds is not None,
          "R2 credentials found",
          "EPSILON_DATA_ROOT is s3:// but no R2 credentials — set EPSILON_R2_KEY_ID / _SECRET / "
          "_ENDPOINT in the shell or in polymarket/research/.env (or configure an rclone [r2] "
          "remote). See the README credentials section.")
elif dr:
    p = pathlib.Path(dr) / "tokens.parquet"
    check(p.exists(),
          f"found tokens.parquet under {dr}",
          f"no tokens.parquet under {dr} — fetch the data with  python scripts/fetch_data.py --dest \"{dr}\"  "
          f"(or fix EPSILON_DATA_ROOT if the data lives elsewhere).")

# the real test: can the loader read?
if not problems:
    try:
        import epsilon_data as ed
        n = len(ed.catalog())
        rec = ed.reconciliation()
        check(n > 0, f"loader read {n:,} tokens; reconciliation match={bool(rec['match'].all())}",
              "loader returned no tokens — the data root looks empty or unreadable.")
    except Exception as e:
        check(False, "", f"loader failed to read the data: {str(e)[:160]}")

print("-" * 40)
if problems:
    print(f"{len(problems)} problem(s) to fix:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print("All good — run:  streamlit run dashboard/app.py")
