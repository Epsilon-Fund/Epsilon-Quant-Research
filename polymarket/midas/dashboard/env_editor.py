from __future__ import annotations

import subprocess
from pathlib import Path

# (key, type, display label)
EDITABLE_KEYS: list[tuple[str, type, str]] = [
    ("STRATEGY_BID_THRESHOLD",          float, "Bid threshold (0–1)"),
    ("STRATEGY_MIN_REPRICE_TICKS",       int,   "Min reprice ticks"),
    ("OMS_ORDER_QTY",                   int,   "Order qty per market"),
    ("RISK_DAILY_LOSS_CAP_USDC",        float, "Daily loss cap (USDC)"),
    ("RISK_MAX_NOTIONAL_PER_EVENT_USDC", int,   "Max position qty per event (shares)"),
    ("RISK_ENABLE_AUTO_KILL",           str,   "Auto kill on loss cap (true/false)"),
    ("RISK_MIN_TOKEN_PRICE",           float, "Stop-loss token price (0 = disabled)"),
    ("LOG_LEVEL",                       str,   "Log level (DEBUG/INFO/WARNING/ERROR)"),
    ("DRY_RUN",                         str,   "Dry run (0/1)"),
]


def read_env(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    result: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        result[key.strip()] = val.strip()
    return result


def write_env(path: str, updates: dict[str, str]) -> None:
    """Update specific keys in the .env file, preserving all other lines."""
    p = Path(path)
    existing = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    written: set[str] = set()
    new_lines: list[str] = []

    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.partition("=")[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            written.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}={val}")

    p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def restart_bot() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["systemctl", "restart", "midas-harvester"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "Bot restarted successfully"
        return False, result.stderr.strip() or "systemctl returned non-zero exit code"
    except FileNotFoundError:
        return False, "systemctl not available (not on a Linux/systemd host)"
    except Exception as exc:
        return False, str(exc)
