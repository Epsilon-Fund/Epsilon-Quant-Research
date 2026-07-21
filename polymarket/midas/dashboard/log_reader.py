from __future__ import annotations

import json
from pathlib import Path


def tail_log(path: str, n: int = 200) -> list[dict]:
    """Return the last n parsed JSON lines from the log file."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    result: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            result.append({"ts": "", "level": "INFO", "event": line})
    return result


def filter_logs(
    lines: list[dict],
    *,
    level: str | None = None,
    keyword: str | None = None,
) -> list[dict]:
    """Filter log lines by minimum level and/or keyword substring."""
    _LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    result = lines
    if level:
        min_lvl = _LEVELS.get(level, 0)
        result = [r for r in result if _LEVELS.get(r.get("level", "INFO"), 1) >= min_lvl]
    if keyword:
        kw = keyword.lower()
        result = [r for r in result if kw in json.dumps(r).lower()]
    return result
