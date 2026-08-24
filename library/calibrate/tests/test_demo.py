"""Smoke test: the shipped example must run clean and tell a true story."""
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo.py"


@pytest.mark.skipif(not DEMO.exists(), reason="examples/ not present (installed wheel)")
def test_demo_runs_and_reports(tmp_path):
    out = subprocess.run(
        [sys.executable, str(DEMO)], capture_output=True, text=True,
        timeout=120, cwd=tmp_path,   # PNG (if any) lands in tmp, not the repo
    )
    assert out.returncode == 0, out.stderr
    assert "WELL-CALIBRATED forecaster" in out.stdout
    assert "OVER-CONFIDENT forecaster" in out.stdout
    assert "isotonic recalibration" in out.stdout
    assert "markets layer" in out.stdout
    assert "Spiegelhalter Z" in out.stdout
