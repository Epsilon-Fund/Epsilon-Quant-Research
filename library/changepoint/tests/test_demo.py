"""Smoke test: the shipped example must run clean and tell a true story."""
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo.py"


@pytest.mark.skipif(not DEMO.exists(), reason="examples/ not present (installed wheel)")
def test_demo_runs_and_reports():
    out = subprocess.run(
        [sys.executable, str(DEMO)], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    # the story the demo claims to tell, pinned
    assert "true breaks at bars [250, 425]" in out.stdout
    assert "detector comparison" in out.stdout
    assert "changepoint_features around the first TRUE break" in out.stdout
    assert "fresh_break_gate" in out.stdout
    assert "embargo_indices_from_breaks" in out.stdout
    assert "IDENTICAL to the batch run (asserted)" in out.stdout
