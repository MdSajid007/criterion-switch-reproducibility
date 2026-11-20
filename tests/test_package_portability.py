"""Portability check for the self-contained verification command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def test_verifier_runs_from_an_unrelated_working_directory(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts" / "verify_paper_results.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REPRODUCTION CHECK PASSED" in completed.stdout
