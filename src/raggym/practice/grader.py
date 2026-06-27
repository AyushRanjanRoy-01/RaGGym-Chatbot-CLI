"""Run a learner's solution against the exercise's pytest checks (no LLM)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from raggym.core import get_logger

log = get_logger(__name__)


def run_tests(exercise_dir: str | Path) -> dict:
    """Run ``test_exercise.py`` in the exercise dir; return pass/fail + output.

    pytest's default (prepend) import mode puts the exercise dir on ``sys.path``,
    so the test's ``from solution import ...`` resolves to the learner's file.
    """
    exercise_dir = Path(exercise_dir)
    test_file = exercise_dir / "test_exercise.py"
    if not test_file.exists():
        return {
            "passed": False,
            "returncode": -1,
            "output": f"No test_exercise.py in {exercise_dir}",
        }

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_exercise.py"],
        cwd=str(exercise_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    passed = proc.returncode == 0
    log.info("grade_run", dir=str(exercise_dir), passed=passed, returncode=proc.returncode)
    return {"passed": passed, "returncode": proc.returncode, "output": output}
