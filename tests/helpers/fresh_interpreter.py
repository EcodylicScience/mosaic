"""Programs run in a fresh interpreter, for the checks of what an import loads.

An in-process probe answers about whatever an earlier test already imported, so
each check starts its own interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence

__all__ = ["modules_loaded_by", "run_fresh"]


def run_fresh(code: str) -> str:
    """Run *code* in a fresh interpreter and return its stripped stdout.

    Raises:
        AssertionError: The program exits nonzero. The message is its stderr.
    """
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def modules_loaded_by(statement: str, candidates: Sequence[str]) -> list[str]:
    """Return the *candidates* that a fresh interpreter holds after *statement*, sorted."""
    code = (
        "import json, sys\n"
        f"{statement}\n"
        f"print(json.dumps(sorted(m for m in {tuple(candidates)!r} if m in sys.modules)))"
    )
    loaded: list[str] = json.loads(run_fresh(code))
    return loaded
